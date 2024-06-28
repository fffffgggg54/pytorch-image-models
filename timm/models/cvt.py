""" CvT: Convolutional Vision Transformer

reference impl of CvT in PyTorch from https://github.com/microsoft/CvT

'CvT: Introducing Convolutions to Vision Transformers'
    - https://arxiv.org/abs/2103.15808


"""

from collections import OrderedDict
from functools import partial
from typing import List, Final, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ConvNormAct, LayerNorm, LayerNorm2d, Mlp, QuickGELU, trunc_normal_, use_fused_attn
from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model

from einops import rearrange
from einops.layers.torch import Rearrange

__all__ = ['CvT']

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )
        self.probe = nn.Identity()

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.probe(x)
        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = (patch_size,patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)


        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.probe = nn.Identity()

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H*W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens


class CvT(nn.Module):
    def __init__(self,
            in_chans=3,
            num_classes=1000,
            init='trunc_norm',
            spec=None,
            dims: Tuple[int, ...] = (64, 192, 384),
            depths: Tuple[int, ...] = (1, 2, 10),
            embed_kernel_size: Tuple[int, ...] = (7, 3, 3),
            embed_stride: Tuple[int, ...] = (4, 2, 2),
            embed_padding: Tuple[int, ...] = (2, 1, 1),
            embed_norm_layer: nn.Module = partial(LayerNorm2d, eps=1e-5),
            kernel_size: int = 3,
            stride_q: int = 1,
            stride_kv: int = 2,
            padding: int = 1,
            conv_bias: bool = False,
            conv_norm_layer: nn.Module = nn.BatchNorm2d,
            conv_act_layer: nn.Module = nn.Identity,
            num_heads: Tuple[int, ...] = (1, 3, 6),
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            input_norm_layer = partial(LayerNorm, eps=1e-5),
            norm_layer: nn.Module = partial(LayerNorm, eps=1e-5),
            init_values: Optional[float] = None,
            drop_path_rate: float = 0.,
            mlp_layer: nn.Module = Mlp,
            mlp_ratio: float = 4.,
            mlp_act_layer: nn.Module = QuickGELU,
            use_cls_token: Tuple[bool, ...] = (False, False, True),
            drop_rate: float = 0.,
    ):
        super().__init__()
        self.num_classes = num_classes




        self.num_stages = len(dims)
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': embed_kernel_size[i],
                'patch_stride': embed_stride[i],
                'patch_padding': embed_padding[i],
                'embed_dim': dims[i],
                'depth': depths[i],
                'num_heads': num_heads[i],
                'mlp_ratio': mlp_ratio,
                'qkv_bias': qkv_bias,
                'drop_rate': drop_rate,
                'attn_drop_rate': attn_drop,
                'drop_path_rate': proj_drop,
                'with_cls_token': use_cls_token[i],
                'method': 'dw_bn',
                'kernel_size': kernel_size,
                'padding_q': padding,
                'padding_kv': padding,
                'stride_kv': stride_kv,
                'stride_q': stride_q,
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=mlp_act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = dims[i]

        dim_embed = dims[-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = use_cls_token[-1]

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)



    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')

        return layers

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

    

    
    
def _create_cvt(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 2, 10))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        CvT,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (14, 14),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stage0.patch_embed.proj', 'classifier': 'head',
        **kwargs
    }
    
default_cfgs = generate_default_cfgs({
    'cvt_13.msft_in1k': _cfg(
        url='https://github.com/fffffgggg54/pytorch-image-models/releases/download/cvt/CvT-13-224x224-IN-1k.pth'),
    'cvt_13.msft_in1k_384': _cfg(
        url='https://github.com/fffffgggg54/pytorch-image-models/releases/download/cvt/CvT-13-384x384-IN-1k.pth',
        input_size=(3, 384, 384), pool_size=(24, 24)),
    'cvt_13.msft_in22k_ft_in1k_384': _cfg(url='https://github.com/fffffgggg54/pytorch-image-models/releases/download/cvt/CvT-13-384x384-IN-22k.pth',
        input_size=(3, 384, 384), pool_size=(24, 24)),
    
    'cvt_21.msft_in1k': _cfg(url='https://github.com/fffffgggg54/pytorch-image-models/releases/download/cvt/CvT-21-224x224-IN-1k.pth'),
    'cvt_21.msft_in1k_384': _cfg(url='https://github.com/fffffgggg54/pytorch-image-models/releases/download/cvt/CvT-21-384x384-IN-1k.pth',
        input_size=(3, 384, 384), pool_size=(24, 24)),
    'cvt_21.msft_in22k_ft_in1k_384': _cfg(url='https://github.com/fffffgggg54/pytorch-image-models/releases/download/cvt/CvT-21-384x384-IN-22k.pth',
        input_size=(3, 384, 384), pool_size=(24, 24)),
    
    'cvt_w24.msft_in22k_ft_in1k_384': _cfg(url='https://github.com/fffffgggg54/pytorch-image-models/releases/download/cvt/CvT-w24-384x384-IN-22k.pth',
        input_size=(3, 384, 384), pool_size=(24, 24)),
})


@register_model
def cvt_13(pretrained=False, **kwargs) -> CvT:
    model_args = dict(depths=(1, 2, 10), dims=(64, 192, 384), num_heads=(1, 3, 6))
    return _create_cvt('cvt_13', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def cvt_21(pretrained=False, **kwargs) -> CvT:
    model_args = dict(depths=(1, 4, 16), dims=(64, 192, 384), num_heads=(1, 3, 6))
    return _create_cvt('cvt_21', pretrained=pretrained, **dict(model_args, **kwargs))
    
@register_model
def cvt_w24(pretrained=False, **kwargs) -> CvT:
    model_args = dict(depths=(2, 2, 20), dims=(192, 768, 1024), num_heads=(3, 12, 16))
    return _create_cvt('cvt_w24', pretrained=pretrained, **dict(model_args, **kwargs))