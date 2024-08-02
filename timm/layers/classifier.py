""" Classifier head and layer factory

Hacked together by / Copyright 2020 Ross Wightman
"""
from collections import OrderedDict
from functools import partial
from typing import Optional, Union, Callable

import torch
from torch.jit import Final
import torch.nn as nn
from torch.nn import functional as F

from .adaptive_avgmax_pool import SelectAdaptivePool2d
from .config import use_fused_attn
from .create_act import get_act_layer, create_act_layer
from .create_norm import get_norm_layer, create_norm_layer 
from .drop import DropPath
from .mlp import GluMlp, Mlp


def _create_pool(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc


def create_classifier(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: str = 'NCHW',
        drop_rate: Optional[float] = None,
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    if drop_rate is not None:
        dropout = nn.Dropout(drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            use_conv: bool = False,
            input_fmt: str = 'NCHW',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
        """
        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        global_pool, fc = create_classifier(
            in_features,
            num_classes,
            pool_type,
            use_conv=use_conv,
            input_fmt=input_fmt,
        )
        self.global_pool = global_pool
        self.drop = nn.Dropout(drop_rate)
        self.fc = fc
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = nn.Flatten(1) if self.use_conv and pool_type else nn.Identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
            )

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)


class NormMlpClassifierHead(nn.Module):

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            hidden_size: Optional[int] = None,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            norm_layer: Union[str, Callable] = 'layernorm2d',
            act_layer: Union[str, Callable] = 'tanh',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            hidden_size: The hidden size of the MLP (pre-logits FC layer) if not None.
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
            norm_layer: Normalization layer type.
            act_layer: MLP activation layer type (only used if hidden_size is not None).
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        self.use_conv = not pool_type
        norm_layer = get_norm_layer(norm_layer)
        act_layer = get_act_layer(act_layer)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear

        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        if hidden_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', linear_layer(in_features, hidden_size)),
                ('act', act_layer()),
            ]))
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = linear_layer(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
            self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        self.use_conv = self.global_pool.is_identity()
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear
        if self.hidden_size:
            if ((isinstance(self.pre_logits.fc, nn.Conv2d) and not self.use_conv) or
                    (isinstance(self.pre_logits.fc, nn.Linear) and self.use_conv)):
                with torch.no_grad():
                    new_fc = linear_layer(self.in_features, self.hidden_size)
                    new_fc.weight.copy_(self.pre_logits.fc.weight.reshape(new_fc.weight.shape))
                    new_fc.bias.copy_(self.pre_logits.fc.bias)
                    self.pre_logits.fc = new_fc
        self.fc = linear_layer(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x

class StarAct(nn.Module):
    """
    StarAct: s * act(x) ** 2 + b
    """

    def __init__(
            self,
            act_fn='relu',
            scale_value=1.0,
            bias_value=0.0,
            scale_learnable=True,
            bias_learnable=True,
            mode=None,
            inplace=False
    ):
        super().__init__()
        self.inplace = inplace
        self.act = create_act_layer(act_fn, inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.act(x) ** 2 + self.bias
class PyramidFeatureAggregationModel(nn.Module):
    def __init__(
        self,
        model,
        num_classes = None,
        head_type = "glu",
        head_act = "silu",
    ):
        super().__init__()
        self.model = model

        self.feature_dims = model.feature_info.channels()
        self.num_features = sum(self.feature_dims)

        self.norms = nn.ModuleList([create_norm_layer('layernorm2d', dim) for dim in self.feature_dims])
        self.pools = nn.ModuleList([SelectAdaptivePool2d(pool_type='fast_avg', flatten=True) for dim in self.feature_dims])
        self.num_classes = 1000 if num_classes is None else num_classes
        if(head_type == "fc"):
            self.head = nn.Linear(self.num_features, self.num_classes)
        elif(head_type == "mlp"):
            self.head = Mlp(
                in_features = self.num_features,
                hidden_features = int(4*self.num_features),
                out_features = self.num_classes,
                act_layer = get_act_layer(head_act),
                norm_layer = None,
            )
        elif(head_type == "glu"):
            self.head = GluMlp(
                in_features = self.num_features,
                hidden_features = int(2*2.5*self.num_features),
                out_features = self.num_classes,
                act_layer = get_act_layer(head_act),
                norm_layer = None,
            )

    def forward(self, x):
        x=self.model(x)
        #x = torch.column_stack([nn.functional.gelu(out).mean((-2, -1)) for out in x]) # NCHW only for now
        #return self.head(x)

        x = torch.column_stack([pool(norm(out)) for pool, norm, out in zip(self.pools, self.norms, x)])
        return self.head(x)

class Attention(nn.Module):
    fused_attn: Final[bool]
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            proj_bias=False,
    ):
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.2,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            **kwargs
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
        
class CrossAttention(nn.Module):
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            query_dim = None,
            kv_dim = None,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            norm_layer: nn.Module = nn.LayerNorm,
            **kwargs
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim
        self.query_dim = self.dim if query_dim is None else query_dim
        self.kv_dim = self.dim if kv_dim is None else kv_dim
        self.fused_attn = use_fused_attn()
        
        self.q = nn.Linear(self.query_dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.kv_dim, self.dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, x) -> torch.Tensor:
        N_q, _ = q.shape # [N_q, C_q]
        B, N, C = x.shape # [B, N_kv, C]
        q = self.q(q).reshape(1, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [1, n_h, N_q, d_h]
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [2, B, n_h, N_kv, d_h]
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) # [B, n_h, N_q, N_kv]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v # [B, n_h, N_q, d_h]

        x = x.permute(0, 2, 1, 3).reshape(B, N_q, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
class FeaturePyramid2Token(nn.Module):
    def __init__(
        self,
        model,
        embed_dim = 384,
        depth = 8,
        num_classes = None,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.num_classes = 1000 if num_classes is None else num_classes
        feature_dims = self.model.feature_info.channels()
        self.decoders = nn.ModuleList([CrossAttention(embed_dim, kv_dim = dim, **kwargs) for dim in feature_dims])
        self.queries = nn.ParameterList([nn.Parameter(torch.randn(32, embed_dim)) for _ in feature_dims])
        self.norms = nn.ModuleList([create_norm_layer('layernorm2d', dim) for dim in feature_dims])
        self.encoder = nn.Sequential(*[TransformerBlock(embed_dim, **kwargs) for _ in range(depth)])
        self.head_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, self.num_classes)
    def forward(self, x):
        x = self.model(x)
        x = torch.column_stack([
            decoder(query, norm(out).flatten(2).permute(0,2,1)) \
            for decoder, query, norm, out in zip(self.decoders, self.queries, self.norms, x)
        ])
        x = self.encoder(x)
        x = x.mean(-2)
        x = self.head_norm(x)
        x = self.fc(x)
        return x
