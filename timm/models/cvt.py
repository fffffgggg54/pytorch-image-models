from typing import Optional, Tuple

import torch
import torch.nn
import torch.nn.functional as F

from timm.layers import ConvNormAct, LayerNorm2d, Mlp, QuickGELU, trunc_normal_, use_fused_attn


class ConvEmbed(nn.Module):
    def __init__(
        self,
        in_chs: int = 3,
        out_chs: int = 64,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 2,
        norm_layer: nn.Module = nn.LayerNorm2d,
    ) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.norm = norm_layer(out_chs) if norm_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # [B, C, H, W] -> [B, C, H, W]
        x = self.conv(x)
        x = self.norm(x)
        return x

class ConvProj(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        stride_q: int = 1,
        stride_kv: int = 2,
        padding: int = 1,
        bias: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.Identity(),
    ) -> None:
        super().__init__()
        self.dim = dim

        self.conv_q = ConvNormAct(
            dim,
            dim,
            kernel_size,
            stride=stride_q,
            padding=padding,
            groups=in_chs,
            bias=bias,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        
        self.conv_k = ConvNormAct(
            dim,
            dim,
            kernel_size,
            stride=stride_kv,
            padding=padding,
            groups=in_chs,
            bias=conv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        
        self.conv_v = ConvNormAct(
            dim,
            dim,
            kernel_size,
            stride=stride_kv,
            padding=padding,
            groups=in_chs,
            bias=conv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, H*W, C]
        q = self.conv_q(x).flatten(2).transpose(1, 2)
        k = self.conv_k(x).flatten(2).transpose(1, 2)
        v = self.conv_v(x).flatten(2).transpose(1, 2)
        return q, k, v

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 1,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_chs, out_chs)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N, C = q.shape
        
        # [B, H*W, C] -> [B, H*W, n_h, d_h] -> [B, n_h, H*W, d_h]
        q = self.proj_q(q).reshape(B, q.shape[2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.proj_k(k).reshape(B, k.shape[2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.proj_v(v).reshape(B, v.shape[2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        # [B, n_h, H*W, d_h], [B, n_h, H*W/4, d_h], [B, n_h, H*W/4, d_h]

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CvTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        stride_q: int = 1,
        stride_kv: int = 2,
        padding: int = 1,
        conv_bias: bool = False,
        conv_norm_layer: nn.Module = nn.BatchNorm2d,
        conv_act_layer: nn.Module = nn.Identity(),
        num_heads: int = 1,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        input_norm_layer = LayerNorm2d,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        mlp_layer: nn.Module = Mlp,
        mlp_ratio: float = 4.,
        mlp_act_layer: nn.Module = QuickGELU,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.use_cls_token = use_cls_token

        self.norm1 = norm_layer(dim)
        self.conv_proj = ConvProj(
            dim = dim,
            kernel_size = kernel_size,
            stride_q = stride_q,
            stride_kv = stride_kv,
            padding = padding,
            bias = conv_bias,
            norm_layer = conv_norm_layer,
            act_layer = conv_act_layer,
        )
        self.attn = Attention(
            dim = dim,
            num_heads = num_heads,
            qkv_bias = qkv_bias,
            qk_norm = qk_norm,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            norm_layer = norm_layer
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def add_cls_token(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        cls_token: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)
        return q, k, v

    def fw_attn(self, x: torch.Tensor, cls_token: Optional[torch.Tensor]) -> torch.Tensor:
        return self.attn(*self.add_cls_token(*self.conv_proj(x), cls_token))

    def forward(self, x: torch.Tensor, cls_token: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2) + self.drop_path1(self.ls1(self.fw(attn(self.norm1(x)))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if self.use_cls_token:
            cls_token, x = torch.split(x, [1, H*W], 1)
        
        return x, cls_token

class CvTStage(nn.Module):
    def __init__(
        in_chs: int,
        dim: int,
        depth: int,
        embed_kernel_size: int = 7,
        embed_stride: int = 4,
        embed_padding: int = 2,
        kernel_size: int = 3,
        stride_q: int = 1,
        stride_kv: int = 2,
        padding: int = 1,
        conv_bias: bool = False,
        conv_norm_layer: nn.Module = nn.BatchNorm2d,
        conv_act_layer: nn.Module = nn.Identity(),
        num_heads: int = 1,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        input_norm_layer = LayerNorm2d,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: Optional[float] = None,
        drop_path_rates: List[float] = [0.],
        mlp_layer: nn.Module = Mlp,
        mlp_ratio: float = 4.,
        mlp_act_layer: nn.Module = QuickGELU,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        
        self.conv_embed = ConvEmbed(
            in_chs = in_chs,
            out_chs = dim,
            kernel_size = embed_kernel_size,
            stride = embed_stride,
            padding = embed_padding,
            norm_layer = input_norm_layer,
        )
        self.embed_drop = nn.Dropout(proj_drop)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_cls_token else None

        blocks = []
        for i in range(depth):
            block = CvTBlock(
                dim = dim,
                kernel_size = kernel_size,
                stride_q = stride_q,
                stride_kv = stride_kv,
                padding = padding,
                conv_bias = conv_bias,
                conv_norm_layer = conv_norm_layer,
                conv_act_layer = conv_act_layer,
                num_heads = num_heads,
                qkv_bias = qkv_bias,
                qk_norm = qk_norm,
                attn_drop = attn_drop,
                proj_drop = proj_drop,
                input_norm_layer = input_norm_layer,
                norm_layer = norm_layer,
                init_values = init_values,
                drop_path = drop_path_rates[i],
                mlp_layer = mlp_layer,
                mlp_ratio = mlp_ratio,
                mlp_act_layer = mlp_act_layer,
                use_cls_token = use_cls_token,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_embed(x)
        x = self.embed_drop(x)

        cls_token = self.cls_token
        for block in self.blocks: # technically possible to exploit nn.Sequential's untyped intermediate results if each block takes in a tensor
            x, cls_token = block(x, cls_token)
        
        return x, cls_token

class CvT(nn.Module):
    def __init__(
        in_chans: int = 3,
        num_classes: int = 1000,
        dims: Tuple[int, ...] = (64, 192, 384),
        depths: Tuple[int, ...] = (1, 2, 10),
        embed_kernel_size: Tuple[int, ...] = (7, 3, 3),
        embed_stride: Tuple[int, ...] = (4, 2, 2),
        embed_padding: Tuple[int, ...] = (2, 1, 1),
        kernel_size: int = 3,
        stride_q: int = 1,
        stride_kv: int = 2,
        padding: int = 1,
        conv_bias: bool = False,
        conv_norm_layer: nn.Module = nn.BatchNorm2d,
        conv_act_layer: nn.Module = nn.Identity(),
        num_heads: Tuple[int, ...] = (1, 3, 6),
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        input_norm_layer = LayerNorm2d,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: Optional[float] = None,
        drop_path_rate: float = 0.,
        mlp_layer: nn.Module = Mlp,
        mlp_ratio: float = 4.,
        mlp_act_layer: nn.Module = QuickGELU,
        use_cls_token: Tuple[bool, ...] = (False, False, True),
    ) -> None:
        super().__init__()
        num_stages = len(dims)
        assert num_stages == len(depths) == len(embed_kernel_size) == len(embed_stride)
        assert num_stages == len(embed_padding) == len(num_heads) == len(use_cls_token)
        self.num_classes = num_classes
        self.num_features = dims[-1]
        self.drop_rate = drop_rate
        
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        
        in_chs = in_chans
        
        stages = []
        for stage_idx in range(num_stages):
            dim = dims[stage_idx]
            stage = CvTStage(
                in_chs = in_chs,
                dim = dim,
                depth = depths[stage_idx],
                embed_kernel_size = embed_kernel_size[stage_idx],
                embed_stride = embed_stride[stage_idx],
                embed_padding = embed_padding[stage_idx],
                kernel_size = kernel_size,
                stride_q = stride_q,
                stride_kv = stride_kv,
                padding = padding,
                conv_bias = conv_bias,
                conv_norm_layer = conv_norm_layer,
                conv_act_layer = conv_act_layer,
                num_heads = num_heads[stage_idx],
                qkv_bias = qkv_bias,
                qk_norm = qk_norm,
                attn_drop = attn_drop,
                proj_drop = proj_drop,
                input_norm_layer = input_norm_layer,
                norm_layer = norm_layer,
                init_values = init_values,
                drop_path_rates = dpr[stage_idx],
                mlp_layer = mlp_layer,
                mlp_ratio = mlp_ratio,
                mlp_act_layer = mlp_act_layer,
                use_cls_token = use_cls_token[stage_idx],
            )
            in_chs = dim
            stages.append(stage)