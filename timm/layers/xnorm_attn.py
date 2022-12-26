from torch import nn
import torch.nn.functional as F
import torch.linalg as LA



class XNorm(nn.Module):
    def __init__(self, initial_gamma = 0.2):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(gamma_size))
        nn.init.constant_(self.gamma, initial_gamma)

    def forward(self, x):
        x = LA.norm(x, dim=0) * self.gamma
        return x

class XNormAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., stride = 1):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.xnorm = Xnorm()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # n x d_q, n x d_k, n x d_v
        
        attn = self.xnorm(k.transpose(-2, -1) @ v)
        x = (self.xnorm(q) @ attn).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        

        return x
    
