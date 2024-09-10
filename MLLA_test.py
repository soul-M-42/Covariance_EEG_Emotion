from MLLA import MLLABlock
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torchtune.modules import RotaryPositionalEmbeddings as RoPE
from utils_new import stratified_layerNorm
# num_heads = 2
# input_resolution = (64, 64)
# block = MLLABlock(dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

# class RoPE(torch.nn.Module):
#     r"""Rotary Positional Embedding.
#     """
#     def __init__(self, shape, base=10000):
#         super(RoPE, self).__init__()

#         channel_dims, feature_dim = shape[:-1], shape[-1]
#         k_max = feature_dim // (2 * len(channel_dims))

#         assert feature_dim % k_max == 0

#         # angles
#         theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
#         angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

#         # rotation
#         rotations_re = torch.cos(angles).unsqueeze(dim=-1)
#         rotations_im = torch.sin(angles).unsqueeze(dim=-1)
#         rotations = torch.cat([rotations_re, rotations_im], dim=-1)
#         self.register_buffer('rotations', rotations)

#     def forward(self, x):
#         if x.dtype != torch.float32:
#             x = x.to(torch.float32)
#         x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
#         pe_x = torch.view_as_complex(self.rotations) * x
#         return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        # self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))
        self.head_dim = dim // self.num_heads
        self.rope = RoPE(dim=self.head_dim, max_seq_len=200, base=10000)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, n, num_heads, head_dim)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, n, num_heads, head_dim)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        # q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        # k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        # kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
        x = x + self.lepe(v).permute(0, 2, 1)

        return x

# class LinearAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, qkv_bias=True):
#         super().__init__()
#         self.dim = dim
#         self.heads = heads
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5
        
#         inner_dim = dim_head * heads
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        
#         self.to_out = nn.Linear(inner_dim, dim)
        
#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
        
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.view(b, n, h, -1), qkv)
        
#         q = q.softmax(dim=-1)
#         k = k.softmax(dim=-1)
        
#         context = torch.einsum('bnhd,bnhv->bnhd', k, v)
#         out = torch.einsum('bnhd,bnhd->bnhd', q, context)
        
#         out = out.reshape(b, n, -1)
#         return self.to_out(out)

class MLLA_EEG_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, **kwargs):
        super().__init__()
        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = [B, L, C]
        x = x + self.cpe1(x.permute(0, 2, 1)).permute(0, 2, 1)
        shortcut = x
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x)
        x = self.act(self.dwc(x.permute(0, 2, 1)).permute(0, 2, 1))

        # Linear Attention
        x = self.attn(x)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = self.norm_out(x)
        return x

class MLLA_BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, in_dim, hidden_dim, n_filter, filterLen, out_dim, depth, num_heads, drop_path, qkv_bias=True):

        super().__init__()
        self.in_dim = in_dim
        # self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        # self.conv_in = conv_in(n_filter=n_filter, patch_size=in_dim, filterLen=filterLen)
        self.read_in = nn.Linear(in_dim, hidden_dim)
        self.read_out = nn.Linear(self.hidden_dim, out_dim)
        self.act = nn.ReLU()
        
        # build blocks
        self.blocks = nn.ModuleList([
            MLLA_EEG_Block(dim=self.hidden_dim, num_heads=num_heads, qkv_bias=qkv_bias, drop_path=drop_path)
            for i in range(depth)])

    def forward(self, x):
        # for name, param in self.read_in.named_parameters():
        #     if param.requires_grad:
        #         print(f"Gradient of {name}: {param.grad}")
        # print(f'Before MLLA: mean={torch.mean(x)}, std={torch.std(x)}')
        # x = self.conv_in(x)
        x = self.act(self.read_in(x))
        x = stratified_layerNorm(x, n_samples=x.shape[0]/2)
        for blk in self.blocks:
            x = blk(x)
            # print(f'In MLLA: mean={torch.mean(x)}, std={torch.std(x)}')
            
        x = self.act(self.read_out(x))
        x = stratified_layerNorm(x, n_samples=x.shape[0]/2)
        # print(f'After MLLA: mean={torch.mean(x)}, std={torch.std(x)}')
        
        return x

class conv_in(nn.Module):
    def __init__(self, n_filter=16, patch_size=30, filterLen=1):
        super().__init__()
        self.stride = filterLen // 2
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=n_filter, 
            kernel_size=(1, filterLen), 
            stride=(1,self.stride))
        self.out_dim = n_filter * ((patch_size - filterLen) // self.stride + 1)
        self.norm = nn.LayerNorm(self.out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = x.transpose(1,2)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.norm(x)
        x = self.act(x)
        return x

# batch = 64
# n_channel = 60
# N = 100
# C = 128
# x = torch.randn((batch, N, C))
# print(x.shape)
# attention = LinearAttention(dim=C, heads=8, dim_head=64, qkv_bias=True)
# layer = MLLA_BasicLayer(dim=C, depth=2, num_heads=8)
# x = layer(x)
# print(x.shape)

