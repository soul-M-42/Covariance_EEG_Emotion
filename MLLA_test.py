from MLLA import MLLABlock
import torch
import torch.nn as nn
# dim = 64
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

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1), qkv)
        
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)
        
        context = torch.einsum('bnhd,bnhv->bnhd', k, v)
        out = torch.einsum('bnhd,bnhd->bnhd', q, context)
        
        out = out.reshape(b, n, -1)
        return self.to_out(out)

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
        self.attn = LinearAttention(dim=dim, heads=num_heads, qkv_bias=qkv_bias)
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
        x = shortcut + x
        x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)

        # FFN
        x = x + self.mlp(self.norm2(x))
        x = self.norm_out(x)
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

    def __init__(self, in_dim, hidden_dim, out_dim, depth, num_heads, qkv_bias=True):

        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.read_in = nn.Linear(in_dim, hidden_dim)
        self.read_out = nn.Linear(hidden_dim, out_dim)
        # build blocks
        self.blocks = nn.ModuleList([
            MLLA_EEG_Block(dim=hidden_dim, num_heads=num_heads, qkv_bias=qkv_bias)
            for i in range(depth)])

    def forward(self, x):
        # for name, param in self.read_in.named_parameters():
        #     if param.requires_grad:
        #         print(f"Gradient of {name}: {param.grad}")
        x = self.read_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.read_out(x)
        
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

