import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# Assuming you have the optimized RoPE implementation
# If not, you can use the one provided here
class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(base)) / dim))
        sinusoid_inp = position * div_term
        sin, cos = torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)

    def forward(self, x):
        sin = self.sin[:x.size(-2), :].unsqueeze(0).unsqueeze(0)
        cos = self.cos[:x.size(-2), :].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x.flatten(-2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class LinearAttention(nn.Module):
    """Optimized Linear Attention with LePE and RoPE."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.rope = RoPE(dim=self.head_dim, max_seq_len=512, base=10000)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply ELU activation and add 1
        q = self.elu(q) + 1
        k = self.elu(k) + 1

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, D)

        # Combine heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Apply LePE
        v_conv = v.transpose(1, 2).reshape(B, N, C).permute(0, 2, 1)  # (B, C, N)
        lepe_output = self.lepe(v_conv).permute(0, 2, 1)  # (B, N, C)

        x = attn_output + lepe_output  # (B, N, C)

        return x


class vanillaMultiHeadAtt(nn.Module):
    """Optimized Linear Attention with LePE and RoPE."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.rope = RoPE(dim=self.head_dim, max_seq_len=512, base=10000)
        self.att = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_output, attn_output_weights = self.att(q, k, v)


        x = attn_output + x  # (B, N, C)

        return x

class MLLA_EEG_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.cpe = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        # self.attn = vanillaMultiHeadAtt(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Convolutional Position Encoding
        x = x + self.cpe(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Attention Block
        x_res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x_res + self.drop_path(x)

        # MLP Block
        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_res + self.drop_path(x)

        return x

class MLLA_BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage."""

    def __init__(self, in_dim, hidden_dim, out_dim, depth, num_heads, drop_path=0., qkv_bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth

        # Input and Output Projections
        self.read_in = nn.Linear(in_dim, hidden_dim)
        self.read_out = nn.Linear(hidden_dim, out_dim)

        # Build blocks
        self.blocks = nn.ModuleList([
            MLLA_EEG_Block(dim=hidden_dim, num_heads=num_heads, qkv_bias=qkv_bias, drop_path=drop_path)
            for _ in range(depth)
        ])

    def forward(self, x):
        x = F.relu(self.read_in(x))
        # print(x.shape)
        for blk in self.blocks:
            x = blk(x)
        x = F.relu(self.read_out(x))
        return x
