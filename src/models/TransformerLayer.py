import torch.nn as nn
import torch.nn.functional as F
import torch

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        inv_freq = 1.0 / (500 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x):
        # x: (B, H, L, D)
        L = x.size(-2)
        return self.cos[:L].unsqueeze(0).unsqueeze(0), \
               self.sin[:L].unsqueeze(0).unsqueeze(0)

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, key_padding_mask=None):
        # x: (B, L, D)
        B, L, D = x.shape

        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B,H,L,D)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.attn = RoPEMultiheadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x