import torch.nn as nn
from .TransformerLayer import RoPETransformerLayer
import torch 

class TransformerRSA(nn.Module):
    def __init__(self,vocab_size=21,d_model=256,nhead=4,num_layers=3,
                 dim_feedforward=758,esm_dim=1280,dropout=0.2):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.fuse_proj = nn.Linear(d_model + esm_dim, d_model)

        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, tokens, esm_emb, mask_bool):
        x_seq = self.token_emb(tokens)          
        x = torch.cat([x_seq, esm_emb], dim=-1)
        x = self.fuse_proj(x)

        key_padding_mask = ~mask_bool
        for layer in self.layers:
            x = layer(x, key_padding_mask)

        return self.head(x).squeeze(-1)