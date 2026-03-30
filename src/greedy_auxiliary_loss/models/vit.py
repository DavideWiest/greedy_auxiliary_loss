from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class LayerwiseViT(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        hidden_dim: int = 192,
        num_layers: int = 6,
        patch_size: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels, height, width = input_shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")
        num_patches = (height // patch_size) * (width // patch_size)
        self.patch_embed = nn.Conv2d(channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dims = [hidden_dim for _ in range(num_layers)]
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x + self.pos_embed)
        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x[:, 0])
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits, hidden_states
