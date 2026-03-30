from __future__ import annotations

import torch
from torch import nn


class TextTransformerBlock(nn.Module):
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

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = attention_mask == 0
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class LayerwiseTextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        num_classes: int,
        hidden_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TextTransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dims = [hidden_dim for _ in range(num_layers)]
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.unsqueeze(-1).float()
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.token_embed(input_ids) + self.pos_embed[:, : input_ids.shape[1]]
        x = self.dropout(x)
        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
            hidden_states.append(self._pool(x, attention_mask))
        x = self.norm(x)
        pooled = self._pool(x, attention_mask)
        logits = self.head(pooled)
        return logits, hidden_states
