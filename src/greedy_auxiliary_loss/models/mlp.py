from __future__ import annotations

import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LayerwiseMLP(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.stem = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([MLPBlock(hidden_dim, dropout) for _ in range(num_layers - 1)])
        self.head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dims = [hidden_dim for _ in range(num_layers)]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states: list[torch.Tensor] = []
        x = self.stem(x)
        hidden_states.append(x)
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)
        logits = self.head(x)
        return logits, hidden_states
