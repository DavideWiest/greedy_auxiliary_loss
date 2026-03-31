from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class LayerwiseResNet18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        head_layers: list[nn.Module] = []
        if dropout > 0.0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(backbone.fc.in_features, num_classes))
        self.head = nn.Sequential(*head_layers)
        self.hidden_dims = [64, 128, 256, 512]

    def _pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.stem(x)
        hidden_states: list[torch.Tensor] = []

        x = self.layer1(x)
        hidden_states.append(self._pooled(x))

        x = self.layer2(x)
        hidden_states.append(self._pooled(x))

        x = self.layer3(x)
        hidden_states.append(self._pooled(x))

        x = self.layer4(x)
        final_representation = self._pooled(x)
        hidden_states.append(final_representation)

        logits = self.head(final_representation)
        return logits, hidden_states
