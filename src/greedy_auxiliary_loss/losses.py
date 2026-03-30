from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def compute_future_weights(
    total_candidates: int,
    layer_index: int,
    strategy: str,
    lookahead: int,
    sigma: float,
) -> torch.Tensor:
    positions = torch.arange(layer_index + 1, total_candidates, dtype=torch.float32)
    if positions.numel() == 0:
        return positions
    normalized = strategy.lower()
    if normalized == "fixed":
        target_index = min(total_candidates - 1, layer_index + max(1, lookahead))
        weights = (positions == float(target_index)).float()
        return weights / weights.sum()
    if normalized == "gaussian":
        center = float(layer_index + max(1, lookahead))
        width = max(float(sigma), 1e-3)
        weights = torch.exp(-0.5 * ((positions - center) / width) ** 2)
        return weights / weights.sum()
    if normalized == "uniform":
        return torch.full_like(positions, fill_value=1.0 / positions.numel())
    if normalized == "output":
        weights = torch.zeros_like(positions)
        weights[-1] = 1.0
        return weights
    if normalized == "exponential":
        offsets = positions - float(layer_index)
        weights = torch.exp(-offsets / max(float(lookahead), 1.0))
        return weights / weights.sum()
    raise ValueError(f"Unsupported strategy: {strategy}")


def _orthogonal_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(max(in_dim, out_dim), max(in_dim, out_dim), generator=generator)
    q, _ = torch.linalg.qr(raw)
    return q[:in_dim, :out_dim]


class LayerwiseAuxiliaryObjective(nn.Module):
    def __init__(
        self,
        layer_dims: list[int],
        num_classes: int,
        strategy: str,
        lookahead: int,
        sigma: float,
        include_output: bool,
        detach_target: bool,
        aux_dim: int = 0,
        loss_type: str = "cosine",
        projector_seed: int = 17,
    ) -> None:
        super().__init__()
        if not layer_dims:
            raise ValueError("Layerwise auxiliary loss requires at least one hidden state.")
        self.strategy = strategy
        self.lookahead = lookahead
        self.sigma = sigma
        self.include_output = include_output
        self.detach_target = detach_target
        self.loss_type = loss_type
        self.aux_dim = aux_dim or layer_dims[0]
        self.predictors = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.aux_dim)) for dim in layer_dims]
        )
        candidate_dims = list(layer_dims)
        if include_output:
            candidate_dims.append(num_classes)
        for idx, dim in enumerate(candidate_dims):
            projection = (
                torch.eye(dim, self.aux_dim)
                if dim == self.aux_dim
                else _orthogonal_projection(dim, self.aux_dim, seed=projector_seed + idx)
            )
            self.register_buffer(f"projection_{idx}", projection, persistent=False)

    def _project_candidate(self, tensor: torch.Tensor, candidate_index: int) -> torch.Tensor:
        projection = getattr(self, f"projection_{candidate_index}")
        return F.normalize(tensor @ projection, dim=-1)

    def _pairwise_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)
        if self.loss_type == "mse":
            return F.mse_loss(prediction, target)
        if self.loss_type == "cosine":
            return (1.0 - (prediction * target).sum(dim=-1)).mean()
        raise ValueError(f"Unsupported auxiliary loss: {self.loss_type}")

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        candidates = [self._project_candidate(hidden, idx) for idx, hidden in enumerate(hidden_states)]
        if self.include_output:
            candidates.append(self._project_candidate(logits, len(hidden_states)))
        losses: list[torch.Tensor] = []
        per_layer: dict[str, float] = {}
        total_candidates = len(candidates)
        for layer_index, hidden in enumerate(hidden_states):
            weights = compute_future_weights(
                total_candidates=total_candidates,
                layer_index=layer_index,
                strategy=self.strategy,
                lookahead=self.lookahead,
                sigma=self.sigma,
            ).to(hidden.device)
            if weights.numel() == 0:
                continue
            future_candidates = candidates[layer_index + 1 :]
            target = sum(weight * candidate for weight, candidate in zip(weights, future_candidates))
            if self.detach_target:
                target = target.detach()
            prediction = self.predictors[layer_index](hidden)
            layer_loss = self._pairwise_loss(prediction, target)
            losses.append(layer_loss)
            per_layer[f"layer_{layer_index}_aux_loss"] = float(layer_loss.detach().cpu())
        if not losses:
            zero = logits.new_tensor(0.0)
            return zero, {"active_aux_layers": 0.0}
        aux_loss = torch.stack(losses).mean()
        per_layer["active_aux_layers"] = float(len(losses))
        return aux_loss, per_layer
