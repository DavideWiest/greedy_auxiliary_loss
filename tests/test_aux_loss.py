from __future__ import annotations

import torch

from greedy_auxiliary_loss.losses import LayerwiseAuxiliaryObjective, compute_future_weights


def test_fixed_lookahead_chooses_single_future_index() -> None:
    weights = compute_future_weights(total_candidates=5, layer_index=1, strategy="fixed", lookahead=2, sigma=1.0)
    assert weights.tolist() == [0.0, 1.0, 0.0]


def test_uniform_weights_sum_to_one() -> None:
    weights = compute_future_weights(total_candidates=6, layer_index=2, strategy="uniform", lookahead=1, sigma=1.0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_detached_target_blocks_future_gradients() -> None:
    objective = LayerwiseAuxiliaryObjective(
        layer_dims=[4, 4],
        num_classes=3,
        strategy="fixed",
        lookahead=1,
        sigma=1.0,
        include_output=False,
        detach_target=True,
        aux_dim=4,
    )
    hidden_0 = torch.randn(2, 4, requires_grad=True)
    hidden_1 = torch.randn(2, 4, requires_grad=True)
    logits = torch.randn(2, 3, requires_grad=True)
    loss, _ = objective([hidden_0, hidden_1], logits)
    loss.backward()
    assert hidden_0.grad is not None
    assert hidden_1.grad is None
    assert logits.grad is None
