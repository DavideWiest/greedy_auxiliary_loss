from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from greedy_auxiliary_loss.config import (
    AuxiliaryLossConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
)
from greedy_auxiliary_loss.runner import run_experiment
from greedy_auxiliary_loss.utils.results import write_json


def aggregate(results: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for result in results:
        grouped[tuple(result[field] for field in key_fields)].append(result)
    aggregated: list[dict] = []
    for key, values in grouped.items():
        row = {field: value for field, value in zip(key_fields, key)}
        row["mean_test_acc"] = mean([item["test_acc"] for item in values])
        row["std_test_acc"] = pstdev([item["test_acc"] for item in values]) if len(values) > 1 else 0.0
        row["mean_val_acc"] = mean([item["val_acc"] for item in values])
        row["std_val_acc"] = pstdev([item["val_acc"] for item in values]) if len(values) > 1 else 0.0
        row["runs"] = len(values)
        aggregated.append(row)
    return aggregated


def load_stage1_selection() -> dict:
    path = Path("results/stage1_selection.json")
    if not path.exists():
        raise FileNotFoundError("Stage 1 selection file not found. Run scripts/run_stage1_pilot.py first.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mnist_config(beta: float, seed: int, selection: dict) -> RunConfig:
    return RunConfig(
        run_name=f"stage2_mnist_beta{beta:.1f}_seed{seed}",
        seed=seed,
        dataset=DatasetConfig(name="mnist", batch_size=256, num_workers=0),
        model=ModelConfig(name="mlp", hidden_dim=256, num_layers=4, dropout=0.1),
        optimizer=OptimizerConfig(lr=3e-4, weight_decay=1e-4, max_epochs=8),
        auxiliary=AuxiliaryLossConfig(
            enabled=beta > 0.0,
            beta=beta,
            strategy=selection["best_strategy"]["strategy"],
            lookahead=selection["best_strategy"]["lookahead"],
            sigma=selection["best_strategy"]["sigma"],
            include_output=True,
            detach_target=selection["detach_target"],
            normalize_gradients=selection["normalize_gradients"] and beta > 0.0,
            aux_dim=256,
        ),
    )


def cifar_config(beta: float, seed: int, selection: dict, epochs: int) -> RunConfig:
    return RunConfig(
        run_name=f"stage2_cifar100_beta{beta:.1f}_seed{seed}",
        seed=seed,
        dataset=DatasetConfig(
            name="cifar100",
            batch_size=128,
            num_workers=0,
            train_subset=0.10,
            val_subset=1.0,
            test_subset=1.0,
        ),
        model=ModelConfig(
            name="vit",
            hidden_dim=128,
            num_layers=4,
            dropout=0.1,
            patch_size=8,
            num_heads=4,
            mlp_ratio=4.0,
        ),
        optimizer=OptimizerConfig(lr=3e-4, weight_decay=5e-4, max_epochs=epochs),
        auxiliary=AuxiliaryLossConfig(
            enabled=beta > 0.0,
            beta=beta,
            strategy=selection["best_strategy"]["strategy"],
            lookahead=selection["best_strategy"]["lookahead"],
            sigma=selection["best_strategy"]["sigma"],
            include_output=True,
            detach_target=selection["detach_target"],
            normalize_gradients=False,
            aux_dim=128,
        ),
    )


def main() -> None:
    selection = load_stage1_selection()
    results: list[dict] = []

    for beta in [round(step * 0.1, 1) for step in range(11)]:
        for seed in [0, 1]:
            results.append(run_experiment(mnist_config(beta=beta, seed=seed, selection=selection)))

    for beta in [round(step * 0.1, 1) for step in range(11)]:
        results.append(run_experiment(cifar_config(beta=beta, seed=0, selection=selection, epochs=4)))

    cifar_summary = aggregate(
        [result for result in results if result["dataset"] == "cifar100"],
        key_fields=("dataset", "beta"),
    )
    cifar_best = max(cifar_summary, key=lambda row: row["mean_test_acc"])
    confirm_betas = sorted(
        {
            0.0,
            cifar_best["beta"],
            max(0.0, round(cifar_best["beta"] - 0.1, 1)),
            min(1.0, round(cifar_best["beta"] + 0.1, 1)),
        }
    )
    for beta in confirm_betas:
        if beta == 0.0:
            continue
        results.append(run_experiment(cifar_config(beta=beta, seed=1, selection=selection, epochs=4)))

    summary = {
        "selection": selection,
        "aggregated": aggregate(results, key_fields=("dataset", "beta")),
        "raw_results": results,
    }
    write_json("results/stage2_summary.json", summary)


if __name__ == "__main__":
    main()
