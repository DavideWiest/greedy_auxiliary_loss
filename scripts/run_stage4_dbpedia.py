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


def load_stage1_selection() -> dict:
    path = Path("results/stage1_selection.json")
    if not path.exists():
        raise FileNotFoundError("Stage 1 selection file not found. Run scripts/run_stage1_pilot.py first.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate(results: list[dict]) -> list[dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for result in results:
        grouped[result["beta"]].append(result)
    aggregated = []
    for beta, values in sorted(grouped.items()):
        aggregated.append(
            {
                "beta": beta,
                "mean_val_acc": mean([item["val_acc"] for item in values]),
                "std_val_acc": pstdev([item["val_acc"] for item in values]) if len(values) > 1 else 0.0,
                "mean_test_acc": mean([item["test_acc"] for item in values]),
                "std_test_acc": pstdev([item["test_acc"] for item in values]) if len(values) > 1 else 0.0,
                "runs": len(values),
            }
        )
    return aggregated


def dbpedia_config(beta: float, seed: int, selection: dict) -> RunConfig:
    return RunConfig(
        run_name=f"stage4_dbpedia_beta{beta:.1f}_seed{seed}",
        seed=seed,
        dataset=DatasetConfig(
            name="dbpedia_14",
            batch_size=64,
            num_workers=0,
            train_subset=0.02,
            val_subset=0.10,
            test_subset=0.10,
        ),
        model=ModelConfig(
            name="text_transformer",
            hidden_dim=128,
            num_layers=4,
            dropout=0.1,
            num_heads=4,
            mlp_ratio=4.0,
        ),
        optimizer=OptimizerConfig(lr=3e-4, weight_decay=1e-4, max_epochs=2),
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
    for beta in [0.0, 0.2]:
        for seed in [0, 1]:
            results.append(run_experiment(dbpedia_config(beta=beta, seed=seed, selection=selection)))

    summary = {
        "selection": selection,
        "aggregated": aggregate(results),
        "raw_results": results,
    }
    write_json("results/stage4_dbpedia_summary.json", summary)


if __name__ == "__main__":
    main()
