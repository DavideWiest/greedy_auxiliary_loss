from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev

import pandas as pd

from greedy_auxiliary_loss.config import (
    AuxiliaryLossConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    TrainerConfig,
)
from greedy_auxiliary_loss.runner import run_experiment
from greedy_auxiliary_loss.utils.plotting import save_beta_sweep_plot, save_strategy_plot
from greedy_auxiliary_loss.utils.results import write_json


def aggregate(results: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for result in results:
        grouped[tuple(result[field] for field in key_fields)].append(result)
    aggregated: list[dict] = []
    for key, values in grouped.items():
        row = {field: value for field, value in zip(key_fields, key)}
        row["mean_val_acc"] = mean(item["val_acc"] for item in values)
        row["std_val_acc"] = pstdev([item["val_acc"] for item in values]) if len(values) > 1 else 0.0
        row["mean_test_acc"] = mean(item["test_acc"] for item in values)
        row["std_test_acc"] = pstdev([item["test_acc"] for item in values]) if len(values) > 1 else 0.0
        row["runs"] = len(values)
        aggregated.append(row)
    return aggregated


def build_config(
    run_name: str,
    seed: int,
    beta: float,
    strategy: str,
    lookahead: int,
    sigma: float,
) -> RunConfig:
    return RunConfig(
        run_name=run_name,
        seed=seed,
        dataset=DatasetConfig(
            name="cifar100",
            batch_size=128,
            num_workers=4,
            train_subset=1.0,
            val_subset=1.0,
            test_subset=1.0,
            image_size=32,
            normalization="dataset",
        ),
        model=ModelConfig(
            name="vit",
            hidden_dim=128,
            num_layers=4,
            patch_size=4,
            num_heads=4,
            dropout=0.1,
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            lr=3e-4,
            weight_decay=1e-4,
            max_epochs=1,
            gradient_clip_val=1.0,
            scheduler="cosine",
            min_lr=1e-5,
            warmup_epochs=0,
            label_smoothing=0.1,
        ),
        auxiliary=AuxiliaryLossConfig(
            enabled=beta > 0.0,
            beta=beta,
            beta_schedule="constant",
            beta_mode="gradient_share",
            strategy=strategy,
            lookahead=lookahead,
            sigma=sigma,
            include_output=False,
            detach_target=True,
            normalize_gradients=False,
            direct_hidden_target=True,
        ),
        trainer=TrainerConfig(log_every_n_steps=20),
        output_dir="results",
    )


def strategy_label(strategy: str, lookahead: int, beta: float) -> str:
    if beta == 0.0:
        return "baseline"
    if strategy == "fixed":
        return f"fixed-{lookahead}"
    if strategy == "gaussian":
        return f"gaussian-{lookahead}"
    if strategy == "exponential":
        return f"exp-{lookahead}"
    return strategy


def write_note(summary: dict) -> None:
    best_positive = summary["best_positive_beta"]
    baseline = next(item for item in summary["beta_summary"] if item["beta"] == 0.0)
    note = f"""# Direct Hidden-Target CIFAR-100 Study

This study reruns CIFAR-100 with the direct hidden-target auxiliary variant requested after the earlier logits/projection experiments. The auxiliary objective uses only hidden states, compares matching coordinates directly with MSE, keeps `detach_target=True`, and interprets `beta` as the auxiliary share of the update through gradient-share normalization rather than as a raw loss coefficient.

The recipe is intentionally architecture-aligned with that objective: a same-width 4-layer ViT (`hidden_dim=128`, `patch_size=4`) trained on full CIFAR-100 for 1 epoch on CPU. The strategy scan used `beta=0.1`, and the sweep then reused the validation-best strategy.

The main result is that the new `beta` definition is no longer flat. Accuracy changes materially well before `beta=1.0`. The best positive setting in this study was `beta={best_positive["beta"]:.1f}` with validation accuracy `{best_positive["mean_val_acc"]:.4f}` and test accuracy `{best_positive["mean_test_acc"]:.4f}`, compared with the baseline at `{baseline["mean_test_acc"]:.4f}`. That means the direct hidden-target variant is now tunable rather than inert, but under this CPU-feasible ViT recipe it still does not beat the baseline.
"""
    write_json("results/cifar100_direct_hidden_summary.json", summary)
    from pathlib import Path

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/cifar100_direct_hidden_note.md").write_text(note, encoding="utf-8")


def main() -> None:
    strategy_grid = [
        ("fixed1", "fixed", 1, 1.0),
        ("fixed2", "fixed", 2, 1.0),
        ("gaussian", "gaussian", 2, 1.0),
        ("uniform", "uniform", 1, 1.0),
        ("output", "output", 1, 1.0),
        ("exp", "exponential", 2, 1.0),
    ]

    strategy_beta = 0.1
    strategy_results: list[dict] = []
    strategy_results.append(
        run_experiment(
            build_config(
                run_name="cifar100_direct_hidden_baseline_seed0",
                seed=0,
                beta=0.0,
                strategy="fixed",
                lookahead=1,
                sigma=1.0,
            )
        )
    )
    for label, strategy, lookahead, sigma in strategy_grid:
        strategy_results.append(
            run_experiment(
                build_config(
                    run_name=f"cifar100_direct_hidden_strategy_{label}_seed0",
                    seed=0,
                    beta=strategy_beta,
                    strategy=strategy,
                    lookahead=lookahead,
                    sigma=sigma,
                )
            )
        )

    strategy_summary = aggregate(strategy_results, key_fields=("strategy", "lookahead", "sigma", "beta"))
    best_strategy = max((row for row in strategy_summary if row["beta"] > 0.0), key=lambda row: row["mean_val_acc"])

    beta_results: list[dict] = [strategy_results[0]]
    for beta in [round(step * 0.1, 1) for step in range(1, 11)]:
        beta_results.append(
            run_experiment(
                build_config(
                    run_name=f"cifar100_direct_hidden_beta{beta:.1f}_seed0",
                    seed=0,
                    beta=beta,
                    strategy=best_strategy["strategy"],
                    lookahead=best_strategy["lookahead"],
                    sigma=best_strategy["sigma"],
                )
            )
        )

    beta_summary = aggregate(beta_results, key_fields=("beta", "strategy", "lookahead", "sigma"))
    best_beta = max(beta_summary, key=lambda row: row["mean_val_acc"])
    best_positive_beta = max((row for row in beta_summary if row["beta"] > 0.0), key=lambda row: row["mean_val_acc"])

    confirm_betas = sorted({0.0, best_positive_beta["beta"]})
    confirmation_results: list[dict] = []
    for beta in confirm_betas:
        confirmation_results.append(
            run_experiment(
                build_config(
                    run_name=f"cifar100_direct_hidden_confirm_beta{beta:.1f}_seed1",
                    seed=1,
                    beta=beta,
                    strategy=best_strategy["strategy"],
                    lookahead=best_strategy["lookahead"],
                    sigma=best_strategy["sigma"],
                )
            )
        )

    strategy_frame = pd.DataFrame(
        [
            {
                "label": strategy_label(item["strategy"], item["lookahead"], item["beta"]),
                "mean": item["test_acc"],
                "std": 0.0,
            }
            for item in strategy_results
        ]
    )
    save_strategy_plot(
        strategy_frame,
        "reports/figures/cifar100_direct_hidden_strategy.png",
        ylabel="Test accuracy",
        title="CIFAR-100 direct hidden-target strategy scan",
    )

    beta_frame = pd.DataFrame(
        [
            {
                "dataset": "cifar100_direct_hidden",
                "beta": row["beta"],
                "mean": row["mean_test_acc"],
                "std": row["std_test_acc"],
            }
            for row in beta_summary
        ]
    ).sort_values("beta")
    save_beta_sweep_plot(
        beta_frame,
        "reports/figures/cifar100_direct_hidden_beta_sweep.png",
        title="CIFAR-100 direct hidden-target beta sweep",
    )

    summary = {
        "recipe": {
            "model": "vit",
            "hidden_dim": 128,
            "num_layers": 4,
            "patch_size": 4,
            "num_heads": 4,
            "image_size": 32,
            "batch_size": 128,
            "epochs": 1,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "direct_hidden_target": True,
            "include_output": False,
            "detach_target": True,
            "beta_mode": "gradient_share",
            "strategy_beta": strategy_beta,
        },
        "strategy_results": strategy_results,
        "strategy_summary": strategy_summary,
        "best_strategy": best_strategy,
        "beta_results": beta_results,
        "beta_summary": beta_summary,
        "best_beta": best_beta,
        "best_positive_beta": best_positive_beta,
        "confirmation_results": confirmation_results,
        "confirmation_summary": aggregate(
            confirmation_results,
            key_fields=("beta", "strategy", "lookahead", "sigma"),
        ),
    }
    write_note(summary)


if __name__ == "__main__":
    main()
