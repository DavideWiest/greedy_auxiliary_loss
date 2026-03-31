from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev

from greedy_auxiliary_loss.config import (
    AuxiliaryLossConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    TrainerConfig,
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
            batch_size=64,
            num_workers=4,
            train_subset=1.0,
            val_subset=1.0,
            test_subset=1.0,
            image_size=96,
            normalization="imagenet",
        ),
        model=ModelConfig(
            name="resnet18",
            dropout=0.1,
            pretrained=True,
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            lr=3e-4,
            weight_decay=1e-4,
            max_epochs=2,
            gradient_clip_val=1.0,
            scheduler="cosine",
            min_lr=1e-5,
            warmup_epochs=1,
            label_smoothing=0.1,
        ),
        auxiliary=AuxiliaryLossConfig(
            enabled=beta > 0.0,
            beta=beta,
            strategy=strategy,
            lookahead=lookahead,
            sigma=sigma,
            include_output=True,
            detach_target=True,
            normalize_gradients=False,
            aux_dim=256,
        ),
        trainer=TrainerConfig(log_every_n_steps=20),
        output_dir="results",
    )


def main() -> None:
    strategy_grid = [
        ("fixed1", "fixed", 1, 1.0),
        ("fixed2", "fixed", 2, 1.0),
        ("gaussian", "gaussian", 2, 1.0),
        ("uniform", "uniform", 1, 1.0),
        ("output", "output", 1, 1.0),
        ("exp", "exponential", 2, 1.0),
    ]

    strategy_results: list[dict] = []
    strategy_results.append(
        run_experiment(
            build_config(
                run_name="cifar100_full_baseline_seed0",
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
                    run_name=f"cifar100_full_strategy_{label}_seed0",
                    seed=0,
                    beta=0.2,
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
                    run_name=f"cifar100_full_beta{beta:.1f}_seed0",
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
    confirm_betas = sorted(
        {
            0.0,
            best_beta["beta"],
        }
    )
    confirmation_results: list[dict] = []
    for beta in confirm_betas:
        confirmation_results.append(
            run_experiment(
                build_config(
                    run_name=f"cifar100_full_confirm_beta{beta:.1f}_seed1",
                    seed=1,
                    beta=beta,
                    strategy=best_strategy["strategy"],
                    lookahead=best_strategy["lookahead"],
                    sigma=best_strategy["sigma"],
                )
            )
        )

    summary = {
        "recipe": {
            "model": "resnet18",
            "pretrained": True,
            "image_size": 96,
            "batch_size": 64,
            "epochs": 2,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
        },
        "strategy_results": strategy_results,
        "strategy_summary": strategy_summary,
        "best_strategy": best_strategy,
        "beta_results": beta_results,
        "beta_summary": beta_summary,
        "best_beta": best_beta,
        "confirmation_results": confirmation_results,
        "confirmation_summary": aggregate(
            confirmation_results,
            key_fields=("beta", "strategy", "lookahead", "sigma"),
        ),
    }
    write_json("results/cifar100_full_budget_summary.json", summary)


if __name__ == "__main__":
    main()
