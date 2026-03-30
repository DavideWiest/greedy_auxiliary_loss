from __future__ import annotations

from collections import defaultdict
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


def summarize(results: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for result in results:
        grouped[result["run_name"].rsplit("_seed", maxsplit=1)[0]].append(result["val_acc"])
    summary: dict[str, dict[str, float]] = {}
    for key, values in grouped.items():
        summary[key] = {
            "mean_val_acc": mean(values),
            "std_val_acc": pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def build_base_config(
    run_name: str,
    seed: int,
    beta: float,
    strategy: str,
    lookahead: int,
    sigma: float,
    detach_target: bool,
    normalize_gradients: bool,
) -> RunConfig:
    return RunConfig(
        run_name=run_name,
        seed=seed,
        dataset=DatasetConfig(name="mnist", batch_size=256, num_workers=0),
        model=ModelConfig(name="mlp", hidden_dim=256, num_layers=4, dropout=0.1),
        optimizer=OptimizerConfig(lr=3e-4, weight_decay=1e-4, max_epochs=6, gradient_clip_val=1.0),
        auxiliary=AuxiliaryLossConfig(
            enabled=beta > 0.0,
            beta=beta,
            strategy=strategy,
            lookahead=lookahead,
            sigma=sigma,
            include_output=True,
            detach_target=detach_target,
            normalize_gradients=normalize_gradients,
            aux_dim=256,
        ),
    )


def main() -> None:
    seeds = [0, 1]
    results: list[dict] = []

    for seed in seeds:
        baseline = build_base_config(
            run_name=f"stage1_baseline_seed{seed}",
            seed=seed,
            beta=0.0,
            strategy="fixed",
            lookahead=1,
            sigma=1.0,
            detach_target=True,
            normalize_gradients=False,
        )
        results.append(run_experiment(baseline))

    strategy_grid = [
        ("fixed1", "fixed", 1, 1.0),
        ("fixed2", "fixed", 2, 1.0),
        ("gaussian", "gaussian", 2, 1.0),
        ("uniform", "uniform", 1, 1.0),
        ("output", "output", 1, 1.0),
        ("exp", "exponential", 2, 1.0),
    ]
    for label, strategy, lookahead, sigma in strategy_grid:
        for seed in seeds:
            config = build_base_config(
                run_name=f"stage1_strategy_{label}_seed{seed}",
                seed=seed,
                beta=0.4,
                strategy=strategy,
                lookahead=lookahead,
                sigma=sigma,
                detach_target=True,
                normalize_gradients=False,
            )
            results.append(run_experiment(config))

    summary = summarize(results)
    strategy_scores = {
        label: summary[f"stage1_strategy_{label}"]["mean_val_acc"]
        for label, *_ in strategy_grid
    }
    best_strategy_label = max(strategy_scores, key=strategy_scores.get)
    best_strategy_entry = next(entry for entry in strategy_grid if entry[0] == best_strategy_label)

    for detach_target in [True, False]:
        for seed in seeds:
            config = build_base_config(
                run_name=f"stage1_detach_{detach_target}_seed{seed}",
                seed=seed,
                beta=0.4,
                strategy=best_strategy_entry[1],
                lookahead=best_strategy_entry[2],
                sigma=best_strategy_entry[3],
                detach_target=detach_target,
                normalize_gradients=False,
            )
            results.append(run_experiment(config))

    for normalize_gradients in [False, True]:
        for seed in seeds:
            config = build_base_config(
                run_name=f"stage1_gradnorm_{normalize_gradients}_seed{seed}",
                seed=seed,
                beta=0.5,
                strategy=best_strategy_entry[1],
                lookahead=best_strategy_entry[2],
                sigma=best_strategy_entry[3],
                detach_target=True,
                normalize_gradients=normalize_gradients,
            )
            results.append(run_experiment(config))

    final_summary = summarize(results)
    detach_choice = max(
        ["stage1_detach_True", "stage1_detach_False"],
        key=lambda key: final_summary[key]["mean_val_acc"],
    )
    gradnorm_choice = max(
        ["stage1_gradnorm_False", "stage1_gradnorm_True"],
        key=lambda key: final_summary[key]["mean_val_acc"],
    )

    selection = {
        "best_strategy_label": best_strategy_label,
        "best_strategy": {
            "strategy": best_strategy_entry[1],
            "lookahead": best_strategy_entry[2],
            "sigma": best_strategy_entry[3],
        },
        "detach_target": detach_choice.endswith("True"),
        "normalize_gradients": gradnorm_choice.endswith("True"),
        "summary": final_summary,
    }
    write_json("results/stage1_selection.json", selection)


if __name__ == "__main__":
    main()
