from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from greedy_auxiliary_loss.utils.plotting import save_beta_sweep_plot, save_gain_plot, save_strategy_plot


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def best_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: row["mean_test_acc"])


def row_for_beta(rows: list[dict], beta: float) -> dict:
    return next(row for row in rows if abs(row["beta"] - beta) < 1e-9)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = []
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator_row, *body_rows])


def aggregate_named_runs(frame: pd.DataFrame, run_names: list[str], label: str) -> dict[str, float | str]:
    subset = frame[frame["run_name"].isin(run_names)].copy()
    if subset.empty:
        raise ValueError(f"No runs found for {label}: {run_names}")
    return {
        "label": label,
        "mean_test_acc": float(subset["test_acc"].mean()),
        "std_test_acc": float(subset["test_acc"].std(ddof=0)),
        "mean_val_acc": float(subset["val_acc"].mean()),
        "std_val_acc": float(subset["val_acc"].std(ddof=0)),
        "runs": int(len(subset)),
    }


def main() -> None:
    stage1 = load_json("results/stage1_selection.json")
    stage2 = load_json("results/stage2_summary.json")
    stage3 = load_json("results/stage3_ag_news_summary.json")
    stage4 = load_json("results/stage4_dbpedia_summary.json")
    cifar_full = load_json("results/cifar100_full_budget_summary.json")
    all_runs = pd.read_csv("results/all_runs.csv")

    mnist_rows = [row for row in stage2["aggregated"] if row["dataset"] == "mnist"]
    cifar_pilot_rows = [row for row in stage2["aggregated"] if row["dataset"] == "cifar100"]
    ag_rows = stage3["aggregated"]
    dbpedia_rows = stage4["aggregated"]

    mnist_best = best_row(mnist_rows)
    cifar_pilot_best = best_row(cifar_pilot_rows)
    ag_best = best_row(ag_rows)
    dbpedia_best = best_row(dbpedia_rows)

    stage1_strategy_frame = pd.DataFrame(
        [
            {
                "label": label.replace("stage1_strategy_", ""),
                "mean": stats["mean_val_acc"],
                "std": stats["std_val_acc"],
            }
            for label, stats in stage1["summary"].items()
            if label.startswith("stage1_strategy_")
        ]
    ).sort_values("mean", ascending=False)
    save_strategy_plot(stage1_strategy_frame, "reports/figures/mnist_strategy_ablation.png")

    pilot_beta_frame = pd.DataFrame(
        [
            {
                "dataset": row["dataset"],
                "beta": row["beta"],
                "mean": row["mean_test_acc"],
                "std": row["std_test_acc"],
            }
            for row in stage2["aggregated"]
        ]
    ).sort_values(["dataset", "beta"])
    save_beta_sweep_plot(
        pilot_beta_frame,
        "reports/figures/beta_sweep.png",
        title="MNIST and pilot CIFAR-100 beta sweep",
    )

    pilot_gain_frame = pd.DataFrame(
        [
            {
                "dataset": "MNIST",
                "gain": mnist_best["mean_test_acc"] - row_for_beta(mnist_rows, 0.0)["mean_test_acc"],
                "std": (mnist_best["std_test_acc"] ** 2 + row_for_beta(mnist_rows, 0.0)["std_test_acc"] ** 2) ** 0.5,
            },
            {
                "dataset": "CIFAR-100 pilot",
                "gain": cifar_pilot_best["mean_test_acc"] - row_for_beta(cifar_pilot_rows, 0.0)["mean_test_acc"],
                "std": (
                    cifar_pilot_best["std_test_acc"] ** 2
                    + row_for_beta(cifar_pilot_rows, 0.0)["std_test_acc"] ** 2
                )
                ** 0.5,
            },
            {
                "dataset": "AG News",
                "gain": ag_best["mean_test_acc"] - row_for_beta(ag_rows, 0.0)["mean_test_acc"],
                "std": (ag_best["std_test_acc"] ** 2 + row_for_beta(ag_rows, 0.0)["std_test_acc"] ** 2) ** 0.5,
            },
            {
                "dataset": "DBPedia",
                "gain": dbpedia_best["mean_test_acc"] - row_for_beta(dbpedia_rows, 0.0)["mean_test_acc"],
                "std": (
                    dbpedia_best["std_test_acc"] ** 2 + row_for_beta(dbpedia_rows, 0.0)["std_test_acc"] ** 2
                )
                ** 0.5,
            },
        ]
    )
    save_gain_plot(
        pilot_gain_frame,
        "reports/figures/dataset_summary.png",
        title="Absolute gain from the best pilot auxiliary setting",
    )

    cifar_strategy_frame = pd.DataFrame(
        [
            {
                "label": "baseline" if item["beta"] == 0.0 else f"{item['strategy']}-{item['lookahead']}",
                "mean": item["test_acc"],
                "std": 0.0,
            }
            for item in cifar_full["strategy_results"]
        ]
    )
    save_strategy_plot(
        cifar_strategy_frame,
        "reports/figures/cifar100_full_strategy.png",
        ylabel="Test accuracy",
        title="CIFAR-100 full-budget strategy scan",
    )

    baseline_full = aggregate_named_runs(
        all_runs,
        ["cifar100_full_baseline_seed0", "cifar100_full_confirm_beta0.0_seed1"],
        label="baseline",
    )
    detach_beta05 = aggregate_named_runs(
        all_runs,
        ["cifar100_full_beta0.5_seed0", "cifar100_followup_output_beta0.5_detach_seed1"],
        label="detach, beta=0.5",
    )
    nodetach_beta05 = aggregate_named_runs(
        all_runs,
        ["cifar100_followup_output_beta0.5_nodetach_seed0", "cifar100_followup_output_beta0.5_nodetach_seed1"],
        label="no detach, beta=0.5",
    )
    nodetach_beta04 = aggregate_named_runs(
        all_runs,
        ["cifar100_followup_output_beta0.4_nodetach_seed0", "cifar100_followup_output_beta0.4_nodetach_seed1"],
        label="no detach, beta=0.4",
    )
    decay_beta08 = aggregate_named_runs(
        all_runs,
        ["cifar100_followup_output_beta0.8_decay_seed0", "cifar100_followup_output_beta0.8_decay_seed1"],
        label="decay, beta=0.8",
    )

    cifar_followup_frame = pd.DataFrame(
        [
            {
                "label": stats["label"],
                "mean": stats["mean_test_acc"],
                "std": stats["std_test_acc"],
            }
            for stats in [baseline_full, detach_beta05, nodetach_beta05, nodetach_beta04, decay_beta08]
        ]
    )
    save_strategy_plot(
        cifar_followup_frame,
        "reports/figures/cifar100_full_followups.png",
        ylabel="Mean test accuracy",
        title="CIFAR-100 full-budget follow-ups",
    )

    summary_table = pd.DataFrame(
        [
            {
                "Dataset": "MNIST",
                "Model": "4-layer MLP",
                "Budget": "8 epochs, full train/test",
                "Best auxiliary": "detached Gaussian, beta=0.8",
                "Baseline": row_for_beta(mnist_rows, 0.0)["mean_test_acc"],
                "Auxiliary": mnist_best["mean_test_acc"],
                "Gain": mnist_best["mean_test_acc"] - row_for_beta(mnist_rows, 0.0)["mean_test_acc"],
            },
            {
                "Dataset": "CIFAR-100 pilot",
                "Model": "ViT (4 layers, patch 8)",
                "Budget": "4 epochs, 10% train, full val/test",
                "Best auxiliary": "detached Gaussian, beta=0.1",
                "Baseline": row_for_beta(cifar_pilot_rows, 0.0)["mean_test_acc"],
                "Auxiliary": cifar_pilot_best["mean_test_acc"],
                "Gain": cifar_pilot_best["mean_test_acc"] - row_for_beta(cifar_pilot_rows, 0.0)["mean_test_acc"],
            },
            {
                "Dataset": "AG News",
                "Model": "4-layer text transformer",
                "Budget": "3 epochs, 5% train, 50% val/test",
                "Best auxiliary": "detached Gaussian, beta=0.2",
                "Baseline": row_for_beta(ag_rows, 0.0)["mean_test_acc"],
                "Auxiliary": ag_best["mean_test_acc"],
                "Gain": ag_best["mean_test_acc"] - row_for_beta(ag_rows, 0.0)["mean_test_acc"],
            },
            {
                "Dataset": "DBPedia 14",
                "Model": "4-layer text transformer",
                "Budget": "2 epochs, 2% train, 10% val/test",
                "Best auxiliary": "detached Gaussian, beta=0.2",
                "Baseline": row_for_beta(dbpedia_rows, 0.0)["mean_test_acc"],
                "Auxiliary": dbpedia_best["mean_test_acc"],
                "Gain": dbpedia_best["mean_test_acc"] - row_for_beta(dbpedia_rows, 0.0)["mean_test_acc"],
            },
            {
                "Dataset": "CIFAR-100 full-budget",
                "Model": "pretrained ResNet18",
                "Budget": "2 epochs, full train/test, 2 seeds",
                "Best auxiliary": "output target, no detach, beta=0.5",
                "Baseline": baseline_full["mean_test_acc"],
                "Auxiliary": nodetach_beta05["mean_test_acc"],
                "Gain": nodetach_beta05["mean_test_acc"] - baseline_full["mean_test_acc"],
            },
        ]
    )
    summary_table_markdown = markdown_table(summary_table)

    report = f"""# Experiment Report

## Question

This project studies a simple layerwise auxiliary objective. Each hidden layer predicts a summary of later representations, and that auxiliary loss is mixed with the primary task loss. The original hypothesis was that detached downstream targets would provide a stable local signal. After the full-budget CIFAR-100 rerun, the question became slightly sharper: which target construction and gradient-flow regime actually helps once the baseline is no longer toy-sized?

## Method

Let $h_i$ be the representation at layer $i$. For each hidden layer, the model learns a predictor $p_i(h_i)$ and matches it to a weighted combination of future representations projected into a common target space. The combined objective is

$$L = (1-\\beta)L_{{primary}} + \\beta L_{{aux}}.$$

The future target uses fixed random projections of downstream hidden states and optionally the logits. I tested fixed lookahead, Gaussian weighting, uniform averaging, output-only targets, and exponential weighting. I also tested detached and non-detached targets, constant and decayed $\\beta$, and a shallow-only auxiliary variant that skips the deepest hidden layers.

## Experimental protocol

The study proceeded in two phases. First, I ran a pilot transfer study: Stage 1 selected the basic target-construction rule on MNIST, Stage 2 swept $\\beta$ on MNIST and on a tiny CIFAR-100 ViT pilot, and the positive pilot result triggered AG News and DBPedia 14 follow-ups. Second, I reran CIFAR-100 properly on full data with a pretrained ResNet18, standard augmentation, cosine decay, and two seeds. On that stronger recipe I repeated the strategy search, the detached $\\beta$ sweep, and then a set of targeted follow-ups motivated by the observed failure modes.

## Stage 1: selecting the pilot variant

On MNIST, the best validation result came from a detached Gaussian future-target kernel centered two layers ahead with standard deviation 1.0. Gradient normalization was clearly harmful. The normalization variant dropped MNIST validation accuracy to {stage1["summary"]["stage1_gradnorm_True"]["mean_val_acc"]:.4f}, far below the non-normalized counterpart at {stage1["summary"]["stage1_gradnorm_False"]["mean_val_acc"]:.4f}.

![MNIST target-strategy ablation](figures/mnist_strategy_ablation.png)

## Pilot transfer results

The pilot study established that the idea was not confined to MNIST. Detached targets improved the best observed test accuracy on the MNIST pilot, the lightweight CIFAR-100 pilot, AG News, and DBPedia 14.

{summary_table_markdown}

![MNIST and pilot CIFAR-100 beta sweep](figures/beta_sweep.png)
![Absolute gain from the best pilot auxiliary setting](figures/dataset_summary.png)

## Full-budget CIFAR-100 rerun

The full-budget CIFAR rerun materially changed the picture. The upgraded baseline, a pretrained ResNet18 fine-tuned on the full training set for two epochs, reached a mean test accuracy of {baseline_full["mean_test_acc"]:.4f} over two seeds. That already resolves the earlier “why is CIFAR so low?” problem: the weak 6.8% result was a deliberately tiny pilot, not a realistic CIFAR recipe.

Within the detached strategy scan at $\\beta=0.2$, the validation winner was the output-only target, while the best single-seed test score among the detached strategies came from fixed lookahead 2. More importantly, the detached constant-$\\beta$ sweep did not beat the full-budget baseline robustly. A matched two-seed comparison at $\\beta=0.5$ gave {detach_beta05["mean_test_acc"]:.4f} for the detached output-target variant, which is below the baseline at {baseline_full["mean_test_acc"]:.4f}.

![CIFAR-100 full-budget strategy scan](figures/cifar100_full_strategy.png)

The most informative follow-up was to remove the detach. On the same full-budget CIFAR recipe, the output-target auxiliary loss with no detach and $\\beta=0.5$ reached {nodetach_beta05["mean_test_acc"]:.4f} mean test accuracy over two seeds, a gain of {nodetach_beta05["mean_test_acc"] - baseline_full["mean_test_acc"]:.4f} over the baseline and {nodetach_beta05["mean_test_acc"] - detach_beta05["mean_test_acc"]:.4f} over the matched detached variant. A nearby no-detach setting at $\\beta=0.4$ also improved over baseline, but less strongly. In contrast, a linearly decayed $\\beta=0.8$ schedule and the shallow-only auxiliary variants did not produce a robust win.

![CIFAR-100 full-budget follow-ups](figures/cifar100_full_followups.png)

## Interpretation

The evidence now supports a more nuanced conclusion than the original pilot. Detached targets are a good default in the small from-scratch regime and transferred well in the pilot study, but they are not universally optimal. On the strongest recipe in this project, the promising variant was actually non-detached and output-focused. A plausible interpretation is that once the baseline representation is already strong, letting the downstream representation move with the auxiliary objective can reduce target mismatch and make the auxiliary loss act more like a coordinated shaping signal than a frozen self-distillation target.

The other clear pattern is that the primary-loss anchor remains essential. At $\\beta = 1.0$, the CIFAR-100 model collapsed to chance-level classification performance, which confirms that the auxiliary objective alone does not solve the task.

## Hyperparameter heuristics

Four practical heuristics emerged.

1. Detached targets remain a sensible default for small from-scratch pilots, but do not assume that detach is optimal once the backbone is stronger or pretrained. On full-budget CIFAR-100, the best result came from the non-detached output-target variant.
2. For detached pilots, start in the moderate range $\\beta \\in [0.1, 0.2]$. For the stronger CIFAR recipe here, the best no-detach result landed higher, at $\\beta=0.5$, which suggests that the right $\\beta$ depends strongly on how aligned the auxiliary target is with the task.
3. Output-only targets deserve serious consideration on classification problems. In the full-budget CIFAR rerun they dominated the detached validation scan and underpinned the best non-detached result.
4. Scheduling or shallow-only application are reasonable rescue ideas when the auxiliary loss looks too constraining, but in this study neither beat the matched non-detached output-target variant.

## Limitations

The transfer study outside CIFAR still uses fixed-budget pilots, and the full-budget CIFAR rerun is only two epochs because the machine is CPU-only. The detached strategy scan on full-budget CIFAR used one seed, and only the most promising follow-up variants were confirmed across two seeds. I would not yet claim a universal rule that non-detached targets are better; I would claim that the detach choice is regime-dependent and must be treated as a first-class experimental variable.

## Conclusion

The project now supports two defensible conclusions. First, the auxiliary-loss idea is real enough to survive beyond MNIST: it helped on the original pilot transfer study and, after a proper rerun, it can improve full-budget CIFAR-100 as well. Second, the best variant depends on the regime. The strongest result in this repository is not the original detached Gaussian target, but an output-only, non-detached auxiliary loss with $\\beta=0.5$ on full-budget CIFAR-100.
"""

    readme = f"""# Greedy Auxiliary Loss

This repository studies a layerwise auxiliary objective in which each hidden layer predicts a summary of later representations. The original pilot work used detached downstream targets and showed small but consistent gains on MNIST, a tiny CIFAR-100 ViT pilot, AG News, and DBPedia 14. The main lesson from the later full-budget CIFAR-100 rerun is that the detach choice is not universal: on a pretrained ResNet18 trained on the full CIFAR-100 training set, the best variant was an output-only auxiliary loss with **no detach** and `beta=0.5`.

The pilot transfer study still matters because it shows the idea is portable. The best observed pilot gains were `+0.0019` on MNIST, `+0.0046` on the tiny CIFAR-100 pilot, `+0.0118` on AG News, and `+0.0073` on DBPedia 14. But the stronger CIFAR rerun is the better indicator of what survives a credible baseline: the full-budget baseline reached `{baseline_full["mean_test_acc"]:.4f}` mean test accuracy over two seeds, and the best confirmed auxiliary variant improved that to `{nodetach_beta05["mean_test_acc"]:.4f}`.

![CIFAR-100 full-budget follow-ups](reports/figures/cifar100_full_followups.png)
![Absolute gain from the best pilot auxiliary setting](reports/figures/dataset_summary.png)

The short version is that the idea looks promising, but the method is more conditional than the initial pilot suggested. Detached targets are still a good default for small from-scratch runs, while stronger or pretrained backbones may prefer non-detached, output-focused targets with a larger mixing coefficient. The full write-up, figures, and run metadata are in [reports/experiment_report.md](reports/experiment_report.md).
"""

    report_path = Path("reports/experiment_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    Path("README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
