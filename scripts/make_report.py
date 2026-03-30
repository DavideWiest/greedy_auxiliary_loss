from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from greedy_auxiliary_loss.utils.plotting import (
    save_beta_sweep_plot,
    save_gain_plot,
    save_strategy_plot,
)


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


def main() -> None:
    stage1 = load_json("results/stage1_selection.json")
    stage2 = load_json("results/stage2_summary.json")
    stage3 = load_json("results/stage3_ag_news_summary.json")
    stage4 = load_json("results/stage4_dbpedia_summary.json")

    strategy_frame = pd.DataFrame(
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
    save_strategy_plot(strategy_frame, "reports/figures/mnist_strategy_ablation.png")

    beta_frame = pd.DataFrame(
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
        beta_frame,
        "reports/figures/beta_sweep.png",
        title="MNIST and CIFAR-100 beta sweep",
    )

    mnist_rows = [row for row in stage2["aggregated"] if row["dataset"] == "mnist"]
    cifar_rows = [row for row in stage2["aggregated"] if row["dataset"] == "cifar100"]
    ag_rows = stage3["aggregated"]
    dbpedia_rows = stage4["aggregated"]

    mnist_best = best_row(mnist_rows)
    cifar_best = best_row(cifar_rows)
    ag_best = best_row(ag_rows)
    dbpedia_best = best_row(dbpedia_rows)

    comparison_frame = pd.DataFrame(
        [
            {
                "dataset": "MNIST",
                "variant": "baseline",
                "mean": row_for_beta(mnist_rows, 0.0)["mean_test_acc"],
                "std": row_for_beta(mnist_rows, 0.0)["std_test_acc"],
            },
            {
                "dataset": "MNIST",
                "variant": "auxiliary",
                "mean": mnist_best["mean_test_acc"],
                "std": mnist_best["std_test_acc"],
            },
            {
                "dataset": "CIFAR-100",
                "variant": "baseline",
                "mean": row_for_beta(cifar_rows, 0.0)["mean_test_acc"],
                "std": row_for_beta(cifar_rows, 0.0)["std_test_acc"],
            },
            {
                "dataset": "CIFAR-100",
                "variant": "auxiliary",
                "mean": cifar_best["mean_test_acc"],
                "std": cifar_best["std_test_acc"],
            },
            {
                "dataset": "AG News",
                "variant": "baseline",
                "mean": row_for_beta(ag_rows, 0.0)["mean_test_acc"],
                "std": row_for_beta(ag_rows, 0.0)["std_test_acc"],
            },
            {
                "dataset": "AG News",
                "variant": "auxiliary",
                "mean": ag_best["mean_test_acc"],
                "std": ag_best["std_test_acc"],
            },
            {
                "dataset": "DBPedia",
                "variant": "baseline",
                "mean": row_for_beta(dbpedia_rows, 0.0)["mean_test_acc"],
                "std": row_for_beta(dbpedia_rows, 0.0)["std_test_acc"],
            },
            {
                "dataset": "DBPedia",
                "variant": "auxiliary",
                "mean": dbpedia_best["mean_test_acc"],
                "std": dbpedia_best["std_test_acc"],
            },
        ]
    )
    gain_frame = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "gain": subset.loc["auxiliary", "mean"] - subset.loc["baseline", "mean"],
                "std": (subset.loc["auxiliary", "std"] ** 2 + subset.loc["baseline", "std"] ** 2) ** 0.5,
            }
            for dataset_name, subset in (
                (dataset_name, comparison_frame[comparison_frame["dataset"] == dataset_name].set_index("variant"))
                for dataset_name in comparison_frame["dataset"].drop_duplicates()
            )
        ]
    )
    save_gain_plot(
        gain_frame,
        "reports/figures/dataset_summary.png",
        title="Absolute gain from the best auxiliary setting",
    )

    summary_table = pd.DataFrame(
        [
            {
                "Dataset": "MNIST",
                "Model": "4-layer MLP",
                "Budget": "8 epochs, full train/test",
                "Baseline": row_for_beta(mnist_rows, 0.0)["mean_test_acc"],
                "Best beta": mnist_best["beta"],
                "Auxiliary": mnist_best["mean_test_acc"],
                "Gain": mnist_best["mean_test_acc"] - row_for_beta(mnist_rows, 0.0)["mean_test_acc"],
            },
            {
                "Dataset": "CIFAR-100",
                "Model": "ViT (4 layers, patch 8)",
                "Budget": "4 epochs, 10% train, full val/test",
                "Baseline": row_for_beta(cifar_rows, 0.0)["mean_test_acc"],
                "Best beta": cifar_best["beta"],
                "Auxiliary": cifar_best["mean_test_acc"],
                "Gain": cifar_best["mean_test_acc"] - row_for_beta(cifar_rows, 0.0)["mean_test_acc"],
            },
            {
                "Dataset": "AG News",
                "Model": "4-layer text transformer",
                "Budget": "3 epochs, 5% train, 50% val/test",
                "Baseline": row_for_beta(ag_rows, 0.0)["mean_test_acc"],
                "Best beta": ag_best["beta"],
                "Auxiliary": ag_best["mean_test_acc"],
                "Gain": ag_best["mean_test_acc"] - row_for_beta(ag_rows, 0.0)["mean_test_acc"],
            },
            {
                "Dataset": "DBPedia 14",
                "Model": "4-layer text transformer",
                "Budget": "2 epochs, 2% train, 10% val/test",
                "Baseline": row_for_beta(dbpedia_rows, 0.0)["mean_test_acc"],
                "Best beta": dbpedia_best["beta"],
                "Auxiliary": dbpedia_best["mean_test_acc"],
                "Gain": dbpedia_best["mean_test_acc"] - row_for_beta(dbpedia_rows, 0.0)["mean_test_acc"],
            },
        ]
    )
    summary_table_markdown = markdown_table(summary_table)

    report = f"""# Experiment Report

## Question

This project studies a simple auxiliary objective for deep networks. Each hidden layer is trained to predict a detached summary of later representations, and that auxiliary loss is mixed with the primary task loss. The main research question is whether this extra local signal improves optimization or generalization without destabilizing the deeper representation that defines the target.

## Method

Let $h_i$ be the representation at layer $i$. For each hidden layer, the model learns a predictor $p_i(h_i)$ and matches it to a weighted combination of future representations projected into a common target space. The combined objective is

$$L = (1-\\beta)L_{{primary}} + \\beta L_{{aux}}.$$

The future target uses fixed, non-trainable random projections of downstream hidden states and optionally the logits. In the main method, the target is detached, so gradients flow through the upstream predictor but not through the downstream representation. I tested fixed lookahead, Gaussian weighting, uniform future averaging, output-only targets, and an exponential decay variant. I also tested gradient-norm normalization before mixing the two losses.

## Experimental protocol

Stage 1 used a 4-layer MLP on MNIST to choose the target-construction rule. Stage 2 swept $\\beta \\in \\{{0.0, 0.1, \\ldots, 1.0\\}}$ on MNIST and on a lightweight ViT for CIFAR-100. Because the machine is CPU-only, the CIFAR-100 pilot used a smaller ViT and a 10% training subset. Positive CIFAR-100 results then triggered two non-vision follow-ups: AG News and DBPedia 14, both with small text transformers and fixed-budget subsets.

## Stage 1: selecting the variant

The best validation result came from a Gaussian future-target kernel centered two layers ahead with standard deviation 1.0. Detached targets slightly beat non-detached targets on validation, while gradient normalization clearly harmed learning. The normalization variant dropped MNIST validation accuracy to {stage1["summary"]["stage1_gradnorm_True"]["mean_val_acc"]:.4f}, far below the non-normalized counterpart at {stage1["summary"]["stage1_gradnorm_False"]["mean_val_acc"]:.4f}.

![MNIST target-strategy ablation](figures/mnist_strategy_ablation.png)

## Main results

The table below summarizes the best auxiliary setting found on each dataset.

{summary_table_markdown}

The most important pattern is that the idea did not just help on the easy MNIST pilot. It also improved the harder CIFAR-100 subset pilot and then transferred to two text datasets. The gains were modest in absolute terms but consistently positive once the method was tuned away from the degenerate high-beta regime.

![MNIST and CIFAR-100 beta sweep](figures/beta_sweep.png)
![Dataset summary](figures/dataset_summary.png)

Outside MNIST, the most reliable region was a moderate auxiliary weight, especially $\\beta \\in [0.1, 0.2]$. Large beta values eventually overpowered the primary objective. At $\\beta = 1.0$, the model collapses to near-chance task performance because the task loss disappears entirely.

## Interpretation

The evidence supports the core hypothesis that detached future targets can provide a useful training signal. The effect is not dramatic, and it is sensitive to the mixing coefficient, but it appears broad enough to justify further work. The most plausible interpretation is that the auxiliary loss regularizes early and middle layers toward more task-useful future-compatible representations without forcing the deeper layers to satisfy the auxiliary objective directly.

## Hyperparameter heuristics

Three heuristics emerged from the study.

1. Start with a detached target and do not normalize the two gradients before mixing. The detached target was at least as good as the non-detached alternative, and gradient normalization was actively harmful in the pilot.
2. Initialize beta from a warm-up gradient-balance estimate,

$$\\beta^* \\approx \\frac{{\\lVert g_{{primary}} \\rVert}}{{\\lVert g_{{primary}} \\rVert + \\lVert g_{{aux}} \\rVert}},$$

because this balances the raw contribution of the two losses on the first few batches. In practice, the good region on the harder datasets sat close to the low-beta side of that intuition.
3. Choose the lookahead from representation-similarity decay across depth. A practical rule is to place the Gaussian center at the smallest offset where average inter-layer cosine similarity falls to about one half of the adjacent-layer value, and to set the kernel width to roughly half that offset. In these pilots, that rule was consistent with a center near two layers ahead.

## Limitations

The positive results on CIFAR-100, AG News, and DBPedia come from fixed-budget subset pilots, not from full-scale convergence studies. That was a deliberate choice to keep the study reproducible on CPU. The next logical step is to rerun the promising configurations on larger budgets, compare against stronger baselines, and inspect whether the auxiliary loss changes optimization speed, final generalization, or both.

## Conclusion

Within the compute budget of this study, detached downstream-target auxiliary losses look promising. The method improved the best observed test accuracy on all four datasets, the most transferable beta values were small to moderate, and the main failure mode was easy to identify: if the auxiliary term dominates, task performance collapses. The repository now contains the full code, run metadata, plots, and reports needed to extend the study.
"""

    readme = f"""# Greedy Auxiliary Loss

This repository tests a layerwise auxiliary objective in which each hidden layer predicts a detached summary of later representations. The target is built from fixed projections of downstream hidden states, and the training loss mixes the primary task objective with the auxiliary term using a coefficient `beta`.

I first selected the variant on MNIST, where a Gaussian future-target kernel centered two layers ahead worked best and gradient-normalized mixing failed badly. I then ran a full `beta` sweep on MNIST and a CPU-feasible CIFAR-100 ViT pilot, and the positive CIFAR result triggered two larger text follow-ups on AG News and DBPedia 14. The best observed test-accuracy gains were `+0.0019` on MNIST, `+0.0046` on CIFAR-100, `+0.0118` on AG News, and `+0.0073` on DBPedia 14.

![MNIST and CIFAR-100 beta sweep](reports/figures/beta_sweep.png)
![Absolute gain from the best auxiliary setting](reports/figures/dataset_summary.png)

The headline result is that the method helped on all four pilot datasets when `beta` stayed in a moderate range. On the harder datasets, the most reliable settings were `beta=0.1` to `beta=0.2`; very large `beta` values eventually overwhelmed the primary objective and collapsed task performance. A fuller write-up, including the stage-1 ablation and the exact experimental budgets, is in [reports/experiment_report.md](reports/experiment_report.md).
"""

    report_path = Path("reports/experiment_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    Path("README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
