from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_paper_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.18,
        }
    )


def save_strategy_plot(frame: pd.DataFrame, path: str | Path) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    palette = sns.color_palette("Set2", n_colors=max(frame["label"].nunique(), 3))
    sns.barplot(data=frame, x="label", y="mean", hue="label", palette=palette, dodge=False, ax=ax)
    ax.errorbar(
        x=range(len(frame)),
        y=frame["mean"],
        yerr=frame["std"].fillna(0.0),
        fmt="none",
        ecolor="#3C3C3C",
        capsize=3,
        linewidth=1.0,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Validation accuracy")
    ax.tick_params(axis="x", rotation=18)
    lower = max(0.0, float(frame["mean"].min() - 0.0015))
    upper = min(1.0, float(frame["mean"].max() + 0.0015))
    ax.set_ylim(lower, upper)
    if ax.legend_ is not None:
        ax.legend_.remove()
    fig.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_beta_sweep_plot(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    set_paper_style()
    dataset_order = list(dict.fromkeys(frame["dataset"].tolist()))
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(3.8 * len(dataset_order), 3.2), sharex=True)
    if len(dataset_order) == 1:
        axes = [axes]
    palette = sns.color_palette("colorblind", n_colors=max(len(dataset_order), 2))
    for axis, dataset_name, color in zip(axes, dataset_order, palette):
        subset = frame[frame["dataset"] == dataset_name].sort_values("beta")
        axis.plot(
            subset["beta"],
            subset["mean"],
            marker="o",
            markersize=5.2,
            linewidth=1.8,
            color=color,
        )
        axis.fill_between(
            subset["beta"],
            subset["mean"] - subset["std"].fillna(0.0),
            subset["mean"] + subset["std"].fillna(0.0),
            alpha=0.16,
            color=color,
        )
        axis.set_title(str(dataset_name).replace("cifar100", "CIFAR-100").replace("mnist", "MNIST"))
        axis.set_xlabel(r"$\beta$")
        axis.set_ylabel("Test accuracy")
        axis.set_xticks(sorted(subset["beta"].unique()))
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_dataset_comparison_plot(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    palette = sns.color_palette(["#3B7EA1", "#D97B29"])
    order = list(dict.fromkeys(frame["dataset"].tolist()))
    sns.barplot(
        data=frame,
        x="dataset",
        y="mean",
        hue="variant",
        palette=palette,
        order=order,
        ax=ax,
    )
    variants = ["baseline", "auxiliary"]
    for dataset_index, dataset_name in enumerate(order):
        subset = frame[frame["dataset"] == dataset_name].set_index("variant")
        for variant_index, variant_name in enumerate(variants):
            x_position = dataset_index + (-0.2 if variant_name == "baseline" else 0.2)
            ax.errorbar(
                x=x_position,
                y=subset.loc[variant_name, "mean"],
                yerr=subset.loc[variant_name, "std"],
                fmt="none",
                ecolor="#2F2F2F",
                linewidth=1.0,
                capsize=3,
            )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Test accuracy")
    ax.legend(title="")
    fig.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_gain_plot(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    palette = sns.color_palette("Set2", n_colors=max(len(frame), 3))
    sns.barplot(data=frame, x="dataset", y="gain", hue="dataset", palette=palette, dodge=False, ax=ax)
    ax.errorbar(
        x=range(len(frame)),
        y=frame["gain"],
        yerr=frame["std"].fillna(0.0),
        fmt="none",
        ecolor="#2F2F2F",
        linewidth=1.0,
        capsize=3,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Absolute gain in test accuracy")
    if ax.legend_ is not None:
        ax.legend_.remove()
    fig.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
