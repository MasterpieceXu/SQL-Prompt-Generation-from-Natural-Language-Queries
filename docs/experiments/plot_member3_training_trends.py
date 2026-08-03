"""Generate report-ready Baseline and Member 3 V3 training trend plots."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comp9444-matplotlib")
)

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_LOG = PROJECT_ROOT / "results" / "baseline_training_log.csv"
V3_LOG = PROJECT_ROOT / "results" / "member3" / "v3_full_run" / "improved_training_log.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "figures" / "member3_training_trends.png"


def read_training_log(path: Path) -> list[dict[str, float]]:
    """Read a training log and convert every numeric field to float."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def style_axis(axis) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9ca3af")
    axis.spines["bottom"].set_color("#9ca3af")
    axis.tick_params(colors="#374151")


def annotate_endpoint(
    axis,
    epoch: float,
    value: float,
    label: str,
    color: str,
    offset: tuple[int, int] = (7, 6),
) -> None:
    axis.annotate(
        label,
        (epoch, value),
        xytext=offset,
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        weight="bold",
    )


def main() -> None:
    baseline = read_training_log(BASELINE_LOG)
    v3 = read_training_log(V3_LOG)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "text.color": "#111827",
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    figure.patch.set_facecolor("white")

    baseline_epochs = [row["epoch"] for row in baseline]
    baseline_train = [row["train_loss"] for row in baseline]
    baseline_validation = [row["val_loss"] for row in baseline]

    axes[0].plot(
        baseline_epochs,
        baseline_train,
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Train loss",
    )
    axes[0].plot(
        baseline_epochs,
        baseline_validation,
        color="#d97706",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Validation loss",
    )
    axes[0].set_title("A. Member 2 Baseline loss", loc="left", weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_xticks(baseline_epochs)
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(frameon=False, fontsize=8.5)
    annotate_endpoint(
        axes[0],
        baseline_epochs[-1],
        baseline_train[-1],
        f"{baseline_train[-1]:.3f}",
        "#1d4ed8",
        (7, 8),
    )
    annotate_endpoint(
        axes[0],
        baseline_epochs[-1],
        baseline_validation[-1],
        f"{baseline_validation[-1]:.3f}",
        "#b45309",
        (7, -12),
    )
    style_axis(axes[0])

    v3_epochs = [row["epoch"] for row in v3]
    v3_train = [row["train_loss"] for row in v3]
    v3_validation = [row["val_loss"] for row in v3]
    v3_exact_match = [row["val_exact_match"] * 100 for row in v3]

    axes[1].plot(
        v3_epochs,
        v3_train,
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Train loss",
    )
    axes[1].plot(
        v3_epochs,
        v3_validation,
        color="#d97706",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Validation loss",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("B. Member 3 V3 loss", loc="left", weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss (log scale)")
    axes[1].set_xticks(v3_epochs)
    axes[1].legend(frameon=False, fontsize=8.5)
    annotate_endpoint(
        axes[1],
        v3_epochs[-1],
        v3_train[-1],
        f"{v3_train[-1]:.3f}",
        "#1d4ed8",
        (7, 8),
    )
    annotate_endpoint(
        axes[1],
        v3_epochs[-1],
        v3_validation[-1],
        f"{v3_validation[-1]:.3f}",
        "#b45309",
        (7, -12),
    )
    style_axis(axes[1])

    axes[2].plot(
        v3_epochs,
        v3_exact_match,
        color="#059669",
        linewidth=2.5,
        marker="D",
        markersize=5,
        label="Validation exact match",
    )
    axes[2].fill_between(v3_epochs, v3_exact_match, color="#10b981", alpha=0.10)
    for epoch, value in zip(v3_epochs, v3_exact_match):
        axes[2].annotate(
            f"{value:.1f}%",
            (epoch, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color="#047857",
            fontsize=8,
        )
    axes[2].set_title("C. V3 validation task metric", loc="left", weight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Normalised exact match (%)")
    axes[2].set_xticks(v3_epochs)
    axes[2].set_ylim(0, 17.5)
    style_axis(axes[2])

    figure.suptitle(
        "COMP9444 Text-to-SQL training and validation trends",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.055,
        0.012,
        "Source: verified Kaggle logs. Baseline: 10 epochs, t5-small. V3: 8 epochs, FLAN-T5-base, 940-example validation generation each epoch.",
        color="#6b7280",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.16, wspace=0.30)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
