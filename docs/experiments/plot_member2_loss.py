"""Generate the report-ready T5-small baseline loss curve."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comp9444-matplotlib")
)

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Copied from results/baseline_training_log.csv in the verified Kaggle output.
EPOCHS = list(range(1, 11))
TRAIN_LOSS = [
    1.0609924704470532,
    0.5774083773902756,
    0.4639583017597807,
    0.40063822138261923,
    0.35707026592832297,
    0.32537521619666765,
    0.2991999267778815,
    0.2793937857995959,
    0.2625532419955794,
    0.24874390515082695,
]
VALIDATION_LOSS = [
    0.5069119820569424,
    0.391915359205388,
    0.34320044599949046,
    0.31008938481198983,
    0.2831777810574846,
    0.2684713452895905,
    0.2574971141650322,
    0.24544839960463505,
    0.23570749693094417,
    0.23062983213904056,
]


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "member2_baseline_loss.png"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.edgecolor": "#9ca3af",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
        }
    )

    figure, axis = plt.subplots(figsize=(8, 5), dpi=180)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    axis.plot(
        EPOCHS,
        TRAIN_LOSS,
        color="#2563eb",
        linewidth=2.4,
        marker="o",
        markersize=7,
        label="Train loss",
    )
    axis.plot(
        EPOCHS,
        VALIDATION_LOSS,
        color="#d97706",
        linewidth=2.4,
        linestyle="--",
        marker="s",
        markersize=6.5,
        label="Validation loss",
    )

    for epoch, value in zip(EPOCHS, TRAIN_LOSS):
        axis.annotate(
            f"{value:.3f}",
            (epoch, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="#1e40af",
            fontsize=9,
        )

    for epoch, value in zip(EPOCHS, VALIDATION_LOSS):
        axis.annotate(
            f"{value:.3f}",
            (epoch, value),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            color="#92400e",
            fontsize=9,
        )

    axis.set_title("T5-small baseline loss by epoch", loc="left", pad=24, weight="bold")
    axis.text(
        0,
        1.02,
        "7,517 training and 940 validation examples; batch size 4; random seed 42",
        transform=axis.transAxes,
        color="#4b5563",
        fontsize=9.5,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Token-level cross-entropy loss")
    axis.set_xticks(EPOCHS)
    axis.set_xlim(0.7, 10.3)
    axis.set_ylim(0, 1.2)
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper right")

    figure.text(
        0.125,
        0.015,
        "Source: verified Kaggle baseline_training_log.csv; training completed 29 July 2026.",
        color="#6b7280",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.16)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
