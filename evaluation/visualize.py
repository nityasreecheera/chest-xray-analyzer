"""
Visualization helpers for evaluation results.
Produces confusion matrix heatmap and per-image prediction grid.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from PIL import Image


def plot_confusion_matrix(matrix: np.ndarray, conditions: list, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(conditions)))
    ax.set_yticks(range(len(conditions)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in conditions], fontsize=9)
    ax.set_yticklabels(conditions, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix — BiomedCLIP\nChest X-ray Classification", fontsize=12)

    # Annotate each cell
    thresh = matrix.max() / 2.0
    for i in range(len(conditions)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {output_path}")


def plot_per_class_metrics(per_class: dict, output_path: str):
    conditions = list(per_class.keys())
    precision  = [per_class[c]["precision"] for c in conditions]
    recall     = [per_class[c]["recall"]    for c in conditions]
    f1         = [per_class[c]["f1"]        for c in conditions]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#3498db")
    ax.bar(x,         recall,    width, label="Recall",    color="#2ecc71")
    ax.bar(x + width, f1,        width, label="F1",        color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in conditions], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved per-class metrics → {output_path}")


def plot_prediction_grid(results: list[dict], image_dir: str, output_path: str, max_images: int = 16):
    """
    A grid showing each image with true label, predicted label, and top score.
    Green border = correct, Red border = incorrect.
    """
    results = results[:max_images]
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        img_path = Path(image_dir) / r["filename"]

        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img, cmap="gray")
        except Exception:
            ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

        correct = r["true_label"] == r["predicted_label"]
        color = "#2ecc71" if correct else "#e74c3c"

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        ax.set_title(
            f"True:  {r['true_label']}\nPred: {r['predicted_label']} ({r['top_score']:.0%})",
            fontsize=7.5,
            color=color if not correct else "black",
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    correct_patch = mpatches.Patch(color="#2ecc71", label="Correct")
    wrong_patch   = mpatches.Patch(color="#e74c3c", label="Incorrect")
    fig.legend(handles=[correct_patch, wrong_patch], loc="lower right", fontsize=10)

    plt.suptitle("Per-image Predictions — BiomedCLIP", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction grid → {output_path}")


def plot_score_distribution(results: list[dict], output_path: str):
    """Histogram of top confidence scores split by correct/incorrect."""
    correct   = [r["top_score"] for r in results if r["true_label"] == r["predicted_label"]]
    incorrect = [r["top_score"] for r in results if r["true_label"] != r["predicted_label"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct,   bins=bins, alpha=0.7, color="#2ecc71", label=f"Correct (n={len(correct)})")
    ax.hist(incorrect, bins=bins, alpha=0.7, color="#e74c3c", label=f"Incorrect (n={len(incorrect)})")
    ax.axvline(0.3, color="gray", linestyle="--", label="Confidence threshold (0.3)")
    ax.set_xlabel("Top prediction score")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved score distribution → {output_path}")
