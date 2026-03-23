"""
Evaluation metrics for the chest X-ray classifier.
Computes accuracy, per-class precision/recall/F1, and confusion matrix.
"""

import numpy as np
from collections import defaultdict
from config import CONDITIONS


def compute_metrics(results: list[dict]) -> dict:
    """
    Args:
        results: list of dicts with keys:
            - filename
            - true_label
            - predicted_label
            - top_score
            - all_scores (dict condition -> float)
            - low_confidence

    Returns:
        metrics dict with overall + per-class stats
    """
    true_labels = [r["true_label"] for r in results]
    pred_labels = [r["predicted_label"] for r in results]
    n = len(results)

    # --- Overall accuracy ---
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / n if n > 0 else 0.0

    # --- Top-2 accuracy (true label in top 2 predictions) ---
    top2_correct = 0
    for r in results:
        sorted_conditions = sorted(r["all_scores"].items(), key=lambda x: -x[1])
        top2 = [c for c, _ in sorted_conditions[:2]]
        if r["true_label"] in top2:
            top2_correct += 1
    top2_accuracy = top2_correct / n if n > 0 else 0.0

    # --- Per-class metrics ---
    per_class = {}
    for condition in CONDITIONS:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == condition and p == condition)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != condition and p == condition)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == condition and p != condition)
        tn = sum(1 for t, p in zip(true_labels, pred_labels) if t != condition and p != condition)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = sum(1 for t in true_labels if t == condition)

        per_class[condition] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    # --- Macro averages ---
    macro_precision = np.mean([v["precision"] for v in per_class.values()])
    macro_recall    = np.mean([v["recall"]    for v in per_class.values()])
    macro_f1        = np.mean([v["f1"]        for v in per_class.values()])

    # --- Confusion matrix (rows = true, cols = predicted) ---
    label_index = {c: i for i, c in enumerate(CONDITIONS)}
    matrix = np.zeros((len(CONDITIONS), len(CONDITIONS)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t in label_index and p in label_index:
            matrix[label_index[t]][label_index[p]] += 1

    # --- Mean confidence on correct vs incorrect predictions ---
    correct_scores  = [r["top_score"] for r in results if r["true_label"] == r["predicted_label"]]
    incorrect_scores = [r["top_score"] for r in results if r["true_label"] != r["predicted_label"]]

    return {
        "n": n,
        "accuracy": accuracy,
        "top2_accuracy": top2_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "conditions": CONDITIONS,
        "mean_confidence_correct": np.mean(correct_scores) if correct_scores else 0.0,
        "mean_confidence_incorrect": np.mean(incorrect_scores) if incorrect_scores else 0.0,
        "low_confidence_count": sum(1 for r in results if r["low_confidence"]),
    }


def format_report(metrics: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Total images evaluated : {metrics['n']}")
    lines.append(f"Top-1 Accuracy         : {metrics['accuracy']:.1%}")
    lines.append(f"Top-2 Accuracy         : {metrics['top2_accuracy']:.1%}")
    lines.append(f"Macro Precision        : {metrics['macro_precision']:.1%}")
    lines.append(f"Macro Recall           : {metrics['macro_recall']:.1%}")
    lines.append(f"Macro F1               : {metrics['macro_f1']:.1%}")
    lines.append(f"Low-confidence flags   : {metrics['low_confidence_count']}/{metrics['n']}")
    lines.append(f"Mean conf (correct)    : {metrics['mean_confidence_correct']:.1%}")
    lines.append(f"Mean conf (incorrect)  : {metrics['mean_confidence_incorrect']:.1%}")
    lines.append("")

    lines.append("Per-class breakdown:")
    lines.append(f"  {'Condition':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    lines.append("  " + "-" * 54)
    for condition, m in metrics["per_class"].items():
        lines.append(
            f"  {condition:<22} {m['precision']:>6.1%} {m['recall']:>6.1%} "
            f"{m['f1']:>6.1%} {m['support']:>8}"
        )

    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=predicted):")
    header = "  " + " ".join(f"{c[:6]:>8}" for c in metrics["conditions"])
    lines.append(header)
    for i, condition in enumerate(metrics["conditions"]):
        row_vals = " ".join(f"{v:>8}" for v in metrics["confusion_matrix"][i])
        lines.append(f"  {condition[:6]:<6}  {row_vals}")

    return "\n".join(lines)
