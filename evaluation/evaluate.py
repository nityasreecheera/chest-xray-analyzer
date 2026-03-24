"""
Main evaluation script.

Usage:
    # Download eval set and run evaluation
    python evaluation/evaluate.py

    # Run on a custom labeled directory
    python evaluation/evaluate.py --image-dir data/eval --labels data/eval/labels.csv

    # Run without downloading (use existing eval data)
    python evaluation/evaluate.py --no-download

Labels CSV format:
    filename,label,source,notes
    image01.jpg,pneumonia,NIH,right lower lobe consolidation
    image02.jpg,no finding,NIH,normal chest
"""

import os
import csv
import json
import argparse
from pathlib import Path
from PIL import Image

from models.clip_classifier import ChestXrayClassifier
from evaluation.metrics import compute_metrics, format_report
from evaluation.visualize import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_prediction_grid,
    plot_score_distribution,
)


RESULTS_DIR = "evaluation/results"


def load_labels(labels_csv: str) -> list[dict]:
    with open(labels_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_evaluation(image_dir: str, labels: list[dict], output_dir: str) -> list[dict]:
    classifier = ChestXrayClassifier()
    results = []
    n = len(labels)

    print(f"\nEvaluating {n} images...\n")
    for i, row in enumerate(labels):
        img_path = Path(image_dir) / row["filename"]
        print(f"[{i+1}/{n}] {row['filename']}  (true: {row['label']})")

        if not img_path.exists():
            print(f"  ⚠️  Image not found, skipping.")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            clip_result = classifier.classify(image)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

        predicted = clip_result["top_condition"]
        correct = predicted == row["label"]
        marker = "✓" if correct else "✗"
        print(f"  {marker}  pred={predicted} ({clip_result['top_score']:.1%})  "
              f"low_conf={clip_result['low_confidence']}")

        results.append({
            "filename": row["filename"],
            "true_label": row["label"],
            "predicted_label": predicted,
            "top_score": clip_result["top_score"],
            "all_scores": clip_result["scores"],
            "low_confidence": clip_result["low_confidence"],
            "correct": correct,
            "source": row.get("source", ""),
            "notes": row.get("notes", ""),
        })

    return results


def save_results(results: list[dict], metrics: dict, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Raw results JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Text report
    report_path = os.path.join(output_dir, "report.txt")
    report_text = format_report(metrics)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\n✓ Results saved to {output_dir}/")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate chest X-ray classifier")
    parser.add_argument("--image-dir", default="data/eval",
                        help="Directory containing evaluation images")
    parser.add_argument("--labels", default="data/eval/labels.csv",
                        help="Path to labels CSV file")
    parser.add_argument("--output-dir", default=RESULTS_DIR,
                        help="Where to save results and plots")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloading eval images")
    args = parser.parse_args()

    # Download eval data unless told not to
    if not args.no_download and not Path(args.labels).exists():
        print("Downloading evaluation dataset...")
        from evaluation.download_eval_data import download_eval_set
        download_eval_set(args.image_dir)

    if not Path(args.labels).exists():
        print(f"Labels file not found: {args.labels}")
        print("Run without --no-download to fetch the eval set automatically.")
        return

    labels = load_labels(args.labels)
    results = run_evaluation(args.image_dir, labels, args.output_dir)

    if not results:
        print("No results — check that images downloaded correctly.")
        return

    metrics = compute_metrics(results)
    report_text = save_results(results, metrics, args.output_dir)

    # Generate plots
    print("\nGenerating plots...")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["conditions"],
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    plot_per_class_metrics(
        metrics["per_class"],
        os.path.join(args.output_dir, "per_class_metrics.png"),
    )
    plot_prediction_grid(
        results,
        args.image_dir,
        os.path.join(args.output_dir, "prediction_grid.png"),
    )
    plot_score_distribution(
        results,
        os.path.join(args.output_dir, "score_distribution.png"),
    )

    print("\n" + report_text)
    print(f"\n✓ All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
