"""
Downloads a labeled subset of NIH ChestX-ray14 via the Kaggle API.

Setup (one-time):
  1. Go to https://www.kaggle.com/settings → API → Create New Token
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json

What this downloads:
  - The NIH sample folder (880 images, ~300MB) from nih-chest-xrays/data
  - Data_Entry_2017.csv (labels for all 112K images)
  - Selects up to MAX_PER_CONDITION images per condition and copies
    them to data/eval/ with the correct label rows appended to labels.csv

NIH conditions mapped to our 25-condition set:
  Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
  Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule,
  Pleural_Thickening, Pneumonia, Pneumothorax, No Finding
"""

import os
import csv
import shutil
import zipfile
from pathlib import Path

# How many NIH images to add per condition (keep eval set manageable)
MAX_PER_CONDITION = 3

# Map NIH label names → our condition names
NIH_TO_CONDITION = {
    "Atelectasis":       "atelectasis",
    "Cardiomegaly":      "cardiomegaly",
    "Consolidation":     "consolidation",
    "Edema":             "edema",
    "Effusion":          "pleural effusion",
    "Emphysema":         "emphysema",
    "Fibrosis":          "fibrosis",
    "Hernia":            "hernia",
    "Infiltration":      "infiltration",
    "Mass":              "mass",
    "Nodule":            "nodule",
    "Pleural_Thickening": "pleural thickening",
    "Pneumonia":         "pneumonia",
    "Pneumothorax":      "pneumothorax",
    "No Finding":        "no finding",
}

# Conditions we most want from NIH (ones missing from our current eval set)
PRIORITY_CONDITIONS = {
    "emphysema", "infiltration", "mass", "nodule",
    "heart failure",  # not in NIH set but keep as reminder
    "hernia", "fibrosis", "consolidation",
}


def download_nih_sample(dest_dir: str = "data/nih_raw"):
    """Download NIH sample images + labels CSV via Kaggle API."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("kaggle package not installed. Run: pip install kaggle")
        return False

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "Kaggle credentials not found.\n"
            "  1. Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "  2. Save kaggle.json to ~/.kaggle/kaggle.json\n"
            "  3. Run: chmod 600 ~/.kaggle/kaggle.json\n"
            "  4. Re-run this script"
        )
        return False

    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    print("Downloading NIH ChestX-ray14 sample (880 images, ~300MB)...")
    os.system(
        f".venv/bin/kaggle datasets download nih-chest-xrays/data "
        f"--path {dest_dir} --unzip -f sample.zip"
    )
    print("Downloading labels CSV...")
    os.system(
        f".venv/bin/kaggle datasets download nih-chest-xrays/data "
        f"--path {dest_dir} --unzip -f Data_Entry_2017.csv"
    )
    return True


def add_nih_to_eval(
    nih_dir: str = "data/nih_raw",
    eval_dir: str = "data/eval",
    max_per: int = MAX_PER_CONDITION,
):
    """Copy selected NIH images to eval dir and append rows to labels.csv."""
    labels_csv = Path(nih_dir) / "Data_Entry_2017.csv"
    sample_dir = Path(nih_dir) / "sample" / "images"

    if not labels_csv.exists():
        print(f"Labels CSV not found: {labels_csv}")
        return

    if not sample_dir.exists():
        # Try alternate path
        sample_dir = Path(nih_dir) / "images"

    if not sample_dir.exists():
        print(f"Sample images not found under {nih_dir}")
        return

    # Build index: condition → list of filenames in sample
    sample_files = {f.name for f in sample_dir.glob("*.png")}
    print(f"Found {len(sample_files)} sample images")

    condition_files: dict[str, list[str]] = {c: [] for c in NIH_TO_CONDITION.values()}

    with open(labels_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["Image Index"]
            if fname not in sample_files:
                continue
            findings = row["Finding Labels"].split("|")
            # Only use single-label images for clean eval
            if len(findings) != 1:
                continue
            nih_label = findings[0].strip()
            our_label = NIH_TO_CONDITION.get(nih_label)
            if our_label and len(condition_files[our_label]) < max_per:
                condition_files[our_label].append(fname)

    # Copy images and build new label rows
    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)
    existing_labels_path = eval_path / "labels.csv"

    # Read existing labels to avoid duplicates
    existing_files = set()
    if existing_labels_path.exists():
        with open(existing_labels_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_files.add(row["filename"])

    new_rows = []
    for condition, files in condition_files.items():
        for src_name in files:
            dest_name = f"nih_{src_name}"
            if dest_name in existing_files:
                print(f"  Already exists: {dest_name}")
                continue
            src = sample_dir / src_name
            dst = eval_path / dest_name
            shutil.copy2(src, dst)
            print(f"  Copied: {dest_name}  ({condition})")
            new_rows.append({
                "filename": dest_name,
                "label": condition,
                "source": "NIH ChestX-ray14",
                "notes": f"NIH sample — single label: {condition}",
            })

    if not new_rows:
        print("No new images to add.")
        return

    # Append to labels.csv
    with open(existing_labels_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "label", "source", "notes"]
        )
        writer.writerows(new_rows)

    print(f"\n✓ Added {len(new_rows)} NIH images to {eval_dir}/")
    print(f"  Conditions covered: {sorted({r['label'] for r in new_rows})}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14 sample")
    parser.add_argument("--nih-dir", default="data/nih_raw",
                        help="Where to store raw NIH download")
    parser.add_argument("--eval-dir", default="data/eval",
                        help="Eval directory to copy images into")
    parser.add_argument("--max-per", type=int, default=MAX_PER_CONDITION,
                        help="Max images per condition to add")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip Kaggle download (use existing nih-dir)")
    args = parser.parse_args()

    if not args.no_download:
        ok = download_nih_sample(args.nih_dir)
        if not ok:
            exit(1)

    add_nih_to_eval(args.nih_dir, args.eval_dir, args.max_per)
