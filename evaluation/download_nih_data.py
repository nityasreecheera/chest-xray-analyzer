"""
Downloads a labeled subset of NIH ChestX-ray14 via the Kaggle API.

Setup (one-time):
  1. Go to https://www.kaggle.com/settings → API → Create New Token
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json

What this does:
  - Downloads Data_Entry_2017.csv (labels for all 112K images)
  - Finds single-label images for target conditions
  - Downloads only those specific images (not the full 42GB dataset)
  - Copies them to data/eval/ with labels appended to labels.csv
"""

import os
import csv
import shutil
import zipfile
import subprocess
from pathlib import Path

DATASET = "nih-chest-xrays/data"

# Max images to add per condition
MAX_PER_CONDITION = 3

# Map NIH label names → our condition names
NIH_TO_CONDITION = {
    "Atelectasis":        "atelectasis",
    "Cardiomegaly":       "cardiomegaly",
    "Consolidation":      "consolidation",
    "Edema":              "edema",
    "Effusion":           "pleural effusion",
    "Emphysema":          "emphysema",
    "Fibrosis":           "fibrosis",
    "Hernia":             "hernia",
    "Infiltration":       "infiltration",
    "Mass":               "mass",
    "Nodule":             "nodule",
    "Pleural_Thickening": "pleural thickening",
    "Pneumonia":          "pneumonia",
    "Pneumothorax":       "pneumothorax",
    "No Finding":         "no finding",
}

# Which conditions to prioritise (ones missing or under-represented in eval set)
TARGET_CONDITIONS = {
    "emphysema", "infiltration", "mass", "nodule",
    "fibrosis", "hernia", "consolidation", "pleural thickening",
    "atelectasis", "pneumonia", "pneumothorax", "no finding",
    "edema", "cardiomegaly", "pleural effusion",
}


def run(cmd: str) -> int:
    return subprocess.call(cmd, shell=True)


def kaggle_download_file(remote_path: str, dest_dir: str) -> bool:
    """Download a single file from the NIH Kaggle dataset."""
    cmd = (
        f".venv/bin/kaggle datasets download {DATASET} "
        f"--file \"{remote_path}\" --path \"{dest_dir}\" --unzip -q"
    )
    return run(cmd) == 0


def ensure_labels_csv(nih_dir: str) -> Path:
    """Download and unzip Data_Entry_2017.csv if not already present."""
    csv_path = Path(nih_dir) / "Data_Entry_2017.csv"
    if csv_path.exists():
        return csv_path

    print("Downloading Data_Entry_2017.csv...")
    Path(nih_dir).mkdir(parents=True, exist_ok=True)
    zip_path = Path(nih_dir) / "Data_Entry_2017.csv.zip"

    run(
        f".venv/bin/kaggle datasets download {DATASET} "
        f"--file Data_Entry_2017.csv --path \"{nih_dir}\" -q"
    )

    # Unzip if needed
    if zip_path.exists() and not csv_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(nih_dir)

    return csv_path


def select_images(csv_path: Path, max_per: int) -> dict[str, list[str]]:
    """Parse CSV and return {condition: [image_filenames]} for single-label images."""
    selected: dict[str, list[str]] = {c: [] for c in TARGET_CONDITIONS}

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            findings = row["Finding Labels"].split("|")
            if len(findings) != 1:
                continue  # skip multi-label images
            nih_label = findings[0].strip()
            our_label = NIH_TO_CONDITION.get(nih_label)
            if our_label in selected and len(selected[our_label]) < max_per:
                selected[our_label].append(row["Image Index"])

    return selected


def image_folder(filename: str) -> str:
    """NIH images are split across images_001 … images_012 by filename range."""
    # The dataset has 12 zip files; images are sequentially numbered.
    # Map image number to folder number (each folder has ~9,000 images)
    num = int(filename.split("_")[0])
    folder = min((num // 9000) + 1, 12)
    return f"images_{folder:03d}"


def download_and_add(
    nih_dir: str = "data/nih_raw",
    eval_dir: str = "data/eval",
    max_per: int = MAX_PER_CONDITION,
):
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "Kaggle credentials not found.\n"
            "  1. Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "  2. Save kaggle.json to ~/.kaggle/kaggle.json\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json"
        )
        return

    csv_path = ensure_labels_csv(nih_dir)
    if not csv_path.exists():
        print("Could not download labels CSV.")
        return

    print("Selecting images from labels CSV...")
    selected = select_images(csv_path, max_per)

    total = sum(len(v) for v in selected.values())
    print(f"Will download {total} images for {len([k for k,v in selected.items() if v])} conditions\n")

    # Read existing eval filenames to avoid duplicates
    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)
    labels_path = eval_path / "labels.csv"
    existing = set()
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(row["filename"])

    new_rows = []
    for condition, files in selected.items():
        for fname in files:
            dest_name = f"nih_{fname}"
            if dest_name in existing:
                print(f"  Already exists: {dest_name}")
                continue

            folder = image_folder(fname)
            remote = f"{folder}/images/{fname}"
            tmp_dir = Path(nih_dir) / folder / "images"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / fname

            print(f"  Downloading {fname}  ({condition})...")
            ok = kaggle_download_file(remote, str(tmp_dir.parent.parent))

            # Kaggle may place it at a different path — find it
            if not tmp_path.exists():
                found = list(Path(nih_dir).rglob(fname))
                if found:
                    tmp_path = found[0]

            if tmp_path.exists():
                shutil.copy2(tmp_path, eval_path / dest_name)
                print(f"    ✓ Saved as {dest_name}")
                new_rows.append({
                    "filename": dest_name,
                    "label": condition,
                    "source": "NIH ChestX-ray14",
                    "notes": f"NIH single-label: {condition}",
                })
            else:
                print(f"    ✗ Could not find downloaded file")

    if new_rows:
        with open(labels_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "label", "source", "notes"]
            )
            writer.writerows(new_rows)
        print(f"\n✓ Added {len(new_rows)} NIH images")
        print(f"  Conditions: {sorted({r['label'] for r in new_rows})}")
    else:
        print("No new images added.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14 subset")
    parser.add_argument("--nih-dir", default="data/nih_raw")
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--max-per", type=int, default=MAX_PER_CONDITION)
    args = parser.parse_args()

    download_and_add(args.nih_dir, args.eval_dir, args.max_per)
