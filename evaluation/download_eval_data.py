"""
Downloads a labeled evaluation set of chest X-ray images.

Sources:
- Primary: NIH Chest X-ray 14 dataset (requires manual download from Kaggle/NIH)
- Fallback: PadChest / Indiana University CXR (open access)
- Demo mode: Small set of CC-licensed labeled images from OpenI (Indiana Univ)

OpenI dataset: https://openi.nlm.nih.gov/
- 7,470 frontal chest X-rays with radiology reports
- CC BY license, no registration required for image download
"""

import os
import csv
import json
import urllib.request
from pathlib import Path

# A hand-labeled set of OpenI images with known ground-truth conditions.
# These are mapped to our 6 condition labels.
# Image UIDs from the OpenI/Indiana University CXR dataset (public domain).
EVAL_SAMPLES = [
    {
        "filename": "eval_normal_01.png",
        "url": "https://openi.nlm.nih.gov/imgs/512/0/0/CXR1_1_IM-0001-3001.png",
        "label": "no finding",
        "source": "OpenI",
        "notes": "Normal PA view, no significant findings",
    },
    {
        "filename": "eval_normal_02.png",
        "url": "https://openi.nlm.nih.gov/imgs/512/1/1/CXR1_1_IM-0001-4001.png",
        "label": "no finding",
        "source": "OpenI",
        "notes": "Normal chest, clear lung fields",
    },
    {
        "filename": "eval_cardiomegaly_01.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/"
            "Cardiomegaly_labeled.jpg/420px-Cardiomegaly_labeled.jpg"
        ),
        "label": "cardiomegaly",
        "source": "Wikimedia Commons (CC BY-SA)",
        "notes": "Enlarged cardiac silhouette",
    },
    {
        "filename": "eval_pneumonia_01.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/"
            "Pneumonia_x-ray.jpg/401px-Pneumonia_x-ray.jpg"
        ),
        "label": "pneumonia",
        "source": "Wikimedia Commons (CC BY-SA)",
        "notes": "Right lower lobe consolidation consistent with pneumonia",
    },
    {
        "filename": "eval_pleural_effusion_01.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/"
            "Pleural_effusion_with_labels.jpg/400px-Pleural_effusion_with_labels.jpg"
        ),
        "label": "pleural effusion",
        "source": "Wikimedia Commons (CC BY-SA)",
        "notes": "Right-sided pleural effusion with blunting of costophrenic angle",
    },
    {
        "filename": "eval_pneumothorax_01.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/"
            "Pneumothorax_on_chest_xray.jpg/401px-Pneumothorax_on_chest_xray.jpg"
        ),
        "label": "pneumothorax",
        "source": "Wikimedia Commons (CC BY-SA)",
        "notes": "Left-sided pneumothorax with visible lung edge",
    },
    {
        "filename": "eval_atelectasis_01.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/"
            "Plate_atelectasis.jpg/400px-Plate_atelectasis.jpg"
        ),
        "label": "atelectasis",
        "source": "Wikimedia Commons (CC BY-SA)",
        "notes": "Plate-like atelectasis at right base",
    },
    {
        "filename": "eval_normal_03.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/"
            "Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg/"
            "409px-Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg"
        ),
        "label": "no finding",
        "source": "Wikimedia Commons (CC BY-SA)",
        "notes": "Normal PA chest radiograph",
    },
]


def download_eval_set(output_dir: str = "data/eval") -> str:
    """
    Downloads images and writes a labels CSV.
    Returns path to the labels CSV.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    labels_path = os.path.join(output_dir, "labels.csv")

    rows = []
    for sample in EVAL_SAMPLES:
        dest = os.path.join(output_dir, sample["filename"])
        if not os.path.exists(dest):
            print(f"Downloading {sample['filename']}...")
            try:
                req = urllib.request.Request(
                    sample["url"],
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    with open(dest, "wb") as f:
                        f.write(resp.read())
                print(f"  ✓ {dest}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
        else:
            print(f"  Already exists: {sample['filename']}")

        rows.append({
            "filename": sample["filename"],
            "label": sample["label"],
            "source": sample["source"],
            "notes": sample["notes"],
        })

    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "source", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Labels CSV written to {labels_path} ({len(rows)} images)")
    return labels_path


if __name__ == "__main__":
    download_eval_set()
