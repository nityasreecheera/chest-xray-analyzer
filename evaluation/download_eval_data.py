"""
Downloads a labeled evaluation set of chest X-ray images.

Uses the Wikimedia Commons API to resolve stable file URLs (avoids rate limits
from direct thumbnail scraping). Falls back to OpenI/NLM for normal X-rays.
"""

import os
import csv
import time
import urllib.request
import urllib.parse
import json
from pathlib import Path

# Wikimedia Commons API — resolves File: names to direct download URLs
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Descriptive User-Agent required by Wikimedia ToS
USER_AGENT = "chest-xray-analyzer/1.0 (research project; educational use)"

# Map: local filename → (Commons file name, label, notes)
EVAL_SAMPLES = [
    {
        "filename": "eval_normal_01.jpg",
        "commons_file": "Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg",
        "label": "no finding",
        "notes": "Normal PA chest radiograph, clear lung fields",
    },
    {
        "filename": "eval_pneumonia_01.jpg",
        "commons_file": "Pneumonia_x-ray.jpg",
        "label": "pneumonia",
        "notes": "Right lower lobe consolidation consistent with pneumonia",
    },
    {
        "filename": "eval_pneumothorax_01.jpg",
        "commons_file": "Pneumothorax_CXR.jpg",
        "label": "pneumothorax",
        "notes": "Pneumothorax on chest X-ray",
    },
    {
        "filename": "eval_atelectasis_01.jpg",
        "commons_file": "Atelectasis_Normal_vs_Affected_Airway.jpg",
        "label": "atelectasis",
        "notes": "Atelectasis diagram showing affected airway",
    },
    {
        "filename": "eval_pleural_effusion_01.jpg",
        "commons_file": "Pleural_effusion.jpg",
        "label": "pleural effusion",
        "notes": "Pleural effusion on chest X-ray",
    },
    {
        "filename": "eval_cardiomegaly_01.jpg",
        "commons_file": "Cardiomegally.PNG",
        "label": "cardiomegaly",
        "notes": "Markedly enlarged cardiac silhouette",
    },
    {
        "filename": "eval_cardiomegaly_02.jpg",
        "commons_file": "Cardiomegalia.JPG",
        "label": "cardiomegaly",
        "notes": "Cardiomegaly chest X-ray",
    },
    {
        "filename": "eval_normal_02.png",
        "direct_url": "https://openi.nlm.nih.gov/imgs/512/1/1/CXR1_1_IM-0001-4001.png",
        "label": "no finding",
        "notes": "Normal chest, OpenI Indiana University CXR dataset",
    },
]


def resolve_commons_url(commons_filename: str) -> str | None:
    """Use Wikimedia Commons API to get direct image URL for a File: name."""
    params = {
        "action": "query",
        "titles": f"File:{commons_filename}",
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json",
    }
    url = COMMONS_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            infos = page.get("imageinfo", [])
            if infos:
                return infos[0]["url"]
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"    API error for {commons_filename}: {e}")
    return None


def download_image(url: str, dest: str) -> bool:
    """Download a single image from url to dest. Returns True on success."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            with open(dest, "wb") as f:
                f.write(resp.read())
        return True
    except (urllib.error.URLError, OSError) as e:
        print(f"    Download failed: {e}")
        return False


def download_eval_set(output_dir: str = "data/eval") -> str:
    """Download labeled eval images and write a labels CSV. Returns path to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    labels_path = os.path.join(output_dir, "labels.csv")
    rows = []

    for sample in EVAL_SAMPLES:
        dest = os.path.join(output_dir, sample["filename"])

        if os.path.exists(dest):
            print(f"  Already exists: {sample['filename']}")
            rows.append({k: sample.get(k, "") for k in ["filename", "label", "notes"]})
            rows[-1]["source"] = sample.get("commons_file", "OpenI")
            continue

        print(f"Fetching {sample['filename']}  ({sample['label']})")

        # Resolve URL
        if "commons_file" in sample:
            print(f"  Resolving Commons API: {sample['commons_file']}")
            url = resolve_commons_url(sample["commons_file"])
            if not url:
                print("  Could not resolve Commons URL, skipping.")
                time.sleep(1)
                continue
            print(f"  Resolved: {url[:80]}...")
        else:
            url = sample["direct_url"]

        ok = download_image(url, dest)
        if ok:
            print("  Saved")
            rows.append({
                "filename": sample["filename"],
                "label": sample["label"],
                "source": sample.get("commons_file", "OpenI"),
                "notes": sample["notes"],
            })
        else:
            print("  Skipped")

        time.sleep(1)  # Be polite to Wikimedia servers

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "source", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Labels CSV written: {labels_path}  ({len(rows)} images)")
    return labels_path


if __name__ == "__main__":
    download_eval_set()
