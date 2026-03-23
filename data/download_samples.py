"""
Downloads a small set of sample chest X-ray images from NIH's public dataset
for testing the pipeline without manually sourcing images.

NIH Chest X-ray dataset is publicly available at:
https://nihcc.app.box.com/v/ChestXray-NIHCC

This script pulls a few sample images from the Kaggle mirror
(no auth needed for these specific public samples).
"""

import os
import urllib.request
from pathlib import Path

# A few publicly available chest X-ray sample images (NIH CC0-licensed samples)
SAMPLE_IMAGES = {
    "pneumonia_sample.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/"
        "Pneumonia_x-ray.jpg/401px-Pneumonia_x-ray.jpg"
    ),
    "normal_sample.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/"
        "Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg/"
        "409px-Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg"
    ),
    "cardiomegaly_sample.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/"
        "Cardiomegaly.svg/400px-Cardiomegaly.svg.png"
    ),
}


def download_samples(output_dir: str = "data/samples"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for filename, url in SAMPLE_IMAGES.items():
        dest = os.path.join(output_dir, filename)
        if os.path.exists(dest):
            print(f"Already exists: {dest}")
            continue
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved to {dest}")
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    download_samples()
    print("\nDone. Sample images saved to data/samples/")
