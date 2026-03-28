# Chest X-ray Analyzer

An AI-assisted chest X-ray analysis tool using **BiomedCLIP** for zero-shot classification and **BLIP VQA** for visual question answering. Runs entirely locally — no API calls required.

> ⚠️ **Disclaimer**: For research and educational purposes only. This is NOT a medical device and must NOT be used for clinical diagnosis.

---

## What it does

Given a chest X-ray image, the pipeline:

1. **BiomedCLIP** (Microsoft) — zero-shot classification against 25 conditions using a CLIP model trained on 15M biomedical image-text pairs
2. **BLIP VQA** (Salesforce) — answers 6 targeted medical questions about the image (findings, lung fields, heart, pleura, nodules/masses, impression)
3. **Report synthesis** — combines scores and visual descriptions into a structured radiology-style report (Findings / Impression / Recommendation)

### Detectable conditions (25 total)

**NIH ChestX-ray14 (14 conditions)**
| | | |
|---|---|---|
| Atelectasis | Cardiomegaly | Consolidation |
| Edema | Pleural Effusion | Emphysema |
| Fibrosis | Hernia | Infiltration |
| Mass | Nodule | Pleural Thickening |
| Pneumonia | Pneumothorax | |

**Extended conditions (10 + normal)**
| | | |
|---|---|---|
| Tuberculosis | Lung Abscess | Heart Failure |
| Aortic Aneurysm | Pericardial Effusion | Rib Fractures |
| Spine Abnormalities | Scoliosis | Mediastinal Mass |
| Hilar Enlargement | No Finding | |

---

## Setup

```bash
git clone https://github.com/nityasreecheera/chest-xray-analyzer
cd chest-xray-analyzer

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running the app

```bash
source .venv/bin/activate
python app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## Evaluation

The eval set contains **71 labeled chest X-rays** from two sources:

- **Wikimedia Commons** — 26 images across 19 conditions
- **NIH ChestX-ray14** — 45 images across 15 conditions (via Kaggle)

### Run evaluation

```bash
python evaluation/evaluate.py --no-download
```

### Download additional NIH images

Requires a [Kaggle API token](https://www.kaggle.com/settings) saved to `~/.kaggle/kaggle.json`.

```bash
python evaluation/download_nih_data.py
```

---

## Project structure

```
chest-xray-analyzer/
├── app.py                        # Gradio UI (dark theme)
├── config.py                     # 25 conditions + BiomedCLIP prompts
├── models/
│   ├── clip_classifier.py        # BiomedCLIP zero-shot classifier
│   └── blip_captioner.py         # BLIP VQA (6 medical questions)
├── pipeline/
│   └── report_generator.py       # Orchestrates full pipeline
├── evaluation/
│   ├── evaluate.py               # Metrics + confusion matrix + plots
│   ├── download_eval_data.py     # Wikimedia Commons image downloader
│   ├── download_nih_data.py      # NIH ChestX-ray14 downloader (Kaggle)
│   ├── metrics.py                # Precision / recall / F1
│   └── visualize.py              # Plot generation
├── notebooks/                    # Jupyter notebooks (01–04)
└── data/
    ├── eval/                     # 71 labeled eval images + labels.csv
    └── nih_raw/                  # Raw NIH downloads
```

---

## Models used

| Model | Source | Purpose |
|---|---|---|
| `BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Microsoft / HuggingFace | Zero-shot classification |
| `blip-vqa-base` | Salesforce / HuggingFace | Visual question answering |

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
