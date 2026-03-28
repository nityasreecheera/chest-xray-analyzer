# Chest X-ray Analyzer

An AI-assisted chest X-ray analysis tool using **BiomedCLIP** for zero-shot classification and **BLIP VQA** for visual question answering. Runs entirely locally — no API calls required.

> ⚠️ **Disclaimer**: For research and educational purposes only. This is NOT a medical device and must NOT be used for clinical diagnosis.

---

## What it does

Given a chest X-ray image, the pipeline:

1. **BiomedCLIP** (Microsoft) — classifies the image against 6 conditions using a CLIP model trained on 15M biomedical image-text pairs
2. **BLIP VQA** (Salesforce) — answers 4 targeted medical questions about the image (opacity, effusion, cardiomegaly, pneumothorax)
3. **Report synthesis** — combines scores and visual descriptions into a structured radiology-style report (Findings / Impression / Recommendation)

### Detectable conditions

- Pneumonia
- Pleural Effusion
- Cardiomegaly
- Atelectasis
- Pneumothorax
- No Finding

---

## Setup

```bash
git clone https://github.com/your-username/chest-xray-analyzer
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

## Project structure

```
chest-xray-analyzer/
├── app.py                  # Gradio UI
├── config.py               # Model names, conditions, thresholds
├── models/
│   ├── clip_classifier.py  # BiomedCLIP zero-shot classifier
│   └── blip_captioner.py   # BLIP VQA visual descriptions
├── pipeline/
│   └── report_generator.py # Orchestrates full pipeline
├── evaluation/
│   └── evaluate.py         # Evaluation metrics and plots
├── notebooks/              # Jupyter notebooks for exploration
└── data/                   # Sample and evaluation images
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
