"""
Report synthesis pipeline.
Combines BiomedCLIP scores + BLIP captions into a structured report.
"""

from PIL import Image

from models.clip_classifier import ChestXrayClassifier
from models.blip_captioner import ChestXrayCaptioner


def build_report(clip_result: dict, blip_captions: dict) -> str:
    top = clip_result["top_condition"].title()
    score = clip_result["top_score"]
    low_conf = clip_result["low_confidence"]

    scores_lines = "\n".join(
        f"  • {cond.title()}: {s:.1%}"
        for cond, s in sorted(clip_result["scores"].items(), key=lambda x: -x[1])
    )

    blip_lines = "\n".join(
        f"  {key.replace('_', ' ').title()}: {text}"
        for key, text in blip_captions.items()
    )

    confidence_note = (
        f"⚠️  LOW CONFIDENCE ({score:.1%}) — treat results with extra caution."
        if low_conf
        else f"Top prediction: {top} ({score:.1%})"
    )

    return f"""FINDINGS
--------
{blip_lines}

CLASSIFICATION SCORES (BiomedCLIP)
-----------------------------------
{scores_lines}

IMPRESSION
----------
{confidence_note}

RECOMMENDATION
--------------
Correlate with clinical history and prior imaging. This is AI-assisted
analysis only and is NOT a clinical diagnosis. A qualified radiologist
should review all findings before any clinical decision is made.
"""


class ReportGenerator:
    def __init__(self, device: str = None):
        self.classifier = ChestXrayClassifier(device=device)
        self.captioner = ChestXrayCaptioner(device=device)

    def analyze(self, image: Image.Image) -> dict:
        print("Running BiomedCLIP classification...")
        clip_result = self.classifier.classify(image)

        print("Running BLIP VQA captioning...")
        blip_captions = self.captioner.caption_with_questions(image)

        print("Building report...")
        report = build_report(clip_result, blip_captions)

        return {
            "clip": clip_result,
            "blip": blip_captions,
            "report": report,
        }


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not image_path:
        print("Usage: python pipeline/report_generator.py <image_path>")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    generator = ReportGenerator()
    result = generator.analyze(image)

    print("\n" + "=" * 60)
    print("GENERATED REPORT")
    print("=" * 60)
    print(result["report"])
