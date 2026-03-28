"""
BLIP VQA image captioner for chest X-rays.
Uses blip-vqa-base (CPU-friendly, ~1GB) instead of BLIP-2 (14GB).
Generates free-text answers to targeted radiology questions.
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"

QUESTIONS = {
    "findings":          "What abnormalities are visible in this chest X-ray?",
    "lung_fields":       "Describe the lung fields in this chest X-ray.",
    "heart":             "Describe the heart size and shape in this chest X-ray.",
    "pleura":            "Is there any pleural effusion or thickening visible?",
    "nodules_masses":    "Are there any nodules, masses, or infiltrates visible?",
    "impression":        "What is the overall impression of this chest X-ray?",
}


class ChestXrayCaptioner:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BLIP VQA on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(BLIP_VQA_MODEL)
        self.model = BlipForQuestionAnswering.from_pretrained(
            BLIP_VQA_MODEL, torch_dtype=torch.float32
        ).to(self.device).eval()
        print("BLIP VQA loaded.")

    def ask(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=80)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def caption_with_questions(self, image: Image.Image) -> dict:
        return {key: self.ask(image, q) for key, q in QUESTIONS.items()}


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not image_path:
        print("Usage: python models/blip_captioner.py <image_path>")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    captioner = ChestXrayCaptioner()

    print("\nGenerating captions...\n")
    results = captioner.caption_with_questions(image)
    for section, text in results.items():
        print(f"{section.upper()}:\n  {text}\n")
