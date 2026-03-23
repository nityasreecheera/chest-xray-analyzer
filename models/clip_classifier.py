"""
BiomedCLIP-based zero-shot chest X-ray classifier.
Uses Microsoft's BiomedCLIP model trained on PubMed + MIMIC data.
"""

import torch
import open_clip
from PIL import Image
from config import BIOMED_CLIP_MODEL, CONDITION_PROMPTS, CONFIDENCE_THRESHOLD


class ChestXrayClassifier:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BiomedCLIP on {self.device}...")
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(BIOMED_CLIP_MODEL)
        self.tokenizer = open_clip.get_tokenizer(BIOMED_CLIP_MODEL)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("BiomedCLIP loaded.")

        # Pre-encode all condition text prompts (only done once)
        self._text_features = self._encode_conditions()

    def _encode_conditions(self) -> torch.Tensor:
        prompts = list(CONDITION_PROMPTS.values())
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def classify(self, image: Image.Image) -> dict:
        """
        Returns a dict of condition -> probability score.
        Flags low-confidence results.
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity → softmax probabilities
        logits = (image_features @ self._text_features.T).squeeze(0)
        probs = torch.softmax(logits * 100, dim=0).cpu().numpy()

        conditions = list(CONDITION_PROMPTS.keys())
        scores = {condition: float(prob) for condition, prob in zip(conditions, probs)}

        top_condition = max(scores, key=scores.get)
        top_score = scores[top_condition]
        low_confidence = top_score < CONFIDENCE_THRESHOLD

        return {
            "scores": scores,
            "top_condition": top_condition,
            "top_score": top_score,
            "low_confidence": low_confidence,
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not image_path:
        print("Usage: python models/clip_classifier.py <image_path>")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    classifier = ChestXrayClassifier()
    result = classifier.classify(image)

    print(f"\nTop condition: {result['top_condition']} ({result['top_score']:.1%})")
    print(f"Low confidence: {result['low_confidence']}")
    print("\nAll scores:")
    for condition, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        print(f"  {condition:<20} {score:.1%}  {bar}")
