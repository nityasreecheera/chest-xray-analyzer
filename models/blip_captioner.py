"""
BLIP-2 image captioner for chest X-rays.
Generates a free-text visual description of what the model sees.
"""

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from config import BLIP2_MODEL


class ChestXrayCaptioner:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BLIP-2 on {self.device}...")

        self.processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
        # Load in 8-bit if on GPU to save memory, otherwise float32
        load_kwargs = {}
        if self.device == "cuda":
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2_MODEL, **load_kwargs
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("BLIP-2 loaded.")

    def caption(self, image: Image.Image, prompt: str = None) -> str:
        """
        Generate a description of the X-ray image.
        Optionally provide a directing prompt.
        """
        if prompt is None:
            prompt = "Question: What abnormalities or findings are visible in this chest X-ray? Answer:"

        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                min_new_tokens=20,
                do_sample=False,
                num_beams=5,
            )

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        # Strip the prompt from the output if echoed back
        if prompt in caption:
            caption = caption.replace(prompt, "").strip()
        return caption

    def caption_with_questions(self, image: Image.Image) -> dict:
        """Run multiple targeted questions for richer structured output."""
        questions = {
            "findings": "Question: What abnormalities are visible in this chest X-ray? Answer:",
            "lung_fields": "Question: Describe the lung fields in this chest X-ray. Answer:",
            "heart": "Question: Describe the heart size and shape in this chest X-ray. Answer:",
            "impression": "Question: What is the overall impression of this chest X-ray? Answer:",
        }
        return {key: self.caption(image, prompt=q) for key, q in questions.items()}


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
