"""
Gradio UI for the Chest X-ray Analyzer.
"""

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import io
import os

matplotlib.use("Agg")

from pipeline.report_generator import ReportGenerator

generator = None


def load_generator():
    global generator
    if generator is None:
        generator = ReportGenerator()
    return generator


def make_bar_chart(scores: dict) -> Image.Image:
    conditions = list(scores.keys())
    values = [scores[c] for c in conditions]

    colors = ["#e74c3c" if v == max(values) else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(conditions, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("BiomedCLIP Condition Scores")

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def analyze_xray(image: Image.Image):
    if image is None:
        return None, "Please upload a chest X-ray image.", ""

    try:
        gen = load_generator()
        result = gen.analyze(image)

        chart = make_bar_chart(result["clip"]["scores"])

        top = result["clip"]["top_condition"]
        score = result["clip"]["top_score"]
        low_conf = result["clip"]["low_confidence"]

        confidence_str = (
            f"⚠️ Low confidence ({score:.1%}) — results may be unreliable."
            if low_conf
            else f"✅ Top prediction: **{top}** ({score:.1%})"
        )

        return chart, confidence_str, result["report"]

    except Exception as e:
        return None, f"Error: {str(e)}", ""


with gr.Blocks(title="Chest X-ray Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🩻 Chest X-ray Analyzer
    **AI-assisted analysis using BiomedCLIP + BLIP-2 + Claude**

    > ⚠️ **Disclaimer**: This tool is for research and educational purposes only.
    > It is NOT a medical device and should NOT be used for clinical diagnosis.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Chest X-ray")
            analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=2):
            confidence_output = gr.Markdown(label="Prediction")
            chart_output = gr.Image(label="Condition Scores", type="pil")

    report_output = gr.Textbox(
        label="Generated Radiology Report",
        lines=15,
        interactive=False,
    )

    analyze_btn.click(
        fn=analyze_xray,
        inputs=[image_input],
        outputs=[chart_output, confidence_output, report_output],
    )

    gr.Markdown("""
    ### How it works
    1. **BiomedCLIP** (Microsoft) — zero-shot classification against 6 conditions
    2. **BLIP-2** (Salesforce) — visual description of what the model sees
    3. **Claude** — synthesizes both into a structured radiology-style report
    """)


if __name__ == "__main__":
    demo.launch(share=False)
