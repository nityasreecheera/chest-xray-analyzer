"""
Gradio UI for the Chest X-ray Analyzer.
"""

import io

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

matplotlib.use("Agg")

from pipeline.report_generator import ReportGenerator

generator = None

DARK_CSS = """
/* ── Root palette ─────────────────────────────────────────── */
:root {
  --bg:        #0a0f1e;
  --bg-card:   #111827;
  --border:    #1e293b;
  --accent:    #38bdf8;
  --accent2:   #818cf8;
  --danger:    #f87171;
  --text:      #e2e8f0;
  --text-dim:  #64748b;
}

/* Full-page dark background */
body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: "Inter", "SF Pro Display", system-ui, sans-serif !important;
}

/* Header card */
#header-md {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 24px 28px !important;
  margin-bottom: 8px !important;
}
#header-md h1 { color: var(--accent) !important; font-size: 1.6rem !important; margin: 0 !important; }
#header-md p  { color: var(--text-dim) !important; margin: 4px 0 0 !important; }
#header-md blockquote {
  border-left: 3px solid var(--danger) !important;
  background: rgba(248,113,113,0.06) !important;
  padding: 8px 14px !important;
  border-radius: 0 8px 8px 0 !important;
  color: #fca5a5 !important;
  margin: 12px 0 0 !important;
}

/* All panels / cards */
.block, .panel, .wrap, .form {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}

/* Labels */
label span, .label-wrap span {
  color: var(--text-dim) !important;
  font-size: 0.72rem !important;
  text-transform: uppercase !important;
  letter-spacing: .08em !important;
}

/* Image upload zone */
.upload-container, .svelte-1ipelgc {
  background: #0d1526 !important;
  border: 2px dashed var(--border) !important;
  border-radius: 12px !important;
}

/* Primary button */
#analyze-btn {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: #0a0f1e !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 10px !important;
  font-size: 1rem !important;
  padding: 12px !important;
  transition: opacity .2s;
}
#analyze-btn:hover { opacity: .85 !important; }

/* Textbox (report) */
textarea {
  background: #0d1526 !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-family: "JetBrains Mono", "Fira Code", monospace !important;
  font-size: 0.82rem !important;
  line-height: 1.65 !important;
}

/* Confidence markdown */
#confidence-md {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 14px 18px !important;
  font-size: 0.95rem !important;
  color: var(--text) !important;
}

/* How-it-works footer */
#footer-md {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 20px 24px !important;
  margin-top: 8px !important;
}
#footer-md h3 { color: var(--accent2) !important; margin-top: 0 !important; }
#footer-md li { color: var(--text-dim) !important; }
#footer-md strong { color: var(--text) !important; }

/* Pipeline badge row */
#pipeline-badges { text-align: center !important; margin: 10px 0 !important; }
"""


def load_generator():
    global generator
    if generator is None:
        generator = ReportGenerator()
    return generator


def make_bar_chart(scores: dict) -> Image.Image:
    conditions = list(scores.keys())
    values = [scores[c] for c in conditions]
    max_val = max(values)

    # Color palette: top bar gets cyan accent, rest get muted blues
    colors = ["#38bdf8" if v == max_val else "#1e3a5f" for v in values]
    edge_colors = ["#38bdf8" if v == max_val else "#2563eb" for v in values]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#0d1526")

    bars = ax.barh(conditions, values, color=colors,
                   edgecolor=edge_colors, linewidth=0.8, height=0.55)

    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Probability", color="#64748b", fontsize=9)
    ax.set_title("Condition Probabilities  ·  BiomedCLIP", color="#94a3b8",
                 fontsize=10, pad=10)

    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[:].set_color("#1e293b")
    ax.xaxis.label.set_color("#64748b")

    for bar, val in zip(bars, values):
        color = "#38bdf8" if val == max_val else "#94a3b8"
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9, color=color,
                fontweight="bold" if val == max_val else "normal")

    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([c.title() for c in conditions], color="#e2e8f0", fontsize=9)

    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def analyze_xray(image: Image.Image):
    if image is None:
        return None, "⬆️  Upload a chest X-ray to begin.", ""

    try:
        gen = load_generator()
        result = gen.analyze(image)

        chart = make_bar_chart(result["clip"]["scores"])

        top = result["clip"]["top_condition"]
        score = result["clip"]["top_score"]
        low_conf = result["clip"]["low_confidence"]

        if low_conf:
            confidence_str = (
                f"⚠️ **Low confidence** ({score:.1%}) — results may be unreliable.\n\n"
                f"Top prediction: **{top.title()}**"
            )
        else:
            confidence_str = (
                f"✅ **Top prediction: {top.title()}** &nbsp;·&nbsp; {score:.1%} confidence"
            )

        return chart, confidence_str, result["report"]

    except Exception as e:
        return None, f"❌ Error: {str(e)}", ""


# ── UI ──────────────────────────────────────────────────────────
DARK_THEME = gr.themes.Base(
    primary_hue="sky",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0a0f1e",
    body_text_color="#e2e8f0",
    block_background_fill="#111827",
    block_border_color="#1e293b",
    input_background_fill="#0d1526",
    button_primary_background_fill="linear-gradient(135deg,#38bdf8,#818cf8)",
    button_primary_text_color="#0a0f1e",
)

with gr.Blocks(title="Chest X-ray Analyzer") as demo:

    gr.Markdown(
        """
# 🩻 Chest X-ray Analyzer
**AI-assisted analysis — BiomedCLIP &nbsp;·&nbsp; BLIP VQA &nbsp;·&nbsp; 15 NIH conditions**

> ⚠️ **Disclaimer**: For research and educational purposes only.
> This is NOT a medical device and must NOT be used for clinical diagnosis.
        """,
        elem_id="header-md",
    )

    with gr.Row():
        # ── Left column ──────────────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Chest X-ray")
            analyze_btn = gr.Button("⚡ Analyze", variant="primary", elem_id="analyze-btn")
            confidence_output = gr.Markdown(
                "⬆️  Upload an image to begin.",
                elem_id="confidence-md",
            )

        # ── Right column ─────────────────────────────────────
        with gr.Column(scale=2):
            chart_output = gr.Image(label="Condition Scores", type="pil")
            report_output = gr.Textbox(
                label="Generated Radiology Report",
                lines=18,
                interactive=False,
                placeholder="The structured radiology report will appear here after analysis...",
            )

    analyze_btn.click(
        fn=analyze_xray,
        inputs=[image_input],
        outputs=[chart_output, confidence_output, report_output],
    )

    gr.Markdown(
        """
### How it works
1. **BiomedCLIP** (Microsoft) — zero-shot classification against 6 conditions
   (trained on 15M biomedical image-text pairs)
2. **BLIP VQA** (Salesforce) — answers 4 targeted medical questions
   (opacity, effusion, cardiomegaly, pneumothorax)
3. **Report synthesis** — combines scores + visual descriptions into a
   structured radiology-style report
        """,
        elem_id="footer-md",
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=DARK_THEME, css=DARK_CSS)
