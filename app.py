"""
Radiology Report Generation & Evaluation Web App
==================================================
A simple Gradio-based web interface for:
  1. Generating radiology reports from chest X-ray images using LLMs
  2. Evaluating generated reports against ground truth (BLEU, ROUGE, METEOR)

Supports two modes:
  - API mode: Use OpenAI-compatible API (e.g., vLLM, Ollama, OpenAI, etc.)
  - Local mode: Load a local HuggingFace model (requires GPU)

Usage:
    pip install gradio openai
    python app.py

    # With custom port
    python app.py --port 7860 --share
"""

import os
import re
import json
import time
import argparse
import base64
import tempfile
from pathlib import Path

import gradio as gr

# ============================================================
# NLG Metrics (lightweight, no GPU needed)
# ============================================================

def compute_bleu_scores(reference: str, hypothesis: str) -> dict:
    """Compute BLEU-1/2/3/4 for a single pair."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    except ImportError:
        return {"error": "nltk not installed. Run: pip install nltk"}

    smooth = SmoothingFunction().method1
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return {f"BLEU-{n}": 0.0 for n in range(1, 5)}

    results = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights,
                              smoothing_function=smooth)
        results[f"BLEU-{n}"] = round(score, 4)
    return results


def compute_rouge_score(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-L for a single pair."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"error": "rouge-score not installed. Run: pip install rouge-score"}

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {"ROUGE-L": round(scores['rougeL'].fmeasure, 4)}


def compute_meteor_score(reference: str, hypothesis: str) -> dict:
    """Compute METEOR for a single pair."""
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        for resource, name in [('corpora/wordnet', 'wordnet'), ('corpora/omw-1.4', 'omw-1.4')]:
            try:
                nltk.data.find(resource)
            except (LookupError, Exception):
                nltk.download(name, quiet=True)
    except ImportError:
        return {"error": "nltk not installed. Run: pip install nltk"}

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return {"METEOR": 0.0}

    score = meteor_score([ref_tokens], hyp_tokens)
    return {"METEOR": round(score, 4)}


def compute_all_metrics(reference: str, hypothesis: str) -> str:
    """Compute all available NLG metrics and return formatted string."""
    if not reference or not reference.strip():
        return "⚠️ No ground truth report provided. Cannot compute metrics."
    if not hypothesis or not hypothesis.strip():
        return "⚠️ No generated report available. Cannot compute metrics."

    results = {}

    # BLEU
    bleu = compute_bleu_scores(reference, hypothesis)
    if "error" not in bleu:
        results.update(bleu)
    else:
        results["BLEU"] = bleu["error"]

    # ROUGE
    rouge = compute_rouge_score(reference, hypothesis)
    if "error" not in rouge:
        results.update(rouge)
    else:
        results["ROUGE-L"] = rouge["error"]

    # METEOR
    meteor = compute_meteor_score(reference, hypothesis)
    if "error" not in meteor:
        results.update(meteor)
    else:
        results["METEOR"] = meteor["error"]

    # Format output
    lines = ["📊 **Evaluation Metrics**\n"]
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for metric, value in results.items():
        if isinstance(value, float):
            lines.append(f"| {metric} | {value:.4f} |")
        else:
            lines.append(f"| {metric} | {value} |")

    return "\n".join(lines)


# ============================================================
# Report Generation - API Mode
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_report_api(
    image_path: str,
    api_key: str,
    api_base: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
) -> str:
    """Generate a report using an OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        return "❌ Error: openai package not installed. Run: pip install openai"

    if not api_key or not api_key.strip():
        return "❌ Error: Please provide an API key."
    if not image_path:
        return "❌ Error: Please upload an image."

    # Encode image
    base64_image = encode_image_to_base64(image_path)

    # Determine image MIME type
    ext = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif"}
    mime_type = mime_map.get(ext, "image/jpeg")

    client = OpenAI(api_key=api_key, base_url=api_base if api_base.strip() else None)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        report = response.choices[0].message.content
        # Clean thinking tags if present
        report = re.sub(r'<think>.*?</think>', '', report, flags=re.DOTALL).strip()
        return report
    except Exception as e:
        return f"❌ API Error: {str(e)}"


# ============================================================
# Report Generation - Local Mode
# ============================================================

# Global model cache
_local_model = None
_local_processor = None
_local_model_name = None


def load_local_model(model_name: str):
    """Load a local HuggingFace model (cached globally)."""
    global _local_model, _local_processor, _local_model_name

    if _local_model_name == model_name and _local_model is not None:
        return _local_model, _local_processor

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[INFO] Loading local model: {model_name}")
    _local_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _local_processor = AutoProcessor.from_pretrained(model_name)
    _local_model_name = model_name
    print(f"[INFO] Model loaded successfully.")
    return _local_model, _local_processor


def generate_report_local(
    image_path: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
) -> str:
    """Generate a report using a local HuggingFace model."""
    if not image_path:
        return "❌ Error: Please upload an image."

    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return "❌ Error: Required packages not installed. Run: pip install torch transformers qwen-vl-utils"

    try:
        model, processor = load_local_model(model_name)
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ]
        }
    ]

    try:
        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed = generated[0][inputs["input_ids"].shape[-1]:]
        output_text = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
        return output_text
    except Exception as e:
        return f"❌ Error during generation: {str(e)}"


# ============================================================
# Main Gradio Interface
# ============================================================

def generate_and_evaluate(
    image,
    mode,
    api_key,
    api_base,
    model_name,
    prompt,
    max_tokens,
    ground_truth,
):
    """Main function: generate report and optionally evaluate."""
    if image is None:
        return "❌ Please upload a chest X-ray image.", ""

    # Generate report
    start_time = time.time()

    if mode == "API (OpenAI-compatible)":
        report = generate_report_api(
            image_path=image,
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            prompt=prompt,
            max_tokens=int(max_tokens),
        )
    else:  # Local Model
        report = generate_report_local(
            image_path=image,
            model_name=model_name,
            prompt=prompt,
            max_tokens=int(max_tokens),
        )

    elapsed = time.time() - start_time

    # Add generation info
    report_display = f"{report}\n\n---\n⏱️ Generation time: {elapsed:.1f}s | Model: {model_name}"

    # Evaluate if ground truth provided
    if ground_truth and ground_truth.strip() and not report.startswith("❌"):
        metrics_output = compute_all_metrics(ground_truth, report)
    else:
        if report.startswith("❌"):
            metrics_output = "⚠️ Report generation failed. Cannot compute metrics."
        else:
            metrics_output = "💡 Provide a ground truth report (left panel) to compute evaluation metrics."

    return report_display, metrics_output


def build_app():
    """Build the Gradio interface."""

    with gr.Blocks(
        title="🏥 Radiology Report Generation & Evaluation",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("""
# 🏥 Radiology Report Generation & Evaluation

Upload a chest X-ray image and generate a radiology report using an LLM.
Optionally provide a ground truth report to compute evaluation metrics (BLEU, ROUGE-L, METEOR).

**Two modes available:**
- **API mode**: Use any OpenAI-compatible API (vLLM, Ollama, OpenAI, Together AI, etc.)
- **Local mode**: Load a HuggingFace model locally (requires GPU)
        """)

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")

                image_input = gr.Image(
                    label="Chest X-ray Image",
                    type="filepath",
                    height=300,
                )

                mode_selector = gr.Radio(
                    choices=["API (OpenAI-compatible)", "Local Model"],
                    value="API (OpenAI-compatible)",
                    label="Generation Mode",
                )

                with gr.Group() as api_group:
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="sk-... or your API key",
                        type="password",
                    )
                    api_base_input = gr.Textbox(
                        label="API Base URL (optional)",
                        placeholder="https://api.openai.com/v1 (leave empty for OpenAI default)",
                        value="",
                    )

                model_name_input = gr.Textbox(
                    label="Model Name",
                    placeholder="e.g., gpt-4o, Qwen/Qwen2.5-VL-7B-Instruct",
                    value="gpt-4o",
                )

                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="Please generate a detailed radiology report for this chest X-ray image. "
                          "Include findings and impressions.",
                    lines=3,
                )

                max_tokens_input = gr.Slider(
                    label="Max Tokens",
                    minimum=64,
                    maximum=4096,
                    value=512,
                    step=64,
                )

                gr.Markdown("### 📋 Ground Truth (optional)")
                ground_truth_input = gr.Textbox(
                    label="Ground Truth Report",
                    placeholder="Paste the reference radiology report here for evaluation...",
                    lines=5,
                )

                generate_btn = gr.Button("🚀 Generate Report", variant="primary", size="lg")

            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Generated Report")
                report_output = gr.Textbox(
                    label="Generated Report",
                    lines=12,
                    show_copy_button=True,
                )

                gr.Markdown("### 📊 Evaluation Metrics")
                metrics_output = gr.Markdown(
                    value="💡 Generate a report and provide ground truth to see metrics here."
                )

        # Event handlers
        generate_btn.click(
            fn=generate_and_evaluate,
            inputs=[
                image_input,
                mode_selector,
                api_key_input,
                api_base_input,
                model_name_input,
                prompt_input,
                max_tokens_input,
                ground_truth_input,
            ],
            outputs=[report_output, metrics_output],
        )

        # Show/hide API fields based on mode
        def update_mode_visibility(mode):
            if mode == "API (OpenAI-compatible)":
                return gr.update(visible=True), gr.update(value="gpt-4o")
            else:
                return gr.update(visible=True), gr.update(value="Qwen/Qwen2.5-VL-7B-Instruct")

        mode_selector.change(
            fn=update_mode_visibility,
            inputs=[mode_selector],
            outputs=[api_group, model_name_input],
        )

        # Examples
        gr.Markdown("""
---
### 💡 Tips

- **API Mode**: Works with any OpenAI-compatible endpoint. For local vLLM servers, set the base URL to `http://localhost:8000/v1`.
- **Local Mode**: Requires a GPU with sufficient VRAM. The model is cached after first load.
- **Metrics**: BLEU, ROUGE-L, and METEOR are computed instantly (no GPU needed).
- **Supported API providers**: OpenAI, Azure OpenAI, vLLM, Ollama (`http://localhost:11434/v1`), Together AI, etc.
        """)

    return app


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radiology Report Generation Web App")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name/IP")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
