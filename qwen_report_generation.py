"""
Qwen VL Radiology Report Generation
=====================================
Run Qwen2.5-VL / Qwen3-VL / Qwen3.5 on chest X-ray images for report generation.

Input:
  - test_dataset.json: Qwen VL conversation format
    Format: {"test": [{"id": "sample_id", "role": "user", "content": [{"type": "image", "image": "path/to/image.jpg"}, {"type": "text", "text": "prompt"}]}, ...]}
  - Model: Qwen2.5-VL / Qwen3-VL / Qwen3.5 (HuggingFace)

Output:
  - Generated reports JSON: {"test": [{"id": "sample_id", "output": "report text"}, ...]}

Usage:
    # Qwen2.5-VL (default, uses AutoModelForImageTextToText)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python qwen_report_generation.py \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --question_file ./data/test_dataset.json \
        --output_file ./results/qwen_output.json \
        --max_tokens 512

    # Qwen3-VL (uses Qwen3VLForConditionalGeneration)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python qwen_report_generation.py \
        --model_name Qwen/Qwen3-VL-8B-Instruct \
        --model_type qwen3vl \
        --question_file ./data/test_dataset.json \
        --output_file ./results/qwen_output_qwen3vl.json \
        --max_tokens 512
"""

import json
import argparse
import os
import re
import time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Qwen3-VL specific import (lazy, only used when --model_type qwen3vl)
Qwen3VLForConditionalGeneration = None


def _load_qwen3vl_class():
    """Lazy import Qwen3VLForConditionalGeneration to avoid import errors
    when the class is not available in the installed transformers version."""
    global Qwen3VLForConditionalGeneration
    if Qwen3VLForConditionalGeneration is None:
        from transformers import Qwen3VLForConditionalGeneration as _cls
        Qwen3VLForConditionalGeneration = _cls
    return Qwen3VLForConditionalGeneration


def parse_thinking_output(text: str) -> tuple:
    """
    Parse Qwen3.5 thinking mode output.
    The model outputs in format: <think>...</think>answer

    Returns (thinking_text, answer_text, is_truncated).
    """
    text = re.sub(r"<\|im_end\|>", "", text).strip()

    # Case 1: Complete <think>...</think> pattern
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = match.group(2).strip()
        return thinking, answer, False

    # Case 2: Only </think> present (opening <think> was part of input prompt)
    match_close = re.search(r"^(.*?)</think>(.*)", text, re.DOTALL)
    if match_close:
        thinking = match_close.group(1).strip()
        answer = match_close.group(2).strip()
        return thinking, answer, False

    # Case 3: <think> opened but never closed (truncated)
    match_open = re.search(r"<think>(.*)", text, re.DOTALL)
    if match_open:
        thinking = match_open.group(1).strip()
        return thinking, "", True

    # Fallback: no thinking tags
    return "", text.strip(), False


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen VL inference for radiology report generation"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model name or local path of the Qwen VL model",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "qwen3vl"],
        help="Model loading type. 'auto': use AutoModelForImageTextToText (for Qwen2.5-VL, Qwen3.5); "
             "'qwen3vl': use Qwen3VLForConditionalGeneration (for Qwen3-VL series)",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        required=True,
        help="Path to test dataset JSON",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/qwen_output.json",
        help="Path to save generated reports",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=256,
        help="Min pixels for image processing (multiplied by 28*28)",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1280,
        help="Max pixels for image processing (multiplied by 28*28)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate (recommend >=2048 for thinking mode)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for Qwen3.5 models",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to process (for debugging). None = all.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing output file (skip already processed samples)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save intermediate results every N samples",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        default=False,
        help="Use flash_attention_2 for faster inference and lower memory",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # ============================================================
    # Load model and processor
    # ============================================================
    print(f"[INFO] Loading model: {args.model_name}")
    print(f"[INFO] Model type: {args.model_type}")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"[INFO] Using flash_attention_2")

    if args.model_type == "qwen3vl":
        Qwen3VLCls = _load_qwen3vl_class()
        model = Qwen3VLCls.from_pretrained(
            args.model_name,
            **model_kwargs,
        )
        print(f"[INFO] Loaded with Qwen3VLForConditionalGeneration")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            **model_kwargs,
        )
        print(f"[INFO] Loaded with AutoModelForImageTextToText")

    min_pixels = args.min_pixels * 28 * 28
    max_pixels = args.max_pixels * 28 * 28
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Print multi-GPU info
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"[INFO] Model loaded across {n_gpus} GPU(s) via device_map='auto'")
        for i in range(n_gpus):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"        GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    else:
        print(f"[INFO] Model loaded on CPU")
    print(f"[INFO] Image pixels range: {min_pixels} - {max_pixels}")
    print(f"[INFO] Max new tokens: {args.max_tokens}")
    if args.enable_thinking:
        print(f"[INFO] Thinking mode ENABLED")
        if args.max_tokens < 2048:
            print(f"[WARN] Thinking mode usually needs >=2048 max_tokens, current={args.max_tokens}")

    # ============================================================
    # Load test dataset
    # ============================================================
    print(f"[INFO] Loading test dataset: {args.question_file}")
    with open(args.question_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["test"] if isinstance(data, dict) and "test" in data else data
    print(f"[INFO] Total test samples: {len(questions)}")

    if args.max_samples is not None:
        questions = questions[: args.max_samples]
        print(f"[INFO] Limited to {len(questions)} samples")

    # ============================================================
    # Resume support
    # ============================================================
    processed_ids = set()
    outputs = {"test": []}

    if args.resume and os.path.exists(args.output_file):
        print(f"[INFO] Resuming from {args.output_file}")
        with open(args.output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        outputs = existing
        processed_ids = {item["id"] for item in existing.get("test", [])}
        print(f"[INFO] Already processed: {len(processed_ids)} samples")

    # ============================================================
    # Run inference
    # ============================================================
    print(f"\n[INFO] Starting inference...")
    start_time = time.time()
    n_processed = 0
    n_skipped = 0
    n_errors = 0

    for i, question in enumerate(tqdm(questions, desc="Inference")):
        sample_id = question["id"]

        if sample_id in processed_ids:
            n_skipped += 1
            continue

        try:
            messages = [question]
            image_inputs, video_inputs = process_vision_info(messages)

            if args.model_type == "qwen3vl":
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = processor.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=args.enable_thinking,
                )

            inputs = processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            generate_kwargs = {
                "max_new_tokens": args.max_tokens,
            }
            if args.model_type == "qwen3vl" and args.enable_thinking:
                generate_kwargs["enable_thinking"] = True

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    **generate_kwargs,
                )

            generated_ids_trimmed = generated[0][inputs["input_ids"].shape[-1]:]
            output_text = processor.decode(
                generated_ids_trimmed,
                skip_special_tokens=not args.enable_thinking,
            )

            result_entry = {"id": sample_id}
            if args.enable_thinking:
                thinking, answer, truncated = parse_thinking_output(output_text)
                result_entry["thinking"] = thinking
                result_entry["output"] = answer
                if truncated:
                    result_entry["truncated"] = True
            else:
                result_entry["output"] = output_text

            outputs["test"].append(result_entry)
            n_processed += 1

            if n_processed <= 5:
                print(f"\n{'─'*60}")
                print(f"  Sample [{n_processed}] ID: {sample_id}")
                print(f"{'─'*60}")
                if args.enable_thinking and result_entry.get("thinking"):
                    print(f"  [Thinking]\n{result_entry['thinking']}")
                    print(f"  [Answer]")
                if result_entry.get("truncated"):
                    print(f"  ⚠️  OUTPUT TRUNCATED (thinking not finished, increase --max_tokens)")
                print(f"{result_entry['output']}")
                print(f"{'─'*60}")

        except Exception as e:
            print(f"\n[ERROR] Sample {sample_id}: {str(e)[:200]}")
            outputs["test"].append({
                "id": sample_id,
                "output": "",
                "error": str(e)[:500],
            })
            n_errors += 1

        # Periodic save
        if (n_processed + n_errors) % args.save_every == 0 and (n_processed + n_errors) > 0:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
            elapsed = time.time() - start_time
            speed = (n_processed + n_errors) / elapsed * 3600
            print(f"\n[SAVE] {n_processed + n_errors} samples processed, "
                  f"{n_errors} errors, {speed:.0f} samples/hour")

    # ============================================================
    # Final save
    # ============================================================
    elapsed = time.time() - start_time
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Inference Complete")
    print(f"{'='*60}")
    print(f"  Model:       {args.model_name}")
    print(f"  Processed:   {n_processed}")
    print(f"  Skipped:     {n_skipped}")
    print(f"  Errors:      {n_errors}")
    print(f"  Total time:  {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if n_processed > 0:
        print(f"  Speed:       {n_processed/elapsed*3600:.0f} samples/hour")
    print(f"  Output:      {args.output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
