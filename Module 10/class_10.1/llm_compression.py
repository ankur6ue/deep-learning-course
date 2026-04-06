#!/usr/bin/env python3
"""
Quantize Llama 3 Instruct with llmcompressor using different methods.

Examples:

  # Run both GPTQ and NVFP4 with defaults
  python quantize_llama.py \
      --model-id /home/ankur/dev/models/Llama-3.1-8B-Instruct \
      --methods gptq,nvfp4

  # GPTQ only, with fewer calibration samples and shorter max seq length
  python quantize_llama.py \
      --methods gptq \
      --num-calibration-samples 256 \
      --max-seq-length 1024 \
      --gptq-scheme W4A16

  # NVFP4 only, with custom scheme (if supported) and no lm_head ignore
  python quantize_llama.py \
      --methods nvfp4 \
      --nvfp4-scheme NVFP4 \
      --no-ignore-lm-head
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        type=str,
        default="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16",
        help="Base HF model path / ID to quantize.",
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="gptq,nvfp4",
        help=(
            "Comma-separated quantization methods to run. "
            "Supported: gptq, nvfp4"
        ),
    )

    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=256,
        help="Number of calibration samples to use from the dataset.",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max sequence length for calibration.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="HF dataset used for calibration.",
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train_sft",
        help="Dataset split prefix used for calibration.",
    )

    parser.add_argument(
        "--targets",
        type=str,
        default="Linear",
        help="Module target pattern(s) for quantization, e.g. 'Linear'.",
    )

    parser.add_argument(
        "--ignore-lm-head",
        dest="ignore_lm_head",
        action="store_true",
        default=True,
        help="(Default) Ignore lm_head layer during quantization.",
    )

    parser.add_argument(
        "--no-ignore-lm-head",
        dest="ignore_lm_head",
        action="store_false",
        help="Quantize lm_head as well.",
    )

    # GPTQ-specific knobs
    parser.add_argument(
        "--gptq-scheme",
        type=str,
        default="W4A16",
        help="GPTQ quantization scheme string (e.g. 'W4A16').",
    )

    # NVFP4-specific knobs
    parser.add_argument(
        "--nvfp4-scheme",
        type=str,
        default="NVFP4",
        help="NVFP4 quantization scheme string (usually 'NVFP4').",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help="Root directory where quantized models will be saved.",
    )

    return parser.parse_args()


def load_calibration_dataset(
    tokenizer,
    dataset_name: str,
    dataset_split: str,
    num_samples: int,
    max_seq_length: int,
):
    """
    Load and preprocess calibration dataset for oneshot quantization.
    Uses the Ultrachat chat template format you already had.
    """
    split = f"{dataset_split}[:{num_samples}]"
    print(f"[INFO] Loading dataset {dataset_name} split {split}")
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=42)

    # Preprocess into text with chat template.
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    # Tokenize (no extra special tokens; chat template already added them).
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    return ds


def build_save_dir_name(
    model_id: str,
    method: str,
    scheme: str,
    num_calib: int,
    max_seq_length: int,
) -> str:
    """
    Build a save directory name that encodes the quantization config.
    Example:
      Llama-3.1-8B-Instruct-quant-gptq-W4A16-calib512-msl2048
    """
    base_name = Path(model_id.rstrip("/")).name
    return f"{base_name}-quant-{method}-{scheme}-calib{num_calib}-msl{max_seq_length}"


def patch_tokenizer_config(save_dir: Path):
    import json
    cfg_path = save_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print("[INFO] Patched tokenizer_class -> PreTrainedTokenizerFast")


def quantize_with_gptq(
    model_id: str,
    tokenizer,
    ds,
    args: argparse.Namespace,
):
    print("\n[INFO] Running GPTQ quantization...")

    # Load a fresh FP model for this method
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=None,  # let llmcompressor place it as needed
    )

    ignore_list = ["lm_head"] if args.ignore_lm_head else []
    recipe = GPTQModifier(
        targets=args.targets,
        scheme=args.gptq_scheme,
        ignore=ignore_list,
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    save_dir_name = build_save_dir_name(
        model_id=model_id,
        method="gptq",
        scheme=args.gptq_scheme,
        num_calib=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )
    save_dir = Path(args.output_root) / save_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving GPTQ-quantized model to {save_dir}")
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    patch_tokenizer_config(save_dir)

def quantize_with_nvfp4(
    model_id: str,
    tokenizer,
    ds,
    args: argparse.Namespace,
):
    print("\n[INFO] Running NVFP4 quantization...")

    # Load a fresh FP model for this method
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=None,
    )

    ignore_list = ["lm_head"] if args.ignore_lm_head else []
    ignore_list += [
        r"re:.*embed_tokens",
        r"re:model\.vision_tower.*",
        r"re:model\.multi_modal_projector.*",
    ]
    recipe = QuantizationModifier(
        targets=args.targets,
        scheme=args.nvfp4_scheme,  # typically "NVFP4"
        ignore=ignore_list,
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    save_dir_name = build_save_dir_name(
        model_id=model_id,
        method="nvfp4",
        scheme=args.nvfp4_scheme,
        num_calib=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )
    save_dir = Path(args.output_root) / save_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving NVFP4-quantized model to {save_dir}")
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    patch_tokenizer_config(save_dir)

def main():
    args = parse_args()

    model_id = args.model_id
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    print(f"[INFO] Base model: {model_id}")
    print(f"[INFO] Methods to run: {methods}")
    print(
        f"[INFO] Calibration: num_samples={args.num_calibration_samples}, "
        f"max_seq_length={args.max_seq_length}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    ds = load_calibration_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        num_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )

    if "gptq" in methods:
        quantize_with_gptq(model_id, tokenizer, ds, args)

    if "nvfp4" in methods:
        quantize_with_nvfp4(model_id, tokenizer, ds, args)

    print("\n[INFO] Done with all requested quantization methods.")


if __name__ == "__main__":
    main()
