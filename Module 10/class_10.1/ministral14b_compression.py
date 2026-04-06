#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

# Note for GPTQ, we may need to:
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --num-calibration-samples 64
# --max-seq-length 1024
# to avoid OOM errors

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        type=str,
        default="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours",
        help="Path to extracted text-only BF16 checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16", # load tokenizer from original BF16 directory
        help="Path to the canonical tokenizer source.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours-GPTQ-W4A16",
        help="Where to save the quantized checkpoint.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["gptq", "nvfp4"],
        default="gptq",
        help="Quantization method to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train_sft",
        help="Dataset split.",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=128,
        help="Number of calibration examples.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Calibration max sequence length.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="W4A16",
        help="GPTQ scheme, e.g. W4A16.",
    )
    parser.add_argument(
        "--nvfp4-scheme",
        type=str,
        default="NVFP4",
        help="NVFP4 scheme string.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="Linear",
        help="Quantization targets.",
    )
    parser.add_argument(
        "--ignore-lm-head",
        action="store_true",
        default=True,
        help="Ignore lm_head during quantization.",
    )

    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"[INFO] Copied {src.name}")


def copy_tokenizer_assets(src_dir: Path, dst_dir: Path) -> None:
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tekken.json",
        "tokenizer.model",
        "added_tokens.json",
    ]
    for name in tokenizer_files:
        copy_if_exists(src_dir / name, dst_dir / name)


def load_calibration_dataset(
    tokenizer,
    dataset_name: str,
    dataset_split: str,
    num_samples: int,
    max_seq_length: int,
):
    split = f"{dataset_split}[:{num_samples}]"
    print(f"[INFO] Loading calibration dataset: {dataset_name} {split}")
    ds = load_dataset(dataset_name, split=split).shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    ds = ds.map(preprocess)

    def tokenize_fn(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    return ds


def build_recipe(args: argparse.Namespace, ignore: list[str]):
    if args.method == "gptq":
        return GPTQModifier(
            targets=args.targets,
            scheme=args.scheme,
            ignore=ignore,
        )

    if args.method == "nvfp4":
        return QuantizationModifier(
            targets=args.targets,
            scheme=args.nvfp4_scheme,
            ignore=ignore,
        )

    raise ValueError(f"Unsupported method: {args.method}")


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()

    model_id = args.model_id
    tokenizer_id = args.tokenizer_id
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading tokenizer from: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        fix_mistral_regex=True,
        trust_remote_code=True,
    )

    ds = load_calibration_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        num_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )

    print(f"[INFO] Loading model from: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    ignore = ["lm_head"] if args.ignore_lm_head else []
    recipe = build_recipe(args, ignore)

    print(f"[INFO] Running {args.method.upper()} oneshot quantization...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        processor=tokenizer,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    print(f"[INFO] Saving quantized model to: {output_dir}")
    model.save_pretrained(
        output_dir,
        save_compressed=True,
    )

    print(f"[INFO] Copying canonical tokenizer assets from: {tokenizer_id}")
    copy_tokenizer_assets(Path(tokenizer_id), output_dir)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()