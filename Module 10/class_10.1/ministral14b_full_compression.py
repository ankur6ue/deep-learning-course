#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from transformers import MistralForCausalLM
import copy
import gc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        type=str,
        default="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-PATCHED-457",
        help="Path to full BF16 Ministral checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-Full-NVFP4",
        help="Where to save the quantized checkpoint.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["gptq", "nvfp4"],
        default="nvfp4",
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
    parser.add_argument(
        "--include-vision",
        action="store_true",
        default=False,
        help="Also quantize vision/projection modules. Off by default.",
    )

    return parser.parse_args()


def load_calibration_dataset(
    processor,
    dataset_name: str,
    dataset_split: str,
    num_samples: int,
    max_seq_length: int,
):
    split = f"{dataset_split}[:{num_samples}]"
    print(f"[INFO] Loading calibration dataset: {dataset_name} {split}")
    ds = load_dataset(dataset_name, split=split).shuffle(seed=42)

    def preprocess(example):
        text = processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(preprocess)

    def tokenize_fn(sample):
        # Text-only calibration, even though model is multimodal.
        out = processor(
            text=sample["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

        # llmcompressor / trainers generally expect plain dict fields.
        # Keep only text-side tensors.
        keep = {}
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in out:
                keep[key] = out[key]
        return keep

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    return ds


def build_ignore_list(args: argparse.Namespace) -> list[str]:
    ignore = ["lm_head"] if args.ignore_lm_head else []

    # if not args.include_vision:
    #     ignore += [
    #         r"re:.*vision_tower.*",
    #         r"re:.*multi_modal_projector.*",
    #         r"re:.*vision_model.*",
    #         r"re:.*visual.*",
    #     ]

    return ignore


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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()

    model_id = args.model_id
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading processor from: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    ds = load_calibration_dataset(
        processor=processor,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        num_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )

    print(f"[INFO] Loading full model from: {model_id}")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    ignore = build_ignore_list(args)
    recipe = build_recipe(args, ignore)

    print(f"[INFO] Running {args.method.upper()} oneshot quantization...")
    print(f"[INFO] Running {args.method.upper()} oneshot quantization...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        processor=processor,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    print("[INFO] Building quantized text-only Mistral model...")

    text_config = copy.deepcopy(model.model.language_model.config)
    text_config.tie_word_embeddings = False

    lm = model.model.language_model
    lm_head = model.lm_head

    print("[INFO] Releasing multimodal wrapper references...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[INFO] Saving quantized language model to: {output_dir}")
    lm.save_pretrained(
        output_dir,
        save_compressed=True,
        max_shard_size="1GB",
    )

    print("[INFO] Saving lm_head separately...")
    torch.save(lm_head.state_dict(), Path(output_dir) / "lm_head_state_dict.pt")

    print(f"[INFO] Saving processor to: {output_dir}")
    processor.save_pretrained(output_dir)

    print("[INFO] Done.")

if __name__ == "__main__":
    main()