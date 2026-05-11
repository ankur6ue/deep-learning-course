#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


CATEGORY_NAMES = [
    "Groceries",
    "Travel",
    "Dining",
    "Gas",
    "Entertainment",
    "Shopping",
]

MERCHANT_NAMES = [
    "Amazon",
    "Uber",
    "WholeFoods",
    "Netflix",
    "Starbucks",
    "Target",
    "Delta",
    "Shell",
    "Apple",
    "Airbnb",
]

MONTH_SETS_3 = [
    ["Jan", "Feb", "Mar"],
    ["Apr", "May", "Jun"],
    ["Jul", "Aug", "Sep"],
    ["Oct", "Nov", "Dec"],
]

MONTH_SETS_4 = [
    ["Jan", "Feb", "Mar", "Apr"],
    ["May", "Jun", "Jul", "Aug"],
    ["Sep", "Oct", "Nov", "Dec"],
]

MONTH_SETS_5 = [
    ["Jan", "Feb", "Mar", "Apr", "May"],
    ["Jun", "Jul", "Aug", "Sep", "Oct"],
    ["Nov", "Dec", "Jan", "Feb", "Mar"],
]


@dataclass
class PromptData:
    idx: int
    kind: str
    months: list[str]
    items: list[str]
    values: dict[str, dict[str, int]]
    user_prompt: str


def build_values(
    rng: random.Random,
    items: list[str],
    months: list[str],
    kind: str,
) -> dict[str, dict[str, int]]:
    values: dict[str, dict[str, int]] = {}

    for item in items:
        if kind == "category":
            base = rng.randint(120, 600)
            noise = rng.randint(10, 60)
        else:
            if item == "Netflix":
                base = rng.randint(15, 20)
                noise = rng.randint(0, 2)
            elif item in {"Delta", "Airbnb", "Apple"}:
                base = rng.randint(0, 260)
                noise = rng.randint(30, 120)
            else:
                base = rng.randint(40, 320)
                noise = rng.randint(8, 50)

        trend = rng.uniform(-0.18, 0.22)
        cur = float(base)
        month_map: dict[str, int] = {}

        for month in months:
            seasonal = rng.uniform(-noise, noise)
            cur = max(0.0, cur * (1.0 + trend) + seasonal)

            if kind == "merchant" and item in {"Delta", "Airbnb", "Apple"} and rng.random() < 0.18:
                cur = max(0.0, cur * rng.uniform(0.0, 0.35))
            if kind == "merchant" and item in {"Delta", "Airbnb"} and rng.random() < 0.15:
                cur += rng.uniform(120, 260)

            month_map[month] = int(round(cur))

        values[item] = month_map

    return values


def render_user_prompt(
    kind: str,
    months: list[str],
    items: list[str],
    values: dict[str, dict[str, int]],
) -> str:
    lines: list[str] = []
    label = "category" if kind == "category" else "merchant"
    lines.append(f"Data type: {label}-level spending")
    lines.append("")

    for month in months:
        parts = [f"{item} ${values[item][month]}" for item in items]
        lines.append(f"{month}: " + ", ".join(parts))

    lines.append("")
    lines.append("Please summarize how each item changed month over month, including dollar and percentage changes.")
    return "\n".join(lines)


def generate_prompt_data(
    num_prompts: int,
    seed: int,
    include_categories: bool,
    include_merchants: bool,
) -> list[PromptData]:
    if not include_categories and not include_merchants:
        raise ValueError("At least one of include_categories/include_merchants must be true.")

    rng = random.Random(seed)
    rows: list[PromptData] = []

    available_kinds: list[str] = []
    if include_categories:
        available_kinds.append("category")
    if include_merchants:
        available_kinds.append("merchant")

    for idx in range(num_prompts):
        kind = available_kinds[idx % len(available_kinds)]
        if len(available_kinds) > 1:
            kind = rng.choice(available_kinds)

        month_count = rng.choice([3, 4, 5])
        if month_count == 3:
            months = list(rng.choice(MONTH_SETS_3))
        elif month_count == 4:
            months = list(rng.choice(MONTH_SETS_4))
        else:
            months = list(rng.choice(MONTH_SETS_5))

        items = list(CATEGORY_NAMES if kind == "category" else MERCHANT_NAMES)
        values = build_values(rng, items, months, kind)
        user_prompt = render_user_prompt(kind, months, items, values)

        rows.append(
            PromptData(
                idx=idx,
                kind=kind,
                months=months,
                items=items,
                values=values,
                user_prompt=user_prompt,
            )
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--categories-only", action="store_true")
    parser.add_argument("--merchants-only", action="store_true")
    args = parser.parse_args()

    if args.categories_only and args.merchants_only:
        raise ValueError("Choose at most one of --categories-only / --merchants-only")

    rows = generate_prompt_data(
        num_prompts=args.num_prompts,
        seed=args.seed,
        include_categories=not args.merchants_only,
        include_merchants=not args.categories_only,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(rows)} prompt records to {output_path}")


if __name__ == "__main__":
    main()