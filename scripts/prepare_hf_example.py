"""Recreate the small public car-parts example from a pinned Hugging Face file.

Download SOURCE_URL to outputs/hf_car_parts_source.tsf, then run this script with
--source outputs/hf_car_parts_source.tsf. The source file stays outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


SOURCE_URL = (
    "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/"
    "ea89753da2b451928436adb333c7a2e892461c7d/forecasting/monash/"
    "car_parts_dataset_without_missing_values.tsf"
)
SOURCE_SHA256 = "4308d4c75319f14ea3cfd853f5e7805d2f0c2eeb9a9a77e3ff517a8979c0a4f7"


def convert(source: Path, output: Path, count: int = 5) -> pd.DataFrame:
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source checksum differs from the reviewed Hugging Face version")
    series = []
    for line in content.decode("utf-8").splitlines():
        if not line or line.startswith(("#", "@")):
            continue
        name, start, values = line.split(":", 2)
        quantities = [float(value) for value in values.split(",")]
        series.append((name, pd.Timestamp(start.split(" ")[0]), quantities))
    if not series or count < 1 or count > len(series):
        raise ValueError("Invalid source or requested item count")

    # Pick busy series for an easier chart, without looking at any model results.
    selected = sorted(
        series,
        key=lambda row: (-sum(q > 0 for q in row[2]), -sum(row[2]), row[0]),
    )[:count]
    rows = []
    for name, start, quantities in selected:
        for month, quantity in zip(pd.date_range(start, periods=len(quantities), freq="MS"), quantities):
            rows.append((month.strftime("%Y-%m-%d"), name, quantity))
    example = pd.DataFrame(rows, columns=["month", "item_id", "quantity"])
    example = example.sort_values(["item_id", "month"])
    if example["quantity"].mod(1).eq(0).all():
        example["quantity"] = example["quantity"].astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    example.to_csv(output, index=False)
    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("examples/hf_car_parts_sample.csv"))
    parser.add_argument("--items", type=int, default=5)
    args = parser.parse_args()
    result = convert(args.source, args.output, args.items)
    print(f"Wrote {len(result)} rows for {result['item_id'].nunique()} items to {args.output}")
    print("Selected IDs:", ", ".join(sorted(result["item_id"].unique())))
