"""Recreate the fictional, seasonally patterned Random Forest teaching CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEASONAL_MULTIPLIER = (0.72, 0.80, 0.95, 1.05, 1.18, 1.32,
                       1.47, 1.34, 1.14, 0.98, 0.89, 1.15)
SEED = 20261003


def generate(path: str | Path) -> pd.DataFrame:
    """Monthly quantities = item scale × recurring month pattern + random noise."""
    rng = np.random.default_rng(SEED)
    months = pd.date_range("2018-01-01", periods=96, freq="MS")
    records = []
    for item_number in range(1, 7):
        base = 80 + 30 * item_number
        for month in months:
            expected = base * SEASONAL_MULTIPLIER[month.month - 1]
            quantity = max(0, round(expected + rng.normal(0, base * 0.12)))
            records.append((month.strftime("%Y-%m-01"), f"DEMO-{item_number:02d}", quantity))
    frame = pd.DataFrame(records, columns=["month", "item_id", "quantity"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "examples" / "synthetic_rf_demo.csv")
