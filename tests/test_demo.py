from pathlib import Path

import pandas as pd

from mfg_forecast.core import backtest, evaluate, load_history


def test_teaching_demo_uses_later_months_and_beats_baselines():
    sample = Path(__file__).resolve().parents[1] / "examples" / "synthetic_rf_demo.csv"
    predictions = backtest(load_history(sample), horizon=1, origins=12)
    assert len(predictions) == 3 * 6 * 12
    assert predictions["target_month"].min() == pd.Timestamp("2025-01-01")
    assert predictions["target_month"].max() == pd.Timestamp("2025-12-01")
    metrics = evaluate(predictions).set_index("model")
    assert metrics.loc["random_forest", "wape"] < metrics.loc["seasonal_naive", "wape"]
    assert metrics.loc["random_forest", "hit_rate_25"] > metrics.loc["seasonal_naive", "hit_rate_25"]
