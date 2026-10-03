from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mfg_forecast.core import (
    add_benchmark,
    backtest,
    evaluate,
    forecast_at_origin,
    load_history,
    make_synthetic_history,
)


@pytest.fixture
def history(tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "synthetic.csv"
    make_synthetic_history(path, items=3, months=48)
    return load_history(path)


def test_future_actuals_cannot_change_past_forecast(history: pd.DataFrame):
    origin = pd.Timestamp("2022-12-01")
    original = forecast_at_origin(history, origin, models=("random_forest",))
    changed = history.copy()
    changed.loc[changed["month"] > origin, "quantity"] *= 100
    again = forecast_at_origin(changed, origin, models=("random_forest",))
    pd.testing.assert_frame_equal(original, again)


def test_backtest_has_matched_actuals_and_expected_rows(history: pd.DataFrame):
    result = backtest(history, origins=3, models=("naive", "seasonal_naive", "random_forest"))
    assert len(result) == 3 * 3 * 3
    assert (result["target_month"] > result["origin_month"]).all()
    assert result["actual_qty"].notna().all()
    assert result["forecast_qty"].ge(0).all()


def test_benchmark_restricts_models_to_same_observations(history: pd.DataFrame, tmp_path: Path):
    result = backtest(history, origins=3, models=("naive", "random_forest"))
    single = result[result["model"] == "naive"].head(2).copy()
    benchmark = single[["origin_month", "target_month", "item_id", "forecast_qty"]]
    file = tmp_path / "benchmark.csv"
    benchmark.to_csv(file, index=False)
    matched = add_benchmark(result, file)
    assert set(matched["model"]) == {"naive", "random_forest", "external_benchmark"}
    assert matched.groupby("model").size().nunique() == 1


def test_hit_rate_is_not_wape():
    rows = pd.DataFrame({"model": ["example", "example"], "forecast_qty": [80, 150], "actual_qty": [100, 100]})
    result = evaluate(rows).iloc[0]
    assert result["hit_rate_25"] == 0.5
    assert np.isclose(result["wape"], 0.35)


def test_missing_month_is_rejected(tmp_path: Path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"month": ["2025-01-01", "2025-03-01"], "item_id": ["A", "A"], "quantity": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing month"):
        load_history(path)

