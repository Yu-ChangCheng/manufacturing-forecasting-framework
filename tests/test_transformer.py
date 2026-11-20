from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mfg_forecast.core import backtest, forecast_at_origin, load_history
from mfg_forecast.transformer import TransformerConfig


def test_transformer_settings_reject_incompatible_heads():
    with pytest.raises(ValueError, match="positive multiple"):
        TransformerConfig(width=16, heads=3)


def test_transformer_uses_same_rows_and_cannot_see_future_actuals():
    pytest.importorskip("tensorflow")
    root = Path(__file__).resolve().parents[1]
    history = load_history(root / "examples" / "synthetic_rf_demo.csv")
    config = TransformerConfig(epochs=1, width=8, heads=2)
    results = backtest(
        history, horizon=2, origins=1, models=("naive", "transformer"), transformer_config=config,
    )
    keys = ["origin_month", "target_month", "item_id"]
    naive = results[results["model"].eq("naive")].set_index(keys).sort_index()
    transformer = results[results["model"].eq("transformer")].set_index(keys).sort_index()
    assert len(transformer) == history["item_id"].nunique()
    pd.testing.assert_series_equal(naive["actual_qty"], transformer["actual_qty"])
    assert np.isfinite(transformer["forecast_qty"]).all()
    assert transformer["forecast_qty"].ge(0).all()

    origin = transformer.index.get_level_values("origin_month")[0]
    changed = history.copy()
    changed.loc[changed["month"] > origin, "quantity"] *= 100
    replayed = forecast_at_origin(
        changed, origin, horizon=2, models=("transformer",), transformer_config=config,
    )
    np.testing.assert_allclose(
        transformer["forecast_qty"], replayed.set_index(keys).sort_index()["forecast_qty"], rtol=1e-6,
    )
