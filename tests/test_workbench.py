from pathlib import Path

import pandas as pd

from mfg_forecast.core import load_history
from mfg_forecast.workbench import (
    FeatureConfig, build_supervised, fit_forest, forecast_future,
    save_forecaster, load_forecaster, score_holdout, split_time, tune_forest,
)


SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_rf_demo.csv"


def test_three_month_horizon_keeps_future_actuals_out_of_training():
    history = load_history(SAMPLE)
    features = FeatureConfig(lags=(1, 12), rolling_windows=(3,), include_month=True)
    rows = build_supervised(history, horizon=3, features=features)
    split = split_time(rows, test_origins=6)
    assert split.train["target_month"].max() <= split.first_test_origin
    assert split.test["origin_month"].min() == split.first_test_origin

    changed = history.copy()
    changed.loc[changed["month"] > split.first_test_origin, "quantity"] *= 10
    changed_split = split_time(build_supervised(changed, horizon=3, features=features), test_origins=6)
    pd.testing.assert_frame_equal(split.train, changed_split.train)
    first_rows = split.test[split.test["origin_month"] == split.first_test_origin]
    changed_first = changed_split.test[changed_split.test["origin_month"] == split.first_test_origin]
    pd.testing.assert_frame_equal(first_rows[features.columns], changed_first[features.columns])


def test_tune_score_future_and_save_model(tmp_path: Path):
    history = load_history(SAMPLE)
    features = FeatureConfig(lags=(1, 12, 24), rolling_windows=(3, 12), include_month=True)
    rows = build_supervised(history, features=features)
    split = split_time(rows, test_origins=12)
    tuning = tune_forest(split.train, features, trials=2, folds=2, months_per_fold=2)
    assert tuning["cv_forecasts"].min() > 0
    selected = dict(tuning.iloc[0]["params"])
    model = fit_forest(split.train, features, selected)
    predictions, metrics = score_holdout(model, split.test, features)
    assert predictions.groupby("model").size().to_dict() == {
        "last_month": 72, "same_month_last_year": 72, "random_forest": 72,
    }
    assert metrics["n"].eq(72).all()

    final_model = fit_forest(rows, features, selected)
    future = forecast_future(history, final_model, features)
    assert len(future) == 6
    assert future["target_month"].eq(pd.Timestamp("2026-01-01")).all()
    saved = save_forecaster(tmp_path / "model.joblib", final_model, features, 1, history["month"].max())
    loaded_model, loaded_features, loaded_horizon, trained_through = load_forecaster(saved)
    assert loaded_horizon == 1
    assert loaded_features == features
    assert trained_through == history["month"].max()
    pd.testing.assert_frame_equal(forecast_future(history, loaded_model, loaded_features), future)
