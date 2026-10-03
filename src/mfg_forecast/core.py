"""Data validation, time-respecting forecasts, and transparent evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LOOKBACK = 24
HISTORY_COLUMNS = ["month", "item_id", "quantity"]
PREDICTION_COLUMNS = ["origin_month", "target_month", "item_id", "model", "forecast_qty"]


def _month(value: object) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if date != date.to_period("M").to_timestamp():
        raise ValueError(f"Expected the first day of a month, got {value!r}")
    return date


def load_history(path: str | Path) -> pd.DataFrame:
    """Read one complete monthly series per item; never guess missing demand."""
    return validate_history(pd.read_csv(path, dtype={"item_id": "string"}))


def validate_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an in-memory monthly table for use by the CLI or local app."""
    missing = set(HISTORY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"History is missing columns: {sorted(missing)}")
    df = df[HISTORY_COLUMNS].copy()
    df["month"] = pd.to_datetime(df["month"], errors="raise")
    if not df["month"].eq(df["month"].dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("month must use the first day of each month (YYYY-MM-01)")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="raise")
    if df["quantity"].isna().any() or not np.isfinite(df["quantity"].astype(float)).all() or (df["quantity"] < 0).any():
        raise ValueError("quantity must be a finite, nonnegative number on every row")
    if df["item_id"].isna().any() or df["item_id"].str.strip().eq("").any():
        raise ValueError("item_id must be present on every row")
    if df.duplicated(["item_id", "month"]).any():
        raise ValueError("There must be exactly one row per item_id and month")
    df = df.sort_values(["item_id", "month"]).reset_index(drop=True)
    for item, group in df.groupby("item_id", sort=False):
        expected = pd.date_range(group["month"].iloc[0], group["month"].iloc[-1], freq="MS")
        if len(expected) != len(group):
            raise ValueError(f"Missing month for item {item!r}; decide whether it means zero or unknown")
    return df


def make_synthetic_history(path: str | Path, *, items: int = 6, months: int = 72) -> pd.DataFrame:
    """Create clearly fictional data for learning and screenshots."""
    if items < 1 or months < 36:
        raise ValueError("Use at least one item and 36 months")
    rng = np.random.default_rng(20261003)
    dates = pd.date_range("2020-01-01", periods=months, freq="MS")
    rows = []
    for i in range(items):
        base = 75 + 35 * i
        for t, date in enumerate(dates):
            seasonal = 1 + 0.19 * np.sin(2 * np.pi * (date.month - 1) / 12 + i / 3)
            trend = 1 + 0.0025 * t
            qty = max(0, round(base * seasonal * trend + rng.normal(0, base * 0.07)))
            rows.append((date.strftime("%Y-%m-01"), f"SYNTH-{i+1:03d}", qty))
    df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _rf_features(item: str, values: np.ndarray, anchor: pd.Timestamp, horizon: int) -> dict:
    recent = values[-LOOKBACK:]
    target_month = anchor + pd.DateOffset(months=horizon)
    row = {"item_id": item, "target_month_num": target_month.month, "horizon": horizon}
    for lag in (1, 2, 3, 6, 12, 24):
        row[f"lag_{lag}"] = float(recent[-lag])
    for window in (3, 6, 12):
        row[f"mean_{window}"] = float(np.mean(recent[-window:]))
    return row


def _training_rows(df: pd.DataFrame, origin: pd.Timestamp, horizon: int):
    features, targets = [], []
    for item, group in df[df["month"] <= origin].groupby("item_id", sort=True):
        values = group["quantity"].to_numpy(dtype=float)
        dates = group["month"].tolist()
        # The label must be observed by the forecast origin. This is the leakage boundary.
        for t in range(LOOKBACK - 1, len(values) - horizon):
            history = values[: t + 1]
            features.append(_rf_features(str(item), history, dates[t], horizon))
            targets.append(float(values[t + horizon]))
    return features, np.asarray(targets)


def _predict_rf(df: pd.DataFrame, origin: pd.Timestamp, horizon: int, items: list[str]) -> dict[str, float]:
    features, targets = _training_rows(df, origin, horizon)
    if len(features) < 8:
        raise ValueError("Not enough historical training examples for Random Forest at this origin/horizon")
    X = pd.DataFrame(features)
    model = Pipeline([
        ("encode", ColumnTransformer([
            ("item", OneHotEncoder(handle_unknown="ignore"), ["item_id"]),
        ], remainder="passthrough")),
        ("forest", RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=1)),
    ])
    model.fit(X, targets)
    predict_rows = []
    for item in items:
        group = df[(df["item_id"] == item) & (df["month"] <= origin)]
        predict_rows.append(_rf_features(item, group["quantity"].to_numpy(dtype=float), origin, horizon))
    values = model.predict(pd.DataFrame(predict_rows))
    return dict(zip(items, np.maximum(0, values).astype(float)))


def _predict_lstm(df: pd.DataFrame, origin: pd.Timestamp, horizon: int, items: list[str], epochs: int) -> dict[str, float]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("LSTM is optional. Install with: pip install -e '.[lstm]'") from exc
    sequences, ratios = [], []
    for _, group in df[df["month"] <= origin].groupby("item_id", sort=True):
        values = group["quantity"].to_numpy(dtype=float)
        for t in range(LOOKBACK - 1, len(values) - horizon):
            recent = values[t - LOOKBACK + 1:t + 1]
            scale = max(float(recent.mean()), 1.0)
            sequences.append(recent / scale)
            ratios.append(values[t + horizon] / scale)
    if len(sequences) < 24:
        raise ValueError("Not enough historical training examples for LSTM at this origin/horizon")
    tf.keras.utils.set_random_seed(42)
    X = np.asarray(sequences, dtype="float32")[..., np.newaxis]
    y = np.asarray(ratios, dtype="float32")
    network = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, 1)),
        tf.keras.layers.LSTM(24),
        tf.keras.layers.Dense(1),
    ])
    network.compile(optimizer="adam", loss="mse")
    network.fit(X, y, epochs=epochs, batch_size=min(32, len(X)), verbose=0, shuffle=False)
    recent, scales = [], []
    for item in items:
        group = df[(df["item_id"] == item) & (df["month"] <= origin)]
        values = group["quantity"].to_numpy(dtype=float)[-LOOKBACK:]
        scale = max(float(values.mean()), 1.0)
        recent.append(values / scale)
        scales.append(scale)
    predictions = network.predict(np.asarray(recent, dtype="float32")[..., np.newaxis], verbose=0).ravel()
    # Release the graph between rolling-origin fits.
    tf.keras.backend.clear_session()
    return dict(zip(items, np.maximum(0, predictions * np.asarray(scales)).astype(float)))


def forecast_at_origin(df: pd.DataFrame, origin: object, horizon: int = 1,
                       models: tuple[str, ...] = ("naive", "seasonal_naive", "random_forest"),
                       lstm_epochs: int = 15) -> pd.DataFrame:
    """Fit only on history known at origin and predict origin + horizon months."""
    origin = _month(origin)
    if horizon < 1 or horizon > 12:
        raise ValueError("horizon must be 1 through 12 months")
    allowed = {"naive", "seasonal_naive", "random_forest", "lstm"}
    if not models or set(models) - allowed:
        raise ValueError(f"models must be selected from {sorted(allowed)}")
    available = df[df["month"] <= origin]
    if available.empty or origin not in set(available["month"]):
        raise ValueError("origin must be an observed history month")
    groups = {}
    for item, group in available.groupby("item_id", sort=True):
        if group["month"].iloc[-1] == origin and len(group) >= LOOKBACK:
            groups[str(item)] = group["quantity"].to_numpy(dtype=float)
    if not groups:
        raise ValueError(f"No item has {LOOKBACK} complete months through the origin")
    items = sorted(groups)
    predictions: dict[str, dict[str, float]] = {}
    if "naive" in models:
        predictions["naive"] = {item: float(groups[item][-1]) for item in items}
    if "seasonal_naive" in models:
        predictions["seasonal_naive"] = {item: float(groups[item][-12 + ((horizon - 1) % 12)]) for item in items}
    if "random_forest" in models:
        predictions["random_forest"] = _predict_rf(available, origin, horizon, items)
    if "lstm" in models:
        predictions["lstm"] = _predict_lstm(available, origin, horizon, items, lstm_epochs)
    target = origin + pd.DateOffset(months=horizon)
    return pd.DataFrame([
        (origin, target, item, model, value)
        for model, values in predictions.items() for item, value in values.items()
    ], columns=PREDICTION_COLUMNS)


def backtest(df: pd.DataFrame, *, horizon: int = 1, origins: int = 6,
             models: tuple[str, ...] = ("naive", "seasonal_naive", "random_forest"),
             lstm_epochs: int = 15) -> pd.DataFrame:
    """Rolling-origin evaluation; each model sees only data available at that origin."""
    dates = sorted(df["month"].unique())
    if origins < 1 or len(dates) < LOOKBACK + horizon + origins:
        raise ValueError("Insufficient months for requested lookback, horizon, and origin count")
    selected = dates[-(origins + horizon):-horizon]
    results = []
    actuals = df.rename(columns={"month": "target_month", "quantity": "actual_qty"})
    for origin in selected:
        forecast = forecast_at_origin(df, origin, horizon, models, lstm_epochs)
        forecast = forecast.merge(actuals, on=["target_month", "item_id"], how="inner", validate="many_to_one")
        results.append(forecast)
    if not results:
        raise ValueError("No comparable forecasts and actuals found")
    return pd.concat(results, ignore_index=True)


def evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report sample size, +/-25% hit rate, WAPE, and bias by model."""
    required = {"model", "forecast_qty", "actual_qty"}
    if required - set(predictions.columns):
        raise ValueError(f"Predictions need columns: {sorted(required)}")
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        actual = pd.to_numeric(group["actual_qty"], errors="raise").to_numpy(dtype=float)
        forecast = pd.to_numeric(group["forecast_qty"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
            raise ValueError("Evaluation values must be finite")
        hit = np.abs(forecast - actual) <= 0.25 * np.abs(actual) + 1e-9
        denominator = np.abs(actual).sum()
        rows.append((model, len(group), float(hit.mean()),
                     float(np.abs(forecast - actual).sum() / denominator) if denominator else np.nan,
                     float((forecast - actual).sum() / denominator) if denominator else np.nan))
    return pd.DataFrame(rows, columns=["model", "n", "hit_rate_25", "wape", "bias_ratio"])


def add_benchmark(backtested: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Compare an independently saved benchmark on exactly matched item/origin/target rows."""
    return add_benchmark_frame(backtested, pd.read_csv(path, dtype={"item_id": "string"}))


def add_benchmark_frame(backtested: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """In-memory version for a user-supplied CSV in the local app."""
    benchmark = benchmark.copy()
    keys = ["origin_month", "target_month", "item_id"]
    if set(keys + ["forecast_qty"]) - set(benchmark.columns):
        raise ValueError("Benchmark needs origin_month,target_month,item_id,forecast_qty")
    for col in keys[:2]:
        benchmark[col] = pd.to_datetime(benchmark[col], errors="raise")
        if not benchmark[col].eq(benchmark[col].dt.to_period("M").dt.to_timestamp()).all():
            raise ValueError(f"Benchmark {col} must be the first day of a month")
    benchmark["item_id"] = benchmark["item_id"].astype("string")
    benchmark["forecast_qty"] = pd.to_numeric(benchmark["forecast_qty"], errors="raise")
    if benchmark["forecast_qty"].isna().any() or not np.isfinite(benchmark["forecast_qty"].astype(float)).all() or (benchmark["forecast_qty"] < 0).any():
        raise ValueError("Benchmark forecasts must be finite, nonnegative quantities")
    if benchmark.duplicated(keys).any():
        raise ValueError("Benchmark has duplicate item/origin/target rows")
    benchmark = benchmark[keys + ["forecast_qty"]].copy()
    benchmark["model"] = "external_benchmark"
    actuals = backtested[keys + ["actual_qty"]].drop_duplicates(keys)
    benchmark = benchmark.merge(actuals, on=keys, how="inner", validate="one_to_one")
    comparable = backtested.merge(benchmark[keys], on=keys, how="inner", validate="many_to_one")
    if benchmark.empty:
        raise ValueError("Benchmark has no rows matching this backtest")
    return pd.concat([comparable, benchmark], ignore_index=True)
