"""Reusable, time-safe Random Forest experiments for monthly item demand."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .core import evaluate, validate_history


@dataclass(frozen=True)
class FeatureConfig:
    """Only features observable at a forecast origin may be selected."""

    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    rolling_windows: tuple[int, ...] = (3, 6, 12)
    include_month: bool = True
    include_item: bool = True
    include_year_over_year_change: bool = False

    def __post_init__(self) -> None:
        for name, values in (("lags", self.lags), ("rolling_windows", self.rolling_windows)):
            if any(type(value) is not int or value < 1 for value in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain distinct positive integers")
        if not self.columns:
            raise ValueError("Select at least one model feature")

    @property
    def lookback(self) -> int:
        # The same-month-last-year baseline also needs 12 past months.
        return max((12, *self.lags, *self.rolling_windows))

    @property
    def columns(self) -> list[str]:
        columns = []
        if self.include_item:
            columns.append("item_id")
        if self.include_month:
            columns.append("target_month_num")
        columns.extend(f"lag_{lag}" for lag in self.lags)
        columns.extend(f"mean_{window}" for window in self.rolling_windows)
        if self.include_year_over_year_change:
            columns.append("year_over_year_change")
        return columns


def _feature_values(item: str, past: np.ndarray, origin: pd.Timestamp,
                    horizon: int, config: FeatureConfig) -> dict[str, object]:
    row: dict[str, object] = {}
    if config.include_item:
        row["item_id"] = item
    if config.include_month:
        row["target_month_num"] = (origin + pd.DateOffset(months=horizon)).month
    row.update({f"lag_{lag}": float(past[-lag]) for lag in config.lags})
    row.update({f"mean_{window}": float(np.mean(past[-window:]))
                for window in config.rolling_windows})
    if config.include_year_over_year_change:
        row["year_over_year_change"] = float((past[-1] - past[-12]) / max(past[-12], 1.0))
    return row


def build_supervised(history: pd.DataFrame, *, horizon: int = 1,
                     features: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Make labeled rows; every feature is calculated from data through its origin."""
    if type(horizon) is not int or not 1 <= horizon <= 12:
        raise ValueError("horizon must be an integer from 1 to 12 months")
    history = validate_history(history)
    rows = []
    for item, group in history.groupby("item_id", sort=True):
        values = group["quantity"].to_numpy(dtype=float)
        dates = group["month"].tolist()
        for anchor in range(features.lookback - 1, len(values) - horizon):
            target = anchor + horizon
            past = values[:anchor + 1]
            row = {
                "origin_month": dates[anchor],
                "target_month": dates[target],
                "item_id": str(item),
                "actual_qty": float(values[target]),
                "last_month_qty": float(past[-1]),
                "same_month_last_year_qty": float(values[target - 12]),
            }
            row.update(_feature_values(str(item), past, dates[anchor], horizon, features))
            rows.append(row)
    if not rows:
        raise ValueError(f"Need more than {features.lookback + horizon} consecutive months for at least one item")
    return pd.DataFrame(rows).sort_values(["origin_month", "item_id"]).reset_index(drop=True)


@dataclass(frozen=True)
class TimeHoldout:
    train: pd.DataFrame
    test: pd.DataFrame
    first_test_origin: pd.Timestamp


def split_time(frame: pd.DataFrame, *, test_origins: int = 12) -> TimeHoldout:
    """Reserve the last forecast origins; train labels existed by the first one."""
    dates = sorted(frame["origin_month"].unique())
    if type(test_origins) is not int or test_origins < 1 or len(dates) <= test_origins:
        raise ValueError("test_origins must leave earlier months for training")
    selected = dates[-test_origins:]
    cutoff = pd.Timestamp(selected[0])
    train = frame[frame["target_month"] <= cutoff].copy()
    test = frame[frame["origin_month"].isin(selected)].copy()
    if len(train) < 10 or test.empty:
        raise ValueError("Not enough labeled rows before the test period; use more history or fewer test origins")
    return TimeHoldout(train=train, test=test, first_test_origin=cutoff)


DEFAULT_RF_PARAMS = {
    "n_estimators": 160,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": 1.0,
}


def fit_forest(train: pd.DataFrame, features: FeatureConfig,
               params: dict | None = None) -> Pipeline:
    """Fit a forest with one-hot item ID and the selected numeric features."""
    params = dict(DEFAULT_RF_PARAMS if params is None else params)
    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", 1)
    numeric = [column for column in features.columns if column != "item_id"]
    transforms = []
    if features.include_item:
        transforms.append(("item", OneHotEncoder(handle_unknown="ignore"), ["item_id"]))
    if numeric:
        transforms.append(("numeric", "passthrough", numeric))
    model = Pipeline([
        ("prepare", ColumnTransformer(transforms, remainder="drop", verbose_feature_names_out=False)),
        ("forest", RandomForestRegressor(**params)),
    ])
    model.fit(train[features.columns], train["actual_qty"])
    return model


def _validation_folds(train: pd.DataFrame, folds: int, months_per_fold: int):
    dates = sorted(train["origin_month"].unique())
    if folds < 2 or months_per_fold < 1 or len(dates) <= folds * months_per_fold:
        raise ValueError("Tuning needs more earlier months; reduce folds or months_per_fold")
    start = len(dates) - folds * months_per_fold
    results = []
    for index in range(folds):
        validation_origins = dates[start + index * months_per_fold:start + (index + 1) * months_per_fold]
        first_origin = pd.Timestamp(validation_origins[0])
        earlier_labels = train[train["target_month"] <= first_origin]
        validation = train[train["origin_month"].isin(validation_origins)]
        if len(earlier_labels) < 10 or validation.empty:
            raise ValueError("Not enough labeled rows for time-safe tuning; add history or use fewer folds")
        results.append((earlier_labels, validation))
    return results


def tune_forest(train: pd.DataFrame, features: FeatureConfig, *, trials: int = 8,
                folds: int = 3, months_per_fold: int = 3,
                search_space: dict | None = None, random_state: int = 42) -> pd.DataFrame:
    """Search earlier rolling folds only, sorted by pooled validation WAPE."""
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    if search_space is None:
        search_space = {
            "n_estimators": [100, 160, 240],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [0.7, 1.0],
        }
    validation_folds = _validation_folds(train, folds, months_per_fold)
    candidates = list(ParameterSampler(search_space, n_iter=trials, random_state=random_state))
    results = []
    for params in candidates:
        actuals, forecasts = [], []
        for earlier, validation in validation_folds:
            model = fit_forest(earlier, features, params)
            forecasts.append(np.maximum(0, model.predict(validation[features.columns])))
            actuals.append(validation["actual_qty"].to_numpy(dtype=float))
        actual = np.concatenate(actuals)
        forecast = np.concatenate(forecasts)
        total_actual = actual.sum()
        wape = float(np.abs(forecast - actual).sum() / total_actual) if total_actual else np.inf
        hit = float(np.mean(np.abs(forecast - actual) <= 0.25 * actual + 1e-9))
        display_params = {**params, "max_depth": "unlimited" if params.get("max_depth") is None else str(params["max_depth"])}
        results.append({**display_params, "params": params.copy(), "cv_wape": wape, "cv_hit_rate_25": hit,
                        "cv_forecasts": len(actual)})
    return pd.DataFrame(results).sort_values("cv_wape").reset_index(drop=True)


def score_holdout(model: Pipeline, test: pd.DataFrame,
                  features: FeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one untouched test period on the same rows as two simple baselines."""
    common = test[["origin_month", "target_month", "item_id", "actual_qty"]].copy()
    methods = {
        "last_month": test["last_month_qty"].to_numpy(dtype=float),
        "same_month_last_year": test["same_month_last_year_qty"].to_numpy(dtype=float),
        "random_forest": np.maximum(0, model.predict(test[features.columns])),
    }
    predictions = pd.concat(
        [common.assign(model=name, forecast_qty=value) for name, value in methods.items()],
        ignore_index=True,
    )
    return predictions, evaluate(predictions)


def feature_importance(model: Pipeline) -> pd.DataFrame:
    """Tree impurity importance is descriptive; correlated features share credit."""
    names = model.named_steps["prepare"].get_feature_names_out()
    importance = model.named_steps["forest"].feature_importances_
    return pd.DataFrame({"feature": names, "importance": importance}).sort_values(
        "importance", ascending=False, ignore_index=True
    )


def forecast_future(history: pd.DataFrame, model: Pipeline, features: FeatureConfig,
                    *, horizon: int = 1) -> pd.DataFrame:
    """Predict one future target for items current through the latest observed month."""
    if type(horizon) is not int or not 1 <= horizon <= 12:
        raise ValueError("horizon must be an integer from 1 to 12 months")
    history = validate_history(history)
    origin = history["month"].max()
    records = []
    for item, group in history.groupby("item_id", sort=True):
        if group["month"].iloc[-1] != origin or len(group) < features.lookback:
            continue
        values = group["quantity"].to_numpy(dtype=float)
        inputs = _feature_values(str(item), values, origin, horizon, features)
        records.append({"origin_month": origin, "target_month": origin + pd.DateOffset(months=horizon),
                        "item_id": str(item), **inputs})
    if not records:
        raise ValueError("No item has enough history through the latest observed month")
    frame = pd.DataFrame(records)
    frame["forecast_qty"] = np.maximum(0, model.predict(frame[features.columns]))
    return frame[["origin_month", "target_month", "item_id", "forecast_qty"]]


def save_forecaster(path: str | Path, model: Pipeline, features: FeatureConfig,
                    horizon: int, trained_through: object) -> Path:
    """Persist a local trusted model; load joblib files only from trusted sources."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": asdict(features), "horizon": horizon,
                 "trained_through": str(pd.Timestamp(trained_through).date())}, path)
    return path


def load_forecaster(path: str | Path) -> tuple[Pipeline, FeatureConfig, int, pd.Timestamp]:
    """Load only a trusted locally saved bundle with compatible package versions."""
    bundle = joblib.load(path)
    required = {"model", "features", "horizon", "trained_through"}
    if not isinstance(bundle, dict) or required - bundle.keys():
        raise ValueError("This is not a complete forecasting-model bundle")
    return (bundle["model"], FeatureConfig(**bundle["features"]),
            int(bundle["horizon"]), pd.Timestamp(bundle["trained_through"]))


def plot_diagnostics(predictions: pd.DataFrame, metrics: pd.DataFrame,
                     importances: pd.DataFrame, *, item_id: str | None = None):
    """Show model comparison, an item-level trace, and top feature importance."""
    if item_id is None:
        item_id = str(predictions["item_id"].iloc[0])
    subset = predictions[predictions["item_id"].astype(str) == str(item_id)]
    if subset.empty:
        raise ValueError(f"No test predictions for item {item_id!r}")
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ordered = metrics.sort_values("wape")
    axes[0].barh(ordered["model"], ordered["wape"] * 100, color="#3486a3")
    axes[0].set(xlabel="WAPE (%) · lower is better", title="Held-out test")
    actual = subset.drop_duplicates("target_month").sort_values("target_month")
    axes[1].plot(actual["target_month"], actual["actual_qty"], color="black", marker="o", label="Actual")
    for name, group in subset.groupby("model"):
        group = group.sort_values("target_month")
        axes[1].plot(group["target_month"], group["forecast_qty"], marker=".", label=name)
    axes[1].set(title=f"Item {item_id}", xlabel="Target month", ylabel="Quantity")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].legend(fontsize=8)
    top = importances.head(10).iloc[::-1]
    axes[2].barh(top["feature"], top["importance"], color="#69a869")
    axes[2].set(title="Forest feature importance", xlabel="Impurity importance")
    figure.tight_layout()
    return figure
