"""Regenerate the editable tutorial notebook without saved private outputs."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip(), id=f"workbench-{len(cells) + 1:02d}"))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip(), id=f"workbench-{len(cells) + 1:02d}"))


md("""
# Monthly demand forecasting workbench

Start with **Run All** on the fictional data. The notebook cleans monthly demand, tunes a Random Forest on earlier months, and compares it with two simple forecasts on a later test period. Then change `DATA_PATH` and the column names to use your local CSV or Excel file.

To add the Transformer to the same test, install `python -m pip install -e ".[notebook,transformer]"` from the project root and set `RUN_TRANSFORMER = True` below. The forest and Transformer use the same training cutoff and scored item-month rows.

**Keep confidential files outside this public repository.** The notebook does not upload data. Check your organization's rules before using any business records. Do not claim the fictional demonstration scores as results from your own data.
""")

code("""
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent  # Jupyter may start inside notebooks/
if not (ROOT / "pyproject.toml").exists():
    raise RuntimeError("Start Jupyter from the repository root or notebooks/ folder")
sys.path.insert(0, str(ROOT / "src"))

from mfg_forecast.core import evaluate
from mfg_forecast.prepare import prepare_monthly_data
from mfg_forecast.transformer import TransformerConfig, fit_transformer
from mfg_forecast.workbench import (
    FeatureConfig, DEFAULT_RF_PARAMS, build_supervised, split_time,
    tune_forest, fit_forest, score_holdout, feature_importance,
    plot_diagnostics, forecast_future, save_forecaster, load_forecaster,
)
""")

md("""
## 1 · Choose a file and settings

For your own file, change `DATA_PATH` and map its actual column names. Use a date column **or** separate year and month columns. Set `ROWS_ARE_TRANSACTIONS=True` only when multiple rows in a month should be summed. Missing months are rejected unless you explicitly confirm that they represent true zero demand. Choose at least one feature. Lags and rolling windows use only quantities already known when the forecast is made.
""")

code("""
DATA_PATH = ROOT / "examples" / "synthetic_rf_demo.csv"  # replace with your own local CSV or .xlsx
# DATA_PATH = ROOT / "examples" / "hf_car_parts_sample.csv"
DATE_COLUMN = "month"          # set to None if using YEAR_COLUMN + MONTH_COLUMN
YEAR_COLUMN = None
MONTH_COLUMN = None
ITEM_COLUMN = "item_id"        # set to None for one combined total
QUANTITY_COLUMN = "quantity"
ROWS_ARE_TRANSACTIONS = False  # True: sum rows within each item-month
FILL_MISSING_WITH_ZERO = False # True ONLY for verified zero-demand months

HORIZON = 1                    # predict 1–12 months ahead
TEST_ORIGINS = 12              # last 12 forecast dates, untouched during tuning
FEATURES = FeatureConfig(
    lags=(1, 2, 3, 6, 12, 24),
    rolling_windows=(3, 6, 12),
    include_month=True,
    include_item=True,
    include_year_over_year_change=False,
)
AUTO_TUNE = True
TUNE_TRIALS = 8                # increase for a wider search, at more compute cost
TUNE_FOLDS = 3
TUNE_MONTHS_PER_FOLD = 3
SEARCH_SPACE = {
    "n_estimators": [100, 160, 240],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [0.7, 1.0],
}
FOREST_PARAMS = dict(DEFAULT_RF_PARAMS)  # used when AUTO_TUNE=False

RUN_TRANSFORMER = False        # install the transformer extra before enabling
TRANSFORMER = TransformerConfig(epochs=15, width=32, heads=4)
PLOT_ITEM = None               # None picks the first scored item, or set an ID like "DEMO-04"
""")

md("""
## 2 · Load and clean

The cleaner converts dates to calendar months, checks quantities and item IDs, detects duplicates, and checks for gaps. It never silently converts an unknown month to zero. Inspect the preview and counts before training.
""")

code("""
if DATA_PATH.suffix.lower() == ".csv":
    raw = pd.read_csv(DATA_PATH, dtype="string")
elif DATA_PATH.suffix.lower() == ".xlsx":
    raw = pd.read_excel(DATA_PATH, sheet_name=0, dtype="string")
else:
    raise ValueError("Use a .csv or .xlsx file")

monthly, cleaning = prepare_monthly_data(
    raw,
    quantity_col=QUANTITY_COLUMN,
    date_col=DATE_COLUMN,
    year_col=YEAR_COLUMN,
    month_col=MONTH_COLUMN,
    item_col=ITEM_COLUMN,
    source_shape="transactions" if ROWS_ARE_TRANSACTIONS else "monthly",
    fill_missing_with_zero=FILL_MISSING_WITH_ZERO,
)
print("Cleaning summary:", cleaning)
display(monthly.head(10))
""")

code("""
quality = monthly.groupby("item_id").agg(
    first_month=("month", "min"), last_month=("month", "max"),
    months=("month", "size"), zero_months=("quantity", lambda q: int((q == 0).sum())),
    total_quantity=("quantity", "sum"),
)
display(quality)
ax = monthly[monthly["item_id"] == monthly["item_id"].iloc[0]].plot(
    x="month", y="quantity", figsize=(11, 3), title="First item's observed monthly quantity", legend=False
)
ax.set_ylabel("Quantity")
plt.show()
""")

md("""
## 3 · Build features and reserve later months

One supervised row describes what was known at an **origin month** and the actual quantity at its later **target month**. The last `TEST_ORIGINS` origin months form the final holdout. Training labels must already have been observed by the first test origin. This matters especially when `HORIZON > 1`.
""")

code("""
supervised = build_supervised(monthly, horizon=HORIZON, features=FEATURES)
split = split_time(supervised, test_origins=TEST_ORIGINS)
print("Selected model features:", FEATURES.columns)
print("Training rows:", len(split.train), "· test rows:", len(split.test))
print("First test origin:", split.first_test_origin.date())
print("Latest training label:", split.train["target_month"].max().date())
print("Test targets:", split.test["target_month"].min().date(), "to", split.test["target_month"].max().date())
assert split.train["target_month"].max() <= split.first_test_origin
display(split.test[["origin_month", "target_month", "item_id", "actual_qty"]].head())
first_item = str(monthly["item_id"].iloc[0])
item_history = monthly[monthly["item_id"] == first_item]
ax = item_history.plot(x="month", y="quantity", figsize=(11, 3), legend=False,
                       title=f"Item {first_item}: first test origin")
ax.axvline(split.first_test_origin, color="darkorange", linestyle="--", label="First test origin")
ax.set_ylabel("Quantity")
ax.legend()
plt.show()
""")

md("""
## 4 · Tune on earlier rolling folds, then train

The optional search scores candidate forests on earlier rolling validation folds using WAPE. Each fold trains only on labels available at its first validation origin. **The final holdout is not used to choose parameters.** Set `AUTO_TUNE=False` to use your values in `FOREST_PARAMS`.
""")

code("""
if AUTO_TUNE:
    tuning = tune_forest(
        split.train, FEATURES, trials=TUNE_TRIALS,
        folds=TUNE_FOLDS, months_per_fold=TUNE_MONTHS_PER_FOLD,
        search_space=SEARCH_SPACE,
    )
    display(tuning.drop(columns="params"))
    best_params = dict(tuning.iloc[0]["params"])
else:
    tuning = None
    best_params = dict(FOREST_PARAMS)
print("Chosen forest parameters:", best_params)
model = fit_forest(split.train, FEATURES, best_params)
""")

md("""
## 5 · Compare models on the same test rows

The optional Transformer reads the same number of past months as the forest's lookback. Each month carries scaled demand, its place in the window, and its month-of-year encoding. Self-attention lets the final month draw information from the entire known window. Both fitted models stop training at the first test origin. At later origins the input window moves forward as new actuals become known, while the weights stay fixed.

A ±25% hit means absolute error is at most one-quarter of actual quantity. WAPE divides total absolute error by total actual quantity. Report both scores and the number of forecasts. Choose Transformer settings before looking at the held-out scores.
""")

code("""
predictions, metrics = score_holdout(model, split.test, FEATURES)
if RUN_TRANSFORMER:
    attention_model = fit_transformer(
        monthly, through=split.first_test_origin, horizon=HORIZON,
        lookback=FEATURES.lookback, config=TRANSFORMER,
    )
    transformer_test = split.test[["origin_month", "target_month", "item_id", "actual_qty"]].copy()
    transformer_test["model"] = "transformer"
    transformer_test["forecast_qty"] = attention_model.predict(monthly, transformer_test)
    predictions = pd.concat([predictions, transformer_test], ignore_index=True)
    metrics = evaluate(predictions)

test_keys = ["origin_month", "target_month", "item_id"]
assert not predictions.duplicated([*test_keys, "model"]).any()
assert predictions.groupby(test_keys)["model"].nunique().eq(metrics["model"].nunique()).all()
display(metrics.sort_values("wape").style.format(
    {"hit_rate_25": "{:.1%}", "wape": "{:.1%}", "bias_ratio": "{:+.1%}"}, na_rep="n/a"
))
display(predictions.head(10))
""")

md("""
## 6 · Visualize errors and feature importance

Set `PLOT_ITEM` to inspect another item. Compare its score and chart with the pooled results above. Feature importance describes the fitted forest. Correlated inputs can split credit, and that chart cannot establish cause and effect.
""")

code("""
importance = feature_importance(model)
item_to_plot = str(PLOT_ITEM) if PLOT_ITEM is not None else str(split.test["item_id"].iloc[0])
item_rows = predictions[predictions["item_id"] == item_to_plot]
figure = plot_diagnostics(predictions, metrics, importance, item_id=item_to_plot)
display(figure)
plt.close(figure)
display(evaluate(item_rows).style.format(
    {"hit_rate_25": "{:.1%}", "wape": "{:.1%}", "bias_ratio": "{:+.1%}"}, na_rep="n/a"
))
display(importance.head(15))
""")

md("""
## 7 · Refit for a future forecast and save local artifacts

After recording the untouched test scores, refit on all observed labeled rows and predict the selected future month. If enabled, the Transformer also produces its own forecast CSV. Only load the saved forest `.joblib` file from trusted sources. Files are written into Git-ignored `outputs/`.
""")

code("""
final_model = fit_forest(supervised, FEATURES, best_params)
future = forecast_future(monthly, final_model, FEATURES, horizon=HORIZON)
display(future)

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
monthly.to_csv(output_dir / "workbench_cleaned_monthly.csv", index=False)
predictions.to_csv(output_dir / "workbench_test_predictions.csv", index=False)
metrics.to_csv(output_dir / "workbench_test_metrics.csv", index=False)
future.to_csv(output_dir / "workbench_future_forecast.csv", index=False)
if RUN_TRANSFORMER:
    final_attention_model = fit_transformer(
        monthly, through=monthly["month"].max(), horizon=HORIZON,
        lookback=FEATURES.lookback, config=TRANSFORMER,
    )
    transformer_future = future[["origin_month", "target_month", "item_id"]].copy()
    transformer_future["forecast_qty"] = final_attention_model.predict(monthly, transformer_future)
    transformer_future.to_csv(output_dir / "workbench_transformer_forecast.csv", index=False)
    display(transformer_future)

model_path = save_forecaster(
    output_dir / "workbench_model.joblib", final_model, FEATURES,
    HORIZON, monthly["month"].max(),
)
print("Saved results and model in", output_dir)
loaded_model, loaded_features, loaded_horizon, trained_through = load_forecaster(model_path)
print("Reloaded model trained through", trained_through.date())
display(forecast_future(monthly, loaded_model, loaded_features, horizon=loaded_horizon))
""")

md("""
### Before using a forecast

- Verify that the input is actual demand or sales quantity, that missing months were handled honestly, and that each selected item has enough history.
- Try different **earlier** validation periods and realistic forecast horizons. Do not keep adjusting settings to make the final holdout look good; reserve a new later period if you do.
- Record future forecasts before actual outcomes arrive. This notebook provides point estimates, not calibrated prediction intervals, inventory decisions, or evidence of real-world adoption.
- See [`docs/technical-workbench.md`](../docs/technical-workbench.md) for API details and limitations.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
destination = ROOT / "notebooks" / "monthly_demand_workbench.ipynb"
destination.parent.mkdir(exist_ok=True)
nbf.write(notebook, destination)
print(destination)
