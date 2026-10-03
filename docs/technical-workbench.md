# Technical workbench: from a CSV to a tested forest

The editable [monthly demand workbench notebook](../notebooks/monthly_demand_workbench.ipynb) provides a complete local experiment. It calls functions in [`mfg_forecast.workbench`](../src/mfg_forecast/workbench.py), so the data rules and model code are reusable outside Jupyter.

## Run it

From the repository root, create a Python 3.11+ environment and install the notebook dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe -m jupyter lab
```

Open `notebooks/monthly_demand_workbench.ipynb` in JupyterLab and choose **Run All**. The default file is the invented seasonal-demand example, so the first run needs no private data. JupyterLab serves a local browser page; keep the terminal open while using it. On macOS/Linux, activate `.venv` and use `python -m jupyter lab`.

The install/run pattern follows the [JupyterLab installation guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). The chronological validation follows the principle in scikit-learn's [lagged-feature forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html); this workbench uses a Random Forest and an explicit origin/target cutoff suited to a direct multi-month horizon.

## Substitute your data

Edit the notebook's **Choose a file and settings** cell. Set `DATA_PATH` to your local `.csv` or `.xlsx` path and map the date, item, and actual-quantity columns. One date column or separate year and month columns are supported. Set `ITEM_COLUMN=None` if every row belongs to one aggregate series.

For example, on Windows: `DATA_PATH = Path(r"C:\path\to\my_monthly_sales.xlsx")`. Keep the `r` before the quoted path so backslashes are interpreted literally.

| Input situation | Setting |
| --- | --- |
| Exactly one row per item-month | `ROWS_ARE_TRANSACTIONS=False` |
| Daily or individual transactions that should be summed monthly | `ROWS_ARE_TRANSACTIONS=True` |
| A missing month is verified to mean zero demand | `FILL_MISSING_WITH_ZERO=True` |
| A month is missing because records are unavailable or a product was inactive | Correct/filter the source; keep zero fill off |

The cleaning step rejects invalid dates, negative or nonnumeric quantities, duplicate monthly rows, and unexplained gaps. It prints a cleaning summary and per-item span/zero counts. Do not treat sales during stockouts as complete demand without business review.

## Select features and model settings

`FeatureConfig` controls the features available to the forest:

- `lags=(1, 2, 3, 6, 12, 24)` takes quantities from known months before the target. `lag_1` is the latest quantity available at the forecast origin.
- `rolling_windows=(3, 6, 12)` adds averages of the latest known quantities.
- `include_month` adds the calendar month of the target, useful for recurring seasonality.
- `include_item` one-hot encodes item ID so one forest can train on multiple series.
- `include_year_over_year_change` optionally adds the latest-versus-year-ago relative change.

`HORIZON` selects 1–12 months ahead. `TEST_ORIGINS` reserves the latest forecast dates as the final test. With `AUTO_TUNE=True`, `TUNE_TRIALS` candidates are sampled from the editable `SEARCH_SPACE` grid of tree count, depth, leaf size, and feature sampling. The default search uses three earlier rolling folds of three origins each and minimizes pooled validation WAPE. `AUTO_TUNE=False` uses the editable `FOREST_PARAMS` dictionary. Larger searches take longer and are not guaranteed to improve future accuracy. A single item with only 3–4 years of history may need fewer test origins/folds or no tuning; several years of earlier training data are preferable.

The code fits one model on the training portion, scores it once on the later holdout, and then fits a final model on all observed history for a future estimate. The model is a scikit-learn `Pipeline`, with one-hot item preprocessing and `RandomForestRegressor`. The saved `.joblib` bundle includes the fitted pipeline, feature configuration, horizon, and latest training month. `load_forecaster` reopens that bundle for a later local prediction. Load it only from a trusted source and keep compatible package versions.

## Time boundaries

For each row, **origin** is the last month whose actual quantity could be used as an input; **target** is the later month being predicted. The test uses the last `TEST_ORIGINS` origin months. Training labels are limited to targets on or before the **first test origin**. For a three-month horizon, this prevents the model from learning a December outcome before making an October forecast for January.

Auto tuning uses only the earlier training portion. Each validation fold fits with labels available by its first validation origin. Later validation origins may use actual quantities that have since become known as lag features, reflecting a rolling operational setting. The final test actuals are not used for feature selection or tuning. If you change the configuration after inspecting the final test score, collect a new later test period before presenting it as an unbiased result.

The notebook compares Random Forest on exactly the same test item-month rows as **last month** and **same month last year**. It reports the ±25% hit rate, WAPE, and signed bias, plus a chart of one item and forest impurity importance. Importance can split credit between correlated features and should not be described as a causal effect.

## Saved local outputs

The notebook writes these files under Git-ignored `outputs/`:

```text
workbench_cleaned_monthly.csv
workbench_test_predictions.csv
workbench_test_metrics.csv
workbench_future_forecast.csv
workbench_model.joblib
```

The `.ipynb` file in Git contains code and explanatory text only, with no run outputs. Review every file before publishing changes. The notebook generator is [`scripts/build_workbench_notebook.py`](../scripts/build_workbench_notebook.py); rerun it only when you intend to replace notebook edits with the maintained version.

## Use the API without Jupyter

```python
from mfg_forecast.core import load_history
from mfg_forecast.workbench import (
    FeatureConfig, build_supervised, split_time, tune_forest,
    fit_forest, score_holdout,
)

history = load_history("path/to/monthly_history.csv")
features = FeatureConfig(lags=(1, 3, 12, 24), rolling_windows=(3, 12))
rows = build_supervised(history, horizon=1, features=features)
split = split_time(rows, test_origins=12)
search = tune_forest(split.train, features, trials=8)
model = fit_forest(split.train, features, dict(search.iloc[0]["params"]))
predictions, metrics = score_holdout(model, split.test, features)
print(metrics)
```

The framework does not currently handle future-known promotions, prices, stockout corrections, item hierarchy reconciliation, calibrated uncertainty intervals, or automatic deployment. Add a driver only if its value would truly have been known at each forecast origin, then recheck the time boundary and baselines.
