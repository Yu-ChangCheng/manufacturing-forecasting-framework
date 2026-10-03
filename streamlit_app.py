"""Local, guided CSV workflow for people who do not use Python day to day."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from mfg_forecast.core import (
    LOOKBACK,
    add_benchmark_frame,
    backtest,
    evaluate,
    forecast_at_origin,
    load_history,
)
from mfg_forecast.prepare import prepare_monthly_data


ROOT = Path(__file__).resolve().parent
PUBLIC_SAMPLE = ROOT / "examples" / "hf_car_parts_sample.csv"
RF_DEMO = ROOT / "examples" / "synthetic_rf_demo.csv"
MODEL_NAMES = {
    "naive": "Last month",
    "seasonal_naive": "Same month last year",
    "random_forest": "Random Forest",
    "external_benchmark": "Your saved forecast",
}


def _guess(columns: list[str], names: tuple[str, ...]) -> int:
    lowered = [column.lower().strip() for column in columns]
    for name in names:
        if name in lowered:
            return lowered.index(name)
    return 0


def _read_table(data: bytes, name: str) -> pd.DataFrame:
    try:
        if name.lower().endswith(".xlsx"):
            return pd.read_excel(BytesIO(data), sheet_name=0, dtype="string", engine="openpyxl")
        return pd.read_csv(BytesIO(data), dtype="string")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, OSError) as exc:
        raise ValueError("Could not read this file. Use a UTF-8 CSV or an ordinary .xlsx workbook and try again") from exc


def _history() -> tuple[pd.DataFrame, str, str]:
    source = st.radio(
        "Choose a starting point",
        ["See the Random Forest teaching example", "Explore public car-parts example", "Use my own file"],
        horizontal=True,
    )
    if source.startswith("See the Random Forest"):
        st.info(
            "Fictional demand for six items over eight years. Each item follows a repeating seasonal pattern, "
            "plus random month-to-month noise. This gives a clear example of a pattern Random Forest can learn. "
            "It does not establish performance on real business data."
        )
        with st.expander("How this example was made"):
            st.write(
                "For each item and month, the generator multiplies an item-specific base quantity by a fixed "
                "month-of-year factor, then adds seeded random noise. Random Forest can use calendar month and "
                "past quantities to learn the repeating pattern. The full recipe and backtest are in "
                "docs/random-forest-demo.md in this repository."
            )
        return load_history(RF_DEMO), "Reproducible fictional seasonal-demand example", "Fixed seasonal pattern plus random noise; not observed sales."
    if source.startswith("Explore"):
        st.info(
            "This is a five-item sample from the Monash car-parts monthly sales data on Hugging Face. "
            "The source dataset replaced missing observations with zeros, so treat the scores as a learning example."
        )
        st.markdown(
            "[Hugging Face source](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/blob/main/forecasting/monash/car_parts_dataset_without_missing_values.tsf)"
            " · [Original Monash dataset](https://huggingface.co/datasets/Monash-University/monash_tsf)"
        )
        return load_history(PUBLIC_SAMPLE), "Monash car-parts teaching sample", "Original missing observations were replaced with zero by the source dataset."

    st.caption("Run this app on your own computer for private data. The app does not send your file to Hugging Face or another service.")
    uploaded = st.file_uploader("Choose a CSV or Excel file with dates and quantities", type=["csv", "xlsx"])
    if uploaded is None:
        st.info("Start with a file containing a date (or year and month), an item ID if needed, and a quantity. For Excel, the app reads the first sheet.")
        st.stop()
    if uploaded.size > 20 * 1024 * 1024:
        st.error("This beginner interface accepts files up to 20 MB. Narrow the file or use the command line for larger data.")
        st.stop()
    try:
        raw = _read_table(uploaded.getvalue(), uploaded.name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if raw.empty:
        st.error("This CSV has no rows.")
        st.stop()
    st.write("**First rows of your file**")
    st.dataframe(raw.head(8), hide_index=True)
    columns = list(raw.columns)
    date_layout = st.radio("Where is the date?", ["One date column", "Separate year and month columns"], horizontal=True)
    if date_layout == "One date column":
        date_col = st.selectbox("Date column", columns, index=_guess(columns, ("date", "month", "invoice_date", "invoicedate")))
        year_col = month_col = None
    else:
        date_col = None
        year_col = st.selectbox("Year column", columns, index=_guess(columns, ("year",)))
        month_col = st.selectbox("Month number column", columns, index=_guess(columns, ("month", "month_num")))
    item_choice = st.selectbox("What identifies each item?", ["All rows are one total"] + columns,
                               index=columns.index("item_id") + 1 if "item_id" in columns else 0)
    item_col = None if item_choice == "All rows are one total" else item_choice
    quantity_col = st.selectbox("Which column is the quantity sold or demanded?", columns,
                                index=_guess(columns, ("quantity", "qty", "sales", "units")))
    source_shape_label = st.radio(
        "What does each row represent?",
        ["One item in one month", "A transaction or daily record (sum rows within each month)"],
    )
    fill_missing = st.checkbox(
        "I confirm missing months mean zero demand; fill them with 0",
        value=False,
        help="Leave this unchecked if a missing month may mean missing records, a stockout, or an item that was not yet offered.",
    )
    try:
        prepared, summary = prepare_monthly_data(
            raw,
            quantity_col=quantity_col,
            date_col=date_col,
            year_col=year_col,
            month_col=month_col,
            item_col=item_col,
            source_shape="monthly" if source_shape_label.startswith("One") else "transactions",
            fill_missing_with_zero=fill_missing,
        )
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
        st.stop()
    st.success(
        f"Prepared {summary['monthly_rows']:,} monthly rows from {summary['source_rows']:,} source rows. "
        f"Added {summary['zero_filled_months']:,} explicitly approved zero months."
    )
    st.dataframe(prepared.head(8), hide_index=True)
    st.download_button("Download prepared monthly CSV", prepared.to_csv(index=False),
                       file_name="prepared_monthly_history.csv", mime="text/csv")
    note = f"User CSV: {summary['source_rows']} rows; {summary['zero_filled_months']} missing months filled with zero."
    return prepared, uploaded.name, note


def _chart(predictions: pd.DataFrame, item: str) -> None:
    subset = predictions[predictions["item_id"].astype(str) == item]
    actual = subset.drop_duplicates("target_month").set_index("target_month")["actual_qty"].rename("Actual")
    model_lines = subset.assign(model=subset["model"].map(MODEL_NAMES)).pivot(
        index="target_month", columns="model", values="forecast_qty"
    )
    chart = actual.to_frame().join(model_lines).sort_index()
    st.line_chart(chart, x_label="Month being forecast", y_label="Quantity")


def _report(source: str, note: str, history: pd.DataFrame, items: list[str], horizon: int,
            origins: int, metrics: pd.DataFrame) -> str:
    lines = [
        "Monthly demand forecasting: retrospective teaching report",
        f"Data source: {source}",
        f"Data note: {note}",
        f"Selected items ({len(items)}): {', '.join(items)}",
        f"Observed months: {history['month'].min():%Y-%m} through {history['month'].max():%Y-%m}",
        f"Forecast horizon: {horizon} month(s); backtest origins: {origins}",
        "Each method was scored on the same observed item-month rows when an external forecast was provided.",
        "Hit: absolute error is at most 25% of actual quantity; when actual is zero, only a zero forecast hits.",
        "WAPE: total absolute error divided by total actual quantity; lower is better.",
        "These are retrospective scores and do not establish future performance.",
        "",
        "Method | item-month forecasts | hits within 25% | WAPE | signed bias",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"{MODEL_NAMES.get(row.model, row.model)} | {row.n} | {row.hit_rate_25:.1%} | "
            f"{row.wape:.1%} | {row.bias_ratio:.1%}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    st.set_page_config(page_title="Monthly Demand Forecasting", page_icon="📈", layout="wide")
    st.title("Understand your monthly demand forecast")
    st.write("Start with a fictional teaching example, explore public data, or bring your own CSV. Check the data, compare forecasts against past months, then download a future forecast.")
    st.image(str(ROOT / "docs" / "workflow.svg"), caption="The same steps apply to your data. Backtesting comes before a future forecast.")

    st.header("1. Prepare the data")
    history, source, note = _history()
    st.caption(
        f"{history['item_id'].nunique()} items · {history['month'].min():%b %Y} to "
        f"{history['month'].max():%b %Y} · {len(history):,} monthly rows"
    )
    totals = history.groupby("item_id")["quantity"].sum().sort_values(ascending=False)
    all_items = list(totals.index.astype(str))
    default_items = all_items if len(all_items) <= 10 else all_items[:10]
    chosen = st.multiselect("Items to analyze", options=all_items, default=default_items,
                            help="For a large file, begin with the ten highest-volume items. Results describe selected items only.")
    if not chosen:
        st.info("Select at least one item to continue.")
        st.stop()
    if len(chosen) < len(all_items):
        st.warning(f"This run covers {len(chosen)} of {len(all_items)} items. The reported scores are for this selection only.")
    chosen_history = history[history["item_id"].astype(str).isin(chosen)].copy()
    spans = chosen_history.groupby("item_id")["month"].agg(["min", "max", "count"])
    if spans["max"].nunique() != 1:
        st.error("Selected items end in different months. Choose items with the same last observed month for this simple comparison.")
        st.stop()

    st.header("2. Test against months we already know")
    st.write("A backtest acts as if earlier months were still in the future. Each method sees only the data available at that time.")
    horizon = int(st.selectbox("How far ahead should each forecast look?", [1, 2, 3, 6, 12], index=0,
                               format_func=lambda value: f"{value} month{'s' if value != 1 else ''}"))
    min_months = int(spans["count"].min())
    max_origins = min(12, min_months - LOOKBACK - horizon)
    if max_origins < 1:
        st.error(f"These items need at least {LOOKBACK + horizon + 1} consecutive months for a backtest at this horizon. Choose more history or a shorter horizon.")
        st.stop()
    origins = int(st.slider("How many past forecast dates should we test?", 1, max_origins, min(6, max_origins)))
    use_rf = st.checkbox("Include Random Forest", value=True,
                         help="Compare a machine-learning model with the two simple reference forecasts.")
    models = ("naive", "seasonal_naive", "random_forest") if use_rf else ("naive", "seasonal_naive")
    with st.expander("Optional: compare a forecast saved by your team"):
        st.write("Only upload forecasts recorded before their actual quantities were known. Dates must be the first of the month.")
        st.download_button(
            "Download comparison template",
            "origin_month,target_month,item_id,forecast_qty\n2025-01-01,2025-02-01,ITEM-A,112\n",
            file_name="saved_forecast_template.csv", mime="text/csv",
        )
        benchmark_upload = st.file_uploader("Saved forecast CSV", type="csv", key="benchmark")
    benchmark_bytes = benchmark_upload.getvalue() if benchmark_upload else b""
    fingerprint = sha256(
        chosen_history.to_csv(index=False).encode("utf-8")
        + repr((horizon, origins, models)).encode("utf-8") + benchmark_bytes
    ).hexdigest()
    if st.button("Run the backtest and make a future forecast", type="primary"):
        try:
            with st.spinner("Testing earlier months and preparing the next forecast..."):
                predictions = backtest(chosen_history, horizon=horizon, origins=origins, models=models)
                if benchmark_bytes:
                    predictions = add_benchmark_frame(predictions, _read_table(benchmark_bytes, benchmark_upload.name))
                metrics = evaluate(predictions)
                future = forecast_at_origin(chosen_history, chosen_history["month"].max(), horizon, models)
                st.session_state["analysis"] = (fingerprint, predictions, metrics, future)
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
            st.stop()

    saved = st.session_state.get("analysis")
    if not saved or saved[0] != fingerprint:
        st.info("Choose your settings, then run the backtest. If you change the data or settings, run it again.")
        st.stop()
    _, predictions, metrics, future = saved

    st.header("3. Read the result")
    display = metrics.assign(
        Method=metrics["model"].map(MODEL_NAMES),
        **{"Forecasts tested": metrics["n"], "Hits within ±25%": metrics["hit_rate_25"].map(lambda x: f"{x:.1%}"),
           "WAPE (lower is better)": metrics["wape"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "n/a"),
           "Bias": metrics["bias_ratio"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "n/a")},
    )[["Method", "Forecasts tested", "Hits within ±25%", "WAPE (lower is better)", "Bias"]]
    st.dataframe(display, hide_index=True)
    st.caption("One forecast = one selected item at one target month. A hit means error ≤25% of actual; if actual is 0, only forecast 0 is a hit. WAPE weights larger quantities more heavily.")
    if metrics["wape"].notna().any():
        low_wape = metrics.loc[metrics["wape"].idxmin()]
        high_hit = metrics.loc[metrics["hit_rate_25"].idxmax()]
        st.info(
            f"In these past months, {MODEL_NAMES[low_wape['model']]} had the lowest WAPE "
            f"({low_wape['wape']:.1%}); {MODEL_NAMES[high_hit['model']]} had the highest ±25% hit rate "
            f"({high_hit['hit_rate_25']:.1%}). These measures can favor different methods."
        )
    chart_item = st.selectbox("View one item's past forecasts", chosen)
    _chart(predictions, chart_item)
    st.write("**Forecast for the selected future month**")
    future_view = future.copy()
    future_view["model"] = future_view["model"].map(MODEL_NAMES)
    for column in ("origin_month", "target_month"):
        future_view[column] = future_view[column].dt.strftime("%Y-%m")
    future_view["forecast_qty"] = future_view["forecast_qty"].round(1)
    future_view = future_view.rename(columns={
        "origin_month": "Last known month", "target_month": "Month forecasted",
        "item_id": "Item", "model": "Method", "forecast_qty": "Estimated quantity",
    })
    st.dataframe(future_view, hide_index=True)
    st.caption("A future forecast is a planning estimate, not a guarantee. Review promotions, stockouts, launches, and unusual orders before acting on it.")

    st.header("4. Keep your results")
    left, middle, right = st.columns(3)
    left.download_button("Download past predictions", predictions.to_csv(index=False),
                         file_name="backtest_predictions.csv", mime="text/csv")
    middle.download_button("Download future forecast", future.to_csv(index=False),
                           file_name="future_forecast.csv", mime="text/csv")
    right.download_button("Download plain-language report",
                          _report(source, note, chosen_history, chosen, horizon, origins, metrics),
                          file_name="forecast_report.txt", mime="text/plain")
    st.caption("Uploaded data and results stay in this running local app session unless you download them. For confidential data, keep the app on a trusted computer.")


if __name__ == "__main__":
    main()
