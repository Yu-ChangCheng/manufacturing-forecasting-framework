"""Command-line entry point for the public forecasting example."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .core import add_benchmark, backtest, evaluate, forecast_at_origin, load_history, make_synthetic_history


def _write(frame: pd.DataFrame, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(frame)} rows to {destination}")


def _models(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mfg-forecast",
        description="Transparent monthly demand forecasting with baselines, Random Forest, and optional LSTM.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample-data", help="Create fictional monthly demand data")
    sample.add_argument("--output", default="examples/synthetic_demand.csv")
    sample.add_argument("--items", type=int, default=6)
    sample.add_argument("--months", type=int, default=72)

    test = commands.add_parser("backtest", help="Run rolling-origin forecasts against later actuals")
    test.add_argument("--data", required=True)
    test.add_argument("--output", default="outputs/backtest_predictions.csv")
    test.add_argument("--metrics-output", default="outputs/backtest_metrics.csv")
    test.add_argument("--horizon", type=int, default=1)
    test.add_argument("--origins", type=int, default=6)
    test.add_argument("--models", type=_models, default=("naive", "seasonal_naive", "random_forest"))
    test.add_argument("--benchmark", help="Optional independently saved comparison forecasts CSV")
    test.add_argument("--lstm-epochs", type=int, default=15)

    forecast = commands.add_parser("forecast", help="Predict a future month from available history")
    forecast.add_argument("--data", required=True)
    forecast.add_argument("--origin", help="Last known month, YYYY-MM-01; defaults to latest")
    forecast.add_argument("--horizon", type=int, default=1)
    forecast.add_argument("--models", type=_models, default=("naive", "seasonal_naive", "random_forest"))
    forecast.add_argument("--lstm-epochs", type=int, default=15)
    forecast.add_argument("--output", default="outputs/future_forecast.csv")

    score = commands.add_parser("evaluate", help="Score an existing CSV with model,forecast_qty,actual_qty")
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", default="outputs/metrics.csv")

    chart = commands.add_parser("plot", help="Plot a backtest for one item")
    chart.add_argument("--predictions", required=True)
    chart.add_argument("--item", required=True)
    chart.add_argument("--output", default="outputs/backtest_chart.png")

    args = parser.parse_args(argv)
    if args.command == "sample-data":
        make_synthetic_history(args.output, items=args.items, months=args.months)
        print(f"Wrote fictional example data to {args.output}")
    elif args.command == "backtest":
        history = load_history(args.data)
        predictions = backtest(history, horizon=args.horizon, origins=args.origins,
                               models=args.models, lstm_epochs=args.lstm_epochs)
        if args.benchmark:
            predictions = add_benchmark(predictions, args.benchmark)
        _write(predictions, args.output)
        metrics = evaluate(predictions)
        _write(metrics, args.metrics_output)
        print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    elif args.command == "forecast":
        history = load_history(args.data)
        origin = args.origin or history["month"].max()
        predictions = forecast_at_origin(history, origin, args.horizon, args.models, args.lstm_epochs)
        _write(predictions, args.output)
        print(predictions.to_string(index=False))
    elif args.command == "evaluate":
        metrics = evaluate(pd.read_csv(args.predictions))
        _write(metrics, args.output)
        print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    elif args.command == "plot":
        predictions = pd.read_csv(args.predictions)
        subset = predictions[predictions["item_id"].astype(str) == args.item].copy()
        if subset.empty:
            raise ValueError(f"No backtest rows for item {args.item!r}")
        subset["target_month"] = pd.to_datetime(subset["target_month"])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        actual = subset.drop_duplicates("target_month").sort_values("target_month")
        ax.plot(actual["target_month"], actual["actual_qty"], marker="o", color="black", label="Actual")
        for model, group in subset.groupby("model"):
            group = group.sort_values("target_month")
            ax.plot(group["target_month"], group["forecast_qty"], marker=".", label=model)
        ax.set(title=f"Rolling-origin forecast: {args.item}", xlabel="Target month", ylabel="Quantity")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160)
        plt.close(fig)
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()

