import pandas as pd
import pytest

from mfg_forecast.prepare import prepare_monthly_data


def test_daily_transactions_are_summed_into_months():
    raw = pd.DataFrame({
        "sale_date": ["2024-01-02", "2024-01-18", "2024-02-01"],
        "sku": ["A", "A", "A"],
        "units": [2, 3, 4],
    })
    prepared, summary = prepare_monthly_data(
        raw, date_col="sale_date", item_col="sku", quantity_col="units", source_shape="transactions"
    )
    assert prepared["quantity"].tolist() == [5, 4]
    assert summary["combined_rows"] == 1


def test_monthly_duplicates_are_not_silently_summed():
    raw = pd.DataFrame({"month": ["2024-01", "2024-01"], "item": ["A", "A"], "qty": [2, 3]})
    with pytest.raises(ValueError, match="Multiple rows"):
        prepare_monthly_data(raw, date_col="month", item_col="item", quantity_col="qty")


def test_missing_month_requires_explicit_zero_decision():
    raw = pd.DataFrame({"year": [2024, 2024], "month": [1, 3], "units": [5, 7]})
    settings = {"year_col": "year", "month_col": "month", "quantity_col": "units"}
    with pytest.raises(ValueError, match="missing"):
        prepare_monthly_data(raw, **settings)
    prepared, summary = prepare_monthly_data(raw, fill_missing_with_zero=True, **settings)
    assert prepared["quantity"].tolist() == [5, 0, 7]
    assert summary["zero_filled_months"] == 1


def test_nonnumeric_quantity_does_not_turn_into_zero():
    raw = pd.DataFrame({"month": pd.Series(["2024-01", "2024-02"], dtype="string"),
                        "units": pd.Series(["5", "unknown"], dtype="string")})
    with pytest.raises(ValueError, match="Quantities must be numbers"):
        prepare_monthly_data(raw, date_col="month", quantity_col="units")
