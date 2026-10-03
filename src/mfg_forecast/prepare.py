"""Turn a familiar sales CSV into the framework's exact monthly format."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import HISTORY_COLUMNS, validate_history


def prepare_monthly_data(
    raw: pd.DataFrame,
    *,
    quantity_col: str,
    date_col: str | None = None,
    year_col: str | None = None,
    month_col: str | None = None,
    item_col: str | None = None,
    source_shape: str = "monthly",
    fill_missing_with_zero: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map columns and optionally sum transaction rows or explicitly fill zero months.

    Returns validated monthly data and counts of original, combined, and filled rows.
    The caller must decide whether absent months truly mean zero demand.
    """
    if raw.empty:
        raise ValueError("The CSV has no rows")
    if source_shape not in {"monthly", "transactions"}:
        raise ValueError("source_shape must be monthly or transactions")
    if bool(date_col) == bool(year_col or month_col):
        raise ValueError("Choose one date column OR both a year and a month column")
    if not date_col and not (year_col and month_col):
        raise ValueError("Select both year and month columns")
    chosen = [quantity_col, date_col, year_col, month_col, item_col]
    chosen = [name for name in chosen if name is not None]
    if len(chosen) != len(set(chosen)) or any(name not in raw.columns for name in chosen):
        raise ValueError("Choose distinct columns that exist in your CSV")

    if date_col:
        dates = pd.to_datetime(raw[date_col], errors="coerce")
    else:
        years = pd.to_numeric(raw[year_col], errors="coerce")
        months = pd.to_numeric(raw[month_col], errors="coerce")
        if years.isna().any() or months.isna().any() or not years.mod(1).eq(0).all() or not months.mod(1).eq(0).all():
            raise ValueError("Year and month columns must contain whole numbers")
        dates = pd.to_datetime(pd.DataFrame({"year": years, "month": months, "day": 1}), errors="coerce")
    if dates.isna().any():
        raise ValueError("Some dates could not be read. Use YYYY-MM-DD or numeric year and month columns")

    quantities = pd.to_numeric(raw[quantity_col], errors="coerce")
    if quantities.isna().any() or not np.isfinite(quantities.astype(float)).all() or (quantities < 0).any():
        raise ValueError("Quantities must be numbers of zero or more. Resolve returns and missing values before upload")
    items = raw[item_col].astype("string").str.strip() if item_col else pd.Series("ALL_ITEMS", index=raw.index, dtype="string")
    if items.isna().any() or items.eq("").any():
        raise ValueError("Every row needs an item ID, or choose the single-total option")

    monthly = pd.DataFrame({
        "month": dates.dt.to_period("M").dt.to_timestamp(),
        "item_id": items,
        "quantity": quantities,
    })
    duplicate_count = int(monthly.duplicated(["item_id", "month"]).sum())
    if source_shape == "monthly" and duplicate_count:
        raise ValueError("Multiple rows have the same item and month. Choose transaction rows only if summing them is intended")
    if source_shape == "transactions":
        monthly = monthly.groupby(["month", "item_id"], as_index=False, sort=True)["quantity"].sum()

    missing = []
    for item, group in monthly.groupby("item_id", sort=False):
        expected = pd.date_range(group["month"].min(), group["month"].max(), freq="MS")
        absent = expected.difference(group["month"])
        if len(absent):
            missing.append((str(item), absent))
    missing_count = sum(len(dates) for _, dates in missing)
    if missing_count and not fill_missing_with_zero:
        item, absent = missing[0]
        raise ValueError(
            f"{missing_count} item-months are missing; first example: {item} in {absent[0]:%Y-%m}. "
            "Correct the source or explicitly mark missing active months as zero."
        )
    if missing_count:
        extra = pd.DataFrame(
            [(month, item, 0.0) for item, absent in missing for month in absent],
            columns=HISTORY_COLUMNS,
        )
        monthly = pd.concat([monthly, extra], ignore_index=True)

    prepared = validate_history(monthly)
    return prepared, {
        "source_rows": len(raw),
        "combined_rows": duplicate_count if source_shape == "transactions" else 0,
        "zero_filled_months": missing_count,
        "monthly_rows": len(prepared),
    }
