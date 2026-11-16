"""Cleaning helpers: type conversion, duplicates, zero flags, daily means."""

from datetime import date

import numpy as np
import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply basic type conversion, sorting, and duplicate handling."""
    cleaned = data.copy()
    cleaned["datetime"] = pd.to_datetime(cleaned["datetime"], utc=True, errors="coerce")
    cleaned = cleaned.dropna(subset=["datetime"])
    cleaned.attrs["rows_before_cleaning"] = len(cleaned)
    cleaned.attrs["duplicate_timestamps"] = int(
        cleaned.duplicated(subset=["datetime"]).sum()
    )
    cleaned = cleaned.drop_duplicates(subset=["datetime"]).sort_values("datetime")

    for column in cleaned.columns:
        if column != "datetime":
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.reset_index(drop=True)


def format_date_range(data: pd.DataFrame) -> tuple[date, date]:
    """Return the calendar dates available in the cleaned data."""
    return data["datetime"].min().date(), data["datetime"].max().date()


def valid_series(data: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric observations without treating zero as missing."""
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    return values[np.isfinite(values)]


def suspicious_zero_flags(data: pd.DataFrame, column: str) -> pd.Series:
    """Flag zero readings that match common sensor/reporting error patterns."""
    numeric = data.apply(lambda values: pd.to_numeric(values, errors="coerce"))
    values = numeric[column]
    zero = values.eq(0)
    previous_nonzero = values.shift(1).gt(0)
    next_nonzero = values.shift(-1).gt(0)
    isolated = zero & previous_nonzero & next_nonzero

    run_id = zero.ne(zero.shift(fill_value=False)).cumsum()
    run_lengths = zero.groupby(run_id).transform("sum")
    long_run = zero & run_lengths.ge(3)

    numeric_pollutants = [name for name in numeric.columns if name != "datetime"]
    simultaneous = (
        numeric[numeric_pollutants].eq(0).sum(axis=1).ge(2)
        if len(numeric_pollutants) >= 2
        else pd.Series(False, index=data.index)
    )
    return (isolated | long_run | (zero & simultaneous)).fillna(False)


def finite_values(data: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric observations without treating zero as missing."""
    values = pd.to_numeric(data[column], errors="coerce")
    return values[np.isfinite(values)].dropna()


def daily_average_data(data: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """Create one valid daily mean per day for analyses needing independence."""
    frame = data[["datetime", pollutant]].copy()
    frame[pollutant] = pd.to_numeric(frame[pollutant], errors="coerce")
    frame = frame.dropna(subset=[pollutant])
    frame = frame[np.isfinite(frame[pollutant])]
    if frame.empty:
        return frame.assign(day=pd.Series(dtype="object"))
    frame["day"] = frame["datetime"].dt.date
    return frame.groupby("day", as_index=False)[pollutant].mean()


def quality_summary(data: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Per-variable data-quality table: valid / missing / zero / suspicious zero counts.

    Zeros are counted, never removed: a real 0 reading can be valid, so only the
    heuristic 'suspicious' zeros (isolated, long runs, simultaneous) are flagged.
    """
    rows = []
    for column in variables:
        values = pd.to_numeric(data[column], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        rows.append(
            {
                "Variable": column,
                "Valid": int(valid.sum()),
                "Missing": int((~valid).sum()),
                "Missing %": float((~valid).mean() * 100) if len(values) else float("nan"),
                "Zeros": int((values == 0).sum()),
                "Suspicious zeros": int(suspicious_zero_flags(data, column).sum()),
            }
        )
    return pd.DataFrame(rows)


def iqr_outlier_bounds(values: pd.Series) -> tuple[float, float, int, int]:
    """Tukey fences: Q1 - 1.5*IQR and Q3 + 1.5*IQR. Returns (low, high, n_low, n_high)."""
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = q3 - q1
    low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
    return float(low), float(high), int((values < low).sum()), int((values > high).sum())
