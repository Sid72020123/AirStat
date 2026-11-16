"""Data Quality page: how complete and trustworthy is the selected data?"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.common import AppContext
from utils.config import display_name
from utils.data_cleaning import quality_summary


def render(ctx: AppContext) -> None:
    st.header("Data Quality")
    quality = {
        "Total rows downloaded": ctx.data.attrs.get("rows_before_cleaning", len(ctx.data)),
        "Missing values in selected variable": int(
            ctx.analysis_data[ctx.pollutant].isna().sum()
        ),
        "Suspicious zero values": int(ctx.filtered["suspicious_zero"].sum()),
        "Suspicious zeros excluded": (
            int(ctx.filtered["suspicious_zero"].sum())
            if not ctx.include_suspicious_zeros
            else 0
        ),
        "Duplicate timestamps removed": ctx.data.attrs.get("duplicate_timestamps", 0),
        "Rows remaining after cleaning": len(ctx.data),
    }
    st.dataframe(
        pd.DataFrame(
            [
                {"Measure": measure, "Value": str(value)}
                for measure, value in quality.items()
            ]
        ),
        hide_index=True,
    )

    st.subheader("All variables (selected date range)")
    summary = quality_summary(ctx.filtered, ctx.variables)
    summary["Variable"] = summary["Variable"].map(display_name)
    st.dataframe(
        summary.style.format({"Missing %": "{:.1f}%"}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Zeros are never removed automatically. 'Suspicious zeros' are zeros that are isolated between "
        "positive readings, part of a run of 3+ zeros, or occur in two pollutants at the same time; "
        "they are heuristic flags, not proof of sensor failure."
    )

    st.subheader("Completeness over time")
    numeric = ctx.filtered.set_index("datetime")[ctx.variables].apply(pd.to_numeric, errors="coerce")
    if numeric.empty:
        st.info("No data in the selected date range.")
        return
    daily_valid = numeric.notna().groupby(numeric.index.date).sum()
    daily_valid.index.name = "Date"
    long = daily_valid.rename(columns=display_name).reset_index().melt(
        id_vars="Date", var_name="Variable", value_name="Valid readings"
    )
    st.plotly_chart(
        px.line(long, x="Date", y="Valid readings", color="Variable",
                title="Valid readings per day (a full day has about 24)"),
        width="stretch",
    )
    gaps = ctx.filtered["datetime"].sort_values().diff().dropna()
    if not gaps.empty:
        st.write(f"Longest gap between consecutive readings: **{gaps.max()}**")
