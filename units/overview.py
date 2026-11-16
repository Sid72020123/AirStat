"""Overview page: what dataset is loaded, a first look, and a map of the syllabus."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import AppContext
from utils.config import display_name


def render(ctx: AppContext) -> None:
    st.header("Dataset Overview")
    overview = st.columns(6)
    overview[0].metric("City", ctx.city)
    overview[1].metric("Station", ctx.station)
    overview[2].metric("Observations", len(ctx.filtered))
    overview[3].metric("Selected variable", display_name(ctx.pollutant))
    overview[4].metric("Unit", ctx.units.get(ctx.pollutant, "Not provided"))
    overview[5].metric("Rows after cleaning", len(ctx.data))
    st.write(
        f"Selected period: **{ctx.start_date} to {ctx.end_date}** | "
        f"Available data: **{ctx.first_date} to {ctx.last_date}**"
    )
    st.dataframe(
        ctx.analysis_data.drop(columns=["suspicious_zero"]).head(10), width="stretch"
    )

    st.subheader("Where to find each syllabus topic")
    st.dataframe(
        pd.DataFrame(
            {
                "Section": [
                    "AQI", "Unit I", "Unit II", "Unit III", "Unit IV", "Data Quality",
                ],
                "What it covers": [
                    "CPCB Air Quality Index from rolling averages",
                    "Data types, collection, sampling, cleaning, descriptive statistics, EDA, correlation, predictive analytics intro",
                    "Probability, conditional probability, Bayes, random variables, binomial, Poisson, normal, Z, normal approximation",
                    "Sampling distribution, CLT, estimation, confidence intervals, hypothesis tests (Z, t, paired, ANOVA)",
                    "Correlation, simple & multiple linear regression, assumptions, evaluation metrics, interpretation, workflow",
                    "Missing values, zeros, duplicates and gaps in the data",
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("Use the sidebar to switch sections. City, station, date range and variables chosen in the "
               "sidebar apply to every section.")
