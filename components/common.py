"""Shared UI helpers and the AppContext handed to every unit page."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

from utils.config import display_name


@dataclass
class AppContext:
    """Everything a unit page needs, so each unit can be read on its own.

    data              full cleaned station data (all dates, all columns)
    filtered          rows inside the sidebar date range (+ suspicious_zero flag)
    analysis_data     filtered, with flagged zeros removed if the user chose so
    units             {column: measurement unit}
    variables         numeric measurement columns available in this dataset
    pollutant         variable chosen in the sidebar
    threshold         exceedance threshold chosen in the sidebar
    variable_x / _y   comparison variables chosen in the sidebar
    """

    data: pd.DataFrame
    filtered: pd.DataFrame
    analysis_data: pd.DataFrame
    units: dict[str, str]
    variables: list[str]
    city: str
    station: str
    pollutant: str
    threshold: float
    variable_x: str
    variable_y: str
    include_suspicious_zeros: bool
    start_date: date
    end_date: date
    first_date: date
    last_date: date
    extras: dict = field(default_factory=dict)

    @property
    def unit(self) -> str:
        return self.units.get(self.pollutant, "")

    def label(self, column: str) -> str:
        """Readable name plus unit, e.g. 'PM2.5 (µg/m³)'."""
        unit = self.units.get(column, "")
        return f"{display_name(column)} ({unit})" if unit else display_name(column)


def probability_sentence(probability: float) -> str:
    """Explain a probability in simple percentage language."""
    return f"{probability:.1%} ({probability:.4f})"


def correlation_strength(value: float) -> tuple[str, str]:
    """Return (strength, direction) words for a correlation coefficient.

    |r| >= 0.7 strong, >= 0.4 moderate, otherwise weak (the thresholds used
    throughout the dashboard).
    """
    strength = (
        "strong" if abs(value) >= 0.7 else "moderate" if abs(value) >= 0.4 else "weak"
    )
    direction = "positive" if value > 0 else "negative" if value < 0 else "no"
    return strength, direction


def correlation_interpretation(value: float) -> str:
    """Give a deliberately simple interpretation without implying causation."""
    if pd.isna(value):
        return "There are not enough paired observations to calculate this correlation."
    strength, direction = correlation_strength(value)
    return f"This is a {strength} {direction} relationship. Correlation does not prove causation."


def render_statistical_conclusion(
    null_hypothesis: str,
    alternative_hypothesis: str,
    alpha: float,
    statistic_name: str,
    statistic: float,
    p_value: float,
) -> None:
    """Show the same plain-language conclusion for every hypothesis test."""
    st.markdown("#### Statistical conclusion")
    result = "Reject H₀" if p_value < alpha else "Fail to reject H₀"
    st.write(f"**H₀:** {null_hypothesis}")
    st.write(f"**H₁:** {alternative_hypothesis}")
    metrics = st.columns(3)
    metrics[0].metric("Significance level (α)", f"{alpha:.2f}")
    metrics[1].metric(statistic_name, f"{statistic:.4g}")
    metrics[2].metric("p-value", f"{p_value:.4g}")
    if p_value < alpha:
        st.error(f"**{result}.** The result is statistically significant at α = {alpha:.2f}.")
    else:
        st.info(
            f"**{result}.** There is not enough evidence of a difference at α = {alpha:.2f}."
        )


def metric_cards(items: list[tuple[str, str]], help_texts: list[str] | None = None) -> None:
    """Render a row of metric cards: [(label, value), ...]."""
    columns = st.columns(len(items))
    for index, (label, value) in enumerate(items):
        columns[index].metric(
            label, value, help=help_texts[index] if help_texts else None
        )
