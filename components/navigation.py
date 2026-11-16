"""Sidebar: section navigation, data selection and the shared AppContext."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from openaq.core.exceptions import OpenAQError

from components.common import AppContext
from utils.config import (
    CITY_COORDINATES,
    DEFAULT_CITY,
    LOOKBACK_DAYS,
    LOOKBACK_OPTIONS,
    display_name,
)
from utils.data_cleaning import format_date_range, suspicious_zero_flags, valid_series
from utils.data_loader import (
    clear_station_cache,
    fetch_data,
    find_stations,
    load_offline_default_data,
    station_label,
)

# Main navigation: key -> label shown in the sidebar.
SECTIONS = {
    "Overview": "🏠 Overview",
    "AQI": "🌫️ AQI",
    "Unit I": "📘 Unit I — Data Exploration & Statistics",
    "Unit II": "🎲 Unit II — Probability & Distributions",
    "Unit III": "🧪 Unit III — Sampling & Inference",
    "Unit IV": "📈 Unit IV — Correlation & Regression",
    "Data Quality": "🧹 Data Quality",
}
NAV_KEY = "dashboard_section"


def go_to_section(section: str) -> None:
    """on_click callback that switches the main navigation (used by Unit IV)."""
    st.session_state[NAV_KEY] = section


def render_navigation() -> str:
    """Draw the section selector at the top of the sidebar and return its key."""
    with st.sidebar:
        st.header("Navigate")
        return st.radio(
            "Dashboard section",
            list(SECTIONS),
            format_func=SECTIONS.get,
            key=NAV_KEY,
            label_visibility="collapsed",
        )


def build_context() -> AppContext:
    """Draw the data-selection sidebar, load the data and return an AppContext.

    Data comes from the existing cache system: a fresh cached CSV is reused,
    the OpenAQ API is only called when an API key exists and the cache is
    missing or stale, and bundled Pune CSVs are used when offline.
    """
    api_key = os.getenv("OPENAQ_API_KEY", "").strip()

    with st.sidebar:
        st.divider()
        st.header("Data Selection")
        st.caption(
            "Choose a city. AirStat automatically finds the best recent OpenAQ "
            "station nearby."
        )
        city_choice = st.selectbox(
            "Indian city",
            list(CITY_COORDINATES),
            index=list(CITY_COORDINATES).index(DEFAULT_CITY),
            help="Pune is selected by default. You can change this anytime.",
        )
        lookback_days = st.selectbox(
            "Live data lookback",
            LOOKBACK_OPTIONS,
            index=LOOKBACK_OPTIONS.index(LOOKBACK_DAYS),
            format_func=lambda days: f"{days} days",
            help=(
                "When an API key is available, request this much recent data "
                "from the selected station. Older history may not exist for "
                "every sensor."
            ),
        )
        if st.button("Refresh Data"):
            fetch_data.clear()
            find_stations.clear()
            clear_station_cache(city_choice)
            st.session_state["refresh_city"] = city_choice
            st.rerun()

    fallback_used = False
    fallback_reason = ""
    fallback_path: Path | None = None
    force_refresh = st.session_state.pop("refresh_city", None) == city_choice
    try:
        if not api_key:
            raise RuntimeError("No OpenAQ API key was provided.")
        station_options = find_stations(
            api_key,
            city_choice,
            force_refresh=force_refresh,
        )
        selected_station = station_options[0]
        selected_station_label = st.session_state.get(
            f"station_choice_{city_choice}", station_label(selected_station)
        )
        labels = [station_label(option) for option in station_options]
        if selected_station_label not in labels:
            selected_station_label = labels[0]
        selected_station = station_options[labels.index(selected_station_label)]
        data, units, city, station = fetch_data(
            api_key,
            selected_station,
            force_refresh=force_refresh,
            days_back=lookback_days,
        )
    except (RuntimeError, OpenAQError):
        (
            data,
            units,
            city,
            station,
            selected_station,
            fallback_path,
        ) = load_offline_default_data()
        station_options = [selected_station]
        labels = [station_label(selected_station)]
        selected_station_label = labels[0]
        fallback_used = True
        fallback_reason = (
            "No API key was provided."
            if not api_key
            else "OpenAQ could not be reached or rejected the API key."
        )

    with st.sidebar:
        if fallback_used:
            st.warning(
                f"{fallback_reason} Showing the original downloaded Pune data "
                "instead."
            )
            if fallback_path is not None:
                st.caption(
                    f"Offline source: `{fallback_path.name}` | "
                    f"latest observation: {data['datetime'].max().date()}"
                )
        else:
            st.success(
                "Live OpenAQ data loaded through "
                f"{data['datetime'].max().strftime('%Y-%m-%d %H:%M UTC')}."
            )
        st.write(f"**City:** {city}")
        station_choice = st.selectbox(
            "Monitoring station",
            labels,
            index=labels.index(selected_station_label),
            key=f"station_choice_{city_choice}",
            help=(
                "Stations are ranked by recent missing-data rate. "
                "The first option is the automatically recommended station."
            ),
        )
        if station_choice != selected_station_label:
            st.rerun()
        st.write(f"**Selected station:** {station}")
        st.success(
            f"Recommended station has {selected_station['null_rate']:.1%} "
            "missing values in the recent sample."
        )

    first_date, last_date = format_date_range(data)
    with st.sidebar:
        selected_dates = st.date_input(
            "Date range",
            value=(first_date, last_date),
            min_value=first_date,
            max_value=last_date,
        )
        variables = [column for column in data.columns if column != "datetime"]
        if not variables:
            st.error("The API returned no numerical variables.")
            st.stop()
        pollutant = st.selectbox(
            "Pollutant / variable",
            variables,
            format_func=display_name,
        )
        pollutant_values = valid_series(data, pollutant)
        default_threshold = (
            float(pollutant_values.median()) if not pollutant_values.empty else 0.0
        )
        threshold = st.number_input(
            f"Exceedance threshold ({units.get(pollutant, '')})",
            value=default_threshold,
            key=f"threshold_{city_choice}_{pollutant}",
        )
        variable_x = st.selectbox(
            "Variable X", variables, index=variables.index(pollutant)
        )
        variable_y = st.selectbox(
            "Variable Y", variables, index=1 if len(variables) > 1 else 0
        )
        include_suspicious_zeros = st.checkbox(
            "Include suspicious zeros in calculations",
            value=True,
            help=(
                "Zeros are retained by default. Uncheck this to exclude zeros "
                "flagged by the quality heuristic from statistical calculations."
            ),
        )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates
    filtered = data[
        (data["datetime"].dt.date >= start_date)
        & (data["datetime"].dt.date <= end_date)
    ].copy()
    filtered[pollutant] = pd.to_numeric(filtered[pollutant], errors="coerce")
    filtered["suspicious_zero"] = suspicious_zero_flags(data, pollutant).loc[
        filtered.index
    ]
    analysis_data = filtered.copy()
    if not include_suspicious_zeros:
        analysis_data.loc[analysis_data["suspicious_zero"], pollutant] = np.nan

    with st.sidebar:
        st.caption("Data quality for selected pollutant")
        quality_values = pd.to_numeric(filtered[pollutant], errors="coerce")
        finite = quality_values.notna() & np.isfinite(quality_values)
        zero_count = int((quality_values == 0).sum())
        suspicious_count = int(filtered["suspicious_zero"].sum())
        st.write(f"Valid numeric: **{int(finite.sum())}**")
        st.write(f"Missing/invalid: **{int((~finite).sum())}**")
        st.write(f"Zero values: **{zero_count}**")
        st.write(f"Suspicious zeros: **{suspicious_count}**")
        if suspicious_count:
            st.caption(
                "Suspicious zeros are heuristic flags, not proof of sensor failure."
            )

    return AppContext(
        data=data,
        filtered=filtered,
        analysis_data=analysis_data,
        units=units,
        variables=variables,
        city=city,
        station=station,
        pollutant=pollutant,
        threshold=threshold,
        variable_x=variable_x,
        variable_y=variable_y,
        include_suspicious_zeros=include_suspicious_zeros,
        start_date=start_date,
        end_date=end_date,
        first_date=first_date,
        last_date=last_date,
    )
