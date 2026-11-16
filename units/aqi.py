"""AQI page: CPCB Air Quality Index from the latest rolling averages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.common import AppContext
from utils.aqi_calculator import latest_valid_aqi
from utils.config import display_name


def render(ctx: AppContext) -> None:
    current_aqi, current_aqi_category, aqi_indices, aqi_time = latest_valid_aqi(
        ctx.filtered, ctx.units
    )
    st.header("Current AQI")
    if current_aqi is None:
        st.info(
            "Insufficient data for CPCB AQI. At least three pollutant indices, "
            "including PM2.5 or PM10, are required."
        )
    else:
        aqi_columns = st.columns(2)
        aqi_columns[0].metric("CPCB AQI", f"{current_aqi:.0f}")
        aqi_columns[1].metric("Category", current_aqi_category)
        st.caption(
            "Calculated from the latest available rolling averages in the selected "
            "date range. PM2.5, PM10, NO2, and SO2 use 24 hours; O3 and CO use 8 hours."
        )
        st.caption(f"AQI is for the latest hour with enough valid data: {aqi_time:%Y-%m-%d %H:%M} UTC.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Pollutant": display_name(name), "Sub-index": value}
                    for name, value in aqi_indices.items()
                ]
            ),
            hide_index=True,
        )
