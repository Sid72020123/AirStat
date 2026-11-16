"""AirStat: statistical analysis of real-time air quality across Indian cities.

This file only wires things together:
  1. page set-up,
  2. sidebar navigation + data selection (components/navigation.py),
  3. calling the page module for the selected section (units/*.py).
"""

import streamlit as st
from dotenv import load_dotenv

from components.navigation import build_context, render_navigation
from units import aqi, data_quality, overview, unit1, unit2, unit3, unit4

# Section key (see components.navigation.SECTIONS) -> page renderer.
PAGES = {
    "Overview": overview.render,
    "AQI": aqi.render,
    "Unit I": unit1.render,
    "Unit II": unit2.render,
    "Unit III": unit3.render,
    "Unit IV": unit4.render,
    "Data Quality": data_quality.render,
}


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="AirStat", page_icon="🌍", layout="wide")
    st.title("AirStat")
    st.caption("Explore air quality across India's major cities")

    section = render_navigation()
    ctx = build_context()
    PAGES[section](ctx)


if __name__ == "__main__":
    main()
