"""CPCB Air Quality Index (AQI) calculation."""

import numpy as np
import pandas as pd

AQI_BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, np.inf, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, np.inf, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, np.inf, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, np.inf, 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, np.inf, 401, 500),
    ],
    "co": [
        (0, 0.5, 0, 50),
        (0.6, 1.0, 51, 100),
        (1.1, 2.0, 101, 200),
        (2.1, 3.0, 201, 300),
        (3.1, 3.5, 301, 400),
        (3.6, np.inf, 401, 500),
    ],
}

AQI_AVERAGING = {
    "pm25": ("24h", 16),
    "pm10": ("24h", 16),
    "no2": ("24h", 16),
    "so2": ("24h", 16),
    "o3": ("8h", 8),
    "co": ("8h", 8),
}


def aqi_sub_index(concentration: float, pollutant: str) -> float:
    """Calculate one CPCB pollutant sub-index by linear interpolation.

    Sub-index = I_low + (I_high - I_low) * (C - C_low) / (C_high - C_low)
    where [C_low, C_high] is the concentration band that contains C.
    """
    if pd.isna(concentration) or concentration < 0:
        return np.nan
    for concentration_low, concentration_high, index_low, index_high in AQI_BREAKPOINTS[
        pollutant
    ]:
        # The CPCB table has small gaps between bands (e.g. PM2.5 30 -> 31).
        # Real averages such as 30.4 fall in a gap, so a value is placed in the
        # first band whose upper limit it does not exceed.
        if concentration <= concentration_high:
            if np.isinf(concentration_high):
                return float(index_low)
            concentration = max(concentration, concentration_low)
            return round(
                index_low
                + (index_high - index_low)
                * (concentration - concentration_low)
                / (concentration_high - concentration_low),
                2,
            )
    return np.nan


def aqi_category(aqi: float | None) -> str:
    """Return the CPCB category for an AQI value."""
    if aqi is None or pd.isna(aqi):
        return "Insufficient data"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def _co_is_mg_per_m3(values: pd.Series, unit: str) -> bool:
    """Decide whether CO values are in mg/m3 (the unit the CPCB table uses).

    The unit string is trusted when present. Offline CSVs lose their units, so
    then a typical-magnitude check is used: ambient CO is ~0.2-5 in mg/m3 but
    hundreds to thousands in ug/m3.
    """
    unit = unit.lower().replace("µ", "u")
    if "mg/m" in unit:
        return True
    if "ug/m" in unit or "ppm" in unit:
        return False
    return bool(values.dropna().median() < 50)


def latest_valid_aqi(
    data: pd.DataFrame, units: dict[str, str]
) -> tuple[float | None, str, dict[str, float], pd.Timestamp | None]:
    """Return (AQI, category, sub-indices, timestamp) for the latest valid hour.

    CPCB rule: PM2.5/PM10/NO2/SO2 use a 24-hour mean (at least 16 readings),
    O3/CO an 8-hour mean (at least 8 readings). AQI = the largest sub-index,
    and needs at least 3 pollutants including PM2.5 or PM10. The newest hour
    can have gaps, so the newest timestamp satisfying the rule is used.
    """
    if "datetime" not in data:
        return None, "Insufficient data", {}, None

    indexed = data.set_index("datetime").sort_index()
    averages = {}
    for pollutant, (window, minimum_periods) in AQI_AVERAGING.items():
        if pollutant not in indexed:
            continue
        values = pd.to_numeric(indexed[pollutant], errors="coerce")
        average = values.rolling(window, min_periods=minimum_periods).mean()
        if pollutant == "co" and _co_is_mg_per_m3(values, units.get("co", "")) is False:
            average = average / 1000  # ug/m3 -> mg/m3
        averages[pollutant] = average.where(average >= 0)
    if not averages:
        return None, "Insufficient data", {}, None

    table = pd.DataFrame(averages)
    has_particulate = table.reindex(columns=["pm25", "pm10"]).notna().any(axis=1)
    usable = table[(table.notna().sum(axis=1) >= 3) & has_particulate]
    if usable.empty:
        # Report whatever sub-indices exist at the last row for transparency.
        last = table.iloc[-1].dropna()
        partial = {p: aqi_sub_index(float(v), p) for p, v in last.items()}
        return None, "Insufficient data", partial, None

    timestamp = usable.index[-1]
    row = usable.iloc[-1].dropna()
    sub_indices = {p: aqi_sub_index(float(v), p) for p, v in row.items()}
    aqi = max(sub_indices.values())
    return aqi, aqi_category(aqi), sub_indices, timestamp


def calculate_current_aqi(
    data: pd.DataFrame, units: dict[str, str]
) -> tuple[float | None, str, dict[str, float]]:
    """Calculate the latest CPCB AQI (see latest_valid_aqi for the rules)."""
    aqi, category, sub_indices, _ = latest_valid_aqi(data, units)
    return aqi, category, sub_indices
