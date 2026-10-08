import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openaq import OpenAQ
from openaq.core.exceptions import OpenAQError

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not found. Put it in your .env file.")


# ------------------------------------------------------------
# SELECTED STATION
# ------------------------------------------------------------
# Katraj Dairy, Pune - MPCB
LOCATION_ID = 3409438
LOCATION_NAME = "Katraj Dairy, Pune - MPCB"
CITY = "Pune"


# ------------------------------------------------------------
# HISTORICAL PERIOD
# ------------------------------------------------------------
# Start with 30 days.
# You can change this to 60 or 90 later.
DAYS_BACK = 30

END_DATE = datetime.now(timezone.utc)
START_DATE = END_DATE - timedelta(days=DAYS_BACK)


# ------------------------------------------------------------
# SENSOR IDs
# ------------------------------------------------------------
# We deliberately use the µg/m³ sensors for the main
# statistical analysis so that pollutant units remain consistent.

SENSORS = {
    "pm25": 12248370,
    "pm10": 12248369,
    "o3": 12248368,
    "no2": 12248367,
    "so2": 12248372,
    "co": 12248365,
    "temperature": 12248373,
    "relative_humidity": 12248371,
}

DATA_WINDOWS = []


# ============================================================
# OPENAQ DATA COLLECTION
# ============================================================


def fetch_sensor_data(client, sensor_id, parameter_name):
    """
    Download hourly historical data for one sensor.

    OpenAQ returns up to 1000 records per page, so we continue
    requesting pages until no more records are returned.
    """

    all_rows = []
    page = 1

    try:
        sensor = client.sensors.get(sensor_id).results[0]
    except OpenAQError as error:
        print(f"  ERROR reading {parameter_name} metadata: {error}")
        return pd.DataFrame()

    if sensor.datetime_last is None:
        print(f"  No available timestamps for {parameter_name}; skipping.")
        return pd.DataFrame()

    # OpenAQ keeps historical sensors after they stop reporting. Querying
    # through the current date therefore returns no rows for those sensors.
    sensor_end = datetime.fromisoformat(sensor.datetime_last.utc.replace("Z", "+00:00"))
    query_end = min(END_DATE, sensor_end)
    query_start = query_end - timedelta(days=DAYS_BACK)
    DATA_WINDOWS.append((query_start, query_end))

    while True:

        print(f"  Downloading {parameter_name}: " f"page {page}...")

        try:
            response = client.measurements.list(
                sensors_id=sensor_id,
                data="hours",
                datetime_from=query_start,
                datetime_to=query_end,
                page=page,
                limit=1000,
            )

        except OpenAQError as error:
            print(f"  ERROR downloading {parameter_name}: {error}")
            break

        results = response.results

        if not results:
            break

        for item in results:

            # Timestamp
            timestamp = item.period.datetime_from.utc

            # Value
            value = item.value

            all_rows.append(
                {
                    "datetime": timestamp,
                    parameter_name: value,
                }
            )

        # If fewer than 1000 results came back,
        # we have reached the final page.
        if len(results) < 1000:
            break

        page += 1

    print(f"  → {len(all_rows)} records collected " f"for {parameter_name}")

    return pd.DataFrame(all_rows)


# ============================================================
# DOWNLOAD ALL VARIABLES
# ============================================================

print("=" * 70)
print("AirStat - Historical Air Quality Data Collection")
print("=" * 70)

print(f"City       : {CITY}")
print(f"Location   : {LOCATION_NAME}")
print(f"Location ID: {LOCATION_ID}")
print(f"From       : {START_DATE}")
print(f"To         : {END_DATE}")
print("=" * 70)


with OpenAQ(api_key=API_KEY) as client:

    dataframes = []

    for parameter, sensor_id in SENSORS.items():

        print(f"\nFetching {parameter} " f"(sensor {sensor_id})...")

        df_sensor = fetch_sensor_data(client, sensor_id, parameter)

        if not df_sensor.empty:
            dataframes.append(df_sensor)


# ============================================================
# MERGE ALL SENSORS
# ============================================================

if not dataframes:
    raise RuntimeError("No data was downloaded from OpenAQ.")


df = dataframes[0]

for other_df in dataframes[1:]:

    df = pd.merge(df, other_df, on="datetime", how="outer")


# ============================================================
# BASIC CLEANING
# ============================================================

print("\nCleaning data...")

# Convert timestamp
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

# Sort chronologically
df = df.sort_values("datetime")

# Remove duplicate timestamps
df = df.drop_duplicates(subset=["datetime"])

# Reset index
df = df.reset_index(drop=True)


# Convert numeric columns
for column in df.columns:

    if column != "datetime":
        df[column] = pd.to_numeric(df[column], errors="coerce")


# ============================================================
# ADD DATE / TIME FEATURES
# ============================================================

df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.day_name()


# ============================================================
# DATA QUALITY REPORT
# ============================================================

print("\n")
print("=" * 70)
print("DATA QUALITY REPORT")
print("=" * 70)

print(f"Total rows: {len(df)}")

print("\nMissing values:")

missing = df.isna().sum()

print(missing)


print("\nMissing percentage:")

missing_percent = (df.isna().mean() * 100).round(2)

print(missing_percent)


# ============================================================
# SAVE RAW CLEANED HOURLY DATA
# ============================================================

df.to_csv("katraj_dairy_hourly_cleaned.csv", index=False)

print("\nSaved: katraj_dairy_hourly_cleaned.csv")


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

STATISTICAL_COLUMNS = [
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
    "temperature",
    "relative_humidity",
    "wind_speed",
]


# Keep only columns that actually exist
STATISTICAL_COLUMNS = [col for col in STATISTICAL_COLUMNS if col in df.columns]


# ------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

statistics = []

for column in STATISTICAL_COLUMNS:

    series = df[column].dropna()

    if len(series) == 0:
        continue

    statistics.append(
        {
            "Variable": column,
            "Count": len(series),
            "Mean": series.mean(),
            "Median": series.median(),
            "Mode": (series.mode().iloc[0] if not series.mode().empty else np.nan),
            # Sample variance
            "Variance": series.var(),
            # Sample standard deviation
            "Std_Dev": series.std(),
            "Minimum": series.min(),
            "Maximum": series.max(),
            "Range": series.max() - series.min(),
            "Q1": series.quantile(0.25),
            "Q3": series.quantile(0.75),
            "IQR": (series.quantile(0.75) - series.quantile(0.25)),
        }
    )


statistics_df = pd.DataFrame(statistics)


# ============================================================
# PRINT STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(statistics_df.to_string(index=False))


statistics_df.to_csv("midsem_descriptive_statistics.csv", index=False)

print("\nSaved: midsem_descriptive_statistics.csv")


# ============================================================
# COVARIANCE
# ============================================================

covariance_df = df[STATISTICAL_COLUMNS].cov()

print("\n")
print("=" * 70)
print("COVARIANCE MATRIX")
print("=" * 70)

print(covariance_df)


covariance_df.to_csv("midsem_covariance.csv")

print("\nSaved: midsem_covariance.csv")


# ============================================================
# PEARSON CORRELATION
# ============================================================

pearson_df = df[STATISTICAL_COLUMNS].corr(method="pearson")

print("\n")
print("=" * 70)
print("PEARSON CORRELATION MATRIX")
print("=" * 70)

print(pearson_df)


pearson_df.to_csv("midsem_pearson_correlation.csv")

print("\nSaved: midsem_pearson_correlation.csv")


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

spearman_df = df[STATISTICAL_COLUMNS].corr(method="spearman")

print("\n")
print("=" * 70)
print("SPEARMAN CORRELATION MATRIX")
print("=" * 70)

print(spearman_df)


spearman_df.to_csv("midsem_spearman_correlation.csv")

print("\nSaved: midsem_spearman_correlation.csv")


# ============================================================
# CPCB AQI CALCULATION
# ============================================================

"""
CPCB AQI methodology:

PM10  -> 24-hour average
PM2.5 -> 24-hour average
NO2   -> 24-hour average
SO2   -> 24-hour average
O3    -> 8-hour average
CO    -> 8-hour average

The overall AQI is the maximum pollutant sub-index.

This implementation uses the CPCB breakpoint table.

Important:
AQI is a DERIVED variable and is not the primary variable
for the mid-sem statistical analysis.
"""


# ------------------------------------------------------------
# CPCB BREAKPOINT TABLE
# ------------------------------------------------------------

AQI_BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, float("inf"), 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, float("inf"), 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, float("inf"), 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, float("inf"), 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, float("inf"), 401, 500),
    ],
    # CO is converted from µg/m³ to mg/m³ before
    # applying these breakpoints.
    "co": [
        (0, 0.5, 0, 50),
        (0.6, 1.0, 51, 100),
        (1.1, 2.0, 101, 200),
        (2.1, 3.0, 201, 300),
        (3.1, 3.5, 301, 400),
        (3.6, float("inf"), 401, 500),
    ],
}


def calculate_sub_index(concentration, pollutant):
    """
    Calculate CPCB pollutant sub-index
    using linear interpolation.
    """

    if pd.isna(concentration):
        return np.nan

    breakpoints = AQI_BREAKPOINTS[pollutant]

    for blo, bhi, ilo, ihi in breakpoints:

        if blo <= concentration <= bhi:

            # CPCB linear interpolation
            if bhi == blo:
                return ilo

            index = ((ihi - ilo) / (bhi - blo)) * (concentration - blo) + ilo

            return round(index, 2)

    return np.nan


def aqi_category(aqi):

    if pd.isna(aqi):
        return "Insufficient data"

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ============================================================
# CREATE DAILY / ROLLING AQI INPUTS
# ============================================================

aqi_data = df.copy()

aqi_data = aqi_data.set_index("datetime")


# ------------------------------------------------------------
# 24-HOUR AVERAGES
# ------------------------------------------------------------

for pollutant in [
    "pm25",
    "pm10",
    "no2",
    "so2",
]:

    if pollutant in aqi_data.columns:

        aqi_data[f"{pollutant}_24h"] = (
            aqi_data[pollutant].rolling("24h", min_periods=16).mean()
        )


# ------------------------------------------------------------
# 8-HOUR ROLLING AVERAGES
# ------------------------------------------------------------

for pollutant in [
    "o3",
    "co",
]:

    if pollutant in aqi_data.columns:

        aqi_data[f"{pollutant}_8h"] = (
            aqi_data[pollutant].rolling("8h", min_periods=8).mean()
        )


# ------------------------------------------------------------
# CONVERT CO
# ------------------------------------------------------------
# OpenAQ CO sensor selected above is µg/m³.
# CPCB AQI breakpoint table expresses CO in mg/m³.

if "co_8h" in aqi_data.columns:

    aqi_data["co_8h_mg_m3"] = aqi_data["co_8h"] / 1000


# ============================================================
# CALCULATE POLLUTANT SUB-INDICES
# ============================================================

if "pm25_24h" in aqi_data.columns:

    aqi_data["AQI_PM25"] = aqi_data["pm25_24h"].apply(
        lambda x: calculate_sub_index(x, "pm25")
    )


if "pm10_24h" in aqi_data.columns:

    aqi_data["AQI_PM10"] = aqi_data["pm10_24h"].apply(
        lambda x: calculate_sub_index(x, "pm10")
    )


if "no2_24h" in aqi_data.columns:

    aqi_data["AQI_NO2"] = aqi_data["no2_24h"].apply(
        lambda x: calculate_sub_index(x, "no2")
    )


if "so2_24h" in aqi_data.columns:

    aqi_data["AQI_SO2"] = aqi_data["so2_24h"].apply(
        lambda x: calculate_sub_index(x, "so2")
    )


if "o3_8h" in aqi_data.columns:

    aqi_data["AQI_O3"] = aqi_data["o3_8h"].apply(lambda x: calculate_sub_index(x, "o3"))


if "co_8h_mg_m3" in aqi_data.columns:

    aqi_data["AQI_CO"] = aqi_data["co_8h_mg_m3"].apply(
        lambda x: calculate_sub_index(x, "co")
    )


# ============================================================
# OVERALL AQI
# ============================================================

AQI_COLUMNS = [
    "AQI_PM25",
    "AQI_PM10",
    "AQI_NO2",
    "AQI_SO2",
    "AQI_O3",
    "AQI_CO",
]

AQI_COLUMNS = [col for col in AQI_COLUMNS if col in aqi_data.columns]


# CPCB requires at least 3 pollutants,
# including PM2.5 or PM10, for an overall AQI.

available_pollutant_subindices = aqi_data[AQI_COLUMNS].notna().sum(axis=1)


has_particulate_index = (
    aqi_data[[col for col in ["AQI_PM25", "AQI_PM10"] if col in aqi_data.columns]]
    .notna()
    .any(axis=1)
)


aqi_data["AQI"] = np.nan


valid_aqi = (available_pollutant_subindices >= 3) & has_particulate_index


aqi_data.loc[valid_aqi, "AQI"] = aqi_data.loc[valid_aqi, AQI_COLUMNS].max(axis=1)


aqi_data["AQI_Category"] = aqi_data["AQI"].apply(aqi_category)


# ============================================================
# SAVE AQI DATA
# ============================================================

aqi_output = aqi_data.reset_index()

aqi_output.to_csv("katraj_dairy_aqi_analysis.csv", index=False)

print("\nSaved: katraj_dairy_aqi_analysis.csv")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("AIRSTAT MID-SEM DATA COLLECTION COMPLETE")
print("=" * 70)

print(f"Station : {LOCATION_NAME}")

if DATA_WINDOWS:
    data_start = min(window[0] for window in DATA_WINDOWS)
    data_end = max(window[1] for window in DATA_WINDOWS)
    print(f"Period  : {data_start.date()} to {data_end.date()}")
else:
    print(f"Period  : {START_DATE.date()} to {END_DATE.date()}")

print(f"Rows    : {len(df)}")

print("\nFiles created:")

print("1. katraj_dairy_hourly_cleaned.csv")

print("2. midsem_descriptive_statistics.csv")

print("3. midsem_covariance.csv")

print("4. midsem_pearson_correlation.csv")

print("5. midsem_spearman_correlation.csv")

print("6. katraj_dairy_aqi_analysis.csv")

print("\n")
print("Done.")
