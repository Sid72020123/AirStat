"""AirStat: simple descriptive analysis of Indian air-quality stations."""

from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from typing import TypedDict

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openaq import OpenAQ
from openaq.core.exceptions import HTTPRateLimitError, OpenAQError


DATA_CACHE_DIR = Path(__file__).parent / "data_cache"
DATA_CACHE_TTL = timedelta(hours=1)
STATION_CACHE_TTL = timedelta(days=90)
LOOKBACK_DAYS = 30
CITY_COORDINATES = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882),
    "Surat": (21.1702, 72.8311),
    "Bhopal": (23.2599, 77.4126),
    "Patna": (25.5941, 85.1376),
    "Chandigarh": (30.7333, 76.7794),
    "Bhubaneswar": (20.2961, 85.8245),
    "Kochi": (9.9312, 76.2673),
    "Coimbatore": (11.0168, 76.9558),
}
DEFAULT_CITY = "Pune"
SEARCH_RADIUS_METERS = 25_000

DISPLAY_NAMES = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "o3": "O3",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "temperature": "Temperature",
    "relativehumidity": "Relative Humidity",
    "windspeed": "Wind Speed",
}


class StationInfo(TypedDict):
    """Metadata needed to select and download one station."""

    id: int
    name: str
    city: str
    sensors: dict[str, tuple[int, str]]
    sensor_count: int
    last_seen: datetime
    null_rate: float
    observed_rows: int


def parameter_key(name: str) -> str:
    """Make API parameter names safe to use as DataFrame column names."""
    return name.lower().replace(" ", "").replace(".", "")


def display_name(value: str) -> str:
    """Return a readable label for an API parameter."""
    return DISPLAY_NAMES.get(value, value)


def location_city(location) -> str:
    """Use OpenAQ locality, or infer a readable city from the station name."""
    if location.locality:
        return location.locality
    if "," in location.name:
        return location.name.rsplit(",", 1)[-1].split("-", 1)[0].strip()
    return "Unknown city"


def sensor_parameters(location) -> dict[str, tuple[int, str]]:
    """Return one sensor ID and unit for each supported parameter."""
    result: dict[str, tuple[int, str]] = {}
    for sensor in location.sensors:
        key = parameter_key(sensor.parameter.name)
        if key in DISPLAY_NAMES and key not in result:
            result[key] = (sensor.id, sensor.parameter.units)
    return result


def station_last_seen(location) -> datetime:
    """Return a sortable timestamp, using the oldest possible value if absent."""
    timestamp = getattr(location, "datetime_last", None)
    if timestamp is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(timestamp.utc.replace("Z", "+00:00"))


def safe_city_filename(city: str) -> str:
    """Create a stable, filesystem-safe filename component."""
    return re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")


def station_label(station: StationInfo) -> str:
    """Create a useful station option label for the sidebar."""
    return (
        f"{station['name']} (ID {station['id']}) | "
        f"{station['null_rate']:.1%} missing | "
        f"{station['observed_rows']} sample rows"
    )


def station_cache_path(city: str) -> Path:
    """Return the persistent three-month station metadata cache path."""
    return DATA_CACHE_DIR / f"stations_{safe_city_filename(city)}.json"


def station_to_json(station: StationInfo) -> dict[str, object]:
    """Convert station metadata into JSON-safe values."""
    return {
        **station,
        "sensors": {
            key: {"id": sensor_id, "unit": unit}
            for key, (sensor_id, unit) in station["sensors"].items()
        },
        "last_seen": station["last_seen"].isoformat(),
    }


def station_from_json(raw: dict[str, object]) -> StationInfo:
    """Restore and validate one station from the persistent cache."""
    sensors_raw = raw["sensors"]
    if not isinstance(sensors_raw, dict):
        raise ValueError("Invalid station sensor metadata.")
    sensors: dict[str, tuple[int, str]] = {}
    for key, value in sensors_raw.items():
        if not isinstance(value, dict):
            raise ValueError("Invalid sensor entry.")
        sensor_id = value.get("id")
        unit = value.get("unit")
        if not isinstance(sensor_id, int) or not isinstance(unit, str):
            raise ValueError("Invalid sensor values.")
        sensors[str(key)] = (sensor_id, unit)

    station_id = raw.get("id")
    sensor_count = raw.get("sensor_count")
    null_rate = raw.get("null_rate")
    observed_rows = raw.get("observed_rows")
    if (
        not isinstance(station_id, int)
        or not isinstance(sensor_count, int)
        or not isinstance(null_rate, int | float)
        or not isinstance(observed_rows, int)
    ):
        raise ValueError("Invalid station quality metadata.")
    return {
        "id": station_id,
        "name": str(raw["name"]),
        "city": str(raw["city"]),
        "sensors": sensors,
        "sensor_count": sensor_count,
        "last_seen": datetime.fromisoformat(str(raw["last_seen"])),
        "null_rate": float(null_rate),
        "observed_rows": observed_rows,
    }


def read_station_cache(city: str) -> list[StationInfo] | None:
    """Read a fresh three-month station cache, returning None when unavailable."""
    path = station_cache_path(city)
    if not path.exists() or (
        datetime.now(timezone.utc)
        - datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        >= STATION_CACHE_TTL
    ):
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            return None
        return [station_from_json(item) for item in raw]
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def write_station_cache(city: str, stations: list[StationInfo]) -> None:
    """Persist ranked station metadata atomically for the next three months."""
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    path = station_cache_path(city)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps([station_to_json(station) for station in stations], indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def clear_station_cache(city: str) -> None:
    """Remove one city's persistent station cache."""
    station_cache_path(city).unlink(missing_ok=True)


def load_measurement_cache(
    cache_path: Path, station: StationInfo
) -> tuple[pd.DataFrame, dict[str, str], str, str] | None:
    """Load a station CSV regardless of age for rate-limit fallback."""
    if not cache_path.exists():
        return None
    try:
        data = clean_data(pd.read_csv(cache_path, parse_dates=["datetime"]))
        if data.empty or "datetime" not in data:
            return None
        return (
            data,
            {key: unit for key, (_, unit) in station["sensors"].items()},
            station["city"],
            station["name"],
        )
    except (KeyError, OSError, ValueError, pd.errors.ParserError):
        return None


def station_sample(
    client: OpenAQ, station: StationInfo, sample_days: int = 3
) -> tuple[float, int]:
    """Measure recent missingness across a station's supported sensors."""
    end = min(datetime.now(timezone.utc), station["last_seen"])
    if end == datetime.min.replace(tzinfo=timezone.utc):
        end = datetime.now(timezone.utc)
    start = end - timedelta(days=sample_days)
    frames = []
    for parameter, (sensor_id, _) in station["sensors"].items():
        frame = fetch_sensor_measurements(client, sensor_id, parameter, start, end)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return 1.0, 0
    sample = frames[0]
    for frame in frames[1:]:
        sample = pd.merge(sample, frame, on="datetime", how="outer")
    values = sample.drop(columns=["datetime"])
    return float(values.isna().mean().mean()), len(sample)


@st.cache_data(ttl=3600)
def find_stations(
    api_key: str, city: str, force_refresh: bool = False
) -> list[StationInfo]:
    """Find and rank nearby stations by recent null rate, then recency."""
    if not force_refresh:
        cached_stations = read_station_cache(city)
        if cached_stations is not None:
            return cached_stations

    latitude, longitude = CITY_COORDINATES[city]
    try:
        with OpenAQ(api_key=api_key) as client:
            locations = client.locations.list(
                coordinates=(latitude, longitude),
                radius=SEARCH_RADIUS_METERS,
                iso="IN",
                limit=100,
            ).results
    except HTTPRateLimitError:
        fallback = read_station_cache(DEFAULT_CITY)
        if fallback is not None:
            return fallback
        raise RuntimeError(
            "OpenAQ is temporarily rate-limited and no local station cache is available. "
            "Please try again later."
        )
    except OpenAQError as error:
        raise RuntimeError(f"Could not search stations near {city}: {error}") from error

    candidates: list[StationInfo] = []
    for location in locations:
        sensors = sensor_parameters(location)
        if sensors:
            candidates.append(
                {
                    "id": location.id,
                    "name": location.name,
                    "city": city,
                    "sensors": sensors,
                    "sensor_count": len(sensors),
                    "last_seen": station_last_seen(location),
                    "null_rate": 1.0,
                    "observed_rows": 0,
                }
            )

    if not candidates:
        raise RuntimeError(
            f"No OpenAQ station with supported sensors was found near {city}."
        )

    try:
        with OpenAQ(api_key=api_key) as client:
            for station in candidates:
                null_rate, observed_rows = station_sample(client, station)
                station["null_rate"] = null_rate
                station["observed_rows"] = observed_rows
    except HTTPRateLimitError:
        fallback = read_station_cache(DEFAULT_CITY)
        if fallback is not None:
            return fallback
        raise RuntimeError(
            "OpenAQ is temporarily rate-limited and no local station cache is available. "
            "Please try again later."
        )
    except OpenAQError as error:
        raise RuntimeError(f"Could not evaluate stations near {city}: {error}") from error

    candidates.sort(
        key=lambda item: (
            item["null_rate"],
            -item["observed_rows"],
            -item["sensor_count"],
            -item["last_seen"].timestamp(),
        )
    )
    write_station_cache(city, candidates)
    return candidates


def fetch_sensor_measurements(
    client: OpenAQ,
    sensor_id: int,
    parameter: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Download all hourly pages for one sensor."""
    rows: list[dict[str, object]] = []
    page = 1

    while True:
        response = client.measurements.list(
            sensors_id=sensor_id,
            data="hours",
            datetime_from=start,
            datetime_to=end,
            page=page,
            limit=1000,
        )
        if not response.results:
            break

        rows.extend(
            {
                "datetime": measurement.period.datetime_from.utc,
                parameter: measurement.value,
            }
            for measurement in response.results
            if measurement.value is not None
        )
        if len(response.results) < 1000:
            break
        page += 1

    return pd.DataFrame(rows, columns=["datetime", parameter])


@st.cache_data(ttl=3600)
def fetch_data(
    api_key: str,
    station: StationInfo,
    force_refresh: bool = False,
    days_back: int = LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, dict[str, str], str, str]:
    """Load cached station data or fetch, clean, and save a fresh CSV."""
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    location_id = int(station["id"])
    city_name = str(station["city"])
    station_name = str(station["name"])
    cache_path = DATA_CACHE_DIR / (
        f"{safe_city_filename(city_name)}_{location_id}_{days_back}d.csv"
    )
    if (
        not force_refresh
        and cache_path.exists()
        and datetime.now(timezone.utc) - datetime.fromtimestamp(
            cache_path.stat().st_mtime, timezone.utc
        )
        < DATA_CACHE_TTL
    ):
        try:
            cached_result = load_measurement_cache(cache_path, station)
            if cached_result is not None:
                return cached_result
        except (KeyError, OSError, ValueError, pd.errors.ParserError):
            # A partial or malformed cache is replaced by a fresh API download.
            cache_path.unlink(missing_ok=True)

    end = datetime.now(timezone.utc)
    frames: list[pd.DataFrame] = []
    units: dict[str, str] = {}

    try:
        with OpenAQ(api_key=api_key) as client:
            location_response = client.locations.get(location_id)
            if not location_response.results:
                raise RuntimeError(f"No metadata found for OpenAQ location {location_id}.")

            location = location_response.results[0]
            station_name = location.name
            sensors = sensor_parameters(location)
            if not sensors:
                raise RuntimeError(f"No supported sensors found for {station_name}.")

            for parameter, (sensor_id, unit) in sensors.items():
                sensor_response = client.sensors.get(sensor_id)
                sensor = sensor_response.results[0] if sensor_response.results else None
                sensor_end = end
                if sensor is not None and sensor.datetime_last is not None:
                    sensor_end = datetime.fromisoformat(
                        sensor.datetime_last.utc.replace("Z", "+00:00")
                    )
                query_end = min(end, sensor_end)
                query_start = query_end - timedelta(days=days_back)
                frame = fetch_sensor_measurements(
                    client, sensor_id, parameter, query_start, query_end
                )
                if not frame.empty:
                    frames.append(frame)
                    units[parameter] = unit
    except HTTPRateLimitError:
        cached_result = load_measurement_cache(cache_path, station)
        if cached_result is not None:
            return cached_result
        raise RuntimeError(
            "OpenAQ is temporarily rate-limited and no downloaded data is available "
            f"for {station_name}. Please try again later."
        )
    except OpenAQError as error:
        raise RuntimeError(f"OpenAQ request failed: {error}") from error

    if not frames:
        raise RuntimeError(
            f"No measurements were returned for {station_name} in the last {days_back} days."
        )

    data = frames[0]
    for frame in frames[1:]:
        data = pd.merge(data, frame, on="datetime", how="outer")

    cleaned = clean_data(data)
    cleaned.to_csv(cache_path, index=False)
    return cleaned, units, city_name, station_name


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply basic type conversion, sorting, and duplicate handling."""
    cleaned = data.copy()
    cleaned["datetime"] = pd.to_datetime(cleaned["datetime"], utc=True, errors="coerce")
    cleaned = cleaned.dropna(subset=["datetime"])
    cleaned.attrs["rows_before_cleaning"] = len(cleaned)
    cleaned.attrs["duplicate_timestamps"] = int(cleaned.duplicated(subset=["datetime"]).sum())
    cleaned = cleaned.drop_duplicates(subset=["datetime"]).sort_values("datetime")

    for column in cleaned.columns:
        if column != "datetime":
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.reset_index(drop=True)


def descriptive_statistics(series: pd.Series) -> pd.DataFrame:
    """Calculate the Unit I descriptive statistics for one variable."""
    values = series.dropna()
    if values.empty:
        return pd.DataFrame(columns=["Statistic", "Value"])

    mode = values.mode()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return pd.DataFrame(
        {
            "Statistic": [
                "Mean", "Median", "Mode", "Variance", "Standard Deviation",
                "Minimum", "Maximum", "Range", "Q1", "Q3", "IQR",
            ],
            "Value": [
                f"{values.mean():.4g}", f"{values.median():.4g}",
                ", ".join(f"{value:g}" for value in mode),
                f"{values.var():.4g}", f"{values.std():.4g}",
                f"{values.min():.4g}", f"{values.max():.4g}",
                f"{values.max() - values.min():.4g}", f"{q1:.4g}",
                f"{q3:.4g}", f"{q3 - q1:.4g}",
            ],
        }
    )


def correlation_interpretation(value: float) -> str:
    """Give a deliberately simple interpretation without implying causation."""
    if pd.isna(value):
        return "There are not enough paired observations to calculate this correlation."
    strength = "strong" if abs(value) >= 0.7 else "moderate" if abs(value) >= 0.4 else "weak"
    direction = "positive" if value > 0 else "negative" if value < 0 else "no"
    return f"This is a {strength} {direction} relationship. Correlation does not prove causation."


def format_date_range(data: pd.DataFrame) -> tuple[date, date]:
    """Return the calendar dates available in the cleaned data."""
    return data["datetime"].min().date(), data["datetime"].max().date()


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="AirStat", page_icon="🌍", layout="wide")
    st.title("AirStat")
    st.caption("Explore air quality across India's major cities")

    api_key = os.getenv("OPENAQ_API_KEY")
    if api_key is None:
        st.error("OPENAQ_API_KEY is missing. Add it to a .env file and restart the app.")
        st.stop()

    with st.sidebar:
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
        if st.button("Refresh Data"):
            fetch_data.clear()
            find_stations.clear()
            clear_station_cache(city_choice)
            st.session_state["refresh_city"] = city_choice
            st.rerun()

    fallback_used = False
    try:
        station_options = find_stations(
            api_key,
            city_choice,
            force_refresh=st.session_state.get("refresh_city") == city_choice,
        )
        selected_station = station_options[0]
        selected_station_label = st.session_state.get(
            f"station_choice_{city_choice}", station_label(selected_station)
        )
        labels = [station_label(option) for option in station_options]
        if selected_station_label not in labels:
            selected_station_label = labels[0]
        selected_station = station_options[labels.index(selected_station_label)]
        force_refresh = st.session_state.pop("refresh_city", None) == city_choice
        data, units, city, station = fetch_data(
            api_key, selected_station, force_refresh=force_refresh
        )
    except RuntimeError as error:
        fallback_options = read_station_cache(DEFAULT_CITY)
        if fallback_options is None:
            st.error(str(error))
            st.stop()
        fallback_station = fallback_options[0]
        fallback_path = DATA_CACHE_DIR / (
            f"{safe_city_filename(DEFAULT_CITY)}_{fallback_station['id']}_{LOOKBACK_DAYS}d.csv"
        )
        fallback_data = load_measurement_cache(fallback_path, fallback_station)
        if fallback_data is None:
            st.error(str(error))
            st.stop()
        station_options = fallback_options
        selected_station = fallback_station
        labels = [station_label(option) for option in station_options]
        selected_station_label = labels[0]
        data, units, city, station = fallback_data
        fallback_used = True

    with st.sidebar:
        if fallback_used:
            st.warning(
                "OpenAQ is temporarily rate-limited. Showing the latest downloaded "
                "Pune data instead."
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
        variable_x = st.selectbox(
            "Variable X", variables, index=variables.index(pollutant)
        )
        variable_y = st.selectbox(
            "Variable Y", variables, index=1 if len(variables) > 1 else 0
        )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates
    filtered = data[
        (data["datetime"].dt.date >= start_date)
        & (data["datetime"].dt.date <= end_date)
    ].copy()
    values = filtered[pollutant].dropna()

    st.header("1. Dataset Overview")
    overview = st.columns(6)
    overview[0].metric("City", city)
    overview[1].metric("Station", station)
    overview[2].metric("Observations", len(filtered))
    overview[3].metric("Selected variable", display_name(pollutant))
    overview[4].metric("Unit", units.get(pollutant, "Not provided"))
    overview[5].metric("Rows after cleaning", len(data))
    st.write(
        f"Selected period: **{start_date} to {end_date}** | "
        f"Available data: **{first_date} to {last_date}**"
    )
    st.dataframe(filtered.head(10), width="stretch")

    st.header("2. Descriptive Statistics")
    if values.empty:
        st.warning("No values are available for this variable and date range.")
    else:
        st.dataframe(descriptive_statistics(filtered[pollutant]), hide_index=True)

    st.header("3. Exploratory Visualizations")
    if values.empty:
        st.info("Choose a date range containing measurements to display charts.")
    else:
        line_data = filtered[["datetime", pollutant]].dropna()
        st.plotly_chart(
            px.line(
                line_data,
                x="datetime",
                y=pollutant,
                labels={pollutant: f"{display_name(pollutant)} ({units.get(pollutant, '')})"},
                title="Time series",
            ),
            width="stretch",
        )
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(
                px.histogram(line_data, x=pollutant, title="Distribution (histogram)"),
                width="stretch",
            )
        with chart_right:
            st.plotly_chart(
                px.box(line_data, y=pollutant, title="Distribution (box plot)"),
                width="stretch",
            )

    st.header("4. Covariance and Correlation")
    paired = filtered[[variable_x, variable_y]].dropna()
    if variable_x == variable_y:
        st.warning("Choose two different variables for relationship analysis.")
    elif len(paired) < 2:
        st.warning("At least two paired observations are needed for this analysis.")
    else:
        covariance = paired[variable_x].cov(paired[variable_y])
        pearson = paired[variable_x].corr(paired[variable_y], method="pearson")
        spearman = paired[variable_x].corr(paired[variable_y], method="spearman")
        metrics = st.columns(3)
        metrics[0].metric("Covariance", f"{covariance:.4g}")
        metrics[1].metric("Pearson correlation", f"{pearson:.4f}")
        metrics[2].metric("Spearman correlation", f"{spearman:.4f}")
        st.info(f"Pearson interpretation: {correlation_interpretation(pearson)}")
        scatter = paired.rename(
            columns={variable_x: display_name(variable_x),
                     variable_y: display_name(variable_y)}
        )
        st.plotly_chart(
            px.scatter(
                scatter,
                x=display_name(variable_x),
                y=display_name(variable_y),
                title="Scatter plot",
            ),
            width="stretch",
        )

    st.header("5. Data Quality")
    quality = {
        "Total rows downloaded": data.attrs.get("rows_before_cleaning", len(data)),
        "Missing values in selected variable": int(data[pollutant].isna().sum()),
        "Duplicate timestamps removed": data.attrs.get("duplicate_timestamps", 0),
        "Rows remaining after cleaning": len(data),
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


if __name__ == "__main__":
    main()
