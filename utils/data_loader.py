"""Station discovery, OpenAQ download and the local CSV/JSON cache."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import TypedDict

import pandas as pd
import streamlit as st
from openaq import OpenAQ
from openaq.core.exceptions import HTTPRateLimitError, OpenAQError

from utils.config import (
    CITY_COORDINATES,
    DATA_CACHE_DIR,
    DATA_CACHE_TTL,
    DEFAULT_CITY,
    DISPLAY_NAMES,
    LIVE_STATION_MAX_AGE,
    LOOKBACK_DAYS,
    SEARCH_RADIUS_METERS,
    STATION_CACHE_TTL,
)
from utils.data_cleaning import clean_data


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


def offline_station_for_data(
    data: pd.DataFrame, station_id: int, name: str
) -> StationInfo:
    """Build station metadata for a locally cached measurement file."""
    sensors = {
        column: (index + 1, "")
        for index, column in enumerate(data.columns)
        if column != "datetime"
    }
    return {
        "id": station_id,
        "name": f"{name} (offline cached data)",
        "city": DEFAULT_CITY,
        "sensors": sensors,
        "sensor_count": len(sensors),
        "last_seen": datetime.now(timezone.utc),
        "null_rate": float(data.drop(columns=["datetime"]).isna().mean().mean()),
        "observed_rows": len(data),
    }


def load_offline_default_data() -> (
    tuple[pd.DataFrame, dict[str, str], str, str, StationInfo, Path]
):
    """Select the best valid Pune measurement CSV from data_cache."""
    station_metadata = {
        station["id"]: station for station in (read_station_cache(DEFAULT_CITY) or [])
    }
    candidates: list[
        tuple[tuple[float, float, int], pd.DataFrame, StationInfo, Path]
    ] = []
    for path in sorted(DATA_CACHE_DIR.glob("pune_*.csv")):
        try:
            data = clean_data(pd.read_csv(path, parse_dates=["datetime"]))
            numeric = data.drop(columns=["datetime"], errors="ignore").apply(
                pd.to_numeric, errors="coerce"
            )
            if data.empty or numeric.empty or not numeric.notna().any().any():
                continue
            data = data.rename(columns={"relative_humidity": "relativehumidity"})
            match = re.match(r"pune_(\d+)_", path.name)
            station_id = int(match.group(1)) if match else 0
            metadata = station_metadata.get(station_id)
            station = offline_station_for_data(
                data,
                station_id,
                metadata["name"] if metadata else f"Pune station {station_id}",
            )
            null_rate = float(numeric.isna().mean().mean())
            valid_values = int(numeric.notna().sum().sum())
            latest_timestamp = data["datetime"].max().timestamp()
            # Offline mode must prefer current observations over a more
            # complete but obsolete station cache.
            score = (-latest_timestamp, null_rate, -valid_values)
            candidates.append((score, data, station, path))
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            continue
    if not candidates:
        raise RuntimeError("No valid Pune CSV data was found in data_cache.")
    _, data, station, path = min(candidates, key=lambda item: item[0])
    units = {column: "" for column in data.columns if column != "datetime"}
    return data, units, DEFAULT_CITY, station["name"], station, path


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
        newest_cached = max(
            (station["last_seen"] for station in (cached_stations or [])),
            default=datetime.min.replace(tzinfo=timezone.utc),
        )
        if (
            cached_stations is not None
            and datetime.now(timezone.utc) - newest_cached <= LIVE_STATION_MAX_AGE
        ):
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
    cutoff = datetime.now(timezone.utc) - LIVE_STATION_MAX_AGE
    for location in locations:
        sensors = sensor_parameters(location)
        last_seen = station_last_seen(location)
        if sensors and last_seen >= cutoff:
            candidates.append(
                {
                    "id": location.id,
                    "name": location.name,
                    "city": city,
                    "sensors": sensors,
                    "sensor_count": len(sensors),
                    "last_seen": last_seen,
                    "null_rate": 1.0,
                    "observed_rows": 0,
                }
            )

    if not candidates:
        raise RuntimeError(
            f"No active OpenAQ station near {city} has reported within the "
            f"last {LIVE_STATION_MAX_AGE.days} days."
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
        raise RuntimeError(
            f"Could not evaluate stations near {city}: {error}"
        ) from error

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
        and datetime.now(timezone.utc)
        - datetime.fromtimestamp(cache_path.stat().st_mtime, timezone.utc)
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
                raise RuntimeError(
                    f"No metadata found for OpenAQ location {location_id}."
                )

            location = location_response.results[0]
            station_name = location.name
            sensors = sensor_parameters(location)
            if not sensors:
                raise RuntimeError(f"No supported sensors found for {station_name}.")

            for parameter, (sensor_id, unit) in sensors.items():
                # Query through now. Sensor metadata can lag behind the
                # measurements endpoint, so datetime_last is not a safe cap.
                query_start = end - timedelta(days=days_back)
                frame = fetch_sensor_measurements(
                    client, sensor_id, parameter, query_start, end
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

