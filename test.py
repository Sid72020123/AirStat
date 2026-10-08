import math
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openaq import OpenAQ
from openaq.core.exceptions import OpenAQError


load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not found. Put it in your .env file.")


PUNE_COORDINATES = (18.5204, 73.8567)
SEARCH_RADIUS_METERS = 24_000
LOOKBACK_DAYS = 30
MAX_STATIONS_TO_TEST = 6

# These are the variables used by the mid-sem statistical and AQI analysis.
REQUIRED_PARAMETERS = {
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
    "temperature",
    "relativehumidity",
}


def sensor_map(location):
    """Return one sensor ID for each required parameter at a location."""
    sensors = {}
    for sensor in location.sensors:
        parameter = sensor.parameter.name
        if parameter in REQUIRED_PARAMETERS and parameter not in sensors:
            sensors[parameter] = sensor.id
    return sensors


def measure_completeness(client, location, sensors):
    """Measure hourly coverage over the common 30-day station window."""
    if location.datetime_last is None:
        return None

    end = datetime.fromisoformat(location.datetime_last.utc.replace("Z", "+00:00"))
    start = end - timedelta(days=LOOKBACK_DAYS)
    expected_hours = math.ceil((end - start).total_seconds() / 3600)
    counts = {}

    for parameter, sensor_id in sensors.items():
        try:
            response = client.measurements.list(
                sensors_id=sensor_id,
                data="hours",
                datetime_from=start,
                datetime_to=end,
                limit=1000,
            )
        except OpenAQError as error:
            print(f"  Could not read {parameter} for {location.name}: {error}")
            return None

        # Count only rows with an actual numeric value, not merely timestamps.
        counts[parameter] = sum(
            measurement.value is not None for measurement in response.results
        )

    total_expected = expected_hours * len(REQUIRED_PARAMETERS)
    total_observed = sum(counts.values())
    missing = total_expected - total_observed

    return {
        "start": start,
        "end": end,
        "expected_hours": expected_hours,
        "counts": counts,
        "observed": total_observed,
        "missing": missing,
        "completeness": total_observed / total_expected,
    }


with OpenAQ(api_key=API_KEY) as client:
    locations = client.locations.list(
        coordinates=PUNE_COORDINATES,
        radius=SEARCH_RADIUS_METERS,
        limit=100,
    ).results

    candidates = []
    for location in locations:
        sensors = sensor_map(location)
        if REQUIRED_PARAMETERS <= sensors.keys() and location.datetime_last is not None:
            candidates.append((location, sensors))

    # Test the newest complete stations first, keeping API usage bounded.
    candidates.sort(
        key=lambda item: item[0].datetime_last.utc,
        reverse=True,
    )
    candidates = candidates[:MAX_STATIONS_TO_TEST]

    if not candidates:
        raise RuntimeError(
            "No Pune station has all required parameters: "
            + ", ".join(sorted(REQUIRED_PARAMETERS))
        )

    results = []
    for location, sensors in candidates:
        print(f"Checking {location.name} (ID {location.id})...")
        coverage = measure_completeness(client, location, sensors)
        if coverage is not None:
            results.append((location, sensors, coverage))


if not results:
    raise RuntimeError("The candidate stations could not be measured.")

results.sort(
    key=lambda item: (item[2]["completeness"], item[2]["observed"]),
    reverse=True,
)

print("\n" + "=" * 78)
print("PUNE STATION COMPLETENESS RANKING")
print("=" * 78)

for rank, (location, _, coverage) in enumerate(results, start=1):
    print(
        f"{rank}. {location.name} (ID {location.id}) - "
        f"{coverage['completeness']:.2%} complete, "
        f"{coverage['missing']} missing values "
        f"({coverage['start'].date()} to {coverage['end'].date()})"
    )

best_location, best_sensors, best_coverage = results[0]

print("\n" + "=" * 78)
print("RECOMMENDED STATION FOR MID-SEM REVIEW")
print("=" * 78)
print(f"Station : {best_location.name}")
print(f"ID      : {best_location.id}")
print(f"Period  : {best_coverage['start'].date()} to {best_coverage['end'].date()}")
print(f"Coverage: {best_coverage['completeness']:.2%}")
print(f"Missing : {best_coverage['missing']} of "
      f"{best_coverage['expected_hours'] * len(REQUIRED_PARAMETERS)} values")
print("\nSensor IDs:")
for parameter in sorted(best_sensors):
    print(f"  {parameter}: {best_sensors[parameter]}")

print("\nValues by parameter:")
for parameter in sorted(best_coverage["counts"]):
    print(f"  {parameter}: {best_coverage['counts'][parameter]}")
