"""Project-wide constants: cache settings, city list and variable names."""

from datetime import timedelta
from pathlib import Path

# data_cache/ sits next to app.py (one level above this utils/ package).
DATA_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
DATA_CACHE_TTL = timedelta(hours=1)
STATION_CACHE_TTL = timedelta(days=90)
LIVE_STATION_MAX_AGE = timedelta(days=14)
LOOKBACK_DAYS = 90
LOOKBACK_OPTIONS = (30, 90, 180, 365)
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
    "relative_humidity": "Relative Humidity",
}


def display_name(value: str) -> str:
    """Return a readable label for an API parameter."""
    return DISPLAY_NAMES.get(value, value)
