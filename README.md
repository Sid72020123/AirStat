# AirStat

AirStat is a beginner-friendly Streamlit dashboard for statistical analysis of
hourly air-quality measurements from OpenAQ monitoring stations across major
Indian cities. **Pune** is selected by default.

The project is intentionally limited to the mid-semester Unit I syllabus:

- data collection, cleaning, missing-value awareness, sorting, and duplicate handling
- mean, median, mode, variance, standard deviation, minimum, maximum, range, quartiles, and IQR
- covariance, Pearson correlation, and Spearman correlation
- a time-series chart, histogram, box plot, and scatter plot

AQI and machine-learning models are deliberately not included.

## Setup

1. Create and activate a Python virtual environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in this folder:

   ```text
   OPENAQ_API_KEY=your_64_character_openaq_key
   ```

   The key is read from the environment and is never stored in the Python code.

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

## Using the dashboard

Choose a city from the sidebar. AirStat searches OpenAQ within 25 km of that
city, downloads a short recent sample from each candidate, calculates the
percentage of missing values, and ranks stations from least missing data to
most missing data. The first station is selected automatically. You can also
choose another station from the **Monitoring station** dropdown if you want to
compare locations. No station ID is required. Pune's result is commonly a
current Pune IITM or MPCB station, depending on recent data quality.

After choosing a city, choose a date range and variable. The dashboard filters
the cleaned Pandas DataFrame, displays descriptive statistics, and updates the
charts and relationship analysis.

Station rankings are saved as JSON inside `data_cache/` for three months, so the
city and station dropdowns open quickly on later visits without repeating the
station-quality API checks. Cleaned measurements are saved as CSV files in the
same folder. A recent CSV is loaded locally on the next visit, making the
dashboard faster and avoiding repeated API requests. Measurement files older
than one hour are refreshed automatically. Use **Refresh Data** to force a
fresh station search and download. If OpenAQ rate-limits a request, AirStat
uses the latest downloaded Pune data when it is available and shows a warning
instead of exposing a traceback.

OpenAQ may not provide every pollutant or environmental variable at all times.
The dashboard uses the sensors actually available at the selected station and
shows a clear message when no measurements are returned. If the API key is
missing, invalid, expired, or OpenAQ is temporarily unavailable, the dashboard
automatically selects the best valid, non-empty Pune CSV from `data_cache/`.
It chooses the cached station with the lowest missing-data rate, then the most
valid values and latest data. This offline fallback keeps the dashboard usable
without showing an API traceback.

## Limitations

The dashboard uses one automatically selected station at a time and a recent
30-day window. Some cities may not have a station with recent data or all
supported variables. Correlation describes association only; it does not
establish causation. The dashboard does not calculate AQI, forecast pollution,
or train prediction models.
