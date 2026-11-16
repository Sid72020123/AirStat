"""Unit I: data exploration and statistical foundations.

Tabs follow the syllabus order:
  Introduction · Data Types · Collection & Study Types · Sampling Techniques ·
  Cleaning & Preprocessing · Descriptive Statistics · EDA ·
  Covariance & Correlation · Predictive Analytics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.common import AppContext, correlation_interpretation, metric_cards
from components.navigation import go_to_section
from utils.aqi_calculator import aqi_category
from utils.config import display_name
from utils.data_cleaning import iqr_outlier_bounds, quality_summary, valid_series

LOCAL_TIMEZONE = "Asia/Kolkata"   # readings are stored in UTC; Indian hours are UTC+5:30


# --------------------------------------------------------------------------- #
# Calculations
# --------------------------------------------------------------------------- #
def descriptive_statistics(series: pd.Series) -> pd.DataFrame:
    """Calculate the Unit I descriptive statistics for one variable."""
    values = series.dropna()
    if values.empty:
        return pd.DataFrame(columns=["Statistic", "Value"])

    mode = values.mode()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    mean = values.mean()
    # Coefficient of variation = SD / mean (spread relative to the average);
    # skewness > 0 means a long right tail; kurtosis is "excess" (normal = 0).
    cv = f"{values.std() / mean:.4g}" if mean != 0 else "n/a"
    return pd.DataFrame(
        {
            "Statistic": [
                "Mean",
                "Median",
                "Mode",
                "Variance",
                "Standard Deviation",
                "Minimum",
                "Maximum",
                "Range",
                "Q1",
                "Q3",
                "IQR",
                "Coefficient of Variation",
                "Skewness",
                "Excess Kurtosis",
            ],
            "Value": [
                f"{values.mean():.4g}",
                f"{values.median():.4g}",
                ", ".join(f"{value:g}" for value in mode),
                f"{values.var():.4g}",
                f"{values.std():.4g}",
                f"{values.min():.4g}",
                f"{values.max():.4g}",
                f"{values.max() - values.min():.4g}",
                f"{q1:.4g}",
                f"{q3:.4g}",
                f"{q3 - q1:.4g}",
                cv,
                f"{values.skew():.4g}",
                f"{values.kurt():.4g}",
            ],
        }
    )


# How each AirStat variable is classified (type, scale, role).
VARIABLE_KINDS = {
    "pm25": ("Quantitative – continuous", "Ratio", "Pollutant concentration"),
    "pm10": ("Quantitative – continuous", "Ratio", "Pollutant concentration"),
    "no2": ("Quantitative – continuous", "Ratio", "Pollutant concentration"),
    "so2": ("Quantitative – continuous", "Ratio", "Pollutant concentration"),
    "o3": ("Quantitative – continuous", "Ratio", "Pollutant concentration"),
    "co": ("Quantitative – continuous", "Ratio", "Pollutant concentration"),
    "temperature": ("Quantitative – continuous", "Interval", "Weather"),
    "relativehumidity": ("Quantitative – continuous", "Ratio", "Weather"),
    "relative_humidity": ("Quantitative – continuous", "Ratio", "Weather"),
    "windspeed": ("Quantitative – continuous", "Ratio", "Weather"),
}


def sampling_frame(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Valid readings (time-ordered) that a sample can be drawn from."""
    frame = data[["datetime", column]].copy()
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("datetime").reset_index(drop=True)
    local = frame["datetime"].dt.tz_convert(LOCAL_TIMEZONE)
    frame["day"] = local.dt.date
    frame["period"] = pd.cut(
        local.dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["Night (00–05)", "Morning (06–11)", "Afternoon (12–17)", "Evening (18–23)"],
    )
    return frame


def draw_sample(frame: pd.DataFrame, technique: str, size: int, seed: int) -> pd.DataFrame:
    """Draw one sample with the chosen technique (reproducible through the seed)."""
    rng = np.random.default_rng(seed)
    population = len(frame)
    size = min(size, population)
    if technique == "Simple random":
        # every row has the same chance of selection
        return frame.iloc[np.sort(rng.choice(population, size, replace=False))]
    if technique == "Systematic":
        # every k-th row after a random start
        step = max(population // size, 1)
        start = int(rng.integers(step))
        return frame.iloc[np.arange(start, population, step)[:size]]
    if technique == "Stratified":
        # sample inside each time-of-day group, in proportion to its size
        parts = []
        for _, group in frame.groupby("period", observed=True):
            take = max(1, int(round(size * len(group) / population)))
            parts.append(group.iloc[np.sort(rng.choice(len(group), min(take, len(group)), replace=False))])
        return pd.concat(parts).sort_values("datetime")
    # Cluster: pick whole days at random until the sample is big enough
    days = frame["day"].unique()
    rng.shuffle(days)
    chosen, count = [], 0
    for day in days:
        chosen.append(day)
        count += int((frame["day"] == day).sum())
        if count >= size:
            break
    return frame[frame["day"].isin(chosen)]


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_introduction(ctx: AppContext) -> None:
    st.subheader("Introduction to data science and statistics")
    st.markdown(
        "**Data science** turns raw data into decisions by combining statistics, programming and "
        "domain knowledge. **Statistics** supplies the tools to describe data, measure uncertainty "
        "and test claims. AirStat applies both to hourly air-quality readings from Indian cities."
    )
    columns = st.columns(3)
    columns[0].markdown(
        "##### 1 · Describe\n*What does the data look like?*\n\nMean, spread, charts, correlation "
        "(this unit)."
    )
    columns[1].markdown(
        "##### 2 · Infer\n*What can a sample tell us about the whole?*\n\nProbability (Unit II), "
        "confidence intervals and tests (Unit III)."
    )
    columns[2].markdown(
        "##### 3 · Predict\n*What will a new value probably be?*\n\nRegression and model "
        "evaluation (Unit IV)."
    )
    st.markdown("##### The data-science cycle in AirStat")
    st.markdown(
        "`Collect` → `Clean` → `Explore` → `Analyse` → `Model` → `Evaluate` → `Communicate`  \n"
        "The full workflow is shown with live values in **Unit IV → 7 · Analytics Workflow**."
    )
    metric_cards(
        [
            ("Rows in dataset", f"{len(ctx.data):,}"),
            ("Variables", f"{len(ctx.variables)}"),
            ("First reading", f"{ctx.first_date}"),
            ("Last reading", f"{ctx.last_date}"),
        ]
    )
    st.caption(
        "A *population* is every reading that could ever be taken at the station; the downloaded "
        "rows are a *sample* of it. Descriptive statistics summarise the sample; inferential statistics "
        "(Unit III) reason from the sample to the population."
    )


def tab_data_types(ctx: AppContext) -> None:
    st.subheader("Types of data and variables")
    rows = []
    for column in ctx.variables:
        kind, scale, role = VARIABLE_KINDS.get(
            column, ("Quantitative – continuous", "Ratio", "Measured variable")
        )
        sample = valid_series(ctx.data, column)
        rows.append(
            {
                "Variable": display_name(column),
                "Unit": ctx.units.get(column, "") or "—",
                "Type": kind,
                "Scale": scale,
                "Role": role,
                "Example value": f"{sample.iloc[-1]:.4g}" if not sample.empty else "—",
            }
        )
    rows.insert(
        0,
        {"Variable": "datetime", "Unit": "UTC", "Type": "Temporal", "Scale": "Interval",
         "Role": "Time index", "Example value": str(ctx.data["datetime"].iloc[-1])[:16]},
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Quantitative (numerical)")
        st.markdown(
            "- **Continuous:** can take any value in a range, e.g. PM2.5 = 41.7 µg/m³.\n"
            "- **Discrete:** counted whole numbers, e.g. *hours per day above a limit* (used in Unit II's Poisson tab)."
        )
    with right:
        st.markdown("##### Qualitative (categorical)")
        st.markdown(
            "- **Nominal:** names without order, e.g. city, station.\n"
            "- **Ordinal:** ordered categories, e.g. the AQI category below."
        )
    st.markdown("##### Ordinal example: CPCB AQI categories")
    st.dataframe(
        pd.DataFrame(
            {
                "AQI value": [25, 75, 150, 250, 350, 450],
                "Category (ordered)": [aqi_category(v) for v in (25, 75, 150, 250, 350, 450)],
            }
        ),
        hide_index=True,
    )
    with st.expander("Measurement scales (NOIR) explained"):
        st.markdown(
            "| Scale | Property | AirStat example |\n|---|---|---|\n"
            "| Nominal | Labels only | Station name |\n"
            "| Ordinal | Order matters, gaps do not | AQI category |\n"
            "| Interval | Equal gaps, **no true zero** | Temperature in °C (0 °C is not 'no heat') |\n"
            "| Ratio | Equal gaps **and** true zero | PM2.5, humidity, wind speed (0 means none) |\n\n"
            "The scale decides which statistics make sense: ratios such as 'twice as polluted' are "
            "meaningful for PM2.5 but not for °C."
        )


def tab_collection(ctx: AppContext) -> None:
    st.subheader("Data collection methods and study types")
    times = ctx.data["datetime"].sort_values()
    gap_hours = times.diff().dropna().dt.total_seconds().median() / 3600 if len(times) > 1 else float("nan")
    metric_cards(
        [
            ("Source", "OpenAQ"),
            ("Station", ctx.station),
            ("Typical interval", f"{gap_hours:.1f} h" if np.isfinite(gap_hours) else "n/a"),
            ("Rows collected", f"{ctx.data.attrs.get('rows_before_cleaning', len(ctx.data)):,}"),
        ]
    )
    st.markdown("##### How AirStat collects its data")
    st.markdown(
        "1. **Automated sensor measurement** – monitoring stations record pollutant and weather values every hour.\n"
        "2. **Secondary data via an API** – AirStat does not run the sensors; it downloads readings that "
        "another organisation (OpenAQ) already collected.\n"
        "3. **Local cache** – downloaded data is saved in `data_cache/` and reused, so the API is called only when "
        "needed and the dashboard still works offline."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Method": ["Primary data", "Secondary data", "Survey / questionnaire", "Experiment", "Sensor / automated logging"],
                "Meaning": [
                    "Collected by you for your own question",
                    "Collected by someone else, reused",
                    "Asking people questions",
                    "Controlling a factor and measuring the result",
                    "Instruments record values automatically",
                ],
                "Used in AirStat?": ["No", "**Yes** (OpenAQ)", "No", "No", "**Yes** (station sensors)"],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown("##### Observational vs experimental studies")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Observational study**  \nResearchers *watch* what happens and record it, without changing anything.\n\n"
            "- Can reveal **associations** (e.g. PM2.5 and PM10 move together)\n"
            "- Cannot prove cause and effect, because other factors may differ"
        )
    with right:
        st.markdown(
            "**Experimental study**  \nResearchers *assign* a treatment to some units and compare with a control, "
            "ideally at random.\n\n"
            "- Can support **causal** conclusions\n"
            "- Example: randomly closing roads on some days and measuring the PM2.5 change"
        )
    st.info(
        "**AirStat is observational.** The stations record air as it is; nothing is manipulated. "
        "That is why every regression and correlation result in this dashboard is described as an "
        "association, not a cause."
    )


def tab_sampling(ctx: AppContext) -> None:
    st.subheader("Sampling techniques")
    st.write(
        "A **sample** is a subset chosen to represent the **population** (all valid readings of the selected "
        "variable). Try each technique and compare the sample with the population."
    )
    frame = sampling_frame(ctx.analysis_data, ctx.pollutant)
    if len(frame) < 30:
        st.warning("At least 30 valid readings are needed for the sampling demonstration.")
        return
    controls = st.columns(3)
    technique = controls[0].selectbox(
        "Sampling technique", ["Simple random", "Systematic", "Stratified", "Cluster"], key="u1_sampling_technique"
    )
    size = controls[1].slider("Sample size", 10, min(500, len(frame)), min(100, len(frame)), key="u1_sampling_size")
    seed = controls[2].number_input("Random seed", 0, 9999, 42, key="u1_sampling_seed",
                                    help="Change the seed to draw a different sample; the same seed repeats it.")
    sample = draw_sample(frame, technique, size, int(seed))
    column = ctx.pollutant
    population_mean, sample_mean = float(frame[column].mean()), float(sample[column].mean())
    metric_cards(
        [
            ("Population mean", f"{population_mean:.4g}"),
            ("Sample mean", f"{sample_mean:.4g}"),
            ("Difference (sampling error)", f"{sample_mean - population_mean:+.3g}"),
            ("Sample size", f"{len(sample)}"),
        ]
    )
    plot_data = pd.concat(
        [
            frame.assign(Group="Population")[[column, "Group"]],
            sample.assign(Group="Sample")[[column, "Group"]],
        ]
    )
    st.plotly_chart(
        px.histogram(plot_data, x=column, color="Group", barmode="overlay", histnorm="probability density",
                     opacity=0.6, nbins=30, labels={column: ctx.label(column)},
                     title=f"Population vs {technique.lower()} sample"),
        width="stretch",
    )
    if technique == "Stratified":
        allocation = (
            frame.groupby("period", observed=True)[column].size().to_frame("Population rows")
            .join(sample.groupby("period", observed=True)[column].size().rename("Sample rows"))
        )
        st.dataframe(allocation, width="stretch")
        st.caption("Each time-of-day stratum contributes in proportion to its share of the population.")
    if technique == "Cluster":
        st.caption(f"Whole days were chosen at random: {sample['day'].nunique()} days, "
                   f"{len(sample)} rows (sample size is approximate because days differ in size).")
    with st.expander("How the four techniques differ", expanded=True):
        st.markdown(
            "| Technique | How it works | Good when | Watch out for |\n|---|---|---|---|\n"
            "| **Simple random** | Every reading equally likely to be chosen | Population is fairly uniform | Can miss rare periods by chance |\n"
            "| **Systematic** | Every k-th reading after a random start | Data is ordered and simple to sample | Hidden cycles (e.g. a 24-hour pattern when k = 24) bias the sample |\n"
            "| **Stratified** | Sample inside groups (night, morning, ...) in proportion | Groups differ a lot | Needs a sensible grouping variable |\n"
            "| **Cluster** | Pick whole groups (days) at random | Groups are natural and costly to split | Readings within a day are similar, so precision is lower |"
        )
        st.caption("Because air quality changes through the day, stratified sampling often tracks the population mean most closely.")


def tab_cleaning(ctx: AppContext) -> None:
    st.subheader("Data cleaning and preprocessing")
    st.markdown(
        "Raw sensor data is never analysis-ready. AirStat's `clean_data` step already:\n"
        "1. converts timestamps to UTC date-times and drops unreadable ones,\n"
        "2. sorts rows by time and removes **duplicate timestamps**,\n"
        "3. converts every measurement to numbers (invalid text becomes *missing*)."
    )
    metric_cards(
        [
            ("Rows before cleaning", f"{ctx.data.attrs.get('rows_before_cleaning', len(ctx.data)):,}"),
            ("Duplicate timestamps removed", f"{ctx.data.attrs.get('duplicate_timestamps', 0)}"),
            ("Rows after cleaning", f"{len(ctx.data):,}"),
        ]
    )
    st.markdown("##### Missing values and zeros by variable (selected date range)")
    summary = quality_summary(ctx.filtered, ctx.variables)
    summary["Variable"] = summary["Variable"].map(display_name)
    st.dataframe(
        summary.style.format({"Missing %": "{:.1f}%"}).background_gradient(subset=["Missing %"], cmap="Reds", vmin=0, vmax=100),
        hide_index=True,
        width="stretch",
    )
    st.caption("Zeros are **not** deleted automatically: a real zero can be a valid reading. Only zeros that match "
               "error patterns are flagged as 'suspicious' (sidebar checkbox decides whether to exclude them).")

    values = valid_series(ctx.analysis_data, ctx.pollutant)
    if len(values) < 4:
        st.info("Not enough valid values for outlier and missing-value demonstrations.")
        return
    st.markdown(f"##### Outliers in {display_name(ctx.pollutant)} (IQR rule)")
    low, high, n_low, n_high = iqr_outlier_bounds(values)
    metric_cards(
        [("Lower fence", f"{low:.4g}"), ("Upper fence", f"{high:.4g}"),
         ("Low outliers", f"{n_low}"), ("High outliers", f"{n_high}")]
    )
    st.caption("Outliers are values beyond Q1 − 1.5·IQR or Q3 + 1.5·IQR. They are *flagged, not removed*: a pollution "
               "spike can be real and important.")

    st.markdown("##### How different missing-value strategies change the result")
    hourly = (
        ctx.analysis_data.set_index("datetime")[ctx.pollutant].apply(pd.to_numeric, errors="coerce").resample("1h").mean()
    )
    strategies = {
        "Drop missing hours (used by the dashboard)": hourly.dropna(),
        "Linear interpolation (gaps ≤ 3 h)": hourly.interpolate(method="time", limit=3).dropna(),
        "Forward fill (gaps ≤ 3 h)": hourly.ffill(limit=3).dropna(),
        "Fill with median": hourly.fillna(hourly.median()),
    }
    comparison = pd.DataFrame(
        {name: {"Hours kept": len(series), "Mean": series.mean(), "Std. deviation": series.std()}
         for name, series in strategies.items()}
    ).T
    st.dataframe(comparison.style.format({"Hours kept": "{:.0f}", "Mean": "{:.4g}", "Std. deviation": "{:.4g}"}),
                 width="stretch")
    st.caption(f"{int(hourly.isna().sum())} of {len(hourly)} hourly slots are empty. Filling gaps invents data, so the "
               "analysis pages simply drop rows with missing values and report how many remain.")


def tab_descriptive(ctx: AppContext) -> None:
    st.subheader(f"Measures of central tendency and dispersion — {display_name(ctx.pollutant)}")
    values = valid_series(ctx.analysis_data, ctx.pollutant)
    if values.empty:
        st.warning("No values are available for this variable and date range.")
        return
    mode = values.mode()
    metric_cards(
        [
            ("Mean", f"{values.mean():.4g}"),
            ("Median", f"{values.median():.4g}"),
            ("Mode", f"{mode.iloc[0]:.4g}" if not mode.empty else "n/a"),
            ("Std. deviation", f"{values.std():.4g}"),
            ("IQR", f"{values.quantile(0.75) - values.quantile(0.25):.4g}"),
        ],
        ["Arithmetic average.", "Middle value; robust to outliers.", "Most frequent value.",
         "Typical distance from the mean.", "Spread of the middle 50% of values."],
    )
    st.dataframe(descriptive_statistics(ctx.analysis_data[ctx.pollutant]), hide_index=True)
    if values.mean() > values.median():
        st.info("Mean > median: a few high readings pull the average up (right-skewed). The median describes a "
                "'typical' reading better.")
    elif values.mean() < values.median():
        st.info("Mean < median: a few low readings pull the average down (left-skewed).")
    with st.expander("Which measure should I use?"):
        st.markdown(
            "- **Central tendency** – *mean* (uses every value, sensitive to extremes), *median* (middle value, robust), "
            "*mode* (most common value).\n"
            "- **Dispersion** – *range* (max − min, only two values), *variance / standard deviation* (average distance from the "
            "mean), *IQR* (spread of the middle half, robust), *coefficient of variation* (spread relative to the mean, to compare variables).\n"
            "- **Shape** – *skewness* (asymmetry) and *kurtosis* (heaviness of the tails)."
        )
        st.latex(r"s^2=\frac{1}{n-1}\sum_{i}(x_i-\bar x)^2,\qquad s=\sqrt{s^2},\qquad CV=\frac{s}{\bar x}")


def tab_eda(ctx: AppContext) -> None:
    st.subheader(f"Exploratory data analysis — {display_name(ctx.pollutant)}")
    values = valid_series(ctx.analysis_data, ctx.pollutant)
    if values.empty:
        st.info("Choose a date range containing measurements to display charts.")
        return
    line_data = ctx.analysis_data[["datetime", ctx.pollutant]].dropna()
    st.plotly_chart(
        px.line(
            line_data,
            x="datetime",
            y=ctx.pollutant,
            labels={ctx.pollutant: f"{display_name(ctx.pollutant)} ({ctx.units.get(ctx.pollutant, '')})"},
            title="Time series",
        ),
        width="stretch",
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(px.histogram(line_data, x=ctx.pollutant, title="Distribution (histogram)"), width="stretch")
    with chart_right:
        st.plotly_chart(px.box(line_data, y=ctx.pollutant, title="Distribution (box plot)"), width="stretch")
    # Typical daily pattern: average by local hour of day
    local_hour = line_data["datetime"].dt.tz_convert(LOCAL_TIMEZONE).dt.hour
    profile = line_data.groupby(local_hour)[ctx.pollutant].mean().rename_axis("Hour of day (IST)").reset_index()
    st.plotly_chart(
        px.line(profile, x="Hour of day (IST)", y=ctx.pollutant, markers=True,
                title="Average by hour of day (daily pattern)"),
        width="stretch",
    )
    st.caption("EDA asks: what is the shape, are there trends or cycles, and are there odd values? "
               "The answers guide every later test and model.")


def tab_correlation(ctx: AppContext) -> None:
    data, variables = ctx.analysis_data, ctx.variables
    variable_x, variable_y = ctx.variable_x, ctx.variable_y
    comparison_label = f"{ctx.label(variable_x)} vs {ctx.label(variable_y)}"
    st.subheader(f"Covariance and correlation — {comparison_label}")
    st.caption("Variable X and Variable Y are chosen in the sidebar. The full correlation lab (heatmap, "
               "interactive scatter) is in Unit IV.")
    paired = data[[variable_x, variable_y]].dropna()
    if variable_x == variable_y:
        st.warning("Choose two different variables for relationship analysis.")
    elif len(paired) < 2:
        st.warning("At least two paired observations are needed for this analysis.")
    else:
        covariance = paired[variable_x].cov(paired[variable_y])
        pearson = paired[variable_x].corr(paired[variable_y], method="pearson")
        spearman = paired[variable_x].corr(paired[variable_y], method="spearman")
        metrics = st.columns(3)
        metrics[0].metric(f"Covariance ({comparison_label})", f"{covariance:.4g}")
        metrics[1].metric(f"Pearson correlation ({comparison_label})", f"{pearson:.4f}")
        metrics[2].metric(f"Spearman correlation ({comparison_label})", f"{spearman:.4f}")
        st.info(f"Pearson interpretation: {correlation_interpretation(pearson)}")
        scatter = paired.rename(columns={variable_x: display_name(variable_x), variable_y: display_name(variable_y)})
        st.plotly_chart(
            px.scatter(scatter, x=display_name(variable_x), y=display_name(variable_y), title="Scatter plot"),
            width="stretch",
        )
    with st.expander("Covariance, Pearson and Spearman"):
        st.latex(r"\operatorname{cov}(x,y)=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{n-1},\qquad "
                 r"r=\frac{\operatorname{cov}(x,y)}{s_x s_y}")
        st.markdown(
            "- **Covariance** shows the direction of the joint movement, but its size depends on the units.\n"
            "- **Pearson r** rescales covariance to −1…+1 (straight-line relationships).\n"
            "- **Spearman ρ** is Pearson's r applied to the *ranks*; it detects any steadily rising/falling pattern and resists outliers."
        )

    st.markdown("##### Relationship matrix")
    matrix_options = {"Covariance": "covariance", "Pearson correlation": "pearson", "Spearman correlation": "spearman"}
    matrix_choice = st.selectbox(
        "Choose a relationship measure",
        list(matrix_options),
        help=("Select a measure to compare every pair of numerical variables. "
              "The diagonal shows each variable compared with itself."),
    )
    matrix_data = data[[column for column in variables if column in data]].apply(pd.to_numeric, errors="coerce")
    if len(matrix_data.columns) < 2:
        st.info("At least two numerical variables are needed for a relationship matrix.")
        return
    method = matrix_options[matrix_choice]
    if method == "covariance":
        relationship_matrix, color_range = matrix_data.cov(), None
    elif method == "pearson":
        relationship_matrix, color_range = matrix_data.corr(method="pearson"), (-1, 1)
    else:
        relationship_matrix, color_range = matrix_data.corr(method="spearman"), (-1, 1)
    relationship_matrix = relationship_matrix.rename(index=display_name, columns=display_name)
    styled_matrix = relationship_matrix.style.format("{:.3f}").background_gradient(
        cmap="RdYlGn",
        vmin=color_range[0] if color_range else None,
        vmax=color_range[1] if color_range else None,
    )
    st.caption(f"{matrix_choice} for all numerical variables.")
    st.dataframe(styled_matrix, width="stretch")


def tab_predictive(ctx: AppContext) -> None:
    st.subheader("Introduction to predictive analytics")
    st.markdown(
        "**Predictive analytics** uses patterns in past data to estimate an unknown value. In AirStat the question is, "
        "for example, *\"If PM10 is 80 µg/m³ right now, what PM2.5 should I expect?\"*"
    )
    columns = st.columns(3)
    columns[0].markdown("##### Regression\nPredict a **number** (PM2.5 tomorrow).")
    columns[1].markdown("##### Classification\nPredict a **category** (will the AQI be 'Poor'?).")
    columns[2].markdown("##### Forecasting\nPredict the **future** of a time series.")
    st.markdown(
        "**Core ideas:** *features* (inputs, X) and a *target* (output, Y); a model **learns** from training rows and is "
        "**tested** on rows it never saw; good predictions need a relationship in the data, but a relationship is not a cause."
    )

    st.markdown("##### Mini demo: does knowing X help predict Y?")
    x_column, y_column = ctx.variable_x, ctx.variable_y
    pairs = ctx.analysis_data[["datetime", x_column, y_column]].apply(
        lambda s: pd.to_numeric(s, errors="coerce") if s.name != "datetime" else s).dropna()
    if x_column == y_column or len(pairs) < 30:
        st.info("Pick two different variables with at least 30 paired rows in the sidebar (Variable X and Variable Y).")
    else:
        shuffled = np.random.default_rng(42).permutation(len(pairs))
        n_test = max(5, int(0.2 * len(pairs)))
        test, train = pairs.iloc[shuffled[:n_test]], pairs.iloc[shuffled[n_test:]]
        slope, intercept = np.polyfit(train[x_column], train[y_column], 1)
        line_error = float(np.sqrt(np.mean((test[y_column] - (intercept + slope * test[x_column])) ** 2)))
        average_error = float(np.sqrt(np.mean((test[y_column] - train[y_column].mean()) ** 2)))
        metric_cards(
            [
                (f"Error if we always guess the average {display_name(y_column)}", f"{average_error:.4g}"),
                (f"Error when predicting from {display_name(x_column)}", f"{line_error:.4g}"),
                ("Improvement", f"{(1 - line_error / average_error):.0%}" if average_error > 0 else "n/a"),
            ]
        )
        st.caption("Errors are RMSE on 20% of the rows (fixed random split, seed 42) that were kept out of the fitting. "
                   "A useful model must beat the 'always guess the average' baseline.")
    st.info("Ready for the real thing? Unit IV builds simple and multiple regression models, checks their assumptions "
            "and evaluates them with MAE, MSE, RMSE and R².")
    st.button("Open Unit IV", on_click=go_to_section, args=("Unit IV",), key="u1_open_unit4")


def render(ctx: AppContext) -> None:
    st.header("Unit I — Data Exploration and Statistical Foundations")
    tabs = st.tabs(
        [
            "Introduction",
            "Data Types",
            "Collection & Study Types",
            "Sampling Techniques",
            "Cleaning & Preprocessing",
            "Descriptive Statistics",
            "EDA",
            "Covariance & Correlation",
            "Predictive Analytics",
        ]
    )
    for tab, function in zip(
        tabs,
        [tab_introduction, tab_data_types, tab_collection, tab_sampling, tab_cleaning,
         tab_descriptive, tab_eda, tab_correlation, tab_predictive],
    ):
        with tab:
            function(ctx)
