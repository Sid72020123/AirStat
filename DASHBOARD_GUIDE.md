# AirStat Dashboard Guide

This guide explains what each AirStat dashboard component does, what statistical
analysis or inference it supports, and two examples of how it can be used.

AirStat uses hourly air-quality observations from an OpenAQ monitoring station.
Unless stated otherwise, results describe the selected station and selected date
range only. They should not automatically be interpreted as causal or
city-wide conclusions.

## 1. Data Selection

### Indian city

Selects the city around which AirStat searches for OpenAQ monitoring stations.
The station search uses a 25 km radius and ranks recent stations using data
completeness and recency.

### Live data lookback

Controls how much recent history is requested from OpenAQ: 30, 90, 180, or
365 days. A longer period provides more observations but may contain more
missing data and older station conditions.

### Refresh Data

Clears the relevant application caches, refreshes station metadata, and requests
new measurements. The newest timestamp still depends on when OpenAQ has received
and processed data from the station.

### Monitoring station

Allows a different nearby station to be selected. This is important because
stations can measure different pollutant combinations and can have different
missing-data rates.

### Date range

Restricts all displayed statistical analysis to the selected dates. Changing the
date range changes the sample used by descriptive statistics, charts,
correlations, probabilities, and the relationship matrix.

### Pollutant / variable

Selects the main variable for descriptive statistics and Unit II probability
analysis. Possible variables include PM2.5, PM10, O3, NO2, SO2, CO, temperature,
humidity, and wind speed when those sensors are available.

### Exceedance threshold

Defines the value used to classify an observation as an exceedance. Unit II uses
the rule:

```text
exceedance = selected variable > threshold
```

### Variable X and Variable Y

Select the two variables compared in the covariance, Pearson correlation,
Spearman correlation, conditional probability, and Bayes analyses.

### Suspicious-zero control

AirStat can flag possible sensor-quality issues such as isolated zeros, long
runs of zeros, or simultaneous zeros across variables. The checkbox controls
whether flagged zeros remain in calculations. A flag is a heuristic and does
not prove that a sensor malfunctioned.

### Examples

1. Select Pune, a 90-day lookback, PM2.5 as the main variable, and a threshold
   of 60 µg/m³ to study how often PM2.5 exceeded a chosen level.
2. Select PM2.5 as Variable X and temperature as Variable Y to investigate
   whether the two measurements move together during the selected period.

## 2. Data Quality Summary

The sidebar reports:

- valid numeric observations
- missing or invalid observations
- zero values
- suspicious-zero values

This is a data-quality diagnostic, not a pollution estimate. It helps determine
whether a result is based on a sufficiently complete sample.

### Statistical use

Missingness can reduce sample size and may bias results if missing observations
are not random. Suspicious zeros should be investigated before treating them as
real environmental measurements.

### Examples

1. If PM2.5 has 2,000 valid values and 20 missing values, descriptive statistics
   are based on a nearly complete sample.
2. If PM10 has many consecutive zeros, compare results with the checkbox enabled
   and disabled to see whether those readings materially change the conclusions.

## 3. Unit I — Descriptive Statistics

Unit I summarizes the selected station and selected date range. It is
descriptive: it reports patterns in the observed sample and does not by itself
prove that the same pattern exists at every station or on every day.

## 4. Dataset Overview

The overview reports the city, station, number of observations, selected
variable, measurement unit, and rows remaining after cleaning. Cleaning converts
values to numeric form, removes invalid timestamps, sorts observations, and
removes duplicate timestamps.

### Statistical use

These values establish the scope and size of the sample before interpreting
statistics. A result based on 20 observations should be treated less
confidently than a similar result based on 2,000 observations.

### Examples

1. Compare two stations with the same date range and check whether one has many
   fewer usable rows before comparing their pollution summaries.
2. Confirm that PM2.5 is measured in µg/m³ before comparing its mean with a
   published concentration limit.

## 5. Current CPCB AQI

The AQI panel calculates a simple CPCB AQI locally from the selected station's
pollutants. It does not call a second AQI API.

The calculation uses:

- 24-hour rolling averages for PM2.5, PM10, NO2, and SO2
- 8-hour rolling averages for O3 and CO
- CPCB breakpoint interpolation for each pollutant sub-index
- the highest valid pollutant sub-index as the overall AQI

At least three pollutant indices, including PM2.5 or PM10, are required.
Otherwise the panel reports insufficient data. The panel also shows the
individual pollutant sub-indices.

### Statistical use

AQI converts multiple pollutant concentrations into a health-communication
scale. It is useful for categorizing the latest sufficiently supported
measurement period, but it should not be treated as an average of pollutants or
as a causal health-risk estimate.

### Examples

1. If PM2.5 produces a sub-index of 180 and every other valid sub-index is
   lower, the overall AQI is 180 and the category is Moderate.
2. If only PM2.5 and NO2 are available, the dashboard reports insufficient data
   instead of presenting an unsupported overall AQI.

## 6. Descriptive Statistics

The descriptive-statistics table includes count, mean, median, mode, variance,
standard deviation, minimum, maximum, range, quartiles, and IQR for the selected
variable.

### Statistical use

- **Mean:** arithmetic average; sensitive to extreme values.
- **Median:** middle observation; more robust to skewed pollution data.
- **Mode:** most frequent value or values.
- **Variance and standard deviation:** spread around the mean.
- **Minimum and maximum:** observed extremes.
- **Range:** maximum minus minimum.
- **Quartiles and IQR:** central spread and resistance to outliers.

### Examples

1. If the mean PM2.5 is much higher than the median, the distribution may be
   right-skewed by a small number of pollution episodes.
2. A large IQR for NO2 indicates that the middle half of observations varies
   substantially, even if the minimum and maximum are not considered.

## 7. Exploratory Visualizations

### Time-series chart

Shows the selected variable against time.

**Analysis and inference:**

- identify changes, spikes, dips, and possible recurring patterns
- compare pollution episodes with dates or operational events

**Examples:**

1. A repeated PM2.5 increase every evening may suggest a recurring daily pattern.
2. A sudden isolated spike can identify a period worth checking for weather,
   traffic, construction, or sensor issues.

### Histogram

Shows how frequently values fall into numeric ranges.

**Analysis and inference:**

- assess skewness, concentration, multiple peaks, and unusual values
- decide whether mean-based summaries adequately represent the sample

**Examples:**

1. A long right tail in PM10 indicates occasional high-concentration events.
2. Two visible peaks in temperature may indicate observations from different
   daily or seasonal conditions.

### Box plot

Shows the median, quartiles, IQR, and potential outliers.

**Analysis and inference:**

- compare spread and central tendency quickly
- identify observations that deserve investigation

**Examples:**

1. A high upper whisker and several points above it may indicate exceptional
   pollution episodes.
2. Comparing box plots for PM2.5 and PM10 can show which pollutant has greater
   relative spread, while remembering that their units and scales differ.

## 8. Covariance and Correlation

This section compares the selected Variable X and Variable Y and labels both
variables in the section heading and metric names.

### Covariance

Measures whether two variables tend to increase or decrease together. Its
magnitude depends on the units, so covariance values are not directly comparable
across differently scaled variable pairs.

### Pearson correlation

Measures linear association from -1 to +1. Values near +1 indicate strong
positive linear association, values near -1 indicate strong negative linear
association, and values near 0 indicate weak linear association.

### Spearman correlation

Measures monotonic association using ranks. It is useful when the relationship is
consistently increasing or decreasing but not necessarily linear, and is often
less sensitive to extreme values than Pearson correlation.

### Statistical caution

Correlation is association, not causation. A third variable, time trend, weather
condition, or measurement issue may influence both variables.

### Examples

1. A positive PM2.5-temperature Pearson correlation means higher temperatures
   and PM2.5 tended to occur together in this sample; it does not prove that
   temperature caused the PM2.5 increase.
2. If Pearson correlation is near zero but Spearman correlation is strongly
   positive, the variables may have a nonlinear but consistently increasing
   relationship.

## 9. Data Quality Section

The main Data Quality section summarizes the number of downloaded rows,
missing values for the selected variable, suspicious zeros, excluded suspicious
zeros, duplicate timestamps removed, and rows remaining after cleaning.

### Statistical use

This section documents the reliability conditions behind the displayed results.
It can reveal whether an apparent pattern is based on a small or heavily
filtered sample.

### Examples

1. If duplicate timestamps were removed, the cleaned row count explains why it
   differs from the raw downloaded count.
2. If excluding suspicious zeros removes a large fraction of the selected
   variable, conclusions should be reported with that sensitivity clearly stated.

## 10. Relationship Matrix

The Relationship Matrix compares every numerical variable pair. The dropdown
offers:

- Covariance
- Pearson correlation
- Spearman correlation

Pearson and Spearman matrices use a fixed -1 to +1 color scale. Covariance uses
the actual covariance values, because covariance has unit-dependent magnitude.
The diagonal compares each variable with itself.

### Statistical use

The matrix is an exploratory screening tool. It helps identify pairs for a
focused analysis but does not establish causation or account for confounding
variables.

### Examples

1. A strong positive PM2.5-PM10 cell suggests that the two particulate measures
   often rise and fall together at the selected station.
2. A strong negative humidity-temperature cell suggests an inverse association
   in the selected sample, which could motivate a time-series or weather-adjusted
   analysis.

## 11. Unit II — Probability and Distributions

Unit II uses the selected variable, threshold, comparison variable, and cleaned
date range. It requires at least two valid observations.

### 11.1 Basic Probability

Calculates the fraction of valid observations above the threshold:

```text
P(A) = number of exceedances / number of valid observations
```

It also calculates conditional probability when Variable Y is available:

```text
P(A | B)
```

where A is the selected variable exceeding its threshold and B is Variable Y
exceeding the same threshold.

**Examples:**

1. If 120 of 1,000 PM2.5 observations exceed 60 µg/m³, the estimated
   exceedance probability is 12%.
2. If 40% of observations with high temperature also have high PM2.5, the
   estimated conditional probability is P(high PM2.5 | high temperature) = 40%.

### 11.2 Bayes Theorem

Computes the same conditional relationship using Bayes' theorem:

```text
P(A | B) = P(B | A) P(A) / P(B)
```

The direct and Bayes results should be nearly equal apart from rounding.

**Examples:**

1. Estimate the probability of high PM2.5 given high temperature using the
   observed base rate of high PM2.5 and the observed reverse conditional rate.
2. Compare P(high NO2 | high PM2.5) with P(high PM2.5 | high NO2) to show that
   conditional probabilities are directional and are not interchangeable.

### 11.3 Random Variables

Treats each valid measurement of the selected pollutant as an observed
continuous random variable and reports count, minimum, mean, standard deviation,
maximum, and sample values.

**Examples:**

1. Use the reported mean and standard deviation to describe the typical PM10
   level and its observed variability.
2. Compare the minimum and maximum with the IQR from Unit I to determine whether
   extreme values are far outside the central sample.

### 11.4 Binomial Distribution

Models the number of threshold exceedances in `n` trials using the observed
exceedance probability `p`:

```text
P(X = k) = C(n, k) p^k (1-p)^(n-k)
```

This is most appropriate when trials are treated as approximately independent
and have a stable probability.

**Examples:**

1. With p = 0.12 and n = 10 hourly observations, estimate the probability of
   exactly two PM2.5 exceedances.
2. With p = 0.30 and n = 20 daily observations, estimate the probability of at
   least a chosen number of high-pollution days by summing binomial outcomes.

### 11.5 Poisson Distribution

Counts daily threshold exceedances and estimates λ as the average daily count.
It then calculates the probability of a selected count on one day:

```text
P(X = k) = exp(-λ) λ^k / k!
```

**Examples:**

1. If the station averages two PM2.5 exceedances per day, estimate the
   probability of exactly one exceedance on a future comparable day.
2. Compare λ for PM2.5 and NO2 to identify which threshold is exceeded more
   frequently per day in the selected sample.

### 11.6 Normal Distribution

Fits a normal curve using the selected variable's sample mean and standard
deviation. The histogram is shown with the fitted probability-density curve.

**Examples:**

1. A roughly symmetric histogram with one central peak may be reasonably
   summarized by a normal approximation.
2. A strongly skewed histogram or multiple peaks indicates that the normal curve
   may be a poor model even if it is displayed for comparison.

### 11.7 Z-Score

Calculates how far an entered value is from the sample mean in standard-deviation
units:

```text
z = (value - mean) / standard deviation
```

**Examples:**

1. A PM2.5 value with z = 2 is two sample standard deviations above the mean
   and is unusually high relative to this dataset.
2. A temperature value with z = -1.5 is below the sample mean but is not
   automatically an abnormal real-world temperature.

### 11.8 Normal Approximation to Binomial

Compares an exact binomial probability with a normal approximation using a
continuity correction. It is useful for sufficiently large `n` when both
`n*p` and `n*(1-p)` are not too small.

**Examples:**

1. For a large number of trials, compare the exact probability of a chosen
   exceedance count with the normal approximation.
2. If the approximation differs substantially from the exact result, increase
   the sample size or avoid using the approximation for that setting.

### 11.9 Probability Insights

Provides plain-language interpretations of the selected variable's threshold
probability, mean, standard deviation, and z-score context. These statements
summarize the sample and are not medical or regulatory advice.

**Examples:**

1. A high exceedance probability can be reported as frequent threshold
   exceedance, while still naming the date range and station.
2. A large standard deviation can be reported as substantial variation rather
   than as evidence that every future observation will be extreme.

## 12. Good Reporting Practice

When presenting a result from AirStat, include:

1. city and station
2. date range and lookback period
3. pollutant or variables and their units
4. number of valid observations
5. selected threshold, when relevant
6. whether suspicious zeros were included or excluded
7. the statistic or probability and its interpretation

For example:

> At the selected Pune station, during the chosen 90-day period, 12% of valid
> PM2.5 observations exceeded 60 µg/m³. This is an observed sample proportion,
> not a guarantee about future readings or every location in Pune.
