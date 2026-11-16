"""Unit IV: correlation, regression and predictive analytics.

Everything for Unit IV is in this one file, in four parts:

  1. DATA PREPARATION  - choosing usable variables, complete cases, train/test split
  2. MODEL MATHS       - least squares, error metrics, coefficient statistics
  3. DIAGNOSTICS       - the five regression assumptions
  4. STREAMLIT PAGES   - one function per tab (correlation ... workflow)

The regression is written with NumPy/SciPy so every formula is visible:
    beta = argmin sum (y - X beta)^2        (ordinary least squares)
No test data is used when fitting, so there is no data leakage.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from components import charts
from components.common import AppContext, correlation_strength, metric_cards
from components.navigation import go_to_section
from utils.config import display_name

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42          # fixed seed -> the same split every run
MIN_ROWS = 30              # minimum complete rows to fit any model
MIN_TEST_ROWS = 5
NON_NUMERIC = {"datetime", "suspicious_zero"}   # never used as variables
PREFERRED_TARGET = "pm25"
PREFERRED_X = "pm10"
PREFERRED_PREDICTORS = ["pm10", "no2", "temperature", "relativehumidity", "windspeed"]
SPLIT_METHODS = {
    f"Random shuffle (seed {RANDOM_STATE})": "random",
    "Chronological (oldest rows train, newest rows test)": "chronological",
}


class ModelError(ValueError):
    """Raised with a user-friendly message when a model cannot be fitted."""


# --------------------------------------------------------------------------- #
# 1. DATA PREPARATION
# --------------------------------------------------------------------------- #
def numeric_candidates(data: pd.DataFrame) -> list[str]:
    """Measured numeric variables usable in a model.

    Excludes the timestamp and flag columns (IDs/timestamps are not valid
    numeric predictors) and any variable that is almost empty or constant.
    """
    columns = []
    for column in data.columns:
        if column in NON_NUMERIC:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values) >= MIN_ROWS and values.nunique() > 1:
            columns.append(column)
    return columns


def complete_cases(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rows where every selected column has a valid number (zeros are kept).

    Missing values are dropped row-wise (listwise deletion), never filled with
    made-up values, and the rows stay in time order.
    """
    columns = list(dict.fromkeys(columns))
    frame = data[["datetime", *columns]].copy()
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame.sort_values("datetime").reset_index(drop=True)


def paired_coverage(data: pd.DataFrame, target: str, columns: list[str]) -> dict[str, float]:
    """Share of rows with a valid target that also have a valid value of each column."""
    target_rows = data[pd.to_numeric(data[target], errors="coerce").notna()]
    if target_rows.empty:
        return {column: 0.0 for column in columns}
    return {
        column: float(pd.to_numeric(target_rows[column], errors="coerce").notna().mean())
        for column in columns
    }


def split_indices(n_rows: int, test_size: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_positions, test_positions) for a reproducible split.

    random        : shuffle with a fixed seed (same result every run)
    chronological : oldest rows train, newest rows test (no look-ahead)
    The split depends only on n_rows, test_size, method and the seed, so two
    models fitted on the same rows always get exactly the same split.
    """
    n_test = int(round(n_rows * test_size))
    n_test = min(max(n_test, 1), n_rows - 2)
    if method == "chronological":
        positions = np.arange(n_rows)
        return positions[: n_rows - n_test], positions[n_rows - n_test :]
    shuffled = np.random.default_rng(RANDOM_STATE).permutation(n_rows)
    return np.sort(shuffled[n_test:]), np.sort(shuffled[:n_test])


# --------------------------------------------------------------------------- #
# 2. MODEL MATHS
# --------------------------------------------------------------------------- #
def fit_ols(features: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Ordinary least squares: minimise the sum of squared errors.

    Solves  [1 | X] * beta ~= y  with lstsq. Returns (intercept, coefficients).
    For ONE predictor this equals the textbook formulas
        slope = cov(x, y) / var(x),   intercept = mean(y) - slope * mean(x).
    """
    design = np.column_stack([np.ones(len(features)), features])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(beta[0]), beta[1:]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE, MSE, RMSE and R-squared.

    MAE  = mean(|y - yhat|)          MSE = mean((y - yhat)^2)
    RMSE = sqrt(MSE)                 R2  = 1 - SS_res / SS_tot
    """
    error = actual - predicted
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    mse = ss_res / len(actual)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R²": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def coefficient_statistics(
    features: np.ndarray, target: np.ndarray, intercept: float, coefficients: np.ndarray
) -> pd.DataFrame:
    """Standard error, t statistic and p-value for intercept + each coefficient."""
    n_rows, n_features = features.shape
    design = np.column_stack([np.ones(n_rows), features])
    beta = np.concatenate([[intercept], coefficients])
    residuals = target - design @ beta
    dof = n_rows - n_features - 1
    nan = np.full(len(beta), np.nan)
    if dof <= 0:
        return pd.DataFrame({"std_error": nan, "t": nan, "p_value": nan})
    sigma_squared = float(residuals @ residuals) / dof
    covariance = sigma_squared * np.linalg.pinv(design.T @ design)
    with np.errstate(divide="ignore", invalid="ignore"):
        std_error = np.sqrt(np.diag(covariance))
        t_values = beta / std_error
    p_values = 2 * stats.t.sf(np.abs(t_values), dof)
    return pd.DataFrame({"std_error": std_error, "t": t_values, "p_value": p_values})


@dataclass
class ModelResult:
    """Everything produced by one fitted regression model."""

    name: str
    target: str
    predictors: list[str]
    intercept: float
    coefficients: np.ndarray
    train: pd.DataFrame
    test: pd.DataFrame
    train_pred: np.ndarray
    test_pred: np.ndarray
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    baseline_test_metrics: dict[str, float]   # "always predict the training mean"
    statistics: pd.DataFrame                   # intercept first, then predictors
    n_available: int                           # rows with a valid target
    split_method: str
    test_size: float
    extras: dict = field(default_factory=dict)

    @property
    def residuals_train(self) -> np.ndarray:
        return self.train[self.target].to_numpy() - self.train_pred

    def predict(self, values: np.ndarray) -> float:
        return float(self.intercept + np.dot(values, self.coefficients))


def fit_model(
    data: pd.DataFrame,
    name: str,
    target: str,
    predictors: list[str],
    test_size: float,
    method: str,
) -> ModelResult:
    """Prepare rows, split, fit on TRAIN only, evaluate on TEST only."""
    if not predictors:
        raise ModelError("Choose at least one independent variable.")
    if target in predictors:
        raise ModelError("The dependent variable cannot also be a predictor.")
    n_available = int(pd.to_numeric(data[target], errors="coerce").notna().sum())
    frame = complete_cases(data, [target, *predictors])
    if len(frame) < MIN_ROWS:
        raise ModelError(
            f"Only {len(frame)} rows have valid values for all selected variables; "
            f"at least {MIN_ROWS} are needed. Try fewer variables or a wider date range."
        )
    train_pos, test_pos = split_indices(len(frame), test_size, method)
    if len(test_pos) < MIN_TEST_ROWS or len(train_pos) < len(predictors) + 10:
        raise ModelError(
            "Too few rows to split into training and test sets with this many variables."
        )
    train, test = frame.iloc[train_pos], frame.iloc[test_pos]
    x_train, y_train = train[predictors].to_numpy(), train[target].to_numpy()
    x_test, y_test = test[predictors].to_numpy(), test[target].to_numpy()

    intercept, coefficients = fit_ols(x_train, y_train)      # TRAIN rows only
    train_pred = intercept + x_train @ coefficients
    test_pred = intercept + x_test @ coefficients             # unseen rows
    return ModelResult(
        name=name,
        target=target,
        predictors=list(predictors),
        intercept=intercept,
        coefficients=coefficients,
        train=train,
        test=test,
        train_pred=train_pred,
        test_pred=test_pred,
        train_metrics=regression_metrics(y_train, train_pred),
        test_metrics=regression_metrics(y_test, test_pred),
        baseline_test_metrics=regression_metrics(y_test, np.full(len(y_test), y_train.mean())),
        statistics=coefficient_statistics(x_train, y_train, intercept, coefficients),
        n_available=n_available,
        split_method=method,
        test_size=test_size,
    )


def standardized_coefficients(model: ModelResult) -> np.ndarray:
    """Coefficient x sd(x) / sd(y): effect of +1 standard deviation, comparable across units."""
    sd_y = model.train[model.target].std(ddof=1)
    sd_x = model.train[model.predictors].std(ddof=1).to_numpy()
    return model.coefficients * sd_x / sd_y


def compare_models(
    data: pd.DataFrame, simple: ModelResult, multiple: ModelResult
) -> tuple[pd.DataFrame, int, int]:
    """Fit both models on IDENTICAL rows and the IDENTICAL split for a fair comparison.

    The rows are those valid for every variable used by EITHER model. Because the
    split depends only on the number of rows and the seed, both models get exactly
    the same training and test rows. Returns (metrics table, n_train, n_test).
    """
    common = complete_cases(data, [simple.target, *simple.predictors, *multiple.predictors])
    fitted = {
        "Simple linear regression": fit_model(
            common, "Simple linear regression", simple.target, simple.predictors,
            simple.test_size, simple.split_method),
        "Multiple linear regression": fit_model(
            common, "Multiple linear regression", simple.target, multiple.predictors,
            simple.test_size, simple.split_method),
    }
    rows = {label: model.test_metrics for label, model in fitted.items()}
    first = fitted["Simple linear regression"]
    rows["Baseline (always predict training mean)"] = first.baseline_test_metrics
    return pd.DataFrame(rows).T, len(first.train), len(first.test)


# --------------------------------------------------------------------------- #
# 3. DIAGNOSTICS (the five regression assumptions)
# --------------------------------------------------------------------------- #
def durbin_watson(residuals: np.ndarray) -> float:
    """sum((e_t - e_{t-1})^2) / sum(e_t^2).  ~2 = independent, <2 = positive autocorrelation."""
    return float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2))


def breusch_pagan(residuals: np.ndarray, features: np.ndarray) -> tuple[float, float]:
    """Breusch-Pagan test for constant variance (Koenker form).

    Regress squared residuals on the predictors: LM = n * R^2 ~ chi-square(k).
    A small p-value means the spread of the errors changes with the predictors.
    """
    squared = residuals**2
    intercept, coefficients = fit_ols(features, squared)
    fitted = intercept + features @ coefficients
    r2 = regression_metrics(squared, fitted)["R²"]
    lm = len(residuals) * r2
    return float(lm), float(stats.chi2.sf(lm, features.shape[1]))


def ramsey_reset(target: np.ndarray, fitted: np.ndarray, features: np.ndarray) -> tuple[float, float]:
    """Ramsey RESET test for linearity (adds fitted^2 and tests it with an F-test).

    A small p-value means a curved term improves the model, i.e. the straight-line
    (linear) form is probably too simple.
    """
    n_rows, n_features = features.shape
    dof = n_rows - n_features - 2
    if dof <= 0:
        return float("nan"), float("nan")
    rss_restricted = float(np.sum((target - fitted) ** 2))
    extended = np.column_stack([features, fitted**2])
    intercept, coefficients = fit_ols(extended, target)
    rss_full = float(np.sum((target - (intercept + extended @ coefficients)) ** 2))
    if rss_full <= 0:
        return float("nan"), float("nan")
    f_stat = (rss_restricted - rss_full) / (rss_full / dof)
    return float(f_stat), float(stats.f.sf(f_stat, 1, dof))


def variance_inflation_factors(features: np.ndarray) -> np.ndarray:
    """VIF_j = 1 / (1 - R_j^2), where R_j^2 comes from regressing predictor j on the others."""
    n_features = features.shape[1]
    result = np.empty(n_features)
    for j in range(n_features):
        others = np.delete(features, j, axis=1)
        intercept, coefficients = fit_ols(others, features[:, j])
        r2 = regression_metrics(features[:, j], intercept + others @ coefficients)["R²"]
        result[j] = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    return result


def run_diagnostics(model: ModelResult) -> dict:
    """Run all five assumption checks on the TRAINING residuals of a model."""
    residuals, fitted = model.residuals_train, model.train_pred
    features = model.train[model.predictors].to_numpy()
    target = model.train[model.target].to_numpy()
    n_rows, n_features = features.shape
    checks: list[dict] = []

    # 1. Linearity
    f_stat, p_value = ramsey_reset(target, fitted, features)
    checks.append({
        "Assumption": "Linearity", "Check": "Ramsey RESET test",
        "Statistic": f_stat, "p-value": p_value,
        "violated": bool(p_value < 0.05) if np.isfinite(p_value) else None,
        "warning": "A curved term improves the fit, so a straight line may be too simple.",
    })
    # 2. Independence of errors
    dw = durbin_watson(residuals)
    lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    checks.append({
        "Assumption": "Independence of errors", "Check": "Durbin-Watson (≈2 is ideal)",
        "Statistic": dw, "p-value": np.nan,
        "violated": bool(dw < 1.5 or dw > 2.5),
        "warning": (f"Neighbouring residuals are correlated (lag-1 correlation {lag1:.2f}). "
                    "Hourly air-quality readings are a time series, so this is common."),
    })
    # 3. Homoscedasticity
    lm, bp_p = breusch_pagan(residuals, features)
    checks.append({
        "Assumption": "Constant variance (homoscedasticity)", "Check": "Breusch-Pagan test",
        "Statistic": lm, "p-value": bp_p,
        "violated": bool(bp_p < 0.05) if np.isfinite(bp_p) else None,
        "warning": "The size of the errors changes with the predictors (funnel shape).",
    })
    # 4. Normality of residuals
    sample = residuals
    if n_rows > 5000:  # Shapiro-Wilk is only reliable up to 5000 points
        sample = np.random.default_rng(RANDOM_STATE).choice(residuals, 5000, replace=False)
    if np.ptp(sample) > 0:
        sw_stat, sw_p = stats.shapiro(sample)
    else:
        sw_stat, sw_p = float("nan"), float("nan")
    skewness = float(stats.skew(residuals))
    kurtosis = float(stats.kurtosis(residuals))   # excess kurtosis: 0 for a normal curve
    checks.append({
        "Assumption": "Normality of residuals", "Check": "Shapiro-Wilk test",
        "Statistic": float(sw_stat), "p-value": float(sw_p),
        "violated": bool(sw_p < 0.05) if np.isfinite(sw_p) else None,
        "warning": (f"Residuals depart from a bell shape (skewness {skewness:.2f}, "
                    f"excess kurtosis {kurtosis:.2f}; both are 0 for a normal curve)."),
    })
    # 5. Multicollinearity (needs 2+ predictors)
    vif = variance_inflation_factors(features) if n_features >= 2 else None
    if vif is None:
        checks.append({
            "Assumption": "No multicollinearity", "Check": "VIF",
            "Statistic": np.nan, "p-value": np.nan, "violated": None,
            "warning": "",
        })
    else:
        checks.append({
            "Assumption": "No multicollinearity", "Check": "Largest VIF (<5 good, ≥10 serious)",
            "Statistic": float(np.max(vif)), "p-value": np.nan,
            "violated": bool(np.max(vif) >= 10),
            "warning": "Some predictors carry almost the same information, so their "
                       "individual coefficients are unstable.",
        })
    return {
        "checks": checks, "vif": vif, "durbin_watson": dw, "lag1": lag1,
        "skewness": skewness, "kurtosis": kurtosis, "residuals": residuals, "fitted": fitted,
    }


def verdict(check: dict) -> str:
    """Short status text used in the summary table."""
    if check["violated"] is None:
        return "Not applicable" if check["Assumption"].startswith("No multi") else "Not testable"
    return "⚠️ May be violated" if check["violated"] else "✅ Looks acceptable"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt(value: float, digits: int = 4) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def fmt_p(p_value: float) -> str:
    """p-values below 0.001 are shown as '<0.001' instead of rounding to 0."""
    if p_value is None or not np.isfinite(p_value):
        return "n/a"
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def with_unit(value: float, unit: str, digits: int = 4) -> str:
    return f"{fmt(value, digits)} {unit}".strip()


def equation_text(model: ModelResult, ctx: AppContext) -> str:
    """Plain-text regression equation, e.g.  PM2.5 = 3.2 + 0.54 × PM10."""
    text = f"{display_name(model.target)} = {fmt(model.intercept)}"
    for name, coefficient in zip(model.predictors, model.coefficients):
        sign = "+" if coefficient >= 0 else "−"
        text += f" {sign} {fmt(abs(coefficient))} × {display_name(name)}"
    return text


def equation_latex(model: ModelResult) -> str:
    """The same equation as LaTeX for st.latex."""
    text = rf"\widehat{{\text{{{display_name(model.target)}}}}} = {fmt(model.intercept)}"
    for name, coefficient in zip(model.predictors, model.coefficients):
        sign = "+" if coefficient >= 0 else "-"
        text += rf" {sign} {fmt(abs(coefficient))} \times \text{{{display_name(name)}}}"
    return text


def unit_phrase(ctx: AppContext, column: str) -> str:
    """Unit name for amounts, e.g. 'µg/m³' (or the word 'units' when the unit is unknown)."""
    unit = ctx.units.get(column, "")
    return f"{unit}" if unit else "units"


def one_unit(ctx: AppContext, column: str) -> str:
    """'1 µg/m³' or, when the unit is unknown, '1 unit'."""
    unit = ctx.units.get(column, "")
    return f"1 {unit}" if unit else "1 unit"


def r2_words(r2: float) -> str:
    if not np.isfinite(r2):
        return "could not be judged"
    if r2 < 0:
        return "worse than simply predicting the average"
    if r2 < 0.1:
        return "almost none of the variation"
    if r2 < 0.3:
        return "a weak share of the variation"
    if r2 < 0.5:
        return "a moderate share of the variation"
    if r2 < 0.7:
        return "a good share of the variation"
    return "a large share of the variation"


def coefficient_table(model: ModelResult, ctx: AppContext) -> pd.DataFrame:
    """Coefficients with direction, significance and a standardised effect size."""
    beta_std = standardized_coefficients(model)
    stats_table = model.statistics.iloc[1:].reset_index(drop=True)
    rows = []
    for index, name in enumerate(model.predictors):
        coefficient = model.coefficients[index]
        rows.append({
            "Predictor": ctx.label(name),
            "Coefficient": coefficient,
            "Direction": "↑ positive" if coefficient > 0 else "↓ negative" if coefficient < 0 else "none",
            "Std. error": stats_table.loc[index, "std_error"],
            "p-value": stats_table.loc[index, "p_value"],
            "Standardised effect": beta_std[index],
        })
    return pd.DataFrame(rows)


def show_coefficient_table(model: ModelResult, ctx: AppContext) -> None:
    table = coefficient_table(model, ctx)
    st.dataframe(
        table.style.format(
            {"Coefficient": "{:.4g}", "Std. error": "{:.3g}",
             "p-value": fmt_p, "Standardised effect": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )


def setup_ensure_choice(key: str, options: list[str], default: str) -> None:
    """Make sure a selectbox's stored value is valid before the widget is drawn."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default


def sync_target(source: str, other: str) -> None:
    """on_change callback: keep the dependent variable identical in both regression tabs."""
    st.session_state[other] = st.session_state[source]


def rows_note(model: ModelResult) -> str:
    total = len(model.train) + len(model.test)
    method = "random shuffle (seed 42)" if model.split_method == "random" else "chronological"
    return (
        f"Rows used: **{total}** complete rows (of {model.n_available} with a valid "
        f"{display_name(model.target)} value) → **{len(model.train)} training** / "
        f"**{len(model.test)} test**, {method} split. Coefficients are learned from "
        "training rows only; all metrics below use the unseen test rows."
    )


def prediction_range_warning(model: ModelResult, name: str, value: float, ctx: AppContext) -> None:
    low, high = model.train[name].min(), model.train[name].max()
    if value < low or value > high:
        st.warning(
            f"{display_name(name)} = {value:.4g} is outside the range the model learned from "
            f"({low:.4g} to {high:.4g}). Predictions outside this range (extrapolation) are unreliable."
        )


# --------------------------------------------------------------------------- #
# 4. STREAMLIT PAGES (one function per tab)
# --------------------------------------------------------------------------- #
def tab_correlation(ctx: AppContext, candidates: list[str], shared: dict) -> None:
    st.subheader("Correlation analysis")
    st.write(
        "Correlation measures how strongly two numeric variables move together, "
        "from −1 (perfect opposite movement) through 0 (no linear pattern) to +1 "
        "(perfect same-direction movement)."
    )
    default_x = PREFERRED_TARGET if PREFERRED_TARGET in candidates else candidates[0]
    default_y = PREFERRED_X if PREFERRED_X in candidates and PREFERRED_X != default_x else next(
        c for c in candidates if c != default_x
    )
    setup_ensure_choice("u4_corr_x", candidates, default_x)
    setup_ensure_choice("u4_corr_y", candidates, default_y)
    selectors = st.columns(2)
    variable_a = selectors[0].selectbox("Variable 1 (X axis)", candidates, key="u4_corr_x",
                                        format_func=display_name)
    variable_b = selectors[1].selectbox("Variable 2 (Y axis)", candidates, key="u4_corr_y",
                                        format_func=display_name)
    if variable_a == variable_b:
        st.warning("Choose two different variables.")
    else:
        pairs = complete_cases(ctx.analysis_data, [variable_a, variable_b])
        if len(pairs) < 3:
            st.warning("At least 3 rows with valid values for both variables are needed.")
        elif pairs[variable_a].nunique() < 2 or pairs[variable_b].nunique() < 2:
            st.warning("One of the variables is constant in this selection, so correlation is undefined.")
        else:
            pearson_r, pearson_p = stats.pearsonr(pairs[variable_a], pairs[variable_b])
            spearman_r, spearman_p = stats.spearmanr(pairs[variable_a], pairs[variable_b])
            shared["correlation"] = (variable_a, variable_b, float(pearson_r), len(pairs))
            metric_cards(
                [
                    ("Paired observations (n)", f"{len(pairs)}"),
                    ("Pearson r", f"{pearson_r:.3f}"),
                    ("Spearman ρ", f"{spearman_r:.3f}"),
                    ("Pearson p-value", fmt_p(pearson_p)),
                    ("Spearman p-value", fmt_p(spearman_p)),
                ],
                [
                    "Rows where both variables have a valid value.",
                    "Strength of the straight-line relationship.",
                    "Strength of the rank (monotonic) relationship; robust to outliers.",
                    "Probability of seeing |r| this large if the true correlation were 0.",
                    "Same idea for Spearman.",
                ],
            )
            strength, direction = correlation_strength(pearson_r)
            label_a, label_b = display_name(variable_a), display_name(variable_b)
            if direction == "positive":
                meaning = f"when {label_a} is higher, {label_b} tends to be higher too"
            elif direction == "negative":
                meaning = f"when {label_a} is higher, {label_b} tends to be lower"
            else:
                meaning = "there is no linear pattern"
            message = (
                f"**Interpretation:** a **{strength} {direction}** linear relationship "
                f"(r = {pearson_r:.2f}); {meaning}. About {pearson_r**2:.0%} of the variation in "
                f"one variable is linearly shared with the other (r² = {pearson_r**2:.2f})."
            )
            (st.success if strength == "strong" else st.info)(message)
            if abs(pearson_r - spearman_r) > 0.15:
                st.caption(
                    "Pearson and Spearman differ noticeably, which hints at a curved "
                    "relationship or the influence of extreme values."
                )
            colour_choice = st.selectbox(
                "Colour points by", ["None", "Hour of day", "Date"], key="u4_corr_colour"
            )
            plot_frame = pairs.copy()
            colour, colour_label = None, None
            if colour_choice == "Hour of day":
                plot_frame["Hour"] = plot_frame["datetime"].dt.hour
                colour, colour_label = "Hour", "Hour of day"
            elif colour_choice == "Date":
                plot_frame["Date ordinal"] = plot_frame["datetime"].map(pd.Timestamp.toordinal)
                colour, colour_label = "Date ordinal", "Date"
            st.plotly_chart(
                charts.scatter_with_trend(plot_frame, variable_a, variable_b,
                                          ctx.label(variable_a), ctx.label(variable_b),
                                          colour, colour_label),
                width="stretch", key="u4_chart_1")
    st.warning(
        "**Correlation does not imply causation.** Two variables can move together because "
        "of a third factor (for example, weather or traffic) or by coincidence. Hourly "
        "readings are also repeated measures of the same air, so p-values here look more "
        "certain than they really are."
    )

    st.markdown("#### Correlation heatmap")
    controls = st.columns([3, 1])
    default_heatmap = candidates[: min(len(candidates), 8)]
    chosen = controls[0].multiselect(
        "Variables to include", candidates, default=default_heatmap,
        format_func=display_name, key="u4_heatmap_vars",
    )
    method = controls[1].radio("Method", ["Pearson", "Spearman"], horizontal=True,
                               key="u4_heatmap_method")
    if len(chosen) < 2:
        st.info("Select at least two variables for the heatmap.")
    else:
        numeric = ctx.analysis_data[chosen].apply(pd.to_numeric, errors="coerce")
        matrix = numeric.corr(method=method.lower(), min_periods=3).rename(
            index=display_name, columns=display_name
        )
        st.plotly_chart(
            charts.correlation_heatmap(matrix, f"{method} correlation matrix (pairwise complete rows)"),
            width="stretch", key="u4_chart_2")
        st.caption("Each cell uses every row where that pair of variables is available. "
                   "Blank cells mean too few shared observations.")

    with st.expander("How to read a correlation coefficient"):
        st.markdown(
            "| |r| | Strength | Meaning |\n|---|---|---|\n"
            "| 0.70 – 1.00 | Strong | The points hug a line |\n"
            "| 0.40 – 0.69 | Moderate | A visible trend with plenty of scatter |\n"
            "| 0.00 – 0.39 | Weak | Almost no straight-line pattern |\n\n"
            "- **Positive** (r > 0): both variables rise together, e.g. PM2.5 and PM10 are often "
            "positively related because they share sources such as dust and smoke.\n"
            "- **Negative** (r < 0): one rises while the other falls, e.g. temperature and "
            "relative humidity often move in opposite directions during a day.\n"
            "- **Weak / near 0**: knowing one variable tells you little about the other.\n"
            "- **Pearson** measures straight-line association and is sensitive to outliers. "
            "**Spearman** compares ranks, so it also captures steadily rising or falling "
            "(monotonic) curves and is more robust to outliers."
        )


def pick_target_and_predictors(
    ctx: AppContext, candidates: list[str], mode: str
) -> tuple[str, list[str]] | None:
    """Draw the variable selectors for 'simple' or 'multiple' regression."""
    default_target = PREFERRED_TARGET if PREFERRED_TARGET in candidates else candidates[0]
    # Both regression tabs share one dependent variable so their models are comparable.
    own_key, other_key = (("u4_slr_y", "u4_mlr_y") if mode == "simple" else ("u4_mlr_y", "u4_slr_y"))
    other_value = st.session_state.get(other_key)
    setup_ensure_choice(own_key, candidates, other_value if other_value in candidates else default_target)
    target = st.selectbox(
        "Dependent variable (Y) — what we want to predict", candidates,
        key=own_key, format_func=display_name,
        on_change=sync_target, args=(own_key, other_key),
        help="Shared with the other regression tab so the two models can be compared fairly.",
    )
    options = [c for c in candidates if c != target]
    coverage = paired_coverage(ctx.analysis_data, target, options)
    if not options:
        st.warning("At least two usable variables are needed.")
        return None
    if mode == "simple":
        default_x = PREFERRED_X if PREFERRED_X in options else max(options, key=coverage.get)
        setup_ensure_choice("u4_slr_x", options, default_x)
        predictor = st.selectbox(
            "Independent variable (X) — the input used for prediction", options,
            key="u4_slr_x",
            format_func=lambda c: f"{display_name(c)} — {coverage[c]:.0%} of rows overlap with {display_name(target)}",
        )
        return target, [predictor]

    preferred = [c for c in PREFERRED_PREDICTORS if c in options and coverage[c] >= 0.5]
    default_predictors = preferred or sorted(options, key=coverage.get, reverse=True)[:2]
    current = [c for c in st.session_state.get("u4_mlr_x", default_predictors) if c in options]
    st.session_state["u4_mlr_x"] = current
    predictors = st.multiselect(
        "Independent variables (X) — choose one or more", options, key="u4_mlr_x",
        format_func=lambda c: f"{display_name(c)} — {coverage[c]:.0%} overlap",
        help="The dependent variable is excluded automatically. Variables with low overlap "
             "remove many rows from the model.",
    )
    return target, predictors


def tab_simple(ctx: AppContext, candidates: list[str], test_size: float, method: str) -> ModelResult | None:
    st.subheader("Simple linear regression")
    st.write(
        "One input (X) is used to predict one outcome (Y) with a straight line: "
        "**Y = intercept + slope × X**. Example: predict PM2.5 from PM10."
    )
    picked = pick_target_and_predictors(ctx, candidates, "simple")
    if picked is None:
        return None
    target, predictors = picked
    x = predictors[0]
    try:
        model = fit_model(ctx.analysis_data, "Simple linear regression", target, predictors, test_size, method)
    except ModelError as error:
        st.warning(str(error))
        return None

    slope, intercept = float(model.coefficients[0]), model.intercept
    st.caption(rows_note(model))
    st.latex(equation_latex(model))
    metric_cards(
        [
            ("Slope", f"{slope:.4g}"),
            ("Intercept", f"{intercept:.4g}"),
            ("Test R²", fmt(model.test_metrics["R²"], 3)),
            ("Test RMSE", with_unit(model.test_metrics["RMSE"], ctx.units.get(target, ""))),
            ("Test MAE", with_unit(model.test_metrics["MAE"], ctx.units.get(target, ""))),
        ],
        ["Change in predicted Y for a 1-unit increase in X.",
         "Predicted Y when X = 0.",
         "Share of Y's variation explained, measured on unseen test rows.",
         "Typical prediction error (same unit as Y), test rows.",
         "Average absolute prediction error, test rows."],
    )

    plot_slot = st.container()
    st.markdown("##### Make a prediction")
    x_default = float(model.train[x].mean())
    entered = st.number_input(
        f"Enter a value of {ctx.label(x)}", value=x_default, key=f"u4_slr_input_{x}",
        help="Defaults to the training average.",
    )
    predicted = model.predict(np.array([entered]))
    result_columns = st.columns(2)
    result_columns[0].metric(f"Predicted {ctx.label(target)}", f"{predicted:.4g}")
    result_columns[1].caption(
        f"Calculation: {fmt(intercept)} + {fmt(slope)} × {entered:.4g} = {predicted:.4g}"
    )
    prediction_range_warning(model, x, entered, ctx)
    with plot_slot:
        st.plotly_chart(
            charts.regression_scatter(
                model.train, model.test, x, target, ctx.label(x), ctx.label(target),
                intercept, slope, (entered, predicted),
            ),
            width="stretch", key="u4_chart_3")

    st.markdown("##### Actual versus predicted (test rows)")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.actual_vs_predicted(model.test[target].to_numpy(), model.test_pred,
                                       display_name(target)),
            width="stretch", key="u4_chart_4")
    with right:
        comparison = pd.DataFrame({
            "Time (UTC)": model.test["datetime"].dt.strftime("%Y-%m-%d %H:%M"),
            f"{display_name(x)} (input)": model.test[x].to_numpy(),
            "Actual": model.test[target].to_numpy(),
            "Predicted": model.test_pred,
        })
        comparison["Error (actual − predicted)"] = comparison["Actual"] - comparison["Predicted"]
        st.dataframe(comparison.style.format(
            {col: "{:.3f}" for col in comparison.columns if col != "Time (UTC)"}),
            hide_index=True, height=360, width="stretch")

    with st.expander("What do the slope and intercept mean?", expanded=True):
        direction = "increase" if slope > 0 else "decrease"
        st.markdown(
            f"- **Slope = {slope:.4g}.** On average, when {display_name(x)} goes up by "
            f"{one_unit(ctx, x)}, the model predicts {display_name(target)} will {direction} by "
            f"{abs(slope):.4g} {unit_phrase(ctx, target)}. It describes an association in this "
            "data, not a cause.\n"
            f"- **Intercept = {intercept:.4g}.** The predicted {display_name(target)} when "
            f"{display_name(x)} = 0. Since {display_name(x)} in the data ranges from "
            f"{model.train[x].min():.4g} to {model.train[x].max():.4g}, the intercept is "
            "mostly a mathematical anchor for the line, not a meaningful real-world value.\n"
            "- **How the line is chosen:** least squares picks the slope and intercept that make "
            "the sum of squared vertical gaps between the points and the line as small as possible."
        )
        with st.container():
            st.latex(r"\text{slope}=\frac{\operatorname{cov}(x,y)}{\operatorname{var}(x)},\qquad "
                     r"\text{intercept}=\bar y-\text{slope}\cdot\bar x")
    with st.expander("Coefficient statistics (training data)"):
        show_coefficient_table(model, ctx)
        intercept_row = model.statistics.iloc[0]
        st.caption(f"Intercept: std. error {intercept_row['std_error']:.3g}, p-value {fmt_p(intercept_row['p_value'])}. "
                   f"For simple regression, training R² ({model.train_metrics['R²']:.3f}) equals r².")
    return model


def tab_multiple(ctx: AppContext, candidates: list[str], test_size: float, method: str) -> ModelResult | None:
    st.subheader("Multiple linear regression")
    st.write(
        "Several inputs predict one outcome together: "
        "**Y = intercept + b₁X₁ + b₂X₂ + …**. Example: predict PM2.5 from PM10, NO2, "
        "temperature, humidity and wind speed (whichever exist in this dataset)."
    )
    picked = pick_target_and_predictors(ctx, candidates, "multiple")
    if picked is None:
        return None
    target, predictors = picked
    if not predictors:
        st.info("Select at least one independent variable.")
        return None
    try:
        model = fit_model(ctx.analysis_data, "Multiple linear regression", target, predictors, test_size, method)
    except ModelError as error:
        st.warning(str(error))
        return None

    total_rows = len(model.train) + len(model.test)
    st.caption(rows_note(model))
    if total_rows < 0.6 * model.n_available:
        st.warning(
            f"Only {total_rows} of {model.n_available} rows with a valid {display_name(target)} also "
            "have every selected predictor. Variables with many gaps shrink the data the model can use."
        )
    if len(predictors) == 1:
        st.info("With one predictor this is the same as simple linear regression.")
    st.latex(equation_latex(model))
    metric_cards(
        [
            ("Intercept", f"{model.intercept:.4g}"),
            ("Predictors", f"{len(predictors)}"),
            ("Test R²", fmt(model.test_metrics["R²"], 3)),
            ("Test RMSE", with_unit(model.test_metrics["RMSE"], ctx.units.get(target, ""))),
            ("Test MAE", with_unit(model.test_metrics["MAE"], ctx.units.get(target, ""))),
        ],
        ["Predicted Y when every predictor is 0.",
         "Number of independent variables used.",
         "Share of Y's variation explained, measured on unseen test rows.",
         "Typical prediction error (same unit as Y), test rows.",
         "Average absolute prediction error, test rows."],
    )

    st.markdown("##### Coefficients")
    show_coefficient_table(model, ctx)
    beta_std = standardized_coefficients(model)
    st.plotly_chart(
        charts.horizontal_bars(
            [display_name(name) for name in predictors], list(beta_std),
            "Relative contribution: change in Y (in standard deviations) per +1 SD of each predictor",
            "Standardised coefficient",
        ),
        width="stretch", key="u4_chart_5")

    st.markdown("##### Make a prediction")
    st.caption("Enter a value for each predictor (defaults are the training averages).")
    input_columns = st.columns(min(len(predictors), 4))
    entered = []
    for index, name in enumerate(predictors):
        value = input_columns[index % len(input_columns)].number_input(
            ctx.label(name), value=float(model.train[name].mean()), key=f"u4_mlr_input_{name}",
        )
        entered.append(value)
        prediction_range_warning(model, name, value, ctx)
    predicted = model.predict(np.array(entered))
    st.metric(f"Predicted {ctx.label(target)}", f"{predicted:.4g}")

    st.markdown("##### Actual versus predicted (test rows)")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.actual_vs_predicted(model.test[target].to_numpy(), model.test_pred,
                                       display_name(target)),
            width="stretch", key="u4_chart_6")
    with right:
        comparison = model.test[["datetime"]].assign(
            **{"Actual": model.test[target].to_numpy(), "Predicted": model.test_pred})
        comparison["Error (actual − predicted)"] = comparison["Actual"] - comparison["Predicted"]
        comparison["datetime"] = comparison["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        comparison = comparison.rename(columns={"datetime": "Time (UTC)"})
        st.dataframe(comparison.style.format(
            {col: "{:.3f}" for col in comparison.columns if col != "Time (UTC)"}),
            hide_index=True, height=360, width="stretch")

    with st.expander("How to read each coefficient", expanded=True):
        lines = []
        for name, coefficient, standardised in zip(predictors, model.coefficients, beta_std):
            direction = "higher" if coefficient > 0 else "lower"
            lines.append(
                f"- **{display_name(name)} = {coefficient:.4g}:** holding the other predictors fixed, "
                f"a {one_unit(ctx, name)} increase in {display_name(name)} goes with a predicted "
                f"{abs(coefficient):.4g} {unit_phrase(ctx, target)} {direction} {display_name(target)} "
                f"(standardised effect {standardised:+.2f})."
            )
        st.markdown("\n".join(lines))
        st.markdown(
            f"- **Intercept = {model.intercept:.4g}:** the prediction when all predictors are 0 "
            "(often outside the observed data, so it is just an anchor for the equation).\n"
            "- **Contribution:** the standardised effect puts predictors on one scale (per 1 standard "
            "deviation), so the largest absolute value is the most influential *within this model*.\n"
            "- Coefficients describe association after accounting for the other predictors; "
            "they are not proof of cause and can change if predictors overlap heavily "
            "(see the multicollinearity check)."
        )
    return model


def tab_assumptions(models: dict[str, ModelResult | None], shared: dict) -> None:
    st.subheader("Regression assumptions")
    st.write(
        "A regression line is only trustworthy when its errors (residuals = actual − predicted) "
        "behave sensibly. Here the five standard assumptions are checked on the **training** residuals."
    )
    available = {name: model for name, model in models.items() if model is not None}
    if not available:
        st.info("Fit a simple or multiple regression first (previous tabs).")
        return
    choice = st.radio("Model to check", list(available), horizontal=True, key="u4_assump_model")
    model = available[choice]
    diagnostics = run_diagnostics(model)

    summary = pd.DataFrame([
        {"Assumption": c["Assumption"], "Check": c["Check"], "Statistic": c["Statistic"],
         "p-value": c["p-value"], "Result": verdict(c)}
        for c in diagnostics["checks"]
    ])
    st.dataframe(
        summary.style.format({"Statistic": lambda v: fmt(v, 4), "p-value": lambda v: fmt_p(v)}),
        hide_index=True, width="stretch",
    )
    flagged = [c for c in diagnostics["checks"] if c["violated"]]
    if flagged:
        st.warning(
            f"{len(flagged)} of the checks raised a warning: "
            + ", ".join(c["Assumption"].lower() for c in flagged)
            + ". Interpret the model's p-values and intervals with caution."
        )
    else:
        st.success("No assumption check raised a warning.")
    st.caption(
        "With thousands of rows, formal tests flag even tiny departures, so always look at the charts too."
    )

    checks = {c["Assumption"]: c for c in diagnostics["checks"]}
    residuals, fitted = diagnostics["residuals"], diagnostics["fitted"]

    with st.expander("1 · Linearity — is a straight line appropriate?"):
        st.markdown("**In simple words:** the real relationship should look like a line, not a curve.")
        check = checks["Linearity"]
        st.plotly_chart(charts.residual_vs_fitted(fitted, residuals), width="stretch", key="u4_chart_7")
        st.caption("Look at the red line: if it stays near zero the line fits well; a U-shape or wave means curvature.")
        (st.warning if check["violated"] else st.success)(
            f"RESET test p-value = {fmt_p(check['p-value'])}. "
            + (check["warning"] if check["violated"] else "No clear sign of curvature.")
        )
    with st.expander("2 · Independence of errors — is each error unrelated to its neighbours?"):
        st.markdown("**In simple words:** one hour's mistake should not predict the next hour's mistake. "
                    "Durbin-Watson near 2 is good; well below 2 means errors come in runs.")
        check = checks["Independence of errors"]
        st.plotly_chart(charts.residuals_over_time(model.train["datetime"], residuals), width="stretch", key="u4_chart_8")
        (st.warning if check["violated"] else st.success)(
            f"Durbin-Watson = {diagnostics['durbin_watson']:.2f}, lag-1 residual correlation = "
            f"{diagnostics['lag1']:.2f}. " + (check["warning"] if check["violated"] else "No strong serial pattern.")
        )
        if model.split_method == "random":
            st.caption("With a random split, some hours are missing from the training series, so this check is approximate.")
    with st.expander("3 · Homoscedasticity — is the error size roughly constant?"):
        st.markdown("**In simple words:** the scatter of points around the line should have the same "
                    "width everywhere. A funnel (narrow then wide) breaks this.")
        check = checks["Constant variance (homoscedasticity)"]
        st.plotly_chart(charts.residual_vs_fitted(fitted, residuals), width="stretch",
                        key="u4_resid_fitted_homo")
        (st.warning if check["violated"] else st.success)(
            f"Breusch-Pagan p-value = {fmt_p(check['p-value'])}. "
            + (check["warning"] if check["violated"] else "Error spread looks roughly constant.")
        )
    with st.expander("4 · Normality of residuals — are errors roughly bell-shaped?"):
        st.markdown("**In simple words:** most errors should be small, with large errors rare and "
                    "balanced on both sides. This matters for p-values and confidence intervals, "
                    "much less for the predictions themselves.")
        check = checks["Normality of residuals"]
        left, right = st.columns(2)
        left.plotly_chart(charts.residual_histogram(residuals), width="stretch", key="u4_chart_9")
        right.plotly_chart(charts.qq_plot(residuals), width="stretch", key="u4_chart_10")
        (st.warning if check["violated"] else st.success)(
            f"Shapiro-Wilk p-value = {fmt_p(check['p-value'])}, skewness = {diagnostics['skewness']:.2f}. "
            + (check["warning"] if check["violated"] else "Residuals look reasonably normal.")
        )
        st.caption("In the Q-Q plot, points close to the line mean normal-looking residuals.")
    with st.expander("5 · Multicollinearity — are predictors too similar to each other?"):
        st.markdown("**In simple words:** if two inputs say nearly the same thing (such as PM10 and "
                    "PM2.5), the model cannot tell which one deserves the credit, so coefficients "
                    "become unstable. VIF < 5 is fine, 5–10 is moderate, ≥ 10 is serious.")
        vif = diagnostics["vif"]
        if vif is None:
            st.info("Multicollinearity needs two or more predictors. Use the Multiple Linear "
                    "Regression model to see VIF values.")
        else:
            names = [display_name(p) for p in model.predictors]
            st.plotly_chart(
                charts.horizontal_bars(names, [float(min(v, 100)) for v in vif],
                                       "Variance Inflation Factor (VIF)", "VIF",
                                       [(5, "5"), (10, "10")]),
                width="stretch", key="u4_chart_11")
            st.dataframe(pd.DataFrame({"Predictor": names, "VIF": vif}).style.format({"VIF": "{:.2f}"}),
                         hide_index=True, width="stretch")
            check = checks["No multicollinearity"]
            (st.warning if check["violated"] else st.success)(
                f"Largest VIF = {fmt(check['Statistic'], 3)}. "
                + (check["warning"] if check["violated"] else
                   ("Moderate overlap between some predictors." if np.max(vif) >= 5 else
                    "Predictors are not strongly overlapping."))
            )


def tab_evaluation(ctx: AppContext, models: dict[str, ModelResult | None], shared: dict) -> None:
    st.subheader("Model evaluation metrics")
    st.write(
        "A model must be judged on rows it has **never seen**. All numbers here use the held-out "
        "test rows from a reproducible split; training scores are shown separately only to spot over-fitting."
    )
    available = {name: model for name, model in models.items() if model is not None}
    if not available:
        st.info("Fit a simple or multiple regression first (previous tabs).")
        return

    for name, model in available.items():
        st.markdown(f"##### {name} — test-set results")
        unit = ctx.units.get(model.target, "")
        metrics = model.test_metrics
        metric_cards(
            [
                ("MAE", with_unit(metrics["MAE"], unit)),
                ("MSE", fmt(metrics["MSE"], 4)),
                ("RMSE", with_unit(metrics["RMSE"], unit)),
                ("R² score", fmt(metrics["R²"], 3)),
            ],
            ["Mean Absolute Error — lower is better.",
             "Mean Squared Error — lower is better.",
             "Root Mean Squared Error — lower is better.",
             "Coefficient of determination — higher is better (max 1)."],
        )
        st.caption(f"Predicting {display_name(model.target)} from "
                   f"{', '.join(display_name(p) for p in model.predictors)} · "
                   f"{len(model.test)} test rows · {len(model.train)} training rows.")

    with st.expander("What does each metric mean, and which direction is better?", expanded=True):
        st.markdown(
            "| Metric | Plain-language meaning | Better when |\n|---|---|---|\n"
            "| **MAE** | Average size of the prediction mistakes, ignoring direction. Same unit as Y. | **Lower** |\n"
            "| **MSE** | Average of the squared mistakes. Big mistakes count much more. Unit² (hard to read). | **Lower** |\n"
            "| **RMSE** | Square root of MSE: a typical mistake in the same unit as Y. Always ≥ MAE. | **Lower** |\n"
            "| **R²** | Share of the variation in Y the model explains. 1 = perfect, 0 = no better than "
            "always predicting the average, negative = worse than that. | **Higher** |"
        )
        st.latex(r"MAE=\tfrac1n\sum|y-\hat y|,\quad MSE=\tfrac1n\sum(y-\hat y)^2,\quad "
                 r"RMSE=\sqrt{MSE},\quad R^2=1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}")

    st.markdown("#### Simple vs multiple regression (identical rows and split)")
    simple, multiple = models.get("Simple linear regression"), models.get("Multiple linear regression")
    if simple is None or multiple is None:
        st.info("Fit both models to compare them.")
    elif simple.target != multiple.target:
        st.info("Choose the same dependent variable (Y) in both regression tabs to compare them.")
    else:
        try:
            table, n_train, n_test = compare_models(ctx.analysis_data, simple, multiple)
        except ModelError as error:
            st.warning(f"Comparison not possible: {error}")
        else:
            st.caption(
                f"Both models are re-fitted on the same {n_train + n_test} rows (those valid for every "
                f"variable used by either model), with the same {n_train}/{n_test} train/test split, so "
                "the comparison is fair."
            )
            st.dataframe(
                table.style.format({"MAE": "{:.4g}", "MSE": "{:.4g}", "RMSE": "{:.4g}", "R²": "{:.3f}"})
                .highlight_min(subset=["MAE", "MSE", "RMSE"], color="rgba(0,204,150,0.35)")
                .highlight_max(subset=["R²"], color="rgba(0,204,150,0.35)"),
                width="stretch",
            )
            st.plotly_chart(charts.metric_comparison_bars(table, ["MAE", "RMSE"]), width="stretch", key="u4_chart_12")
            real = table.drop(index="Baseline (always predict training mean)")
            best = real["RMSE"].idxmin()
            gain = real.loc["Simple linear regression", "RMSE"] - real.loc["Multiple linear regression", "RMSE"]
            st.info(
                f"Lowest test RMSE: **{best}**. Adding predictors changed RMSE by "
                f"{gain:+.4g} {ctx.units.get(simple.target, '')} (positive = multiple regression is more accurate). "
                "A more complex model is only worthwhile if it improves unseen-data accuracy clearly."
            )

    with st.expander("Training vs test scores (over-fitting check)"):
        rows = {}
        for name, model in available.items():
            rows[f"{name} — training"] = model.train_metrics
            rows[f"{name} — test"] = model.test_metrics
        st.dataframe(pd.DataFrame(rows).T.style.format({"MAE": "{:.4g}", "MSE": "{:.4g}",
                                                       "RMSE": "{:.4g}", "R²": "{:.3f}"}),
                     width="stretch")
        st.caption("Training scores are optimistic because the model was fitted to those rows. "
                   "A large drop from training to test suggests over-fitting.")


def interpretation_paragraphs(model: ModelResult, ctx: AppContext, diagnostics: dict) -> list[str]:
    """Build the automatic plain-language summary from the REAL model results."""
    target = display_name(model.target)
    unit = unit_phrase(ctx, model.target)
    metrics, baseline = model.test_metrics, model.baseline_test_metrics
    paragraphs = []
    predictors_text = ", ".join(display_name(p) for p in model.predictors)
    paragraphs.append(
        f"The model predicts **{target}** from **{predictors_text}** using "
        f"{len(model.train)} training rows and was tested on {len(model.test)} unseen rows."
    )
    for name, coefficient, p_value in zip(
        model.predictors, model.coefficients, model.statistics["p_value"].iloc[1:]
    ):
        direction = "higher" if coefficient > 0 else "lower"
        significance = (
            "statistically distinguishable from zero"
            if np.isfinite(p_value) and p_value < 0.05
            else "not clearly different from zero"
        )
        qualifier = " holding the other predictors fixed," if len(model.predictors) > 1 else ""
        paragraphs.append(
            f"- **{display_name(name)}** has a {'positive' if coefficient > 0 else 'negative'} coefficient "
            f"({coefficient:.4g}): when {display_name(name)} is {one_unit(ctx, name)} higher,{qualifier} "
            f"predicted {target} is {abs(coefficient):.4g} {unit} {direction} on average "
            f"(p = {fmt_p(p_value)}, {significance})."
        )
    r2 = metrics["R²"]
    improvement = (1 - metrics["RMSE"] / baseline["RMSE"]) if baseline["RMSE"] > 0 else float("nan")
    paragraphs.append(
        f"On the test rows, R² = **{fmt(r2, 3)}**: the model captures {r2_words(r2)} in {target}. "
        f"The typical error (RMSE) is **{fmt(metrics['RMSE'], 4)} {unit}** and the average absolute error "
        f"(MAE) is {fmt(metrics['MAE'], 4)} {unit}"
        + (f", which is {improvement:.0%} {'smaller' if improvement >= 0 else 'larger'} in RMSE than simply "
           "predicting the training average." if np.isfinite(improvement) else ".")
    )
    gap = model.train_metrics["R²"] - r2
    if np.isfinite(gap) and gap > 0.1:
        paragraphs.append(
            f"Training R² ({fmt(model.train_metrics['R²'], 3)}) is clearly higher than test R² "
            f"({fmt(r2, 3)}), a sign the model may not generalise as well as the training fit suggests."
        )
    paragraphs.append(
        "These results describe **associations** in this station's data. They do not show that changing "
        "one variable would cause a change in another."
    )
    return paragraphs


def limitations(model: ModelResult, ctx: AppContext, diagnostics: dict) -> list[str]:
    items = []
    flagged = [c["Assumption"] for c in diagnostics["checks"] if c["violated"]]
    if flagged:
        items.append("Assumption checks raising warnings: " + ", ".join(flagged).lower()
                     + ". Standard errors and p-values may be unreliable.")
    if diagnostics["durbin_watson"] < 1.5:
        items.append("Consecutive hourly readings are strongly related, so the effective amount of "
                     "independent information is smaller than the row count.")
        if model.split_method == "random":
            items.append("The random split places neighbouring hours in both training and test sets, "
                         "which can make test scores look better than real forecasting accuracy. "
                         "Try the chronological split for a stricter test.")
    total = len(model.train) + len(model.test)
    if total < 100:
        items.append(f"Only {total} rows were used, so the estimates can change a lot with new data.")
    if total < 0.8 * model.n_available:
        items.append(f"{model.n_available - total} rows with a valid {display_name(model.target)} were dropped "
                     "because a selected predictor was missing.")
    items.extend([
        "The model is a straight-line (linear) model; curved relationships are not captured.",
        "It uses one station and the selected date range; results may not apply to other places or seasons.",
        "Predictions outside the range of the training data (extrapolation) are unreliable.",
        "Important drivers of air quality (traffic, season, fires, meteorology not in the data) may be missing.",
        "Sensor errors and suspicious zero readings can influence the fit (see the Data Quality page).",
    ])
    return items


def tab_interpretation(ctx: AppContext, models: dict[str, ModelResult | None], shared: dict) -> None:
    st.subheader("Interpretation of regression results")
    available = {name: model for name, model in models.items() if model is not None}
    if not available:
        st.info("Fit a simple or multiple regression first (previous tabs).")
        return
    choice = st.radio("Model to interpret", list(available), horizontal=True, key="u4_interp_model")
    model = available[choice]
    diagnostics = run_diagnostics(model)

    st.markdown("#### Regression equation")
    st.latex(equation_latex(model))
    st.caption(f"Intercept: {fmt(model.intercept)} {unit_phrase(ctx, model.target)}")

    metrics = model.test_metrics
    st.markdown("#### Model evaluation (test data)")
    metric_cards([("MAE", fmt(metrics["MAE"], 4)), ("MSE", fmt(metrics["MSE"], 4)),
                  ("RMSE", fmt(metrics["RMSE"], 4)), ("R²", fmt(metrics["R²"], 3))])

    st.markdown("#### Coefficients and direction of relationships")
    show_coefficient_table(model, ctx)

    st.markdown("#### Automatic interpretation")
    with st.container(border=True):
        for paragraph in interpretation_paragraphs(model, ctx, diagnostics):
            st.markdown(paragraph)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.actual_vs_predicted(model.test[model.target].to_numpy(), model.test_pred,
                                       display_name(model.target)),
            width="stretch", key="u4_interp_avp",
        )
    with right:
        st.markdown("#### Limitations of the model")
        for item in limitations(model, ctx, diagnostics):
            st.markdown(f"- {item}")


def tab_workflow(ctx: AppContext, models: dict[str, ModelResult | None], shared: dict) -> None:
    st.subheader("End-to-end data analytics workflow")
    st.write("Every AirStat analysis follows the same eight steps. The cards show live values from the "
             "dataset and model you currently have selected.")

    data = ctx.analysis_data
    rows_total = len(ctx.data)
    numeric = data[ctx.variables].apply(pd.to_numeric, errors="coerce")
    missing_pct = float(numeric.isna().mean().mean()) if not numeric.empty else float("nan")
    correlation = shared.get("correlation")
    simple, multiple = models.get("Simple linear regression"), models.get("Multiple linear regression")
    best = multiple or simple

    steps = [
        ("Dataset Selection", f"{ctx.city} · {ctx.station}", "Sidebar", "Pick city, station, date range"),
        ("Data Collection", f"{rows_total:,} hourly rows", "Overview", "OpenAQ sensors, cached locally"),
        ("Cleaning & Preprocessing", f"{missing_pct:.1%} values missing", "Data Quality",
         "Types, duplicates, missing values, zeros"),
        ("Exploratory Data Analysis", f"{len(ctx.variables)} variables", "Unit I",
         "Summaries, charts, relationships"),
        ("Statistical Analysis", (f"r = {correlation[2]:.2f} ({display_name(correlation[0])} vs "
                                  f"{display_name(correlation[1])})" if correlation else "Probability & inference"),
         "Unit II · Unit III · Correlation tab", "Probability, tests, correlation"),
        ("Regression Modeling", (f"{len(best.predictors)} predictor(s) → {display_name(best.target)}"
                                 if best else "Fit a model in tabs 2–3"), "Tabs 2 – 3", "Simple & multiple linear regression"),
        ("Model Evaluation", (f"Test R² {fmt(best.test_metrics['R²'], 3)} · RMSE {fmt(best.test_metrics['RMSE'], 3)}"
                              if best else "MAE · MSE · RMSE · R²"), "Tabs 4 – 5", "Assumptions + test-set metrics"),
        ("Result Interpretation", "Plain-language findings & limits", "Tab 6", "Automatic summary, no causal claims"),
    ]
    cards = []
    for number, (title, live, where, description) in enumerate(steps, start=1):
        cards.append(
            f'<div class="as-step"><div class="as-num">{number}</div>'
            f'<div class="as-title">{html.escape(title)}</div>'
            f'<div class="as-desc">{html.escape(description)}</div>'
            f'<div class="as-live">{html.escape(live)}</div>'
            f'<div class="as-where">📍 {html.escape(where)}</div></div>'
        )
    st.markdown(
        """
<style>
.as-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:26px 30px;margin:8px 0 18px}
.as-step{position:relative;border:1px solid rgba(128,128,128,.35);border-radius:14px;
  padding:16px 14px 12px;background:rgba(128,128,128,.07)}
.as-step:not(:nth-child(4n))::after{content:"➜";position:absolute;right:-24px;top:42%;
  font-size:1.3rem;color:#636EFA}
.as-num{position:absolute;top:-14px;left:12px;width:28px;height:28px;border-radius:50%;
  background:#636EFA;color:#fff;font-weight:700;text-align:center;line-height:28px}
.as-title{font-weight:700;margin:6px 0 2px;font-size:1.02rem}
.as-desc{opacity:.75;font-size:.85rem;margin-bottom:8px}
.as-live{font-weight:600;font-size:.9rem;color:#00A67E;margin-bottom:6px}
.as-where{font-size:.8rem;opacity:.85}
@media (max-width:900px){.as-flow{grid-template-columns:repeat(2,1fr)}
  .as-step:not(:nth-child(4n))::after{content:none}}
</style>
<div class="as-flow">"""
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Jump to the dashboard section for a step:**")
    buttons = st.columns(5)
    for column, (label, section) in zip(
        buttons,
        [("Overview", "Overview"), ("Data Quality", "Data Quality"), ("Unit I", "Unit I"),
         ("Unit II", "Unit II"), ("Unit III", "Unit III")],
    ):
        column.button(f"Open {label}", key=f"u4_goto_{section}", on_click=go_to_section,
                      args=(section,), width="stretch")
    with st.expander("Why this order matters"):
        st.markdown(
            "- **Clean before you model:** missing values, duplicates and impossible readings change every statistic.\n"
            "- **Explore before you model:** charts and correlations show which relationships are worth modelling.\n"
            "- **Split before you fit:** evaluating on rows the model never saw avoids over-optimistic results (data leakage).\n"
            "- **Check assumptions and evaluate before you interpret:** a model with poor test scores or broken "
            "assumptions should be described cautiously."
        )


def render(ctx: AppContext) -> None:
    st.header("Unit IV — Correlation, Regression and Predictive Analytics")
    candidates = numeric_candidates(ctx.analysis_data)
    if len(candidates) < 2:
        st.warning(
            f"At least two numeric variables with {MIN_ROWS}+ valid readings are needed. "
            "Choose a wider date range or another station."
        )
        return

    with st.container(border=True):
        st.markdown("**Modeling setup** (applies to all regression tabs)")
        columns = st.columns([1, 2])
        test_percent = columns[0].slider("Test share of the data", 10, 40, 20, 5,
                                         format="%d%%", key="u4_test_size",
                                         help="Share of rows kept aside to test the model.")
        test_size = test_percent / 100
        method_label = columns[1].selectbox(
            "Train/test split", list(SPLIT_METHODS), key="u4_split",
            help="A fixed random seed makes the split reproducible. The chronological split "
                 "is stricter for time-ordered data.",
        )
        method = SPLIT_METHODS[method_label]
        st.caption(
            "Model coefficients are learned from the training rows only. Rows with missing values are "
            "dropped (never filled with invented numbers) and real zero readings are kept. "
            "Timestamps and IDs are never used as predictors."
        )

    tabs = st.tabs([
        "1 · Correlation", "2 · Simple Linear Regression", "3 · Multiple Linear Regression",
        "4 · Regression Assumptions", "5 · Model Evaluation", "6 · Regression Interpretation",
        "7 · Analytics Workflow",
    ])
    shared: dict = {}
    models: dict[str, ModelResult | None] = {}
    with tabs[0]:
        tab_correlation(ctx, candidates, shared)
    with tabs[1]:
        models["Simple linear regression"] = tab_simple(ctx, candidates, test_size, method)
    with tabs[2]:
        models["Multiple linear regression"] = tab_multiple(ctx, candidates, test_size, method)
    with tabs[3]:
        tab_assumptions(models, shared)
    with tabs[4]:
        tab_evaluation(ctx, models, shared)
    with tabs[5]:
        tab_interpretation(ctx, models, shared)
    with tabs[6]:
        tab_workflow(ctx, models, shared)
