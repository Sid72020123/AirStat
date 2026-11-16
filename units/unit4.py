"""Unit IV: correlation, regression and predictive analytics.

Everything for Unit IV is in this one file, in four parts:

  1. DATA PREPARATION  - usable variables, complete rows, train/test split
  2. MODEL MATHS       - least squares, MAE / MSE / RMSE / R-squared
  3. ASSUMPTION CHECKS - linearity, independence, constant variance, normality, VIF
  4. STREAMLIT PAGES   - one function per tab

The regression is written with NumPy so every formula is visible:
    coefficients = the values that make  sum (actual - predicted)^2  as small as possible.
The test rows are never used while fitting, so there is no data leakage.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

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
RANDOM_STATE = 42          # fixed seed -> the same train/test split every run
MIN_ROWS = 30              # minimum complete rows needed to fit a model
MIN_TEST_ROWS = 5
NON_NUMERIC = {"datetime", "suspicious_zero"}   # never used as variables
PREFERRED_TARGET = "pm25"
PREFERRED_X = "pm10"
PREFERRED_PREDICTORS = ["pm10", "no2", "temperature", "relativehumidity", "windspeed"]


class ModelError(ValueError):
    """Raised with a friendly message when a model cannot be fitted."""


# --------------------------------------------------------------------------- #
# 1. DATA PREPARATION
# --------------------------------------------------------------------------- #
def numeric_candidates(data: pd.DataFrame) -> list[str]:
    """Measured numeric variables that can be used in a model.

    The timestamp and flag columns are excluded (IDs and timestamps are not
    meaningful numeric predictors), as are variables that are nearly empty or constant.
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
    """Rows where every selected column has a valid number.

    Missing values are dropped (never filled with invented numbers) and real
    zero readings are kept. Rows stay in time order.
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


def split_indices(n_rows: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible random train/test split: returns (train_positions, test_positions).

    The rows are shuffled with a fixed seed, so every run gives the same split, and
    two models fitted on the same rows always get exactly the same train and test rows.
    """
    n_test = int(round(n_rows * test_size))
    n_test = min(max(n_test, 1), n_rows - 2)
    shuffled = np.random.default_rng(RANDOM_STATE).permutation(n_rows)
    return np.sort(shuffled[n_test:]), np.sort(shuffled[:n_test])


# --------------------------------------------------------------------------- #
# 2. MODEL MATHS
# --------------------------------------------------------------------------- #
def fit_ols(features: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Ordinary least squares: choose the line that minimises the squared errors.

    Returns (intercept, coefficients). With ONE predictor this equals the textbook formulas
        slope = cov(x, y) / var(x),    intercept = mean(y) - slope * mean(x).
    """
    design = np.column_stack([np.ones(len(features)), features])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(beta[0]), beta[1:]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE, MSE, RMSE and R-squared.

    MAE  = mean(|actual - predicted|)        MSE = mean((actual - predicted)^2)
    RMSE = sqrt(MSE)                         R2  = 1 - SS_residual / SS_total
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
    test_metrics: dict[str, float]
    n_available: int          # rows that have a valid target value
    test_size: float

    @property
    def residuals_train(self) -> np.ndarray:
        """Residual = actual - predicted (on the training rows)."""
        return self.train[self.target].to_numpy() - self.train_pred

    def predict(self, values: np.ndarray) -> float:
        return float(self.intercept + np.dot(values, self.coefficients))


def fit_model(
    data: pd.DataFrame, name: str, target: str, predictors: list[str], test_size: float
) -> ModelResult:
    """Prepare rows, split, fit on TRAIN rows only, evaluate on TEST rows only."""
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
    train_pos, test_pos = split_indices(len(frame), test_size)
    if len(test_pos) < MIN_TEST_ROWS or len(train_pos) < len(predictors) + 10:
        raise ModelError("Too few rows to split into training and test sets with this many variables.")
    train, test = frame.iloc[train_pos], frame.iloc[test_pos]
    x_train, y_train = train[predictors].to_numpy(), train[target].to_numpy()
    x_test, y_test = test[predictors].to_numpy(), test[target].to_numpy()

    intercept, coefficients = fit_ols(x_train, y_train)      # learns from TRAIN rows only
    test_pred = intercept + x_test @ coefficients             # predictions for unseen rows
    return ModelResult(
        name=name,
        target=target,
        predictors=list(predictors),
        intercept=intercept,
        coefficients=coefficients,
        train=train,
        test=test,
        train_pred=intercept + x_train @ coefficients,
        test_pred=test_pred,
        test_metrics=regression_metrics(y_test, test_pred),
        n_available=n_available,
        test_size=test_size,
    )


def compare_models(
    data: pd.DataFrame, simple: ModelResult, multiple: ModelResult
) -> tuple[pd.DataFrame, int, int]:
    """Fit both models on IDENTICAL rows and the IDENTICAL split so the comparison is fair.

    The rows used are those valid for every variable of either model. Returns
    (test-metrics table, n_train, n_test).
    """
    common = complete_cases(data, [simple.target, *simple.predictors, *multiple.predictors])
    first = fit_model(common, "Simple linear regression", simple.target, simple.predictors, simple.test_size)
    second = fit_model(common, "Multiple linear regression", simple.target, multiple.predictors, simple.test_size)
    table = pd.DataFrame({first.name: first.test_metrics, second.name: second.test_metrics}).T
    return table, len(first.train), len(first.test)


# --------------------------------------------------------------------------- #
# 3. ASSUMPTION CHECKS (simple rules of thumb, shown next to the plots)
# --------------------------------------------------------------------------- #
def variance_inflation_factors(features: np.ndarray) -> np.ndarray:
    """VIF_j = 1 / (1 - R_j^2), where R_j^2 comes from regressing predictor j on the others."""
    result = np.empty(features.shape[1])
    for j in range(features.shape[1]):
        others = np.delete(features, j, axis=1)
        intercept, coefficients = fit_ols(others, features[:, j])
        r2 = regression_metrics(features[:, j], intercept + others @ coefficients)["R²"]
        result[j] = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    return result


def run_assumption_checks(model: ModelResult) -> dict:
    """Check the five assumptions on the TRAINING residuals.

    Each check is a simple rule of thumb; the plots on the page are the main evidence.
    """
    residuals, fitted = model.residuals_train, model.train_pred
    spread = float(residuals.std(ddof=1))
    checks = []

    # 1. Linearity: residuals should average about 0 everywhere along the predictions.
    bins = pd.qcut(pd.Series(fitted), q=5, duplicates="drop")
    bin_means = pd.Series(residuals).groupby(bins, observed=True).mean()
    worst = float(bin_means.abs().max() / spread) if spread > 0 else 0.0
    checks.append({
        "Assumption": "Linearity",
        "Rule of thumb": "Average residual in each fifth of the predictions stays near 0 (within 0.3 standard deviations)",
        "Value": worst, "violated": worst > 0.3,
        "warning": "The residuals are systematically above or below zero in parts of the range, "
                   "so the relationship may be curved.",
    })
    # 2. Independence: neighbouring residuals (in time order) should not be correlated.
    lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    checks.append({
        "Assumption": "Independence of errors",
        "Rule of thumb": "Correlation between each residual and the next is small (|r| < 0.3)",
        "Value": lag1, "violated": abs(lag1) > 0.3,
        "warning": "Neighbouring residuals are alike. Hourly air-quality readings form a time "
                   "series, so this is common.",
    })
    # 3. Homoscedasticity: error spread should be similar for low and high predictions.
    order = np.argsort(fitted)
    third = max(len(order) // 3, 2)
    low_sd, high_sd = residuals[order[:third]].std(ddof=1), residuals[order[-third:]].std(ddof=1)
    ratio = float(high_sd / low_sd) if low_sd > 0 else float("inf")
    checks.append({
        "Assumption": "Constant variance (homoscedasticity)",
        "Rule of thumb": "Residual spread in the top third ÷ bottom third of predictions is between 0.67 and 1.5",
        "Value": ratio, "violated": not (1 / 1.5 <= ratio <= 1.5),
        "warning": "The size of the errors changes with the predicted value (funnel shape).",
    })
    # 4. Normality: residual distribution should be roughly symmetric and bell-shaped.
    skewness, kurtosis = float(stats.skew(residuals)), float(stats.kurtosis(residuals))
    checks.append({
        "Assumption": "Normality of residuals",
        "Rule of thumb": "Skewness within ±1 and excess kurtosis within ±3 (both are 0 for a normal curve)",
        "Value": skewness, "violated": abs(skewness) > 1 or abs(kurtosis) > 3,
        "warning": "The residuals are lopsided or heavy-tailed, so they are not bell-shaped.",
    })
    # 5. Multicollinearity: needs two or more predictors.
    features = model.train[model.predictors].to_numpy()
    vif = variance_inflation_factors(features) if features.shape[1] >= 2 else None
    checks.append({
        "Assumption": "No multicollinearity",
        "Rule of thumb": "Every VIF below 10 (below 5 is comfortable)",
        "Value": float(np.max(vif)) if vif is not None else np.nan,
        "violated": bool(np.max(vif) >= 10) if vif is not None else None,
        "warning": "Some predictors carry almost the same information, so their coefficients are unstable.",
    })
    return {"checks": checks, "vif": vif, "residuals": residuals, "fitted": fitted,
            "skewness": skewness, "lag1": lag1}


def verdict(check: dict) -> str:
    if check["violated"] is None:
        return "Needs 2+ predictors"
    return "⚠️ May be violated" if check["violated"] else "✅ Looks acceptable"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt(value: float, digits: int = 4) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def with_unit(value: float, unit: str, digits: int = 4) -> str:
    return f"{fmt(value, digits)} {unit}".strip()


def equation_latex(model: ModelResult) -> str:
    """The regression equation as LaTeX for st.latex."""
    text = rf"\widehat{{\text{{{display_name(model.target)}}}}} = {fmt(model.intercept)}"
    for name, coefficient in zip(model.predictors, model.coefficients):
        sign = "+" if coefficient >= 0 else "-"
        text += rf" {sign} {fmt(abs(coefficient))} \times \text{{{display_name(name)}}}"
    return text


def unit_phrase(ctx: AppContext, column: str) -> str:
    """Unit for amounts, e.g. 'µg/m³' (the word 'units' when the unit is unknown)."""
    return ctx.units.get(column, "") or "units"


def one_unit(ctx: AppContext, column: str) -> str:
    """'1 µg/m³' or, when the unit is unknown, '1 unit'."""
    unit = ctx.units.get(column, "")
    return f"1 {unit}" if unit else "1 unit"


def r2_words(r2: float) -> str:
    if not np.isfinite(r2):
        return "could not be judged"
    if r2 < 0:
        return "worse than simply predicting the average"
    if r2 < 0.3:
        return "a weak share of the variation"
    if r2 < 0.5:
        return "a moderate share of the variation"
    if r2 < 0.7:
        return "a good share of the variation"
    return "a large share of the variation"


def show_coefficient_table(model: ModelResult, ctx: AppContext) -> None:
    table = pd.DataFrame({
        "Predictor": [ctx.label(name) for name in model.predictors],
        "Coefficient": model.coefficients,
        "Direction": ["↑ positive" if c > 0 else "↓ negative" if c < 0 else "none" for c in model.coefficients],
    })
    st.dataframe(table.style.format({"Coefficient": "{:.4g}"}), hide_index=True, width="stretch")


def setup_ensure_choice(key: str, options: list[str], default: str) -> None:
    """Make sure a selectbox's stored value is valid before the widget is drawn."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default


def sync_target(source: str, other: str) -> None:
    """on_change callback: keep the dependent variable the same in both regression tabs."""
    st.session_state[other] = st.session_state[source]


def rows_note(model: ModelResult) -> str:
    total = len(model.train) + len(model.test)
    return (
        f"Rows used: **{total}** (rows with missing values are dropped) → **{len(model.train)} training** / "
        f"**{len(model.test)} test**. The model learns from the training rows only; all metrics use the unseen test rows."
    )


def show_actual_vs_predicted(model: ModelResult, key: str) -> None:
    st.markdown("##### Actual versus predicted (test rows)")
    left, right = st.columns(2)
    left.plotly_chart(
        charts.actual_vs_predicted(model.test[model.target].to_numpy(), model.test_pred,
                                   display_name(model.target)),
        width="stretch", key=key,
    )
    table = pd.DataFrame({
        "Time (UTC)": model.test["datetime"].dt.strftime("%Y-%m-%d %H:%M"),
        "Actual": model.test[model.target].to_numpy(),
        "Predicted": model.test_pred,
    })
    table["Error (actual − predicted)"] = table["Actual"] - table["Predicted"]
    right.dataframe(
        table.style.format({c: "{:.3f}" for c in table.columns if c != "Time (UTC)"}),
        hide_index=True, height=360, width="stretch",
    )


# --------------------------------------------------------------------------- #
# 4. STREAMLIT PAGES (one function per tab)
# --------------------------------------------------------------------------- #
def tab_correlation(ctx: AppContext, candidates: list[str], shared: dict) -> None:
    st.subheader("Correlation analysis")
    st.write(
        "Correlation measures how strongly two numeric variables move together, from −1 "
        "(perfect opposite movement) through 0 (no linear pattern) to +1 (perfect same-direction movement)."
    )
    default_x = PREFERRED_TARGET if PREFERRED_TARGET in candidates else candidates[0]
    default_y = PREFERRED_X if PREFERRED_X in candidates and PREFERRED_X != default_x else next(
        c for c in candidates if c != default_x
    )
    setup_ensure_choice("u4_corr_x", candidates, default_x)
    setup_ensure_choice("u4_corr_y", candidates, default_y)
    selectors = st.columns(2)
    variable_a = selectors[0].selectbox("Variable 1 (X axis)", candidates, key="u4_corr_x", format_func=display_name)
    variable_b = selectors[1].selectbox("Variable 2 (Y axis)", candidates, key="u4_corr_y", format_func=display_name)
    if variable_a == variable_b:
        st.warning("Choose two different variables.")
    else:
        pairs = complete_cases(ctx.analysis_data, [variable_a, variable_b])
        if len(pairs) < 3:
            st.warning("At least 3 rows with valid values for both variables are needed.")
        elif pairs[variable_a].nunique() < 2 or pairs[variable_b].nunique() < 2:
            st.warning("One of the variables is constant in this selection, so correlation is undefined.")
        else:
            pearson_r = float(pairs[variable_a].corr(pairs[variable_b], method="pearson"))
            spearman_r = float(pairs[variable_a].corr(pairs[variable_b], method="spearman"))
            shared["correlation"] = (variable_a, variable_b, pearson_r, len(pairs))
            metric_cards(
                [("Paired observations (n)", f"{len(pairs)}"), ("Pearson r", f"{pearson_r:.3f}"),
                 ("Spearman ρ", f"{spearman_r:.3f}")],
                ["Rows where both variables have a valid value.",
                 "Strength of the straight-line relationship.",
                 "Strength of the rank (steadily rising or falling) relationship."],
            )
            strength, direction = correlation_strength(pearson_r)
            label_a, label_b = display_name(variable_a), display_name(variable_b)
            if direction == "positive":
                meaning = f"when {label_a} is higher, {label_b} tends to be higher too"
            elif direction == "negative":
                meaning = f"when {label_a} is higher, {label_b} tends to be lower"
            else:
                meaning = "there is no linear pattern"
            st.info(f"**Interpretation:** a **{strength} {direction}** relationship (r = {pearson_r:.2f}); {meaning}.")
            st.plotly_chart(
                charts.scatter_with_trend(pairs, variable_a, variable_b, ctx.label(variable_a), ctx.label(variable_b)),
                width="stretch", key="u4_corr_scatter",
            )
    st.warning(
        "**Correlation does not imply causation.** Two variables can move together because of a "
        "third factor (for example weather or traffic) or by coincidence."
    )

    st.markdown("#### Correlation heatmap")
    controls = st.columns([3, 1])
    chosen = controls[0].multiselect(
        "Variables to include", candidates, default=candidates[: min(len(candidates), 8)],
        format_func=display_name, key="u4_heatmap_vars",
    )
    method = controls[1].radio("Method", ["Pearson", "Spearman"], horizontal=True, key="u4_heatmap_method")
    if len(chosen) < 2:
        st.info("Select at least two variables for the heatmap.")
    else:
        numeric = ctx.analysis_data[chosen].apply(pd.to_numeric, errors="coerce")
        matrix = numeric.corr(method=method.lower(), min_periods=3).rename(index=display_name, columns=display_name)
        st.plotly_chart(
            charts.correlation_heatmap(matrix, f"{method} correlation matrix"),
            width="stretch", key="u4_heatmap",
        )

    with st.expander("How to read a correlation coefficient"):
        st.markdown(
            "| |r| | Strength |\n|---|---|\n"
            "| 0.70 – 1.00 | Strong |\n| 0.40 – 0.69 | Moderate |\n| 0.00 – 0.39 | Weak |\n\n"
            "- **Positive** (r > 0): both variables rise together, e.g. PM2.5 and PM10.\n"
            "- **Negative** (r < 0): one rises while the other falls, e.g. temperature and humidity often do this.\n"
            "- **Weak / near 0**: knowing one variable tells you little about the other.\n"
            "- **Pearson** measures straight-line relationships. **Spearman** compares ranks, so it also "
            "captures steadily rising or falling curves and is less affected by extreme values."
        )


def pick_target_and_predictors(ctx: AppContext, candidates: list[str], mode: str) -> tuple[str, list[str]] | None:
    """Draw the variable selectors for 'simple' or 'multiple' regression."""
    default_target = PREFERRED_TARGET if PREFERRED_TARGET in candidates else candidates[0]
    # Both regression tabs share one dependent variable so their models can be compared.
    own_key, other_key = ("u4_slr_y", "u4_mlr_y") if mode == "simple" else ("u4_mlr_y", "u4_slr_y")
    other_value = st.session_state.get(other_key)
    setup_ensure_choice(own_key, candidates, other_value if other_value in candidates else default_target)
    target = st.selectbox(
        "Dependent variable (Y) — what we want to predict", candidates, key=own_key,
        format_func=display_name, on_change=sync_target, args=(own_key, other_key),
        help="Shared with the other regression tab so the two models can be compared fairly.",
    )
    options = [c for c in candidates if c != target]
    if not options:
        st.warning("At least two usable variables are needed.")
        return None
    coverage = paired_coverage(ctx.analysis_data, target, options)
    if mode == "simple":
        default_x = PREFERRED_X if PREFERRED_X in options else max(options, key=coverage.get)
        setup_ensure_choice("u4_slr_x", options, default_x)
        predictor = st.selectbox(
            "Independent variable (X) — the input used for prediction", options, key="u4_slr_x",
            format_func=display_name,
        )
        return target, [predictor]

    preferred = [c for c in PREFERRED_PREDICTORS if c in options and coverage[c] >= 0.5]
    default_predictors = preferred or sorted(options, key=coverage.get, reverse=True)[:2]
    st.session_state["u4_mlr_x"] = [c for c in st.session_state.get("u4_mlr_x", default_predictors) if c in options]
    predictors = st.multiselect(
        "Independent variables (X) — choose one or more", options, key="u4_mlr_x", format_func=display_name,
        help="The dependent variable is excluded automatically.",
    )
    return target, predictors


def tab_simple(ctx: AppContext, candidates: list[str], test_size: float) -> ModelResult | None:
    st.subheader("Simple linear regression")
    st.write(
        "One input (X) predicts one outcome (Y) with a straight line: "
        "**Y = intercept + slope × X**. Example: predict PM2.5 from PM10."
    )
    picked = pick_target_and_predictors(ctx, candidates, "simple")
    if picked is None:
        return None
    target, predictors = picked
    x = predictors[0]
    try:
        model = fit_model(ctx.analysis_data, "Simple linear regression", target, predictors, test_size)
    except ModelError as error:
        st.warning(str(error))
        return None

    slope, intercept = float(model.coefficients[0]), model.intercept
    st.caption(rows_note(model))
    st.latex(equation_latex(model))
    metric_cards(
        [("Slope", f"{slope:.4g}"), ("Intercept", f"{intercept:.4g}"),
         ("Test R²", fmt(model.test_metrics["R²"], 3)),
         ("Test RMSE", with_unit(model.test_metrics["RMSE"], ctx.units.get(target, "")))],
        ["Change in predicted Y for a 1-unit increase in X.", "Predicted Y when X = 0.",
         "Share of Y's variation explained, measured on unseen test rows.",
         "Typical prediction error (same unit as Y), test rows."],
    )

    plot_slot = st.container()
    st.markdown("##### Make a prediction")
    entered = st.number_input(
        f"Enter a value of {ctx.label(x)}", value=float(model.train[x].mean()), key=f"u4_slr_input_{x}",
        help="Starts at the training average.",
    )
    predicted = model.predict(np.array([entered]))
    result_columns = st.columns(2)
    result_columns[0].metric(f"Predicted {ctx.label(target)}", f"{predicted:.4g}")
    result_columns[1].caption(f"Calculation: {fmt(intercept)} + {fmt(slope)} × {entered:.4g} = {predicted:.4g}")
    with plot_slot:
        st.plotly_chart(
            charts.regression_scatter(model.train, model.test, x, target, ctx.label(x), ctx.label(target),
                                      intercept, slope, (entered, predicted)),
            width="stretch", key="u4_slr_scatter",
        )

    show_actual_vs_predicted(model, "u4_slr_avp")

    with st.expander("What do the slope and intercept mean?", expanded=True):
        direction = "increase" if slope > 0 else "decrease"
        st.markdown(
            f"- **Slope = {slope:.4g}.** When {display_name(x)} goes up by {one_unit(ctx, x)}, the model predicts "
            f"{display_name(target)} will {direction} by {abs(slope):.4g} {unit_phrase(ctx, target)} on average. "
            "It describes a pattern in this data, not a cause.\n"
            f"- **Intercept = {intercept:.4g}.** The predicted {display_name(target)} when {display_name(x)} = 0. "
            f"Here {display_name(x)} ranges from {model.train[x].min():.4g} to {model.train[x].max():.4g}, so the "
            "intercept is mainly a starting point for the line rather than a realistic value.\n"
            "- **How the line is chosen:** least squares picks the slope and intercept that make the sum of "
            "squared gaps between the points and the line as small as possible."
        )
        st.latex(r"\text{slope}=\frac{\operatorname{cov}(x,y)}{\operatorname{var}(x)},\qquad "
                 r"\text{intercept}=\bar y-\text{slope}\cdot\bar x")
    return model


def tab_multiple(ctx: AppContext, candidates: list[str], test_size: float) -> ModelResult | None:
    st.subheader("Multiple linear regression")
    st.write(
        "Several inputs predict one outcome together: **Y = intercept + b₁X₁ + b₂X₂ + …**. "
        "Example: predict PM2.5 from PM10, NO2, temperature and humidity (whichever exist in this dataset)."
    )
    picked = pick_target_and_predictors(ctx, candidates, "multiple")
    if picked is None:
        return None
    target, predictors = picked
    if not predictors:
        st.info("Select at least one independent variable.")
        return None
    try:
        model = fit_model(ctx.analysis_data, "Multiple linear regression", target, predictors, test_size)
    except ModelError as error:
        st.warning(str(error))
        return None

    total_rows = len(model.train) + len(model.test)
    st.caption(rows_note(model))
    if total_rows < 0.6 * model.n_available:
        st.warning(
            f"Only {total_rows} of {model.n_available} rows with a valid {display_name(target)} also have every "
            "selected predictor. Variables with many gaps shrink the data the model can use."
        )
    if len(predictors) == 1:
        st.info("With one predictor this is the same as simple linear regression.")
    st.latex(equation_latex(model))
    metric_cards(
        [("Intercept", f"{model.intercept:.4g}"), ("Predictors", f"{len(predictors)}"),
         ("Test R²", fmt(model.test_metrics["R²"], 3)),
         ("Test RMSE", with_unit(model.test_metrics["RMSE"], ctx.units.get(target, "")))],
        ["Predicted Y when every predictor is 0.", "Number of independent variables used.",
         "Share of Y's variation explained, measured on unseen test rows.",
         "Typical prediction error (same unit as Y), test rows."],
    )

    st.markdown("##### Intercept and coefficients")
    show_coefficient_table(model, ctx)

    st.markdown("##### Make a prediction")
    st.caption("Enter a value for each predictor (they start at the training averages).")
    input_columns = st.columns(min(len(predictors), 4))
    entered = [
        input_columns[index % len(input_columns)].number_input(
            ctx.label(name), value=float(model.train[name].mean()), key=f"u4_mlr_input_{name}")
        for index, name in enumerate(predictors)
    ]
    st.metric(f"Predicted {ctx.label(target)}", f"{model.predict(np.array(entered)):.4g}")

    show_actual_vs_predicted(model, "u4_mlr_avp")

    with st.expander("How to read each coefficient", expanded=True):
        lines = []
        for name, coefficient in zip(predictors, model.coefficients):
            direction = "higher" if coefficient > 0 else "lower"
            lines.append(
                f"- **{display_name(name)} = {coefficient:.4g}:** keeping the other predictors the same, a "
                f"{one_unit(ctx, name)} increase in {display_name(name)} goes with a predicted "
                f"{abs(coefficient):.4g} {unit_phrase(ctx, target)} {direction} {display_name(target)}."
            )
        st.markdown("\n".join(lines))
        st.markdown(
            f"- **Intercept = {model.intercept:.4g}:** the prediction when every predictor is 0 (often outside "
            "the observed data, so it is just a starting point for the equation).\n"
            "- **Contribution:** a positive coefficient pushes the prediction up as that variable increases, a "
            "negative one pushes it down. Coefficients are in different units, so compare their *direction*, "
            "not their raw size.\n"
            "- These describe associations in the data, not causes."
        )
    return model


def tab_assumptions(models: dict[str, ModelResult | None]) -> None:
    st.subheader("Regression assumptions")
    st.write(
        "A regression line is only trustworthy when its errors (**residual = actual − predicted**) "
        "behave sensibly. The five standard assumptions are checked below using the training residuals."
    )
    available = {name: model for name, model in models.items() if model is not None}
    if not available:
        st.info("Fit a simple or multiple regression first (previous tabs).")
        return
    choice = st.radio("Model to check", list(available), horizontal=True, key="u4_assump_model")
    model = available[choice]
    results = run_assumption_checks(model)

    summary = pd.DataFrame([
        {"Assumption": c["Assumption"], "Rule of thumb": c["Rule of thumb"], "Result": verdict(c)}
        for c in results["checks"]
    ])
    st.dataframe(summary, hide_index=True, width="stretch")
    flagged = [c for c in results["checks"] if c["violated"]]
    if flagged:
        st.warning(
            f"{len(flagged)} check(s) raised a warning: " + ", ".join(c["Assumption"].lower() for c in flagged)
            + ". Use the model's results with caution."
        )
    else:
        st.success("No assumption check raised a warning.")

    checks = {c["Assumption"]: c for c in results["checks"]}
    residuals, fitted = results["residuals"], results["fitted"]

    def show_result(check: dict, detail: str) -> None:
        (st.warning if check["violated"] else st.success)(
            f"{detail} " + (check["warning"] if check["violated"] else "No warning.")
        )

    with st.expander("1 · Linearity — is a straight line appropriate?"):
        st.markdown("**In simple words:** the real relationship should look like a line, not a curve.")
        st.plotly_chart(charts.residual_vs_fitted(fitted, residuals), width="stretch", key="u4_resid_linearity")
        st.caption("The red line shows the average residual along the predictions. Near zero everywhere = good; "
                   "a U-shape or wave = curvature.")
        check = checks["Linearity"]
        show_result(check, f"Largest average residual = {check['Value']:.2f} standard deviations.")
    with st.expander("2 · Independence of errors — is each error unrelated to the next?"):
        st.markdown("**In simple words:** one hour's mistake should not predict the next hour's mistake.")
        st.plotly_chart(charts.residuals_over_time(model.train["datetime"], residuals), width="stretch",
                        key="u4_resid_time")
        st.caption("Long runs above or below zero mean neighbouring errors are alike.")
        check = checks["Independence of errors"]
        show_result(check, f"Correlation of neighbouring residuals = {results['lag1']:.2f}.")
    with st.expander("3 · Homoscedasticity — is the error size roughly constant?"):
        st.markdown("**In simple words:** the scatter around the line should have the same width everywhere. "
                    "A funnel (narrow, then wide) breaks this.")
        st.plotly_chart(charts.residual_vs_fitted(fitted, residuals), width="stretch", key="u4_resid_variance")
        check = checks["Constant variance (homoscedasticity)"]
        show_result(check, f"Residual spread (top ÷ bottom third of predictions) = {check['Value']:.2f}.")
    with st.expander("4 · Normality of residuals — are errors roughly bell-shaped?"):
        st.markdown("**In simple words:** most errors should be small, with large errors rare and balanced on both sides.")
        left, right = st.columns(2)
        left.plotly_chart(charts.residual_histogram(residuals), width="stretch", key="u4_resid_hist")
        right.plotly_chart(charts.qq_plot(residuals), width="stretch", key="u4_resid_qq")
        st.caption("In the Q-Q plot, points close to the line mean normal-looking residuals.")
        check = checks["Normality of residuals"]
        show_result(check, f"Skewness = {results['skewness']:.2f}.")
    with st.expander("5 · Multicollinearity — are the predictors too similar to each other?"):
        st.markdown("**In simple words:** if two inputs say almost the same thing, the model cannot tell which "
                    "deserves the credit. VIF < 5 is fine, 5–10 is moderate, ≥ 10 is serious.")
        vif = results["vif"]
        if vif is None:
            st.info("Multicollinearity needs two or more predictors. Use the multiple regression model to see VIF values.")
        else:
            names = [display_name(p) for p in model.predictors]
            st.plotly_chart(
                charts.horizontal_bars(names, [float(min(v, 100)) for v in vif], "Variance Inflation Factor (VIF)",
                                       "VIF", [(5, "5"), (10, "10")]),
                width="stretch", key="u4_vif",
            )
            st.dataframe(pd.DataFrame({"Predictor": names, "VIF": vif}).style.format({"VIF": "{:.2f}"}),
                         hide_index=True, width="stretch")
            check = checks["No multicollinearity"]
            show_result(check, f"Largest VIF = {check['Value']:.2f}.")


def tab_evaluation(ctx: AppContext, models: dict[str, ModelResult | None]) -> None:
    st.subheader("Model evaluation metrics")
    st.write(
        "A model must be judged on rows it has **never seen**. All numbers below use the held-out test rows "
        "from a fixed-seed (reproducible) split."
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
            [("MAE", with_unit(metrics["MAE"], unit)), ("MSE", fmt(metrics["MSE"], 4)),
             ("RMSE", with_unit(metrics["RMSE"], unit)), ("R² score", fmt(metrics["R²"], 3))],
            ["Mean Absolute Error — lower is better.", "Mean Squared Error — lower is better.",
             "Root Mean Squared Error — lower is better.", "Coefficient of determination — higher is better (max 1)."],
        )
        st.caption(f"Predicting {display_name(model.target)} from {', '.join(display_name(p) for p in model.predictors)} · "
                   f"{len(model.test)} test rows · {len(model.train)} training rows.")

    with st.expander("What does each metric mean, and which direction is better?", expanded=True):
        st.markdown(
            "| Metric | Meaning | Better when |\n|---|---|---|\n"
            "| **MAE** | Average size of the prediction mistakes (same unit as Y). | **Lower** |\n"
            "| **MSE** | Average of the squared mistakes; big mistakes count more. | **Lower** |\n"
            "| **RMSE** | Square root of MSE: a typical mistake in the unit of Y. | **Lower** |\n"
            "| **R²** | Share of Y's variation the model explains. 1 = perfect, 0 = no better than always "
            "predicting the average, negative = worse than that. | **Higher** |"
        )
        st.latex(r"MAE=\tfrac1n\sum|y-\hat y|,\quad MSE=\tfrac1n\sum(y-\hat y)^2,\quad "
                 r"RMSE=\sqrt{MSE},\quad R^2=1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}")

    st.markdown("#### Simple vs multiple regression")
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
                f"Both models use the same {n_train + n_test} rows (valid for every variable of either model) and the "
                f"same {n_train}/{n_test} train/test split, so the comparison is fair."
            )
            st.dataframe(
                table.style.format({"MAE": "{:.4g}", "MSE": "{:.4g}", "RMSE": "{:.4g}", "R²": "{:.3f}"})
                .highlight_min(subset=["MAE", "MSE", "RMSE"], color="rgba(0,204,150,0.35)")
                .highlight_max(subset=["R²"], color="rgba(0,204,150,0.35)"),
                width="stretch",
            )
            st.plotly_chart(charts.metric_comparison_bars(table, ["MAE", "RMSE"]), width="stretch", key="u4_compare")
            better = table["RMSE"].idxmin()
            st.info(f"The lower test RMSE belongs to the **{better}**. (Green cells mark the better value for each metric.)")


def interpretation_paragraphs(model: ModelResult, ctx: AppContext) -> list[str]:
    """Build the automatic plain-language summary from the REAL model results."""
    target, unit = display_name(model.target), unit_phrase(ctx, model.target)
    metrics = model.test_metrics
    paragraphs = [
        f"The model predicts **{target}** from **{', '.join(display_name(p) for p in model.predictors)}** using "
        f"{len(model.train)} training rows and was tested on {len(model.test)} unseen rows."
    ]
    for name, coefficient in zip(model.predictors, model.coefficients):
        direction = "higher" if coefficient > 0 else "lower"
        qualifier = " keeping the other predictors the same," if len(model.predictors) > 1 else ""
        paragraphs.append(
            f"- **{display_name(name)}** has a {'positive' if coefficient > 0 else 'negative'} coefficient "
            f"({coefficient:.4g}): when it is {one_unit(ctx, name)} higher,{qualifier} predicted {target} is "
            f"{abs(coefficient):.4g} {unit} {direction} on average."
        )
    paragraphs.append(
        f"On the test rows, R² = **{fmt(metrics['R²'], 3)}**, so the model captures {r2_words(metrics['R²'])} in {target}. "
        f"The typical prediction error (RMSE) is **{fmt(metrics['RMSE'], 4)} {unit}** and the average absolute error "
        f"(MAE) is {fmt(metrics['MAE'], 4)} {unit}."
    )
    paragraphs.append(
        "These results describe **associations** in this station's data. They do not show that changing one "
        "variable would cause a change in another."
    )
    return paragraphs


def limitations(model: ModelResult, results: dict) -> list[str]:
    items = []
    flagged = [c["Assumption"] for c in results["checks"] if c["violated"]]
    if flagged:
        items.append("Assumption checks that raised warnings: " + ", ".join(flagged).lower() + ".")
    if abs(results["lag1"]) > 0.3:
        items.append("Neighbouring hourly readings are alike, so the data holds less independent information than "
                     "the number of rows suggests and the test scores may look better than real accuracy.")
    total = len(model.train) + len(model.test)
    if total < 100:
        items.append(f"Only {total} rows were used, so the results can change a lot with new data.")
    if total < 0.8 * model.n_available:
        items.append(f"{model.n_available - total} rows with a valid {display_name(model.target)} were dropped because "
                     "a selected predictor was missing.")
    items.extend([
        "It is a straight-line (linear) model; curved relationships are not captured.",
        "It uses one station and the selected dates; results may not apply to other places or seasons.",
        "Predictions outside the range of the training data are unreliable.",
        "Other drivers of air quality (traffic, season, weather not in the data) may be missing.",
        "Sensor errors and suspicious zero readings can affect the fit (see the Data Quality page).",
    ])
    return items


def tab_interpretation(ctx: AppContext, models: dict[str, ModelResult | None]) -> None:
    st.subheader("Interpretation of regression results")
    available = {name: model for name, model in models.items() if model is not None}
    if not available:
        st.info("Fit a simple or multiple regression first (previous tabs).")
        return
    choice = st.radio("Model to interpret", list(available), horizontal=True, key="u4_interp_model")
    model = available[choice]
    results = run_assumption_checks(model)

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
        for paragraph in interpretation_paragraphs(model, ctx):
            st.markdown(paragraph)

    left, right = st.columns(2)
    left.plotly_chart(
        charts.actual_vs_predicted(model.test[model.target].to_numpy(), model.test_pred, display_name(model.target)),
        width="stretch", key="u4_interp_avp",
    )
    with right:
        st.markdown("#### Limitations of the model")
        for item in limitations(model, results):
            st.markdown(f"- {item}")


def tab_workflow(ctx: AppContext, models: dict[str, ModelResult | None], shared: dict) -> None:
    st.subheader("End-to-end data analytics workflow")
    st.write("Every AirStat analysis follows the same eight steps. The cards show live values from the dataset "
             "and model you currently have selected.")

    numeric = ctx.analysis_data[ctx.variables].apply(pd.to_numeric, errors="coerce")
    missing_pct = float(numeric.isna().mean().mean()) if not numeric.empty else float("nan")
    correlation = shared.get("correlation")
    best = models.get("Multiple linear regression") or models.get("Simple linear regression")

    steps = [
        ("Dataset Selection", f"{ctx.city} · {ctx.station}", "Sidebar", "Pick city, station, date range"),
        ("Data Collection", f"{len(ctx.data):,} hourly rows", "Overview", "OpenAQ sensors, cached locally"),
        ("Cleaning & Preprocessing", f"{missing_pct:.1%} values missing", "Data Quality",
         "Types, duplicates, missing values, zeros"),
        ("Exploratory Data Analysis", f"{len(ctx.variables)} variables", "Unit I", "Summaries, charts, relationships"),
        ("Statistical Analysis",
         (f"r = {correlation[2]:.2f} ({display_name(correlation[0])} vs {display_name(correlation[1])})"
          if correlation else "Probability & inference"),
         "Unit II · Unit III · Correlation tab", "Probability, tests, correlation"),
        ("Regression Modeling",
         f"{len(best.predictors)} predictor(s) → {display_name(best.target)}" if best else "Fit a model in tabs 2–3",
         "Tabs 2 – 3", "Simple & multiple linear regression"),
        ("Model Evaluation",
         f"Test R² {fmt(best.test_metrics['R²'], 3)} · RMSE {fmt(best.test_metrics['RMSE'], 3)}" if best
         else "MAE · MSE · RMSE · R²", "Tabs 4 – 5", "Assumptions + test-set metrics"),
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
        + "".join(cards) + "</div>",
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
        test_percent = st.slider("Test share of the data", 10, 40, 20, 5, format="%d%%", key="u4_test_size",
                                 help="Share of rows kept aside to test the model.")
        st.caption(
            f"Rows are shuffled with a fixed seed ({RANDOM_STATE}), so the train/test split is the same every run. "
            "Rows with missing values are dropped (never filled in), real zero readings are kept, and timestamps "
            "are never used as predictors."
        )
    test_size = test_percent / 100

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
        models["Simple linear regression"] = tab_simple(ctx, candidates, test_size)
    with tabs[2]:
        models["Multiple linear regression"] = tab_multiple(ctx, candidates, test_size)
    with tabs[3]:
        tab_assumptions(models)
    with tabs[4]:
        tab_evaluation(ctx, models)
    with tabs[5]:
        tab_interpretation(ctx, models)
    with tabs[6]:
        tab_workflow(ctx, models, shared)
