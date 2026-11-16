"""Reusable Plotly chart builders (used mainly by Unit IV).

Each function takes plain numbers / DataFrames and returns a Plotly figure,
so the unit pages stay short and the drawing code lives in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Consistent colours across all Unit IV charts.
TRAIN_COLOR = "#636EFA"   # Plotly default blue
TEST_COLOR = "#EF553B"    # Plotly default red
LINE_COLOR = "#00CC96"    # green for fitted lines / references


def correlation_heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    """Annotated heatmap of a correlation matrix, colour-centred on 0."""
    figure = px.imshow(
        matrix,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=title,
    )
    figure.update_layout(coloraxis_colorbar={"title": "r"})
    return figure


def scatter_with_trend(
    frame: pd.DataFrame, x: str, y: str, x_label: str, y_label: str
) -> go.Figure:
    """Scatter plot with a least-squares trend line (visual aid)."""
    figure = px.scatter(
        frame, x=x, y=y, opacity=0.6, labels={x: x_label, y: y_label},
        hover_data={"datetime": "|%Y-%m-%d %H:%M"} if "datetime" in frame else None,
        title=f"{y_label} vs {x_label}",
    )
    if frame[x].nunique() > 1:
        slope, intercept = np.polyfit(frame[x], frame[y], 1)
        xs = np.array([frame[x].min(), frame[x].max()])
        figure.add_scatter(
            x=xs, y=intercept + slope * xs, mode="lines", name="Trend line",
            line={"color": LINE_COLOR, "width": 3},
        )
    return figure


def regression_scatter(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    intercept: float,
    slope: float,
    prediction_point: tuple[float, float] | None = None,
) -> go.Figure:
    """Train/test scatter plot with the fitted simple-regression line."""
    figure = go.Figure()
    figure.add_scatter(
        x=train[x], y=train[y], mode="markers", name="Training data",
        marker={"color": TRAIN_COLOR, "opacity": 0.45},
    )
    figure.add_scatter(
        x=test[x], y=test[y], mode="markers", name="Test data",
        marker={"color": TEST_COLOR, "opacity": 0.7},
    )
    all_x = pd.concat([train[x], test[x]])
    xs = np.array([all_x.min(), all_x.max()])
    figure.add_scatter(
        x=xs, y=intercept + slope * xs, mode="lines", name="Regression line",
        line={"color": LINE_COLOR, "width": 3},
    )
    if prediction_point is not None:
        figure.add_scatter(
            x=[prediction_point[0]], y=[prediction_point[1]], mode="markers",
            name="Your prediction",
            marker={"symbol": "star", "size": 16, "color": "gold",
                    "line": {"color": "black", "width": 1}},
        )
    figure.update_layout(
        title=f"{y_label} vs {x_label} with regression line",
        xaxis_title=x_label, yaxis_title=y_label,
    )
    return figure


def actual_vs_predicted(
    actual: np.ndarray, predicted: np.ndarray, label: str, title_suffix: str = "test data"
) -> go.Figure:
    """Scatter of actual vs predicted values with the perfect-fit diagonal."""
    figure = go.Figure()
    figure.add_scatter(
        x=actual, y=predicted, mode="markers", name="Predictions",
        marker={"color": TEST_COLOR, "opacity": 0.6},
    )
    low = float(min(np.min(actual), np.min(predicted)))
    high = float(max(np.max(actual), np.max(predicted)))
    figure.add_scatter(
        x=[low, high], y=[low, high], mode="lines", name="Perfect prediction (y = x)",
        line={"color": "gray", "dash": "dash"},
    )
    figure.update_layout(
        title=f"Actual vs predicted {label} ({title_suffix})",
        xaxis_title=f"Actual {label}", yaxis_title=f"Predicted {label}",
    )
    return figure


def residual_vs_fitted(fitted: np.ndarray, residuals: np.ndarray) -> go.Figure:
    """Residuals against fitted values, with a binned-mean trend line.

    A flat trend near zero supports linearity; a funnel shape suggests
    non-constant variance.
    """
    figure = go.Figure()
    figure.add_scatter(
        x=fitted, y=residuals, mode="markers", name="Residuals",
        marker={"color": TRAIN_COLOR, "opacity": 0.5},
    )
    bins = pd.qcut(pd.Series(fitted), q=min(10, max(2, len(fitted) // 20)), duplicates="drop")
    trend = pd.DataFrame({"fitted": fitted, "resid": residuals}).groupby(
        bins, observed=True
    ).mean()
    figure.add_scatter(
        x=trend["fitted"], y=trend["resid"], mode="lines+markers",
        name="Average residual per bin", line={"color": TEST_COLOR, "width": 3},
    )
    figure.add_hline(y=0, line_dash="dash", line_color="gray")
    figure.update_layout(
        title="Residuals vs predicted values",
        xaxis_title="Predicted value", yaxis_title="Residual (actual − predicted)",
    )
    return figure


def residual_histogram(residuals: np.ndarray) -> go.Figure:
    """Histogram of residuals with a fitted normal curve for comparison."""
    figure = px.histogram(
        x=residuals, nbins=35, histnorm="probability density",
        labels={"x": "Residual"}, title="Residual distribution",
    )
    spread = float(np.std(residuals, ddof=1))
    if spread > 0:
        grid = np.linspace(residuals.min(), residuals.max(), 200)
        figure.add_scatter(
            x=grid, y=stats.norm.pdf(grid, residuals.mean(), spread),
            mode="lines", name="Normal curve", line={"color": TEST_COLOR, "width": 3},
        )
    return figure


def qq_plot(residuals: np.ndarray) -> go.Figure:
    """Normal Q-Q plot: points near the line mean roughly normal residuals."""
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    figure = go.Figure()
    figure.add_scatter(
        x=theoretical, y=ordered, mode="markers", name="Residual quantiles",
        marker={"color": TRAIN_COLOR, "opacity": 0.6},
    )
    figure.add_scatter(
        x=[theoretical.min(), theoretical.max()],
        y=[intercept + slope * theoretical.min(), intercept + slope * theoretical.max()],
        mode="lines", name="Reference line", line={"color": TEST_COLOR},
    )
    figure.update_layout(
        title="Normal Q-Q plot of residuals",
        xaxis_title="Theoretical normal quantiles", yaxis_title="Sample residual quantiles",
    )
    return figure


def residuals_over_time(times: pd.Series, residuals: np.ndarray) -> go.Figure:
    """Residuals in time order; visible runs/waves suggest dependence."""
    figure = go.Figure()
    figure.add_scatter(
        x=times, y=residuals, mode="lines+markers", name="Residual",
        marker={"size": 4}, line={"width": 1, "color": TRAIN_COLOR},
    )
    figure.add_hline(y=0, line_dash="dash", line_color="gray")
    figure.update_layout(
        title="Residuals in time order", xaxis_title="Time", yaxis_title="Residual",
    )
    return figure


def horizontal_bars(
    names: list[str], values: list[float], title: str, x_title: str,
    reference_lines: list[tuple[float, str]] | None = None,
) -> go.Figure:
    """Simple horizontal bar chart (VIF values, standardised coefficients, ...)."""
    figure = px.bar(
        x=values, y=names, orientation="h", labels={"x": x_title, "y": ""}, title=title,
    )
    for position, text in reference_lines or []:
        figure.add_vline(
            x=position, line_dash="dash", line_color="gray", annotation_text=text,
        )
    figure.update_layout(yaxis={"autorange": "reversed"})
    return figure


def metric_comparison_bars(table: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Grouped bars comparing models on error metrics (lower is better)."""
    long = table.reset_index(names="Model").melt(
        id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value"
    )
    return px.bar(
        long, x="Metric", y="Value", color="Model", barmode="group",
        title="Test-set error metrics by model (lower is better)",
    )
