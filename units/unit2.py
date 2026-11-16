"""Unit II: probability and probability distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binom, norm, poisson

from components.common import AppContext, probability_sentence
from utils.config import display_name
from utils.data_cleaning import valid_series


def render(ctx: AppContext) -> None:
    render_unit_ii(
        data=ctx.analysis_data,
        pollutant=ctx.pollutant,
        threshold=ctx.threshold,
        unit=ctx.unit,
        variable_y=ctx.variable_y,
    )


def render_unit_ii(
    data: pd.DataFrame,
    pollutant: str,
    threshold: float,
    unit: str,
    variable_y: str,
) -> None:
    """Render the Unit II probability and distribution topics in tabs."""
    values = valid_series(data, pollutant)
    if len(values) < 2:
        st.warning("Insufficient valid data for this analysis.")
        return

    st.subheader(f"Unit II: Probability and Distributions — {display_name(pollutant)}")
    tabs = st.tabs(
        [
            "Probability",
            "Bayes Theorem",
            "Random Variables",
            "Binomial",
            "Poisson",
            "Normal Distribution",
            "Z-Score",
            "Normal Approximation",
            "Probability Insights",
        ]
    )

    exceed = values > threshold
    p_exceed = float(exceed.mean())

    # Event B uses the comparison variable's OWN threshold. (Applying the
    # pollutant's threshold to a variable with a different scale/unit would
    # make the conditional probability meaningless.)
    comparison_values = valid_series(data, variable_y) if variable_y in data else pd.Series(dtype=float)
    threshold_b = threshold
    if variable_y != pollutant and not comparison_values.empty:
        threshold_b = st.number_input(
            f"Threshold for the comparison variable {display_name(variable_y)} (event B)",
            value=float(comparison_values.median()),
            key=f"unit2_threshold_b_{variable_y}",
            help="Event A = selected pollutant above its threshold. "
                 "Event B = comparison variable above this threshold. "
                 "Choose the comparison variable (Variable Y) in the sidebar.",
        )
    with tabs[0]:
        st.markdown("### Basic probability")
        st.write(
            f"Success is defined as `{display_name(pollutant)} > {threshold:g} {unit}`."
        )
        st.metric("P(pollutant exceeds threshold)", probability_sentence(p_exceed))
        st.latex(
            r"P(E) = \frac{\text{number of exceedances}}{\text{valid observations}}"
        )
        st.write(
            f"{int(exceed.sum())} of {len(values)} valid observations exceed the threshold."
        )
        paired = (
            data[[pollutant, variable_y]].apply(pd.to_numeric, errors="coerce").dropna()
        )
        if len(paired) >= 2 and variable_y != pollutant:
            condition = paired[variable_y] > threshold_b
            conditional = (
                float((paired[pollutant] > threshold)[condition].mean())
                if condition.any()
                else np.nan
            )
            st.markdown("### Conditional probability")
            st.metric(
                f"P({display_name(pollutant)} > {threshold:g} | {display_name(variable_y)} > {threshold_b:g})",
                (
                    "Insufficient data"
                    if pd.isna(conditional)
                    else probability_sentence(conditional)
                ),
            )
            st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
        else:
            st.info(
                "Choose a different comparison variable with at least two paired observations."
            )

    with tabs[1]:
        st.markdown("### Bayes theorem")
        st.write(
            "Define A as the selected pollutant exceeding its threshold and B as the comparison variable exceeding its own threshold (set above the tabs)."
        )
        paired = (
            data[[pollutant, variable_y]].apply(pd.to_numeric, errors="coerce").dropna()
        )
        if len(paired) < 2 or variable_y == pollutant:
            st.warning(
                "Choose two different variables with at least two paired observations."
            )
        else:
            event_a = paired[pollutant] > threshold
            event_b = paired[variable_y] > threshold_b
            p_a, p_b = float(event_a.mean()), float(event_b.mean())
            p_b_given_a = float(event_b[event_a].mean()) if event_a.any() else np.nan
            p_a_given_b = float(event_a[event_b].mean()) if event_b.any() else np.nan
            st.write(f"P(A) = {probability_sentence(p_a)}")
            st.write(f"P(B) = {probability_sentence(p_b)}")
            st.write(f"P(B|A) = {probability_sentence(p_b_given_a)}")
            st.write(f"P(A|B) direct = {probability_sentence(p_a_given_b)}")
            bayes = p_b_given_a * p_a / p_b if p_b else np.nan
            st.latex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}")
            st.metric(
                "Bayes result",
                "Insufficient data" if pd.isna(bayes) else probability_sentence(bayes),
            )
            if not pd.isna(bayes):
                st.info(
                    "The Bayes result should match the direct conditional probability up to rounding."
                )

    with tabs[2]:
        st.markdown("### Random variable")
        st.write(
            f"Each valid {display_name(pollutant)} measurement can be viewed as a "
            "continuous random variable because it is a numerical measurement."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Distribution information": [
                        "Valid observations",
                        "Minimum",
                        "Mean",
                        "Standard deviation",
                        "Maximum",
                    ],
                    "Value": [
                        str(len(values)),
                        f"{values.min():.4g}",
                        f"{values.mean():.4g}",
                        f"{values.std():.4g}",
                        f"{values.max():.4g}",
                    ],
                }
            ),
            hide_index=True,
        )
        st.write("The sample values below are observations from this random variable.")
        st.dataframe(values.head(15).rename("Value").to_frame(), width="stretch")

        st.markdown("### Discrete vs continuous random variables")
        counts_source = data[["datetime", pollutant]].copy()
        counts_source[pollutant] = pd.to_numeric(counts_source[pollutant], errors="coerce")
        counts_source = counts_source.dropna(subset=[pollutant])
        daily_counts = (
            counts_source.assign(day=counts_source["datetime"].dt.date, exceeded=counts_source[pollutant] > threshold)
            .groupby("day")["exceeded"].sum().astype(int)
        )
        left, right = st.columns(2)
        left.markdown(
            f"**Continuous:** the concentration of {display_name(pollutant)} itself. Any value in a range is possible, "
            "so probability comes from an area under a **density curve (PDF)**; P(X = exact value) is 0."
        )
        right.markdown(
            "**Discrete:** the *number of hours per day above the threshold*. Only whole numbers 0, 1, 2, … occur, so each "
            "value has its own probability in a **probability mass function (PMF)**."
        )
        if len(daily_counts) >= 2:
            pmf = daily_counts.value_counts(normalize=True).sort_index()
            outcomes, probabilities = pmf.index.to_numpy(dtype=float), pmf.to_numpy()
            expected = float(np.sum(outcomes * probabilities))                        # E[X] = sum x * P(x)
            variance = float(np.sum((outcomes - expected) ** 2 * probabilities))      # Var(X) = sum (x-E[X])^2 P(x)
            metric_columns = st.columns(3)
            metric_columns[0].metric("Days observed", f"{len(daily_counts)}")
            metric_columns[1].metric("Expected value E[X] (hours/day)", f"{expected:.3f}")
            metric_columns[2].metric("Variance Var(X)", f"{variance:.3f}")
            pmf_figure = px.bar(
                x=pmf.index, y=pmf.values,
                labels={"x": "Hours above threshold in a day", "y": "Probability"},
                title="Empirical PMF of the discrete random variable",
            )
            cdf_figure = px.line(
                x=pmf.index, y=pmf.cumsum().values, markers=True, line_shape="hv",
                labels={"x": "Hours above threshold in a day", "y": "P(X ≤ x)"},
                title="Cumulative distribution function (CDF)",
            )
            pmf_column, cdf_column = st.columns(2)
            pmf_column.plotly_chart(pmf_figure, width="stretch")
            cdf_column.plotly_chart(cdf_figure, width="stretch")
            st.latex(r"E[X]=\sum_x x\,P(X=x),\qquad \operatorname{Var}(X)=\sum_x (x-E[X])^2P(X=x)")

    with tabs[3]:
        st.markdown("### Binomial distribution")
        n = st.number_input(
            "Number of trials (n)", min_value=1, max_value=1000, value=10, key="binom_n"
        )
        k = st.number_input(
            "Number of successes (k)",
            min_value=0,
            max_value=1000,
            value=2,
            key="binom_k",
        )
        probability = float(binom.pmf(k, n, p_exceed)) if k <= n else 0.0
        st.latex(r"P(X=k) = {n \choose k}p^k(1-p)^{n-k}")
        st.metric(f"P(X = {k})", probability_sentence(probability))
        cumulative = float(binom.cdf(min(k, n), n, p_exceed))
        binomial_columns = st.columns(3)
        binomial_columns[0].metric(f"P(X ≤ {k})", probability_sentence(cumulative))
        binomial_columns[1].metric("Mean n·p", f"{n * p_exceed:.3f}")
        binomial_columns[2].metric("Variance n·p·(1−p)", f"{n * p_exceed * (1 - p_exceed):.3f}")
        st.caption(
            f"Here p = {p_exceed:.4f} is the share of valid hourly readings above the threshold. Binomial assumes a fixed n, "
            "two outcomes, constant p and independent trials; consecutive hours are somewhat dependent, so treat it as a model."
        )
        chart_k = np.arange(0, n + 1)
        st.plotly_chart(
            px.bar(
                x=chart_k,
                y=binom.pmf(chart_k, n, p_exceed),
                labels={"x": "Successes", "y": "Probability"},
                title="Binomial probabilities",
            ),
            width="stretch",
        )

    with tabs[4]:
        st.markdown("### Poisson distribution")
        daily_data = data[["datetime", pollutant]].copy()
        daily_data[pollutant] = pd.to_numeric(daily_data[pollutant], errors="coerce")
        daily_data = daily_data.dropna(subset=[pollutant])
        daily = (
            daily_data.assign(
                day=daily_data["datetime"].dt.date,
                exceeded=daily_data[pollutant] > threshold,
            )
            .groupby("day")["exceeded"]
            .sum()
        )
        if daily.empty:
            st.warning("Insufficient valid dated data for daily exceedance counts.")
        else:
            lam = float(daily.mean())
            poisson_k = st.number_input(
                "Exceedances in one day (k)",
                min_value=0,
                max_value=100,
                value=1,
                key="poisson_k",
            )
            probability = float(poisson.pmf(poisson_k, lam))
            st.write(f"Estimated λ (average daily exceedances) = **{lam:.4g}**")
            st.latex(r"P(X=k) = e^{-\lambda}\frac{\lambda^k}{k!}")
            st.metric(f"P(X = {poisson_k})", probability_sentence(probability))
            chart_k = np.arange(0, max(10, int(daily.max()) + 5))
            poisson_figure = px.bar(
                x=chart_k,
                y=poisson.pmf(chart_k, lam),
                labels={"x": "Daily exceedances", "y": "Probability"},
                title="Poisson probabilities (bars) vs observed days (line)",
            )
            observed = daily.astype(int).value_counts(normalize=True).reindex(chart_k, fill_value=0.0)
            poisson_figure.add_scatter(x=chart_k, y=observed.values, mode="lines+markers", name="Observed share of days")
            st.plotly_chart(poisson_figure, width="stretch")
            st.caption("If the line follows the bars, the Poisson model describes the daily counts well. Hourly exceedances "
                       "often cluster in the same hours or days, which makes the observed counts more spread out than Poisson predicts.")

    with tabs[5]:
        st.markdown("### Normal distribution")
        mean, standard_deviation = float(values.mean()), float(values.std())
        if standard_deviation == 0 or pd.isna(standard_deviation):
            st.warning("Insufficient variation to fit a normal curve.")
        else:
            histogram = px.histogram(
                values,
                nbins=20,
                histnorm="probability density",
                title="Histogram with fitted normal curve",
            )
            x = np.linspace(values.min(), values.max(), 200)
            histogram.add_scatter(
                x=x,
                y=norm.pdf(x, mean, standard_deviation),
                mode="lines",
                name="Fitted normal curve",
            )
            st.plotly_chart(histogram, width="stretch")
            st.write(
                f"Mean = **{mean:.4g}**, standard deviation = **{standard_deviation:.4g}**. The curve is a simple model of the sample distribution, not a guarantee that the data is perfectly normal."
            )

    with tabs[6]:
        st.markdown("### Z-score")
        entered = st.number_input(
            "Enter a pollutant value", value=float(values.mean()), key="z_value"
        )
        mean, standard_deviation = float(values.mean()), float(values.std())
        if standard_deviation == 0:
            st.warning(
                "A Z-score cannot be calculated when all valid values are identical."
            )
        else:
            z = (entered - mean) / standard_deviation
            st.metric("Z-score", f"{z:.4f}")
            interpretation = (
                "near the mean"
                if abs(z) < 1
                else "above the mean" if z > 0 else "below the mean"
            )
            st.write(
                f"Formula: z = (value − mean) / standard deviation = ({entered:.4g} − {mean:.4g}) / {standard_deviation:.4g}. This value is **{interpretation}**."
            )
            st.markdown("### Standard normal (Z) distribution")
            st.write(
                "Converting values to z-scores rescales any normal distribution to mean 0 and standard deviation 1, "
                "so one table (or `norm.cdf`) gives probabilities for every variable."
            )
            z_columns = st.columns(3)
            z_columns[0].metric("P(Z ≤ z)", probability_sentence(float(norm.cdf(z))))
            z_columns[1].metric("P(Z > z)", probability_sentence(float(norm.sf(z))))
            z_columns[2].metric("P(|Z| > |z|)", probability_sentence(float(2 * norm.sf(abs(z)))))
            grid = np.linspace(-4, 4, 400)
            shade = grid <= z
            z_figure = go.Figure()
            z_figure.add_scatter(x=grid, y=norm.pdf(grid), mode="lines", name="Standard normal")
            z_figure.add_scatter(x=grid[shade], y=norm.pdf(grid[shade]), fill="tozeroy", mode="none",
                                 name=f"P(Z ≤ {z:.2f})")
            z_figure.add_vline(x=float(np.clip(z, -4, 4)), line_dash="dash")
            z_figure.update_layout(title="Standard normal curve (shaded area = P(Z ≤ z))",
                                   xaxis_title="z", yaxis_title="Density")
            st.plotly_chart(z_figure, width="stretch")
            st.caption("This assumes the variable is roughly normal. Air-quality data is usually right-skewed, so read these "
                       "as model-based probabilities (compare with the real share of readings below the value in the Normal tab).")
            st.write(f"Actual share of readings ≤ {entered:.4g}: **{probability_sentence(float((values <= entered).mean()))}**")

    with tabs[7]:
        st.markdown("### Normal approximation to binomial")
        n_approx = st.number_input(
            "Trials (n)", min_value=1, max_value=1000, value=30, key="approx_n"
        )
        p_approx = st.slider(
            "Success probability (p)",
            0.0,
            1.0,
            min(max(p_exceed, 0.01), 0.99),
            key="approx_p",
        )
        k_approx = st.number_input(
            "Successes (k)", min_value=0, max_value=1000, value=10, key="approx_k"
        )
        exact = (
            float(binom.pmf(k_approx, n_approx, p_approx))
            if k_approx <= n_approx
            else 0.0
        )
        mu, sigma = n_approx * p_approx, np.sqrt(n_approx * p_approx * (1 - p_approx))
        approx = (
            float(
                norm.cdf(k_approx + 0.5, mu, sigma)
                - norm.cdf(k_approx - 0.5, mu, sigma)
            )
            if sigma
            else (1.0 if k_approx == mu else 0.0)
        )
        left, right = st.columns(2)
        left.metric("Exact binomial P(X = k)", probability_sentence(exact))
        right.metric("Normal approximation", probability_sentence(approx))
        st.latex(r"P(X=k) \approx P(k-0.5 < Y < k+0.5)")
        approx_k = np.arange(0, int(n_approx) + 1)
        approx_figure = px.bar(
            x=approx_k, y=binom.pmf(approx_k, n_approx, p_approx),
            labels={"x": "Successes", "y": "Probability"},
            title="Exact binomial (bars) vs normal curve (line)",
        )
        if sigma > 0:
            curve_x = np.linspace(0, int(n_approx), 300)
            approx_figure.add_scatter(x=curve_x, y=norm.pdf(curve_x, mu, sigma), mode="lines", name="Normal(np, √(np(1−p)))")
        st.plotly_chart(approx_figure, width="stretch")
        if n_approx * p_approx >= 5 and n_approx * (1 - p_approx) >= 5:
            st.success(f"np = {n_approx * p_approx:.1f} and n(1−p) = {n_approx * (1 - p_approx):.1f} are both ≥ 5: the approximation is appropriate.")
        else:
            st.warning(f"np = {n_approx * p_approx:.1f} or n(1−p) = {n_approx * (1 - p_approx):.1f} is below 5: the normal approximation may be poor here.")
        st.info(
            "The approximation is generally appropriate when np and n(1-p) are both at least 5. Continuity correction of 0.5 is used."
        )

    with tabs[8]:
        st.markdown("### Probability insights")
        st.write(
            f"- Estimated probability of exceeding the threshold: **{probability_sentence(p_exceed)}**."
        )
        insight_daily = (
            data[["datetime", pollutant]]
            .assign(
                value=lambda frame: pd.to_numeric(frame[pollutant], errors="coerce")
            )
            .dropna(subset=["value"])
            .assign(
                day=lambda frame: frame["datetime"].dt.date,
                exceeded=lambda frame: frame["value"] > threshold,
            )
            .groupby("day")["exceeded"]
            .sum()
        )
        average_daily = float(insight_daily.mean()) if not insight_daily.empty else 0.0
        st.write(
            f"- Average daily exceedances in the selected period: **{average_daily:.2f}**."
        )
        st.write(
            "- A Z-score with absolute value at least 2 is a simple indicator of an unusually distant observation; it is not a formal hypothesis test."
        )
        hourly = (
            data[["datetime", pollutant]]
            .assign(value=lambda frame: pd.to_numeric(frame[pollutant], errors="coerce"))
            .dropna(subset=["value"])
        )
        if len(hourly) >= 24:
            hourly["Hour (IST)"] = hourly["datetime"].dt.tz_convert("Asia/Kolkata").dt.hour
            profile = hourly.groupby("Hour (IST)")["value"].apply(lambda v: float((v > threshold).mean())).reset_index(name="P(exceed)")
            st.plotly_chart(
                px.bar(profile, x="Hour (IST)", y="P(exceed)",
                       title=f"Application: probability of exceeding {threshold:g} by hour of day"),
                width="stretch",
            )
            peak = profile.loc[profile["P(exceed)"].idxmax()]
            st.write(f"- Exceedances are most likely around **{int(peak['Hour (IST)']):02d}:00 IST** "
                     f"(probability {peak['P(exceed)']:.1%}), which could guide when to issue alerts or limit outdoor activity.")
