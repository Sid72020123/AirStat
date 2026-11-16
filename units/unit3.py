"""Unit III: sampling distributions and statistical inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import f_oneway, levene, norm, t, ttest_1samp, ttest_ind, ttest_rel

from components.common import AppContext, render_statistical_conclusion
from utils.config import display_name
from utils.data_cleaning import daily_average_data, finite_values


# Alternative hypotheses: label shown to the user -> scipy `alternative` argument.
TAILS = {
    "Two-tailed (H₁: μ ≠ μ₀)": "two-sided",
    "Right-tailed (H₁: μ > μ₀)": "greater",
    "Left-tailed (H₁: μ < μ₀)": "less",
}
TAIL_SYMBOL = {"two-sided": "≠", "greater": ">", "less": "<"}


def z_p_value(z_stat: float, alternative: str) -> float:
    """p-value of a Z statistic: area in the tail(s) beyond the observed value."""
    if alternative == "greater":
        return float(norm.sf(z_stat))
    if alternative == "less":
        return float(norm.cdf(z_stat))
    return float(2 * norm.sf(abs(z_stat)))


def render(ctx: AppContext) -> None:
    render_unit_iii(data=ctx.analysis_data, pollutant=ctx.pollutant, unit=ctx.unit)


def render_unit_iii(
    data: pd.DataFrame,
    pollutant: str,
    unit: str,
) -> None:
    """Render Unit III sampling and statistical inference demonstrations."""
    st.subheader(
        f"Unit III: Sampling and Statistical Inference — {display_name(pollutant)}"
    )
    tabs = st.tabs(
        [
            "Sampling Distribution",
            "Central Limit Theorem",
            "Estimation & CIs",
            "Hypothesis Basics",
            "One-Sample Z-Test",
            "One-Sample t-Test",
            "Independent t-Test",
            "Paired t-Test",
            "One-Way ANOVA",
            "Statistical Conclusions",
        ]
    )
    values = finite_values(data, pollutant)
    if len(values) < 2:
        st.warning("Insufficient valid observations for Unit III.")
        return

    max_sample = min(500, len(values))
    with tabs[0]:
        st.markdown("### Sampling distribution of the sample mean")
        st.write(
            "A sample mean is the average of one random sample. Repeating the "
            "sampling creates a sampling distribution of those means."
        )
        controls = st.columns(2)
        sample_size = controls[0].slider(
            "Observations in each sample",
            2,
            max_sample,
            min(30, max_sample),
            key="unit3_sampling_size",
        )
        number_samples = controls[1].slider(
            "Number of random samples", 10, 2000, 500, key="unit3_sampling_count"
        )
        rng = np.random.default_rng(42)
        means = np.array(
            [rng.choice(values.to_numpy(), sample_size, replace=False).mean() for _ in range(number_samples)]
        )
        st.metric("Mean of sample means", f"{means.mean():.4g} {unit}")
        st.metric("Standard error (observed)", f"{means.std(ddof=1):.4g} {unit}")
        # Theory: SE = sigma / sqrt(n); the finite-population factor applies because
        # samples are drawn WITHOUT replacement from the available readings.
        population_n = len(values)
        theoretical_se = values.std(ddof=0) / np.sqrt(sample_size) * np.sqrt(
            (population_n - sample_size) / (population_n - 1)
        )
        st.metric("Standard error (theory: σ/√n, finite-population corrected)", f"{theoretical_se:.4g} {unit}")
        st.plotly_chart(
            px.histogram(
                x=means,
                nbins=35,
                labels={"x": f"Sample mean ({unit})"},
                title="Sampling distribution of sample means",
            ),
            width="stretch",
        )

    with tabs[1]:
        st.markdown("### Central Limit Theorem")
        st.write(
            "Even when individual air-quality readings are skewed, averages of "
            "larger random samples tend to look more like a normal distribution."
        )
        controls = st.columns(2)
        clt_size = controls[0].slider(
            "Sample size", 2, max_sample, min(30, max_sample), key="unit3_clt_size"
        )
        clt_count = controls[1].slider(
            "Number of samples", 20, 2000, 500, key="unit3_clt_count"
        )
        rng = np.random.default_rng(123)
        clt_means = np.array(
            [rng.choice(values.to_numpy(), clt_size, replace=False).mean() for _ in range(clt_count)]
        )
        mean, standard_error = clt_means.mean(), clt_means.std(ddof=1)
        if standard_error > 0:
            x = np.linspace(clt_means.min(), clt_means.max(), 120)
            curve = norm.pdf(x, mean, standard_error) * len(clt_means) * (
                clt_means.max() - clt_means.min()
            ) / 35
            figure = px.histogram(clt_means, nbins=35, opacity=0.7, labels={"x": "Sample mean"})
            figure.add_scatter(x=x, y=curve, mode="lines", name="Normal curve")
            st.plotly_chart(figure, width="stretch")
        else:
            st.info("All sample means are identical, so a normal curve cannot be fitted.")
        st.caption(
            f"Using {clt_count} samples of {clt_size} {display_name(pollutant)} observations."
        )

    with tabs[2]:
        st.markdown("### Estimation and confidence intervals")
        confidence = st.selectbox(
            "Confidence level",
            [0.90, 0.95, 0.99],
            index=1,
            format_func=lambda level: f"{level:.0%}",
            key="unit3_confidence",
        )
        alpha = 1 - confidence
        sample_mean = float(values.mean())
        standard_error = float(values.sem())
        margin = float(t.ppf(1 - alpha / 2, len(values) - 1) * standard_error)
        proportion_threshold = st.number_input(
            f"Threshold for sample proportion ({unit})",
            value=float(values.median()),
            key="unit3_proportion_threshold",
        )
        proportion = float((values > proportion_threshold).mean())
        metrics = st.columns(3)
        metrics[0].metric("Point estimate (mean)", f"{sample_mean:.4g} {unit}")
        metrics[1].metric("Mean CI lower", f"{sample_mean - margin:.4g} {unit}")
        metrics[2].metric("Mean CI upper", f"{sample_mean + margin:.4g} {unit}")
        st.metric(
            "Point estimate (proportion above threshold)",
            f"{proportion:.1%}",
        )
        # Normal-approximation interval for a proportion: p ± z * sqrt(p(1-p)/n)
        z_critical = float(norm.ppf(1 - alpha / 2))
        proportion_margin = z_critical * float(np.sqrt(proportion * (1 - proportion) / len(values)))
        st.write(
            f"{confidence:.0%} confidence interval for the proportion above the threshold: "
            f"**{max(0.0, proportion - proportion_margin):.1%} to {min(1.0, proportion + proportion_margin):.1%}**"
        )
        st.caption("Hourly readings next to each other are similar, so these intervals are narrower than they should be; "
                   "daily averages (used in the t-tests below) give a more honest picture.")
        with st.expander("Formula and interpretation"):
            st.latex(r"\bar{x} \pm t_{\alpha/2,\,n-1}\frac{s}{\sqrt{n}}")
            st.write(
                f"We are {confidence:.0%} confident that the population mean is between "
                f"{sample_mean - margin:.4g} and {sample_mean + margin:.4g} {unit}, "
                "under the sampling assumptions."
            )

    with tabs[3]:
        st.markdown("### Hypothesis testing basics")
        st.write(
            "**H₀ (null):** a baseline claim, such as mean PM2.5 = 35. "
            "**H₁ (alternative):** the claim we look for evidence to support."
        )
        st.write(
            "**α (significance level):** the error threshold chosen before testing. "
            "**p-value:** how unusual the data would be if H₀ were true."
        )
        st.write(
            "**Two-tailed** tests detect any difference; **one-tailed** tests detect "
            "a specified increase or decrease. Type I error rejects a true H₀; "
            "Type II error fails to reject a false H₀."
        )
        st.info(
            "Example: H₀ says the average PM2.5 equals a reference value; H₁ says "
            "it is different. A small p-value is evidence against H₀, not proof of H₁."
        )
        st.markdown("#### Formulate your own hypothesis")
        builder = st.columns(3)
        reference = builder[0].number_input(
            f"Reference mean μ₀ ({unit})", value=float(round(values.mean())), key="unit3_basics_reference"
        )
        tail_label = builder[1].selectbox("Alternative hypothesis", list(TAILS), key="unit3_basics_tail")
        alpha_basics = builder[2].select_slider(
            "Choose significance level (α)",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            format_func=lambda level: f"{level:.2f}",
            key="unit3_basics_alpha",
        )
        alternative = TAILS[tail_label]
        st.write(f"**H₀:** μ = {reference:g}  \n**H₁:** μ {TAIL_SYMBOL[alternative]} {reference:g}")
        grid = np.linspace(-4, 4, 400)
        critical = float(norm.ppf(1 - alpha_basics / 2 if alternative == "two-sided" else 1 - alpha_basics))
        if alternative == "two-sided":
            region = (grid <= -critical) | (grid >= critical)
        elif alternative == "greater":
            region = grid >= critical
        else:
            region = grid <= -critical
        figure = go.Figure()
        figure.add_scatter(x=grid, y=norm.pdf(grid), mode="lines", name="Distribution if H₀ is true")
        figure.add_scatter(x=grid[region], y=norm.pdf(grid[region]), mode="markers", name=f"Rejection region (area = α = {alpha_basics:.2f})",
                           marker={"color": "crimson", "size": 4})
        figure.update_layout(title="Standard normal curve: reject H₀ when the test statistic lands in the red region",
                             xaxis_title="Test statistic (z)", yaxis_title="Density")
        st.plotly_chart(figure, width="stretch")
        st.caption(f"Critical value{'s' if alternative == 'two-sided' else ''}: "
                   f"{'±' if alternative == 'two-sided' else ('+' if alternative == 'greater' else '−')}{critical:.3f}.")

    with tabs[4]:
        st.markdown("### One-sample Z-test")
        st.caption(
            "Use this test only when the population standard deviation is known or "
            "is explicitly supplied, not estimated from this sample."
        )
        controls = st.columns(2)
        hypothesized = controls[0].number_input(
            f"Hypothesized mean ({unit})",
            value=float(values.mean()),
            key="unit3_z_mean",
        )
        population_sd = controls[1].number_input(
            f"Known population standard deviation ({unit})",
            min_value=0.000001,
            value=float(values.std(ddof=1)),
            key="unit3_z_sd",
        )
        alpha = st.select_slider(
            "Significance level (α)",
            [0.01, 0.05, 0.10],
            value=0.05,
            key="unit3_z_alpha",
        )
        z_tail = st.selectbox("Alternative hypothesis", list(TAILS), key="unit3_z_tail")
        alternative = TAILS[z_tail]
        # Z = (sample mean - mu0) / (sigma / sqrt(n))
        z_stat = (float(values.mean()) - hypothesized) / (population_sd / np.sqrt(len(values)))
        p_value = z_p_value(z_stat, alternative)
        render_statistical_conclusion(
            f"μ = {hypothesized:g}", f"μ {TAIL_SYMBOL[alternative]} {hypothesized:g}", alpha, "Z-statistic", z_stat, p_value
        )

    with tabs[5]:
        st.markdown("### One-sample t-test")
        hypothesized = st.number_input(
            f"Reference mean ({unit})",
            value=float(values.mean()),
            key="unit3_t_mean",
        )
        alpha = st.select_slider(
            "Significance level (α)",
            [0.01, 0.05, 0.10],
            value=0.05,
            key="unit3_t_alpha",
        )
        t_tail = st.selectbox("Alternative hypothesis", list(TAILS), key="unit3_t_tail")
        alternative = TAILS[t_tail]
        t_stat, p_value = ttest_1samp(values, hypothesized, alternative=alternative)
        render_statistical_conclusion(
            f"μ = {hypothesized:g}", f"μ {TAIL_SYMBOL[alternative]} {hypothesized:g}", alpha, "t-statistic", float(t_stat), float(p_value)
        )
        st.write(f"Degrees of freedom: **{len(values) - 1}**")
        st.plotly_chart(
            px.histogram(values, nbins=35, labels={"value": display_name(pollutant)}, title="Observed values"),
            width="stretch",
        )

    with tabs[6]:
        st.markdown("### Independent t-test (Welch)")
        daily = daily_average_data(data, pollutant)
        group_mode = st.selectbox(
            "Independent groups",
            ["Weekdays vs weekends", "First half vs second half of dates"],
            key="unit3_independent_groups",
        )
        if group_mode.startswith("Week"):
            daily["group"] = pd.to_datetime(daily["day"]).dt.dayofweek.map(
                lambda day: "Weekday" if day < 5 else "Weekend"
            )
        else:
            midpoint = daily["day"].min() + (daily["day"].max() - daily["day"].min()) / 2
            daily["group"] = daily["day"].map(lambda day: "First half" if day <= midpoint else "Second half")
        # Keep each label together with ITS data (groupby sorts the labels alphabetically,
        # so pairing them with an unsorted list of labels could swap the group names).
        grouped_values = {
            label: group.to_numpy() for label, group in daily.groupby("group")[pollutant] if len(group) >= 2
        }
        labels = list(grouped_values)
        groups = [pd.Series(values_) for values_ in grouped_values.values()]
        if len(groups) != 2 or daily["group"].nunique() != 2:
            st.warning("Both independent groups need at least two valid daily averages.")
        else:
            statistic, p_value = ttest_ind(groups[0], groups[1], equal_var=False)
            alpha = st.select_slider(
                "Significance level (α)",
                [0.01, 0.05, 0.10],
                value=0.05,
                key="unit3_independent_alpha",
            )
            difference = float(groups[0].mean() - groups[1].mean())
            se = np.sqrt(groups[0].var(ddof=1) / len(groups[0]) + groups[1].var(ddof=1) / len(groups[1]))
            degrees = (se**4) / (
                (groups[0].var(ddof=1) / len(groups[0])) ** 2 / (len(groups[0]) - 1)
                + (groups[1].var(ddof=1) / len(groups[1])) ** 2 / (len(groups[1]) - 1)
            )
            margin = float(t.ppf(0.975, degrees) * se)
            metrics = st.columns(3)
            metrics[0].metric(f"{labels[0]} mean", f"{groups[0].mean():.4g}")
            metrics[1].metric(f"{labels[1]} mean", f"{groups[1].mean():.4g}")
            metrics[2].metric("Difference CI (95%)", f"{difference - margin:.4g} to {difference + margin:.4g}")
            plot_data = pd.DataFrame({"Group": np.repeat(labels, [len(groups[0]), len(groups[1])]), "Value": np.concatenate(groups)})
            st.plotly_chart(px.box(plot_data, x="Group", y="Value", points="outliers", title="Independent group comparison"), width="stretch")
            render_statistical_conclusion(
                "the two group means are equal", "the two group means are different",
                alpha, "Welch t-statistic", float(statistic), float(p_value)
            )
            st.caption("Daily averages are used so repeated hourly readings do not dominate the comparison.")
            if min(map(len, groups)) < 30:
                st.warning("Each group has fewer than 30 daily averages; consider the normality assumption carefully.")

    with tabs[7]:
        st.markdown("### Paired t-test")
        dated = data[["datetime", pollutant]].copy()
        dated[pollutant] = pd.to_numeric(dated[pollutant], errors="coerce")
        dated["date"] = dated["datetime"].dt.date
        available_dates = sorted(dated["date"].dropna().unique())
        if len(available_dates) < 2:
            st.warning("At least two dates are required for matched observations.")
        else:
            date_columns = st.columns(2)
            date_a = date_columns[0].selectbox("First matched date", available_dates, key="unit3_pair_a")
            date_b = date_columns[1].selectbox(
                "Second matched date", available_dates, index=min(1, len(available_dates) - 1), key="unit3_pair_b"
            )
            if date_a == date_b:
                st.warning("Choose two different dates.")
            else:
                first = (
                    dated[dated["date"] == date_a]
                    .assign(hour=lambda frame: frame["datetime"].dt.hour)
                    .groupby("hour")[pollutant]
                    .mean()
                )
                second = (
                    dated[dated["date"] == date_b]
                    .assign(hour=lambda frame: frame["datetime"].dt.hour)
                    .groupby("hour")[pollutant]
                    .mean()
                )
                paired = pd.concat([first.rename("First"), second.rename("Second")], axis=1).dropna()
                if len(paired) < 2:
                    st.warning("Fewer than two matching hours have valid observations on both dates.")
                else:
                    statistic, p_value = ttest_rel(paired["First"], paired["Second"])
                    alpha = st.select_slider(
                        "Significance level (α)",
                        [0.01, 0.05, 0.10],
                        value=0.05,
                        key="unit3_paired_alpha",
                    )
                    difference = paired["First"] - paired["Second"]
                    st.metric("Mean paired difference", f"{difference.mean():.4g} {unit}")
                    paired_chart = paired.reset_index(names="Hour")
                    st.plotly_chart(
                        px.line(
                            paired_chart,
                            x="Hour",
                            y=["First", "Second"],
                            markers=True,
                            title="Matched hourly observations",
                        ),
                        width="stretch",
                    )
                    render_statistical_conclusion(
                        "the mean matched difference is 0", "the mean matched difference is not 0",
                        alpha, "Paired t-statistic", float(statistic), float(p_value)
                    )
                    st.write(f"Degrees of freedom: **{len(paired) - 1}**")
                    st.caption("Pairs are matched by hour-of-day across the two selected dates; unrelated rows are not paired.")
                    if len(paired) < 30:
                        st.warning("Fewer than 30 matched hours are available; consider the normality assumption carefully.")

    with tabs[8]:
        st.markdown("### One-way ANOVA")
        daily = daily_average_data(data, pollutant)
        if daily.empty:
            st.warning("No valid daily averages are available.")
        else:
            daily["group"] = pd.to_datetime(daily["day"]).dt.strftime("%Y-%m")
            grouped = [group[pollutant].to_numpy() for _, group in daily.groupby("group") if len(group) >= 2]
            group_names = [name for name, group in daily.groupby("group") if len(group) >= 2]
            if len(grouped) < 3:
                st.warning("At least three months with two valid daily averages each are required.")
            else:
                statistic, p_value = f_oneway(*grouped)
                alpha = st.select_slider(
                    "Significance level (α)",
                    [0.01, 0.05, 0.10],
                    value=0.05,
                    key="unit3_anova_alpha",
                )
                plot_data = daily[daily["group"].isin(group_names)]
                st.plotly_chart(px.box(plot_data, x="group", y=pollutant, points="outliers", title="Daily means by month"), width="stretch")
                render_statistical_conclusion(
                    "all group means are equal", "at least one group mean is different",
                    alpha, "F-statistic", float(statistic), float(p_value)
                )
                levene_stat, levene_p = levene(*grouped)
                st.caption(
                    f"Equal-variance check (Levene): p = {levene_p:.3g}. "
                    + ("Variances look similar, so ANOVA's assumption is reasonable." if levene_p >= 0.05
                       else "Variances differ between groups, so treat the F-test result with caution.")
                )
                st.caption("ANOVA uses daily averages grouped by month; it does not identify which pair differs.")
                if min(map(len, grouped)) < 30:
                    st.warning("Some groups have fewer than 30 daily averages; inspect the normality assumption.")

    with tabs[9]:
        st.markdown("### Interpreting statistical conclusions")
        st.write(
            "For every test, compare the p-value with α. If p-value < α, reject H₀. "
            "If p-value ≥ α, fail to reject H₀. Failing to reject H₀ does not prove "
            "that H₀ is true or accepted."
        )
        st.info(
            "These conclusions describe the selected station and date range. "
            "They depend on valid observations and the assumptions shown with each analysis."
        )
