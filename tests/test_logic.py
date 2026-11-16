"""Logic tests for AQI, cleaning, sampling and the Unit IV regression maths.

Run from the project folder:   python -m unittest discover -s tests -v
No network and no API key are needed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from units import unit1, unit4  # noqa: E402
from utils.aqi_calculator import aqi_category, aqi_sub_index, latest_valid_aqi  # noqa: E402
from utils.data_cleaning import clean_data, iqr_outlier_bounds, quality_summary  # noqa: E402


def make_frame(n=300, seed=0, noise=1.0):
    """Synthetic hourly data with a known relationship: y = 5 + 2*x1 - 3*x2 + noise."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(50, 10, n), rng.normal(20, 5, n)
    y = 5 + 2 * x1 - 3 * x2 + rng.normal(0, noise, n)
    times = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"datetime": times, "x1": x1, "x2": x2, "y": y})


class AqiTests(unittest.TestCase):
    def test_sub_index_known_points(self):
        self.assertEqual(aqi_sub_index(0, "pm25"), 0)
        self.assertEqual(aqi_sub_index(30, "pm25"), 50)
        self.assertEqual(aqi_sub_index(60, "pm25"), 100)
        self.assertAlmostEqual(aqi_sub_index(45, "pm10"), 45)
        # hand calculation: 51 + 49 * (36.8375 - 31) / (60 - 31) = 60.86
        self.assertAlmostEqual(aqi_sub_index(36.8375, "pm25"), 60.86, places=2)

    def test_values_between_bands_are_not_lost(self):
        for pollutant, value in [("pm25", 30.5), ("pm10", 50.5), ("co", 0.55), ("no2", 40.5)]:
            self.assertFalse(np.isnan(aqi_sub_index(value, pollutant)), (pollutant, value))

    def test_invalid_inputs(self):
        self.assertTrue(np.isnan(aqi_sub_index(-1, "pm25")))
        self.assertTrue(np.isnan(aqi_sub_index(float("nan"), "no2")))

    def test_categories(self):
        self.assertEqual(aqi_category(40), "Good")
        self.assertEqual(aqi_category(250), "Poor")
        self.assertEqual(aqi_category(None), "Insufficient data")

    def test_latest_valid_aqi_takes_max_and_skips_empty_tail(self):
        times = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
        frame = pd.DataFrame({"datetime": times, "pm25": 45.0, "pm10": 80.0, "no2": 30.0})
        frame.loc[30:, ["pm25", "pm10", "no2"]] = np.nan   # newest 18 hours missing -> <16 readings in 24 h
        aqi, category, subs, stamp = latest_valid_aqi(frame, {})
        self.assertIsNotNone(aqi)
        self.assertEqual(aqi, max(subs.values()))
        self.assertAlmostEqual(subs["pm25"], aqi_sub_index(45, "pm25"))
        self.assertLess(stamp, times[-1])                     # used an earlier valid hour
        self.assertEqual(category, aqi_category(aqi))

    def test_aqi_needs_three_pollutants_including_pm(self):
        times = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
        only_two = pd.DataFrame({"datetime": times, "pm25": 40.0, "no2": 30.0})
        self.assertIsNone(latest_valid_aqi(only_two, {})[0])
        no_pm = pd.DataFrame({"datetime": times, "no2": 30.0, "so2": 20.0, "o3": 40.0})
        self.assertIsNone(latest_valid_aqi(no_pm, {})[0])

    def test_co_unit_handling(self):
        times = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
        base = pd.DataFrame({"datetime": times, "pm25": 40.0, "pm10": 60.0})
        mg = latest_valid_aqi(base.assign(co=1.5), {"co": "mg/m³"})[2]["co"]
        ug = latest_valid_aqi(base.assign(co=1500.0), {"co": "µg/m³"})[2]["co"]
        unknown = latest_valid_aqi(base.assign(co=1.5), {})[2]["co"]   # offline CSV: no unit
        self.assertAlmostEqual(mg, ug)
        self.assertAlmostEqual(mg, unknown)


class CleaningTests(unittest.TestCase):
    def test_clean_data_duplicates_and_types(self):
        raw = pd.DataFrame({
            "datetime": ["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 01:00", "bad"],
            "pm25": ["10", "11", "abc", "5"],
        })
        cleaned = clean_data(raw)
        self.assertEqual(len(cleaned), 2)                      # duplicate + unreadable date dropped
        self.assertEqual(cleaned.attrs["duplicate_timestamps"], 1)
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["pm25"]))
        self.assertTrue(cleaned["pm25"].isna().iloc[1])        # 'abc' became missing

    def test_zeros_are_kept_and_counted(self):
        frame = make_frame(50)
        frame.loc[[3, 4], "x1"] = 0.0
        summary = quality_summary(frame, ["x1"]).iloc[0]
        self.assertEqual(summary["Zeros"], 2)
        self.assertEqual(summary["Valid"], 50)                 # zeros are valid values
        complete = unit4.complete_cases(frame, ["x1", "y"])
        self.assertEqual(len(complete), 50)

    def test_iqr_outliers(self):
        values = pd.Series(list(range(1, 21)) + [200])
        low, high, n_low, n_high = iqr_outlier_bounds(values)
        self.assertEqual((n_low, n_high), (0, 1))
        self.assertLess(low, 1)
        self.assertLess(high, 200)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        frame = make_frame(500)
        self.frame = unit1.sampling_frame(frame, "y")

    def test_sample_sizes_and_reproducibility(self):
        for technique in ("Simple random", "Systematic", "Stratified", "Cluster"):
            first = unit1.draw_sample(self.frame, technique, 60, 7)
            again = unit1.draw_sample(self.frame, technique, 60, 7)
            self.assertTrue(first.equals(again), technique)
            self.assertGreaterEqual(len(first), 50, technique)
            self.assertLessEqual(len(first), len(self.frame))
        self.assertEqual(len(unit1.draw_sample(self.frame, "Simple random", 60, 7)), 60)
        self.assertEqual(unit1.draw_sample(self.frame, "Simple random", 60, 7).index.is_unique, True)

    def test_stratified_covers_every_period(self):
        sample = unit1.draw_sample(self.frame, "Stratified", 80, 1)
        self.assertEqual(set(sample["period"].dropna().unique()), set(self.frame["period"].dropna().unique()))

    def test_descriptive_statistics(self):
        table = unit1.descriptive_statistics(pd.Series([1.0, 2.0, 3.0, 4.0, 10.0]))
        values = dict(zip(table["Statistic"], table["Value"]))
        self.assertEqual(values["Mean"], "4")
        self.assertEqual(values["Median"], "3")
        self.assertEqual(values["Range"], "9")
        self.assertEqual(values["IQR"], "2")


class RegressionMathTests(unittest.TestCase):
    def test_simple_regression_matches_textbook_formulas(self):
        frame = make_frame(400, noise=3)
        x, y = frame["x1"].to_numpy(), frame["y"].to_numpy()
        intercept, coefficients = unit4.fit_ols(x.reshape(-1, 1), y)
        slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        self.assertAlmostEqual(coefficients[0], slope, places=9)
        self.assertAlmostEqual(intercept, y.mean() - slope * x.mean(), places=9)

    def test_recovers_known_coefficients(self):
        frame = make_frame(2000, noise=0.5)
        model = unit4.fit_model(frame, "m", "y", ["x1", "x2"], 0.2, "random")
        self.assertAlmostEqual(model.intercept, 5, delta=0.5)
        self.assertAlmostEqual(model.coefficients[0], 2, delta=0.05)
        self.assertAlmostEqual(model.coefficients[1], -3, delta=0.05)
        self.assertGreater(model.test_metrics["R²"], 0.99)

    def test_metrics_by_hand(self):
        actual = np.array([3.0, 5.0, 7.0, 9.0])
        predicted = np.array([2.0, 5.0, 8.0, 11.0])
        metrics = unit4.regression_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["MAE"], (1 + 0 + 1 + 2) / 4)
        self.assertAlmostEqual(metrics["MSE"], (1 + 0 + 1 + 4) / 4)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(1.5))
        self.assertAlmostEqual(metrics["R²"], 1 - 6 / 20)       # SS_tot = 20
        perfect = unit4.regression_metrics(actual, actual)
        self.assertEqual((perfect["MAE"], perfect["R²"]), (0.0, 1.0))
        self.assertTrue(np.isnan(unit4.regression_metrics(np.ones(4), np.ones(4) * 2)["R²"]))

    def test_r2_can_be_negative_and_baseline_is_zero_on_train(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertLess(unit4.regression_metrics(y, y[::-1])["R²"], 0)
        self.assertAlmostEqual(unit4.regression_metrics(y, np.full(4, y.mean()))["R²"], 0.0)

    def test_split_is_reproducible_disjoint_and_complete(self):
        for method in ("random", "chronological"):
            train, test = unit4.split_indices(500, 0.2, method)
            again = unit4.split_indices(500, 0.2, method)
            self.assertTrue((train == again[0]).all() and (test == again[1]).all())
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(train) + len(test), 500)
            self.assertEqual(len(test), 100)
        train, test = unit4.split_indices(500, 0.2, "chronological")
        self.assertLess(train.max(), test.min())               # no look-ahead

    def test_no_data_leakage_test_rows_do_not_affect_the_fit(self):
        frame = make_frame(400)
        before = unit4.fit_model(frame, "m", "y", ["x1"], 0.25, "random")
        _, test_pos = unit4.split_indices(len(frame), 0.25, "random")
        altered = frame.copy()
        altered.loc[test_pos, "y"] += 1000                      # corrupt ONLY the test rows
        after = unit4.fit_model(altered, "m", "y", ["x1"], 0.25, "random")
        self.assertEqual(before.intercept, after.intercept)
        self.assertTrue((before.coefficients == after.coefficients).all())
        self.assertNotEqual(before.test_metrics["RMSE"], after.test_metrics["RMSE"])

    def test_models_compared_on_identical_rows_and_split(self):
        frame = make_frame(400)
        frame.loc[::7, "x2"] = np.nan                           # x2 has gaps; x1 does not
        simple = unit4.fit_model(frame, "s", "y", ["x1"], 0.2, "random")
        multiple = unit4.fit_model(frame, "m", "y", ["x1", "x2"], 0.2, "random")
        self.assertNotEqual(len(simple.train), len(multiple.train))   # own rows differ ...
        table, n_train, n_test = unit4.compare_models(frame, simple, multiple)
        common = unit4.complete_cases(frame, ["y", "x1", "x2"])
        a = unit4.fit_model(common, "a", "y", ["x1"], 0.2, "random")
        b = unit4.fit_model(common, "b", "y", ["x1", "x2"], 0.2, "random")
        self.assertTrue(a.train["datetime"].equals(b.train["datetime"]))   # ... comparison uses the same rows
        self.assertTrue(a.test["datetime"].equals(b.test["datetime"]))
        self.assertEqual((n_train, n_test), (len(a.train), len(a.test)))
        self.assertAlmostEqual(table.loc["Simple linear regression", "RMSE"], a.test_metrics["RMSE"])
        self.assertAlmostEqual(table.loc["Multiple linear regression", "RMSE"], b.test_metrics["RMSE"])
        self.assertIn("Baseline (always predict training mean)", table.index)

    def test_prediction_matches_equation(self):
        model = unit4.fit_model(make_frame(300), "m", "y", ["x1", "x2"], 0.2, "random")
        manual = model.intercept + model.coefficients[0] * 50 + model.coefficients[1] * 20
        self.assertAlmostEqual(model.predict(np.array([50.0, 20.0])), manual)

    def test_guards(self):
        frame = make_frame(300)
        with self.assertRaises(unit4.ModelError):
            unit4.fit_model(frame, "m", "y", [], 0.2, "random")           # no predictor
        with self.assertRaises(unit4.ModelError):
            unit4.fit_model(frame, "m", "y", ["y", "x1"], 0.2, "random")  # target as predictor
        with self.assertRaises(unit4.ModelError):
            unit4.fit_model(frame.head(10), "m", "y", ["x1"], 0.2, "random")   # too few rows
        sparse = frame.copy()
        sparse.loc[5:, "x2"] = np.nan                                      # only 5 complete rows
        with self.assertRaises(unit4.ModelError):
            unit4.fit_model(sparse, "m", "y", ["x1", "x2"], 0.2, "random")

    def test_candidates_exclude_time_flags_constants_and_sparse(self):
        frame = make_frame(100)
        frame["suspicious_zero"] = False
        frame["constant"] = 3.0
        frame["sparse"] = np.nan
        frame.loc[:4, "sparse"] = 1.0
        self.assertEqual(unit4.numeric_candidates(frame), ["x1", "x2", "y"])

    def test_standardised_coefficients(self):
        model = unit4.fit_model(make_frame(500, noise=2), "m", "y", ["x1", "x2"], 0.2, "random")
        manual = model.coefficients * model.train[["x1", "x2"]].std().to_numpy() / model.train["y"].std()
        self.assertTrue(np.allclose(unit4.standardized_coefficients(model), manual))


class DiagnosticTests(unittest.TestCase):
    def test_vif_detects_collinearity(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=500)
        independent = np.column_stack([a, rng.normal(size=500)])
        collinear = np.column_stack([a, a * 2 + rng.normal(scale=0.05, size=500)])
        self.assertTrue((unit4.variance_inflation_factors(independent) < 1.2).all())
        self.assertTrue((unit4.variance_inflation_factors(collinear) > 100).all())

    def test_durbin_watson(self):
        rng = np.random.default_rng(2)
        self.assertAlmostEqual(unit4.durbin_watson(rng.normal(size=5000)), 2.0, delta=0.1)
        self.assertLess(unit4.durbin_watson(np.cumsum(rng.normal(size=500))), 0.5)

    def test_heteroscedasticity_and_nonlinearity_flags(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(1, 10, 1500).reshape(-1, 1)
        clean_resid = rng.normal(size=1500)
        funnel_resid = rng.normal(size=1500) * x[:, 0]                  # spread grows with x
        self.assertGreater(unit4.breusch_pagan(clean_resid, x)[1], 0.01)
        self.assertLess(unit4.breusch_pagan(funnel_resid, x)[1], 0.001)
        curved = x[:, 0] ** 2 + rng.normal(size=1500)
        intercept, coefs = unit4.fit_ols(x, curved)
        fitted = intercept + x @ coefs
        self.assertLess(unit4.ramsey_reset(curved, fitted, x)[1], 0.001)
        straight = 3 * x[:, 0] + rng.normal(size=1500)
        intercept, coefs = unit4.fit_ols(x, straight)
        self.assertGreater(unit4.ramsey_reset(straight, intercept + x @ coefs, x)[1], 0.01)

    def test_run_diagnostics_on_good_and_bad_models(self):
        good = unit4.run_diagnostics(unit4.fit_model(make_frame(800), "m", "y", ["x1", "x2"], 0.2, "random"))
        self.assertEqual(len(good["checks"]), 5)
        self.assertEqual(good["vif"].shape, (2,))
        by_name = {c["Assumption"]: c["violated"] for c in good["checks"]}
        # Data built to satisfy the assumptions: these four must pass. (Breusch-Pagan is left out
        # because a correct test rejects ~5% of clean random data; it is tested with a funnel above.)
        for name in ("Linearity", "Independence of errors", "Normality of residuals", "No multicollinearity"):
            self.assertIs(by_name[name], False, name)
        simple = unit4.run_diagnostics(unit4.fit_model(make_frame(300), "m", "y", ["x1"], 0.2, "random"))
        self.assertIsNone(simple["vif"])                                  # needs 2+ predictors
        self.assertIsNone(simple["checks"][-1]["violated"])
        trending = make_frame(600)
        # y depends on x1 plus a slowly drifting level -> errors in a row are alike
        trending["y"] = 5 + 2 * trending["x1"] + np.cumsum(np.random.default_rng(5).normal(size=600))
        bad = unit4.run_diagnostics(unit4.fit_model(trending, "m", "y", ["x1"], 0.2, "random"))
        self.assertTrue(bad["checks"][1]["violated"])                     # serial correlation flagged

    def test_coefficient_p_values_flag_useless_predictor(self):
        frame = make_frame(600)
        frame["noise"] = np.random.default_rng(9).normal(size=600)
        model = unit4.fit_model(frame, "m", "y", ["x1", "noise"], 0.2, "random")
        p_values = model.statistics["p_value"].iloc[1:].to_numpy()
        self.assertLess(p_values[0], 0.001)
        self.assertGreater(p_values[1], 0.01)


class FormattingTests(unittest.TestCase):
    def test_p_value_format(self):
        self.assertEqual(unit4.fmt_p(0.0), "<0.001")
        self.assertEqual(unit4.fmt_p(0.0456), "0.046")
        self.assertEqual(unit4.fmt_p(float("nan")), "n/a")


if __name__ == "__main__":
    unittest.main()
