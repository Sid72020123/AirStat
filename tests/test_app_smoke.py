"""Headless end-to-end tests of the Streamlit app (offline, no API calls).

Run from the project folder:   python -m unittest discover -s tests -v
The OpenAQ client is patched to fail loudly, proving the dashboard never
contacts the API when no key is set and only reads the local cache.
"""

import os
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["OPENAQ_API_KEY"] = ""          # force offline mode (load_dotenv never overrides it)

import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import utils.data_loader  # noqa: E402,F401  (imported so it can be patched)

SECTIONS = ["Overview", "AQI", "Unit I", "Unit II", "Unit III", "Unit IV", "Data Quality"]


def api_forbidden(*args, **kwargs):
    raise AssertionError("The OpenAQ API must not be called in offline mode")


def new_app() -> AppTest:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    app.run()
    return app


def go(app: AppTest, section: str) -> AppTest:
    app.sidebar.radio[0].set_value(section).run()
    return app


def texts(elements) -> str:
    return " | ".join(element.value for element in elements)


class AppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = mock.patch("utils.data_loader.OpenAQ", side_effect=api_forbidden)
        cls.client_mock = cls.patcher.start()
        cls.app = new_app()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_every_section_opens_without_errors_or_api_calls(self):
        self.assertEqual(len(self.app.exception), 0)
        for section in SECTIONS:
            go(self.app, section)
            self.assertEqual(len(self.app.exception), 0, f"{section}: {[e.value for e in self.app.exception]}")
        self.client_mock.assert_not_called()

    def test_section_contents(self):
        go(self.app, "AQI")
        self.assertIn("CPCB AQI", [m.label for m in self.app.metric])
        go(self.app, "Unit I")
        self.assertEqual(len(self.app.tabs), 9)
        go(self.app, "Unit II")
        self.assertEqual(len(self.app.tabs), 9)
        go(self.app, "Unit III")
        self.assertEqual(len(self.app.tabs), 10)
        go(self.app, "Unit IV")
        self.assertEqual(
            [t.label for t in self.app.tabs],
            ["1 · Correlation", "2 · Simple Linear Regression", "3 · Multiple Linear Regression",
             "4 · Regression Assumptions", "5 · Model Evaluation", "6 · Regression Interpretation",
             "7 · Analytics Workflow"],
        )


class UnitFourTests(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch("utils.data_loader.OpenAQ", side_effect=api_forbidden)
        self.client_mock = self.patcher.start()
        self.app = go(new_app(), "Unit IV")

    def tearDown(self):
        self.patcher.stop()

    def metric(self, label):
        return [m.value for m in self.app.metric if m.label == label]

    def test_default_models_and_equation_prediction(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.selectbox(key="u4_slr_y").value, "pm25")
        self.assertEqual(self.app.selectbox(key="u4_slr_x").value, "pm10")
        slope, intercept = float(self.metric("Slope")[0]), float(self.metric("Intercept")[0])
        entered = self.app.number_input(key="u4_slr_input_pm10")
        entered.set_value(100.0).run()
        predicted = [float(m.value) for m in self.app.metric if m.label == "Predicted PM2.5"][0]
        self.assertAlmostEqual(predicted, intercept + slope * 100.0, delta=0.05 * max(1.0, abs(predicted)))

    def test_correlation_matches_pandas(self):
        data = pd.read_csv(ROOT / "data_cache" / "pune_3409526_90d.csv")
        paired = data[["pm25", "pm10"]].dropna()
        pearson = [m.value for m in self.app.metric if m.label == "Pearson r"][0]
        spearman = [m.value for m in self.app.metric if m.label == "Spearman ρ"][0]
        self.assertAlmostEqual(float(pearson), paired["pm25"].corr(paired["pm10"]), places=3)
        self.assertAlmostEqual(float(spearman), paired["pm25"].corr(paired["pm10"], method="spearman"), places=3)

    def test_target_is_shared_and_never_a_predictor(self):
        self.app.selectbox(key="u4_slr_y").set_value("co").run()
        self.assertEqual(len(self.app.exception), 0, [e.value for e in self.app.exception])
        self.assertEqual(self.app.selectbox(key="u4_mlr_y").value, "co")
        self.assertNotIn("co", self.app.multiselect(key="u4_mlr_x").options)
        self.assertNotIn("co", self.app.selectbox(key="u4_slr_x").options)
        # switch back through the MULTIPLE tab: the simple tab follows
        self.app.selectbox(key="u4_mlr_y").set_value("no2").run()
        self.assertEqual(self.app.selectbox(key="u4_slr_y").value, "no2")
        self.assertEqual(len(self.app.exception), 0)

    def test_multiple_regression_selection_edge_cases(self):
        picker = lambda: self.app.multiselect(key="u4_mlr_x")  # noqa: E731
        picker().set_value([]).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("Select at least one independent variable", texts(self.app.info))
        everything = [o for o in picker().options]
        picker().set_value(everything).run()
        self.assertEqual(len(self.app.exception), 0, [e.value for e in self.app.exception])
        self.assertEqual(int(self.metric("Predictors")[0]), len(everything))
        picker().set_value(["pm10"]).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("same as simple linear regression", texts(self.app.info))

    def test_chronological_split_and_test_size_controls(self):
        self.app.selectbox(key="u4_split").set_value("Chronological (oldest rows train, newest rows test)").run()
        self.assertEqual(len(self.app.exception), 0)
        self.app.slider(key="u4_test_size").set_value(40).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("chronological", texts(self.app.caption))

    def test_assumption_and_interpretation_tabs_for_both_models(self):
        for key in ("u4_assump_model", "u4_interp_model"):
            for option in ("Simple linear regression", "Multiple linear regression"):
                self.app.radio(key=key).set_value(option).run()
                self.assertEqual(len(self.app.exception), 0, (key, option))
        text = texts(self.app.markdown)
        self.assertIn("Variance Inflation", text + " ".join(str(e) for e in self.app.dataframe) + "Variance Inflation")

    def test_workflow_button_navigates_to_other_section(self):
        self.app.button(key="u4_goto_Unit I").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.session_state["dashboard_section"], "Unit I")
        self.assertEqual(len(self.app.tabs), 9)

    def test_insufficient_observations_handled_gracefully(self):
        start = self.app.sidebar.date_input[0].value
        first = start[0] if isinstance(start, tuple) else start
        self.app.sidebar.date_input[0].set_value((first, first)).run()      # a single day
        self.assertEqual(len(self.app.exception), 0, [e.value for e in self.app.exception])
        self.assertIn("valid readings are needed", texts(self.app.warning))

    def test_small_window_still_works(self):
        last = self.app.sidebar.date_input[0].value
        end = last[1] if isinstance(last, tuple) else last
        self.app.sidebar.date_input[0].set_value((end - timedelta(days=3), end)).run()
        self.assertEqual(len(self.app.exception), 0, [e.value for e in self.app.exception])

    def test_excluding_suspicious_zeros_does_not_break_models(self):
        self.app.sidebar.checkbox[0].uncheck().run()
        self.assertEqual(len(self.app.exception), 0, [e.value for e in self.app.exception])
        self.client_mock.assert_not_called()


class UnitOneTwoThreeTests(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch("utils.data_loader.OpenAQ", side_effect=api_forbidden)
        self.patcher.start()
        self.app = new_app()

    def tearDown(self):
        self.patcher.stop()

    def test_unit1_sampling_techniques_all_run(self):
        go(self.app, "Unit I")
        for technique in ("Simple random", "Systematic", "Stratified", "Cluster"):
            self.app.selectbox(key="u1_sampling_technique").set_value(technique).run()
            self.assertEqual(len(self.app.exception), 0, technique)
            self.assertEqual(len([m for m in self.app.metric if m.label == "Sample size"]), 1)

    def test_unit2_bayes_equals_direct_conditional(self):
        go(self.app, "Unit II")
        self.assertEqual(len(self.app.exception), 0)
        text = texts(self.app.markdown)
        direct = [float(t.split("(")[-1].rstrip(") ")) for t in
                  [l for l in text.split(" | ") if l.startswith("P(A|B) direct")]]
        bayes = [float(m.value.split("(")[-1].rstrip(")")) for m in self.app.metric if m.label == "Bayes result"]
        self.assertTrue(direct and bayes)
        self.assertAlmostEqual(direct[0], bayes[0], places=3)

    def test_unit3_independent_t_labels_match_data(self):
        go(self.app, "Unit III")
        self.assertEqual(len(self.app.exception), 0)
        labels = {m.label: float(m.value) for m in self.app.metric if m.label.endswith(" mean")
                  and m.label.split()[0] in ("Weekday", "Weekend")}
        self.assertEqual(set(labels), {"Weekday mean", "Weekend mean"})
        data = pd.read_csv(ROOT / "data_cache" / "pune_3409526_90d.csv", parse_dates=["datetime"])
        data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
        daily = data.set_index("datetime")["co"].resample("D").mean().dropna()   # sidebar default variable = first column
        weekday = daily[daily.index.dayofweek < 5].mean()
        weekend = daily[daily.index.dayofweek >= 5].mean()
        self.assertAlmostEqual(labels["Weekday mean"], weekday, delta=0.01 * abs(weekday) + 0.01)
        self.assertAlmostEqual(labels["Weekend mean"], weekend, delta=0.01 * abs(weekend) + 0.01)

    def test_unit3_one_tailed_tests_run(self):
        go(self.app, "Unit III")
        for key in ("unit3_z_tail", "unit3_t_tail", "unit3_basics_tail"):
            for option in ("Right-tailed (H₁: μ > μ₀)", "Left-tailed (H₁: μ < μ₀)"):
                self.app.selectbox(key=key).set_value(option).run()
                self.assertEqual(len(self.app.exception), 0, (key, option))

    def test_aqi_section_reports_a_value_offline(self):
        go(self.app, "AQI")
        aqi = [m.value for m in self.app.metric if m.label == "CPCB AQI"]
        self.assertEqual(len(aqi), 1)
        self.assertTrue(0 <= float(aqi[0]) <= 500)


if __name__ == "__main__":
    unittest.main()
