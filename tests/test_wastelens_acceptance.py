from __future__ import annotations

import builtins
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from wastelens_core import (
    explain_batch,
    forecast,
    recommendations,
    request_key,
    train_models,
    validate_csv,
)
from wastelens_store import (
    clear_workspace_and_runs,
    get_run_by_request_key,
    list_runs,
    load_workspace,
    save_run,
    save_workspace,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "test_datasets"


def golden_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "test_golden_valid.csv", parse_dates=["date"])


class ValidationAcceptance(unittest.TestCase):
    def test_golden_csv_imports(self):
        result = validate_csv((FIXTURES / "test_golden_valid.csv").read_bytes())
        self.assertEqual(result.errors, [])
        self.assertEqual(result.valid_rows, result.total_rows)
        self.assertTrue(result.checksum)

    def test_missing_column_names_the_field(self):
        result = validate_csv(b"date,meal_type,menu_item,meals_prepared,food_waste_kg\n")
        self.assertTrue(any("meals_served" in message for message in result.errors))
        self.assertIsNone(result.frame)

    def test_invalid_rows_have_field_level_reasons(self):
        result = validate_csv((FIXTURES / "test_invalid_rows.csv").read_bytes())
        self.assertGreater(len(result.rejected), 0)
        self.assertTrue(result.rejected["_validation_issue"].str.contains("date|meals_served|food_waste_kg", case=False).any())
        self.assertEqual(result.valid_rows / result.total_rows, .8)

    def test_aliases_are_previewable_and_decimal_comma_is_parsed(self):
        source = golden_frame().rename(columns={
            "date": "service_date", "meal_type": "meal_period", "menu_item": "item_name",
            "meals_prepared": "portions_prepared", "meals_served": "portions_served",
            "food_waste_kg": "waste_kg",
        })
        raw = source.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
        result = validate_csv(raw)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.column_mappings["waste_kg"], "food_waste_kg")
        self.assertTrue(pd.api.types.is_numeric_dtype(result.frame["food_waste_kg"].dtype))
        self.assertEqual(result.warning_rows, len(source))

    def test_duplicate_keys_block_until_keep_last_is_requested(self):
        source = golden_frame()
        duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        raw = duplicated.to_csv(index=False).encode("utf-8")
        blocked = validate_csv(raw)
        self.assertTrue(any("duplicate" in message.lower() for message in blocked.errors))
        resolved = validate_csv(raw, keep_last_duplicates=True)
        self.assertEqual(resolved.errors, [])
        self.assertEqual(resolved.valid_rows, len(source))
        self.assertTrue(any("after explicit confirmation" in message for message in resolved.warnings))
        self.assertTrue(resolved.rejected["_validation_issue"].str.contains("earlier row discarded").any())

    def test_gram_waste_is_normalized_to_kg(self):
        source = golden_frame().head(300)
        original_kg = float(source.iloc[0]["food_waste_kg"])
        source["food_waste_unit"] = "g"
        source["food_waste_kg"] *= 1000
        result = validate_csv(source.to_csv(index=False).encode("utf-8"))
        self.assertEqual(result.errors, [])
        self.assertAlmostEqual(float(result.frame.iloc[0]["food_waste_kg"]), original_kg)
        self.assertTrue(any("converted to kilograms" in warning for warning in result.warnings))

    def test_sparse_history_is_degraded_and_future_date_is_rejected(self):
        sparse = validate_csv((FIXTURES / "test_sparse_14_days.csv").read_bytes())
        self.assertFalse(sparse.errors)
        self.assertTrue(any("Degraded forecast" in warning for warning in sparse.warnings))
        source = golden_frame()
        source.loc[0, "date"] = pd.Timestamp(date.today() + timedelta(days=1))
        result = validate_csv(source.to_csv(index=False).encode("utf-8"))
        self.assertTrue(result.rejected["_validation_issue"].str.contains("Future service date").any())


class ModelAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = golden_frame()
        cls.models = train_models(cls.frame)

    def test_candidates_are_compared_and_temporal_features_are_used(self):
        for key in ["demand_metrics", "waste_metrics"]:
            metrics = self.models[key]
            self.assertEqual(set(metrics["model_comparison"]), {"random_forest", "xgboost"})
            self.assertGreaterEqual(metrics["baseline_mae"], 0)
            self.assertIn("stability_mae_std", metrics["model_comparison"][metrics["algorithm"]])
        self.assertTrue(any(name.startswith("lag_") for name in self.models["demand_features"]))
        self.assertTrue(any(name.startswith("rolling_") for name in self.models["waste_features"]))
        self.assertIn("xgboost", self.models["library_versions"])

    def test_forecast_is_nonnegative_repeatable_and_ceiling_is_visible(self):
        future = self.frame.date.max().date() + timedelta(days=1)
        inputs = {"target_date": future, "meal_type": str(self.frame.meal_type.iloc[0]),
                  "expected_customers": 80, "temperature": 22.0, "is_holiday": False,
                  "has_event": False, "menu_items": self.frame.menu_item.astype(str).unique().tolist()}
        first = forecast(self.models, self.frame, inputs)
        second = forecast(self.models, self.frame, inputs)
        self.assertGreaterEqual(first["demand"], 0)
        self.assertEqual(round(first["demand"]), round(second["demand"]))
        self.assertEqual([round(item["waste_kg"], 2) for item in first["items"]],
                         [round(item["waste_kg"], 2) for item in second["items"]])
        self.assertTrue(any("exceeds the entered customer count" in warning for warning in first["warnings"]) or first["demand"] <= 80)
        self.assertTrue(all(item["waste_kg"] >= 0 for item in first["items"]))

    def test_invalid_customer_input_is_blocked(self):
        result = forecast(self.models, self.frame, {"expected_customers": -1})
        self.assertIn("error", result)

    def test_recommendation_formula_increment_capacity_and_risk(self):
        result = {"demand": 100.2, "items": [{"item": "Rice", "waste_kg": 1.1,
                  "serving_ratio": .5, "portion_mass_kg": .25, "explanation": {"drivers": []}}]}
        frame = recommendations(result, .05, 1, {"Rice": {"increment": 10, "capacity": 40}},
                                {"moderate": .05, "high": .10})
        row = frame.iloc[0]
        self.assertEqual(row["Forecast portions"], 51)
        self.assertEqual(row["Uncapped recommendation"], 60)
        self.assertEqual(row["Recommended portions"], 40)
        self.assertEqual(row["Unmet demand risk"], "High")
        self.assertEqual(row["Waste risk"], "High")
        with self.assertRaises(ValueError):
            recommendations(result, .05, 1, risk_thresholds={"moderate": .1, "high": .05})

    def test_request_key_is_order_independent_and_configuration_sensitive(self):
        inputs = {"target_date": date(2026, 10, 1), "meal_type": "Lunch", "expected_customers": 20,
                  "temperature": 20.0, "is_holiday": False, "has_event": False, "menu_items": ["Rice", "Dal"]}
        settings = {"Rice": {"increment": 1, "capacity": None}}
        key = request_key("d", "m", inputs, .05, 1, settings)
        reordered = dict(inputs, menu_items=["Dal", "Rice"])
        self.assertEqual(key, request_key("d", "m", reordered, .05, 1, settings))
        self.assertNotEqual(key, request_key("d", "m", inputs, .10, 1, settings))

    def test_shap_failure_has_an_explicit_fallback(self):
        class Model:
            feature_importances_ = np.array([1.0])

        original_import = builtins.__import__

        def fail_shap(name, *args, **kwargs):
            if name == "shap":
                raise ImportError("forced fallback")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_shap):
            result = explain_batch(Model(), ["temperature"], pd.DataFrame([[20.0]], columns=["temperature"]))
        self.assertIn("fallback", result[0]["kind"])


class PersistenceAcceptance(unittest.TestCase):
    def test_dataset_run_and_clear_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"WASTELENS_DB_PATH": str(Path(directory) / "wastelens.sqlite3")}):
            frame = golden_frame()
            metadata = {"id": "dataset-1", "checksum": "checksum", "model_version": "model-v1"}
            save_workspace(frame, metadata, None, None, {}, {}, "No override")
            restored = load_workspace()
            self.assertEqual(restored["dataset"]["id"], "dataset-1")
            self.assertEqual(len(restored["data"]), len(frame))
            run = {"id": "run-1", "request_key": "same-request", "dataset_id": "dataset-1",
                   "generated_at": "2026-09-30T10:00:00+00:00", "inputs": {"target_date": date(2026, 10, 1), "meal_type": "Lunch"},
                   "forecast": {"demand": 20, "items": [{"waste_kg": 1.0}]}}
            save_run(run, {"run": run, "overrides": {"Rice": 10}})
            self.assertEqual(get_run_by_request_key("same-request")["run_id"], "run-1")
            retry = dict(run, id="run-2", generated_at="2026-09-30T10:01:00+00:00")
            canonical = save_run(retry, {"run": retry, "overrides": {}})
            self.assertEqual(canonical["id"], "run-1")
            self.assertEqual(get_run_by_request_key("same-request")["payload"]["overrides"], {"Rice": 10})
            self.assertEqual(len(list_runs()), 1)
            clear_workspace_and_runs()
            self.assertIsNone(load_workspace())
            self.assertEqual(list_runs(), [])


class AppScenarioAcceptance(unittest.TestCase):
    def test_labeled_demo_flows_through_dashboard_and_saved_plan(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"WASTELENS_DB_PATH": str(Path(directory) / "demo-app.sqlite3")}):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            next(button for button in app.button if button.label == "Load synthetic demo").click().run()
            self.assertEqual(len(app.exception), 0, [entry.message for entry in app.exception])
            self.assertEqual(app.session_state["dataset"]["provenance"], "Synthetic demo")
            next(button for button in app.button if button.label == "Generate plan").click().run()
            self.assertEqual(len(app.exception), 0, [entry.message for entry in app.exception])
            self.assertIsNotNone(app.session_state["baseline"])
            self.assertEqual(len(list_runs()), 1)
            self.assertTrue(any(metric.label == "Predicted demand" for metric in app.metric))
            baseline_id = app.session_state["baseline"]["id"]
            customers = next(control for control in app.number_input if control.label == "Expected customers")
            customers.set_value(int(customers.value) + 5)
            with patch("wastelens_core.forecast", side_effect=RuntimeError("forced inference failure")):
                next(button for button in app.button if button.label == "Generate plan").click().run()
            self.assertEqual(len(app.exception), 0, [entry.message for entry in app.exception])
            self.assertEqual(app.session_state["baseline"]["id"], baseline_id)
            self.assertTrue(any("WL-MODEL-503" in entry.value for entry in app.error))

    def test_reset_scenario_preserves_the_exact_baseline(self):
        from streamlit.testing.v1 import AppTest

        frame = golden_frame()
        models = train_models(frame)
        future = frame.date.max().date() + timedelta(days=1)
        menu = frame.menu_item.astype(str).unique().tolist()
        inputs = {"target_date": future, "meal_type": str(frame.meal_type.iloc[0]),
                  "expected_customers": 80, "temperature": 22.0, "is_holiday": False,
                  "has_event": False, "menu_items": menu}
        scenario_inputs = dict(inputs, expected_customers=90)
        checksum = "fixture-checksum"
        dataset_id = checksum[:12]
        algorithms = {"demand": models["demand_algorithm"], "waste": models["waste_algorithm"]}
        version = f"demand-{algorithms['demand']}_waste-{algorithms['waste']}-v2-{checksum[:8]}"
        metadata = {"id": dataset_id, "checksum": checksum, "model_version": version,
                    "algorithms": algorithms, "library_versions": models["library_versions"],
                    "rows": len(frame), "provenance": "Acceptance fixture", "warnings": []}
        baseline = {"id": "base-1234", "request_key": "baseline-key", "generated_at": "2026-09-30T10:00:00+00:00",
                    "dataset_id": dataset_id, "model_version": version, "inputs": inputs,
                    "forecast": forecast(models, frame, inputs), "buffer": .05, "increment": 1,
                    "item_settings": {}, "risk_thresholds": {"moderate": .05, "high": .10}}
        scenario = {"baseline_run_id": baseline["id"], "inputs": scenario_inputs,
                    "forecast": forecast(models, frame, scenario_inputs), "generated_at": "2026-09-30T10:01:00+00:00"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"WASTELENS_DB_PATH": str(Path(directory) / "app.sqlite3")}):
            save_workspace(frame, metadata, baseline, scenario, {}, {}, "No override")
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0, [entry.message for entry in app.exception])
            original = app.session_state["baseline"]
            self.assertEqual(app.session_state["scenario"]["baseline_run_id"], original["id"])
            reset = next(button for button in app.button if button.label == "Reset scenario")
            reset.click().run()
            self.assertEqual(len(app.exception), 0, [entry.message for entry in app.exception])
            self.assertIsNone(app.session_state["scenario"])
            self.assertEqual(app.session_state["baseline"], original)


if __name__ == "__main__":
    unittest.main()
