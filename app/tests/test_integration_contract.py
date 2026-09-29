from __future__ import annotations

import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from construction_monitoring.integration import (
    IntegrationValidationError,
    analyze,
    analyze_rfdetr,
    temporal_update,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "integration" / "fixtures"
SCHEMA_DIR = PROJECT_ROOT / "integration" / "schemas"


def assert_subset(test_case: unittest.TestCase, expected, actual, path="response"):
    if isinstance(expected, dict):
        test_case.assertIsInstance(actual, dict, path)
        for key, value in expected.items():
            test_case.assertIn(key, actual, f"{path}.{key}")
            assert_subset(test_case, value, actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        test_case.assertEqual(len(actual), len(expected), path)
        for index, value in enumerate(expected):
            assert_subset(test_case, value, actual[index], f"{path}[{index}]")
        return
    test_case.assertEqual(actual, expected, path)


class IntegrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schemas = {}
        for path in SCHEMA_DIR.glob("*.json"):
            with path.open(encoding="utf-8") as stream:
                schema = json.load(stream)
            Draft202012Validator.check_schema(schema)
            cls.schemas[path.name] = schema
        registry = Registry().with_resources(
            (
                schema["$id"],
                Resource.from_contents(schema),
            )
            for schema in cls.schemas.values()
        )
        cls.validators = {
            name: Draft202012Validator(schema, registry=registry)
            for name, schema in cls.schemas.items()
        }

    def load_fixture(self, name: str):
        with (FIXTURE_DIR / name).open(encoding="utf-8") as stream:
            return json.load(stream)

    def test_01_all_ten_synthetic_fixtures_match_expected_outputs(self) -> None:
        fixture_names = sorted(path.name for path in FIXTURE_DIR.glob("*.json"))
        self.assertEqual(
            fixture_names,
            [
                "a_ok.json",
                "b_warning.json",
                "c_uncertain.json",
                "d_not_monitorable.json",
                "e_low_confidence.json",
                "f_zone_unknown.json",
                "g_other_zone.json",
                "h_temporal_isolated_miss.json",
                "i_temporal_persistent_warning.json",
                "j_active_condition.json",
            ],
        )

        for fixture_name in fixture_names:
            with self.subTest(fixture=fixture_name):
                fixture = self.load_fixture(fixture_name)
                self.assertTrue(fixture["synthetic_fixture"])
                self.assertEqual(fixture["data_origin"]["planned_work"], "source_excel")
                self.assertEqual(fixture["data_origin"]["detections"], "synthetic")
                if fixture["mode"] == "instant":
                    self.validators["analysis_request.schema.json"].validate(
                        fixture["request"]
                    )
                    result = analyze(fixture["request"])
                    self.validators["analysis_response.schema.json"].validate(result)
                else:
                    self.validators["temporal_request.schema.json"].validate(
                        fixture["request"]
                    )
                    result = temporal_update(fixture["request"])
                    self.validators["temporal_response.schema.json"].validate(result)
                assert_subset(self, fixture["expected"], result)
                json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_02_instant_output_uses_existing_nested_contract(self) -> None:
        result = analyze(self.load_fixture("a_ok.json")["request"])
        self.assertIn("instant_status", result)
        self.assertIn("persistent_status", result)
        self.assertIn("observability", result)
        self.assertIn("score", result["match"])
        self.assertIn("coverage", result["match"])
        self.assertIn("satisfied", result["match"])
        self.assertIn("missing", result["match"])
        self.assertIn("uncertain", result["match"])
        self.assertIn("supporting", result["match"])
        self.assertIn("evidence", result)
        self.assertIn("possible_unexpected", result)
        self.assertIn("explanation", result)
        self.assertEqual(result["metadata"]["timestamp"], "2026-09-27T10:00:00+03:00")
        self.assertEqual(result["metadata"]["camera_id"], "cam-01")
        self.assertEqual(result["metadata"]["zone_id"], "pit-a")
        self.assertNotIn("match_score", result)
        self.assertNotIn("coverage", result)

    def test_03_zone_unknown_is_whole_frame_and_not_invented(self) -> None:
        result = analyze(self.load_fixture("f_zone_unknown.json")["request"])
        self.assertEqual(result["status"], "OK")
        self.assertIsNone(result["metadata"]["zone_id"])
        self.assertIsNone(result["metadata"]["target_zone_id"])
        self.assertTrue(
            all(item["zone_verified"] for item in result["detections"]["accepted"])
        )

    def test_04_request_validation_is_strict_and_machine_readable(self) -> None:
        request = self.load_fixture("a_ok.json")["request"]
        malformed = json.loads(json.dumps(request))
        malformed["context"]["timestamp"] = "2026-09-27T10:00:00"
        malformed["detections"][0]["bbox"] = [10, 0, 0, 10]
        malformed["unexpected"] = True

        with self.assertRaises(IntegrationValidationError) as captured:
            analyze(malformed)
        payload = captured.exception.as_dict()
        self.assertEqual(payload["error"], "INVALID_REQUEST")
        paths = {item["path"] for item in payload["issues"]}
        self.assertIn("request.unexpected", paths)
        self.assertIn("request.context.timestamp", paths)
        self.assertIn("request.detections[0].bbox", paths)
        json.dumps(payload, ensure_ascii=False, allow_nan=False)

    def test_05_unknown_work_condition_and_detector_class_are_rejected(self) -> None:
        request = self.load_fixture("a_ok.json")["request"]
        malformed = json.loads(json.dumps(request))
        malformed["context"]["planned_work"] = "Несуществующая работа"
        malformed["context"]["active_conditions"] = {"несуществующее условие": True}
        malformed["detections"][0]["class"] = "unknown_machine"

        with self.assertRaises(IntegrationValidationError) as captured:
            analyze(malformed)
        codes = {item["code"] for item in captured.exception.issues}
        self.assertIn("unknown_planned_work", codes)
        self.assertIn("unknown_detection_class", codes)

        condition_request = self.load_fixture("a_ok.json")["request"]
        condition_request["context"]["active_conditions"] = {
            "несуществующее условие": True
        }
        with self.assertRaises(IntegrationValidationError) as condition_error:
            analyze(condition_request)
        self.assertIn(
            "unknown_active_condition",
            {item["code"] for item in condition_error.exception.issues},
        )

    def test_06_optional_fields_must_be_omitted_instead_of_null(self) -> None:
        request = self.load_fixture("a_ok.json")["request"]
        request["context"]["zone_id"] = None
        request["context"]["active_conditions"] = None
        request["detections"][0]["zone_id"] = None

        with self.assertRaises(IntegrationValidationError) as captured:
            analyze(request)
        paths = {item["path"] for item in captured.exception.issues}
        self.assertIn("request.context.zone_id", paths)
        self.assertIn("request.context.active_conditions", paths)
        self.assertIn("request.detections[0].zone_id", paths)

    def test_07_temporal_request_rejects_mixed_context(self) -> None:
        request = self.load_fixture("h_temporal_isolated_miss.json")["request"]
        malformed = json.loads(json.dumps(request))
        malformed["observations"][1]["context"]["camera_id"] = "cam-02"

        with self.assertRaises(IntegrationValidationError) as captured:
            temporal_update(malformed)
        self.assertIn(
            "mixed_temporal_context",
            {item["code"] for item in captured.exception.issues},
        )

    def test_08_json_schemas_are_valid_json_and_cover_both_endpoints(self) -> None:
        expected = {
            "analysis_request.schema.json",
            "analysis_response.schema.json",
            "temporal_request.schema.json",
            "temporal_response.schema.json",
        }
        self.assertEqual({path.name for path in SCHEMA_DIR.glob("*.json")}, expected)
        for path in SCHEMA_DIR.glob("*.json"):
            with self.subTest(schema=path.name):
                schema = self.schemas[path.name]
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertIn("$id", schema)

    def test_09_rfdetr_native_result_is_canonicalized_at_matcher_threshold(self) -> None:
        class FakeRFDetections:
            data = {"class_name": ["excavator", b"dump_truck"]}
            confidence = [0.90, 0.25]
            xyxy = [
                [10, 20, 210, 220],
                [230, 20, 480, 240],
            ]

        request = {
            "context": {
                "planned_work": "Устройство котлована",
                "active_conditions": {},
                "timestamp": "2026-09-27T13:00:00+03:00",
                "camera_id": "cam-rfdetr",
            },
            "detections": FakeRFDetections(),
        }

        result = analyze_rfdetr(request)

        self.assertEqual(result["status"], "OK")
        self.assertEqual(
            [item["confidence"] for item in result["detections"]["accepted"]],
            [0.90, 0.25],
        )
        self.assertEqual(result["detections"]["rejected_low_confidence"], [])
        self.assertEqual(result["detections"]["accepted"][0]["class"], "excavator")


if __name__ == "__main__":
    unittest.main(verbosity=2)
