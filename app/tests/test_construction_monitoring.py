from __future__ import annotations

import json
import math
import random
import unittest
from datetime import datetime, timezone

from construction_monitoring import (
    calculate_observability,
    MatchingConfig,
    MatchingEngine,
    MODEL_CLASSES,
    DISPLAY_NAMES_RU,
    RULES,
    TemporalConfig,
    WorkCatalog,
    analyze_temporal,
    validate_rules,
)


class MatchingEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = WorkCatalog.load()
        cls.engine = MatchingEngine(catalog=cls.catalog)

    def test_01_all_required_present(self) -> None:
        result = self.engine.analyze_profile(15, ["excavator", "dump_truck"])
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["match"]["score"], 1.0)

    def test_02_one_required_missing(self) -> None:
        result = self.engine.analyze_profile(15, ["excavator"])
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["match"]["score"], 0.5)
        self.assertEqual(
            [item["id"] for item in result["match"]["missing"]],
            ["required:dump_truck"],
        )

    def test_03_all_required_missing(self) -> None:
        result = self.engine.analyze_profile(15, [])
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["match"]["score"], 0.0)
        self.assertEqual(result["match"]["missing_count"], 2)

    def test_04_fully_observable_alternative_satisfied(self) -> None:
        result = self.engine.analyze_profile(
            30, ["truck", "excavator", "crane_manipulator"]
        )
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["match"]["score"], 1.0)

    def test_05_fully_observable_alternative_missing(self) -> None:
        result = self.engine.analyze_profile(30, ["truck", "excavator"])
        self.assertEqual(result["status"], "WARNING")
        self.assertIn(
            "alternative:1",
            [item["id"] for item in result["match"]["missing"]],
        )

    def test_06_alternative_with_undetectable_option(self) -> None:
        result = self.engine.analyze_profile(30, ["truck", "mobile_crane"])
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["match"]["score"], 1.0)
        self.assertEqual(result["match"]["coverage"], 0.667)
        self.assertIn("ямокопатель", result["explanation"])

    def test_07_inactive_unknown_conditional_does_not_warn(self) -> None:
        result = self.engine.analyze_profile(15, ["excavator", "dump_truck"])
        self.assertEqual(result["status"], "OK")
        self.assertNotIn(
            "bulldozer",
            [
                class_name
                for item in result["match"]["missing"]
                for class_name in item["classes"]
            ],
        )

    def test_08_conditional_presence_is_supporting_evidence(self) -> None:
        result = self.engine.analyze_profile(
            15, ["excavator", "dump_truck", "bulldozer"]
        )
        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(result["match"]["supporting"]), 1)

    def test_09_not_monitorable(self) -> None:
        result = self.engine.analyze_profile(2, [])
        self.assertEqual(result["status"], "NOT_MONITORABLE")
        self.assertIsNone(result["match"]["score"])
        self.assertEqual(result["observability"], {"score": 0.0, "level": "NONE"})

    def test_10_unknown_work(self) -> None:
        result = self.engine.analyze_work("Несуществующая работа", [])
        self.assertEqual(result["status"], "UNKNOWN_WORK")

    def test_11_low_detector_confidence(self) -> None:
        engine = MatchingEngine(
            catalog=self.catalog,
            config=MatchingConfig(confidence_threshold=0.6),
        )
        result = engine.analyze_profile(
            15,
            [
                {"class": "excavator", "confidence": 0.91},
                {"class": "dump_truck", "confidence": 0.59},
            ],
        )
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(len(result["detections"]["rejected_low_confidence"]), 1)

    def test_12_duplicate_instances_preserved(self) -> None:
        result = self.engine.analyze_profile(
            15, ["excavator", "excavator", "dump_truck"]
        )
        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(result["detections"]["accepted"]), 3)

    def test_13_single_detector_miss_is_not_persistent(self) -> None:
        observations = [
            {
                "timestamp": "2026-09-27T10:00:00+03:00",
                "detections": ["excavator", "dump_truck"],
            },
            {
                "timestamp": "2026-09-27T10:07:00+03:00",
                "detections": ["excavator"],
            },
            {
                "timestamp": "2026-09-27T10:19:00+03:00",
                "detections": ["excavator", "dump_truck"],
            },
        ]
        result = analyze_temporal(
            self.engine,
            "Устройство котлована",
            observations,
            config=TemporalConfig(min_observations=3, persistence_ratio=0.75),
        )
        self.assertEqual(result["persistent_status"], "OK")

    def test_14_persistent_absence(self) -> None:
        observations = [
            {
                "timestamp": f"2026-09-27T10:{minute:02d}:00+03:00",
                "detections": ["excavator"],
            }
            for minute in (0, 4, 11, 23)
        ]
        result = analyze_temporal(
            self.engine,
            "Устройство котлована",
            observations,
            config=TemporalConfig(min_observations=3, persistence_ratio=0.75),
        )
        self.assertEqual(result["persistent_status"], "WARNING")
        self.assertEqual(
            result["temporal"]["persistent_missing"][0]["requirement"]["id"],
            "required:dump_truck",
        )

    def test_15_warning_and_uncertainty_coexist(self) -> None:
        result = self.engine.analyze_profile(30, ["truck"])
        self.assertEqual(result["status"], "WARNING")
        self.assertTrue(result["uncertainty_present"])
        self.assertTrue(result["match"]["missing"])
        self.assertTrue(result["match"]["uncertain"])

    def test_16_empty_detection_list(self) -> None:
        result = self.engine.analyze_profile(15, [])
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["detections"]["accepted"], [])

    def test_17_unknown_detector_class(self) -> None:
        result = self.engine.analyze_profile(
            15, ["excavator", "dump_truck", "unknown_machine"]
        )
        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(result["detections"]["rejected_unknown_class"]), 1)

    def test_18_zone_mismatch(self) -> None:
        result = self.engine.analyze_profile(
            15,
            [
                {"class": "excavator", "confidence": 0.9, "zone_id": "B"},
                {"class": "dump_truck", "confidence": 0.9, "zone_id": "B"},
            ],
            target_zone_id="A",
        )
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(len(result["detections"]["rejected_zone_mismatch"]), 2)

    def test_19_unverified_zone_yields_uncertainty(self) -> None:
        result = self.engine.analyze_profile(
            15,
            [
                {"class": "excavator", "confidence": 0.9},
                {"class": "dump_truck", "confidence": 0.9},
            ],
            target_zone_id="A",
        )
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertIsNone(result["match"]["score"])

    def test_20_unexpected_equipment_is_not_hard_violation(self) -> None:
        result = self.engine.analyze_profile(
            15, ["excavator", "dump_truck", "tower_crane"]
        )
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["possible_unexpected"][0]["class"], "tower_crane")

    def test_21_stable_profile_mapping(self) -> None:
        record = self.catalog.find("Пересадка зеленых насождений")
        self.assertIsNotNone(record)
        self.assertEqual(record.profile_id, 30)

    def test_22_all_profiles_and_works_are_structurally_valid(self) -> None:
        self.assertEqual(validate_rules(), [])
        self.assertEqual(self.catalog.validate(), [])
        self.assertEqual(set(RULES), set(range(35)))
        self.assertEqual(len(self.catalog.records), 376)

        for profile_id in RULES:
            result = self.engine.analyze_profile(profile_id, [])
            json.dumps(result, ensure_ascii=False)

    def test_23_profile_22_crane_requirement_is_not_optional(self) -> None:
        result = self.engine.analyze_profile(22, [])
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(
            [item["id"] for item in result["match"]["missing"]],
            ["alternative:0"],
        )

    def test_24_profile_27_section_crane_requirement_is_not_optional(self) -> None:
        result = self.engine.analyze_profile(27, ["excavator"])
        self.assertEqual(result["status"], "WARNING")
        self.assertIn(
            "alternative:1",
            [item["id"] for item in result["match"]["missing"]],
        )

    def test_25_slash_separated_undetectable_terms_are_one_alternative(self) -> None:
        expectations = {
            1: 3,
            7: 2,
            8: 1,
            12: 2,
            18: 2,
            25: 2,
            33: 2,
        }
        detections = {
            7: ["crane_manipulator"],
            18: ["excavator", "dump_truck"],
        }
        for profile_id, expected_uncertain_count in expectations.items():
            with self.subTest(profile_id=profile_id):
                result = self.engine.analyze_profile(
                    profile_id, detections.get(profile_id, [])
                )
                self.assertEqual(
                    result["match"]["uncertain_count"], expected_uncertain_count
                )

    def test_26_invalid_bbox_cannot_satisfy_requirement(self) -> None:
        result = self.engine.analyze_profile(
            15,
            [
                {"class": "excavator", "confidence": 0.9, "bbox": [0, 0, 10]},
                {
                    "class": "dump_truck",
                    "confidence": 0.9,
                    "bbox": [0, 0, 10, 10],
                },
            ],
        )
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(len(result["detections"]["rejected_invalid"]), 1)
        self.assertEqual(
            result["detections"]["rejected_invalid"][0]["invalid_reason"],
            "invalid_bbox",
        )

    def test_27_bbox_must_be_finite_ordered_xyxy(self) -> None:
        invalid_boxes = [
            [10, 0, 0, 10],
            [0, 10, 10, 0],
            [0, 0, math.nan, 10],
            [0, 0, math.inf, 10],
            [-1, 0, 10, 10],
        ]
        for bbox in invalid_boxes:
            with self.subTest(bbox=bbox):
                result = self.engine.analyze_profile(
                    15,
                    [
                        {"class": "excavator", "confidence": 0.9, "bbox": bbox},
                        "dump_truck",
                    ],
                )
                self.assertEqual(result["status"], "WARNING")
                self.assertEqual(len(result["detections"]["rejected_invalid"]), 1)

    def test_28_confidence_boundary_is_inclusive(self) -> None:
        cases = (
            (0.249999, "WARNING", 1),
            (0.25, "OK", 0),
            (0.250001, "OK", 0),
        )
        for confidence, expected_status, expected_rejected in cases:
            with self.subTest(confidence=confidence):
                result = self.engine.analyze_profile(
                    15,
                    [
                        {"class": "excavator", "confidence": confidence},
                        {"class": "dump_truck", "confidence": 1.0},
                    ],
                )
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(
                    len(result["detections"]["rejected_low_confidence"]),
                    expected_rejected,
                )

    def test_29_nonfinite_confidence_is_strict_json_safe(self) -> None:
        for confidence in (math.nan, math.inf, -math.inf):
            with self.subTest(confidence=confidence):
                result = self.engine.analyze_profile(
                    15,
                    [
                        {"class": "excavator", "confidence": confidence},
                        "dump_truck",
                    ],
                )
                self.assertEqual(result["status"], "WARNING")
                self.assertIsNone(
                    result["detections"]["rejected_invalid"][0]["confidence"]
                )
                json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_30_none_planned_work_is_unknown_not_exception(self) -> None:
        for planned_work in (None, "", "   "):
            with self.subTest(planned_work=planned_work):
                result = self.engine.analyze_work(planned_work, [])
                self.assertEqual(result["status"], "UNKNOWN_WORK")

    def test_31_detection_collection_shape_is_validated(self) -> None:
        for malformed in (None, "excavator", {"class": "excavator"}):
            with self.subTest(malformed=malformed):
                with self.assertRaises(TypeError):
                    self.engine.analyze_profile(15, malformed)

    def test_32_condition_mapping_requires_boolean_values(self) -> None:
        with self.assertRaises(TypeError):
            self.engine.analyze_profile(
                15,
                ["excavator", "dump_truck"],
                active_conditions={"для планировки/зачистки": "false"},
            )

    def test_33_datetime_metadata_is_json_serializable(self) -> None:
        result = self.engine.analyze_profile(
            15,
            ["excavator", "dump_truck"],
            metadata={"timestamp": datetime(2026, 9, 27, tzinfo=timezone.utc)},
        )
        self.assertEqual(result["metadata"]["timestamp"], "2026-09-27T00:00:00+00:00")
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_34_metadata_nonfinite_numbers_become_null(self) -> None:
        result = self.engine.analyze_profile(
            15,
            ["excavator", "dump_truck"],
            metadata={"diagnostic": math.nan},
        )
        self.assertIsNone(result["metadata"]["diagnostic"])
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_35_temporal_rejects_mixed_cameras(self) -> None:
        observations = [
            {
                "timestamp": "2026-09-27T10:00:00+03:00",
                "camera_id": "cam-a",
                "zone_id": "A",
                "detections": ["excavator"],
            },
            {
                "timestamp": "2026-09-27T10:01:00+03:00",
                "camera_id": "cam-b",
                "zone_id": "A",
                "detections": ["excavator", "dump_truck"],
            },
        ]
        with self.assertRaisesRegex(ValueError, "camera_id"):
            analyze_temporal(self.engine, "Устройство котлована", observations)

    def test_36_temporal_rejects_mixed_zones(self) -> None:
        observations = [
            {
                "timestamp": "2026-09-27T10:00:00+03:00",
                "camera_id": "cam-a",
                "zone_id": "A",
                "detections": ["excavator"],
            },
            {
                "timestamp": "2026-09-27T10:01:00+03:00",
                "camera_id": "cam-a",
                "zone_id": "B",
                "detections": ["excavator", "dump_truck"],
            },
        ]
        with self.assertRaisesRegex(ValueError, "zone_id"):
            analyze_temporal(self.engine, "Устройство котлована", observations)

    def test_37_temporal_rejects_changed_planned_work(self) -> None:
        observations = [
            {
                "timestamp": "2026-09-27T10:00:00+03:00",
                "planned_work": "Устройство котлована",
                "detections": ["excavator"],
            },
            {
                "timestamp": "2026-09-27T10:01:00+03:00",
                "planned_work": "Обратная засыпка",
                "detections": ["excavator"],
            },
        ]
        with self.assertRaisesRegex(ValueError, "planned_work"):
            analyze_temporal(self.engine, "Устройство котлована", observations)

    def test_38_temporal_rejects_invalid_or_duplicate_timestamps(self) -> None:
        invalid = [{"timestamp": "not-a-date", "detections": []}]
        with self.assertRaisesRegex(ValueError, "timestamp"):
            analyze_temporal(self.engine, "Устройство котлована", invalid)

        duplicate = [
            {"timestamp": "2026-09-27T10:00:00+03:00", "detections": []},
            {"timestamp": "2026-09-27T10:00:00+03:00", "detections": []},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_temporal(self.engine, "Устройство котлована", duplicate)

    def test_39_missing_and_uncertain_same_requirement_never_become_ok(self) -> None:
        observations = [
            {
                "timestamp": f"2026-09-27T10:0{index}:00+03:00",
                "detections": (
                    [{"class": "excavator", "zone_id": "A"}]
                    if index < 2
                    else [
                        {"class": "excavator", "zone_id": "A"},
                        {"class": "dump_truck"},
                    ]
                ),
                "target_zone_id": "A",
            }
            for index in range(4)
        ]
        result = analyze_temporal(
            self.engine,
            "Устройство котлована",
            observations,
            config=TemporalConfig(min_observations=3, persistence_ratio=0.75),
        )
        self.assertEqual(result["persistent_status"], "UNCERTAIN")
        self.assertEqual(
            result["temporal"]["persistent_uncertain"][0]["requirement"]["id"],
            "required:dump_truck",
        )

    def test_40_catalog_lookup_is_order_independent(self) -> None:
        shuffled = list(self.catalog.records)
        random.Random(1729).shuffle(shuffled)
        shuffled_catalog = WorkCatalog(shuffled)
        for record in self.catalog.records:
            self.assertEqual(
                shuffled_catalog.find(record.work).profile_id,
                record.profile_id,
            )

    def test_41_requirement_cannot_be_satisfied_and_uncertain(self) -> None:
        for profile_id in RULES:
            result = self.engine.analyze_profile(
                profile_id,
                [
                    {"class": class_name, "confidence": 1.0}
                    for class_name in MODEL_CLASSES
                ],
                target_zone_id="A",
            )
            satisfied = {item["id"] for item in result["match"]["satisfied"]}
            uncertain = {item["id"] for item in result["match"]["uncertain"]}
            self.assertTrue(satisfied.isdisjoint(uncertain))

    def test_42_all_profile_explanations_are_deterministic_and_consistent(self) -> None:
        for profile_id in RULES:
            first = self.engine.analyze_profile(profile_id, [])
            second = self.engine.analyze_profile(profile_id, [])
            self.assertEqual(first["explanation"], second["explanation"])
            if first["status"] == "WARNING":
                self.assertTrue(first["match"]["missing"])
            elif first["status"] == "UNCERTAIN":
                self.assertTrue(first["match"]["uncertain"])
                self.assertNotIn("нарушение доказано", first["explanation"].casefold())
            elif first["status"] == "NOT_MONITORABLE":
                self.assertEqual(first["observability"]["level"], "NONE")

    def test_43_strict_json_round_trip_for_profiles_statuses_and_temporal(self) -> None:
        results = [
            self.engine.analyze_profile(profile_id, []) for profile_id in RULES
        ]
        results.extend(
            [
                self.engine.analyze_profile(15, ["excavator", "dump_truck"]),
                self.engine.analyze_profile(30, ["truck", "mobile_crane"]),
                self.engine.analyze_work("Несуществующая работа", []),
                analyze_temporal(
                    self.engine,
                    "Устройство котлована",
                    [
                        {
                            "timestamp": f"2026-09-27T11:0{index}:00+03:00",
                            "detections": ["excavator"],
                        }
                        for index in range(4)
                    ],
                ),
            ]
        )
        for result in results:
            with self.subTest(status=result["status"], profile=result.get("profile_id")):
                encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
                self.assertEqual(json.loads(encoded), result)

    def test_44_not_monitorable_explanation_keeps_undetectable_alternatives(self) -> None:
        result = self.engine.analyze_profile(1, [])
        self.assertEqual(result["status"], "NOT_MONITORABLE")
        self.assertIn("грузовой подъемник", result["explanation"])
        self.assertIn("мачтовый подъемник", result["explanation"])
        self.assertIn("при высотных работах", result["explanation"])

    def test_45_supporting_detection_is_explained_as_non_confirming(self) -> None:
        result = self.engine.analyze_profile(
            15, ["excavator", "dump_truck", "bulldozer"]
        )
        self.assertEqual(result["status"], "OK")
        self.assertIn("условие не подтверждено", result["explanation"].casefold())

    def test_46_persistence_ratio_boundary_is_inclusive(self) -> None:
        observations = [
            {
                "timestamp": f"2026-09-27T12:0{index}:00+03:00",
                "detections": (
                    ["excavator"]
                    if index < 3
                    else ["excavator", "dump_truck"]
                ),
            }
            for index in range(4)
        ]
        result = analyze_temporal(
            self.engine,
            "Устройство котлована",
            observations,
            config=TemporalConfig(min_observations=3, persistence_ratio=0.75),
        )
        self.assertEqual(result["persistent_status"], "WARNING")
        self.assertEqual(
            result["temporal"]["persistent_missing"][0]["ratio"], 0.75
        )

    def test_47_exhaustive_small_rule_combinations_preserve_invariants(self) -> None:
        for profile_id, rule in RULES.items():
            classes = set(rule["required"])
            for alternative in rule["alternatives"]:
                classes.update(alternative["detectable"])
            for conditional in rule["conditional"]:
                classes.update(conditional["detectable"])
            ordered_classes = sorted(classes)
            condition_sets = [None]
            if rule["conditional"]:
                condition_sets.append(
                    {
                        item["condition"]: True
                        for item in rule["conditional"]
                    }
                )

            for mask in range(1 << len(ordered_classes)):
                detections = [
                    class_name
                    for index, class_name in enumerate(ordered_classes)
                    if mask & (1 << index)
                ]
                for active_conditions in condition_sets:
                    with self.subTest(
                        profile_id=profile_id,
                        detections=detections,
                        active=active_conditions is not None,
                    ):
                        result = self.engine.analyze_profile(
                            profile_id,
                            detections,
                            active_conditions=active_conditions,
                        )
                        match = result["match"]
                        bucket_ids = [
                            {item["id"] for item in match[bucket]}
                            for bucket in ("satisfied", "missing", "uncertain")
                        ]
                        self.assertEqual(
                            sum(len(ids) for ids in bucket_ids),
                            len(set().union(*bucket_ids)),
                        )

                        assessed = len(match["satisfied"]) + len(match["missing"])
                        total = assessed + len(match["uncertain"])
                        expected_score = (
                            round(len(match["satisfied"]) / assessed, 3)
                            if assessed
                            else None
                        )
                        expected_coverage = (
                            round(assessed / total, 3)
                            if total
                            else 0.0
                        )
                        self.assertEqual(match["score"], expected_score)
                        self.assertEqual(match["coverage"], expected_coverage)

                        current_observability = calculate_observability(
                            rule,
                            active_conditions=active_conditions,
                        )
                        self.assertEqual(
                            result["observability"]["score"],
                            current_observability,
                        )
                        if match["missing"]:
                            expected_status = "WARNING"
                        elif match["uncertain"]:
                            expected_status = (
                                "UNCERTAIN"
                                if current_observability > 0
                                else "NOT_MONITORABLE"
                            )
                        elif assessed:
                            expected_status = "OK"
                        else:
                            expected_status = "NOT_MONITORABLE"
                        self.assertEqual(result["status"], expected_status)

    def test_48_model_class_contract_is_exact_and_display_names_are_complete(self) -> None:
        self.assertEqual(
            MODEL_CLASSES,
            (
                "excavator",
                "dump_truck",
                "truck",
                "loader",
                "bulldozer",
                "motor_grader",
                "roller",
                "concrete_mixer",
                "telehandler",
                "piling_machine",
                "crane_manipulator",
                "mobile_crane",
                "tower_crane",
            ),
        )
        self.assertEqual(set(DISPLAY_NAMES_RU), set(MODEL_CLASSES))
        self.assertTrue(all(DISPLAY_NAMES_RU.values()))

    def test_49_user_facing_explanations_do_not_leak_internal_english_terms(self) -> None:
        forbidden = ("hard warning", "detections", "detector", "evidence")
        results = [self.engine.analyze_profile(profile_id, []) for profile_id in RULES]
        results.append(
            self.engine.analyze_profile(
                15,
                [
                    "excavator",
                    "dump_truck",
                    "bulldozer",
                    "unknown_machine",
                ],
            )
        )
        for result in results:
            explanation = result["explanation"].casefold()
            for term in forbidden:
                self.assertNotIn(term, explanation)

    def test_50_profile_0_stays_not_monitorable_until_ppr_override_exists(self) -> None:
        result = self.engine.analyze_profile(0, [])
        self.assertFalse(RULES[0]["monitorable"])
        self.assertEqual(result["status"], "NOT_MONITORABLE")
        self.assertEqual(result["observability"], {"score": 0.0, "level": "NONE"})
        self.assertEqual(result["match"]["uncertain_count"], 1)

    def test_51_profile_2_is_monitorable_only_while_delivery_condition_is_active(self) -> None:
        condition = "при необходимости, для доставки и погрузки"
        inactive = self.engine.analyze_profile(2, [])
        missing = self.engine.analyze_profile(
            2, [], active_conditions={condition: True}
        )
        truck = self.engine.analyze_profile(
            2, ["truck"], active_conditions={condition: True}
        )
        manipulator = self.engine.analyze_profile(
            2, ["crane_manipulator"], active_conditions={condition: True}
        )

        self.assertEqual(inactive["status"], "NOT_MONITORABLE")
        self.assertEqual(inactive["observability"]["score"], 0.0)
        self.assertEqual(missing["status"], "WARNING")
        self.assertEqual(missing["match"]["score"], 0.0)
        self.assertEqual(missing["match"]["coverage"], 1.0)
        self.assertEqual(truck["status"], "OK")
        self.assertEqual(manipulator["status"], "OK")
        self.assertEqual(truck["observability"]["score"], 1.0)

    def test_52_profile_4_hybrid_keeps_primary_uncertainty_and_checks_heavy_blocks(self) -> None:
        condition = "при тяжелых блоках"
        inactive = self.engine.analyze_profile(4, [])
        missing = self.engine.analyze_profile(
            4, [], active_conditions={condition: True}
        )
        present = self.engine.analyze_profile(
            4, ["crane_manipulator"], active_conditions={condition: True}
        )

        self.assertEqual(inactive["status"], "NOT_MONITORABLE")
        self.assertEqual(missing["status"], "WARNING")
        self.assertEqual(missing["match"]["missing_count"], 1)
        self.assertEqual(missing["match"]["uncertain_count"], 1)
        self.assertEqual(missing["match"]["score"], 0.0)
        self.assertEqual(missing["match"]["coverage"], 0.5)
        self.assertEqual(present["status"], "UNCERTAIN")
        self.assertEqual(present["match"]["score"], 1.0)
        self.assertEqual(present["match"]["coverage"], 0.5)

    def test_53_profile_11_keeps_generic_car_undetectable_and_conditions_explicit(self) -> None:
        rule = RULES[11]
        self.assertEqual(rule["alternatives"][0]["detectable"], ["crane_manipulator"])
        self.assertEqual(rule["alternatives"][0]["undetectable"], ["автомобиль"])

        truck = self.engine.analyze_profile(11, ["truck"])
        manipulator = self.engine.analyze_profile(11, ["crane_manipulator"])
        planned_work_text = self.engine.analyze_profile(
            11,
            ["crane_manipulator"],
            planned_work="Монтаж и разметка",
        )

        self.assertEqual(truck["status"], "UNCERTAIN")
        self.assertEqual(truck["match"]["satisfied"], [])
        self.assertEqual(manipulator["status"], "OK")
        self.assertEqual(planned_work_text["match"]["supporting"], [])
        self.assertFalse(
            any(
                item["id"].startswith("conditional:")
                for bucket in ("satisfied", "missing", "uncertain")
                for item in planned_work_text["match"][bucket]
            )
        )

    def test_54_profile_12_active_mixed_alternative_is_uncertain_not_warning(self) -> None:
        condition = "для тяжелого оборудования"
        missing = self.engine.analyze_profile(
            12, [], active_conditions={condition: True}
        )
        present = self.engine.analyze_profile(
            12, ["crane_manipulator"], active_conditions={condition: True}
        )

        self.assertEqual(missing["status"], "UNCERTAIN")
        self.assertEqual(missing["match"]["missing_count"], 0)
        self.assertEqual(missing["match"]["uncertain_count"], 3)
        self.assertIsNone(missing["match"]["score"])
        self.assertEqual(missing["match"]["coverage"], 0.0)
        self.assertEqual(present["status"], "UNCERTAIN")
        self.assertEqual(present["match"]["score"], 1.0)
        self.assertEqual(present["match"]["coverage"], 0.333)

    def test_55_profile_13_mini_excavator_mapping_stays_unverified(self) -> None:
        rule = RULES[13]
        self.assertEqual(rule["alternatives"][0]["detectable"], [])
        self.assertEqual(
            rule["alternatives"][0]["undetectable"],
            ["мини-экскаватор", "ямокопатель"],
        )

        result = self.engine.analyze_profile(13, ["dump_truck", "excavator"])
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertEqual(result["match"]["confirmed_count"], 1)
        self.assertEqual(result["match"]["uncertain_count"], 3)

        self.assertEqual(RULES[27]["alternatives"][0]["detectable"], [])
        self.assertEqual(
            RULES[27]["alternatives"][0]["undetectable"],
            ["ямобур", "мини-экскаватор"],
        )

    def test_56_profile_24_concrete_base_condition_is_never_inferred_from_work_text(self) -> None:
        condition = "для бетонного основания"
        inactive = self.engine.analyze_profile(
            24,
            ["dump_truck", "roller", "motor_grader"],
            planned_work="Устройство бетонного основания",
        )
        active = self.engine.analyze_profile(
            24,
            ["dump_truck", "roller", "motor_grader"],
            active_conditions={condition: True},
            planned_work="Любая работа",
        )

        self.assertEqual(inactive["status"], "OK")
        self.assertFalse(inactive["match"]["missing"])
        self.assertEqual(active["status"], "WARNING")
        self.assertEqual(active["match"]["missing"][0]["classes"], ["concrete_mixer"])

    def test_57_profile_26_cranes_are_a_mixed_alternative(self) -> None:
        rule = RULES[26]
        self.assertEqual(rule["required"], [])
        self.assertEqual(
            rule["alternatives"],
            [
                {
                    "detectable": ["mobile_crane"],
                    "undetectable": ["монтажный кран"],
                }
            ],
        )

        absent = self.engine.analyze_profile(26, [])
        present = self.engine.analyze_profile(26, ["mobile_crane"])
        self.assertEqual(absent["status"], "UNCERTAIN")
        self.assertEqual(absent["match"]["missing_count"], 0)
        self.assertEqual(present["status"], "OK")

    def test_58_profile_33_hybrid_keeps_primary_uncertainty_and_checks_large_blocks(self) -> None:
        condition = "при крупных блоках"
        inactive = self.engine.analyze_profile(33, [])
        missing = self.engine.analyze_profile(
            33, [], active_conditions={condition: True}
        )
        present = self.engine.analyze_profile(
            33, ["mobile_crane"], active_conditions={condition: True}
        )

        self.assertEqual(inactive["status"], "NOT_MONITORABLE")
        self.assertEqual(missing["status"], "WARNING")
        self.assertEqual(missing["match"]["missing_count"], 1)
        self.assertEqual(missing["match"]["uncertain_count"], 2)
        self.assertEqual(missing["match"]["coverage"], 0.333)
        self.assertEqual(present["status"], "UNCERTAIN")
        self.assertEqual(present["match"]["score"], 1.0)
        self.assertEqual(present["match"]["coverage"], 0.333)

    def test_59_observability_uses_only_currently_active_requirements(self) -> None:
        condition = "при необходимости, для доставки и погрузки"
        self.assertEqual(calculate_observability(RULES[2]), 0.0)
        self.assertEqual(
            calculate_observability(RULES[2], active_conditions={condition: True}),
            1.0,
        )
        self.assertEqual(calculate_observability(RULES[4]), 0.0)
        self.assertEqual(
            calculate_observability(
                RULES[4], active_conditions={"при тяжелых блоках": True}
            ),
            0.5,
        )

    def test_60_taxonomy_transition_keeps_ambiguous_loader_terms_unmapped(self) -> None:
        removed_classes = {
            "gazelle",
            "forklift_standard",
            "bucket_loader",
            "tanker",
            "trailer",
            "forklift_giraffe",
            "tractor",
        }
        self.assertTrue(removed_classes.isdisjoint(MODEL_CLASSES))

        self.assertEqual(RULES[29]["required"], ["truck"])
        self.assertIn("трактор", RULES[29]["undetectable"])

        for profile_id in (13, 18, 19, 29):
            rule = RULES[profile_id]
            detectable = set(rule["required"])
            for alternative in rule["alternatives"]:
                detectable.update(alternative["detectable"])
            for conditional in rule["conditional"]:
                detectable.update(conditional["detectable"])
            self.assertTrue(
                {"loader", "telehandler"}.isdisjoint(detectable),
                profile_id,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
