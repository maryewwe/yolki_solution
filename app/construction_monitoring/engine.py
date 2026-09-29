from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .catalog import WorkCatalog
from .config import MatchingConfig
from .knowledge_base import (
    DISPLAY_NAMES_RU,
    MODEL_CLASSES,
    PROFILE_DESCRIPTIONS,
    RULES,
    detectable_classes,
)


DetectionInput = str | Mapping[str, Any]


def _join_ru(items: Sequence[str]) -> str:
    unique = list(dict.fromkeys(item for item in items if item))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return ", ".join(unique[:-1]) + " и " + unique[-1]


def _display(class_name: str) -> str:
    return DISPLAY_NAMES_RU.get(class_name, class_name)


def _condition_states(
    active_conditions: Optional[Mapping[str, bool] | Iterable[str]],
) -> Mapping[str, bool]:
    if active_conditions is None:
        return {}
    if isinstance(active_conditions, Mapping):
        states: Dict[str, bool] = {}
        for key, value in active_conditions.items():
            if not isinstance(value, bool):
                raise TypeError("active_conditions mapping values must be bool")
            states[str(key)] = value
        return states
    if isinstance(active_conditions, (str, bytes)):
        raise TypeError("active_conditions must be a mapping or iterable of names")
    return {str(condition): True for condition in active_conditions}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        converted = item_method()
        if converted is not value:
            return _json_safe(converted)
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def calculate_observability(
    rule: Mapping[str, Any],
    active_conditions: Optional[Mapping[str, bool] | Iterable[str]] = None,
) -> float:
    observable_weight = 0.0
    total_weight = 0.0
    conditions = _condition_states(active_conditions)

    total_weight += len(rule["required"])
    observable_weight += len(rule["required"])

    for alternative in rule["alternatives"]:
        detectable = alternative["detectable"]
        undetectable = alternative["undetectable"]
        total_weight += 1.0
        if detectable and not undetectable:
            observable_weight += 1.0
        elif detectable and undetectable:
            observable_weight += 0.5

    total_weight += len(rule["undetectable"])

    for conditional in rule["conditional"]:
        if conditions.get(conditional["condition"]) is not True:
            continue

        detectable = conditional["detectable"]
        undetectable = conditional["undetectable"]
        relation = conditional["undetectable_relation"]

        if relation == "alternative":
            total_weight += 1.0
            if detectable and not undetectable:
                observable_weight += 1.0
            elif detectable and undetectable:
                observable_weight += 0.5
            continue

        if conditional["mode"] == "all":
            total_weight += len(detectable)
            observable_weight += len(detectable)
        elif detectable:
            total_weight += 1.0
            observable_weight += 1.0
        total_weight += len(undetectable)

    if total_weight == 0:
        return 0.0
    return round(observable_weight / total_weight, 3)


def observability_level(score: float) -> str:
    if score == 0:
        return "NONE"
    if score < 0.4:
        return "LOW"
    if score < 0.75:
        return "MEDIUM"
    return "HIGH"


class MatchingEngine:
    def __init__(
        self,
        catalog: Optional[WorkCatalog] = None,
        config: Optional[MatchingConfig] = None,
    ) -> None:
        self.catalog = catalog or WorkCatalog.load()
        self.config = config or MatchingConfig()

    def _normalize_detection(
        self,
        item: DetectionInput,
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if isinstance(item, str):
            if not self.config.accept_legacy_class_strings:
                return {
                    "class": item,
                    "confidence": None,
                    "bbox": None,
                    "camera_id": metadata.get("camera_id"),
                    "zone_id": metadata.get("zone_id"),
                    "invalid_reason": "legacy_string_disabled",
                }
            raw: Mapping[str, Any] = {"class": item, "confidence": 1.0}
        elif isinstance(item, Mapping):
            raw = item
        else:
            return {
                "class": None,
                "confidence": None,
                "bbox": None,
                "camera_id": metadata.get("camera_id"),
                "zone_id": metadata.get("zone_id"),
                "invalid_reason": "invalid_detection_type",
            }

        class_name = raw.get("class", raw.get("class_name"))
        class_name = str(class_name) if class_name is not None else None
        raw_confidence = raw.get("confidence", 1.0)
        try:
            if isinstance(raw_confidence, bool):
                raise TypeError
            confidence = float(raw_confidence)
        except (TypeError, ValueError, OverflowError):
            confidence = None
        if confidence is not None and (
            not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
        ):
            confidence = None

        bbox = raw.get("bbox")
        invalid_bbox = False
        if bbox is not None:
            if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
                invalid_bbox = True
                bbox = None
            elif len(bbox) != 4:
                invalid_bbox = True
                bbox = None
            else:
                try:
                    if any(isinstance(value, bool) for value in bbox):
                        raise TypeError
                    coordinates = [float(value) for value in bbox]
                except (TypeError, ValueError, OverflowError):
                    invalid_bbox = True
                    bbox = None
                else:
                    x_min, y_min, x_max, y_max = coordinates
                    invalid_bbox = not (
                        all(math.isfinite(value) for value in coordinates)
                        and x_min >= 0
                        and y_min >= 0
                        and x_max > x_min
                        and y_max > y_min
                    )
                    bbox = None if invalid_bbox else coordinates

        result = {
            "class": class_name,
            "display_name": _display(class_name) if class_name else None,
            "confidence": confidence,
            "bbox": bbox,
            "camera_id": _json_safe(
                raw.get("camera_id", metadata.get("camera_id"))
            ),
            "zone_id": _json_safe(raw.get("zone_id", metadata.get("zone_id"))),
        }
        if class_name is None:
            result["invalid_reason"] = "missing_class"
        elif confidence is None:
            result["invalid_reason"] = "invalid_confidence"
        elif invalid_bbox:
            result["invalid_reason"] = "invalid_bbox"
        return result

    def _prepare_detections(
        self,
        detections: Sequence[DetectionInput],
        metadata: Mapping[str, Any],
        target_zone_id: Optional[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        if (
            detections is None
            or isinstance(detections, (str, bytes, Mapping))
            or not isinstance(detections, Sequence)
        ):
            raise TypeError("detections must be a sequence of detection objects")
        prepared = {
            "accepted": [],
            "rejected_low_confidence": [],
            "rejected_unknown_class": [],
            "rejected_zone_mismatch": [],
            "rejected_invalid": [],
        }
        known_classes = set(MODEL_CLASSES)

        for item in detections:
            detection = self._normalize_detection(item, metadata)
            class_name = detection.get("class")
            confidence = detection.get("confidence")

            if detection.get("invalid_reason"):
                prepared["rejected_invalid"].append(detection)
                continue
            if class_name not in known_classes:
                prepared["rejected_unknown_class"].append(detection)
                continue
            if confidence < self.config.confidence_threshold:
                prepared["rejected_low_confidence"].append(detection)
                continue
            if (
                target_zone_id is not None
                and detection.get("zone_id") is not None
                and detection["zone_id"] != target_zone_id
            ):
                prepared["rejected_zone_mismatch"].append(detection)
                continue

            detection["zone_verified"] = not (
                target_zone_id is not None and detection.get("zone_id") is None
            )
            prepared["accepted"].append(detection)
        return prepared

    @staticmethod
    def _requirement_entry(
        requirement_id: str,
        requirement_type: str,
        classes: Sequence[str] = (),
        found_classes: Sequence[str] = (),
        source_terms: Sequence[str] = (),
        condition: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "id": requirement_id,
            "type": requirement_type,
            "classes": list(classes),
            "display_names": [_display(item) for item in classes],
            "found_classes": list(found_classes),
            "found_display_names": [_display(item) for item in found_classes],
            "source_terms": list(source_terms),
        }
        if condition is not None:
            entry["condition"] = condition
        if reason is not None:
            entry["reason"] = reason
        return entry

    def _evaluate_rule(
        self,
        rule: Mapping[str, Any],
        prepared: Mapping[str, List[Dict[str, Any]]],
        active_conditions: Optional[Mapping[str, bool] | Iterable[str]],
        target_zone_id: Optional[str],
    ) -> Dict[str, Any]:
        accepted = prepared["accepted"]
        accepted_classes = {item["class"] for item in accepted}
        verified_classes = {
            item["class"] for item in accepted if item.get("zone_verified", True)
        }
        conditions = _condition_states(active_conditions)

        satisfied: List[Dict[str, Any]] = []
        missing: List[Dict[str, Any]] = []
        uncertain: List[Dict[str, Any]] = []
        supporting: List[Dict[str, Any]] = []

        def add_satisfied_or_spatial_uncertain(entry: Dict[str, Any]) -> None:
            found_classes = entry.get("found_classes", [])
            if (
                target_zone_id is not None
                and found_classes
                and not any(item in verified_classes for item in found_classes)
            ):
                entry["reason"] = "zone_unverified"
                uncertain.append(entry)
            else:
                satisfied.append(entry)

        for class_name in rule["required"]:
            entry = self._requirement_entry(
                f"required:{class_name}",
                "required",
                classes=(class_name,),
                found_classes=(class_name,) if class_name in accepted_classes else (),
            )
            if class_name in accepted_classes:
                add_satisfied_or_spatial_uncertain(entry)
            else:
                missing.append(entry)

        for index, alternative in enumerate(rule["alternatives"]):
            detectable = alternative["detectable"]
            undetectable = alternative["undetectable"]
            found = [item for item in detectable if item in accepted_classes]
            entry = self._requirement_entry(
                f"alternative:{index}",
                "alternative",
                classes=detectable,
                found_classes=found,
                source_terms=undetectable,
            )
            if found:
                add_satisfied_or_spatial_uncertain(entry)
            elif undetectable:
                entry["reason"] = "undetectable_alternative"
                uncertain.append(entry)
            else:
                missing.append(entry)

        for index, source_term in enumerate(rule["undetectable"]):
            uncertain.append(
                self._requirement_entry(
                    f"undetectable:{index}",
                    "undetectable",
                    source_terms=(source_term,),
                    reason="undetectable_requirement",
                )
            )

        for index, conditional in enumerate(rule["conditional"]):
            condition = conditional["condition"]
            state = conditions.get(condition)
            detectable = conditional["detectable"]
            undetectable = conditional["undetectable"]
            found = [item for item in detectable if item in accepted_classes]

            if state is not True:
                if found:
                    supporting.append(
                        self._requirement_entry(
                            f"conditional_support:{index}",
                            "conditional_support",
                            classes=detectable,
                            found_classes=found,
                            condition=condition,
                            reason="condition_not_confirmed",
                        )
                    )
                continue

            if conditional["mode"] == "all":
                if (
                    undetectable
                    and conditional["undetectable_relation"] == "alternative"
                    and any(item not in accepted_classes for item in detectable)
                ):
                    uncertain.append(
                        self._requirement_entry(
                            f"conditional:{index}",
                            "conditional",
                            classes=detectable,
                            found_classes=found,
                            source_terms=undetectable,
                            condition=condition,
                            reason="undetectable_alternative",
                        )
                    )
                else:
                    for class_name in detectable:
                        entry = self._requirement_entry(
                            f"conditional:{index}:{class_name}",
                            "conditional",
                            classes=(class_name,),
                            found_classes=(class_name,)
                            if class_name in accepted_classes
                            else (),
                            condition=condition,
                        )
                        if class_name in accepted_classes:
                            add_satisfied_or_spatial_uncertain(entry)
                        else:
                            missing.append(entry)
            else:
                entry = self._requirement_entry(
                    f"conditional:{index}",
                    "conditional",
                    classes=detectable,
                    found_classes=found,
                    source_terms=undetectable,
                    condition=condition,
                )
                if found:
                    add_satisfied_or_spatial_uncertain(entry)
                elif undetectable and conditional["undetectable_relation"] == "alternative":
                    entry["reason"] = "undetectable_alternative"
                    uncertain.append(entry)
                elif detectable:
                    missing.append(entry)

            if undetectable and conditional["undetectable_relation"] == "additional":
                for source_index, source_term in enumerate(undetectable):
                    uncertain.append(
                        self._requirement_entry(
                            f"conditional_undetectable:{index}:{source_index}",
                            "conditional_undetectable",
                            source_terms=(source_term,),
                            condition=condition,
                            reason="undetectable_requirement",
                        )
                    )

        return {
            "satisfied": satisfied,
            "missing": missing,
            "uncertain": uncertain,
            "supporting": supporting,
        }

    @staticmethod
    def _match_summary(
        evaluation: Mapping[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        satisfied = evaluation["satisfied"]
        missing = evaluation["missing"]
        uncertain = evaluation["uncertain"]
        assessed_count = len(satisfied) + len(missing)
        known_or_unknown_count = assessed_count + len(uncertain)

        score: Optional[float]
        if assessed_count == 0:
            score = None
        else:
            score = round(len(satisfied) / assessed_count, 3)

        coverage = (
            round(assessed_count / known_or_unknown_count, 3)
            if known_or_unknown_count
            else 0.0
        )
        return {
            "score": score,
            "coverage": coverage,
            "confirmed_count": len(satisfied),
            "missing_count": len(missing),
            "uncertain_count": len(uncertain),
            "assessed_count": assessed_count,
            "satisfied": satisfied,
            "missing": missing,
            "uncertain": uncertain,
            "supporting": evaluation["supporting"],
        }

    @staticmethod
    def _build_evidence(
        match: Mapping[str, Any],
        prepared: Mapping[str, List[Dict[str, Any]]],
        possible_unexpected: Sequence[str],
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for item in match["missing"]:
            evidence.append(
                {
                    "code": "MISSING_REQUIREMENT",
                    "severity": "WARNING",
                    "requirement_id": item["id"],
                    "details": item,
                }
            )
        for item in match["uncertain"]:
            evidence.append(
                {
                    "code": "UNVERIFIABLE_REQUIREMENT",
                    "severity": "UNCERTAIN",
                    "requirement_id": item["id"],
                    "details": item,
                }
            )
        if prepared["rejected_low_confidence"]:
            evidence.append(
                {
                    "code": "LOW_CONFIDENCE_DETECTIONS",
                    "severity": "INFO",
                    "count": len(prepared["rejected_low_confidence"]),
                }
            )
        if prepared["rejected_unknown_class"]:
            evidence.append(
                {
                    "code": "UNKNOWN_DETECTOR_CLASSES",
                    "severity": "INFO",
                    "classes": sorted(
                        {
                            item.get("class")
                            for item in prepared["rejected_unknown_class"]
                            if item.get("class")
                        }
                    ),
                }
            )
        if prepared["rejected_zone_mismatch"]:
            evidence.append(
                {
                    "code": "ZONE_MISMATCH",
                    "severity": "INFO",
                    "count": len(prepared["rejected_zone_mismatch"]),
                }
            )
        if possible_unexpected:
            evidence.append(
                {
                    "code": "POSSIBLE_UNEXPECTED_EQUIPMENT",
                    "severity": "INFO",
                    "classes": list(possible_unexpected),
                    "hard_violation": False,
                }
            )
        return evidence

    @staticmethod
    def _explain(
        status: str,
        rule: Mapping[str, Any],
        match: Mapping[str, Any],
        prepared: Mapping[str, List[Dict[str, Any]]],
        possible_unexpected: Sequence[str],
    ) -> str:
        if status == "NOT_MONITORABLE":
            sentences = [
                "Этот профиль нельзя надежно контролировать по текущим классам "
                "CV-модели."
            ]
            primary_terms = list(rule["undetectable"])
            for alternative in rule["alternatives"]:
                primary_terms.extend(alternative["undetectable"])
            if primary_terms:
                sentences.append(
                    "Ненаблюдаемые основные требования или варианты: "
                    + _join_ru(primary_terms)
                    + "."
                )
            for conditional in rule["conditional"]:
                conditional_terms = [
                    *[_display(item) for item in conditional["detectable"]],
                    *conditional["undetectable"],
                ]
                if conditional_terms:
                    sentences.append(
                        f"Условная позиция «{conditional['condition']}»: "
                        + _join_ru(conditional_terms)
                        + "; без подтверждения условия она не считается обязательной."
                    )
            sentences.append("Доказуемое предупреждение не формируется.")
            return " ".join(sentences)

        sentences: List[str] = []
        required_names = [_display(item) for item in rule["required"]]
        if required_names:
            sentences.append(
                "Для запланированной работы ожидаются: "
                + _join_ru(required_names)
                + "."
            )

        found_names: List[str] = []
        for item in match["satisfied"]:
            found_names.extend(item["found_display_names"])
        if found_names:
            sentences.append("Обнаружены: " + _join_ru(found_names) + ".")

        for item in match["missing"]:
            names = item["display_names"]
            if item["type"] in {"alternative", "conditional"} and len(names) > 1:
                sentences.append(
                    "Не обнаружен ни один из допустимых вариантов: "
                    + _join_ru(names)
                    + "."
                )
            elif names:
                sentences.append(names[0].capitalize() + " не обнаружен.")

        for item in match["uncertain"]:
            reason = item.get("reason")
            detectable_names = item["display_names"]
            source_terms = item["source_terms"]
            if reason == "undetectable_alternative":
                absent = _join_ru(detectable_names) or "наблюдаемого варианта"
                allowed = _join_ru(source_terms)
                sentences.append(
                    f"Отсутствие «{absent}» не подтверждает отклонение: "
                    f"база допускает вариант «{allowed}», который текущая "
                    "CV-модель не распознает."
                )
            elif reason == "undetectable_requirement":
                sentences.append(
                    "Требование «"
                    + _join_ru(source_terms)
                    + "» текущая CV-модель проверить не может."
                )
            elif reason == "zone_unverified":
                sentences.append(
                    "Зона для обнаруженной техники «"
                    + _join_ru(item["found_display_names"])
                    + "» не подтверждена, поэтому пространственное соответствие "
                    "остается неопределенным."
                )

        for item in match["supporting"]:
            sentences.append(
                "Дополнительно обнаружены условные признаки: "
                + _join_ru(item["found_display_names"])
                + f"; условие не подтверждено: «{item['condition']}», поэтому "
                "это вспомогательное свидетельство, а не выполнение обязательного "
                "требования."
            )

        if status == "OK" and not match["missing"] and not match["uncertain"]:
            sentences.append("Все проверяемые требования подтверждены.")

        if prepared["rejected_low_confidence"]:
            sentences.append(
                f"Обнаружения с низкой уверенностью исключены: "
                f"{len(prepared['rejected_low_confidence'])}."
            )
        if prepared["rejected_unknown_class"]:
            sentences.append(
                "Неизвестные классы детектора исключены из сопоставления."
            )
        if possible_unexpected:
            sentences.append(
                "Дополнительно обнаружены: "
                + _join_ru([_display(item) for item in possible_unexpected])
                + ". Это информационный сигнал, а не доказанное нарушение."
            )
        return " ".join(sentences) or "Проверяемых требований не найдено."

    def analyze_profile(
        self,
        profile_id: int,
        detections: Sequence[DetectionInput],
        *,
        active_conditions: Optional[Mapping[str, bool] | Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        target_zone_id: Optional[str] = None,
        planned_work: Optional[str] = None,
        work_group: Optional[str] = None,
        source_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if profile_id not in RULES:
            raise KeyError(f"Unknown profile_id: {profile_id}")

        rule = RULES[profile_id]
        source_description = PROFILE_DESCRIPTIONS[profile_id]
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        result_metadata = _json_safe(dict(metadata or {}))
        if target_zone_id is None:
            target_zone_id = result_metadata.get("zone_id")
        else:
            target_zone_id = _json_safe(target_zone_id)
        result_metadata["target_zone_id"] = target_zone_id

        prepared = self._prepare_detections(
            detections=detections,
            metadata=result_metadata,
            target_zone_id=target_zone_id,
        )
        evaluation = self._evaluate_rule(
            rule=rule,
            prepared=prepared,
            active_conditions=active_conditions,
            target_zone_id=target_zone_id,
        )
        match = self._match_summary(evaluation)
        observability_score = calculate_observability(
            rule,
            active_conditions=active_conditions,
        )

        if match["missing"]:
            status = "WARNING"
        elif match["uncertain"]:
            status = "UNCERTAIN" if observability_score > 0 else "NOT_MONITORABLE"
        elif match["assessed_count"]:
            status = "OK"
        else:
            status = "NOT_MONITORABLE"

        expected_classes = detectable_classes(rule)
        accepted_classes = {item["class"] for item in prepared["accepted"]}
        possible_unexpected = (
            sorted(accepted_classes - expected_classes)
            if self.config.flag_possible_unexpected
            else []
        )
        evidence = self._build_evidence(
            match=match,
            prepared=prepared,
            possible_unexpected=possible_unexpected,
        )

        return {
            "planned_work": planned_work,
            "work_group": work_group,
            "profile_id": profile_id,
            "status": status,
            "instant_status": status,
            "persistent_status": None,
            "observability": {
                "score": observability_score,
                "level": observability_level(observability_score),
            },
            "match": match,
            "detections": prepared,
            "possible_unexpected": [
                {"class": item, "display_name": _display(item)}
                for item in possible_unexpected
            ],
            "evidence": evidence,
            "explanation": self._explain(
                status=status,
                rule=rule,
                match=match,
                prepared=prepared,
                possible_unexpected=possible_unexpected,
            ),
            "requirements": {
                "source_text": source_description,
                "source_key": source_key,
                "rule": deepcopy(rule),
            },
            "metadata": result_metadata,
            "uncertainty_present": bool(match["uncertain"]),
        }

    def analyze_work(
        self,
        planned_work: str,
        detections: Sequence[DetectionInput],
        *,
        active_conditions: Optional[Mapping[str, bool] | Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        target_zone_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = self.catalog.find(planned_work)
        if record is None:
            return {
                "planned_work": planned_work,
                "work_group": None,
                "profile_id": None,
                "status": "UNKNOWN_WORK",
                "instant_status": "UNKNOWN_WORK",
                "persistent_status": None,
                "observability": {"score": None, "level": "UNKNOWN"},
                "match": {
                    "score": None,
                    "coverage": 0.0,
                    "confirmed_count": 0,
                    "missing_count": 0,
                    "uncertain_count": 0,
                    "assessed_count": 0,
                    "satisfied": [],
                    "missing": [],
                    "uncertain": [],
                    "supporting": [],
                },
                "detections": {
                    "accepted": [],
                    "rejected_low_confidence": [],
                    "rejected_unknown_class": [],
                    "rejected_zone_mismatch": [],
                    "rejected_invalid": [],
                },
                "possible_unexpected": [],
                "evidence": [
                    {
                        "code": "UNKNOWN_WORK",
                        "severity": "ERROR",
                        "planned_work": planned_work,
                    }
                ],
                "explanation": "Работа отсутствует в базе знаний.",
                "requirements": None,
                "metadata": _json_safe(dict(metadata or {})),
                "uncertainty_present": False,
            }

        return self.analyze_profile(
            profile_id=record.profile_id,
            detections=detections,
            active_conditions=active_conditions,
            metadata=metadata,
            target_zone_id=target_zone_id,
            planned_work=record.work,
            work_group=record.work_group,
            source_key=record.source_key,
        )
