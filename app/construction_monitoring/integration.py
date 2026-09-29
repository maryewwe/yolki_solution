from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .catalog import WorkCatalog, WorkRecord
from .config import TemporalConfig
from .engine import MatchingEngine
from .knowledge_base import MODEL_CLASSES, RULES
from .temporal import analyze_temporal as _analyze_temporal


_CATALOG = WorkCatalog.load()
_ENGINE = MatchingEngine(catalog=_CATALOG)

_REQUEST_KEYS = {"context", "detections"}
_CONTEXT_KEYS = {
    "planned_work",
    "active_conditions",
    "timestamp",
    "camera_id",
    "zone_id",
}
_DETECTION_KEYS = {"class", "confidence", "bbox", "zone_id"}
_TEMPORAL_KEYS = {"observations", "temporal_config"}
_TEMPORAL_CONFIG_KEYS = {
    "window_seconds",
    "min_observations",
    "persistence_ratio",
}


class IntegrationValidationError(ValueError):
    def __init__(self, issues: Sequence[Mapping[str, str]]) -> None:
        self.issues = [dict(item) for item in issues]
        super().__init__("Invalid integration request")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "error": "INVALID_REQUEST",
            "issues": [dict(item) for item in self.issues],
        }


def _issue(
    issues: List[Dict[str, str]],
    path: str,
    code: str,
    message: str,
) -> None:
    issues.append({"path": path, "code": code, "message": message})


def _check_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    path: str,
    issues: List[Dict[str, str]],
) -> None:
    for key in sorted(set(value) - allowed):
        _issue(
            issues,
            f"{path}.{key}",
            "unknown_field",
            "Field is not part of the integration contract.",
        )


def _nonempty_string(
    value: Any,
    path: str,
    issues: List[Dict[str, str]],
    *,
    required: bool,
) -> Optional[str]:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        _issue(issues, path, "invalid_string", "Expected a non-empty string.")
        return None
    return value


def _timestamp(
    value: Any,
    path: str,
    issues: List[Dict[str, str]],
) -> Optional[str]:
    timestamp = _nonempty_string(value, path, issues, required=True)
    if timestamp is None:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        _issue(
            issues,
            path,
            "invalid_timestamp",
            "Expected an ISO 8601 timestamp with timezone.",
        )
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _issue(
            issues,
            path,
            "timezone_required",
            "Timestamp must include an explicit timezone offset or Z.",
        )
        return None
    return timestamp


def _number(
    value: Any,
    path: str,
    issues: List[Dict[str, str]],
) -> Optional[float]:
    if isinstance(value, bool):
        _issue(issues, path, "invalid_number", "Boolean is not a valid number.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _issue(issues, path, "invalid_number", "Expected a finite number.")
        return None
    if not math.isfinite(number):
        _issue(issues, path, "invalid_number", "Expected a finite number.")
        return None
    return number


def _bbox(
    value: Any,
    path: str,
    issues: List[Dict[str, str]],
) -> Optional[List[float]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        _issue(
            issues,
            path,
            "invalid_bbox",
            "Expected [x_min, y_min, x_max, y_max].",
        )
        return None
    coordinates = [
        _number(item, f"{path}[{index}]", issues)
        for index, item in enumerate(value)
    ]
    if any(item is None for item in coordinates):
        return None
    x_min, y_min, x_max, y_max = coordinates
    if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
        _issue(
            issues,
            path,
            "invalid_bbox",
            "BBox must be finite non-negative XYXY with max greater than min.",
        )
        return None
    return [x_min, y_min, x_max, y_max]


def _active_conditions(
    value: Any,
    record: Optional[WorkRecord],
    path: str,
    issues: List[Dict[str, str]],
) -> Dict[str, bool]:
    if value is None:
        _issue(
            issues,
            path,
            "invalid_active_conditions",
            "Expected an object whose values are booleans.",
        )
        return {}
    if not isinstance(value, Mapping):
        _issue(
            issues,
            path,
            "invalid_active_conditions",
            "Expected an object whose values are booleans.",
        )
        return {}

    result: Dict[str, bool] = {}
    known_conditions = (
        {item["condition"] for item in RULES[record.profile_id]["conditional"]}
        if record is not None
        else set()
    )
    for key, item in value.items():
        condition_path = f"{path}.{key}"
        if not isinstance(key, str) or not key:
            _issue(
                issues,
                path,
                "invalid_condition_name",
                "Condition names must be non-empty strings.",
            )
            continue
        if not isinstance(item, bool):
            _issue(
                issues,
                condition_path,
                "invalid_condition_state",
                "Condition state must be boolean.",
            )
            continue
        if record is not None and key not in known_conditions:
            _issue(
                issues,
                condition_path,
                "unknown_active_condition",
                "Condition is not defined for the selected planned_work profile.",
            )
            continue
        result[key] = item
    return result


def _detection(
    value: Any,
    path: str,
    issues: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        _issue(issues, path, "invalid_detection", "Expected a detection object.")
        return None
    _check_unknown_keys(value, _DETECTION_KEYS, path, issues)

    class_name = _nonempty_string(
        value.get("class"), f"{path}.class", issues, required=True
    )
    if class_name is not None and class_name not in MODEL_CLASSES:
        _issue(
            issues,
            f"{path}.class",
            "unknown_detection_class",
            "Class is not part of the current CV model contract.",
        )

    confidence = _number(value.get("confidence"), f"{path}.confidence", issues)
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        _issue(
            issues,
            f"{path}.confidence",
            "invalid_confidence",
            "Confidence must be between 0.0 and 1.0 inclusive.",
        )

    bbox = _bbox(value.get("bbox"), f"{path}.bbox", issues)
    zone_id = (
        _nonempty_string(
            value.get("zone_id"), f"{path}.zone_id", issues, required=True
        )
        if "zone_id" in value
        else None
    )

    if class_name is None or confidence is None or bbox is None:
        return None
    result: Dict[str, Any] = {
        "class": class_name,
        "confidence": confidence,
        "bbox": bbox,
    }
    if zone_id is not None:
        result["zone_id"] = zone_id
    return result


def _normalize_analysis_request(
    request: Any,
    path: str,
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    issues: List[Dict[str, str]] = []
    if not isinstance(request, Mapping):
        _issue(issues, path, "invalid_request", "Expected a JSON object.")
        return None, issues
    _check_unknown_keys(request, _REQUEST_KEYS, path, issues)

    context = request.get("context")
    if not isinstance(context, Mapping):
        _issue(
            issues,
            f"{path}.context",
            "invalid_context",
            "Expected a context object.",
        )
        context = {}
    else:
        _check_unknown_keys(context, _CONTEXT_KEYS, f"{path}.context", issues)

    planned_work = _nonempty_string(
        context.get("planned_work"),
        f"{path}.context.planned_work",
        issues,
        required=True,
    )
    record = _CATALOG.find(planned_work) if planned_work is not None else None
    if planned_work is not None and record is None:
        _issue(
            issues,
            f"{path}.context.planned_work",
            "unknown_planned_work",
            "planned_work is not present in the construction work catalog.",
        )

    active_conditions = _active_conditions(
        context.get("active_conditions", {}),
        record,
        f"{path}.context.active_conditions",
        issues,
    )
    timestamp = _timestamp(
        context.get("timestamp"), f"{path}.context.timestamp", issues
    )
    camera_id = _nonempty_string(
        context.get("camera_id"),
        f"{path}.context.camera_id",
        issues,
        required=True,
    )
    zone_id = (
        _nonempty_string(
            context.get("zone_id"),
            f"{path}.context.zone_id",
            issues,
            required=True,
        )
        if "zone_id" in context
        else None
    )

    raw_detections = request.get("detections")
    detections: List[Dict[str, Any]] = []
    if not isinstance(raw_detections, list):
        _issue(
            issues,
            f"{path}.detections",
            "invalid_detections",
            "Expected an array of detection objects.",
        )
    else:
        for index, item in enumerate(raw_detections):
            normalized = _detection(item, f"{path}.detections[{index}]", issues)
            if normalized is not None:
                detections.append(normalized)

    if issues:
        return None, issues
    assert record is not None and timestamp is not None and camera_id is not None
    return {
        "context": {
            "planned_work": record.work,
            "active_conditions": active_conditions,
            "timestamp": timestamp,
            "camera_id": camera_id,
            "zone_id": zone_id,
        },
        "detections": detections,
    }, issues


def _analyze_with_engine(
    request: Mapping[str, Any],
    engine: MatchingEngine,
) -> Dict[str, Any]:
    normalized, issues = _normalize_analysis_request(request, "request")
    if issues:
        raise IntegrationValidationError(issues)
    assert normalized is not None
    context = normalized["context"]
    return engine.analyze_work(
        planned_work=context["planned_work"],
        detections=normalized["detections"],
        active_conditions=context["active_conditions"],
        metadata={
            "timestamp": context["timestamp"],
            "camera_id": context["camera_id"],
            "zone_id": context["zone_id"],
        },
        target_zone_id=context["zone_id"],
    )


def analyze(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _analyze_with_engine(request, _ENGINE)


def analyze_rfdetr(request: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(request, Mapping):
        return _analyze_with_engine(request, _ENGINE)

    result = request.get("detections")
    data = getattr(result, "data", None)
    confidences = getattr(result, "confidence", None)
    boxes = getattr(result, "xyxy", None)
    names = data.get("class_name") if isinstance(data, Mapping) else None
    if names is None or confidences is None or boxes is None:
        raise IntegrationValidationError(
            [
                {
                    "path": "request.detections",
                    "code": "invalid_rfdetr_result",
                    "message": (
                        "Expected RF-DETR result with data['class_name'], "
                        "confidence and xyxy."
                    ),
                }
            ]
        )

    try:
        lengths = (len(names), len(confidences), len(boxes))
    except TypeError as exc:
        raise IntegrationValidationError(
            [
                {
                    "path": "request.detections",
                    "code": "invalid_rfdetr_result",
                    "message": "RF-DETR result arrays must be sized iterables.",
                }
            ]
        ) from exc
    if len(set(lengths)) != 1:
        raise IntegrationValidationError(
            [
                {
                    "path": "request.detections",
                    "code": "inconsistent_rfdetr_result",
                    "message": "RF-DETR class_name, confidence and xyxy lengths differ.",
                }
            ]
        )

    background_names = {
        "__background__",
        "background",
        "__no_object__",
        "no_object",
        "no-object",
        "no object",
    }
    detections: List[Dict[str, Any]] = []
    for raw_name, confidence, raw_box in zip(names, confidences, boxes):
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8")
        class_name = str(raw_name).strip().lower()
        if class_name in background_names:
            continue
        try:
            bbox = list(raw_box)
        except TypeError:
            bbox = raw_box
        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox,
            }
        )

    canonical_request = dict(request)
    canonical_request["detections"] = detections
    return _analyze_with_engine(canonical_request, _ENGINE)


def _temporal_config(
    value: Any,
    path: str,
    issues: List[Dict[str, str]],
) -> Optional[TemporalConfig]:
    if not isinstance(value, Mapping):
        _issue(
            issues,
            path,
            "invalid_temporal_config",
            "Expected a temporal_config object.",
        )
        return None
    _check_unknown_keys(value, _TEMPORAL_CONFIG_KEYS, path, issues)

    window_seconds = value.get("window_seconds")
    if window_seconds is not None:
        window_seconds = _number(
            window_seconds, f"{path}.window_seconds", issues
        )
        if window_seconds is not None and window_seconds <= 0:
            _issue(
                issues,
                f"{path}.window_seconds",
                "invalid_window_seconds",
                "window_seconds must be positive or null.",
            )

    min_observations = value.get("min_observations", 3)
    if (
        isinstance(min_observations, bool)
        or not isinstance(min_observations, int)
        or min_observations < 1
    ):
        _issue(
            issues,
            f"{path}.min_observations",
            "invalid_min_observations",
            "min_observations must be an integer of at least 1.",
        )

    persistence_ratio = _number(
        value.get("persistence_ratio", 0.75),
        f"{path}.persistence_ratio",
        issues,
    )
    if persistence_ratio is not None and not 0.0 < persistence_ratio <= 1.0:
        _issue(
            issues,
            f"{path}.persistence_ratio",
            "invalid_persistence_ratio",
            "persistence_ratio must be in (0.0, 1.0].",
        )

    if issues:
        return None
    assert isinstance(min_observations, int) and persistence_ratio is not None
    return TemporalConfig(
        window_seconds=window_seconds,
        min_observations=min_observations,
        persistence_ratio=persistence_ratio,
    )


def temporal_update(request: Mapping[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, str]] = []
    if not isinstance(request, Mapping):
        raise IntegrationValidationError(
            [
                {
                    "path": "request",
                    "code": "invalid_request",
                    "message": "Expected a JSON object.",
                }
            ]
        )
    _check_unknown_keys(request, _TEMPORAL_KEYS, "request", issues)

    raw_observations = request.get("observations")
    normalized_observations: List[Dict[str, Any]] = []
    if not isinstance(raw_observations, list) or not raw_observations:
        _issue(
            issues,
            "request.observations",
            "invalid_observations",
            "Expected a non-empty array of analysis requests.",
        )
    else:
        for index, observation in enumerate(raw_observations):
            normalized, observation_issues = _normalize_analysis_request(
                observation,
                f"request.observations[{index}]",
            )
            issues.extend(observation_issues)
            if normalized is not None:
                normalized_observations.append(normalized)

    config = _temporal_config(
        request.get("temporal_config", {}),
        "request.temporal_config",
        issues,
    )

    if normalized_observations:
        contexts = [item["context"] for item in normalized_observations]
        context_keys = {
            (
                item["planned_work"],
                item["camera_id"],
                item["zone_id"],
                tuple(sorted(item["active_conditions"].items())),
            )
            for item in contexts
        }
        if len(context_keys) != 1:
            _issue(
                issues,
                "request.observations",
                "mixed_temporal_context",
                "Temporal observations must share work, camera, zone and active_conditions.",
            )
        timestamps = [item["timestamp"] for item in contexts]
        if len(set(timestamps)) != len(timestamps):
            _issue(
                issues,
                "request.observations",
                "duplicate_timestamps",
                "Temporal observation timestamps must be unique.",
            )

    if issues:
        raise IntegrationValidationError(issues)
    assert normalized_observations and config is not None

    first_context = normalized_observations[0]["context"]
    observations = []
    for item in normalized_observations:
        context = item["context"]
        observations.append(
            {
                "planned_work": context["planned_work"],
                "timestamp": context["timestamp"],
                "camera_id": context["camera_id"],
                "zone_id": context["zone_id"],
                "target_zone_id": context["zone_id"],
                "active_conditions": context["active_conditions"],
                "detections": item["detections"],
            }
        )

    return _analyze_temporal(
        _ENGINE,
        first_context["planned_work"],
        observations,
        config=config,
        target_zone_id=first_context["zone_id"],
    )
