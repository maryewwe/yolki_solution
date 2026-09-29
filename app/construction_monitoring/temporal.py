from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import TemporalConfig
from .engine import MatchingEngine


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            timestamp = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _filter_window(
    observations: Sequence[Mapping[str, Any]],
    window_seconds: Optional[float],
) -> List[Mapping[str, Any]]:
    parsed = [(_parse_timestamp(item.get("timestamp")), item) for item in observations]
    if not observations:
        return []
    if all(timestamp is not None for timestamp, _ in parsed):
        parsed.sort(key=lambda pair: pair[0])
    if window_seconds is None:
        return [item for _, item in parsed]
    valid_times = [timestamp for timestamp, _ in parsed if timestamp is not None]
    if not valid_times:
        return list(observations)
    latest = max(valid_times)
    return [
        item
        for timestamp, item in parsed
        if timestamp is None or (latest - timestamp).total_seconds() <= window_seconds
    ]


def _validate_observations(
    planned_work: str,
    observations: Sequence[Mapping[str, Any]],
    target_zone_id: Optional[str],
) -> None:
    if (
        observations is None
        or isinstance(observations, (str, bytes, Mapping))
        or not isinstance(observations, Sequence)
    ):
        raise TypeError("observations must be a sequence of mappings")

    parsed_timestamps: List[datetime] = []
    expected_work = " ".join(str(planned_work).strip().casefold().split())
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise TypeError(f"observation {index} must be a mapping")
        timestamp = _parse_timestamp(observation.get("timestamp"))
        if timestamp is None:
            raise ValueError(f"observation {index} has invalid timestamp")
        parsed_timestamps.append(timestamp)

        observation_work = observation.get("planned_work")
        if observation_work is not None:
            normalized_work = " ".join(
                str(observation_work).strip().casefold().split()
            )
            if normalized_work != expected_work:
                raise ValueError(
                    f"observation {index} planned_work does not match the series"
                )

    if len(set(parsed_timestamps)) != len(parsed_timestamps):
        raise ValueError("duplicate timestamps are not allowed in one temporal series")

    def validate_stream_field(field_name: str, values: Sequence[Any]) -> None:
        present = [value for value in values if value is not None]
        if any(not isinstance(value, str) or not value for value in present):
            raise TypeError(f"{field_name} must be a non-empty string or null")
        if len(set(present)) > 1 or (present and len(present) != len(values)):
            raise ValueError(
                f"temporal series must contain one consistent {field_name}"
            )

    validate_stream_field(
        "camera_id", [observation.get("camera_id") for observation in observations]
    )
    validate_stream_field(
        "zone_id", [observation.get("zone_id") for observation in observations]
    )
    validate_stream_field(
        "target_zone_id",
        [
            observation.get("target_zone_id", target_zone_id)
            for observation in observations
        ],
    )


def analyze_temporal(
    engine: MatchingEngine,
    planned_work: str,
    observations: Sequence[Mapping[str, Any]],
    *,
    config: Optional[TemporalConfig] = None,
    target_zone_id: Optional[str] = None,
) -> Dict[str, Any]:
    temporal_config = config or TemporalConfig()
    _validate_observations(planned_work, observations, target_zone_id)
    window = _filter_window(observations, temporal_config.window_seconds)
    results: List[Dict[str, Any]] = []

    for observation in window:
        metadata = {
            "timestamp": observation.get("timestamp"),
            "camera_id": observation.get("camera_id"),
            "zone_id": observation.get("zone_id"),
        }
        result = engine.analyze_work(
            planned_work=planned_work,
            detections=observation.get("detections", []),
            active_conditions=observation.get("active_conditions"),
            metadata=metadata,
            target_zone_id=observation.get("target_zone_id", target_zone_id),
        )
        results.append(result)

    if not results:
        return {
            "planned_work": planned_work,
            "status": "INSUFFICIENT_DATA",
            "instant_status": None,
            "persistent_status": "INSUFFICIENT_DATA",
            "observation_count": 0,
            "explanation": "Нет наблюдений для временного анализа.",
            "temporal": {
                "window_seconds": temporal_config.window_seconds,
                "min_observations": temporal_config.min_observations,
                "persistence_ratio": temporal_config.persistence_ratio,
                "persistent_missing": [],
                "persistent_uncertain": [],
            },
            "observations": [],
        }

    latest = results[-1]
    instant_status = latest["instant_status"]

    if instant_status in {"UNKNOWN_WORK", "NOT_MONITORABLE"}:
        persistent_status = instant_status
        top_status = instant_status
        persistent_missing: List[Dict[str, Any]] = []
        persistent_uncertain: List[Dict[str, Any]] = []
        explanation = latest["explanation"]
    elif len(results) < temporal_config.min_observations:
        persistent_status = "INSUFFICIENT_DATA"
        top_status = instant_status
        persistent_missing = []
        persistent_uncertain = []
        explanation = (
            f"Для оценки устойчивости нужно минимум "
            f"{temporal_config.min_observations} наблюдения; получено "
            f"{len(results)}. Показан только instant status."
        )
    else:
        missing_counts: Counter[str] = Counter()
        uncertain_counts: Counter[str] = Counter()
        unresolved_counts: Counter[str] = Counter()
        requirement_details: Dict[str, Dict[str, Any]] = {}
        uncertain_details: Dict[str, Dict[str, Any]] = {}
        for result in results:
            for item in result["match"]["missing"]:
                missing_counts[item["id"]] += 1
                unresolved_counts[item["id"]] += 1
                requirement_details[item["id"]] = item
            for item in result["match"]["uncertain"]:
                uncertain_counts[item["id"]] += 1
                unresolved_counts[item["id"]] += 1
                requirement_details[item["id"]] = item
                uncertain_details[item["id"]] = item

        observation_count = len(results)
        persistent_missing = [
            {
                "requirement": requirement_details[requirement_id],
                "count": count,
                "ratio": round(count / observation_count, 3),
            }
            for requirement_id, count in sorted(missing_counts.items())
            if count / observation_count >= temporal_config.persistence_ratio
        ]
        persistent_uncertain = [
            {
                "requirement": uncertain_details[requirement_id],
                "count": count,
                "ratio": round(count / observation_count, 3),
                "missing_count": missing_counts[requirement_id],
                "uncertain_count": uncertain_counts[requirement_id],
            }
            for requirement_id, count in sorted(unresolved_counts.items())
            if uncertain_counts[requirement_id]
            and count / observation_count >= temporal_config.persistence_ratio
            and missing_counts[requirement_id] / observation_count
            < temporal_config.persistence_ratio
        ]

        if persistent_missing:
            persistent_status = "WARNING"
            if len(persistent_missing) == 1:
                requirement_text = "одного требования"
            else:
                requirement_text = f"{len(persistent_missing)} требований"
            explanation = (
                f"Устойчивое отсутствие подтверждено для {requirement_text} "
                f"в окне из {observation_count} наблюдений."
            )
        elif persistent_uncertain:
            persistent_status = "UNCERTAIN"
            explanation = (
                "Доказуемого устойчивого отсутствия нет, но ограничения "
                "наблюдаемости сохраняются в большинстве наблюдений."
            )
        else:
            persistent_status = "OK"
            transient_warning_count = sum(
                result["instant_status"] == "WARNING" for result in results
            )
            if transient_warning_count:
                warning_text = (
                    "одно неустойчивое предупреждение"
                    if transient_warning_count == 1
                    else f"{transient_warning_count} неустойчивых предупреждения"
                )
                explanation = (
                    f"Зафиксировано {warning_text}; порог persistence не достигнут."
                )
            else:
                explanation = "Устойчивых отклонений в выбранном окне не найдено."
        top_status = persistent_status

    return {
        "planned_work": planned_work,
        "work_group": latest.get("work_group"),
        "profile_id": latest.get("profile_id"),
        "status": top_status,
        "instant_status": instant_status,
        "persistent_status": persistent_status,
        "observation_count": len(results),
        "observability": latest.get("observability"),
        "match": latest.get("match"),
        "detections": latest.get("detections"),
        "possible_unexpected": latest.get("possible_unexpected"),
        "evidence": latest.get("evidence"),
        "metadata": latest.get("metadata"),
        "explanation": explanation,
        "temporal": {
            "window_seconds": temporal_config.window_seconds,
            "min_observations": temporal_config.min_observations,
            "persistence_ratio": temporal_config.persistence_ratio,
            "persistent_missing": persistent_missing,
            "persistent_uncertain": persistent_uncertain,
        },
        "observations": [
            {
                "timestamp": result["metadata"].get("timestamp"),
                "camera_id": result["metadata"].get("camera_id"),
                "zone_id": result["metadata"].get("zone_id"),
                "instant_status": result["instant_status"],
                "match_score": result["match"]["score"],
                "missing_requirement_ids": [
                    item["id"] for item in result["match"]["missing"]
                ],
                "uncertain_requirement_ids": [
                    item["id"] for item in result["match"]["uncertain"]
                ],
            }
            for result in results
        ],
        "latest_result": latest,
    }
