from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .knowledge_base import DESCRIPTION_TO_PROFILE_ID, PROFILE_DESCRIPTIONS, RULES


DEFAULT_CATALOG_PATH = Path(__file__).parent / "data" / "work_catalog.json"


@dataclass(frozen=True)
class WorkRecord:
    work: str
    work_group: str
    source_description: str
    source_key: str
    profile_id: int

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "WorkRecord":
        return cls(
            work=str(item["work"]),
            work_group=str(item["work_group"]),
            source_description=str(item["source_description"]),
            source_key=str(item.get("source_key", "")),
            profile_id=int(item["profile_id"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work": self.work,
            "work_group": self.work_group,
            "source_description": self.source_description,
            "source_key": self.source_key,
            "profile_id": self.profile_id,
        }


class WorkCatalog:
    def __init__(self, records: Iterable[WorkRecord]) -> None:
        self.records = tuple(records)
        self._by_exact: Dict[str, WorkRecord] = {}
        self._by_normalized: Dict[str, List[WorkRecord]] = {}
        for record in self.records:
            self._by_exact[record.work] = record
            normalized = self._normalize_name(record.work)
            self._by_normalized.setdefault(normalized, []).append(record)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CATALOG_PATH) -> "WorkCatalog":
        source_path = Path(path)
        with source_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        records_payload = payload.get("works", payload)
        return cls(WorkRecord.from_mapping(item) for item in records_payload)

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(value.strip().casefold().split())

    def find(self, planned_work: Any) -> Optional[WorkRecord]:
        if not isinstance(planned_work, str) or not planned_work.strip():
            return None
        if planned_work in self._by_exact:
            return self._by_exact[planned_work]
        candidates = self._by_normalized.get(self._normalize_name(planned_work), [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    def works_for_profile(self, profile_id: int) -> List[WorkRecord]:
        return [record for record in self.records if record.profile_id == profile_id]

    def validate(self, expected_work_count: int = 376) -> List[str]:
        issues: List[str] = []
        if len(self.records) != expected_work_count:
            issues.append(
                f"expected {expected_work_count} works, found {len(self.records)}"
            )

        records_by_name: Dict[str, List[WorkRecord]] = {}
        for record in self.records:
            records_by_name.setdefault(record.work, []).append(record)
        for work_name, duplicates in records_by_name.items():
            semantic_keys = {
                (
                    item.work_group,
                    item.source_description,
                    item.source_key,
                    item.profile_id,
                )
                for item in duplicates
            }
            if len(semantic_keys) > 1:
                issues.append(
                    f"work name {work_name!r} maps to conflicting catalog records"
                )

        seen_profile_ids = {record.profile_id for record in self.records}
        if seen_profile_ids != set(RULES):
            issues.append(
                "catalog profile coverage mismatch: "
                f"missing={sorted(set(RULES) - seen_profile_ids)}, "
                f"extra={sorted(seen_profile_ids - set(RULES))}"
            )

        for index, record in enumerate(self.records):
            expected_profile_id = DESCRIPTION_TO_PROFILE_ID.get(
                record.source_description
            )
            if expected_profile_id is None:
                issues.append(
                    f"row {index}: unknown source description {record.source_description!r}"
                )
            elif expected_profile_id != record.profile_id:
                issues.append(
                    f"row {index}: profile {record.profile_id} does not match "
                    f"description profile {expected_profile_id}"
                )
            if record.profile_id not in RULES:
                issues.append(f"row {index}: unknown profile ID {record.profile_id}")

        for profile_id, description in enumerate(PROFILE_DESCRIPTIONS):
            if not any(
                record.profile_id == profile_id
                and record.source_description == description
                for record in self.records
            ):
                issues.append(f"profile {profile_id}: no matching work records")
        return issues

    def assert_valid(self, expected_work_count: int = 376) -> None:
        issues = self.validate(expected_work_count=expected_work_count)
        if issues:
            raise ValueError("Invalid work catalog:\n- " + "\n- ".join(issues))
