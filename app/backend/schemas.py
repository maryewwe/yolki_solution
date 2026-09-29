from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from construction_monitoring import MODEL_CLASSES


Status = Literal["OK", "WARNING", "UNCERTAIN", "NOT_MONITORABLE"]


class DetectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_name: str = Field(
        alias="class",
        min_length=1,
        json_schema_extra={"enum": list(MODEL_CLASSES)},
    )
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Tuple[float, float, float, float]
    zone_id: Optional[str] = Field(default=None, min_length=1)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        x_min, y_min, x_max, y_max = value
        if min(value) < 0 or x_max <= x_min or y_max <= y_min:
            raise ValueError(
                "bbox must be non-negative XYXY with max greater than min"
            )
        return value

    def to_contract(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class AnalyzeDetectionsRequest(BaseModel):
    """JSON mode for callers that already have canonical CV detections."""

    model_config = ConfigDict(extra="forbid")

    planned_work: str = Field(min_length=1)
    detections: List[DetectionInput]
    active_conditions: Dict[str, bool] = Field(default_factory=dict)
    timestamp: Optional[str] = None
    camera_id: str = Field(default="demo-camera", min_length=1)
    zone_id: Optional[str] = Field(default=None, min_length=1)


class ImageAnalysisFields(BaseModel):
    """Validated non-file fields from multipart image requests."""

    model_config = ConfigDict(extra="forbid")

    planned_work: str = Field(min_length=1)
    active_conditions: Dict[str, bool] = Field(default_factory=dict)
    timestamp: Optional[str] = None
    camera_id: str = Field(default="demo-camera", min_length=1)
    zone_id: Optional[str] = Field(default=None, min_length=1)


class SeriesAnalysisFields(BaseModel):
    """Validated non-file fields from multipart series requests."""

    model_config = ConfigDict(extra="forbid")

    planned_work: str = Field(min_length=1)
    active_conditions: Dict[str, bool] = Field(default_factory=dict)
    camera_id: str = Field(min_length=1)
    zone_id: Optional[str] = Field(default=None, min_length=1)
    timestamps: Optional[List[str]] = None

    @field_validator("timestamps")
    @classmethod
    def validate_timestamps(
        cls, value: Optional[List[str]]
    ) -> Optional[List[str]]:
        if value is None:
            return value
        if not value:
            raise ValueError("timestamps must be a non-empty JSON array")
        parsed = []
        for timestamp in value:
            try:
                item = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (AttributeError, ValueError) as exc:
                raise ValueError(
                    "timestamps must contain ISO 8601 strings with timezone"
                ) from exc
            if item.tzinfo is None or item.utcoffset() is None:
                raise ValueError(
                    "timestamps must contain ISO 8601 strings with timezone"
                )
            parsed.append(item)
        if len(set(parsed)) != len(parsed):
            raise ValueError("timestamps must be unique within one series")
        return value


class HealthResponse(BaseModel):
    status: Literal["ok"]
    cv_provider: str
    cv_ready: bool
    confidence_threshold: float


class WorkOption(BaseModel):
    planned_work: str
    work_group: str
    active_conditions: List[str]


class WorksResponse(BaseModel):
    count: int
    works: List[WorkOption]


class ObservabilityResponse(BaseModel):
    score: float
    level: Literal["NONE", "LOW", "MEDIUM", "HIGH"]


class MatchResponse(BaseModel):
    score: Optional[float]
    coverage: float
    confirmed_count: int
    missing_count: int
    uncertain_count: int
    assessed_count: int
    satisfied: List[Dict[str, Any]]
    missing: List[Dict[str, Any]]
    uncertain: List[Dict[str, Any]]
    supporting: List[Dict[str, Any]]


class DetectionBucketsResponse(BaseModel):
    accepted: List[Dict[str, Any]]
    rejected_low_confidence: List[Dict[str, Any]]
    rejected_unknown_class: List[Dict[str, Any]]
    rejected_zone_mismatch: List[Dict[str, Any]]
    rejected_invalid: List[Dict[str, Any]]


class AnalysisResponse(BaseModel):
    """Existing construction_monitoring response, represented for OpenAPI."""

    model_config = ConfigDict(extra="forbid")

    planned_work: str
    work_group: str
    profile_id: int
    status: Status
    instant_status: Status
    persistent_status: None
    observability: ObservabilityResponse
    match: MatchResponse
    detections: DetectionBucketsResponse
    possible_unexpected: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    explanation: str
    requirements: Dict[str, Any]
    metadata: Dict[str, Any]
    uncertainty_present: bool


class SeriesFrameResponse(BaseModel):
    index: int
    filename: str
    timestamp: str
    instant_status: Status
    observability: ObservabilityResponse
    match: MatchResponse
    detections: DetectionBucketsResponse
    possible_unexpected: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    explanation: str
    timing_seconds: Dict[str, float]


class SeriesAnalysisResponse(BaseModel):
    series_summary: Dict[str, Any]
    frames: List[SeriesFrameResponse]
    metadata: Dict[str, Any]


class ErrorIssue(BaseModel):
    path: str
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    issues: List[ErrorIssue] = Field(default_factory=list)
