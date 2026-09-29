from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from time import perf_counter
from typing import Any, Dict, Mapping, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from construction_monitoring import (
    IntegrationValidationError,
    MatchingConfig,
    RULES,
    WorkCatalog,
    analyze,
    analyze_rfdetr,
    temporal_update,
)

from .providers import (
    BadImageError,
    CVInferenceError,
    CVProvider,
    ProviderConfigurationError,
    provider_from_environment,
)
from .schemas import (
    AnalysisResponse,
    AnalyzeDetectionsRequest,
    ErrorResponse,
    HealthResponse,
    ImageAnalysisFields,
    SeriesAnalysisFields,
    SeriesAnalysisResponse,
    WorksResponse,
)


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
ALLOWED_MULTIPART_FIELDS = {
    "image",
    "planned_work",
    "active_conditions",
    "timestamp",
    "camera_id",
    "zone_id",
}
ALLOWED_SERIES_MULTIPART_FIELDS = {
    "files",
    "planned_work",
    "active_conditions",
    "timestamps",
    "camera_id",
    "zone_id",
}
DEFAULT_MAX_SERIES_FRAMES = 30


def _origins_from_environment() -> list[str]:
    configured = os.getenv("CORS_ORIGINS")
    if not configured:
        return list(DEFAULT_CORS_ORIGINS)
    return [item.strip() for item in configured.split(",") if item.strip()]


def _max_series_frames_from_environment() -> int:
    raw_value = os.getenv("MAX_SERIES_FRAMES", str(DEFAULT_MAX_SERIES_FRAMES))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ProviderConfigurationError(
            "MAX_SERIES_FRAMES must be a positive integer."
        ) from exc
    if value < 1:
        raise ProviderConfigurationError(
            "MAX_SERIES_FRAMES must be a positive integer."
        )
    return value


def _validation_issues(exc: ValidationError, prefix: str = "request") -> list[dict]:
    issues = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(item) for item in error["loc"])
        path = f"{prefix}.{location}" if location else prefix
        issues.append(
            {
                "path": path,
                "code": "validation_error",
                "message": error["msg"],
            }
        )
    return issues


def _error(status_code: int, code: str, message: str, path: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": code,
            "issues": [{"path": path, "code": code.lower(), "message": message}],
        },
    )


def _timestamp_or_now(value: Optional[str]) -> str:
    return value or datetime.now(timezone.utc).isoformat()


def _core_request(
    fields: AnalyzeDetectionsRequest | ImageAnalysisFields | SeriesAnalysisFields,
    detections: Any,
    *,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "planned_work": fields.planned_work,
        "active_conditions": fields.active_conditions,
        "timestamp": _timestamp_or_now(
            timestamp if timestamp is not None else getattr(fields, "timestamp", None)
        ),
        "camera_id": fields.camera_id,
    }
    if fields.zone_id is not None:
        context["zone_id"] = fields.zone_id
    return {"context": context, "detections": detections}


def _works_payload(catalog: WorkCatalog) -> Dict[str, Any]:
    works = []
    seen = set()
    for record in catalog.records:
        if record.work in seen:
            continue
        seen.add(record.work)
        conditions = sorted(
            {
                item["condition"]
                for item in RULES[record.profile_id]["conditional"]
            }
        )
        works.append(
            {
                "planned_work": record.work,
                "work_group": record.work_group,
                "active_conditions": conditions,
            }
        )
    return {"count": len(works), "works": works}


def _multipart_openapi_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["image", "planned_work"],
        "properties": {
            "image": {"type": "string", "format": "binary"},
            "planned_work": {"type": "string"},
            "active_conditions": {
                "type": "string",
                "default": "{}",
                "description": "JSON object encoded as a form string.",
            },
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "Optional; current UTC time is used when omitted.",
            },
            "camera_id": {"type": "string", "default": "demo-camera"},
            "zone_id": {"type": "string"},
        },
    }


def _series_multipart_openapi_schema(max_frames: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["files", "planned_work", "camera_id"],
        "properties": {
            "files": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_frames,
                "items": {"type": "string", "format": "binary"},
                "description": (
                    "Ordered image files. Repeat the multipart field name files."
                ),
            },
            "planned_work": {"type": "string"},
            "active_conditions": {
                "type": "string",
                "default": "{}",
                "description": "JSON object encoded as a form string.",
            },
            "timestamps": {
                "type": "string",
                "description": (
                    "Optional JSON array of unique ISO 8601 timestamps with "
                    "timezone, one per file."
                ),
            },
            "camera_id": {"type": "string"},
            "zone_id": {"type": "string"},
        },
    }


def _canonical_accepted_detections(result: Mapping[str, Any]) -> list[dict]:
    detections = []
    for item in result["detections"]["accepted"]:
        detection = {
            "class": item["class"],
            "confidence": item["confidence"],
            "bbox": item["bbox"],
        }
        if item.get("zone_id") is not None:
            detection["zone_id"] = item["zone_id"]
        detections.append(detection)
    return detections


def _add_persistent_frame_indices(
    summary: Dict[str, Any], instant_results: list[Mapping[str, Any]]
) -> None:
    temporal = summary.get("temporal", {})
    for item in temporal.get("persistent_missing", []):
        requirement_id = item["requirement"]["id"]
        item["frame_indices"] = [
            index
            for index, result in enumerate(instant_results)
            if requirement_id
            in {requirement["id"] for requirement in result["match"]["missing"]}
        ]
    for item in temporal.get("persistent_uncertain", []):
        requirement_id = item["requirement"]["id"]
        item["frame_indices"] = [
            index
            for index, result in enumerate(instant_results)
            if requirement_id
            in {
                requirement["id"]
                for bucket in ("missing", "uncertain")
                for requirement in result["match"][bucket]
            }
        ]


def _json_openapi_schema() -> Dict[str, Any]:
    schema = AnalyzeDetectionsRequest.model_json_schema()
    definitions = schema.pop("$defs", {})
    detection_schema = definitions.get("DetectionInput")
    if detection_schema is not None:
        schema["properties"]["detections"]["items"] = detection_schema
    return schema


def create_app(provider: Optional[CVProvider] = None) -> FastAPI:
    cv_provider = provider or provider_from_environment()
    catalog = WorkCatalog.load()
    max_series_frames = _max_series_frames_from_environment()

    application = FastAPI(
        title="Construction Monitoring Demo API",
        version="1.0.0",
        description=(
            "Minimal frontend -> CV -> construction_monitoring integration API. "
            "Swagger UI: /docs"
        ),
    )
    application.state.cv_provider = cv_provider
    application.state.max_series_frames = max_series_frames
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_origins_from_environment(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(IntegrationValidationError)
    async def integration_error_handler(
        request: Request, exc: IntegrationValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content=exc.as_dict())

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "INVALID_REQUEST",
                "issues": _validation_issues(exc),
            },
        )

    @application.get("/health", response_model=HealthResponse)
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "cv_provider": cv_provider.name,
            "cv_ready": cv_provider.ready,
            "confidence_threshold": MatchingConfig().confidence_threshold,
        }

    @application.get("/works", response_model=WorksResponse)
    async def works() -> Dict[str, Any]:
        return _works_payload(catalog)

    @application.post(
        "/analyze",
        response_model=AnalysisResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Unreadable image"},
            415: {"model": ErrorResponse, "description": "Unsupported content type"},
            422: {"model": ErrorResponse, "description": "Invalid request"},
            503: {"model": ErrorResponse, "description": "CV provider unavailable"},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": _json_openapi_schema()
                    },
                    "multipart/form-data": {
                        "schema": _multipart_openapi_schema()
                    },
                },
            }
        },
    )
    async def analyze_endpoint(request: Request) -> Mapping[str, Any] | JSONResponse:
        content_type = request.headers.get("content-type", "").lower()
        if content_type.startswith("application/json"):
            try:
                raw_payload = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                return _error(
                    422,
                    "INVALID_REQUEST",
                    "Request body must contain valid JSON.",
                    "request",
                )
            try:
                payload = AnalyzeDetectionsRequest.model_validate(raw_payload)
            except ValidationError as exc:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "INVALID_REQUEST",
                        "issues": _validation_issues(exc),
                    },
                )
            detections = [item.to_contract() for item in payload.detections]
            return analyze(_core_request(payload, detections))

        if content_type.startswith("multipart/form-data"):
            try:
                form = await request.form()
            except Exception:
                return _error(
                    422,
                    "INVALID_REQUEST",
                    "Malformed multipart form data.",
                    "request",
                )
            unknown_fields = sorted(set(form.keys()) - ALLOWED_MULTIPART_FIELDS)
            if unknown_fields:
                return _error(
                    422,
                    "INVALID_REQUEST",
                    f"Unknown form fields: {', '.join(unknown_fields)}.",
                    "request",
                )
            image = form.get("image")
            if not isinstance(image, UploadFile):
                return _error(
                    422,
                    "INVALID_REQUEST",
                    "A multipart image file is required.",
                    "request.image",
                )
            raw_conditions = form.get("active_conditions", "{}")
            try:
                active_conditions = json.loads(str(raw_conditions))
            except json.JSONDecodeError:
                return _error(
                    422,
                    "INVALID_REQUEST",
                    "active_conditions must be a JSON object string.",
                    "request.active_conditions",
                )
            raw_fields = {
                "planned_work": form.get("planned_work"),
                "active_conditions": active_conditions,
                "camera_id": form.get("camera_id", "demo-camera"),
            }
            if form.get("timestamp") not in (None, ""):
                raw_fields["timestamp"] = form.get("timestamp")
            if form.get("zone_id") not in (None, ""):
                raw_fields["zone_id"] = form.get("zone_id")
            try:
                fields = ImageAnalysisFields.model_validate(raw_fields)
            except ValidationError as exc:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "INVALID_REQUEST",
                        "issues": _validation_issues(exc),
                    },
                )
            try:
                image_bytes = await image.read()
                inference = cv_provider.infer(
                    image_bytes,
                    filename=image.filename,
                    content_type=image.content_type,
                )
            except BadImageError as exc:
                return _error(400, "BAD_IMAGE", str(exc), "request.image")
            except (ProviderConfigurationError, CVInferenceError) as exc:
                return _error(
                    503,
                    "CV_PROVIDER_UNAVAILABLE",
                    str(exc),
                    "cv_provider",
                )
            finally:
                await image.close()
            core_request = _core_request(fields, inference.detections)
            if inference.native_rfdetr:
                return analyze_rfdetr(core_request)
            return analyze(core_request)

        return _error(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            "Use application/json for detections or multipart/form-data for an image.",
            "request",
        )

    @application.post(
        "/analyze-series",
        response_model=SeriesAnalysisResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Unreadable image"},
            413: {"model": ErrorResponse, "description": "Too many frames"},
            422: {"model": ErrorResponse, "description": "Invalid request"},
            503: {"model": ErrorResponse, "description": "CV provider unavailable"},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": _series_multipart_openapi_schema(
                            max_series_frames
                        )
                    }
                },
            }
        },
    )
    async def analyze_series_endpoint(
        request: Request,
    ) -> Mapping[str, Any] | JSONResponse:
        request_started = perf_counter()
        try:
            form = await request.form()
        except Exception:
            return _error(
                422,
                "INVALID_REQUEST",
                "Malformed multipart form data.",
                "request",
            )

        file_values = list(form.getlist("files"))
        uploads = [item for item in file_values if isinstance(item, UploadFile)]
        try:
            unknown_fields = sorted(
                set(form.keys()) - ALLOWED_SERIES_MULTIPART_FIELDS
            )
            if unknown_fields:
                return _error(
                    422,
                    "INVALID_REQUEST",
                    f"Unknown form fields: {', '.join(unknown_fields)}.",
                    "request",
                )
            if not file_values or len(uploads) != len(file_values):
                return _error(
                    422,
                    "INVALID_REQUEST",
                    "At least one multipart image file is required.",
                    "request.files",
                )
            if len(uploads) > max_series_frames:
                return _error(
                    413,
                    "SERIES_TOO_LARGE",
                    f"A series may contain at most {max_series_frames} frames.",
                    "request.files",
                )

            raw_conditions = form.get("active_conditions", "{}")
            try:
                active_conditions = json.loads(str(raw_conditions))
            except json.JSONDecodeError:
                return _error(
                    422,
                    "INVALID_REQUEST",
                    "active_conditions must be a JSON object string.",
                    "request.active_conditions",
                )

            raw_timestamps = form.get("timestamps")
            timestamps_value = None
            if raw_timestamps not in (None, ""):
                try:
                    timestamps_value = json.loads(str(raw_timestamps))
                except json.JSONDecodeError:
                    return _error(
                        422,
                        "INVALID_REQUEST",
                        "timestamps must be a JSON array string.",
                        "request.timestamps",
                    )

            raw_fields: Dict[str, Any] = {
                "planned_work": form.get("planned_work"),
                "active_conditions": active_conditions,
                "camera_id": form.get("camera_id"),
                "timestamps": timestamps_value,
            }
            if form.get("zone_id") not in (None, ""):
                raw_fields["zone_id"] = form.get("zone_id")
            try:
                fields = SeriesAnalysisFields.model_validate(raw_fields)
            except ValidationError as exc:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "INVALID_REQUEST",
                        "issues": _validation_issues(exc),
                    },
                )

            if fields.timestamps is not None:
                if len(fields.timestamps) != len(uploads):
                    return _error(
                        422,
                        "INVALID_REQUEST",
                        "timestamps must contain exactly one item per file.",
                        "request.timestamps",
                    )
                timestamps = list(fields.timestamps)
                timestamps_source = "provided"
            else:
                first_timestamp = datetime.now(timezone.utc).replace(microsecond=0)
                timestamps = [
                    (first_timestamp + timedelta(seconds=index)).isoformat()
                    for index in range(len(uploads))
                ]
                timestamps_source = "generated_upload_order"

            # Validate the shared core context before running costly inference.
            analyze(_core_request(fields, [], timestamp=timestamps[0]))

            instant_results = []
            temporal_observations = []
            frames = []
            inference_timings = []
            matching_timings = []

            for index, (upload, timestamp) in enumerate(zip(uploads, timestamps)):
                try:
                    image_bytes = await upload.read()
                    inference_started = perf_counter()
                    inference = cv_provider.infer(
                        image_bytes,
                        filename=upload.filename,
                        content_type=upload.content_type,
                    )
                    inference_seconds = perf_counter() - inference_started
                except BadImageError as exc:
                    return _error(
                        400,
                        "BAD_IMAGE",
                        str(exc),
                        f"request.files[{index}]",
                    )
                except (ProviderConfigurationError, CVInferenceError) as exc:
                    return _error(
                        503,
                        "CV_PROVIDER_UNAVAILABLE",
                        str(exc),
                        "cv_provider",
                    )

                core_request = _core_request(
                    fields, inference.detections, timestamp=timestamp
                )
                matching_started = perf_counter()
                if inference.native_rfdetr:
                    instant = analyze_rfdetr(core_request)
                else:
                    instant = analyze(core_request)
                matching_seconds = perf_counter() - matching_started

                instant_results.append(instant)
                inference_timings.append(inference_seconds)
                matching_timings.append(matching_seconds)
                temporal_observations.append(
                    _core_request(
                        fields,
                        _canonical_accepted_detections(instant),
                        timestamp=timestamp,
                    )
                )
                frames.append(
                    {
                        "index": index,
                        "filename": upload.filename or f"frame_{index:03d}",
                        "timestamp": timestamp,
                        "instant_status": instant["instant_status"],
                        "observability": instant["observability"],
                        "match": instant["match"],
                        "detections": instant["detections"],
                        "possible_unexpected": instant["possible_unexpected"],
                        "evidence": instant["evidence"],
                        "explanation": instant["explanation"],
                        "timing_seconds": {
                            "inference": round(inference_seconds, 6),
                            "matching": round(matching_seconds, 6),
                        },
                    }
                )

            temporal_started = perf_counter()
            summary = temporal_update({"observations": temporal_observations})
            temporal_seconds = perf_counter() - temporal_started
            summary["frames_count"] = len(frames)
            _add_persistent_frame_indices(summary, instant_results)

            return {
                "series_summary": summary,
                "frames": frames,
                "metadata": {
                    "camera_id": fields.camera_id,
                    "zone_id": fields.zone_id,
                    "active_conditions": fields.active_conditions,
                    "timestamps_source": timestamps_source,
                    "max_series_frames": max_series_frames,
                    "confidence_threshold": MatchingConfig().confidence_threshold,
                    "provider": cv_provider.name,
                    "timings": {
                        "model_loaded_once": cv_provider.name == "rfdetr",
                        "provider_reused": True,
                        "frame_inference_seconds": [
                            round(value, 6) for value in inference_timings
                        ],
                        "frame_matching_seconds": [
                            round(value, 6) for value in matching_timings
                        ],
                        "total_inference_seconds": round(
                            sum(inference_timings), 6
                        ),
                        "total_matching_seconds": round(
                            sum(matching_timings), 6
                        ),
                        "temporal_seconds": round(temporal_seconds, 6),
                        "total_request_seconds": round(
                            perf_counter() - request_started, 6
                        ),
                    },
                },
            }
        finally:
            for upload in uploads:
                await upload.close()

    return application


app = create_app()
