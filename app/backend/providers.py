from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from PIL import Image, UnidentifiedImageError

from construction_monitoring import MatchingConfig


MAX_IMAGE_BYTES = 10 * 1024 * 1024


class BadImageError(ValueError):
    pass


class ProviderConfigurationError(RuntimeError):
    pass


class CVInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceOutput:
    detections: Any
    native_rfdetr: bool = False


class CVProvider(Protocol):
    name: str
    ready: bool

    def infer(
        self,
        image_bytes: bytes,
        *,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> InferenceOutput:
        ...


def decode_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise BadImageError("The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise BadImageError("The uploaded image exceeds the 10 MB demo limit.")
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise BadImageError("The uploaded file is not a readable image.") from exc


class FixtureProvider:
    """Local/demo provider returning configured synthetic detections."""

    name = "fixture"
    ready = True

    def __init__(
        self, detections: Optional[Sequence[Mapping[str, Any]]] = None
    ) -> None:
        self._detections = [dict(item) for item in (detections or [])]

    def infer(
        self,
        image_bytes: bytes,
        *,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> InferenceOutput:
        decode_image(image_bytes)
        return InferenceOutput(
            detections=[dict(item) for item in self._detections]
        )


class RFDETRProvider:
    """Thin wrapper around the CV team's initialized RF-DETR model object."""

    name = "rfdetr"
    ready = True

    def __init__(self, model: Any) -> None:
        if model is None or not callable(getattr(model, "predict", None)):
            raise ProviderConfigurationError(
                "RFDETRProvider requires an initialized model with predict()."
            )
        self._model = model
        self._threshold = MatchingConfig().confidence_threshold

    def infer(
        self,
        image_bytes: bytes,
        *,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> InferenceOutput:
        image = decode_image(image_bytes)
        try:
            detections = self._model.predict(
                image,
                threshold=self._threshold,
            )
        except Exception as exc:
            raise CVInferenceError("RF-DETR inference failed.") from exc
        return InferenceOutput(detections=detections, native_rfdetr=True)


def _load_fixture_detections(path_value: Optional[str]) -> List[Dict[str, Any]]:
    if not path_value:
        return []
    path = Path(path_value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderConfigurationError(
            f"Cannot load CV fixture detections from {path}."
        ) from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ProviderConfigurationError(
            "CV fixture file must contain a JSON array of detection objects."
        )
    return [dict(item) for item in payload]


def _load_model(factory_path: str) -> Any:
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ProviderConfigurationError(
            "RFDETR_MODEL_FACTORY must use the module:function format."
        )
    try:
        factory = getattr(import_module(module_name), attribute_name)
        return factory()
    except (ImportError, AttributeError, TypeError) as exc:
        raise ProviderConfigurationError(
            f"Cannot initialize RF-DETR from {factory_path}."
        ) from exc


def provider_from_environment() -> CVProvider:
    provider_name = os.getenv("CV_PROVIDER", "fixture").strip().lower()
    if provider_name == "fixture":
        detections = _load_fixture_detections(os.getenv("CV_FIXTURE_PATH"))
        return FixtureProvider(detections)
    if provider_name == "rfdetr":
        factory_path = os.getenv("RFDETR_MODEL_FACTORY")
        if not factory_path:
            raise ProviderConfigurationError(
                "CV_PROVIDER=rfdetr requires RFDETR_MODEL_FACTORY=module:function."
            )
        return RFDETRProvider(_load_model(factory_path))
    raise ProviderConfigurationError(
        "CV_PROVIDER must be either 'fixture' or 'rfdetr'."
    )
