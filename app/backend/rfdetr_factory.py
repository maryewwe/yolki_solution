from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
from typing import Any

from construction_monitoring import MODEL_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "rfdetr" / "final_model.pth"
)
EXPECTED_RFDETR_VERSION = "1.10.1"


def _checkpoint_path() -> Path:
    configured = os.getenv("RFDETR_CHECKPOINT")
    path = Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            "FINAL80 RF-DETR final_model.pth is missing. Download "
            "final80_deployment_bundle.zip from the Kaggle output of "
            "lct-final80-rfdetr-medium640.ipynb, place final_model.pth at "
            f"{DEFAULT_CHECKPOINT}, or set RFDETR_CHECKPOINT to its path."
        )
    return path


def _normalized_class_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, Mapping):
        try:
            raw_names = [
                raw_names[index]
                if index in raw_names
                else raw_names[str(index)]
                for index in range(len(raw_names))
            ]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "RF-DETR checkpoint class_names mapping is not zero-indexed."
            ) from exc
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise RuntimeError(
            "RF-DETR checkpoint does not expose an ordered class_names sequence."
        )
    names = []
    for item in raw_names:
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        names.append(str(item).strip().lower())
    return names


def _validate_taxonomy(model: Any) -> None:
    model_context = getattr(model, "model", None)
    names = _normalized_class_names(
        getattr(model_context, "class_names", None)
    )
    expected = list(MODEL_CLASSES)
    if names != expected:
        raise RuntimeError(
            "RF-DETR checkpoint taxonomy mismatch: "
            f"expected {expected}, got {names}."
        )


def create_model() -> Any:
    """Create the FINAL80 RF-DETR Medium @640 deployment model."""

    checkpoint = _checkpoint_path()
    try:
        installed_version = version("rfdetr")
    except PackageNotFoundError:
        installed_version = None
    if (
        installed_version is not None
        and installed_version != EXPECTED_RFDETR_VERSION
    ):
        raise RuntimeError(
            "RF-DETR version mismatch: "
            f"expected {EXPECTED_RFDETR_VERSION}, got {installed_version}."
        )

    try:
        from rfdetr import RFDETRMedium
    except ImportError as exc:
        raise RuntimeError(
            "rfdetr 1.10.1 is not installed. Install requirements-rfdetr.txt."
        ) from exc

    kwargs = {
        "pretrain_weights": str(checkpoint),
        "resolution": 640,
        "num_classes": 13,
    }
    configured_device = os.getenv("RFDETR_DEVICE")
    if configured_device:
        kwargs["device"] = configured_device.strip()

    model = RFDETRMedium(**kwargs)
    _validate_taxonomy(model)
    return model
