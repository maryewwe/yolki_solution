from __future__ import annotations

from io import BytesIO
import json
import os
from typing import Any, Mapping, Sequence
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.providers import InferenceOutput, RFDETRProvider, decode_image


def png_frame(color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color=color).save(output, format="PNG")
    return output.getvalue()


def required_detections(
    *,
    excavator_bbox: Sequence[float] = (10, 20, 110, 120),
    dump_confidence: float = 0.9,
) -> list[dict[str, Any]]:
    return [
        {
            "class": "excavator",
            "confidence": 0.95,
            "bbox": list(excavator_bbox),
        },
        {
            "class": "dump_truck",
            "confidence": dump_confidence,
            "bbox": [130, 20, 230, 120],
        },
    ]


class SequenceProvider:
    name = "sequence-fixture"
    ready = True

    def __init__(self, frames: Sequence[Sequence[Mapping[str, Any]]]) -> None:
        self.frames = [[dict(item) for item in frame] for frame in frames]
        self.calls = 0

    def infer(
        self,
        image_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> InferenceOutput:
        decode_image(image_bytes)
        index = self.calls
        self.calls += 1
        return InferenceOutput(
            detections=[dict(item) for item in self.frames[index]]
        )


class BackendSeriesApiTests(unittest.TestCase):
    @staticmethod
    def post_series(
        client: TestClient,
        frame_count: int,
        *,
        timestamps: list[str] | None = None,
        camera_id: str = "series-camera",
        zone_id: str | None = None,
    ):
        data = {
            "planned_work": "Устройство котлована",
            "active_conditions": "{}",
            "camera_id": camera_id,
        }
        if zone_id is not None:
            data["zone_id"] = zone_id
        if timestamps is not None:
            data["timestamps"] = json.dumps(timestamps)
        files = [
            (
                "files",
                (f"frame_{index:03d}.png", png_frame(), "image/png"),
            )
            for index in range(frame_count)
        ]
        return client.post("/analyze-series", data=data, files=files)

    def test_01_all_frames_satisfied_yield_persistent_ok(self) -> None:
        provider = SequenceProvider([required_detections() for _ in range(3)])
        response = self.post_series(TestClient(create_app(provider)), 3)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["series_summary"]["persistent_status"], "OK")
        self.assertEqual(
            [frame["instant_status"] for frame in payload["frames"]],
            ["OK", "OK", "OK"],
        )

    def test_02_one_isolated_miss_is_not_persistent_warning(self) -> None:
        frames = [required_detections() for _ in range(4)]
        frames[1] = [required_detections()[0]]
        response = self.post_series(
            TestClient(create_app(SequenceProvider(frames))), 4
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["frames"][1]["instant_status"], "WARNING")
        self.assertEqual(payload["series_summary"]["persistent_status"], "OK")
        self.assertEqual(
            payload["series_summary"]["temporal"]["persistent_missing"], []
        )

    def test_03_sustained_miss_yields_persistent_warning_and_evidence(self) -> None:
        frames = [[required_detections()[0]] for _ in range(4)]
        response = self.post_series(
            TestClient(create_app(SequenceProvider(frames))), 4
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        summary = payload["series_summary"]
        self.assertEqual(summary["persistent_status"], "WARNING")
        persistent = summary["temporal"]["persistent_missing"][0]
        self.assertEqual(persistent["requirement"]["id"], "required:dump_truck")
        self.assertEqual(persistent["frame_indices"], [0, 1, 2, 3])

    def test_04_low_confidence_detections_are_ignored(self) -> None:
        frames = [
            required_detections(dump_confidence=0.249999) for _ in range(3)
        ]
        response = self.post_series(
            TestClient(create_app(SequenceProvider(frames))), 3
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["series_summary"]["persistent_status"], "WARNING")
        for frame in payload["frames"]:
            rejected = frame["detections"]["rejected_low_confidence"]
            self.assertEqual([item["class"] for item in rejected], ["dump_truck"])

    def test_05_each_frame_keeps_its_own_bbox(self) -> None:
        boxes = [
            [10, 20, 110, 120],
            [11, 21, 111, 121],
            [12, 22, 112, 122],
        ]
        frames = [required_detections(excavator_bbox=box) for box in boxes]
        response = self.post_series(
            TestClient(create_app(SequenceProvider(frames))), 3
        )

        self.assertEqual(response.status_code, 200)
        actual = [
            frame["detections"]["accepted"][0]["bbox"]
            for frame in response.json()["frames"]
        ]
        self.assertEqual(actual, [[float(value) for value in box] for box in boxes])

    def test_06_upload_order_and_provided_timestamps_are_preserved(self) -> None:
        timestamps = [
            "2026-09-29T12:00:00+03:00",
            "2026-09-29T12:00:05+03:00",
            "2026-09-29T12:00:10+03:00",
        ]
        provider = SequenceProvider([required_detections() for _ in range(3)])
        response = self.post_series(
            TestClient(create_app(provider)), 3, timestamps=timestamps
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual([frame["index"] for frame in payload["frames"]], [0, 1, 2])
        self.assertEqual(
            [frame["filename"] for frame in payload["frames"]],
            ["frame_000.png", "frame_001.png", "frame_002.png"],
        )
        self.assertEqual(
            [frame["timestamp"] for frame in payload["frames"]], timestamps
        )
        self.assertEqual(payload["metadata"]["timestamps_source"], "provided")

    def test_07_every_image_passes_through_provider(self) -> None:
        provider = SequenceProvider([required_detections() for _ in range(5)])
        response = self.post_series(TestClient(create_app(provider)), 5)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(provider.calls, 5)
        self.assertEqual(response.json()["series_summary"]["frames_count"], 5)

    def test_08_one_rfdetr_model_instance_is_reused_for_all_frames(self) -> None:
        class FakeDetections:
            data = {"class_name": ["excavator", "dump_truck"]}
            confidence = [0.95, 0.9]
            xyxy = [[10, 20, 110, 120], [130, 20, 230, 120]]

        class FakeRFDETRModel:
            def __init__(self) -> None:
                self.predict_calls = 0
                self.thresholds: list[float] = []

            def predict(self, image, *, threshold):
                self.predict_calls += 1
                self.thresholds.append(threshold)
                return FakeDetections()

        model = FakeRFDETRModel()
        provider = RFDETRProvider(model)
        application = create_app(provider)
        response = self.post_series(TestClient(application), 4)

        self.assertEqual(response.status_code, 200)
        self.assertIs(application.state.cv_provider, provider)
        self.assertEqual(model.predict_calls, 4)
        self.assertEqual(model.thresholds, [0.25, 0.25, 0.25, 0.25])
        self.assertTrue(response.json()["metadata"]["timings"]["model_loaded_once"])

    def test_09_invalid_image_inside_series_returns_indexed_error(self) -> None:
        provider = SequenceProvider([required_detections() for _ in range(3)])
        client = TestClient(create_app(provider))
        files = [
            ("files", ("frame_000.png", png_frame(), "image/png")),
            ("files", ("broken.jpg", b"not-an-image", "image/jpeg")),
            ("files", ("frame_002.png", png_frame(), "image/png")),
        ]
        response = client.post(
            "/analyze-series",
            data={
                "planned_work": "Устройство котлована",
                "camera_id": "series-camera",
            },
            files=files,
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "BAD_IMAGE")
        self.assertEqual(response.json()["issues"][0]["path"], "request.files[1]")

    def test_10_configured_frame_limit_is_enforced_before_inference(self) -> None:
        provider = SequenceProvider([required_detections() for _ in range(3)])
        with patch.dict(os.environ, {"MAX_SERIES_FRAMES": "2"}):
            client = TestClient(create_app(provider))
        response = self.post_series(client, 3)

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"], "SERIES_TOO_LARGE")
        self.assertEqual(provider.calls, 0)

    def test_11_empty_series_is_validation_error(self) -> None:
        provider = SequenceProvider([])
        response = TestClient(create_app(provider)).post(
            "/analyze-series",
            data={
                "planned_work": "Устройство котлована",
                "camera_id": "series-camera",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "INVALID_REQUEST")
        self.assertEqual(provider.calls, 0)

    def test_12_all_frames_share_one_camera_zone_and_context(self) -> None:
        provider = SequenceProvider([required_detections() for _ in range(3)])
        response = self.post_series(
            TestClient(create_app(provider)),
            3,
            camera_id="cam-a",
            zone_id="pit-a",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["metadata"]["camera_id"], "cam-a")
        self.assertEqual(payload["metadata"]["zone_id"], "pit-a")
        for frame in payload["frames"]:
            self.assertTrue(
                all(
                    item["camera_id"] == "cam-a" and item["zone_id"] == "pit-a"
                    for item in frame["detections"]["accepted"]
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
