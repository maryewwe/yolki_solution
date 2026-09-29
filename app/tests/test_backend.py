from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.providers import FixtureProvider, RFDETRProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "integration" / "fixtures"


def png_1x1() -> bytes:
    output = BytesIO()
    Image.new("RGB", (1, 1), color="white").save(output, format="PNG")
    return output.getvalue()


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image_detections = [
            {
                "class": "excavator",
                "confidence": 0.93,
                "bbox": [100, 80, 420, 360],
            },
            {
                "class": "dump_truck",
                "confidence": 0.88,
                "bbox": [440, 120, 760, 390],
            },
        ]
        app = create_app(provider=FixtureProvider(cls.image_detections))
        cls.client = TestClient(app)

    @staticmethod
    def fixture_request(name: str) -> dict:
        with (FIXTURE_DIR / name).open(encoding="utf-8") as stream:
            fixture = json.load(stream)
        request = fixture["request"]
        return {
            **request["context"],
            "detections": request["detections"],
        }

    def test_01_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["cv_provider"], "fixture")
        self.assertEqual(response.json()["confidence_threshold"], 0.25)

    def test_02_works_returns_dropdown_options(self) -> None:
        response = self.client.get("/works")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], len(payload["works"]))
        self.assertEqual(payload["count"], 359)
        work = next(
            item
            for item in payload["works"]
            if item["planned_work"] == "Устройство котлована"
        )
        self.assertEqual(work["work_group"], "Земляные работы")
        self.assertIn("active_conditions", work)

    def test_03_analyze_ok(self) -> None:
        response = self.client.post(
            "/analyze", json=self.fixture_request("a_ok.json")
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["match"]["score"], 1.0)
        self.assertEqual(len(payload["detections"]["accepted"]), 2)

    def test_04_analyze_warning(self) -> None:
        response = self.client.post(
            "/analyze", json=self.fixture_request("b_warning.json")
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "WARNING")
        self.assertEqual(payload["match"]["missing_count"], 1)

    def test_05_analyze_uncertain(self) -> None:
        response = self.client.post(
            "/analyze", json=self.fixture_request("c_uncertain.json")
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "UNCERTAIN")
        self.assertTrue(payload["uncertainty_present"])

    def test_06_empty_detections_are_valid(self) -> None:
        request = self.fixture_request("a_ok.json")
        request["detections"] = []

        response = self.client.post("/analyze", json=request)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "WARNING")
        self.assertEqual(payload["detections"]["accepted"], [])

    def test_07_invalid_work_returns_422(self) -> None:
        request = self.fixture_request("a_ok.json")
        request["planned_work"] = "Несуществующая работа"

        response = self.client.post("/analyze", json=request)

        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertEqual(payload["error"], "INVALID_REQUEST")
        self.assertIn(
            "unknown_planned_work",
            {issue["code"] for issue in payload["issues"]},
        )

    def test_08_unsupported_detection_class_returns_422(self) -> None:
        request = self.fixture_request("a_ok.json")
        request["detections"][0]["class"] = "forklift_standard"

        response = self.client.post("/analyze", json=request)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "INVALID_REQUEST")

    def test_09_image_mode_uses_provider_and_core_contract(self) -> None:
        response = self.client.post(
            "/analyze",
            data={
                "planned_work": "Устройство котлована",
                "active_conditions": "{}",
                "camera_id": "demo-camera",
                "timestamp": "2026-09-28T12:00:00+03:00",
            },
            files={"image": ("frame.png", png_1x1(), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(len(payload["detections"]["accepted"]), 2)

    def test_10_bad_image_returns_400(self) -> None:
        response = self.client.post(
            "/analyze",
            data={"planned_work": "Устройство котлована"},
            files={"image": ("frame.jpg", b"not-an-image", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "BAD_IMAGE")

    def test_11_rfdetr_provider_uses_025_and_existing_native_adapter(self) -> None:
        class FakeRFDetections:
            data = {"class_name": ["excavator", "dump_truck"]}
            confidence = [0.93, 0.88]
            xyxy = [[100, 80, 420, 360], [440, 120, 760, 390]]

        class FakeRFDETRModel:
            threshold = None

            def predict(self, image, *, threshold):
                self.threshold = threshold
                return FakeRFDetections()

        model = FakeRFDETRModel()
        client = TestClient(create_app(provider=RFDETRProvider(model)))

        response = client.post(
            "/analyze",
            data={"planned_work": "Устройство котлована"},
            files={"image": ("frame.png", png_1x1(), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "OK")
        self.assertEqual(model.threshold, 0.25)

    def test_12_openapi_documents_json_and_multipart_modes(self) -> None:
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        request_content = response.json()["paths"]["/analyze"]["post"][
            "requestBody"
        ]["content"]
        self.assertIn("application/json", request_content)
        self.assertIn("multipart/form-data", request_content)
        json_schema = request_content["application/json"]["schema"]
        self.assertNotIn("$defs", json_schema)
        self.assertIn(
            "enum",
            json_schema["properties"]["detections"]["items"]["properties"][
                "class"
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
