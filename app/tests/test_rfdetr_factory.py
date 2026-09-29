from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from construction_monitoring import MODEL_CLASSES


class RFDETRFactoryTests(unittest.TestCase):
    def test_01_missing_target_checkpoint_fails_before_model_import(self) -> None:
        from backend.rfdetr_factory import create_model

        missing = Path(tempfile.gettempdir()) / "missing-rfdetr-checkpoint.pth"
        with patch.dict(
            "os.environ",
            {"RFDETR_CHECKPOINT": str(missing)},
            clear=False,
        ):
            with self.assertRaises(FileNotFoundError) as captured:
                create_model()

        self.assertIn("final_model.pth", str(captured.exception))
        self.assertIn("RFDETR_CHECKPOINT", str(captured.exception))

    def test_02_factory_matches_kaggle_medium640_constructor(self) -> None:
        from backend.rfdetr_factory import create_model

        calls = []

        class FakeRFDETRMedium:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.model = SimpleNamespace(class_names=list(MODEL_CLASSES))

        fake_module = ModuleType("rfdetr")
        fake_module.RFDETRMedium = FakeRFDETRMedium

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final_model.pth"
            checkpoint.write_bytes(b"synthetic-test-checkpoint")
            with (
                patch.dict(sys.modules, {"rfdetr": fake_module}),
                patch.dict(
                    "os.environ",
                    {"RFDETR_CHECKPOINT": str(checkpoint)},
                    clear=False,
                ),
            ):
                model = create_model()

        self.assertIsInstance(model, FakeRFDETRMedium)
        self.assertEqual(
            calls,
            [
                {
                    "pretrain_weights": str(checkpoint.resolve()),
                    "resolution": 640,
                    "num_classes": 13,
                }
            ],
        )

    def test_03_optional_device_is_forwarded_without_changing_architecture(self) -> None:
        from backend.rfdetr_factory import create_model

        calls = []

        class FakeRFDETRMedium:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.model = SimpleNamespace(class_names=list(MODEL_CLASSES))

        fake_module = ModuleType("rfdetr")
        fake_module.RFDETRMedium = FakeRFDETRMedium

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final_model.pth"
            checkpoint.write_bytes(b"synthetic-test-checkpoint")
            with (
                patch.dict(sys.modules, {"rfdetr": fake_module}),
                patch.dict(
                    "os.environ",
                    {
                        "RFDETR_CHECKPOINT": str(checkpoint),
                        "RFDETR_DEVICE": "cpu",
                    },
                    clear=False,
                ),
            ):
                create_model()

        self.assertEqual(calls[0]["device"], "cpu")
        self.assertEqual(calls[0]["resolution"], 640)
        self.assertEqual(calls[0]["num_classes"], 13)

    def test_04_checkpoint_taxonomy_must_match_canonical_classes(self) -> None:
        from backend.rfdetr_factory import create_model

        class FakeRFDETRMedium:
            def __init__(self, **kwargs):
                self.model = SimpleNamespace(class_names=["wrong_class"])

        fake_module = ModuleType("rfdetr")
        fake_module.RFDETRMedium = FakeRFDETRMedium

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final_model.pth"
            checkpoint.write_bytes(b"synthetic-test-checkpoint")
            with (
                patch.dict(sys.modules, {"rfdetr": fake_module}),
                patch.dict(
                    "os.environ",
                    {"RFDETR_CHECKPOINT": str(checkpoint)},
                    clear=False,
                ),
            ):
                with self.assertRaises(RuntimeError) as captured:
                    create_model()

        self.assertIn("taxonomy", str(captured.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
