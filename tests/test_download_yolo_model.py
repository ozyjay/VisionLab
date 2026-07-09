from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_yolo_model.py"
SPEC = importlib.util.spec_from_file_location("download_yolo_model", SCRIPT_PATH)
assert SPEC is not None
download_yolo_model = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(download_yolo_model)


class DownloadYoloModelTests(unittest.TestCase):
    def test_size_aliases_resolve_to_yolo11_filenames(self) -> None:
        self.assertEqual(download_yolo_model.resolve_model_name("small"), "yolo11s.pt")
        self.assertEqual(download_yolo_model.resolve_model_name("medium"), "yolo11m.pt")
        self.assertEqual(download_yolo_model.resolve_model_name("large"), "yolo11l.pt")
        self.assertEqual(download_yolo_model.resolve_model_name("x"), "yolo11x.pt")

    def test_explicit_yolo_filename_is_allowed(self) -> None:
        self.assertEqual(
            download_yolo_model.resolve_model_name("yolo11m.pt"),
            "yolo11m.pt",
        )

    def test_yoloe_aliases_resolve_to_yoloe_26_filenames(self) -> None:
        self.assertEqual(
            download_yolo_model.resolve_model_name("yoloe-s"),
            "yoloe-26s-seg.pt",
        )
        self.assertEqual(
            download_yolo_model.resolve_model_name("yoloe-medium"),
            "yoloe-26m-seg.pt",
        )
        self.assertEqual(
            download_yolo_model.resolve_model_name("yoloe-prompt-free"),
            "yoloe-26s-seg-pf.pt",
        )

    def test_paths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            download_yolo_model.resolve_model_name("models/yolo11m.pt")

    def test_non_yolo_filename_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            download_yolo_model.resolve_model_name("not-a-yolo-model.pt")


if __name__ == "__main__":
    unittest.main()
