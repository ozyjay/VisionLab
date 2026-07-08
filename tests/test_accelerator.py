from __future__ import annotations

import unittest
from unittest.mock import patch

from src.accelerator import get_torch_accelerator_status


class AcceleratorTests(unittest.TestCase):
    def test_cpu_request_resolves_to_cpu(self) -> None:
        status = get_torch_accelerator_status("cpu")

        self.assertEqual(status.requested_device, "cpu")
        self.assertEqual(status.resolved_device, "cpu")
        self.assertEqual(status.backend, "cpu")

    def test_auto_without_torch_falls_back_to_cpu(self) -> None:
        with patch("src.accelerator._torch_module_available", return_value=False):
            status = get_torch_accelerator_status("auto")

        self.assertEqual(status.resolved_device, "cpu")
        self.assertFalse(status.torch_available)
        self.assertIn("falling back to CPU", status.note)

    def test_explicit_device_is_preserved_when_torch_missing(self) -> None:
        with patch("src.accelerator._torch_module_available", return_value=False):
            status = get_torch_accelerator_status("cuda:0")

        self.assertEqual(status.resolved_device, "cuda:0")
        self.assertFalse(status.torch_available)


if __name__ == "__main__":
    unittest.main()
