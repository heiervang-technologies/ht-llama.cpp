"""Model-free regression checks for validation process and telemetry failures."""
import errno
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import validate


class ValidationInfrastructureTests(unittest.TestCase):
    def test_firmware_read_failure_is_recorded(self):
        target = Path("/sys/firmware/acpi/platform_profile")
        original_exists, original_read = Path.exists, Path.read_text

        def exists(path):
            return path == target or original_exists(path)

        def read(path, *args, **kwargs):
            if path == target:
                raise OSError(errno.EIO, "firmware read failed")
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "exists", exists), patch.object(Path, "read_text", read):
            result = validate.snapshot()
        self.assertIsNone(result[str(target)])
        self.assertIn(str(target), result["read_errors"])

    def test_failed_child_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), timeout_seconds=10)
            with self.assertRaises(RuntimeError):
                validate.run(args, [sys.executable, "-c", "raise SystemExit(3)"], "failure")
            result = json.loads((args.output / "failure.run.json").read_text())
            self.assertEqual(result["returncode"], 3)
            self.assertIsNone(result["runner_error"])

    def test_timeout_stops_child_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), timeout_seconds=0.01)
            with self.assertRaises(TimeoutError):
                validate.run(args, [sys.executable, "-c", "import time; time.sleep(60)"], "timeout")
            result = json.loads((args.output / "timeout.run.json").read_text())
            self.assertIsNotNone(result["returncode"])
            self.assertNotEqual(result["returncode"], 0)
            self.assertIn("TimeoutError", result["runner_error"])


if __name__ == "__main__":
    unittest.main()
