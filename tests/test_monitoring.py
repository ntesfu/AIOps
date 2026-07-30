from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aiops.training.monitoring import TrainingMonitor


class _BrokenWriter:
    def add_scalar(self, *args, **kwargs):
        raise OSError("writer stopped")

    def flush(self):
        raise OSError("flush stopped")

    def close(self):
        pass


class MonitoringTest(unittest.TestCase):
    def test_tensorboard_failure_is_nonfatal_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor = TrainingMonitor(directory, enabled=False)
            monitor.enabled = True
            monitor.writer = _BrokenWriter()
            monitor.log_scalars("validation", {"loss": 1.0}, 3)
            self.assertIsNone(monitor.writer)
            self.assertIn("writer stopped", (Path(directory) / "monitoring_errors.jsonl").read_text())

    def test_tensorboard_flush_failure_is_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor = TrainingMonitor(directory, enabled=False)
            monitor.writer = _BrokenWriter()
            monitor.close()
            self.assertIn("flush stopped", (Path(directory) / "monitoring_errors.jsonl").read_text())
