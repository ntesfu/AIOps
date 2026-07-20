from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


class TrainingMonitor:
    """Reusable JSONL and TensorBoard logger with optional NVIDIA telemetry."""

    def __init__(self, output_dir: str | Path, enabled: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.system_path = self.output_dir / "system.jsonl"
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except (ImportError, ModuleNotFoundError):
                print(
                    "TensorBoard is not installed; JSONL logs remain enabled. "
                    "Install the 'monitor' extra for the live dashboard.",
                    flush=True,
                )

    def log_scalars(self, group: str, values: Mapping[str, Any], step: int) -> None:
        if not self.enabled:
            return
        if self.writer is not None:
            for name, value in values.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{group}/{name}", float(value), step)

    def log_system(self, step: int, device: Any) -> dict[str, float]:
        if not self.enabled:
            return {}
        payload: dict[str, float] = {"wall_time": time.time(), "step": float(step)}
        try:
            import torch

            if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
                index = device.index if device.index is not None else torch.cuda.current_device()
                payload.update(
                    {
                        "gpu_memory_allocated_mb": torch.cuda.memory_allocated(index) / 2**20,
                        "gpu_memory_reserved_mb": torch.cuda.memory_reserved(index) / 2**20,
                        "gpu_memory_peak_mb": torch.cuda.max_memory_allocated(index) / 2**20,
                    }
                )
                payload.update(_nvidia_smi_metrics(index))
        except (ImportError, RuntimeError):
            pass
        with self.system_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self.log_scalars("system", payload, step)
        return payload

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def _nvidia_smi_metrics(index: int) -> dict[str, float]:
    query = (
        "utilization.gpu,memory.used,memory.total,temperature.gpu,"
        "power.draw,power.limit"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [float(value.strip()) for value in result.stdout.strip().split(",")]
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return {}
    if len(values) != 6:
        return {}
    return dict(
        zip(
            (
                "gpu_utilization_percent",
                "gpu_memory_used_mb",
                "gpu_memory_total_mb",
                "gpu_temperature_c",
                "gpu_power_w",
                "gpu_power_limit_w",
            ),
            values,
        )
    )
