from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from gateway import hardware
from gateway.hardware import HardwareSample, sample_hardware


class TestHardwareProfiler(unittest.TestCase):
    def test_sample_returns_cpu_and_ram(self) -> None:
        sample = sample_hardware(cpu_interval=0.0)
        self.assertIsInstance(sample, HardwareSample)
        self.assertGreaterEqual(sample.cpu_percent, 0.0)
        self.assertGreater(sample.ram_total_mb, 0.0)
        self.assertGreaterEqual(sample.ram_percent, 0.0)
        self.assertGreaterEqual(sample.cpu_count_logical, 1)

    def test_runs_without_gpu(self) -> None:
        # Simulate a host with no GPU telemetry: _sample_gpus returns [].
        with patch.object(hardware, "_sample_gpus", return_value=[]):
            sample = sample_hardware(cpu_interval=0.0)
        self.assertFalse(sample.gpu_available)
        self.assertEqual(sample.gpus, [])
        # No measured power -> energy must be flagged as an estimate.
        self.assertFalse(sample.energy_is_measured)
        self.assertIn("ESTIMATE", sample.notes)
        self.assertGreater(sample.estimated_active_power_watts, 0.0)

    def test_measured_gpu_power_flags_energy_as_measured(self) -> None:
        from gateway.hardware import GPUSample

        fake_gpu = GPUSample(
            index=0,
            name="Fake GPU",
            utilization_percent=42.0,
            memory_used_mb=1024.0,
            memory_total_mb=8192.0,
            power_watts=120.5,
            power_is_measured=True,
        )
        with patch.object(hardware, "_sample_gpus", return_value=[fake_gpu]):
            sample = sample_hardware(cpu_interval=0.0)
        self.assertTrue(sample.gpu_available)
        self.assertTrue(sample.energy_is_measured)
        self.assertEqual(sample.estimated_active_power_watts, 120.5)
        self.assertIn("measured", sample.notes.lower())

    def test_gpu_import_is_optional(self) -> None:
        # The module must import and sample even if pynvml is absent.
        with patch.object(hardware, "_PYNVML_IMPORTED", False):
            self.assertFalse(hardware._ensure_nvml())

    def test_profiling_enabled_default_and_env(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("HARDWARE_PROFILING_ENABLED", None)
            self.assertTrue(hardware.profiling_enabled())  # default on
        with patch.dict("os.environ", {"HARDWARE_PROFILING_ENABLED": "false"}):
            self.assertFalse(hardware.profiling_enabled())
        with patch.dict("os.environ", {"HARDWARE_PROFILING_ENABLED": "1"}):
            self.assertTrue(hardware.profiling_enabled())

    def test_gpu_sampling_can_be_disabled_independently(self) -> None:
        # Even if NVML would be ready, HARDWARE_GPU_ENABLED=false skips GPU polling.
        with patch.dict("os.environ", {"HARDWARE_GPU_ENABLED": "false"}), \
             patch.object(hardware, "_ensure_nvml", return_value=True):
            self.assertEqual(hardware._sample_gpus(), [])

    def test_sample_still_works_with_gpu_disabled(self) -> None:
        with patch.dict("os.environ", {"HARDWARE_GPU_ENABLED": "false"}):
            sample = sample_hardware(cpu_interval=0.0)
        self.assertFalse(sample.gpu_available)
        self.assertFalse(sample.energy_is_measured)  # falls back to estimate


if __name__ == "__main__":
    unittest.main()
