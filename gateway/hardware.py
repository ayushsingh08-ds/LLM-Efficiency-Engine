"""Lightweight hardware / resource profiling for the gateway.

Measures what we can actually observe on the host:
- CPU utilisation and core count (psutil)
- RAM usage (psutil)
- GPU utilisation, memory, and power draw IF an NVIDIA GPU + pynvml are present
  (fully optional and import-guarded — the system runs fine without it)

Any energy/power number that is NOT read directly from hardware is labelled as an
ESTIMATE. Specifically, on hosts without NVIDIA power telemetry we do not know
real wattage, so we expose only utilisation plus a clearly-marked rough estimate.
We never hard-code a specific GPU/CPU model's power draw and present it as fact.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Optional

import psutil

# Optional GPU telemetry. Import guarded so absence of NVIDIA/pynvml is a no-op.
try:  # pragma: no cover - depends on host
    import pynvml  # type: ignore

    _PYNVML_IMPORTED = True
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore
    _PYNVML_IMPORTED = False

_nvml_ready = False


def _ensure_nvml() -> bool:
    """Initialise NVML once. Returns False if unavailable (no GPU / no driver)."""
    global _nvml_ready
    if not _PYNVML_IMPORTED:
        return False
    if _nvml_ready:
        return True
    try:  # pragma: no cover - depends on host
        pynvml.nvmlInit()
        _nvml_ready = True
        return True
    except Exception:
        return False


@dataclass
class GPUSample:
    index: int
    name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    power_watts: Optional[float]  # None if the GPU doesn't report power
    power_is_measured: bool


@dataclass
class HardwareSample:
    cpu_percent: float
    cpu_count_logical: int
    cpu_count_physical: Optional[int]
    ram_percent: float
    ram_used_mb: float
    ram_total_mb: float
    gpus: list[dict]
    gpu_available: bool
    # Energy fields are ESTIMATES unless energy_is_measured is True (real GPU power).
    estimated_active_power_watts: float
    energy_is_measured: bool
    notes: str


# Rough fallback estimate (watts) used ONLY when no GPU power telemetry exists.
# This is an illustrative placeholder, not a measurement of this machine.
ESTIMATED_HOST_ACTIVE_WATTS = float(os.getenv("ESTIMATED_HOST_ACTIVE_WATTS", "65"))


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var. Read at call time so config/tests take effect
    without a re-import."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def profiling_enabled() -> bool:
    """Master switch for hardware profiling (env HARDWARE_PROFILING_ENABLED,
    default on). When off, the gateway skips sampling entirely so there is zero
    profiling overhead and no dependency on host telemetry."""
    return _env_bool("HARDWARE_PROFILING_ENABLED", True)


def gpu_enabled() -> bool:
    """Whether to attempt GPU sampling (env HARDWARE_GPU_ENABLED, default on).
    Independent of profiling_enabled so GPU polling can be disabled on its own
    even where pynvml/NVIDIA is present (overhead / noisy-neighbour control)."""
    return _env_bool("HARDWARE_GPU_ENABLED", True)


def _sample_gpus() -> list[GPUSample]:
    if not gpu_enabled() or not _ensure_nvml():
        return []

    samples: list[GPUSample] = []
    try:  # pragma: no cover - depends on host
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W (measured)
                power_measured = True
            except Exception:
                power = None
                power_measured = False
            samples.append(GPUSample(
                index=i,
                name=name,
                utilization_percent=float(util.gpu),
                memory_used_mb=round(mem.used / (1024 ** 2), 1),
                memory_total_mb=round(mem.total / (1024 ** 2), 1),
                power_watts=round(power, 2) if power is not None else None,
                power_is_measured=power_measured,
            ))
    except Exception:
        return samples
    return samples


def sample_hardware(cpu_interval: float = 0.0) -> HardwareSample:
    """Take a point-in-time hardware sample.

    cpu_interval: passed to psutil.cpu_percent. 0.0 = non-blocking (compares to
    the previous call); a small value like 0.1 blocks briefly for an accurate read.
    """
    vm = psutil.virtual_memory()
    gpus = _sample_gpus()

    measured_gpu_power = sum(
        g.power_watts for g in gpus if g.power_watts is not None
    )
    energy_is_measured = any(g.power_is_measured for g in gpus)

    if energy_is_measured:
        active_power = round(measured_gpu_power, 2)
        notes = "GPU power read from NVML (measured)."
    else:
        active_power = ESTIMATED_HOST_ACTIVE_WATTS
        notes = (
            "No GPU power telemetry available; estimated_active_power_watts is a "
            "coarse placeholder ESTIMATE, not a measurement of this host."
        )

    return HardwareSample(
        cpu_percent=psutil.cpu_percent(interval=cpu_interval),
        cpu_count_logical=psutil.cpu_count(logical=True) or 0,
        cpu_count_physical=psutil.cpu_count(logical=False),
        ram_percent=vm.percent,
        ram_used_mb=round(vm.used / (1024 ** 2), 1),
        ram_total_mb=round(vm.total / (1024 ** 2), 1),
        gpus=[asdict(g) for g in gpus],
        gpu_available=bool(gpus),
        estimated_active_power_watts=active_power,
        energy_is_measured=energy_is_measured,
        notes=notes,
    )


def sample_as_dict(cpu_interval: float = 0.0) -> dict:
    return asdict(sample_hardware(cpu_interval=cpu_interval))
