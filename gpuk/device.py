"""Query the machine: what GPU is this, and what are its hard limits?

Two independent sources of truth, because they answer different questions:

* **MLX** tells us what the *runtime* will let us allocate. Available always.
* **Metal** (a tiny Swift probe) tells us the *hardware* limits: how many threads one
  threadgroup may hold, how much threadgroup memory exists, which GPU family this is.
  Requires the Swift toolchain; run it once with ``python -m gpuk.device --metal``.

Nothing in this file is hard-coded to a specific Mac.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
SWIFT_PROBE = REPO_ROOT / "tools" / "device_probe.swift"


@dataclass(frozen=True)
class GpuInfo:
    """What MLX and the OS report about the default GPU."""

    name: str
    architecture: str
    memory_size: int
    max_buffer_length: int
    recommended_working_set: int
    os_version: str
    chip: str
    gpu_cores: int | None

    @property
    def memory_gib(self) -> float:
        return self.memory_size / 2**30

    @property
    def max_buffer_gib(self) -> float:
        return self.max_buffer_length / 2**30

    @property
    def working_set_gib(self) -> float:
        return self.recommended_working_set / 2**30


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_core_count() -> int | None:
    """Parse `system_profiler SPDisplaysDataType` for the GPU core count."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"Total Number of Cores:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def gpu_info() -> GpuInfo:
    """Collect GPU facts from MLX plus a couple of OS queries."""
    info = dict(mx.device_info())
    return GpuInfo(
        name=info.get("device_name", "unknown"),
        architecture=info.get("architecture", "unknown"),
        memory_size=int(info.get("memory_size", 0)),
        max_buffer_length=int(info.get("max_buffer_length", 0)),
        recommended_working_set=int(info.get("max_recommended_working_set_size", 0)),
        os_version=platform.mac_ver()[0] or platform.platform(),
        chip=_sysctl("machdep.cpu.brand_string") or platform.machine(),
        gpu_cores=_gpu_core_count(),
    )


def metal_limits(force: bool = False) -> dict | None:
    """Hardware limits straight from Metal, via `swift tools/device_probe.swift`.

    Returns ``None`` when the Swift toolchain is unavailable. The probe is compiled
    from source each call, so it costs a few seconds; results are cached per process.
    """
    global _METAL_CACHE
    if _METAL_CACHE is not None and not force:
        return _METAL_CACHE
    if not SWIFT_PROBE.exists():
        return None

    cache_dir = REPO_ROOT / ".swiftcache"
    cache_dir.mkdir(exist_ok=True)
    try:
        proc = subprocess.run(
            ["swift", "-module-cache-path", str(cache_dir), str(SWIFT_PROBE)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None

    limits: dict[str, object] = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().split("  ")[0].strip()
        limits[key.strip()] = int(value) if value.lstrip("-").isdigit() else value
    _METAL_CACHE = limits
    return limits


_METAL_CACHE: dict | None = None


def report(as_json: bool = False) -> dict:
    """Print (and return) a one-glance summary of the GPU this code will run on."""
    gpu = gpu_info()
    data: dict[str, object] = {"gpu": asdict(gpu), "mlx_version": mx.__version__}
    limits = metal_limits()
    if limits:
        data["metal"] = limits

    if as_json:
        print(json.dumps(data, indent=2))
        return data

    print("=" * 68)
    print(f"  GPU            : {gpu.name}  ({gpu.architecture})")
    print(f"  SoC            : {gpu.chip}")
    print(f"  GPU cores      : {gpu.gpu_cores if gpu.gpu_cores else 'n/a'}")
    print(f"  macOS          : {gpu.os_version}")
    print(f"  Unified memory : {gpu.memory_gib:.1f} GiB")
    print(f"  Max 1 buffer   : {gpu.max_buffer_gib:.2f} GiB")
    print(f"  Working set    : {gpu.working_set_gib:.1f} GiB (GPU budget before swapping)")
    print(f"  MLX            : {mx.__version__}   default device: {mx.default_device()}")
    if limits:
        print(f"  Metal family   : apple9={limits.get('supportsFamily(.apple9)')} "
              f"metal4={limits.get('supportsFamily(.metal4)')}")
        print(f"  Max threads/tg : {limits.get('maxThreadsPerThreadgroup')}")
        tg_mem = limits.get("maxThreadgroupMemoryLength")
        if isinstance(tg_mem, int):
            print(f"  Threadgroup mem: {tg_mem} B ({tg_mem // 1024} KiB) per threadgroup")
    else:
        print("  Metal limits   : unavailable (run `python -m gpuk.device --metal`)")
    print("=" * 68)
    return data


def main() -> None:
    import sys

    report(as_json="--json" in sys.argv)


if __name__ == "__main__":
    main()
