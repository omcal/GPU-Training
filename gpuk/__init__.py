"""gpuk — the tiny harness the labs share.

Nothing here is a framework. It is deliberately small so you can read all of it:

* :mod:`gpuk.device`  — what GPU is this, and what are its hard limits
* :mod:`gpuk.launch`  — CUDA ``<<<blocks, threads>>>`` -> Metal ``dispatchThreads``
* :mod:`gpuk.bench`   — honest GPU timing, bandwidth/FLOPs, roofline rows
* :mod:`gpuk.check`   — compare a kernel against a reference before trusting it
"""

from gpuk.bench import (
    Timing,
    bandwidth_gbs,
    compare_vs_reference,
    gflops,
    header,
    report_row,
    synchronize,
    throughput_ms,
    time_ms,
)
from gpuk.check import assert_close, max_abs_err, to_numpy
from gpuk.device import GpuInfo, gpu_info, metal_limits, report
from gpuk.launch import DEFAULT_THREADS, Launch, blocks_for, cdiv, cuda

__all__ = [
    "DEFAULT_THREADS",
    "GpuInfo",
    "Launch",
    "Timing",
    "assert_close",
    "bandwidth_gbs",
    "blocks_for",
    "cdiv",
    "compare_vs_reference",
    "gflops",
    "gpu_info",
    "header",
    "max_abs_err",
    "metal_limits",
    "report",
    "report_row",
    "synchronize",
    "throughput_ms",
    "time_ms",
    "to_numpy",
]
