"""Measured hardware ceilings for this specific machine.

Spec sheets are marketing; the numbers your kernels can actually reach depend on
memory clocks, thermals, and how well the vendor library in question is tuned. Since
every lab wants to say "this kernel reached 82% of peak", the peak has to be
*measured*, on the machine that is running the lab.

Two measurement traps, both worth knowing before you trust any GPU number:

* **Not enough warm-up.** The first dispatch of a Metal kernel compiles a shader
  library. With only a few warm-up iterations that cost is averaged into your result
  and can understate bandwidth by 30%+. These functions warm up properly.
* **Timing a busy queue.** MLX hands work to the GPU asynchronously. Read the clock
  without synchronising and you measure how fast Python builds a graph. Every
  measurement in :mod:`gpuk.bench` synchronises.

The ``minimum`` sample is used rather than the mean: the fastest observed run is the
one least polluted by other work happening on the machine, which is the closest
estimate of what the hardware can do.

Both functions are cached per process.
"""

from __future__ import annotations

import mlx.core as mx

from gpuk.bench import bandwidth_gbs, gflops, time_ms

_PEAK_BW: float | None = None
_PEAK_BW_ADD: float | None = None
_PEAK_FLOPS: float | None = None


def peak_bandwidth_gbs(n: int = 1 << 26, warmup: int = 5, iters: int = 20) -> float:
    """Streaming **1 read : 1 write** bandwidth ceiling, in GB/s.

    Probed with a unary elementwise op: ``n * 4`` bytes read, ``n * 4`` written. This
    is the right ceiling for copy-like and normalisation-like kernels.

    The default size (2^26 floats, 512 MiB of traffic) is deliberately large: small
    working sets cannot saturate DRAM because launch overhead and ramp-up dominate.
    A ceiling measured on a 4 MiB array is roughly half the asymptotic figure, so
    comparing a big kernel against a small probe would flatter the kernel.

    There is no single "the bandwidth" of a GPU: the figure depends on the read:write
    mix, on the working-set size, and on how the kernel addresses memory. That is why
    :func:`peak_bandwidth_add_gbs` exists as a second probe rather than one number
    everyone compares against and gets confused by. On this machine the two probes land
    within a few percent of each other (both ≈96-98 GB/s); on other hardware they can
    diverge sharply, so measure the mix you actually care about.
    """
    global _PEAK_BW
    if _PEAK_BW is not None:
        return _PEAK_BW

    src = mx.random.normal(shape=(n,), dtype=mx.float32)
    mx.eval(src)

    def run() -> None:
        mx.eval(mx.maximum(src, 0.0))

    timing = time_ms(run, warmup=warmup, iters=iters)
    _PEAK_BW = bandwidth_gbs(2 * n * 4, timing.minimum)
    return _PEAK_BW


def peak_bandwidth_add_gbs(n: int = 1 << 26, warmup: int = 5, iters: int = 20) -> float:
    """Streaming **2 reads : 1 write** bandwidth ceiling, in GB/s.

    Probed with ``mx.add`` over two arrays: ``2 * n * 4`` read, ``n * 4`` written, and
    the benchmark charges all ``3 * n * 4`` bytes. This is the ceiling to compare
    vector-add-shaped kernels against. Having a second probe with a different read:write
    mix is the point: a single "peak bandwidth" number silently bakes in one mix.
    """
    global _PEAK_BW_ADD
    if _PEAK_BW_ADD is not None:
        return _PEAK_BW_ADD

    a = mx.random.normal(shape=(n,), dtype=mx.float32)
    b = mx.random.normal(shape=(n,), dtype=mx.float32)
    mx.eval(a, b)

    def run() -> None:
        mx.eval(mx.add(a, b))

    timing = time_ms(run, warmup=warmup, iters=iters)
    _PEAK_BW_ADD = bandwidth_gbs(3 * n * 4, timing.minimum)
    return _PEAK_BW_ADD


def peak_flops_fp32(n: int = 2048, warmup: int = 5, iters: int = 15) -> float:
    """FP32 throughput ceiling in GFLOP/s, measured with a large dense matmul.

    ``mx.matmul`` dispatches to Apple's tuned GEMM, which uses the GPU's
    ``simdgroup_matrix`` units — the closest thing Apple GPUs have to the tensor cores
    that PMPP spends its later chapters on. A hand-written kernel that only touches the
    plain FP32 ALUs will land below this number, and knowing *which* of the two limits
    (bandwidth or FLOPs) you are hitting is exactly what the roofline model in
    ``docs/03-how-to-measure.md`` is for.
    """
    global _PEAK_FLOPS
    if _PEAK_FLOPS is not None:
        return _PEAK_FLOPS

    a = mx.random.normal(shape=(n, n), dtype=mx.float32)
    b = mx.random.normal(shape=(n, n), dtype=mx.float32)
    mx.eval(a, b)

    def run() -> None:
        mx.eval(mx.matmul(a, b))

    timing = time_ms(run, warmup=warmup, iters=iters)
    _PEAK_FLOPS = gflops(2.0 * n**3, timing.minimum)
    return _PEAK_FLOPS


def peaks() -> dict[str, float]:
    """All ceilings, handy for printing at the top of a lab."""
    return {
        "bandwidth_1r1w_gbs": peak_bandwidth_gbs(),
        "bandwidth_2r1w_gbs": peak_bandwidth_add_gbs(),
        "flops_fp32": peak_flops_fp32(),
    }
