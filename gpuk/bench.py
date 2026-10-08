"""Timing and roofline helpers.

GPUs run asynchronously: the host returns from a kernel call long before the GPU is
done. MLX is *lazy* on top of that — it builds a graph and only computes when you ask
("``mx.eval``" or "``mx.synchronize``"). A timing harness that forgets this measures
how fast Python builds a graph, not how fast the GPU computes. So every measurement
here synchronises before stopping the clock.

Method used by :func:`time_ms`: warm up (the first launch of a Metal kernel triggers
JIT compilation and can take seconds), then time each iteration separately with a
synchronise inside the loop, and report min/median/mean. The per-iteration
synchronise costs tens of microseconds, which is negligible for kernels above
~0.1 ms and is stated plainly rather than hidden.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx


def synchronize() -> None:
    """Block until every queued GPU operation has finished."""
    sync = getattr(mx, "synchronize", None)
    if sync is not None:
        sync()
    else:  # pragma: no cover - fallback for older MLX
        mx.eval(mx.array([0]))


@dataclass(frozen=True)
class Timing:
    """Timing summary in milliseconds."""

    minimum: float
    median: float
    mean: float
    iters: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.median:.4f} ms (min {self.minimum:.4f})"


def time_ms(fn: Callable[[], object], warmup: int = 10, iters: int = 50) -> Timing:
    """Time ``fn`` on the GPU. See module docstring for the method."""
    if iters < 1:
        raise ValueError("iters must be >= 1")
    for _ in range(warmup):
        fn()
    synchronize()

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return Timing(
        minimum=min(samples),
        median=statistics.median(samples),
        mean=statistics.fmean(samples),
        iters=iters,
    )


def best_ms(
    fn: Callable[[], object], rounds: int = 3, warmup: int = 5, iters: int = 20
) -> Timing:
    """Repeat the whole measurement several times and keep the fastest round.

    A fanless laptop shares its thermal budget with the CPU, and other processes come
    and go. On this machine two consecutive measurements of the same kernel can differ
    by 20%. Repeating the measurement and keeping the best round is the standard way to
    get a number that reflects the *kernel* rather than the *moment*.

    The cost is time: ``rounds * (warmup + iters)`` launches. Use it for the numbers
    you intend to draw conclusions from, not for exploratory plots.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    best: Timing | None = None
    for _ in range(rounds):
        candidate = time_ms(fn, warmup=warmup, iters=iters)
        if best is None or candidate.median < best.median:
            best = candidate
    assert best is not None
    return best


def throughput_ms(fn: Callable[[], object], warmup: int = 10, iters: int = 100) -> Timing:
    """Steady-state cost per call: queue ``iters`` calls, synchronise once, divide.

    This is how GPU vendors measure: it hides per-launch CPU overhead by keeping the
    GPU queue full, so it answers "how fast is the GPU at this job in a loop".

    Contrast with :func:`time_ms`, which synchronises every iteration and therefore
    also counts the cost of *one* launch. For kernels slower than ~0.5 ms the two
    agree. For tiny kernels they diverge wildly, and the gap *is* the launch overhead.
    Choose deliberately; do not compare numbers produced by the two methods.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    for _ in range(warmup):
        fn()
    synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize()
    per_call = (time.perf_counter() - t0) * 1e3 / iters
    return Timing(minimum=per_call, median=per_call, mean=per_call, iters=iters)


def bandwidth_gbs(nbytes: int, ms: float) -> float:
    """Effective bandwidth in GB/s. Uses 1e9, the convention GPU vendors report in."""
    if ms <= 0:
        raise ValueError("ms must be positive")
    return nbytes / (ms * 1e-3) / 1e9


def gflops(flops: float, ms: float) -> float:
    """Effective throughput in GFLOP/s (1e9)."""
    if ms <= 0:
        raise ValueError("ms must be positive")
    return flops / (ms * 1e-3) / 1e9


def report_row(
    name: str,
    timing: Timing,
    nbytes: int | None = None,
    flops: float | None = None,
    peak_bw: float | None = None,
    peak_flops: float | None = None,
) -> dict[str, float | str]:
    """Print one aligned benchmark row and return it as a dict for later reuse."""
    row: dict[str, float | str] = {"name": name, "ms": timing.median}
    parts = [f"{name:<34} {timing.median:>9.4f} ms"]
    if nbytes is not None:
        bw = bandwidth_gbs(nbytes, timing.median)
        row["gbs"] = bw
        parts.append(f"{bw:>9.1f} GB/s")
        if peak_bw:
            parts.append(f"({100 * bw / peak_bw:>5.1f}% of peak)")
            row["pct_peak_bw"] = 100 * bw / peak_bw
    if flops is not None:
        gf = gflops(flops, timing.median)
        row["gflops"] = gf
        parts.append(f"{gf:>9.1f} GFLOP/s")
        if peak_flops:
            parts.append(f"({100 * gf / peak_flops:>5.1f}% of peak)")
            row["pct_peak_flops"] = 100 * gf / peak_flops
    print("  ".join(parts))
    return row


def header(title: str) -> None:
    """Section banner, so lab output stays readable in a terminal."""
    print()
    print(f"--- {title} " + "-" * max(0, 62 - len(title)))


def compare_vs_reference(name: str, ms_kernel: float, ms_reference: float) -> None:
    """Print how a hand-written kernel stacks up against the library implementation.

    Losing to the vendor library is normal and expected. The interesting question is
    *how much* and *why* — that is what the labs are for.
    """
    ratio = ms_kernel / ms_reference if ms_reference > 0 else float("inf")
    verdict = "faster" if ratio < 1 else "slower"
    print(
        f"  {name}: yours {ms_kernel:.4f} ms vs library {ms_reference:.4f} ms "
        f"-> {ratio:.2f}x {verdict}"
    )
