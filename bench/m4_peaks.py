#!/usr/bin/env python
"""Measure this machine's hardware ceilings and print a roofline summary.

    python bench/m4_peaks.py

Run it on AC power with nothing else using the GPU. Numbers on a fanless laptop move by
double-digit percentages between runs; `best_ms` repeats each measurement and keeps the
fastest round, and the spread is printed so you can see how much to trust the result.

Every lab imports these ceilings (`gpuk.peaks`) so that "82% of peak" always means "82%
of what this machine measured today", never a number from a spec sheet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpuk import gflops, header, report  # noqa: E402
from gpuk.bench import bandwidth_gbs, best_ms, time_ms  # noqa: E402
from gpuk.peaks import peak_bandwidth_add_gbs, peak_bandwidth_gbs, peak_flops_fp32  # noqa: E402


def bench_read_bandwidth(n: int = 1 << 26) -> float:
    """Best observed coalesced read bandwidth, with 16 loads in flight per thread.

    Read-only work is not the same shape as read+write work, so this probe exists to keep
    the two apart. On this machine all three mixes land within a few percent of each
    other; on other hardware they can diverge by a lot, which is exactly why you measure
    rather than assume.

    Note the placement of `simd_sum`: OUTSIDE the `if`. A simdgroup shuffle must be
    reached by all 32 lanes; putting it inside a divergent branch gives wrong answers and
    lets the compiler delete the loads that were feeding it. An early version of this
    probe did exactly that and "measured" 167 GB/s on a machine rated for ~120 -- a
    useful reminder that a nonsensically fast number is a bug report, not a discovery.
    """
    src = mx.random.normal(shape=(n,), dtype=mx.float32)
    mx.eval(src)
    count = 16
    threads = n // count
    blocks = threads // 256

    source = """
        uint t = thread_position_in_grid.x;
        uint T = threads_per_grid.x;
        float acc = 0.0f;
        for (uint k = 0; k < COUNT; ++k) acc += inp[t + k * T];
        float s = simd_sum(acc);          // all 32 lanes must reach this
        if (t % 32 == 0) out[t / 32] = s; // only lane 0 stores
    """
    kernel = mx.fast.metal_kernel(
        name="bench_coalesced_read", input_names=["inp"], output_names=["out"], source=source
    )

    def run():
        return kernel(
            inputs=[src],
            template=[("COUNT", count)],
            output_shapes=[(threads // 32,)],
            output_dtypes=[mx.float32],
            grid=(blocks * 256, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]

    # Correctness check: a bandwidth number from a kernel that reads the wrong data is
    # worthless. The reduction output must reproduce the array's total.
    got = float(np.array(run()).astype(np.float64).sum())
    expected = float(np.array(src).astype(np.float64).sum())
    if abs(got - expected) > 1e-3 * abs(expected):
        raise RuntimeError(
            "read-bandwidth probe is not reading the whole array "
            f"(sum {got} vs {expected}) -- the number it reports would be meaningless"
        )

    best = 0.0
    for _ in range(3):
        t = time_ms(lambda: mx.eval(run()), warmup=5, iters=20)
        best = max(best, bandwidth_gbs(n * 4, t.minimum))
    return best


def main() -> None:
    report()

    header("Memory: streaming ceilings (large working sets, warmed up)")
    bw_1r1w = peak_bandwidth_gbs()
    bw_2r1w = peak_bandwidth_add_gbs()
    bw_read = bench_read_bandwidth()
    print(f"  1 read  : 1 write   (unary streaming)     {bw_1r1w:8.1f} GB/s")
    print(f"  2 reads : 1 write   (a + b)               {bw_2r1w:8.1f} GB/s")
    print(f"  read only, coalesced, 16 elem/thread      {bw_read:8.1f} GB/s")
    print(
        "\n  All three mixes land within a few percent of each other here, so this machine\n"
        "  has one effective memory ceiling rather than a read ceiling and a write ceiling.\n"
        "  That is a measurement, not a law: on GPUs with separate read/write paths the\n"
        "  figures spread out, which is why the mix must always be quoted."
    )

    header("Compute: FP32 throughput")
    flops = peak_flops_fp32()
    print(f"  matmul 2048x2048x2048                     {flops:8.1f} GFLOP/s")

    header("Roofline for this machine")
    crossover = flops / bw_1r1w
    print(f"  crossover = peak_flops / peak_bandwidth = {crossover:.1f} FLOP/byte\n")
    print(f"  {'kernel':<28} {'FLOP/byte':>10}   {'bound by':<12}")
    cases = [
        ("vector add", 1 / 12),
        ("axpy (2R:1W)", 2 / 12),
        ("reduction / sum", 1 / 4),
        ("row softmax", 2.0),
        ("matmul 1024^3", 2 * 1024**3 / (4 * 3 * 1024**2)),
        ("matmul 4096^3", 2 * 4096**3 / (4 * 3 * 4096**2)),
    ]
    for name, intensity in cases:
        bound = "compute" if intensity > crossover else "memory"
        print(f"  {name:<28} {intensity:>10.2f}   {bound:<12}")

    header("How to read this")
    print(
        """
  * If your kernel's arithmetic intensity is below the crossover, it is memory bound:
    only moving fewer bytes can help. Fusion, better access patterns, narrower dtypes.
  * Above the crossover it is compute bound: only fewer or cheaper FLOPs help. Wider
    vector types, register blocking, `simdgroup_matrix`.
  * Optimising the wrong side of the roofline produces a flat line. That is why every
    lab here classifies the kernel before it tries to speed it up.

  Spec-sheet cross-check: M4 is rated at ~120 GB/s of LPDDR5X bandwidth. Reaching ~94
  GB/s of usable read+write traffic is ~78% of nominal -- a normal, healthy figure. If
  you measure well above nominal, you have a bug, not a discovery.
"""
    )


if __name__ == "__main__":
    main()
