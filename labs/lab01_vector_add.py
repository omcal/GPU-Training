"""Lab 01 — Your first Metal kernel: vector add, and what "bandwidth bound" means.

Run:  python labs/lab01_vector_add.py

Goal
----
Write a GPU kernel in the Metal Shading Language, launch it from Python, prove it is
correct, then measure it against the machine's memory bandwidth ceiling.

The one thing to internalise
----------------------------
Vector add does 1 FLOP per 12 bytes moved (two float32 reads, one float32 write).
No amount of arithmetic cleverness helps: the kernel is *memory bound*, and the only
question that matters is "how close to peak bandwidth did we get?". This is true for
a large fraction of real GPU work — normalisation, elementwise activations, scaling,
embedding lookups — so it is worth doing once, very carefully.

Two launches are compared:

* ``per_element``  — one thread per element. Simple, correct, and what everyone writes
  first. Grid size grows with n, which becomes a problem at scale.
* ``grid_stride``  — a fixed-size grid where each thread walks the array in strides.
  This is what CUDA/PMPP calls a "grid-stride loop" and it is the pattern to default to.

Metal vs CUDA, in one box
-------------------------
    CUDA                                Metal / MLX
    ----                                -----------
    __global__ void k(...)              [[kernel]] void k(...)   (MLX writes this wrapper)
    thread_position_in_grid is free     blockIdx.x*blockDim.x+threadIdx.x
    k<<<blocks, threads>>>(...)         grid=(blocks*threads,), threadgroup=(threads,)
                                        ^^^ TOTAL threads, not a block count. See gpuk/launch.py.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from gpuk import (
    assert_close,
    blocks_for,
    compare_vs_reference,
    cuda,
    header,
    report,
    report_row,
    time_ms,
)
from gpuk.peaks import peak_bandwidth_add_gbs, peak_bandwidth_gbs

N = 1 << 24  # 16.7M floats = 64 MiB per array, 192 MiB of traffic per launch.
THREADS = 256


# --------------------------------------------------------------------------------------
# Kernel 1: one thread per element.
#
# Note what is NOT in this source: no function signature, no `[[buffer(n)]]`, no kernel
# declaration. MLX generates all of that from `input_names`/`output_names` and the
# metal attributes you use. You write only the body.
#
# `inp_shape` is injected by MLX because the shape array appears in the source. It is a
# device-visible array, so `inp_shape[0]` costs a read — cheap, and cached like any
# uniform. The alternative, passing a scalar buffer, costs a buffer binding instead.
# --------------------------------------------------------------------------------------
PER_ELEMENT_SRC = """
    uint i = thread_position_in_grid.x;
    if (i < inp_shape[0]) {
        out[i] = inp[i] + other[i];
    }
"""

kernel_per_element = mx.fast.metal_kernel(
    name="vector_add_per_element",
    input_names=["inp", "other"],
    output_names=["out"],
    source=PER_ELEMENT_SRC,
)


# --------------------------------------------------------------------------------------
# Kernel 2: grid-stride loop.
#
# `threads_per_grid` is a free Metal attribute holding the total number of threads in
# the dispatch. Striding by it means every thread of every threadgroup works on a
# different part of the array, and each *warp* still touches consecutive addresses
# (because consecutive threads within the loop iteration read consecutive elements).
# Keeping that coalescing property is what makes this fast.
# --------------------------------------------------------------------------------------
GRID_STRIDE_SRC = """
    uint stride = threads_per_grid.x;
    for (uint i = thread_position_in_grid.x; i < inp_shape[0]; i += stride) {
        out[i] = inp[i] + other[i];
    }
"""

kernel_grid_stride = mx.fast.metal_kernel(
    name="vector_add_grid_stride",
    input_names=["inp", "other"],
    output_names=["out"],
    source=GRID_STRIDE_SRC,
)


def add_per_element(a: mx.array, b: mx.array) -> mx.array:
    launch = blocks_for(a.size, THREADS)
    return kernel_per_element(
        inputs=[a, b],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        **launch.kwargs(),
    )[0]


def add_grid_stride(a: mx.array, b: mx.array, blocks: int = 512) -> mx.array:
    launch = cuda(blocks, THREADS)
    return kernel_grid_stride(
        inputs=[a, b],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        **launch.kwargs(),
    )[0]


def main() -> None:
    report()
    peak_bw = peak_bandwidth_add_gbs()
    unary_bw = peak_bandwidth_gbs()
    print(
        f"\nMeasured bandwidth ceilings on this machine:\n"
        f"  2 reads : 1 write (mx.add, what vector add does) : {peak_bw:6.1f} GB/s\n"
        f"  1 read  : 1 write (unary streaming)             : {unary_bw:6.1f} GB/s\n"
        "  The two mixes measure within a few percent of each other here, so this machine\n"
        "  has one effective memory ceiling. On other GPUs the mix matters a lot -- always\n"
        "  quote which one you measured."
    )

    a = mx.random.normal(shape=(N,), dtype=mx.float32)
    b = mx.random.normal(shape=(N,), dtype=mx.float32)
    mx.eval(a, b)
    reference = np.array(a) + np.array(b)
    nbytes = 3 * N * 4  # 2 reads + 1 write

    header(f"Correctness (n = {N:,}, {nbytes / 2**20:.0f} MiB of traffic)")
    assert_close(add_per_element(a, b), reference, label="per_element")
    assert_close(add_grid_stride(a, b), reference, label="grid_stride")
    assert_close(add_grid_stride(a, b, blocks=1), reference, label="grid_stride, 1 block")
    assert_close(add_grid_stride(a, b, blocks=100_000), reference, label="grid_stride, over-sized")

    # A boundary case. 3000 elements does not divide by 256, so the last threadgroup is
    # partly out of range. Off-by-one bugs live here.
    small_a = mx.random.normal(shape=(3000,), dtype=mx.float32)
    small_b = mx.random.normal(shape=(3000,), dtype=mx.float32)
    assert_close(
        add_per_element(small_a, small_b),
        np.array(small_a) + np.array(small_b),
        label="per_element (n=3000, ragged)",
    )

    def bench(fn):
        """Warm up properly: the first launch of a Metal kernel compiles a shader."""
        return time_ms(fn, warmup=5, iters=20)

    header("Benchmarks")
    print(f"  {'kernel':<34} {'time':>12}  {'bandwidth':>12}")
    report_row("mx.add (Apple's own kernel)", bench(lambda: mx.eval(mx.add(a, b))), nbytes, peak_bw=peak_bw)
    report_row("per_element (blocks_for)", bench(lambda: mx.eval(add_per_element(a, b))), nbytes, peak_bw=peak_bw)
    report_row("grid_stride (512 blocks)", bench(lambda: mx.eval(add_grid_stride(a, b))), nbytes, peak_bw=peak_bw)
    report_row("grid_stride (2048 blocks)", bench(lambda: mx.eval(add_grid_stride(a, b, 2048))), nbytes, peak_bw=peak_bw)

    header(f"The grid size is a tuning knob (n = 2^26, {3 * 2**26 * 4 / 2**20:.0f} MiB of traffic)")
    big_n = 1 << 26
    x = mx.random.normal(shape=(big_n,), dtype=mx.float32)
    y = mx.random.normal(shape=(big_n,), dtype=mx.float32)
    mx.eval(x, y)
    big_bytes = 3 * big_n * 4

    # One thread per element: the grid is dictated by n (262144 blocks here).
    report_row(
        "per_element (n/256 blocks)",
        bench(lambda: mx.eval(add_per_element(x, y))),
        big_bytes,
        peak_bw=peak_bw,
    )
    # Grid-stride: the grid is chosen by you, and each thread loops.
    # Too few blocks -> the GPU cannot hide memory latency. Too many -> per-block
    # scheduling overhead and worse instruction cache behaviour.
    for blocks in (64, 256, 1024, 4096, 16384, 65536):
        report_row(
            f"grid_stride ({blocks} blocks)",
            bench(lambda b=blocks: mx.eval(add_grid_stride(x, y, b))),
            big_bytes,
            peak_bw=peak_bw,
        )

    compare_vs_reference(
        "grid_stride vs mx.add",
        bench(lambda: mx.eval(add_grid_stride(a, b))).median,
        bench(lambda: mx.eval(mx.add(a, b))).median,
    )

    header("What to take away")
    print(
        """
  1. `grid` in MLX is the TOTAL number of threads. CUDA's `<<<blocks, threads>>>`
     becomes grid=(blocks*threads,) -- get it wrong and you silently compute garbage.
     gpuk.cuda(blocks, threads) exists so you never have to think about it again.

  2. Vector add is memory bound. Your kernel lands within a few percent of Apple's own
     kernel and of the measured 2R:1W bandwidth ceiling. There is nothing left to win
     here, and any "optimisation" that does not reduce bytes moved is noise. That
     feeling -- "we are at the roofline, stop" -- is the single most useful instinct in
     GPU work, and it is why every later lab measures against a measured ceiling.

  3. `if (i < n)` is not optional. n is almost never a multiple of the threadgroup size,
     and the ragged last threadgroup is where off-by-one bugs live.

  4. Grid-stride loops decouple grid size from problem size. The grid becomes a tuning
     knob (blocks) rather than a consequence of n, and the kernel then works for any n.

  5. Numbers on a laptop fanless Air wobble by several percent between runs. Treat
     differences under ~5% as noise unless you have run the comparison many times.
"""
    )


if __name__ == "__main__":
    main()
