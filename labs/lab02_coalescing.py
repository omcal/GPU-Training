"""Lab 02 — Memory access patterns: cache-line amplification and the tiled transpose.

Run:  python labs/lab02_coalescing.py

Goal
----
See, with numbers, that how you *address* memory can matter more than how much you
move — and learn the standard fix (staging through threadgroup memory).

Background
----------
GPUs fetch memory in cache lines (128 bytes here). When the 32 lanes of a simdgroup
read 32 consecutive float32s they need 128 bytes and the hardware performs **one**
line fetch: a *coalesced* access. When those same 32 lanes read addresses 128 bytes
apart, the hardware performs **32** line fetches to deliver the same 128 useful bytes.

Part A measures exactly that, and makes a distinction that trips people up:

    *useful* bytes  — what the program asked for
    *fetched* bytes — what the memory system actually moved

A kernel can be "using" 2 GB/s while the hardware is fetching 64 GB/s. The gap is
pure waste, and the fix is almost never "make the memory faster" — it is "touch the
bytes you asked for".

Part B is the standard fix applied to matrix transpose: stage a tile through on-chip
threadgroup memory so that *both* the global read and the global write are coalesced.
This is the ancestor of the tiled matmul in lab04 and of every attention kernel.

Measurement honesty
-------------------
These numbers come from a fanless laptop. Repeated runs of the same kernel can differ
by 20%, and sustained load causes real thermal throttling. ``gpuk.bench.best_ms``
repeats each measurement and keeps the fastest round. Treat anything under ~10% as a
tie, and do not run this lab while something else is using the GPU.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from gpuk import (
    assert_close,
    bandwidth_gbs,
    cuda,
    header,
    report,
    report_row,
)
from gpuk.bench import best_ms
from gpuk.peaks import peak_bandwidth_gbs

N = 1 << 24  # 64 MiB of float32
THREADS = 256


def bench(fn):
    """Best-of-3 rounds; see the note on measurement honesty above."""
    return best_ms(fn, rounds=3, warmup=5, iters=20)


# ======================================================================================
# Part A — cache-line amplification.
#
# Every thread reads exactly ONE element, at index `t * STRIDE`. Since the array is
# 64 MiB and the stride only re-orders which elements are touched, the number of
# distinct 128-byte lines touched is the same for every stride in 1..32:
#
#     lines touched = 64 MiB / 128 B = 524288
#
# What changes is how many of those fetched bytes the program actually uses.
#
# The results are summed within each simdgroup and written once per simdgroup, so the
# write traffic is ~2 MiB and never confuses the read measurement.
# ======================================================================================
SPACED_READ_SRC = """
    uint t = thread_position_in_grid.x;
    float v = inp[t * STRIDE];
    float s = simd_sum(v);
    if (thread_index_in_simdgroup == 0) out[t / 32] = s;
"""

kernel_spaced_read = mx.fast.metal_kernel(
    name="spaced_read",
    input_names=["inp"],
    output_names=["out"],
    source=SPACED_READ_SRC,
)


def spaced_read(a: mx.array, stride: int) -> mx.array:
    n_out = a.size // stride
    if n_out % THREADS:
        raise ValueError("pick a size/stride that divides evenly")
    return kernel_spaced_read(
        inputs=[a],
        template=[("STRIDE", stride)],
        output_shapes=[(n_out // 32,)],
        output_dtypes=[a.dtype],
        **cuda(n_out // THREADS, THREADS).kwargs(),
    )[0]


# ======================================================================================
# Part B.1 — naive transpose.
#
# grid is (cols, rows), threadgroup is (32, 8), so the 32 lanes of a simdgroup share
# one `ty` and span 32 consecutive `x` values.
#
#   read  inp[y * cols + x]  -> consecutive lanes read consecutive addresses: COALESCED
#   write out[x * rows + y]  -> consecutive lanes write `rows` floats apart: SCATTERED
#
# Each scattered warp instruction touches 32 different cache lines to deliver 128 useful
# bytes. The large L2 on Apple Silicon absorbs some of this by collecting the pieces of
# each line from many different warps, which is why the naive version is only ~1.5x
# slower rather than 32x. On a NVIDIA GPU the gap is much wider.
# ======================================================================================
TRANSPOSE_NAIVE_SRC = """
    uint x = thread_position_in_grid.x;   // input column
    uint y = thread_position_in_grid.y;   // input row
    if (x < cols[0] && y < rows[0]) {
        out[x * rows[0] + y] = inp[y * cols[0] + x];
    }
"""

kernel_transpose_naive = mx.fast.metal_kernel(
    name="transpose_naive",
    input_names=["inp", "rows", "cols"],
    output_names=["out"],
    source=TRANSPOSE_NAIVE_SRC,
)


def transpose_naive(a: mx.array) -> mx.array:
    rows, cols = a.shape
    return kernel_transpose_naive(
        inputs=[a, mx.array([rows], dtype=mx.uint32), mx.array([cols], dtype=mx.uint32)],
        output_shapes=[(cols, rows)],
        output_dtypes=[a.dtype],
        grid=(cols, rows, 1),
        threadgroup=(32, 8, 1),
    )[0]


# ======================================================================================
# Part B.2 — tiled transpose. The classic CUDA "transpose" kernel, in Metal.
#
# 256 threads (32x8) each handle 4 elements of a 32x32 tile, so there are 4 iterations
# in each direction. Both global accesses are coalesced; the transpose itself happens
# in threadgroup memory.
#
# Two details that are easy to get wrong:
#
#   * `tile[32][33]` — the +1 padding is not decoration. A 32x32 float array has a row
#     stride of 32 floats = 128 bytes. Reading a *column* then makes every lane hit the
#     same memory bank: a 32-way bank conflict. A stride of 33 spreads consecutive rows
#     across different banks and removes it.
#
#   * `threadgroup_barrier(...)` after filling the tile. Without it a thread can read a
#     slot another thread has not written yet. This is `__syncthreads()` in CUDA, and
#     barrier bugs are silent and timing-dependent.
#
# Note the index swap between the two phases: fill uses `tile[ty + j][tx]`, drain uses
# `tile[tx][ty + j]`. That swap *is* the transpose.
# ======================================================================================
TRANSPOSE_TILED_SRC = """
    threadgroup float tile[32][33];

    uint bx = threadgroup_position_in_grid.x;
    uint by = threadgroup_position_in_grid.y;
    uint tx = thread_index_in_threadgroup % 32;
    uint ty = thread_index_in_threadgroup / 32;      // 0..7

    // Fill: 8 rows per iteration, coalesced along tx.
    for (uint j = 0; j < 32; j += 8) {
        tile[ty + j][tx] = inp[(by * 32 + ty + j) * cols[0] + bx * 32 + tx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Drain: transpose on the way out, still coalesced along tx.
    for (uint j = 0; j < 32; j += 8) {
        out[(bx * 32 + ty + j) * rows[0] + by * 32 + tx] = tile[tx][ty + j];
    }
"""

kernel_transpose_tiled = mx.fast.metal_kernel(
    name="transpose_tiled",
    input_names=["inp", "rows", "cols"],
    output_names=["out"],
    source=TRANSPOSE_TILED_SRC,
)


def transpose_tiled(a: mx.array) -> mx.array:
    rows, cols = a.shape
    if rows % 32 or cols % 32:
        raise ValueError(f"tiled transpose needs multiples of 32, got {a.shape}")
    return kernel_transpose_tiled(
        inputs=[a, mx.array([rows], dtype=mx.uint32), mx.array([cols], dtype=mx.uint32)],
        output_shapes=[(cols, rows)],
        output_dtypes=[a.dtype],
        grid=((cols // 32) * 32, (rows // 32) * 8, 1),
        threadgroup=(32, 8, 1),
    )[0]


# ======================================================================================
# Part B.3 — the cautionary variant: a 64x64 tile.
#
# 64 x 65 x 4 B = 16.6 KiB of threadgroup memory per threadgroup, on a GPU whose entire
# budget is 32 KiB and which is shared with the L1 cache. Only one threadgroup can be
# resident, there are barely any warps to hide memory latency with, and throughput
# collapses. Bigger tiles are not free: threadgroup memory trades directly against
# occupancy.
# ======================================================================================
TRANSPOSE_BIG_TILE_SRC = """
    threadgroup float tile[64][65];

    uint bx = threadgroup_position_in_grid.x;
    uint by = threadgroup_position_in_grid.y;
    uint tx = thread_index_in_threadgroup % 16;
    uint ty = thread_index_in_threadgroup / 16;

    for (uint j = 0; j < 64; j += 16) {
        for (uint i = 0; i < 64; i += 16) {
            tile[ty + j][tx + i] = inp[(by * 64 + ty + j) * cols[0] + bx * 64 + tx + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 64; j += 16) {
        for (uint i = 0; i < 64; i += 16) {
            out[(bx * 64 + ty + j) * rows[0] + by * 64 + tx + i] = tile[tx + i][ty + j];
        }
    }
"""

kernel_transpose_big = mx.fast.metal_kernel(
    name="transpose_big_tile",
    input_names=["inp", "rows", "cols"],
    output_names=["out"],
    source=TRANSPOSE_BIG_TILE_SRC,
)


def transpose_big_tile(a: mx.array) -> mx.array:
    rows, cols = a.shape
    if rows % 64 or cols % 64:
        raise ValueError(f"the 64x64 tile variant needs multiples of 64, got {a.shape}")
    return kernel_transpose_big(
        inputs=[a, mx.array([rows], dtype=mx.uint32), mx.array([cols], dtype=mx.uint32)],
        output_shapes=[(cols, rows)],
        output_dtypes=[a.dtype],
        grid=((cols // 64) * 64, (rows // 64) * 16, 1),
        threadgroup=(16, 16, 1),
    )[0]


def main() -> None:
    report()
    peak = peak_bandwidth_gbs()
    print(f"\nMeasured 1R:1W streaming ceiling: {peak:.1f} GB/s")

    a = mx.random.normal(shape=(N,), dtype=mx.float32)
    mx.eval(a)

    # ------------------------------------------------------------------ Part A
    header("A. Cache-line amplification — same lines fetched, less data used")
    print(f"  Array: {N * 4 / 2**20:.0f} MiB = {N * 4 // 128:,} cache lines of 128 B.")
    print("  Each thread reads ONE element at index `t * stride`, so every stride in")
    print("  1..32 touches the same 524288 lines. Only the *useful* fraction changes.\n")
    print(f"  {'stride':>6} {'useful MiB':>11} {'time ms':>9} {'useful GB/s':>12} {'fetched GB/s':>13}")
    rows_a = []
    for stride in (1, 2, 4, 8, 16, 32):
        n_out = N // stride
        t = bench(lambda s=stride: mx.eval(spaced_read(a, s)))
        useful_bytes = n_out * 4
        useful_bw = bandwidth_gbs(useful_bytes, t.median)
        fetched_bw = bandwidth_gbs(N * 4, t.median)
        rows_a.append((stride, useful_bw, fetched_bw, t.median))
        print(
            f"  {stride:>6} {useful_bytes / 2**20:>11.1f} {t.median:>9.4f} "
            f"{useful_bw:>12.1f} {fetched_bw:>13.1f}"
        )
    print(
        f"\n  Useful bandwidth falls {rows_a[0][1] / rows_a[-1][1]:.0f}x from stride 1 to 32,"
        f" while the fetch rate stays in the same ballpark\n"
        f"  ({rows_a[0][2]:.0f} -> {rows_a[-1][2]:.0f} GB/s) and the *time* barely moves. The"
        " hardware did the same work;\n  the program just stopped using what it paid for."
    )

    # ------------------------------------------------------------------ Part B
    rows = cols = 4096
    m = mx.random.normal(shape=(rows, cols), dtype=mx.float32)
    mx.eval(m)
    expected = np.array(m).T
    total_bytes = 2 * rows * cols * 4

    header(f"B. Transpose {rows}x{cols} ({total_bytes / 2**20:.0f} MiB of traffic)")
    assert_close(transpose_naive(m), expected, label="transpose_naive")
    t_naive = bench(lambda: mx.eval(transpose_naive(m)))
    report_row("naive (scattered writes)", t_naive, total_bytes, peak_bw=peak)

    assert_close(transpose_tiled(m), expected, label="transpose_tiled (32x33)")
    t_tiled = bench(lambda: mx.eval(transpose_tiled(m)))
    report_row("tiled 32x33, 256 thr, 4/t", t_tiled, total_bytes, peak_bw=peak)

    assert_close(transpose_big_tile(m), expected, label="transpose_big (64x65)")
    t_big = bench(lambda: mx.eval(transpose_big_tile(m)))
    report_row("tiled 64x65, 256 thr (too big)", t_big, total_bytes, peak_bw=peak)

    # Ragged-ish shapes: the guards in the naive kernel are what make this safe.
    odd = mx.random.normal(shape=(4096, 3072), dtype=mx.float32)
    mx.eval(odd)
    assert_close(transpose_naive(odd), np.array(odd).T, label="naive (non-square)")
    assert_close(transpose_tiled(odd), np.array(odd).T, label="tiled (non-square)")
    assert_close(transpose_big_tile(odd), np.array(odd).T, label="big tile (non-square)")

    header("Result")
    print(
        f"  tiled 32x33 is {t_naive.median / t_tiled.median:.2f}x faster than naive "
        f"({t_naive.median:.3f} -> {t_tiled.median:.3f} ms)\n"
        f"  while moving exactly the same useful bytes.\n\n"
        f"  The 64x65 variant is {t_big.median / t_tiled.median:.2f}x SLOWER than the "
        f"32x33 one. The transpose logic is\n"
        f"  identical; it just asks for 16.6 KiB of the 32 KiB threadgroup budget, which\n"
        f"  leaves room for only one resident threadgroup and no latency hiding."
    )

    header("What to take away")
    print(
        """
  1. Separate "useful" from "fetched" bytes in your head. A kernel at 2 GB/s useful
     may be saturating a 64 GB/s memory system. Optimising such a kernel is not about
     making memory faster; it is about touching the bytes you asked for.

  2. Coalescing is about what one simdgroup does in one instruction. 32 lanes on 32
     consecutive float32s = 1 line. 32 lanes on addresses 128 B apart = 32 lines.

  3. Apple Silicon's large L2 hides a lot of access-pattern sin. The classic tiled
     transpose wins here, but only by ~1.1-1.5x depending on the run -- not by the 10x
     the NVIDIA literature shows. Measure your own hardware; do not port conclusions.

  4. threadgroup memory is the standard way to re-order data between two global
     accesses that cannot both be coalesced. It is also strictly limited (32 KiB on
     this GPU, verified via tools/device_probe.swift) and it competes with L1, so
     using more of it costs occupancy. The 64x65 tile above is that tradeoff failing.

  5. Bank conflicts are the on-chip version of the same problem, and the +1 padding in
     `float tile[32][33]` is the standard fix. It is not superstition.

  6. `threadgroup_barrier(mem_flags::mem_threadgroup)` is `__syncthreads()`. Missing
     barriers produce results that are *sometimes* right and fail under different
     occupancy — the worst kind of bug.

  Where this leads: the 32x33 tile is the direct ancestor of the tiled matmul in
  lab04. Same idea, one more loop, and the payoff becomes enormous because matmul
  reuses each loaded value many times instead of once.
"""
    )


if __name__ == "__main__":
    main()
