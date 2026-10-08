"""Lab 04 — Matmul: tiling, arithmetic intensity, and the roofline.

Run:  python labs/lab04_matmul.py

Goal
----
Take matrix multiply from a naive kernel that is 10x off the hardware to a tiled,
register-blocked kernel that is competitive with Apple's own GEMM — and understand
*why* each step works, using arithmetic intensity rather than intuition.

Why matmul is different from labs 01-03
---------------------------------------
Vector add and reduction are bandwidth bound: they do ~1 FLOP per byte moved, so no
amount of cleverness beats the memory system. Matmul does ``2*M*N*K`` FLOPs while
moving ``4*(M*K + K*N + M*N)`` bytes. For 1024x1024 that is:

    arithmetic intensity = 2*1024^3 / (4*3*1024^2) = 171 FLOP/byte

At ~90 GB/s and ~2400 GFLOP/s this machine's crossover point is about 27 FLOP/byte.
Anything above that is **compute bound** — meaning memory optimisations are pointless
and the only thing that matters is how many FLOPs per clock each thread can issue.
Matmul is far above the line, which is why it is the first kernel where on-chip
reuse (tiling, registers) turns into orders of magnitude.

The progression
---------------
  v1 naive        one thread per output, K-loop reads global memory every iteration
  v2 tiled        16x16 threadgroup tiles: each element loaded from global memory
                  once per tile instead of once per use
  v3 + registers  32x32 tile with a 2x2 register micro-tile per thread: each value
                  loaded from threadgroup memory several times, reused in registers

The roofline
------------
Every kernel below is plotted against the two measured ceilings. A kernel can never
be above the roofline; the interesting question is which segment you are under and
therefore which resource to attack. This lab makes that concrete.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from gpuk import (
    assert_close,
    gflops,
    header,
    report,
    report_row,
)
from gpuk.bench import best_ms
from gpuk.peaks import peak_flops_fp32, peak_bandwidth_gbs

M = N = K = 1024
FLOPS = 2.0 * M * N * K
BYTES = 4 * (M * K + K * N + M * N)
AI = FLOPS / BYTES


def bench(fn):
    return best_ms(fn, rounds=3, warmup=5, iters=20)


# ======================================================================================
# v1 — naive. One thread per output element.
#
# Thread (row, col) walks the whole K dimension reading A and B from GLOBAL memory on
# every iteration. Each A[row, k] value is re-fetched by all N=1024 threads in the row;
# each B[k, col] by all M threads in the column. Total global traffic is O(M*N*K)
# instead of O(M*K + K*N + M*N) — a factor of ~K/3 = 341x more than necessary.
#
# The L2 cache rescues it partially (neighbouring threads in a warp share B, and
# neighbouring warps share A), which is exactly why the naive kernel is not 300x
# slower — but it is still far off.
#
# Note: `row` and `col` come from a 2D grid. grid.x spans columns (the fast axis),
# so consecutive lanes read consecutive B elements -> coalesced.
# ======================================================================================
V1_NAIVE_SRC = """
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (row < M[0] && col < Ncols[0]) {
        float acc = 0.0f;
        for (uint k = 0; k < K[0]; ++k) {
            acc += A[row * K[0] + k] * B[k * Ncols[0] + col];
        }
        C[row * Ncols[0] + col] = acc;
    }
"""

kernel_v1 = mx.fast.metal_kernel(
    name="mm_v1_naive",
    input_names=["A", "B", "M", "K", "Ncols"],
    output_names=["C"],
    source=V1_NAIVE_SRC,
)


def matmul_naive(a: mx.array, b: mx.array) -> mx.array:
    m, k = a.shape
    _, n = b.shape
    return kernel_v1(
        inputs=[a, b, mx.array([m], mx.uint32), mx.array([k], mx.uint32), mx.array([n], mx.uint32)],
        output_shapes=[(m, n)],
        output_dtypes=[a.dtype],
        grid=(n, m, 1),
        threadgroup=(16, 16, 1),
    )[0]


# ======================================================================================
# v2 — tiled, 16x16 threadgroup tiles.
#
# Each threadgroup computes one 16x16 output tile. Per k-step it loads a 16x16 slab of
# A and a 16x16 slab of B into threadgroup memory (2 KiB each), synchronises, then
# accumulates from on-chip memory.
#
# The win: each global value is now loaded once per *tile* rather than once per *use*.
# The 256 threads of the threadgroup each perform 16 FMAs per slab, but the slab costs
# only 256 loads per array. Global traffic drops by roughly the tile width (16x).
#
# Cost: 2 barriers per k-step, and K/16 = 64 k-steps at K=1024 -> 128 barriers. This
# is where threadgroup memory sizing becomes a real design decision.
# ======================================================================================
V2_TILED_SRC = """
    threadgroup float As[16][16];
    threadgroup float Bs[16][16];

    uint tx = thread_index_in_threadgroup % 16;
    uint ty = thread_index_in_threadgroup / 16;
    uint blocks_per_row = Ncols[0] / 16;
    uint bid = threadgroup_position_in_grid.x;

    uint row = (bid / blocks_per_row) * 16 + ty;
    uint col = (bid % blocks_per_row) * 16 + tx;

    float acc = 0.0f;
    for (uint k0 = 0; k0 < K[0]; k0 += 16) {
        As[ty][tx] = A[row * K[0] + k0 + tx];
        Bs[ty][tx] = B[(k0 + ty) * Ncols[0] + col];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 16; ++k) acc += As[ty][k] * Bs[k][tx];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[row * Ncols[0] + col] = acc;
"""

kernel_v2 = mx.fast.metal_kernel(
    name="mm_v2_tiled",
    input_names=["A", "B", "M", "K", "Ncols"],
    output_names=["C"],
    source=V2_TILED_SRC,
)


def matmul_tiled(a: mx.array, b: mx.array) -> mx.array:
    m, k = a.shape
    _, n = b.shape
    if m % 16 or n % 16 or k % 16:
        raise ValueError("tiled matmul needs dimensions that are multiples of 16")
    blocks = (m // 16) * (n // 16)
    return kernel_v2(
        inputs=[a, b, mx.array([m], mx.uint32), mx.array([k], mx.uint32), mx.array([n], mx.uint32)],
        output_shapes=[(m, n)],
        output_dtypes=[a.dtype],
        grid=(blocks * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]


# ======================================================================================
# v3 — 32x32 tile, 16x16 threads, 2x2 register micro-tile per thread.
#
# Two more ideas, both about reuse:
#
#   1. **Bigger tile.** A 32-wide tile halves the number of global loads per output
#      element compared with 16. 32x32 floats = 4 KiB per slab, 8 KiB for both, well
#      inside the 32 KiB threadgroup budget.
#
#   2. **Register blocking.** Each thread computes a 2x2 block of outputs. For each k
#      it loads ONE value of A and ONE value of B into registers and performs FOUR
#      FMAs with them. The ratio of FMAs to threadgroup loads doubles. Pushing this
#      further (4x4, 8x8 micro-tiles) is exactly how real GEMM kernels are built, and
#      it is why the best GEMMs use a large fraction of the register file.
#
# Each thread loads 2 values per slab instead of 4, so the load:FMA ratio is 4:4
# instead of 4:2 in the untiled variant. On Apple GPUs the compiler can also map the
# FMA pairs onto `simdgroup_matrix`-like packed operations if the tile shapes allow.
# ======================================================================================
V3_REGISTER_SRC = """
    threadgroup float As[32][32];
    threadgroup float Bs[32][32];

    uint lane = thread_index_in_threadgroup;
    uint tx = lane % 16;             // 16x16 = 256 threads
    uint ty = lane / 16;

    uint blocks_per_row = Ncols[0] / 32;
    uint bid = threadgroup_position_in_grid.x;
    uint block_row = bid / blocks_per_row;
    uint block_col = bid % blocks_per_row;

    // This thread owns output rows (block_row*32 + ty*2, +1) and
    // columns (block_col*32 + tx*2, +1).
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    for (uint k0 = 0; k0 < K[0]; k0 += 32) {
        // Load a 32x32 slab of A and B: 1024 values each, 4 per thread.
        for (uint s = 0; s < 4; ++s) {
            uint idx = s * 256 + lane;
            uint r = idx / 32;
            uint c = idx % 32;
            As[r][c] = A[(block_row * 32 + r) * K[0] + k0 + c];
            Bs[r][c] = B[(k0 + r) * Ncols[0] + block_col * 32 + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 32; ++k) {
            float a0 = As[ty * 2 + 0][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2 + 0];
            float b1 = Bs[k][tx * 2 + 1];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint crow = block_row * 32 + ty * 2;
    uint ccol = block_col * 32 + tx * 2;
    C[crow * Ncols[0] + ccol] = c00;
    C[crow * Ncols[0] + ccol + 1] = c01;
    C[(crow + 1) * Ncols[0] + ccol] = c10;
    C[(crow + 1) * Ncols[0] + ccol + 1] = c11;
"""

kernel_v3 = mx.fast.metal_kernel(
    name="mm_v3_registers",
    input_names=["A", "B", "M", "K", "Ncols"],
    output_names=["C"],
    source=V3_REGISTER_SRC,
)


def matmul_register_blocked(a: mx.array, b: mx.array) -> mx.array:
    m, k = a.shape
    _, n = b.shape
    if m % 32 or n % 32 or k % 32:
        raise ValueError("register-blocked matmul needs dimensions that are multiples of 32")
    blocks = (m // 32) * (n // 32)
    return kernel_v3(
        inputs=[a, b, mx.array([m], mx.uint32), mx.array([k], mx.uint32), mx.array([n], mx.uint32)],
        output_shapes=[(m, n)],
        output_dtypes=[a.dtype],
        grid=(blocks * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]


def main() -> None:
    report()
    peak_bw = peak_bandwidth_gbs()
    peak_flops = peak_flops_fp32()
    crossover = peak_flops / peak_bw
    print(
        f"\nMeasured ceilings: {peak_bw:.1f} GB/s, {peak_flops:.0f} GFLOP/s fp32"
        f"\nRoofline crossover: {crossover:.1f} FLOP/byte"
    )
    print(
        f"\nThis problem: {M}x{N}x{K}, arithmetic intensity = {AI:.1f} FLOP/byte"
        f"  -> {'COMPUTE bound' if AI > crossover else 'MEMORY bound'} "
        f"({AI / crossover:.1f}x above the crossover)"
    )

    a = mx.random.normal(shape=(M, K), dtype=mx.float32)
    b = mx.random.normal(shape=(K, N), dtype=mx.float32)
    mx.eval(a, b)
    reference = np.array(a).astype(np.float64) @ np.array(b).astype(np.float64)

    # Tolerance note: a K=1024 float32 dot product accumulates 1024 roundings, and the
    # tiled versions sum in a different order than the reference. The differences are
    # real float32 behaviour, not bugs. Compare against a *float64* reference with a
    # relative tolerance, never for bit equality.
    header("Correctness (vs float64 reference)")
    assert_close(matmul_naive(a, b), reference, rtol=1e-4, atol=1e-2, label="v1 naive")
    assert_close(matmul_tiled(a, b), reference, rtol=1e-4, atol=1e-2, label="v2 tiled 16x16")
    assert_close(
        matmul_register_blocked(a, b), reference, rtol=1e-4, atol=1e-2, label="v3 32x32 + 2x2 regs"
    )

    header("Benchmarks")
    print(f"  {'kernel':<34} {'time':>10}  {'GFLOP/s':>10}  {'% of fp32 peak':>15}")
    report_row("mx.matmul (Apple's tuned GEMM)", bench(lambda: mx.eval(mx.matmul(a, b))), None, FLOPS, peak_flops=peak_flops)
    report_row("v1 naive", bench(lambda: mx.eval(matmul_naive(a, b))), None, FLOPS, peak_flops=peak_flops)
    report_row("v2 tiled 16x16", bench(lambda: mx.eval(matmul_tiled(a, b))), None, FLOPS, peak_flops=peak_flops)
    report_row(
        "v3 32x32 tile + 2x2 registers",
        bench(lambda: mx.eval(matmul_register_blocked(a, b))),
        None,
        FLOPS,
        peak_flops=peak_flops,
    )

    header("Scaling: how each version behaves as the problem grows")
    print(f"  {'size':>6}  {'v1 naive':>12}  {'v2 tiled':>12}  {'v3 registers':>13}  {'mx.matmul':>11}    (GFLOP/s)")
    for size in (256, 512, 1024):
        x = mx.random.normal(shape=(size, size), dtype=mx.float32)
        y = mx.random.normal(shape=(size, size), dtype=mx.float32)
        mx.eval(x, y)
        fl = 2.0 * size**3
        t1 = bench(lambda x=x, y=y: mx.eval(matmul_naive(x, y))).median
        t2 = bench(lambda x=x, y=y: mx.eval(matmul_tiled(x, y))).median
        t3 = bench(lambda x=x, y=y: mx.eval(matmul_register_blocked(x, y))).median
        tm = bench(lambda x=x, y=y: mx.eval(mx.matmul(x, y))).median
        print(
            f"  {size:>6}  {gflops(fl, t1):>12.1f}  {gflops(fl, t2):>12.1f}  "
            f"{gflops(fl, t3):>13.1f}  {gflops(fl, tm):>11.1f}"
        )

    header("What to take away")
    print(
        """
  1. Tiling works because of *reuse*, not because "shared memory is fast". Loading a
     value once into threadgroup memory and using it 16 or 32 times removes that many
     global loads. The metric to reason with is arithmetic intensity (FLOP/byte), and
     the roofline tells you whether it is even worth trying.

  2. Register blocking is the next leverage point after tiling, and usually the bigger
     one. One threadgroup load feeding four FMAs instead of one is a 4x reduction in
     on-chip traffic. Real GEMM kernels push to 8x8 micro-tiles and spend most of the
     register file on it.

  3. Barriers are the tax on tiling. Every k-step needs two, so a small tile means many
     barriers. Tile size is a three-way tradeoff between global traffic, threadgroup
     memory (32 KiB total) and barrier count.

  4. Apple's `mx.matmul` uses the GPU's `simdgroup_matrix` units, which are the closest
     thing this hardware has to the tensor cores PMPP's later chapters are about. A
     plain-FMA kernel like v3 will not catch it, and understanding *why* (different
     execution units, not a better algorithm) is the point.

  5. Float32 matmul is not associative. Your kernel and the reference genuinely produce
     different bits, and the difference grows with K. Always compare against a float64
     reference with a relative tolerance.

  6. This is the kernel shape that everything in modern ML reduces to. FlashAttention is
     a tiled matmul with a softmax folded into the accumulation; a fused MLP is a
     matmul with an activation in the epilogue. Get tiling and register blocking into
     your fingers here, and those kernels stop looking like magic.
"""
    )


if __name__ == "__main__":
    main()
