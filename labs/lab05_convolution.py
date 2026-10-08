"""Lab 05 — Convolution: the halo, and when tiling actually pays.

Run:  python labs/lab05_convolution.py

Goal
----
Write the halo-tiled stencil kernel, then find out *empirically* at which kernel size
tiling starts to win. The answer is not the one the CUDA literature leads you to expect,
and the reason is worth understanding.

The theory you are usually taught
---------------------------------
A (2K+1)x(2K+1) stencil needs (2K+1)^2 global reads per output pixel, and neighbouring
outputs read almost the same pixels. Tiling an image block through threadgroup memory
should turn that into roughly one read per pixel:

    naive:  (2K+1)^2 x image read + 1 x image write
    tiled:  ((T + 2K)^2 / T^2) x image read + 1 x image write     <- the halo overhead

For K=1, T=32: 9x vs 1.13x. An 8x traffic reduction. Sounds decisive.

What actually happens here
--------------------------
A warp's 9 stencil reads for K=1 span 3 rows x 34 columns — about 400 bytes, roughly four
128-byte cache lines. The redundancy is *intra-warp and local*, which is precisely what an
L1 cache exists to absorb. So the naive kernel never pays the 9x in DRAM traffic, and
tiling therefore buys nothing for a small kernel.

This lab measures three kernels across K = 1, 2, 4, 8 so you can see where the picture
changes, and includes a plain copy of the same volume as the achievable floor—so you can
tell "slow because of traffic" apart from "slow because of instruction count".

The three variants
------------------
  v1 naive          image and filter both read from global memory
  v2 naive + filter filter staged in threadgroup memory; PMPP's "put the filter in
                    constant memory" advice, implemented with the facility Metal gives you
  v3 tiled + halo   image staged with a halo and filter staged; isolates the *image* reuse

Both hypotheses get tested and only one survives. Tiling the image is worth about 1.6x
for every K >= 2 and nothing at all for K = 1. Staging the filter — which the CUDA
literature is unanimous about — measures as a slight *loss* here (0.89-1.00x), because the
compiler already hoists loop-invariant loads and the remaining filter reads are served by
cache broadcast. That is the third time in this repo that a confident piece of CUDA advice
failed to reproduce on this hardware, and it is why every lab ends with a measurement
rather than a rule.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from gpuk import (
    assert_close,
    header,
    report,
    report_row,
)
from gpuk.bench import best_ms
from gpuk.peaks import peak_bandwidth_gbs

H = W = 2048
TILE = 32
THREADS = 256
KS = (1, 2, 4, 8)


def bench(fn):
    return best_ms(fn, rounds=3, warmup=5, iters=20)


def ramp_filter(k: int) -> np.ndarray:
    """A separable, **asymmetric** (2K+1)^2 kernel.

    The outer product of a ramp and its reverse: rank-1, hence exactly separable, but not
    symmetric. Asymmetry matters for testing: with a symmetric filter (a Gaussian, a box
    blur) a transposed or axis-swapped index error produces *the same output*, so the bug
    hides. With this one it shows up immediately. Any real stencil test suite needs at
    least one asymmetric kernel for that reason.
    """
    ramp = np.arange(1, 2 * k + 2, dtype=np.float64)
    w = np.outer(ramp, ramp[::-1])
    return (w / w.sum()).astype(np.float32)


# Keep the old name working: it is referenced in prose and in earlier revisions.
pyramid_filter = ramp_filter


def launch_for(h: int, w: int) -> dict:
    """Launch geometry for the untiled kernels, at least one full threadgroup.

    **A threadgroup barrier with a partially-dispatched threadgroup is undefined
    behaviour.** A 1x1 image would naturally launch grid=(1,1) with threadgroup=(32,8),
    and the `threadgroup_barrier` in the filter-staging kernel would then wait for 255
    threads that do not exist. Launching a full threadgroup and guarding with
    `if (row < H && col < W)` is the fix — and it is the same guard you already need for
    the ragged edges.

    This is an easy bug to ship: it only fires on images smaller than one tile, which a
    2048x2048 test set never produces.
    """
    return {"grid": (max(w, 32), max(h, 8), 1), "threadgroup": (32, 8, 1)}


def reference_conv(img: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """Zero-padded convolution, accumulated in float64."""
    h, w = img.shape
    k = (filt.shape[0] - 1) // 2
    padded = np.pad(img.astype(np.float64), k)
    out = np.zeros((h, w), dtype=np.float64)
    for dy in range(2 * k + 1):
        for dx in range(2 * k + 1):
            out += filt[dy, dx] * padded[dy : dy + h, dx : dx + w]
    return out


# ======================================================================================
# v1 — naive. One thread per output pixel; image AND filter read from global memory.
#
# Note `filt[...]` inside the innermost loop: (2K+1)^2 global loads per output pixel,
# all of them for the same handful of values. For K=1 that is 9 loads of 36 bytes; for
# K=8 it is 289 loads of 1156 bytes. The compiler may hoist some of these, but it cannot
# hoist them all into registers when the filter is large.
# ======================================================================================
NAIVE_SRC = """
    int H = int(inp_shape[0]);
    int W = int(inp_shape[1]);
    int row = int(thread_position_in_grid.y);
    int col = int(thread_position_in_grid.x);

    if (row < H && col < W) {
        float acc = 0.0f;
        for (int dy = -K; dy <= K; ++dy) {
            for (int dx = -K; dx <= K; ++dx) {
                int y = row + dy;
                int x = col + dx;
                if (y >= 0 && y < H && x >= 0 && x < W) {
                    acc += inp[y * W + x] * filt[(dy + K) * (2 * K + 1) + (dx + K)];
                }
            }
        }
        out[row * W + col] = acc;
    }
"""

kernel_naive = mx.fast.metal_kernel(
    name="conv_naive",
    input_names=["inp", "filt"],
    output_names=["out"],
    source=NAIVE_SRC,
)


def conv_naive(img: mx.array, filt: mx.array, k: int) -> mx.array:
    h, w = img.shape
    return kernel_naive(
        inputs=[img, filt],
        template=[("K", k)],
        output_shapes=[(h, w)],
        output_dtypes=[img.dtype],
        **launch_for(h, w),
    )[0]


# ======================================================================================
# v2 — naive image access, but the filter is staged in threadgroup memory.
#
# This is PMPP's "__constant__ memory for the filter" advice, implemented with the
# facility Metal actually gives you. It exists to test a hypothesis: how much of the
# naive kernel's cost is the filter reads, which look like pure overhead?
#
# Measured answer: essentially none. See the K sweep in main(). The compiler hoists
# loop-invariant loads, and the remaining filter accesses are a broadcast out of L1 that
# costs nothing next to the image traffic. The cooperative load plus barrier actually
# makes it marginally slower. Kept here because negative results are results.
# ======================================================================================
NAIVE_SMEM_SRC = """
    threadgroup float f[(2 * K + 1) * (2 * K + 1)];
    uint tid = thread_index_in_threadgroup;
    for (uint i = tid; i < uint((2 * K + 1) * (2 * K + 1)); i += 256u) {
        f[i] = filt[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int H = int(inp_shape[0]);
    int W = int(inp_shape[1]);
    int row = int(thread_position_in_grid.y);
    int col = int(thread_position_in_grid.x);

    if (row < H && col < W) {
        float acc = 0.0f;
        for (int dy = -K; dy <= K; ++dy) {
            for (int dx = -K; dx <= K; ++dx) {
                int y = row + dy;
                int x = col + dx;
                if (y >= 0 && y < H && x >= 0 && x < W) {
                    acc += inp[y * W + x] * f[(dy + K) * (2 * K + 1) + (dx + K)];
                }
            }
        }
        out[row * W + col] = acc;
    }
"""

kernel_naive_smem = mx.fast.metal_kernel(
    name="conv_naive_filter_smem",
    input_names=["inp", "filt"],
    output_names=["out"],
    source=NAIVE_SMEM_SRC,
)


def conv_naive_smem(img: mx.array, filt: mx.array, k: int) -> mx.array:
    h, w = img.shape
    return kernel_naive_smem(
        inputs=[img, filt],
        template=[("K", k)],
        output_shapes=[(h, w)],
        output_dtypes=[img.dtype],
        **launch_for(h, w),
    )[0]


# ======================================================================================
# v3 — the halo-tiled stencil.
#
# A threadgroup owns one 32x32 output tile and stages the (32+2K)^2 input region it
# needs, zero-padded at the image edges. Both the image tile and the filter live in
# threadgroup memory, so the compute loop touches no global memory at all.
#
# Three details that matter:
#
#  1. **The halo is zero-padded.** `base = block*32 - K` can be negative. Getting this
#     wrong shows up only at the image borders — and a 2048x2048 image (a multiple of the
#     tile) will pass even with the bounds check deleted. The tests below use 100x130,
#     37x37 and 1x1 for exactly that reason.
#
#  2. **256 threads for a 1024-pixel tile.** 32x8 threads cannot cover 32x32 in one step,
#     so each thread walks 4 rows. Keeping the threadgroup at 256 rather than 1024 is what
#     lets several threadgroups be resident and hides the barrier. A 32x32 threadgroup
#     would use the whole Apple limit and leave no room for a second resident group.
#
#  3. **threadgroup budget.** (32+2K)^2 floats plus (2K+1)^2 floats. For K=8 that is
#     9216 + 1156 = 10.4 KiB of the 32 KiB available. Push K much further and occupancy
#     collapses — the same trap as lab02's 64x65 tile.
# ======================================================================================
TILED_SRC = """
    threadgroup float f[(2 * K + 1) * (2 * K + 1)];
    threadgroup float tile[32 + 2 * K][32 + 2 * K];

    const uint TW = uint(32 + 2 * K);
    uint tid = thread_index_in_threadgroup;
    uint tx = tid % 32u;
    uint ty = tid / 32u;                       // 0..7
    uint bx = threadgroup_position_in_grid.x;
    uint by = threadgroup_position_in_grid.y;

    int H = int(inp_shape[0]);
    int W = int(inp_shape[1]);
    int base_y = int(by) * 32 - K;
    int base_x = int(bx) * 32 - K;

    // Stage the filter.
    for (uint i = tid; i < uint((2 * K + 1) * (2 * K + 1)); i += 256u) {
        f[i] = filt[i];
    }

    // Stage the image tile with its halo, zero-padded outside the image.
    for (uint i = tid; i < TW * TW; i += 256u) {
        uint r = i / TW;
        uint c = i % TW;
        int gy = base_y + int(r);
        int gx = base_x + int(c);
        tile[r][c] = (gy >= 0 && gy < H && gx >= 0 && gx < W)
                         ? inp[gy * W + gx]
                         : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1024 outputs per tile / 256 threads = 4 each, 8 rows apart.
    for (uint j = 0; j < 4u; ++j) {
        uint ry = ty + j * 8u;
        uint oy = by * 32u + ry;
        uint ox = bx * 32u + tx;
        if (oy < uint(H) && ox < uint(W)) {
            float acc = 0.0f;
            for (int dy = 0; dy < 2 * K + 1; ++dy) {
                for (int dx = 0; dx < 2 * K + 1; ++dx) {
                    acc += tile[ry + uint(dy)][tx + uint(dx)]
                         * f[dy * (2 * K + 1) + dx];
                }
            }
            out[oy * uint(W) + ox] = acc;
        }
    }
"""

kernel_tiled = mx.fast.metal_kernel(
    name="conv_tiled_halo",
    input_names=["inp", "filt"],
    output_names=["out"],
    source=TILED_SRC,
)


def conv_tiled(img: mx.array, filt: mx.array, k: int) -> mx.array:
    h, w = img.shape
    blocks_x = (w + TILE - 1) // TILE
    blocks_y = (h + TILE - 1) // TILE
    return kernel_tiled(
        inputs=[img, filt],
        template=[("K", k)],
        output_shapes=[(h, w)],
        output_dtypes=[img.dtype],
        grid=(blocks_x * 32, blocks_y * 8, 1),
        threadgroup=(32, 8, 1),
    )[0]


# ======================================================================================
# The floor: a plain 2D copy of the same volume.
#
# This is the reference point that makes the rest of the lab interpretable. If a stencil
# kernel is no slower than a copy of the same number of bytes, it is not paying for
# traffic at all — it is paying for launch, issue and latency, and tiling cannot help.
# ======================================================================================
COPY_SRC = """
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    if (x < uint(inp_shape[1]) && y < uint(inp_shape[0])) {
        out[y * uint(inp_shape[1]) + x] = inp[y * uint(inp_shape[1]) + x];
    }
"""

kernel_copy = mx.fast.metal_kernel(
    name="copy_floor",
    input_names=["inp"],
    output_names=["out"],
    source=COPY_SRC,
)


def copy_floor(img: mx.array) -> mx.array:
    h, w = img.shape
    return kernel_copy(
        inputs=[img],
        output_shapes=[(h, w)],
        output_dtypes=[img.dtype],
        **launch_for(h, w),
    )[0]


def main() -> None:
    report()
    peak = peak_bandwidth_gbs()

    img_np = np.random.default_rng(0).normal(size=(H, W)).astype(np.float32)
    img = mx.array(img_np)
    mx.eval(img)
    img_bytes = H * W * 4

    print(f"\nImage {H}x{W} float32 = {img_bytes / 2**20:.0f} MiB, tile {TILE}x{TILE}")

    header("The floor: what a plain copy of this volume costs")
    t_copy = bench(lambda: mx.eval(copy_floor(img)))
    report_row("copy (1 read + 1 write)", t_copy, 2 * img_bytes, peak_bw=peak)
    print(
        f"\n  Take this number seriously. A kernel that only reads and writes the image\n"
        f"  takes {t_copy.median:.3f} ms here — far from the {peak:.0f} GB/s streaming ceiling,\n"
        f"  because with one load per thread the dispatch is latency/issue bound, not\n"
        f"  bandwidth bound. Any stencil measured at or below this is not paying for\n"
        f"  traffic, and no amount of tiling will improve it."
    )

    # ---------------------------------------------------------------- correctness
    header("Correctness across kernel sizes (and ragged shapes)")
    for k in KS:
        f = pyramid_filter(k)
        mx_f = mx.array(f.reshape(-1))
        mx.eval(mx_f)
        ref = reference_conv(img_np, f)
        assert_close(conv_naive(img, mx_f, k), ref, rtol=1e-4, atol=1e-4, label=f"K={k} naive")
        assert_close(conv_naive_smem(img, mx_f, k), ref, rtol=1e-4, atol=1e-4, label=f"K={k} naive+smem")
        assert_close(conv_tiled(img, mx_f, k), ref, rtol=1e-4, atol=1e-4, label=f"K={k} tiled")

    for shape in ((100, 130), (37, 37), (1, 1)):
        small = np.random.default_rng(1).normal(size=shape).astype(np.float32)
        mx_small = mx.array(small)
        mx.eval(mx_small)
        for k in (1, 2, 4):
            f = pyramid_filter(k)
            mx_f = mx.array(f.reshape(-1))
            mx.eval(mx_f)
            expected = reference_conv(small, f)
            assert_close(
                conv_naive(mx_small, mx_f, k), expected, rtol=1e-4, atol=1e-4,
                label=f"naive K={k} {shape}", verbose=False,
            )
            assert_close(
                conv_tiled(mx_small, mx_f, k), expected, rtol=1e-4, atol=1e-4,
                label=f"tiled K={k} {shape}", verbose=False,
            )
        print(f"  [PASS] ragged {str(shape):<12} all variants, K=1,2,4 (halo bounds exercised)")

    # ---------------------------------------------------------------- K sweep
    header("The K sweep: where does tiling start to pay?")
    print(
        f"  {'K':>3} {'taps':>5} {'logical traffic':>16} {'copy':>9} "
        f"{'naive':>9} {'naive+f smem':>13} {'tiled+halo':>11}"
    )
    print(
        f"  {'':>3} {'':>5} {'(x image read)':>16} {'ms':>9} {'ms':>9} {'ms':>13} {'ms':>11}"
    )
    results = {}
    for k in KS:
        f = pyramid_filter(k)
        mx_f = mx.array(f.reshape(-1))
        mx.eval(mx_f)
        taps = (2 * k + 1) ** 2
        t1 = bench(lambda f=mx_f, k=k: mx.eval(conv_naive(img, f, k)))
        t2 = bench(lambda f=mx_f, k=k: mx.eval(conv_naive_smem(img, f, k)))
        t3 = bench(lambda f=mx_f, k=k: mx.eval(conv_tiled(img, f, k)))
        results[k] = (t1.median, t2.median, t3.median)
        print(
            f"  {k:>3} {taps:>5} {taps:>15}x {t_copy.median:>9.3f} "
            f"{t1.median:>9.3f} {t2.median:>13.3f} {t3.median:>11.3f}"
        )

    print("\n  Speedup of the two fixes over the naive kernel:")
    print(f"  {'K':>3} {'filter in threadgroup':>22} {'tiled + halo':>14} {'logical traffic cut':>21}")
    for k in KS:
        t1, t2, t3 = results[k]
        halo = ((TILE + 2 * k) ** 2) / (TILE**2)
        print(
            f"  {k:>3} {t1 / t2:>21.2f}x {t1 / t3:>13.2f}x "
            f"{(2 * k + 1) ** 2:>18.1f}x -> {halo:.2f}x"
        )

    header("What this tells you")
    worst = max(KS, key=lambda k: results[k][0])
    print(
        f"""
  * **K=1 is not a memory problem at all.** The naive 3x3 stencil costs the same as a
    plain copy of the image ({results[1][0]:.3f} vs {t_copy.median:.3f} ms). The "9x read
    redundancy" is intra-warp and spans about four cache lines, so L1 absorbs it and the
    kernel is bound by dispatch latency, not traffic. Tiling it is wasted code.

  * **For K >= 2 the naive kernel really does pay for its reads.** Its cost grows with the
    tap count -- {results[2][0]:.2f} ms at K=2 (25 taps), {results[4][0]:.2f} ms at K=4 (81),
    {results[8][0]:.2f} ms at K=8 (289). Tiling then wins by a consistent
    {results[2][0] / results[2][2]:.2f}-{results[8][0] / results[8][2]:.2f}x, and the absolute saving at K=8 is
    {(results[8][0] - results[8][2]) / 1000:.1f} seconds per hundred calls.

  * **But the speedup is nowhere near the traffic ratio.** The table says the logical read
    traffic drops {289}x at K=8 (289x -> 2.25x); the measured gain is
    {results[8][0] / results[8][2]:.2f}x. L1 and L2 are still capturing most of the
    redundancy, so the only redundancy that shows up in the wall clock is the part that
    crosses *warp* boundaries. Logical traffic is an upper bound on the win, never a
    prediction of it.

  * **Staging the filter did not help** -- it was neutral at K=1 and slightly negative
    above ({results[4][0] / results[4][1]:.2f}x at K=4). PMPP is right that reading an
    unchanging (2K+1)^2-element filter from global memory *looks* wasteful; it is wrong
    that fixing it matters on this machine, because the compiler hoists the loop-invariant
    loads and what remains is a cached broadcast. The cooperative load and its barrier cost
    more than they save. Keep the advice in mind for genuinely large filters; do not apply
    it reflexively.

  * The practical rule from this lab: **tile the image, skip the filter staging, and only
    bother at all for K >= 2.** Then measure on your own hardware, because all three of
    those clauses are empirical claims about this GPU.

  * Repeated reminder from labs 02 and 03: the CUDA literature describes hardware with a
    smaller L1, a much wider memory interface and a different cache hierarchy. Port the
    *method* -- measure, compute the redundancy, compare against a floor -- never the
    conclusion.
"""
    )

    header("Halo arithmetic: the cost of tiling, exactly")
    print(f"  {'tile T':>7} {'halo overhead':>14} {'threadgroup mem':>16} {'note':<30}")
    for t in (8, 16, 32, 64, 128):
        overhead = ((t + 2) ** 2) / (t**2)
        mem = (t + 2) ** 2 * 4
        note = ""
        if mem > 32768:
            note = "exceeds the 32 KiB budget"
        elif t == TILE:
            note = "what this lab uses"
        print(f"  {t:>7} {overhead:>13.2f}x {mem:>14,} B  {note:<30}")
    print(
        "\n  (K=1 shown; the overhead is ((T+2K)^2)/T^2 in general.) Bigger tiles mean less\n"
        "  redundant traffic, but the footprint grows as T^2 against 32 KiB shared with L1.\n"
        "  T=128 is mathematically attractive and physically impossible."
    )

    header("Why not do it separably?")
    print(
        f"""  Both the naive and tiled kernels above do (2K+1)^2 multiplies per output. A
  separable filter needs only 2(2K+1). Intuitively that should win. Count the bytes for
  this image ({img_bytes / 2**20:.0f} MiB):

    single-pass tiled (K=1)     read ~1.13x + write 1x        ~ {int(1.13 * img_bytes / 2**20) + 16} MiB
    separable, two passes       pass1 read+write = 2x
                                pass2 read+write = 2x         ~ {4 * img_bytes / 2**20:.0f} MiB   (plus a fused-pass option)

  Separability removes FLOPs but adds a full round trip to DRAM. Below the roofline
  crossover that is a losing trade — unless you *fuse* the two passes so the intermediate
  never leaves on-chip memory. Fusion is the same idea that makes FlashAttention possible,
  and it is the general lesson: **count bytes, not FLOPs, whenever you are memory bound.**
"""
    )

    header("What to take away")
    print(
        """
  1. Compute the redundancy factor *and* check whether the cache already removes it.
     "This kernel reads each pixel 9 times" is a statement about instructions, not about
     DRAM traffic. A warp's 3x3 neighbourhood fits in ~4 cache lines; L1 absorbs it.

  2. Measure a floor for the shape you are working in. The copy kernel at the top of this
     lab is the single most useful number here: without it, "0.69 ms for a 3x3 stencil"
     looks like it needs optimising, and with it you can see there is nothing to win.

  3. The halo is the price of tiling: ((T+2K)^2)/T^2. It shrinks with T while the
     on-chip footprint grows as T^2 against a hard 32 KiB budget shared with L1.

  4. Loop-invariant loads look wasteful but are usually already handled. Staging the
     filter in threadgroup memory measured *slower* here at every K. Before "fixing" a
     redundant load, check whether the compiler hoisted it and whether the cache is
     serving it — the fix can cost more than the problem. (For a genuinely large filter,
     or one that does not fit in cache, the advice comes back.)

  5. Boundary handling is where stencils break. Zero-pad the halo and test on ragged
     sizes: 1x1, 37x37, 100x130. A 2048x2048 image is a multiple of every plausible tile
     size and will pass with the bounds check deleted.

  6. This is the shape of every stencil: image filters, Jacobi/heat-equation solvers, and
     the 3x3/5x5 convolutions in the first layers of a CNN. The halo-tiled pattern is the
     standard solution — and now you also know when it is not worth the code.

  7. **Never dispatch fewer threads than one threadgroup if the kernel has a barrier.**
     Writing this lab produced exactly that bug: a 1x1 image launched `grid=(1,1)` with
     `threadgroup=(32,8)`, and the filter-staging kernel's `threadgroup_barrier` then waited
     for 255 threads that were never dispatched. It is undefined behaviour, it does not
     crash, and it produces wrong answers only for images smaller than one tile — which a
     2048x2048 test set never generates. `launch_for()` above floors the grid at one full
     threadgroup, and the ragged tests in this file are what caught it.

  Your turn: raise K to 16 and 32. The tiled kernel needs (32+2K)^2 floats of threadgroup
  memory, so K=16 needs 16 KiB and K=32 needs 36 KiB — over budget. Shrink the tile to 16
  for large K and find the crossing point where tiling starts to beat the naive version by
  more than noise. Predict it from the halo formula first, then measure.
"""
    )


if __name__ == "__main__":
    main()
