"""Lab 03 — Reduction: the pattern that teaches you how to use a whole GPU.

Run:  python labs/lab03_reduction.py

Goal
----
Sum 16.7M floats four different ways and watch a 10x+ gap open up between the naive
and the good version — while every version is *correct*. This is the lab where the
hardware model becomes concrete.

Why reduction is the canonical second kernel
--------------------------------------------
Vector add (lab01) is embarrassingly parallel: every output depends on exactly one
input, so there is nothing to coordinate. Reduction is the opposite. Every output
depends on *all* inputs, so the kernel must combine partial results across
threads, simdgroups and threadgroups — and each level of that hierarchy has a
different mechanism with a different cost:

    level                 mechanism                      cost
    --------------------  -----------------------------  ---------------------------
    within a thread       plain register accumulate      free
    across a simdgroup    `simd_sum` (warp shuffle)      ~5 instructions, no memory
    across a threadgroup  threadgroup memory + barrier   on-chip traffic + 2 barriers
    across threadgroups   global `atomic_fetch_add`      serialised at one address

Choosing the right level for each step is the whole game. The four kernels below
deliberately get it progressively more right.

The PMPP connection
-------------------
This is PMPP's parallel-reduction chapter, and the "one atomic per threadgroup"
pattern is PMPP's performance model in miniature: launch enough threadgroups to fill
the machine, let each do cheap on-chip work, and touch global memory once per
threadgroup instead of once per thread.
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

N = 1 << 24  # 16.7M floats = 64 MiB
THREADS = 256
WARPS_PER_BLOCK = THREADS // 32


def bench(fn):
    return best_ms(fn, rounds=3, warmup=5, iters=20)


# ======================================================================================
# v1 — one global atomic per thread. Correct, and the worst possible thing to do.
#
# Every one of the 16.7M threads hammers the SAME 4-byte address. The GPU serialises
# those updates: the atomics become a queue longer than the memory system. Global
# atomics to a single address are a scalability trap, and this version exists so you
# can see how bad it is.
# ======================================================================================
V1_ATOMIC_SRC = """
    uint i = thread_position_in_grid.x;
    if (i < inp_shape[0]) {
        atomic_fetch_add_explicit(out + 0, inp[i], memory_order_relaxed);
    }
"""

kernel_v1 = mx.fast.metal_kernel(
    name="reduce_v1_atomic",
    input_names=["inp"],
    output_names=["out"],
    source=V1_ATOMIC_SRC,
    atomic_outputs=True,
)


def reduce_v1(a: mx.array) -> mx.array:
    return kernel_v1(
        inputs=[a],
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
        init_value=0,
        **cuda(a.size // THREADS, THREADS).kwargs(),
    )[0]


# ======================================================================================
# v2 — the classic PMPP tree, in threadgroup memory.
#
# Each threadgroup loads its slice into on-chip memory and halves it log2(256) = 8
# times. Two barriers per step. Note the `if (tid < s)`: half the threads idle each
# step, which is inherent to a tree but also means the work per step is tiny — the
# barriers dominate.
#
# Cost: 8 barriers per threadgroup. On a GPU with many resident threadgroups the
# barrier latency is hidden, but with few it is not.
# ======================================================================================
V2_TREE_SRC = """
    threadgroup float scratch[256];
    uint tid = thread_index_in_threadgroup;
    uint tg = threadgroup_position_in_grid.x;
    uint i = tg * 256 + tid;

    scratch[tid] = (i < inp_shape[0]) ? inp[i] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) atomic_fetch_add_explicit(out + 0, scratch[0], memory_order_relaxed);
"""

kernel_v2 = mx.fast.metal_kernel(
    name="reduce_v2_tree",
    input_names=["inp"],
    output_names=["out"],
    source=V2_TREE_SRC,
    atomic_outputs=True,
)


def reduce_v2(a: mx.array) -> mx.array:
    return kernel_v2(
        inputs=[a],
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
        init_value=0,
        **cuda(a.size // THREADS, THREADS).kwargs(),
    )[0]


# ======================================================================================
# v3 — hierarchical: shuffle within simdgroups, threadgroup memory only across them.
#
# This is the version to internalise. Three levels, each using the cheapest mechanism
# that can do the job:
#
#   1. `simd_sum(v)`  — the 32 lanes of a simdgroup are a hardware unit; shuffles are
#      register-to-register and need no memory or barrier at all.
#   2. threadgroup memory for the 8 partial sums (one per simdgroup), then one more
#      `simd_sum` across them. Note the guard `tid < 32`: the parallel reduction across
#      8 values is done by the *first* simdgroup only, and `simd_sum` inside that guard
#      is safe because the whole simdgroup is still active.
#   3. ONE global atomic per threadgroup (8192 of them for this problem) instead of
#      one per thread (16.7M).
#
# One barrier total, versus eight in v2.
# ======================================================================================
V3_SIMD_SRC = """
    threadgroup float warp_sums[8];   // 256 threads / 32 lanes = 8 simdgroups
    uint tid = thread_index_in_threadgroup;
    uint tg = threadgroup_position_in_grid.x;
    uint i = tg * 256 + tid;

    float v = (i < inp_shape[0]) ? inp[i] : 0.0f;
    float s = simd_sum(v);                            // level 1: free
    if (thread_index_in_simdgroup == 0) warp_sums[tid / 32] = s;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {                                   // level 2: one half-warp
        float w = (tid < 8) ? warp_sums[tid] : 0.0f;
        float total = simd_sum(w);
        if (tid == 0) {
            atomic_fetch_add_explicit(out + 0, total, memory_order_relaxed);
        }
    }
"""

kernel_v3 = mx.fast.metal_kernel(
    name="reduce_v3_simd",
    input_names=["inp"],
    output_names=["out"],
    source=V3_SIMD_SRC,
    atomic_outputs=True,
)


def reduce_v3(a: mx.array) -> mx.array:
    return kernel_v3(
        inputs=[a],
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
        init_value=0,
        **cuda(a.size // THREADS, THREADS).kwargs(),
    )[0]


# ======================================================================================
# v4 — v3 plus several elements per thread.
#
# v1-v3 all read exactly one element per thread, which means one outstanding memory
# request per thread. The usual argument is that this leaves you latency bound rather
# than bandwidth bound, and that giving each thread several independent loads fixes it.
#
# ELEMS is a compile-time template constant so the loop unrolls completely, and the
# loop stride is `k * T` (T = total thread count), NOT 1 -- striding by the grid keeps
# consecutive lanes on consecutive addresses so the access stays coalesced however many
# elements each thread reads. Striding by 1 would break coalescing; see lab02.
#
# Measure the sweep at the bottom of this lab before believing the latency argument.
# ======================================================================================
V4_ILP_SRC = """
    threadgroup float warp_sums[8];
    uint tid = thread_index_in_threadgroup;
    uint tg = threadgroup_position_in_grid.x;
    uint T = threads_per_grid.x;
    uint base = tg * 256 + tid;

    float acc = 0.0f;
    for (uint k = 0; k < ELEMS; ++k) {
        uint i = base + k * T;
        if (i < inp_shape[0]) acc += inp[i];
    }

    float s = simd_sum(acc);
    if (thread_index_in_simdgroup == 0) warp_sums[tid / 32] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        float w = (tid < 8) ? warp_sums[tid] : 0.0f;
        float total = simd_sum(w);
        if (tid == 0) {
            atomic_fetch_add_explicit(out + 0, total, memory_order_relaxed);
        }
    }
"""

kernel_v4 = mx.fast.metal_kernel(
    name="reduce_v4_ilp",
    input_names=["inp"],
    output_names=["out"],
    source=V4_ILP_SRC,
    atomic_outputs=True,
)


def reduce_v4(a: mx.array, elems_per_thread: int = 4) -> mx.array:
    if a.size % (THREADS * elems_per_thread):
        raise ValueError("size must divide evenly for this demo")
    n_threads = a.size // elems_per_thread
    return kernel_v4(
        inputs=[a],
        template=[("ELEMS", elems_per_thread)],
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
        init_value=0,
        **cuda(n_threads // THREADS, THREADS).kwargs(),
    )[0]


def main() -> None:
    report()
    peak = peak_bandwidth_gbs()
    print(f"\nMeasured 1R:1W streaming ceiling: {peak:.1f} GB/s")

    a = mx.random.normal(shape=(N,), dtype=mx.float32)
    mx.eval(a)
    reference = float(np.array(a).astype(np.float64).sum())
    nbytes = N * 4  # reduction is read-dominated; the writes are a rounding error

    header(f"Correctness (sum of {N:,} float32, fp64 reference = {reference:.4f})")
    # Tolerance note: the GPU sums in a different order every run (atomics), and
    # float32 addition is not associative. The *relative* error is what matters.
    for name, fn in (
        ("v1 atomic/thread", reduce_v1),
        ("v2 threadgroup tree", reduce_v2),
        ("v3 simdgroup hierarchical", reduce_v3),
        ("v4 hierarchical + 4/thread", reduce_v4),
    ):
        got = float(np.array(fn(a))[0])
        err = abs(got - reference) / abs(reference)
        print(f"  {name:<28} got {got:>14.4f}   relative error {err:.3e}")
        assert err < 1e-4, f"{name} is wrong: {got} vs {reference}"
    assert_close(reduce_v3(a), np.array([reference]), rtol=1e-4, atol=1e-1, label="v3 (assert_close)")
    assert_close(
        reduce_v4(a), np.array([reference]), rtol=1e-4, atol=1e-1, label="v4 (assert_close)"
    )

    header("Benchmarks (64 MiB read; GB/s counts useful bytes read)")
    print(
        "  Caveat: the ceiling below was probed with a 1-read:1-write kernel, while a\n"
        "  reduction is read-only. On this machine the two mixes happen to measure about\n"
        "  the same, but they are not the same shape, so read these as relative\n"
        "  comparisons. A plain streaming read here measures ~60 GB/s at a 64 MiB working\n"
        "  set and ~96 GB/s at 256 MiB -- working-set size moves the number as much as the\n"
        "  access pattern does.\n"
    )
    print(f"  {'kernel':<34} {'time':>12}  {'bandwidth':>12}")
    report_row("mx.sum (Apple's own reduction)", bench(lambda: mx.eval(mx.sum(a))), nbytes, peak_bw=peak)
    report_row("v1 atomic per thread", bench(lambda: mx.eval(reduce_v1(a))), nbytes, peak_bw=peak)
    report_row("v2 threadgroup tree (8 barriers)", bench(lambda: mx.eval(reduce_v2(a))), nbytes, peak_bw=peak)
    report_row("v3 simdgroup hierarchical", bench(lambda: mx.eval(reduce_v3(a))), nbytes, peak_bw=peak)
    report_row("v4 hierarchical + 4 elem/thread", bench(lambda: mx.eval(reduce_v4(a))), nbytes, peak_bw=peak)

    header("The cost of each level of the hierarchy")
    for elems in (1, 2, 4, 8, 16):
        n_threads = N // elems
        t = bench(lambda e=elems: mx.eval(reduce_v4(a, e)))
        bw = bandwidth_gbs(nbytes, t.median)
        print(
            f"  {elems:>2} elems/thread  {n_threads:>10,} threads  "
            f"{t.median:7.4f} ms  {bw:6.1f} GB/s  ({100 * bw / peak:>3.0f}% of ceiling)"
        )

    header("What to take away")
    print(
        """
  1. Reduction forces you to use the whole memory/execution hierarchy, so it is the
     fastest way to learn what that hierarchy costs. There are four levels and four
     different mechanisms: registers, simdgroup shuffles, threadgroup memory, and
     global atomics -- in increasing order of expense.

  2. **The textbook ordering did not fully hold on this hardware, and that is the most
     useful result in this lab.** Two surprises:

       * v1 (one global atomic per *thread*, 16.7M of them at one address) was NOT the
         disaster the CUDA literature promises. Apple's memory system handles contended
         atomics far better than an NVIDIA GPU does. On NVIDIA this kernel is typically
         10x+ off; here it beat the classic tree reduction.
       * v2, the classic PMPP-style threadgroup tree, was the *slowest* of the four.
         Reason: 8 barrier rounds over only 256 threads per threadgroup. With few warps
         per threadgroup there is nothing to hide the barrier latency behind, and the
         atomics it saves were not the bottleneck in the first place.

     v3 wins because it uses the cheapest mechanism for each level: shuffles inside the
     simdgroup (free), one barrier, then one atomic per threadgroup instead of eight
     barriers and one atomic per threadgroup.

  3. `simd_sum` is nearly free and reduces 32 elements at once. On CUDA this is
     `__shfl_down_sync` and every serious reduction uses it. "Shuffle within
     simdgroups, then one barrier, then combine the simdgroup sums" is the standard
     shape of GPU reduction.

  4. The elems/thread sweep above came out essentially FLAT (71-72 GB/s from 1 to 16
     elements per thread). The common claim is that one load per thread leaves you
     latency bound and that more in-flight loads fix it -- true in general, but not the
     limiter here. Be suspicious of any optimisation you have not measured on your own
     hardware; this is the second time in this lab that the received wisdom did not
     reproduce.

  5. Atomic order is not deterministic, so floating-point sums differ run to run. That
     is why the correctness check uses a relative tolerance, and why reproducible
     bit-exact reduction is a real engineering problem rather than a formality.

  6. Practical guidance for your own kernels on this machine: prefer the v3 shape
     (simdgroup shuffle -> one barrier -> one atomic per threadgroup) and use 512 or
     1024 threads per threadgroup so the barrier has more warps to hide behind. Then
     measure.
"""
    )


if __name__ == "__main__":
    main()
