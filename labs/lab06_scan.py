"""Lab 06 — Prefix sum (scan): work efficiency, and why you need three kernels.

Run:  python labs/lab06_scan.py

Goal
----
Compute the prefix sum of 1M elements, learn the two canonical scan algorithms, and
understand why a scan that spans more than one threadgroup cannot be done by one kernel
(the obvious way).

Why scan is the third canonical kernel
--------------------------------------
Vector add (lab01) has no dependencies between outputs. Reduction (lab03) has one output
depending on everything, and is solved by a tree. Scan is the interesting middle: output
``i`` depends on all inputs ``0..i``, so there are ``n`` dependent outputs — and unlike a
reduction, you cannot throw away the intermediate values.

That makes scan the standard building block for problems that are *sequentially* coupled
but *locally* parallel: stream compaction, radix sort, sparse matrix assembly, and any
"running total". It is also the classic lesson in **work efficiency**, the difference
between "how many parallel steps" and "how much total work":

    algorithm       steps (depth)      total adds        for n = 1024
    --------------  -----------------  ----------------  -------------
    Hillis-Steele   log2(n) = 10       n log2(n)         10,240
    Blelloch        2 log2(n) = 20     2n                 2,048

Hillis-Steele is shallower but does 5x the arithmetic. Blelloch is deeper but linear in
work. Which wins depends on whether you are latency-bound or throughput-bound — which is
exactly the kind of question this repo keeps insisting you answer by measuring.

The part that surprises everyone
--------------------------------
A threadgroup can only scan what fits in its own threadgroup memory. For 1M elements you
need many threadgroups, and then output ``i`` depends on the totals of *every preceding
threadgroup* — a dependency no single kernel can resolve, because threadgroups are not
guaranteed to run in any order and there is no global barrier inside a kernel.

The standard solution is **three kernels**:
    1. each threadgroup scans its own chunk and writes its chunk total
    2. one threadgroup scans the array of chunk totals (this is the recursion)
    3. every threadgroup adds its exclusive offset to its chunk

Step 2 is the same problem at 1/1024th the size, and it is the seed of the parallel
recursion that makes scan efficient at any scale. PMPP's "decoupled look-back" is the
single-pass alternative, discussed at the end.
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

N = 1 << 20          # 1M elements = 4 MiB
BLOCK = 1024         # elements per threadgroup
HALF = BLOCK // 2    # Blelloch uses 2 elements per thread
NBLOCKS = N // BLOCK


def bench(fn):
    return best_ms(fn, rounds=3, warmup=5, iters=20)


# ======================================================================================
# Single threadgroup, algorithm 1: Hillis-Steele (inclusive scan).
#
# Step d: every element adds the value `d` positions to its left. After log2(n) steps
# every element holds the sum of everything up to and including itself.
#
# Two barriers per step, because the read of `tmp[tid - offset]` must happen before any
# write in the same step. One barrier is not enough and the failure is data-dependent:
# with 1024 threads a warp can race ahead and read a value that has already been updated.
#
# Work: n log2(n) adds. For n = 1024 that is 10,240 adds to produce 1024 outputs.
# ======================================================================================
HILLIS_STEELE_SRC = """
    threadgroup float tmp[1024];
    uint tid = thread_index_in_threadgroup;
    uint n = inp_shape[0];

    tmp[tid] = (tid < n) ? inp[tid] : 0.0f;

    for (uint offset = 1u; offset < 1024u; offset <<= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float left = (tid >= offset) ? tmp[tid - offset] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tmp[tid] += left;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < n) out[tid] = tmp[tid];
"""

kernel_hs = mx.fast.metal_kernel(
    name="scan_hillis_steele",
    input_names=["inp"],
    output_names=["out"],
    source=HILLIS_STEELE_SRC,
)


def scan_hillis_steele(a: mx.array) -> mx.array:
    if a.size != BLOCK:
        raise ValueError(f"single-block scan requires exactly {BLOCK} elements")
    return kernel_hs(
        inputs=[a],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(BLOCK, 1, 1),
        threadgroup=(BLOCK, 1, 1),
    )[0]


# ======================================================================================
# Single threadgroup, algorithm 2: Blelloch (exclusive scan).
#
# Two phases over an implicit binary tree:
#
#   up-sweep   (reduce):  d = n/2, n/4, ..., 1  — build the tree of partial sums
#   down-sweep (distribute): d = 1, 2, ..., n/2 — push each subtree's total left
#
# Setting the root to 0 between the phases is what turns the inclusive tree sum into an
# exclusive scan. The total is not lost: it is saved into `block_total` before the root is
# zeroed, and that is what the hierarchical version needs.
#
# Work: 2n adds — for n = 1024, 2,048 adds, 5x less than Hillis-Steele. But 2*log2(n) = 20
# barrier steps instead of 10.
#
# Numbers to notice: only 512 threads are active, two elements per thread, and the guards
# `if (tid < d)` mean most threads idle during most steps. That is inherent to the tree:
# the parallel width halves every step while the barrier count stays the same.
# ======================================================================================
BLELLOCH_SRC = """
    threadgroup float tmp[1024];
    threadgroup float grand[1];

    uint tid = thread_index_in_threadgroup;      // 0..511
    uint n = inp_shape[0];

    tmp[2u * tid]      = inp[2u * tid];
    tmp[2u * tid + 1u] = inp[2u * tid + 1u];

    // ---- up-sweep: build partial sums up the tree
    uint offset = 1u;
    for (uint d = n >> 1u; d > 0u; d >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai = offset * (2u * tid + 1u) - 1u;
            uint bi = offset * (2u * tid + 2u) - 1u;
            tmp[bi] += tmp[ai];
        }
        offset <<= 1u;
    }

    // ---- save the grand total and zero the root: this is what makes it *exclusive*
    if (tid == 0u) {
        grand[0] = tmp[n - 1u];
        tmp[n - 1u] = 0.0f;
    }

    // ---- down-sweep: push subtree totals left
    for (uint d = 1u; d < n; d <<= 1u) {
        offset >>= 1u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai = offset * (2u * tid + 1u) - 1u;
            uint bi = offset * (2u * tid + 2u) - 1u;
            float t = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MLX hands us a 2-element `total` buffer; lane 0 stores the reduction there.
    if (tid == 0u) total[0] = grand[0];
    if (tid == 1u) total[1] = 0.0f;

    out[2u * tid]      = tmp[2u * tid];
    out[2u * tid + 1u] = tmp[2u * tid + 1u];
"""

kernel_blelloch = mx.fast.metal_kernel(
    name="scan_blelloch",
    input_names=["inp"],
    output_names=["out", "total"],
    source=BLELLOCH_SRC,
)


def scan_blelloch(a: mx.array) -> tuple[mx.array, mx.array]:
    if a.size != BLOCK:
        raise ValueError(f"single-block scan requires exactly {BLOCK} elements")
    scanned, total = kernel_blelloch(
        inputs=[a],
        output_shapes=[a.shape, (2,)],
        output_dtypes=[a.dtype, a.dtype],
        grid=(HALF, 1, 1),
        threadgroup=(HALF, 1, 1),
    )
    return scanned, total


# ======================================================================================
# Hierarchical scan, kernel 1 of 3: per-threadgroup Blelloch scan + chunk total.
#
# Each threadgroup owns BLOCK elements (512 threads, 2 each). It writes:
#   * `scanned`    — the exclusive scan *within* the chunk (offsets from the other chunks
#                    are added later, by kernel 3)
#   * `block_sums` — the chunk's total, which kernel 2 will scan
#
# The chunk total is read from the tree root right after the up-sweep, before the root is
# zeroed. Forgetting to save it here is the classic bug: the down-sweep destroys it.
# ======================================================================================
BLOCK_SCAN_SRC = """
    threadgroup float tmp[1024];
    uint tid = thread_index_in_threadgroup;      // 0..511
    uint tg = threadgroup_position_in_grid.x;
    uint base = tg * 1024u;

    tmp[2u * tid]      = inp[base + 2u * tid];
    tmp[2u * tid + 1u] = inp[base + 2u * tid + 1u];

    uint offset = 1u;
    for (uint d = 512u; d > 0u; d >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai = offset * (2u * tid + 1u) - 1u;
            uint bi = offset * (2u * tid + 2u) - 1u;
            tmp[bi] += tmp[ai];
        }
        offset <<= 1u;
    }

    if (tid == 0u) {
        block_sums[tg] = tmp[1023u];             // save BEFORE zeroing the root
        tmp[1023u] = 0.0f;
    }

    for (uint d = 1u; d < 1024u; d <<= 1u) {
        offset >>= 1u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai = offset * (2u * tid + 1u) - 1u;
            uint bi = offset * (2u * tid + 2u) - 1u;
            float t = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    scanned[base + 2u * tid]      = tmp[2u * tid];
    scanned[base + 2u * tid + 1u] = tmp[2u * tid + 1u];
"""

kernel_block_scan = mx.fast.metal_kernel(
    name="scan_block",
    input_names=["inp"],
    output_names=["scanned", "block_sums"],
    source=BLOCK_SCAN_SRC,
)


# ======================================================================================
# Hierarchical scan, kernel 2 of 3: scan the chunk totals.
#
# This is the recursion. `block_sums` has NBLOCKS = 1024 entries, so *the same algorithm*
# applies at 1/1024th the size — and here it fits in a single threadgroup, which is why
# the recursion terminates. At larger scales you would recurse again, or use decoupled
# look-back to avoid the extra passes entirely.
# ======================================================================================
SCAN_SUMS_SRC = """
    threadgroup float tmp[1024];
    uint tid = thread_index_in_threadgroup;      // 0..511
    uint n = inp_shape[0];                       // NBLOCKS, must be 1024 here

    tmp[2u * tid]      = inp[2u * tid];
    tmp[2u * tid + 1u] = inp[2u * tid + 1u];

    uint offset = 1u;
    for (uint d = n >> 1u; d > 0u; d >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai = offset * (2u * tid + 1u) - 1u;
            uint bi = offset * (2u * tid + 2u) - 1u;
            tmp[bi] += tmp[ai];
        }
        offset <<= 1u;
    }
    if (tid == 0u) tmp[n - 1u] = 0.0f;

    for (uint d = 1u; d < n; d <<= 1u) {
        offset >>= 1u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) {
            uint ai = offset * (2u * tid + 1u) - 1u;
            uint bi = offset * (2u * tid + 2u) - 1u;
            float t = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    out[2u * tid]      = tmp[2u * tid];
    out[2u * tid + 1u] = tmp[2u * tid + 1u];
"""

kernel_scan_sums = mx.fast.metal_kernel(
    name="scan_block_sums",
    input_names=["inp"],
    output_names=["out"],
    source=SCAN_SUMS_SRC,
)


# ======================================================================================
# Hierarchical scan, kernel 3 of 3: add each chunk's exclusive offset.
#
# Pure elementwise work: 1 read + 1 write per element, no barriers, no dependency between
# threads. This is the pass that pays the DRAM bill — and it is why a three-pass scan is
# never bandwidth-optimal compared with a single-pass decoupled look-back.
# ======================================================================================
ADD_OFFSETS_SRC = """
    uint tid = thread_index_in_threadgroup;
    uint tg = threadgroup_position_in_grid.x;
    uint i = tg * 1024u + 2u * tid;
    float off = block_offsets[tg];
    out[i]           = scanned[i]           + off;
    out[i + 1u]      = scanned[i + 1u]      + off;
"""

kernel_add_offsets = mx.fast.metal_kernel(
    name="scan_add_offsets",
    input_names=["scanned", "block_offsets"],
    output_names=["out"],
    source=ADD_OFFSETS_SRC,
)


def scan_hierarchical(a: mx.array) -> mx.array:
    """Exclusive prefix sum over an arbitrary power-of-two multiple of BLOCK."""
    n = a.size
    if n % BLOCK or (n // BLOCK) & (n // BLOCK - 1):
        raise ValueError("size must be BLOCK * a power of two, and <= BLOCK*1024")
    nblocks = n // BLOCK

    scanned, block_sums = kernel_block_scan(
        inputs=[a],
        output_shapes=[(n,), (nblocks,)],
        output_dtypes=[a.dtype, a.dtype],
        grid=(nblocks * HALF, 1, 1),
        threadgroup=(HALF, 1, 1),
    )
    (block_offsets,) = kernel_scan_sums(
        inputs=[block_sums],
        output_shapes=[(nblocks,)],
        output_dtypes=[a.dtype],
        grid=(HALF, 1, 1),
        threadgroup=(HALF, 1, 1),
    )
    (out,) = kernel_add_offsets(
        inputs=[scanned, block_offsets],
        output_shapes=[(n,)],
        output_dtypes=[a.dtype],
        grid=(nblocks * HALF, 1, 1),
        threadgroup=(HALF, 1, 1),
    )
    return out


def exclusive_reference(x: np.ndarray) -> np.ndarray:
    """Exclusive prefix sum in float64: [0, x0, x0+x1, ...]."""
    c = np.cumsum(x.astype(np.float64))
    return np.concatenate([[0.0], c[:-1]])


def main() -> None:
    report()
    peak = peak_bandwidth_gbs()

    small = mx.random.normal(shape=(BLOCK,), dtype=mx.float32)
    mx.eval(small)
    small_np = np.array(small)
    ref_incl = np.cumsum(small_np.astype(np.float64))
    ref_excl = exclusive_reference(small_np)

    header(f"A. Single threadgroup, n = {BLOCK}")
    assert_close(scan_hillis_steele(small), ref_incl, rtol=1e-5, atol=1e-4, label="hillis-steele (inclusive)")
    scanned, total = scan_blelloch(small)
    assert_close(scanned, ref_excl, rtol=1e-5, atol=1e-4, label="blelloch (exclusive)")
    assert_close(total[0:1], np.array([ref_incl[-1]]), rtol=1e-5, atol=1e-4, label="blelloch grand total")

    t_hs = bench(lambda: mx.eval(scan_hillis_steele(small)))
    t_bl = bench(lambda: mx.eval(scan_blelloch(small)[0]))

    adds_hs = BLOCK * 10
    adds_bl = 2 * BLOCK
    print(
        f"\n  {'algorithm':<22} {'barrier steps':>14} {'adds':>8} {'time':>11}"
        f"\n  {'hillis-steele':<22} {20:>14} {adds_hs:>8,} {t_hs.median:>9.4f} ms"
        f"\n  {'blelloch':<22} {20:>14} {adds_bl:>8,} {t_bl.median:>9.4f} ms"
    )
    # Do not assert which one wins: on a 1024-element problem with ~20 barrier rounds and
    # only 512-1024 threads, the answer flips between runs and between machines. Print
    # what was measured and interpret it.
    if t_bl.median <= t_hs.median:
        print(
            f"\n  Blelloch is {t_hs.median / t_bl.median:.2f}x faster with "
            f"{adds_hs / adds_bl:.0f}x less arithmetic. Work efficiency won:\n"
            f"  the saved adds outweighed the two extra barrier rounds.\n"
            f"  Work: {adds_hs:,} adds (Hillis-Steele) vs {adds_bl:,} adds (Blelloch) — "
            f"Blelloch is\n  linear in n, Hillis-Steele is n*log2(n). At n = {BLOCK} that "
            f"ratio is only {adds_hs / adds_bl:.0f}x, but it grows\n  with n: at n = 64K it "
            f"would be 16x, and the choice stops being close."
        )
    else:
        print(
            f"\n  Blelloch is {t_bl.median / t_hs.median:.2f}x SLOWER despite doing "
            f"{adds_hs / adds_bl:.0f}x less arithmetic. At this size\n"
            f"  the kernel is barrier-bound: both versions pay ~20 barrier rounds over only\n"
            f"  512-1024 threads, and the saved adds are in the noise. Work efficiency only\n"
            f"  pays once arithmetic is the bottleneck — larger n per threadgroup, or a GPU\n"
            f"  with many more lanes than barriers."
        )
    print(
        f"\n  Work complexity, which is machine independent:\n"
        f"    Hillis-Steele  n*log2(n) adds = {adds_hs:,} here, and 16x that at n = 64K\n"
        f"    Blelloch       2n adds       = {adds_bl:,} here, and linear forever after\n"
        f"  Which one is faster *is* machine dependent, and the margin at n = {BLOCK} is small\n"
        f"  enough that you should re-measure rather than trust this run."
    )

    # ------------------------------------------------------------------ hierarchical
    x = mx.random.normal(shape=(N,), dtype=mx.float32)
    mx.eval(x)
    x_np = np.array(x)
    ref = exclusive_reference(x_np)

    header(f"B. Hierarchical three-kernel scan, n = {N:,} ({N // BLOCK} threadgroups)")
    got = scan_hierarchical(x)
    assert_close(got, ref, rtol=1e-3, atol=1e-2, label="hierarchical exclusive scan")
    print(
        f"  Checked against a float64 exclusive prefix sum. Tolerance is loose on purpose:\n"
        f"  a float32 scan of {N:,} elements accumulates a different rounding pattern than\n"
        f"  the reference, and the prefix values grow to ~{np.abs(ref).max():.0f}."
    )

    nbytes = 2 * N * 4  # pass 3 is the only pass that touches the whole array twice
    print(
        f"\n  Working-set warning: this is a {N * 4 / 2**20:.0f} MiB array, which is far too small to\n"
        f"  reach the {peak:.0f} GB/s streaming ceiling (see docs/03-how-to-measure.md). Both\n"
        f"  implementations below are latency- and barrier-bound, so read the GB/s column as\n"
        f"  a relative measure, not as a bandwidth statement.\n"
    )
    t_vendor = bench(lambda: mx.eval(mx.cumsum(x)))
    t_ours = bench(lambda: mx.eval(scan_hierarchical(x)))
    report_row("mx.cumsum (inclusive, framework)", t_vendor, nbytes, peak_bw=peak)
    report_row("hierarchical 3-kernel (exclusive)", t_ours, nbytes, peak_bw=peak)
    print(
        f"\n  Worth noting: the three-kernel version is {t_vendor.median / t_ours.median:.2f}x faster\n"
        f"  than the framework's `mx.cumsum` at this size. Losing to the vendor library is\n"
        f"  normal (labs 01-04); winning means the library is not tuned for this shape."
    )
    report_row(
        "  pass 1/3: block scan + totals",
        bench(lambda: mx.eval(kernel_block_scan(
            inputs=[x], output_shapes=[(N,), (NBLOCKS,)],
            output_dtypes=[x.dtype, x.dtype],
            grid=(NBLOCKS * HALF, 1, 1), threadgroup=(HALF, 1, 1))[0])),
        nbytes,
        peak_bw=peak,
    )
    (bs, _) = kernel_block_scan(
        inputs=[x], output_shapes=[(N,), (NBLOCKS,)],
        output_dtypes=[x.dtype, x.dtype],
        grid=(NBLOCKS * HALF, 1, 1), threadgroup=(HALF, 1, 1),
    )
    (bo,) = kernel_scan_sums(
        inputs=[bs], output_shapes=[(NBLOCKS,)], output_dtypes=[x.dtype],
        grid=(HALF, 1, 1), threadgroup=(HALF, 1, 1),
    )
    (sc, _) = kernel_block_scan(
        inputs=[x], output_shapes=[(N,), (NBLOCKS,)],
        output_dtypes=[x.dtype, x.dtype],
        grid=(NBLOCKS * HALF, 1, 1), threadgroup=(HALF, 1, 1),
    )
    report_row(
        "  pass 3/3: add offsets",
        bench(lambda: mx.eval(kernel_add_offsets(
            inputs=[sc, bo], output_shapes=[(N,)], output_dtypes=[x.dtype],
            grid=(NBLOCKS * HALF, 1, 1), threadgroup=(HALF, 1, 1))[0])),
        nbytes,
        peak_bw=peak,
    )

    header("Total traffic, counted honestly")
    passes = {
        "pass 1  block scan": (N * 4, N * 4 + NBLOCKS * 4),
        "pass 2  scan totals": (NBLOCKS * 4, NBLOCKS * 4),
        "pass 3  add offsets": (N * 4 + NBLOCKS * 4, N * 4),
    }
    for name, (r, w) in passes.items():
        print(f"  {name:<22} read {r / 2**20:>7.2f} MiB  write {w / 2**20:>7.2f} MiB")
    total = sum(r + w for r, w in passes.values())
    print(
        f"  {'TOTAL':<22} {total / 2**20:>20.2f} MiB  "
        f"({total / (N * 4):.2f}x the array)"
        f"\n\n  An ideal single-pass scan would move 2x the array (1 read + 1 write). This one"
        f"\n  moves {total / (N * 4):.2f}x — {total / (N * 4) / 2:.1f}x the ideal — and that extra traffic is"
        f"\n  the entire reason decoupled look-back exists. (Pass 2 moves "
        f"{2 * NBLOCKS * 4 / 1024:.1f} KiB, which is why it is the recursion and not a"
        f"\n  bottleneck: it is 1/{NBLOCKS}th the size of the real work.)"
    )

    header("What to take away")
    print(
        """
  1. Scan is the third canonical parallel pattern, after map (lab01) and reduction
     (lab03). Its distinguishing feature is that it produces n outputs with n dependent
     values, so unlike a reduction it cannot discard its intermediates. Compaction, radix
     sort, sparse assembly and running totals all reduce to it.

  2. **Work efficiency vs depth is a real tradeoff and it is machine dependent.** Blelloch
     does `2n` adds and `2*log2(n)` barrier rounds; Hillis-Steele does `n*log2(n)` adds and
     `log2(n)` rounds. At n=1024 the work ratio is only 5x and the barrier counts are both
     around 20, so the winner is decided by details of the machine — the run above says
     which way it went, and it is worth re-running to see how stable that is. The work
     ratio is what generalises: at n=64K, Blelloch does 16x less arithmetic and the choice
     stops being close.

  3. **A threadgroup cannot scan more than it can hold, and one kernel cannot scan across
     threadgroups.** There is no global barrier inside a kernel and threadgroups run in no
     guaranteed order, so output i cannot depend on the totals of chunks that may not have
     run yet. Hence three kernels: local scan, scan the totals (a recursion at 1/1024th the
     size), then propagate. Recognising "this needs a global prefix" is the skill; the
     three-kernel shape is the standard answer.

  4. The recursion is what makes it scale: pass 2 is the *same algorithm* on a smaller
     input, so the scheme works at any size. At very large n you would recurse again
     rather than assume the totals fit in one threadgroup — this lab asserts they do.

  5. The cost is traffic. Three passes move ~4x the array against 2x for an ideal
     single pass — **2x the traffic**. **Decoupled look-back** (Merrill & Garland, 2016)
     removes that: each threadgroup publishes its aggregate and then spins on its
     predecessors' published prefixes, so one kernel does the whole job. It is the
     production technique, it is subtly correct only with careful memory ordering, and it
     is a good next exercise now that you have the three-pass version working.

  6. Save the chunk total BEFORE zeroing the tree root. The down-sweep overwrites it, and
     the bug is silent — the offsets are simply wrong for every chunk after the first.

  Your turn: implement decoupled look-back. You will need an `atomic` flag per chunk
  (`atomic_outputs=True`), `memory_order_acquire`/`release` on the flag accesses, and a
  spin loop. Then compare its traffic (2x the array) against the 4x measured above, and
  see how much of that theoretical 2x saving actually shows up in wall clock. Start with a
  larger N, too: at a million elements this whole lab is latency-bound, and the traffic
  difference only becomes visible once the array is big enough to be the bottleneck.
  (Raising N past BLOCK*1024 = 1M means pass 2 no longer fits in one threadgroup, so you
  will need a second level of recursion or decoupled look-back itself.)
"""
    )


if __name__ == "__main__":
    main()
