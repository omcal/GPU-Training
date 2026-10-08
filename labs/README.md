# Metal / MLX labs

[Project overview](../README.md) · [Documentation](../docs/README.md) · [CUDA labs](../cuda/README.md)

Each lab is a standalone script: read it, run it, then change a number and run it again.
They are written to be read top to bottom — the kernel source *is* the lesson, and the
commentary around it explains what the numbers mean.

```bash
.venv/bin/python labs/lab01_vector_add.py
.venv/bin/python labs/lab02_coalescing.py
.venv/bin/python labs/lab03_reduction.py
.venv/bin/python labs/lab04_matmul.py
.venv/bin/python labs/lab05_convolution.py
.venv/bin/python labs/lab06_scan.py
```

| Lab | Kernel | Teaches | Measured on this M4 |
|---|---|---|---|
| **01** `vector_add` | elementwise | launch geometry, the `grid`-means-threads trap, boundary guards, grid-stride loops | matches Apple's own `mx.add` |
| **02** `coalescing` | strided read, transpose | cache-line amplification, useful vs fetched bytes, threadgroup memory, bank conflicts, the occupancy cost of big tiles | tiled 1.1-1.5x; 64x65 tile **5.9x slower** |
| **03** `reduction` | sum | the four levels of the hierarchy, simdgroup shuffles, barriers vs atomics | 45 → 74 GB/s; the 16.7M-atomic version beat the textbook tree |
| **04** `matmul` | GEMM | tiling, register blocking, arithmetic intensity, the roofline | 248 → 410 → 809 GFLOP/s, vs 1803 for `mx.matmul` |
| **05** `convolution` | 3x3..17x17 stencil | the halo, halo overhead arithmetic, boundary handling, and **when tiling is not worth it** | K=1 identical to a plain copy; K≥2 tiling wins a consistent ~1.6x |
| **06** `scan` | prefix sum | Hillis-Steele vs Blelloch, work efficiency vs depth, why a cross-threadgroup scan needs three kernels | 3-kernel scan is **1.6x faster than `mx.cumsum`** |

## How to work through a lab

1. **Read the kernel source first**, before running anything. Predict what it will do.
2. **Run it.** Check the correctness block passes before you look at any timing.
3. **Read the numbers against the ceiling.** The labs print a measured ceiling and a
   comparable baseline (Apple's own kernel, or a plain copy) so "fast" means something.
4. **Change one thing** and re-run. The "your turn" hints at the bottom of each lab
   suggest what.
5. **Read the matching PMPP chapter afterwards.** See
   [`../docs/04-pmpp-roadmap.md`](../docs/04-pmpp-roadmap.md) for the mapping.

## A pattern you will notice

Labs 02, 03 and 05 each end with a result that **contradicts the CUDA literature**:

* tiled transpose wins by only ~1.1-1.5x here, not ~10x;
* 16.7M contended atomics beat the classic threadgroup tree reduction;
* staging the convolution filter in on-chip memory — unanimous PMPP advice — measured
  *slower* at every kernel size;
* a 3x3 stencil is not bandwidth bound at all on this machine, because L1 absorbs the
  intra-warp redundancy.

That is not a flaw in the labs. Apple's memory system is not NVIDIA's, and the CUDA
literature describes hardware with a smaller L1, a wider memory interface and a different
cache hierarchy. Port the *method* — compute the redundancy, measure a floor, compare
against the best available implementation — and never the conclusion. If you take one
habit away from this repo, take that one.

## Things to try when a lab feels finished

* `lab01` — replace the scalar loads with `float4` (`const device float4* v =
  (const device float4*)inp;`). Does it help? (On this machine: not reliably.)
* `lab02` — remove the `+1` padding from `tile[32][33]` and measure. Then reason about
  which memory bank each lane hits.
* `lab03` — raise the threadgroup size from 256 to 1024 and see what happens to the
  v2-vs-v3 gap. (Prediction: v2's barrier penalty shrinks, because there are more warps
  to hide it behind.)
* `lab04` — push the register micro-tile from 2x2 to 4x4. This is the biggest remaining
  lever, and it is how real GEMM kernels are built.
* `lab05` — raise K to 16 and 32. The tiled kernel needs `(32+2K)^2` floats of threadgroup
  memory, so K=16 needs 16 KiB and K=32 needs 36 KiB (over budget). Shrink the tile to 16
  for large K and find the crossing point where tiling beats naive by more than noise.
  Predict it from the halo formula first, then measure.
* `lab06` — implement **decoupled look-back** (single-pass scan). You need an `atomic`
  flag per chunk (`atomic_outputs=True`), acquire/release ordering, and a spin loop.
  Decoupled look-back moves 2x the array against the 4x measured here.

## Not written yet — good next projects

In rough order of difficulty. You have every primitive these need:

| Topic | Kernel | New idea |
|---|---|---|
| Histogram | 256-bin histogram | atomics, contention, privatisation |
| Softmax | row-wise softmax | online max/sum, rescaling — **the gateway to attention** |
| Merge sort / radix sort | bitonic or radix | sorting networks, co-rank, digit histograms |
| Sparse matrices | SpMV over CSR | irregular access, load balancing |
| Graph search | BFS | frontiers, work efficiency, direction-optimising |
| **FlashAttention** | single-head attention | online softmax + tiling — the capstone |

`softmax` → `FlashAttention` is the natural continuation from here: a tiled matmul with a
numerically-stable reduction folded into the accumulation, where the ideas from labs 02,
03, 04 and 05 all meet at once. It is memory bound, which is exactly why lab05's
"count bytes, not FLOPs" lesson applies, and why it changed how transformers are trained.

## Measurement reminders

Every lab ends with numbers. Before you believe any of them:

* Run on AC power with nothing else using the GPU.
* Treat differences under ~10% as ties — this is a fanless laptop and it throttles.
  `best_ms` already repeats each measurement; if two versions land within noise, say so
  rather than picking a winner.
* Check the working-set size. A 4 MiB kernel measures a quarter of the bandwidth of a
  256 MiB one, and several labs print that warning explicitly.
* Full checklist: [`../docs/03-how-to-measure.md`](../docs/03-how-to-measure.md).
