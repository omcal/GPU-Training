# 05 · The route: how to actually learn this

**CUDA/SSH için güncel program:** [PMPP + Chris, 8 Ekim–1 Kasım 2026](08-cuda-inference-study-plan.md).
[Bağımsız çalışma alanı](../study/README.md) ile başla. Bu dosyanın geri kalanı
Mac/Metal öğrenme rotasıdır; SSH ile CUDA kernel çalıştırılabilen mevcut ortamın
programı yukarıdaki bağlantıdadır.


You asked the right question. Having the labs and having PMPP is not the same as having a
route through them, and the ordering matters more than the material.

Short version: **do not read PMPP cover to cover first.** Alternate — read a chapter
shallowly, do the matching lab, then re-read the chapter. Details and a concrete plan below.

---

## The method: three passes per topic

The instinct is to read the chapter carefully, then do the lab. That is the slow way, and
it fails for a specific reason: **PMPP's code is CUDA you cannot run.** You would be reading
several hundred pages of syntax with no feedback loop, retaining a vocabulary with nothing
to attach it to. Two weeks later it is gone.

Do this instead, per topic, in one sitting:

| Pass | Time | What you do |
|---|---|---|
| **1. Skim** | 15-20 min | Read the chapter *fast* — headings, figures, the kernel listings, the performance graphs. Do not try to absorb the code. Goal: get the vocabulary and the shape of the problem. |
| **2. Lab** | 45-90 min | Do the matching lab in this repo. Read the kernel source, predict the result, run it, then change one thing. This is where the intuition forms. |
| **3. Re-read** | 20-30 min | Now read the chapter properly. The theory attaches to something you have already felt. This pass is fast and it sticks. |

Pass 2 is the one that does the work. Pass 1 exists only so that pass 3 lands.

**Why this works better than PMPP-first on this machine specifically:** every lab here is
a *runnable Metal port* of the idea the chapter describes. `pmpp/` goes further and puts the
book's own kernels side by side in CUDA and Metal, and `python pmpp/run_all.py` executes
them. So you are never reading code you cannot run — you are reading CUDA and immediately
running its Metal translation and watching the numbers come out.

---

## The per-lab loop (the 6 steps that matter)

Most of the learning in a lab happens if you do these in order. Skipping step 1 or 4 is why
people "do the tutorials" and learn nothing.

1. **Read the kernel source and predict.** Before running anything: which version will be
   fastest, and by roughly how much? Write it down. A wrong prediction is the most valuable
   thing that can happen in this repo.
2. **Run it. Check correctness first.** A wrong kernel's speed is meaningless.
3. **Compare against your prediction.** The gap between prediction and measurement *is* the
   lesson. Every "the textbook didn't hold" note in this repo came from that gap.
4. **Break it on purpose.** Delete a `threadgroup_barrier`, delete the bounds guard, launch
   one block instead of `blocks*threads`. Watch how it fails — silently and plausibly, every
   time. This is how you learn what each line is protecting against.
5. **Change one parameter and sweep it.** Tile size, block count, elements per thread. A
   sweep that comes out flat is a real result: that knob is not your bottleneck.
6. **Write three sentences in your own words.** What did you expect, what happened, why.
   Keep them in a file. If you cannot write the third sentence, re-read the chapter.

Checkpoint after each lab: `python -m pytest -q` runs that lab's kernel against a reference.

---

## The route (~3 weeks at 1-1.5 h/day)

Chapter **titles** are given, not numbers — editions differ and the numbering has moved.
Match them against your edition's table of contents.

| Session | PMPP (by title) | This repo | You are done when |
|---|---|---|---|
| 0 · 30 min | — | `docs/00`, `docs/01` | `python -m gpuk` prints your GPU, and you can state the `grid`-means-threads rule out loud |
| 1 · 1 h | *Heterogeneous Data Parallel Computing* | `labs/lab01` | your kernel matches `mx.add`, and you have broken the launch on purpose and seen it fail silently |
| 2 · 30 min | — | `docs/03` | you have run `bench/m4_peaks.py` yourself and know your machine's two numbers |
| 3 · 1 h | *Multidimensional Grids and Data* | extend `lab01` to a 2D grid yourself | you can write a 2D launch with the guards from scratch, no reference |
| 4-5 · 2 h | *Memory Architecture and Data Locality* | `labs/lab02` | you have removed the `+1` padding from `tile[32][33]` and measured what bank conflicts cost |
| 6-7 · 2 h | *Compute Architecture and Scheduling* | `labs/lab03` | you can write a `simd_sum`-based reduction from memory, and you know why v2 lost |
| 8-10 · 3 h | *Performance Considerations* + tiling | `labs/lab04` | 800+ GFLOP/s, and you can explain the roofline crossover without looking |
| 11 · 1 h | *Convolution* (5th ed: *filtering*) | `labs/lab05` | you can predict, from the halo formula, whether tiling will pay before measuring |
| 12-13 · 2 h | *Parallel Patterns: Prefix Sum (Scan)* | `labs/lab06` | the three-kernel scan passes, and you know why one kernel cannot do it |
| 14 · 1-2 h | *Parallel Patterns: Histogram* | write your own in `labs/` | a kernel you wrote from a blank file, tested, benchmarked |
| 15-17 · 3-4 h | *Deep Learning* / **LLM chapter (5th ed)** | softmax → FlashAttention | see below |

**Why this ordering and not PMPP-first:** sessions 1-13 are the fundamentals and they are
strictly sequential — each one uses the previous one's primitive. Softmax/attention needs
tiling (session 8-10), online reductions (6-7), and "count bytes not FLOPs" (11). Put it
first and you will be copying a kernel you cannot debug.

**Softmax → FlashAttention goes last on purpose.** It is the payoff: it is a tiled matmul
with a numerically-stable online reduction folded in, so it uses literally every idea from
sessions 4-13 at once. Done in order, it is a 3-hour exercise. Done first, it is a wall.

If you have the **5th edition (Feb 2026)**, its new *large language models* chapter plus the
new *advanced matrix multiplication optimizations* and *warp-level programming* chapters
line up almost exactly with sessions 8-17. That is the best-matched edition for the route
above.

---

## Rules that save you weeks

1. **Never trust one measurement.** This is a fanless laptop; runs differ by 20% and it
   throttles under sustained load. `gpuk.bench.best_ms` repeats and keeps the best round;
   treat sub-10% differences as ties.
2. **Warm up, synchronise, and size the working set.** Three separate traps, each of which
   can be off by 30%+. See `docs/03-how-to-measure.md`, which has a checklist.
3. **Compute the arithmetic intensity before optimising anything.** Below ~25 FLOP/byte on
   this machine, the win can only come from moving fewer bytes; above it, only from fewer
   FLOPs. Optimising the wrong side produces a flat line, which is the most common wasted
   week in GPU work.
4. **Measure a baseline and a floor.** One of the biggest lessons in this repo is that a
   0.67 ms 3x3 stencil and a 0.67 ms plain copy are *the same number*, and without the copy
   you would "optimise" a kernel that has nothing left to win.
5. **Never port a conclusion, port the method.** Four separate pieces of unanimous CUDA
   advice failed to reproduce on this hardware while building the repo (see
   `docs/02-your-hardware.md`). Apple's memory system is not NVIDIA's.
6. **Keep the CUDA files you read.** When you do rent a GPU (see `docs/04`), the `pmpp/**/*.cu`
   files are copy-paste ready and your git history records the experiments.

---

## When you get stuck

* **Kernel compiles but gives wrong numbers** → pass `verbose=True` to the kernel call. It
  prints the Metal source MLX generated, including the signature and buffer indices. Five
  seconds of reading usually finds it.
* **Kernel crashes the command buffer** → you wrote out of bounds. Re-check that every
  output array's declared shape covers the largest index the kernel touches. Widening a
  pointer type (`float*` → `float4*`) is the classic way to get this wrong.
* **Result is *sometimes* right** → a missing or misplaced `threadgroup_barrier`. Barrier
  bugs are timing-dependent and can pass on your machine and fail under different occupancy.
* **Numbers look too good** → you have a bug, not a discovery. A measurement above the
  hardware's nominal bandwidth means the kernel did not read what you think it did. The repo
  contains two worked examples of exactly this in `docs/02-your-hardware.md`.
* **Everything is slower than the vendor library** → that is normal and expected (labs
  01-04). Losing by 2x to Apple's tuned GEMM is a good result. Ask *why* — usually a
  different execution unit, not a better algorithm.

---

## What to do in week 4

Once the route above is done, pick one:

* **Depth.** Write the histogram / merge sort / sparse-matrix kernels yourself. Irregular
  access is the largest gap in this repo and the place where GPU intuition gets tested
  hardest.
* **Breadth.** Set up the CUDA cloud path from `docs/04` and run the book's own kernels on a
  real NVIDIA GPU. You will finally see the 10x coalescing and 100x tiling effects the
  literature talks about — and you will understand why they did not show up here.
* **Portability.** Set up `triton-msl` so one `@triton.jit` kernel runs on both your Mac and
  an NVIDIA GPU. Useful once you can already write the Metal version by hand.

---

*Back to the [`README`](../README.md) · PMPP topic mapping in [`04-pmpp-roadmap.md`](04-pmpp-roadmap.md).*
