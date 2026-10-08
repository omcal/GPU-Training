# 00 · The mental model

Read this once before the labs, and come back to it after lab03. It is the shortest
version of the model you need in order to predict what a kernel will do before you run
it.

---

## 1. A GPU is a throughput machine, not a fast CPU

A CPU core is built to run *one* instruction stream as fast as possible: branch
prediction, out-of-order execution, deep caches — all so that a single thread's latency
is short. An Apple M4 performance core is a marvel of that design.

A GPU is built to run *enormous numbers of instruction streams* and to keep its
execution units busy even though each individual memory access takes ~500 ns. It has no
branch predictor worth the name and no out-of-order window. Instead it has thousands of
threads, and when one stalls on memory the hardware switches to another.

The consequence shapes every kernel you will write:

> **You do not optimise a GPU kernel by making one thread faster. You optimise it by
> keeping the machine fed.**

An M4 GPU (8 cores) has 8 × 128 = 1024 FP32 lanes. That is why `labs/lab01` shows your
hand-written vector add matching Apple's own kernel: the arithmetic is irrelevant
(1 FLOP per 12 bytes), and the only thing that matters is feeding it.

---

## 2. SIMT and the simdgroup

Threads do not execute independently. The hardware groups them into **simdgroups** of 32
lanes (NVIDIA calls this a *warp*) that share one instruction stream. All 32 lanes
execute the same instruction on different data.

Two consequences:

* **Occupancy is bought in units of 32.** 33 threads cost as much as 64.
* **Divergence is expensive.** If lanes in a simdgroup take different branches, the
  hardware runs both paths serially with lanes masked off. `if (x > 0) ... else ...`
  costs you *both* sides when the 32 lanes disagree.

This is why GPU kernels are written branch-free where possible — e.g. `max(x, 0)`
instead of `if (x < 0) x = 0`.

---

## 3. The hierarchy

```
grid
 └── threadgroup   (CUDA: block)        up to 1024 threads on Apple GPUs
      ├── simdgroup (CUDA: warp)        32 lanes, share an instruction stream
      │    └── thread
      └── threadgroup memory            on-chip, shared by the threadgroup, 32 KiB max
```

Four levels of memory, each roughly an order of magnitude apart:

| Level | Scope | Latency (order) | Size here |
|---|---|---|---|
| Registers | one thread | ~1 cycle | ~thousands per thread |
| threadgroup memory | one threadgroup | ~20-30 cycles | **32 KiB total** |
| L2 cache | whole GPU | ~100 cycles | several MB |
| DRAM (unified) | whole system | ~500 cycles | 16 GiB |

The whole craft of GPU kernel writing is **moving data up this table and reusing it
before it falls back down**. `labs/lab04_matmul.py` is that idea taken to its
conclusion: the naive kernel reads from DRAM O(M·N·K) times, the tiled one O(M·K + K·N +
M·N), and the register-blocked one reuses each on-chip value four times before touching
memory again. That is where the 3x comes from — not from any arithmetic trick.

---

## 4. Latency hiding and occupancy

A memory access takes ~500 cycles. The GPU has no out-of-order engine to cover that, so
it covers it with *other threads*. If enough simdgroups are resident, the scheduler
always has someone to run and the memory system stays saturated.

**Occupancy** is the fraction of the GPU's thread capacity that is actually resident.
The two things that limit it on Apple GPUs:

1. **Threads per threadgroup / total threads.** A threadgroup of 1024 threads is the
   maximum; if your kernel has only 256 threads per threadgroup, you need 4 threadgroups
   resident per core to fill it.
2. **Threadgroup memory per threadgroup.** This is the one that surprises people. The
   32 KiB is shared with the L1 cache. A kernel asking for 16 KiB per threadgroup can
   only have one threadgroup resident — and one threadgroup is not enough warps to hide
   500-cycle latency. `labs/lab02_coalescing.py` contains a transpose kernel that is
   **5.9x slower** than an equivalent one purely because its tile is 64x64 instead of
   32x32.

Rule of thumb: keep threadgroup memory per threadgroup under ~8 KiB unless you have
measured that you need more.

---

## 5. Bandwidth vs FLOPs: the roofline

Every kernel lives under one of two ceilings:

* **bandwidth ceiling** — bytes/second the memory system can move
* **compute ceiling** — FLOPs/second the execution units can issue

and the ratio decides which one you hit:

```
arithmetic intensity (FLOP/byte) = FLOPs performed / bytes moved

if intensity > peak_flops / peak_bandwidth   ->  compute bound
else                                          ->  memory bound
```

On this machine (measured; see `docs/02-your-hardware.md`):

```
~94 GB/s  and  ~2400 GFLOP/s fp32   ->  crossover at ~26 FLOP/byte
```

Concretely:

| Kernel | FLOP/byte | Bound by |
|---|---|---|
| vector add | 1/12 ≈ 0.08 | memory (hopelessly) |
| reduction (sum) | ~1/4 | memory |
| softmax | ~2 | memory |
| matmul 1024³ | ~171 | compute |
| attention (typical) | ~10-50 | usually memory |

**This is the single most useful diagnostic in GPU work.** If you are memory bound, the
only thing that helps is moving fewer bytes (fusion, better access patterns, lower
precision). If you are compute bound, only fewer/cheaper FLOPs help. Optimising the wrong
side of the roofline produces no measurable change, which is why `docs/03` insists you
measure before you optimise.

---

## 6. Asynchrony, and why naive benchmarking lies

A kernel launch is *asynchronous*: the CPU hands work to the GPU and returns immediately.
MLX adds a second layer — it is **lazy**, building a graph and computing only when you
call `mx.eval` or `mx.synchronize`.

So this measures nothing:

```python
t0 = time.perf_counter()
kernel(...)          # just enqueues
t1 = time.perf_counter()   # ~0 ms, GPU hasn't started
```

Correct timing is in `gpuk.bench`, and the traps are worth knowing by name:

* **Not synchronising** → you measured Python's graph-building speed.
* **Not warming up** → the first dispatch compiles a Metal shader library. Averaging
  that into 20 iterations can understate bandwidth by 30%+. Every benchmark here uses
  `warmup=5` or more.
* **A working set that is too small** → launch overhead dominates. A 4 MiB streaming
  kernel measures about half the bandwidth of a 512 MiB one on this machine.
* **Thermal throttling** → a fanless MacBook Air running sustained GPU load will slow
  down. Repeated identical runs can differ by 20%. Use `gpuk.bench.best_ms`.

---

## 7. Unified memory changes the shape of the problem

On a discrete GPU, the CPU and GPU have separate memory and you spend real effort on
transfers. On Apple Silicon there is one pool of RAM. Consequences:

* No copies. An MLX array is already "on the GPU".
* The GPU's usable budget is `recommendedMaxWorkingSetSize` — **11.8 GiB** of your 16 GiB
  here. Beyond that macOS starts swapping and performance falls off a cliff.
* The CPU and GPU share bandwidth. A benchmark running while a browser is compositing
  will read low.
* `max_buffer_length` is 8.88 GiB — a *single* array cannot exceed that.

---

## 8. What to actually remember

1. Kernels are for throughput; keep the machine fed.
2. 32-lane simdgroups; divergence costs both paths.
3. Registers > threadgroup memory > L2 > DRAM. Move data up and reuse it.
4. Occupancy is limited by threads *and* threadgroup memory (32 KiB, shared with L1).
5. Compute the arithmetic intensity first; it tells you where the win is.
6. Measure with warm-up, synchronisation, and a working set big enough to matter.

---

*Next: [`01-cuda-to-metal.md`](01-cuda-to-metal.md) to translate this into the syntax
you will actually type.*
