# GPU-Training

[![CI](https://github.com/omcal/GPU-Training/actions/workflows/ci.yml/badge.svg)](https://github.com/omcal/GPU-Training/actions/workflows/ci.yml)


Hands-on GPU kernel writing on Apple Silicon and NVIDIA CUDA, mapped to PMPP.

**8 Ekim–1 Kasım 2026 çalışma programı:**
[PMPP + AI Systems Performance Engineering planı](docs/08-cuda-inference-study-plan.md).
[Bağımsız egzersizler ve ilk oturum](study/README.md) · [İlerleme](study/progress.md).
Bu takvim CUDA/SSH yolu için güncel programdır; aşağıdaki Metal rotası ayrı kaynaktır.


**Ubuntu / NVIDIA:** the CUDA route is in [`cuda/README.md`](cuda/README.md).
The configured `ssh cuda` machine has CUDA 12.9 and a GTX 1050 Ti. Run all six
CUDA labs and the two PMPP examples with:

```bash
ssh cuda 'cd ~/GPU-Training && make -C cuda test'
```

The rest of this README describes the **Mac / Metal route**.

You have an M4 MacBook Air — **CUDA kernels cannot run locally on this Apple GPU.** This repo
is the workaround, and it is a good one: Apple GPUs have a real, modern compute API
(Metal), and you can write and run actual GPU kernels on it from Python in a second-long
edit-run loop.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
python labs/lab01_vector_add.py
```

---

## The situation, honestly

| Approach | Works on M4? |
|---|---|
| CUDA / `nvcc` / PTX | ❌ No CUDA runtime, no NVIDIA GPU |
| SCALE (the CUDA-compatible compiler) | ❌ Supports NVIDIA and AMD only |
| HIP / ROCm | ❌ AMD only |
| **Metal Shading Language** | ✅ The real GPU compute API on this platform |
| **MLX `mx.fast.metal_kernel`** | ✅ Write MSL from Python, JIT-compiled at runtime |
| Triton via `triton-msl` | ⚠️ Alpha, works, needs a heavy separate install |
| Cloud NVIDIA GPU | ✅ The only way to run PMPP's CUDA verbatim |

**You do not need Xcode.** MLX compiles Metal at runtime through the OS's
`Metal.framework`, so writing kernels works with only the Command Line Tools installed.
(Install full Xcode only when you want the offline `metal` compiler or the Metal
Debugger.)

**You do not need to give up on PMPP.** Chapters 2-10 are about concepts — thread
mapping, coalescing, tiling, reductions, atomics, occupancy — and those transfer to Metal
almost one-to-one. The syntax does not. [`docs/01-cuda-to-metal.md`](docs/01-cuda-to-metal.md)
is the translation table, and [`pmpp/`](pmpp/) has the same kernels written in both
languages, side by side.

---

## The one thing that will bite you

```cuda
my_kernel<<<gridDim, blockDim>>>(args);   // CUDA: first arg counts BLOCKS
```

```python
kernel(grid=(blocks * threads, 1, 1), threadgroup=(threads, 1, 1))   # Metal: THREADS
```

Metal's `dispatchThreads` takes the **total number of threads**, not a block count. Get it
wrong and nothing errors — you just silently compute the wrong thing. Use
`gpuk.launch.cuda(blocks, threads)`, which exists so you only have to get it right once.

---

## Layout

```
docs/                 the reading material
  05-learning-route.md  **START HERE** — the route, the method, the session plan
  00-mental-model.md    SIMT, the memory hierarchy, occupancy, the roofline
  01-cuda-to-metal.md   the full translation table — keep this open beside PMPP
  02-your-hardware.md   what this M4 measures, and what surprised me
  03-how-to-measure.md  how to benchmark a GPU without fooling yourself
  04-pmpp-roadmap.md    PMPP topics -> labs, and how to get a CUDA GPU when you need one

labs/                 the main event — read, run, change, re-run
  lab01_vector_add.py   launch geometry and the memory bound
  lab02_coalescing.py   cache-line amplification, threadgroup memory, bank conflicts
  lab03_reduction.py    the four levels of the GPU hierarchy
  lab04_matmul.py       tiling, register blocking, the roofline
  lab05_convolution.py  the halo, and when tiling a stencil is not worth it
  lab06_scan.py         prefix sum: work efficiency, and why it needs three kernels

cuda/                 native CUDA C++ labs, CPU checks, and CUDA-event benchmarks
pmpp/                 the same kernels in CUDA and Metal, with line-by-line notes
gpuk/                 the small shared harness (device, launch, bench, check, peaks)
bench/m4_peaks.py     re-measure this machine's ceilings
tools/device_probe.swift   asks Metal directly for the hardware limits
tests/                correctness tests for every kernel (python -m pytest)
```

## What the labs actually measure

Measured on an M4 MacBook Air (8-core GPU, 16 GiB), MLX 0.32.2. Re-measure with
`python bench/m4_peaks.py`; laptop numbers move.

```
streaming 1R:1W ceiling         ~92-97 GB/s     (M4 is rated ~120 GB/s nominal)
streaming 2R:1W ceiling         ~94 GB/s
FP32 GEMM ceiling               ~2400 GFLOP/s   (uses simdgroup_matrix)
roofline crossover              ~26 FLOP/byte
```

| Lab | Result |
|---|---|
| 01 vector add | hand-written kernel matches Apple's `mx.add` — it is memory bound and there is nothing left to win |
| 02 transpose | tiled is 1.1-1.5x faster than naive; a **64x65 tile is 5.9x slower** than 32x33 purely from occupancy |
| 03 reduction | 16.7M-atomic version beats the textbook tree; simdgroup-hierarchical wins at 74 GB/s |
| 04 matmul | 248 → 410 → 809 GFLOP/s across three versions, vs 1803 for Apple's tuned GEMM |
| 05 convolution | a 3x3 stencil costs **the same as a plain copy** (so tiling it is wasted code); tiling only pays from K≥2, and then by a consistent ~1.6x |
| 06 scan | three-kernel hierarchical scan is **1.6x faster than `mx.cumsum`** at 1M elements |

Several of those results **contradict the CUDA literature** — contended atomics are cheap
here, tiled transpose wins by little, `float4` did not reliably help, staging the
convolution filter in on-chip memory (unanimous PMPP advice) measured *slower*, and a 3x3
stencil is not bandwidth bound at all. That is not a bug in the labs. It is the first real
lesson of GPU work, and [`docs/02-your-hardware.md`](docs/02-your-hardware.md) goes through
each one: Apple's memory system is not NVIDIA's, and ported conclusions are how people
waste weeks. Measure your own hardware.

## Suggested path

**Follow [`docs/05-learning-route.md`](docs/05-learning-route.md)** — it has the
session-by-session plan and the method. The shape of it:

1. Read [`docs/00-mental-model.md`](docs/00-mental-model.md) once, and keep
   [`docs/01-cuda-to-metal.md`](docs/01-cuda-to-metal.md) open beside PMPP.
2. Per topic: **skim the PMPP chapter → do the matching lab → re-read the chapter.** Never
   read the book cover to cover first: its code is CUDA you cannot run here, so the
   vocabulary has nothing to attach to.
3. Work through `labs/lab01` → `lab06`, in order.
4. Write one of the "not written yet" kernels from [`labs/README.md`](labs/README.md) from a
   blank file.
5. `softmax` → `FlashAttention` last — it uses every idea from labs 02-06 at once.
6. Rent a GPU for a few hours when you want the NVIDIA-specific material. See
   [`docs/04-pmpp-roadmap.md`](docs/04-pmpp-roadmap.md).

## CI and releases

Pushes and pull requests run portable repository checks and compile all CUDA labs
and PMPP examples with CUDA 12.9 targeting `sm_61`. Compiled Linux programs are
available as workflow artifacts. GPU correctness runs separately on real hardware;
a green compile check does not mean a kernel has executed.

Tags such as `v0.1.0` run the same checks and publish source `.tar.gz` / `.zip`
archives with SHA-256 checksums to GitHub Releases. Setup and local validation:
[CI and release guide](docs/09-ci-and-releases.md).

## Environment

```
Python 3.12 · MLX 0.32.2 · numpy · matplotlib · pytest
macOS 26.6.2 · Metal 4 · Apple GPU family apple9
```

`gpuk/` is installed editable, so `import gpuk` works from anywhere in the repo. Run
`python -m gpuk` for a full GPU report and
`swift -module-cache-path .swiftcache tools/device_probe.swift` for the raw Metal limits.
