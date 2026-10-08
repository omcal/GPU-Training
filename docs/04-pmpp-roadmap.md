# 04 · Following PMPP on a Mac

**PMPP** = *Programming Massively Parallel Processors: A Hands-on Approach*, Kirk, Hwu &
El Hajj. It is the standard CUDA textbook and it is worth working through properly. This
page is about doing that without an NVIDIA GPU in the machine.

Editions differ in chapter numbering. For reference:

| Edition | Year | Authors | ISBN |
|---|---|---|---|
| 3rd | 2016 | Kirk & Hwu | 978-0-12-811986-0 |
| 4th | 2022 | Hwu, Kirk & El Hajj | 978-0-323-91231-0 |
| **5th** | **Feb 2026** | Hwu, Kirk & El Hajj | 978-0-443-43900-1 |

The **5th edition is the best match for this repo**: it adds chapters on *filtering*,
*wavefront parallelism*, *advanced matrix multiplication optimizations* and **large language
models (LLMs)**, plus much more on warp-level programming, cooperative groups, CUDA C++
atomics and multi-GPU (NCCL/NVSHMEM). Filtering lines up with `lab05`, advanced matmul with
`lab04`, and the LLM chapter with the softmax/attention work you are heading towards.

Whatever edition you have, match the **topic table** below against your own table of
contents rather than trusting a chapter number from anywhere (including here).

---

## The honest situation

| Approach | Works on M4? | Verdict |
|---|---|---|
| Run CUDA directly | **No** | Apple GPUs have no CUDA runtime and no PTX. |
| SCALE (CUDA-compatible compiler) | **No** | Supports NVIDIA and AMD only. Apple Silicon is not a target. |
| HIP / ROCm | **No** | AMD only. |
| Metal Shading Language | **Yes** | The real GPU programming API on this platform. Used by every lab here. |
| MLX `mx.fast.metal_kernel` | **Yes** | Write MSL from Python, JIT-compiled at runtime. No Xcode needed. |
| Triton (via `triton-msl`) | **Partly** | Alpha. `@triton.jit` compiles to Metal; same source runs on NVIDIA. |
| Cloud NVIDIA GPU | **Yes** | The only way to run the book's CUDA verbatim. |

So the practical plan is:

1. **Do the concepts here, in Metal.** Chapters 2-10 of PMPP are about concepts (thread
   mapping, coalescing, tiling, reductions, scans, atomics, occupancy) and those transfer
   completely. The translation is mechanical — see
   [`01-cuda-to-metal.md`](01-cuda-to-metal.md).
2. **Rent a GPU when you need to see the CUDA-specific effects.** A few hours on a T4 is
   free on Colab; a few hours on an A100/H100 costs a few dollars.
3. **Keep one Triton file if you want portability.** `triton-msl` lets the same
   `@triton.jit` kernel run on your Mac and on a rented GPU — see the bottom of this page.

---

## Topic → lab mapping

| PMPP topic | Concepts | In this repo |
|---|---|---|
| Heterogeneous data-parallel computing | kernel launch, thread mapping, data parallelism | ✅ `labs/lab01_vector_add.py` |
| Multidimensional grids and data | 2D/3D grids, row-major addressing, boundary guards | ✅ `labs/lab01`, `labs/lab02`, `labs/lab04` |
| Compute architecture and scheduling | warps/simdgroups, occupancy, block scheduling, divergence | ✅ `docs/00-mental-model.md`, `labs/lab03` |
| Memory architecture and data locality | coalescing, cache lines, shared memory, tiling, bank conflicts | ✅ `labs/lab02_coalescing.py` |
| Performance considerations | roofline, ILP, occupancy, measuring correctly | ✅ `docs/03-how-to-measure.md`, `labs/lab03`, `labs/lab04` |
| Parallel patterns: **reduction** | tree reduction, warp shuffle, atomics, hierarchical reducers | ✅ `labs/lab03_reduction.py` |
| Parallel patterns: **convolution / stencil** | halo regions, tiled stencils, constant memory, when tiling pays | ✅ `labs/lab05_convolution.py` |
| Parallel patterns: **prefix sum / scan** | Blelloch vs Hillis-Steele, work efficiency, three-kernel hierarchical scan | ✅ `labs/lab06_scan.py` |
| Parallel patterns: **histogram** | atomics, contention, privatisation | 🔨 **your turn** |
| Parallel patterns: **merge sort / radix sort** | co-rank, sorting networks, digit histograms | 🔨 **your turn** |
| Parallel patterns: **sparse matrices / graph search** | CSR, SpMV, BFS frontiers, load balancing | 🔨 **your turn** |
| Deep learning | matmul, tiling, fusion, tensor cores, attention | ✅ `labs/lab04_matmul.py` (+ softmax/attention as the next lab) |

**"Your turn"** means: the concepts are all covered by the labs here, and you now have
every primitive you need (threadgroup memory, barriers, `simd_sum`, atomics, tiling) to
write it in Metal yourself. Each one is one small file in `labs/`, and `pmpp/` shows the
side-by-side translation pattern to copy.

---

## Reference: the canonical kernels, side by side

`pmpp/` contains CUDA and Metal versions of the same kernels, line for line, with notes
explaining each translation:

```
pmpp/01_vector_add/     kernel.cu  kernel.metal  notes.md
pmpp/02_tiled_matmul/   kernel.cu  kernel.metal  notes.md
```

Run them (the Metal ones actually execute and are checked):

```bash
python pmpp/run_all.py
```

---

## When you need an actual NVIDIA GPU

For the parts of PMPP that are genuinely NVIDIA-specific — `cp.async` pipelines, TMA,
`__launch_bounds__`, occupancy tuning against an SM, Nsight Compute, and the tensor-core
programming model — you need hardware. Options, cheapest first:

| Option | Cost | Notes |
|---|---|---|
| **Google Colab** | free | A T4 with ~16 GB. Enough for every non-tensor-core exercise in the book. Sessions time out; keep work in files. |
| **Modal** | free credits | Serverless; good for "run this one kernel on an A100 for 10 seconds". |
| **Vast.ai / RunPod** | ~$0.15-0.50/hr | Cheapest per hour; a 3090 or A5000 is plenty. |
| **Lambda / CoreWeave** | ~$1-3/hr | Better uptime, more setup. |
| **University / work cluster** | free | Ask; most have A100s sitting behind a Slurm queue. |

Practical advice: **write the kernel here first.** Iterate on logic, boundary conditions
and correctness on the M4 where the edit-run loop is a second long. Then take the same
source to the cloud for the performance pass. That is precisely the workflow `triton-msl`
was built for, and it works for CUDA too — the algorithmic content of a kernel port is
mostly mechanical once the logic is right.

Keep the CUDA files in this repo (`pmpp/**/*.cu`) so the cloud session is a copy-paste
and your git history records the experiments.

---

## Want one source that runs on both?

`triton-msl` compiles standard `@triton.jit` kernels to Metal. Same source, same
semantics; the Triton IR is lowered to Metal Shading Language, and the same kernels also
run through Triton's NVIDIA and AMD backends.

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask)
             + tl.load(y_ptr + offsets, mask=mask), mask=mask)
```

This is a genuinely useful middle path: **Triton's programming model is the PMPP model**
(programs = threadgroups, `tl.load`/`tl.store` with masks, block-level tiles, explicit
`tl.constexpr` tile shapes). A kernel written for Triton on your Mac will usually run
unchanged on an H100.

Caveats, stated plainly:

* It is **alpha software**. It refuses unsupported kernels explicitly rather than
  silently miscompiling, which is the right design, but you will hit refusals.
* Requires PyTorch, plus Triton built from a pinned source revision — a real install.
* Performance has known regressions relative to previous versions; treat it as a
  correctness/prototyping tool, not a production path.
* Triton's abstraction hides exactly the low-level details (barriers, bank conflicts,
  occupancy) that the labs here are designed to teach. Use it *after* you can write the
  Metal version by hand, not instead of.

It is deliberately not installed by this repo. If you want it, set it up in a separate
virtualenv — the pinned Triton build is heavy and will fight with your MLX environment.

---

## Suggested order

**The full study route, with session-by-session timing and the three-pass method for
reading a chapter alongside its lab, lives in [`05-learning-route.md`](05-learning-route.md).**
That is the document to follow; this one is the reference map it points into.

The short version:

1. `docs/00-mental-model.md` → read once.
2. `docs/01-cuda-to-metal.md` → keep open beside PMPP.
3. `labs/lab01` → `lab02` → `lab03` → `lab04`, in order. Read the source, run it,
   change a constant, run it again.
4. Read PMPP's matching chapter *after* each lab. The lab gives you the "why it feels
   like this"; the chapter gives you the theory and the vocabulary.
5. Pick a **your turn** topic from the table above and write it in `labs/`. That is where
   the learning actually consolidates.
6. Only then consider the cloud for the NVIDIA-specific material.

---

*Back to the [`README`](../README.md).*
