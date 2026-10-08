# 01 · CUDA → Metal: the translation table

You are learning GPU kernels from CUDA material (PMPP) on a machine that has no CUDA.
This page is the bridge. The concepts transfer almost completely; the spelling does not.

Internalise this first: **a GPU kernel is a function you write once and the hardware
runs N times in parallel.** Everything else — thread ids, memory spaces, barriers — is
about letting those N copies cooperate.

---

## The one difference that will bite you

```cuda
// CUDA: the first argument counts BLOCKS
my_kernel<<<gridDim, blockDim>>>(args);   // 16 blocks x 256 threads = 4096 threads
```

```python
# MLX/Metal: `grid` counts THREADS
kernel(inputs=..., grid=(4096, 1, 1), threadgroup=(256, 1, 1))
```

`grid` is what Metal calls `threadsPerGrid` — the **total** number of threads.
`threadgroup` is `threadsPerThreadgroup` — threads per block.

So `<<<blocks, threads>>>` becomes `grid=(blocks * threads,)`, not `grid=(blocks,)`.

This does not raise an error when you get it wrong. It silently launches the wrong
number of threads and your answer is quietly, plausibly wrong. `gpuk.launch.cuda(blocks,
threads)` exists so you only have to get it right once:

```python
from gpuk import cuda
cuda(16, 256).kwargs()   # -> {'grid': (4096, 1, 1), 'threadgroup': (256, 1, 1)}
```

A second Metal constraint: every `threadgroup` dimension must be `<=` the matching
`grid` dimension.

---

## Identifiers and built-ins

Inside the kernel body, MLX generates the signature for you. You write the body using
Metal attributes directly.

| CUDA | Metal (MSL) | Notes |
|---|---|---|
| `threadIdx.x` | `thread_index_in_threadgroup` | scalar, flattened |
| — | `thread_position_in_threadgroup` | `uint3`, per-axis |
| `blockIdx.x` | `threadgroup_position_in_grid` | this is the "block" |
| `blockDim.x` | `threads_per_threadgroup` | |
| `gridDim.x` | `threadgroups_per_grid` | **not** total threads |
| `blockIdx.x*blockDim.x + threadIdx.x` | `thread_position_in_grid` | free, use this |
| — | `thread_index_in_simdgroup` | `threadIdx.x % 32` |
| — | `threads_per_grid` | total threads in the dispatch |
| — | `simdgroup_index_in_threadgroup` | `threadIdx.x / 32` |

Global thread id: CUDA makes you compute it; Metal hands it to you as
`thread_position_in_grid`. Use it.

---

## Execution model

| CUDA | Metal | Notes |
|---|---|---|
| warp (32 lanes) | **simdgroup** (32 lanes) | same hardware concept |
| `__syncwarp()` | `simdgroup_barrier(mem_flags::mem_threadgroup)` | |
| `__shfl_down_sync(m, v, o)` | `simd_shuffle_down(v, o)` | no mask argument needed |
| `__shfl_xor_sync` | `simd_shuffle_xor` | |
| `__shfl_sync` | `simd_shuffle` | |
| `__ballot_sync` | `simd_ballot` | |
| `__reduce_add_sync` | **`simd_sum(v)`** | one instruction; no mask |
| `__reduce_max_sync` | `simd_max(v)` | |
| — | `simd_prefix_exclusive_sum`, `simd_product`, … | Metal has a fuller set |

`simd_sum` / `simd_max` / `simd_min` / `simd_product` / `simd_prefix_*` collapsing a
whole warp reduction into one builtin is a genuine Metal convenience. In CUDA you write
the shuffle ladder yourself. `__reduce_add_sync` needs CC 8.0+; `simd_sum` works
everywhere.

**Divergence rule:** a `simd_sum` must be reached by the *whole* simdgroup. If you call
it inside `if (lane < 8)`, it is undefined. Guard the *store* by the lane id, never the
shuffle itself:

```metal
float s = simd_sum(v);                   // all 32 lanes execute
if (thread_index_in_simdgroup == 0) out[tg] = s;   // only lane 0 stores
```

---

## Memory

| CUDA | Metal | Notes |
|---|---|---|
| `__shared__ float s[N];` | `threadgroup float s[N];` | on-chip, per threadgroup |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | |
| `__constant__ float c[N];` | `constant float c[N];` | function-scope in MSL |
| global `float* p` | `device float* p` | the default for kernel args |
| `cudaMemcpy` H↔D | *(nothing)* | unified memory; see below |
| `atomicAdd(&x, v)` | `atomic_fetch_add_explicit(&x, v, memory_order_relaxed)` | |
| `atomicCAS` | `atomic_compare_exchange_weak_explicit` | |
| `__ldg(&p[i])` | `p[i]` | Metal caches reads by default |
| `float4` | `float4` | identical; 16 B per lane |
| `cp.async` (Ampere+) | `simdgroup_async_copy` / staging | different shape, same idea |
| `wmma` / `mma.sync` | `simdgroup_matrix<float, 8, 8>` | Apple's "tensor core" |

Hard limits on this machine (measured — see `docs/02-your-hardware.md`):

```
max threads per threadgroup      1024
max threadgroup memory           32768 bytes (32 KiB) per threadgroup
max buffer length                8.88 GiB
```

That 32 KiB is *shared with the L1 cache*. Asking for 16 KiB per threadgroup is legal
and will destroy your occupancy — `labs/lab02_coalescing.py` demonstrates exactly this.

---

## Library and language surface

| CUDA | MLX / Metal |
|---|---|
| `expf`, `sqrtf`, `powf` | `metal::exp`, `metal::sqrt`, `metal::pow` |
| `fmaf(a,b,c)` | `fma(a,b,c)` |
| `__saturatef` | `metal::saturate` |
| `printf` in kernel | not available |
| `malloc` in kernel | not available |
| `__device__` globals | `device` / `constant` address-space globals |
| templates | C++14 templates; MLX passes them via `template=[...]` |
| `constexpr` | `constant` + template parameters |
| `#pragma unroll` | `#pragma unroll` (same) |

---

## Launching, from Python

MLX generates the MSL function signature from the arrays you pass. **You write only the
kernel body.** For an input named `inp`, the generated signature contains:

```metal
const device float* inp [[buffer(0)]],
device float* out      [[buffer(1)]],
uint3 thread_position_in_grid [[thread_position_in_grid]]
```

…and, if the name appears in your source, `inp_shape`, `inp_strides`, `inp_ndim` too:

```python
src = """
    uint i = thread_position_in_grid.x;
    if (i < inp_shape[0]) out[i] = inp[i] * 2.0f;
"""
k = mx.fast.metal_kernel(name="double_it", input_names=["inp"], output_names=["out"], source=src)
out = k(inputs=[x], output_shapes=[x.shape], output_dtypes=[x.dtype],
        grid=(x.size, 1, 1), threadgroup=(256, 1, 1))[0]
mx.eval(out)   # MLX is lazy: nothing runs until you ask
```

Useful options:

| Option | Where | Why |
|---|---|---|
| `template=[("T", mx.float32)]` | call | dtype-generic kernels; instantiated per dtype |
| `template=[("TILE", 32)]` | call | compile-time constants that fully unroll loops |
| `ensure_row_contiguous=False` | constructor | take `inp_strides` and index it yourself |
| `atomic_outputs=True` | constructor | declare outputs `atomic` so you can `atomic_fetch_add` |
| `init_value=0` | call | zero the outputs before the kernel runs |
| `compile_options={"math_mode": "fast"}` | constructor | relax IEEE rules (`safe` is the default) |
| `verbose=True` | call | **print the generated MSL** — the single best debugging tool |

`verbose=True` is how you learn what MLX is actually compiling. Use it early and often.

---

## What has no Metal equivalent

* **Host/device memory split.** Apple GPUs are unified-memory: the CPU and GPU share the
  same physical RAM, and an MLX array lives in it with no copy. There is no
  `cudaMemcpy`, no pinned memory, no "copy to device" step. Practically this removes a
  whole class of CUDA bugs — and adds a new one: the GPU and CPU *compete* for the same
  bandwidth and capacity, which is why `recommendedMaxWorkingSetSize` (11.8 GiB here)
  matters.
* **`cudaMalloc` inside a kernel / dynamic parallelism.** Not available.
* **`__syncthreads_count` / `__syncthreads_and`.** Use `simd_*` vote builtins plus
  threadgroup memory instead.
* **`__launch_bounds__`.** Metal infers it; control occupancy through threadgroup size
  and threadgroup memory usage instead.
* **PTX / inline SASS.** Metal has no inline-assembly escape hatch at all. What the
  compiler does is what you get.
* **Occupancy API.** There is no `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. You
  infer occupancy from threads-per-threadgroup and threadgroup memory, and confirm with
  the Metal Debugger.

---

## Reading PMPP while writing Metal

Keep PMPP open and a Metal file next to it. When PMPP says:

| PMPP (CUDA) | You write (Metal) |
|---|---|
| "each thread computes `blockIdx.x*blockDim.x + threadIdx.x`" | `thread_position_in_grid.x` |
| "load the tile into `__shared__` memory and call `__syncthreads()`" | `threadgroup` array + `threadgroup_barrier` |
| "use warp shuffle to reduce, then one atomic per block" | `simd_sum`, then `atomic_fetch_add` from lane 0 |
| "tile size is limited by shared memory / SM" | limited by 32 KiB threadgroup memory |
| "coalesced access to maximize DRAM bandwidth" | exactly the same, same 128 B lines |
| "use `float4` to improve memory throughput" | identical |
| "tensor cores via `wmma`" | `simdgroup_matrix`, or just use `mx.matmul` |

The conceptual content of chapters 2–10 maps almost one-to-one. What does not map is
anything about a specific NVIDIA feature: PTX, `cp.async` pipelines, TMA, and the
tensor-core programming model beyond the basic idea.

---

*Next: [`02-your-hardware.md`](02-your-hardware.md) for what this specific M4 can do, and
[`04-pmpp-roadmap.md`](04-pmpp-roadmap.md) for the chapter-by-chapter mapping.*
