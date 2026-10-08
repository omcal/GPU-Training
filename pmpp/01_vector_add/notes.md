# 01 · Vector add — CUDA vs Metal

The simplest data-parallel kernel. Everything that differs between the two languages is
already visible here in eight lines.

## Line by line

| CUDA | Metal | Why |
|---|---|---|
| `__global__ void f(...)` | `[[kernel]] void f(...)` | An attribute instead of a keyword. |
| `const float* a` | `const device float* a` | Metal requires an explicit *address space*: `device`, `constant`, `threadgroup`, or `thread`. |
| `int n` | `constant uint* a_shape` | MLX injects array metadata as buffers. In hand-written Metal you would pass `constant uint& n [[buffer(3)]]`. |
| `int i = blockIdx.x*blockDim.x + threadIdx.x` | `uint i = thread_position_in_grid.x` | Metal pre-computes the global index as a *thread attribute* passed as a normal function parameter. |
| `if (i < n)` | `if (i < a_shape[0])` | Identical concept; only the name of `n` changes. |
| `c[i] = a[i] + b[i]` | `c[i] = a[i] + b[i]` | **Identical.** This is the point. |
| `vector_add<<<blocks, 256>>>(...)` | `grid=(blocks*256,), threadgroup=(256,)` | ⚠️ **The one dangerous difference.** |
| `cudaMalloc` + `cudaMemcpy` | *(nothing)* | Unified memory: the array is already on the GPU. |
| `cudaDeviceSynchronize()` | `mx.synchronize()` | Waiting is still required — GPU work is asynchronous on both. |

## The launch

```cuda
const int blocks = (n + 255) / 256;
vector_add<<<blocks, 256>>>(d_a, d_b, d_c, n);   // `blocks` counts BLOCKS
```

```python
launch = cuda(cdiv(n, 256), 256)                 # gpuk.launch.cuda() -> Launch
out = k(inputs=[a, b], output_shapes=[a.shape], output_dtypes=[a.dtype],
        **launch.kwargs())                       # grid=(blocks*256, 1, 1), threadgroup=(256,1,1)
```

Metal's `dispatchThreads(threadsPerGrid, threadsPerThreadgroup)` takes the **total**
number of threads, not a block count. Every other part of the CUDA mental model survives
contact with Metal; this one does not, and the failure mode is silent wrong answers
rather than an error. `gpuk.launch.cuda(blocks, threads)` is the translation, and it
raises if you ask for more than 1024 threads per threadgroup (the Apple GPU limit).

## What to notice about performance

The CUDA version in the book is presented as having three "obvious" optimisations
(coalescing, enough threads, no divergence) — and it is at the roofline already. That is
true on the M4 too: `labs/lab01_vector_add.py` measures a hand-written Metal vector add
within a few percent of Apple's own kernel.

One FLOP per 12 bytes means the kernel is hopelessly memory bound. Arithmetic intensity
≈ 0.08 FLOP/byte against a crossover of ≈26. Nothing you do to the arithmetic will show
up in the timing.

```bash
python labs/lab01_vector_add.py     # the runnable, benchmarked version of this kernel
```
