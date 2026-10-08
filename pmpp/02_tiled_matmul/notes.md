# 02 · Tiled matrix multiply — CUDA vs Metal

This is where the two languages stop looking alike and start being *the same program in
different clothes*. Every structural idea in the CUDA kernel — the tile, the two
barriers, the coalesced load, the block-to-tile mapping — is present in the Metal version
under a different name.

## Line by line

| CUDA | Metal | Notes |
|---|---|---|
| `__shared__ float As[TILE][TILE];` | `threadgroup float As[16][16];` | The name changed; the hardware is the same on-chip scratchpad. |
| `threadIdx.x % TILE` | `thread_index_in_threadgroup % 16` | Flattened per-threadgroup index in both. |
| `threadIdx.x / TILE` | `thread_index_in_threadgroup / 16` | |
| `blockIdx.x` | `threadgroup_position_in_grid.x` | |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | Metal makes the address spaces being ordered explicit. |
| `int K` | `A_shape[1]` | MLX injects operand shapes; hand-written Metal would pass `constant uint& K`. |
| `C[row*N + col]` | `C[row * B_shape[1] + col]` | Identical indexing. |
| `matmul_tiled<<<blocks, threads>>>(...)` | `grid=(blocks*256,), threadgroup=(256,)` | Same gotcha as always. |

## The three things tiling buys, and what they cost

**Buys.** Each element of A and B is loaded from global memory once per *tile* instead of
once per *use*. With a 16-wide tile that is a 16x reduction in global traffic; the naive
kernel re-reads from DRAM O(M·N·K) times while the tiled one reads O(M·K + K·N + M·N).

**Costs.** Two barriers per K-step, so at K=1024 with 16-wide tiles that is 128 barriers
per threadgroup. And the tile must fit in on-chip memory.

**The hard limit on Apple GPUs.** `maxThreadgroupMemoryLength` = **32768 bytes**, shared
with the L1 cache — verified by `tools/device_probe.swift`. Two 16x16 float tiles use
2 KiB, which is comfortable. Going to 64x64 tiles costs 16.6 KiB and collapses to one
resident threadgroup with no warps left to hide memory latency:

```
transpose 4096x4096   32x33 tile, 256 threads    1.69 ms
transpose 4096x4096   64x65 tile, 256 threads    9.92 ms   (5.9x slower, same algorithm)
```

Measured in `labs/lab02_coalescing.py`. On NVIDIA the same tradeoff exists, but the
budget is larger (up to 100+ KiB per SM on recent parts) and the tuning point is
different. Always query the device; never port a tile size.

## Measured on the M4

From `labs/lab04_matmul.py`, at 1024x1024x1024 float32:

```
v1 naive  (one thread/output, global K-loop)   248 GFLOP/s
v2 tiled  16x16, one output per thread         410 GFLOP/s
v3 32x32 tile + 2x2 register micro-tile        809 GFLOP/s
mx.matmul (Apple's tuned GEMM)                1803 GFLOP/s
```

The 7x from v1 to v3 is entirely on-chip reuse. The remaining 2.2x gap to Apple's GEMM is
a *different execution unit*, not a better algorithm: `simdgroup_matrix` is Apple's
equivalent of the tensor cores that PMPP's later chapters are about, and a plain-FMA
kernel cannot reach it.

## Running these

```bash
python pmpp/run_all.py                          # compiles + runs both Metal kernels here
python labs/lab04_matmul.py                     # the full benchmarked progression
```

The `.cu` files are for a CUDA machine — see
[`../../docs/04-pmpp-roadmap.md`](../../docs/04-pmpp-roadmap.md) for how to get a few
hours on an NVIDIA GPU without buying one.
