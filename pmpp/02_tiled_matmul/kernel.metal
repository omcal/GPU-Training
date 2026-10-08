// PMPP tiled matrix multiply, in Metal Shading Language.
//
// Complete, compilable MSL. The region between the markers is what you pass to MLX as
// `source=`; `pmpp/run_all.py` extracts and runs exactly that region.
//
//   python pmpp/run_all.py

#include <metal_stdlib>
using namespace metal;

// CUDA:
//   __global__ void matmul_tiled(const float* A, const float* B, float* C,
//                                int M, int N, int K)
[[kernel]] void matmul_tiled(
    const device float* A                 [[buffer(0)]],
    const device float* B                 [[buffer(1)]],
    device float*       C                 [[buffer(2)]],
    constant uint*      A_shape           [[buffer(3)]],   // (M, K)
    constant uint*      B_shape           [[buffer(4)]],   // (K, N)
    uint                thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3               threadgroup_position_in_grid [[threadgroup_position_in_grid]])
{
    // --- MLX body begin ---
    // CUDA `__shared__ float As[TILE][TILE]`  ->  Metal `threadgroup`.
    //
    // 16*16*4 = 1 KiB each, 2 KiB total. That matters: the Apple GPU budget is 32 KiB
    // per threadgroup and it is shared with the L1 cache. A 64x64 tile (16.6 KiB) is
    // legal and destroys occupancy -- measured in labs/lab02_coalescing.py at 5.9x
    // slower than the equivalent 32x32 version.
    threadgroup float As[16][16];
    threadgroup float Bs[16][16];

    // CUDA: threadIdx.x % TILE and threadIdx.x / TILE
    // Metal: thread_index_in_threadgroup is the same flattened per-threadgroup index.
    uint tx = thread_index_in_threadgroup % 16;
    uint ty = thread_index_in_threadgroup / 16;

    // CUDA: blockIdx.x  ->  Metal: threadgroup_position_in_grid.x
    uint blocks_per_row = B_shape[1] / 16;
    uint bid = threadgroup_position_in_grid.x;
    uint row = (bid / blocks_per_row) * 16 + ty;
    uint col = (bid % blocks_per_row) * 16 + tx;

    float acc = 0.0f;

    // CUDA: for (int k0 = 0; k0 < K; k0 += TILE)
    // Metal: K is A_shape[1].
    for (uint k0 = 0; k0 < A_shape[1]; k0 += 16) {
        // Coalesced: consecutive lanes read consecutive addresses, because tx is the
        // fast-varying part of thread_index_in_threadgroup.
        As[ty][tx] = A[row * A_shape[1] + k0 + tx];
        Bs[ty][tx] = B[(k0 + ty) * B_shape[1] + col];

        // CUDA: __syncthreads()  ->  Metal: threadgroup_barrier(...)
        // The memory-flags argument tells the compiler which address spaces to order.
        // Getting this wrong (or omitting the barrier) is a silent race.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 16; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // The second barrier: nobody may overwrite the tiles until everyone has read
        // them. Same reason as CUDA, same failure mode if omitted.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * B_shape[1] + col] = acc;
    // --- MLX body end ---
}

// Launch geometry, for reference:
//
//   CUDA :  dim3 blocks((M/16)*(N/16));  dim3 threads(256);
//           matmul_tiled<<<blocks, threads>>>(...)
//
//   MLX  :  launch = cuda((M//16) * (N//16), 256)
//           # -> grid=((M//16)*(N//16)*256, 1, 1), threadgroup=(256, 1, 1)
//
// Note that `grid` is the total thread count, not the block count.
