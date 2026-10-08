// PMPP, memory architecture and data locality: the tiled matrix multiply.
//
// This is the kernel that teaches tiling. Compare it against the naive version in the
// book (one thread per output element, reading A and B from global memory on every
// iteration of the K loop) to see what the threadgroup tile buys.
//
// Build (on a machine with CUDA):  nvcc -arch=sm_61 kernel.cu -o tiled_matmul
// Run:                             ./tiled_matmul

#include <cstdio>
#include <cstdlib>
#include "../../cuda/include/common.cuh"

#define TILE 16   // 16x16 tile = 256 threads, 1 KiB of shared memory per array

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // On-chip storage for one tile of each operand.
    //
    // CUDA:      __shared__
    // Metal:     threadgroup
    // Apple cap: 32 KiB per threadgroup, and it is shared with the L1 cache. Two 16x16
    //            float tiles cost 2 KiB, which is why 16x16 is comfortable and 64x64
    //            (16.6 KiB) wrecks occupancy. See ../../labs/lab02_coalescing.py.
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // This thread's position INSIDE the block.
    // Metal equivalent: thread_index_in_threadgroup (and thread_position_in_threadgroup
    // for the per-axis uint3 form).
    int tx = threadIdx.x % TILE;
    int ty = threadIdx.x / TILE;

    // Which output tile this block owns.
    // Metal equivalent: threadgroup_position_in_grid.
    int blocks_per_row = N / TILE;
    int row = (blockIdx.x / blocks_per_row) * TILE + ty;
    int col = (blockIdx.x % blocks_per_row) * TILE + tx;

    float acc = 0.0f;

    // Walk the K dimension one tile at a time.
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // Coalesced global loads: consecutive threads read consecutive addresses,
        // because tx is the fast-varying thread index.
        As[ty][tx] = A[row * K + k0 + tx];
        Bs[ty][tx] = B[(k0 + ty) * N + col];

        // Every thread must see the whole tile before anyone uses it.
        // Metal equivalent: threadgroup_barrier(mem_flags::mem_threadgroup)
        __syncthreads();

        // The inner product, entirely from on-chip memory.
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // Before overwriting the tiles on the next iteration, make sure everyone is
        // done reading them. Omitting this second barrier is a classic race: it usually
        // passes on small inputs and corrupts large ones.
        __syncthreads();
    }

    C[row * N + col] = acc;
}

int main() {
    const int M = 256, N = 256, K = 256;
    const size_t bytes = M * N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // One block per 16x16 output tile, 256 threads each.
    dim3 blocks((M / TILE) * (N / TILE));
    dim3 threads(TILE * TILE);
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // With A and B all ones, every element of C must equal K.
    int bad = 0;
    for (int i = 0; i < M * N; ++i) {
        if (h_C[i] != (float)K) ++bad;
    }
    printf("%d of %d elements wrong\n", bad, M * N);

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    return bad == 0 ? 0 : 1;
}
