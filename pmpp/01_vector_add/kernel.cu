// PMPP, chapter on heterogeneous data-parallel computing: the vector add.
//
// This is the first kernel in the book. Written in CUDA C++.
//
// Build (on a machine with CUDA):  nvcc -arch=sm_61 kernel.cu -o vector_add
// Run:                             ./vector_add

#include <cstdio>
#include <cstdlib>
#include "../../cuda/include/common.cuh"

// ---------------------------------------------------------------------------------
// The kernel.
//
// `__global__` marks it as a kernel: it runs on the GPU and is launched from the host.
// Each of the N threads that the launch creates executes this function exactly once,
// with its own threadIdx/blockIdx.
// ---------------------------------------------------------------------------------
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Global index of THIS thread.
    //
    // blockIdx.x  - which block this thread belongs to
    // blockDim.x  - how many threads per block (256 here)
    // threadIdx.x - this thread's index within its block
    //
    // Expressed in Metal terms: threadgroup_position_in_grid.x * threads_per_threadgroup.x
    //                            + thread_index_in_threadgroup
    // Metal also offers this sum pre-computed as `thread_position_in_grid`.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // The boundary guard. n is essentially never a multiple of blockDim.x, so the last
    // block has threads whose index is past the end of the array. Without this check
    // they write out of bounds.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;                       // ~1M elements
    const size_t bytes = n * sizeof(float);

    // Host memory.
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;
        h_b[i] = 1.0f;
    }

    // Device memory. On Apple Silicon there is no equivalent of this step: memory is
    // unified and the array is already accessible to the GPU.
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Host -> device.
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch: <<<blocks, threads>>>.
    //
    // `blocks` counts BLOCKS, each holding `threads` threads. The ceiling division is
    // the standard idiom for "one thread per element, rounded up".
    //
    // Metal is different here and it matters: `dispatchThreads(threadsPerGrid,
    // threadsPerThreadgroup)` takes the TOTAL thread count, so the equivalent call is
    // grid = blocks * threads. See ../../docs/01-cuda-to-metal.md.
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    // Wait here to surface asynchronous kernel errors before verification.
    // The blocking D2H copy below would also wait for this default-stream kernel.
    // (Metal/MLX equivalent: mx.synchronize() or MTLCommandBuffer.waitUntilCompleted.)
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Device -> host.
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify.
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            ++bad;
        }
    }
    printf("%d of %d elements wrong\n", bad, n);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    return bad == 0 ? 0 : 1;
}
