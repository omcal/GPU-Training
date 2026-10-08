#include "include/common.cuh"
// A bounded launch: one output per thread.
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// A fixed number of blocks can cover any input via a grid-stride loop.
__global__ void add_stride(const float* a, const float* b, float* c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) c[i] = a[i] + b[i];
}
int main(int argc, char**) {
    device_report();
    for (int n : {1, 31, 255, 256, 257, 1003, 1 << 20}) {
        auto a = input(n), b = input(n, 3);
        std::vector<float> ref(n);
        for (int i = 0; i < n; ++i) ref[i] = a[i] + b[i];
        Buffer da(n), db(n), dc(n);
        da.upload(a); db.upload(b);
        add<<<ceil_div(n, 256), 256>>>(da.p, db.p, dc.p, n);
        CUDA_CHECK(cudaGetLastError());
        check("vector add", dc.download(), ref, 0, 0);
        // Poison the output so the second kernel cannot pass using old results.
        CUDA_CHECK(cudaMemset(dc.p, 0xff, n * sizeof(float)));
        add_stride<<<1, 256>>>(da.p, db.p, dc.p, n);
        CUDA_CHECK(cudaGetLastError());
        check("grid stride / one block", dc.download(), ref, 0, 0);
        if (argc == 1 && n == (1 << 20)) {
            benchmark("vector add", [&] { add<<<ceil_div(n, 256), 256>>>(da.p, db.p, dc.p, n); }, 12.0 * n);
            benchmark("grid stride / 128 blocks", [&] { add_stride<<<128, 256>>>(da.p, db.p, dc.p, n); }, 12.0 * n);
            check("grid stride / 128 blocks", dc.download(), ref, 0, 0);
        }
    }
}
