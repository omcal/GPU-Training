#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include <cstdint>

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: %s: %s\n", file, line, expr, cudaGetErrorString(status));
        std::exit(1);
    }
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
inline int ceil_div(int n, int d) { return (n + d - 1) / d; }

// Own device allocations; CUDA data movement remains explicit in the labs.
struct Buffer {
    float* p = nullptr;
    size_t n;
    explicit Buffer(size_t count) : n(count) {
        CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    }
    ~Buffer() { CUDA_CHECK(cudaFree(p)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<float>& x) {
        if (x.size() != n) std::exit(1);
        CUDA_CHECK(cudaMemcpy(p, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    std::vector<float> download() const {
        std::vector<float> x(n);
        CUDA_CHECK(cudaMemcpy(x.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
        return x;
    }
};
inline std::vector<float> input(size_t n, unsigned salt = 0) {
    std::vector<float> x(n);
    uint32_t state = salt + 1;
    for (size_t i = 0; i < n; ++i) {
        state = 1664525u * state + 1013904223u;
        x[i] = float(int((state >> 16) % 101) - 50) / 64.0f;
    }
    return x;
}
inline void check(const char* label, const std::vector<float>& got,
                  const std::vector<float>& ref, float atol = 1e-4f, float rtol = 1e-4f) {
    if (got.size() != ref.size()) std::exit(1);
    float worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        float err = std::fabs(got[i] - ref[i]);
        if (!std::isfinite(got[i]) || err > atol + rtol * std::fabs(ref[i])) {
            std::fprintf(stderr, "FAIL %s index=%zu got=%g ref=%g\n", label, i, got[i], ref[i]);
            std::exit(1);
        }
        worst = std::max(worst, err);
    }
    std::printf("PASS %-24s n=%zu max_error=%g\n", label, got.size(), worst);
}
inline void device_report() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s | sm_%d%d | %.0f MiB\n", prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / 1048576.0);
}
// Events time only queued GPU work. Host allocation and transfers stay outside.
// Call only after checking correctness. Every invocation must overwrite its output.
template<class F> float benchmark(const char* label, F launch, double bytes = 0,
                                  double flops = 0) {
    for (int i = 0; i < 5; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    for (int round = 0; round < 5; ++round) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms / 20);
    }
    std::sort(samples.begin(), samples.end());
    float ms = samples[2];
    std::printf("BENCH %-24s %.4f ms", label, ms);
    if (bytes) std::printf(" | %.2f effective GB/s", bytes / ms / 1e6);
    if (flops) std::printf(" | %.2f GFLOP/s", flops / ms / 1e6);
    std::puts("");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}
