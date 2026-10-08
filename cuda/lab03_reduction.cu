#include "include/common.cuh"
// Each block produces one partial sum, then the host queues further levels.
// All 256 threads participate, including zero-filled tail lanes.
__global__ void reduce_tree(const float* a, float* out, int n) {
    __shared__ float s[256];
    int t = threadIdx.x, i = blockIdx.x * 512 + t;
    float v = i < n ? a[i] : 0;
    if (i + 256 < n) v += a[i+256];
    s[t] = v; __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) s[t] += s[t+stride];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}
float* reduce(const Buffer& a, Buffer& work1, Buffer& work2, int n) {
    const float* src = a.p;
    float* dst = work1.p;
    do {
        int blocks = ceil_div(n,512);
        reduce_tree<<<blocks,256>>>(src,dst,n);
        CUDA_CHECK(cudaGetLastError());
        n = blocks; src = dst; dst = dst == work1.p ? work2.p : work1.p;
    } while (n > 1);
    return const_cast<float*>(src);
}
int main(int argc, char**) {
    device_report();
    for (int n : {1, 31, 255, 256, 257, 511, 512, 513, 10007, 1 << 20}) {
        auto a = input(n); double sum = 0;
        for (float v : a) sum += v;
        Buffer da(n), w1(ceil_div(n,512)), w2(ceil_div(n,512)); da.upload(a);
        float* result = reduce(da,w1,w2,n); float got;
        CUDA_CHECK(cudaMemcpy(&got,result,sizeof(float),cudaMemcpyDeviceToHost));
        std::printf("input elements=%d\n", n);
        check("hierarchical reduction",{got},{float(sum)},1e-3f,1e-4f);
        if (argc == 1 && n == (1<<20))
            benchmark("hierarchical reduction",[&] { reduce(da,w1,w2,n); },4.0*n);
    }
}
