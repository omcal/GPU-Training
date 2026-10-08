#include "include/common.cuh"
__global__ void transpose_naive(const float* a, float* b, int h, int w) {
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 8 + threadIdx.y;
    if (x < w && y < h) b[x * h + y] = a[y * w + x];
}
// 32x8 threads move a 32x32 tile. The extra column avoids shared-memory
// bank conflicts when a warp reads a column of the tile.
__global__ void transpose_tiled(const float* a, float* b, int h, int w) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        if (x < w && y + j < h) tile[threadIdx.y + j][threadIdx.x] = a[(y + j) * w + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x; y = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        if (x < h && y + j < w) b[(y + j) * h + x] = tile[threadIdx.x][threadIdx.y + j];
}
int main(int argc, char**) {
    device_report();
    for (auto shape : {std::pair<int,int>{1,1}, {17,33}, {32,32}, {65,97}, {1024,1536}}) {
        int h = shape.first, w = shape.second;
        auto a = input(h * w); std::vector<float> ref(h * w);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) ref[x*h+y] = a[y*w+x];
        Buffer da(a.size()), db(a.size()); da.upload(a);
        dim3 threads(32,8), naive(ceil_div(w,32),ceil_div(h,8)), tiled(ceil_div(w,32),ceil_div(h,32));
        auto run_naive = [&] { transpose_naive<<<naive,threads>>>(da.p, db.p, h, w); };
        auto run_tiled = [&] { transpose_tiled<<<tiled,threads>>>(da.p, db.p, h, w); };
        run_naive(); CUDA_CHECK(cudaGetLastError()); check("transpose naive",db.download(),ref,0,0);
        CUDA_CHECK(cudaMemset(db.p,0xff,a.size()*sizeof(float)));
        run_tiled(); CUDA_CHECK(cudaGetLastError()); check("transpose tiled",db.download(),ref,0,0);
        if (argc == 1 && h == 1024) {
            benchmark("transpose naive",run_naive,8.0*h*w);
            benchmark("transpose tiled",run_tiled,8.0*h*w);
        }
    }
}
