#include "include/common.cuh"
__global__ void naive(const float* a, const float* b, float* c, int m, int n, int k) {
    int x = blockIdx.x*16+threadIdx.x, y = blockIdx.y*16+threadIdx.y;
    if (x < n && y < m) {
        float acc = 0;
        for (int q = 0; q < k; ++q) acc += a[y*k+q]*b[q*n+x];
        c[y*n+x] = acc;
    }
}
__global__ void tiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[16][16], bs[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x*16+tx, y = blockIdx.y*16+ty;
    float acc = 0;
    for (int q = 0; q < k; q += 16) {
        as[ty][tx] = y < m && q+tx < k ? a[y*k+q+tx] : 0;
        bs[ty][tx] = q+ty < k && x < n ? b[(q+ty)*n+x] : 0;
        __syncthreads();
        for (int j = 0; j < 16; ++j) acc += as[ty][j]*bs[j][tx];
        __syncthreads(); // No thread may overwrite a tile still being read.
    }
    if (x < n && y < m) c[y*n+x] = acc;
}
int main(int argc, char**) {
    device_report();
    struct Shape { int m,n,k; };
    for (auto s : {Shape{1,1,1}, {17,33,19}, {68,52,36}, {256,256,256}}) {
        int m=s.m,n=s.n,k=s.k;
        auto a=input(m*k), b=input(k*n,7); std::vector<float> ref(m*n);
        for(int y=0;y<m;++y) for(int x=0;x<n;++x) {
            double acc=0;
            for(int q=0;q<k;++q) acc+=double(a[y*k+q])*b[q*n+x];
            ref[y*n+x]=float(acc);
        }
        Buffer da(a.size()),db(b.size()),dc(ref.size()); da.upload(a);db.upload(b);
        dim3 threads(16,16),blocks(ceil_div(n,16),ceil_div(m,16));
        auto run_naive=[&]{naive<<<blocks,threads>>>(da.p,db.p,dc.p,m,n,k);};
        auto run_tiled=[&]{tiled<<<blocks,threads>>>(da.p,db.p,dc.p,m,n,k);};
        run_naive(); CUDA_CHECK(cudaGetLastError()); check("matmul naive",dc.download(),ref);
        CUDA_CHECK(cudaMemset(dc.p,0xff,ref.size()*sizeof(float)));
        run_tiled(); CUDA_CHECK(cudaGetLastError()); check("matmul tiled",dc.download(),ref);
        if(argc==1 && m==256) {
            benchmark("matmul naive",run_naive,0,2.0*m*n*k);
            benchmark("matmul tiled",run_tiled,0,2.0*m*n*k);
        }
    }
}
