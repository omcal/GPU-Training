#include "include/common.cuh"
// Zero-padded 2D cross-correlation (filter is NOT reversed), as used by CNNs.
// An asymmetric filter in the reference makes x/y mistakes visible.
__global__ void conv_naive(const float* a, const float* f, float* b, int h, int w, int r) {
    int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    if(x>=w || y>=h) return;
    float acc=0; int d=2*r+1;
    for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx) {
        int yy=y+dy,xx=x+dx;
        if(yy>=0 && yy<h && xx>=0 && xx<w) acc+=a[yy*w+xx]*f[(dy+r)*d+dx+r];
    }
    b[y*w+x]=acc;
}
__global__ void conv_tiled(const float* a, const float* f, float* b, int h, int w, int r) {
    extern __shared__ float tile[];
    int side=16+2*r,t=threadIdx.y*16+threadIdx.x;
    // Cooperatively stage the output tile plus its halo, including zero padding.
    for(int i=t;i<side*side;i+=256) {
        int y=blockIdx.y*16+i/side-r,x=blockIdx.x*16+i%side-r;
        tile[i]=y>=0 && y<h && x>=0 && x<w ? a[y*w+x] : 0;
    }
    __syncthreads();
    int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    if(x<w && y<h) {
        float acc=0; int d=2*r+1;
        for(int dy=0;dy<d;++dy) for(int dx=0;dx<d;++dx)
            acc+=tile[(threadIdx.y+dy)*side+threadIdx.x+dx]*f[dy*d+dx];
        b[y*w+x]=acc;
    }
}
int main(int argc, char**) {
    device_report();
    for(int r : {1,2,4}) for(auto shape : {std::pair<int,int>{1,1},{1,19},{33,17},{256,320}}) {
        int h=shape.first,w=shape.second,d=2*r+1;
        auto a=input(h*w); std::vector<float> f(d*d),ref(h*w);
        for(int i=0;i<d*d;++i) f[i]=float(i+1)/(d*d*(d*d+1)/2);
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
            double acc=0;
            for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx) {
                int yy=y+dy,xx=x+dx;
                if(yy>=0 && yy<h && xx>=0 && xx<w) acc+=double(a[yy*w+xx])*f[(dy+r)*d+dx+r];
            }
            ref[y*w+x]=float(acc);
        }
        Buffer da(a.size()),df(f.size()),db(a.size()); da.upload(a);df.upload(f);
        dim3 threads(16,16),blocks(ceil_div(w,16),ceil_div(h,16));
        auto run_naive=[&]{conv_naive<<<blocks,threads>>>(da.p,df.p,db.p,h,w,r);};
        auto run_tiled=[&]{conv_tiled<<<blocks,threads,(16+2*r)*(16+2*r)*sizeof(float)>>>(da.p,df.p,db.p,h,w,r);};
        run_naive(); CUDA_CHECK(cudaGetLastError()); check("convolution naive",db.download(),ref);
        CUDA_CHECK(cudaMemset(db.p,0xff,a.size()*sizeof(float)));
        run_tiled(); CUDA_CHECK(cudaGetLastError()); check("convolution tiled",db.download(),ref);
        if(argc==1 && h==256) {
            std::printf("filter=%dx%d image=%dx%d\n",d,d,h,w);
            benchmark("convolution naive",run_naive);
            benchmark("convolution tiled",run_tiled);
        }
    }
}
