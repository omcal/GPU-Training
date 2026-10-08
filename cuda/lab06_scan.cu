#include "include/common.cuh"
#include <memory>
// Work-efficient Blelloch exclusive scan, 512 elements / 256 threads.
// Every thread participates in every barrier, even for incomplete final blocks.
__global__ void block_scan(const float* a,float* out,float* sums,int n) {
    __shared__ float s[512];
    int t=threadIdx.x,base=blockIdx.x*512;
    s[t]=base+t<n ? a[base+t] : 0;
    s[t+256]=base+t+256<n ? a[base+t+256] : 0;
    __syncthreads();
    for(int stride=1;stride<512;stride*=2) {
        int i=(t+1)*2*stride-1;
        if(i<512) s[i]+=s[i-stride];
        __syncthreads();
    }
    if(t==0) { sums[blockIdx.x]=s[511];s[511]=0; }
    __syncthreads();
    for(int stride=256;stride;stride/=2) {
        int i=(t+1)*2*stride-1;
        if(i<512) { float left=s[i-stride];s[i-stride]=s[i];s[i]+=left; }
        __syncthreads();
    }
    if(base+t<n) out[base+t]=s[t];
    if(base+t+256<n) out[base+t+256]=s[t+256];
}
__global__ void add_offsets(float* out,const float* offsets,int n) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<n) out[i]+=offsets[i/512];
}
// Preallocate every level so allocation is excluded from event timing.
struct Scan {
    int n,blocks;
    Buffer sums,offsets;
    std::unique_ptr<Scan> next;
    explicit Scan(int size):n(size),blocks(ceil_div(size,512)),sums(blocks),offsets(blocks) {
        if(blocks>1) next=std::make_unique<Scan>(blocks);
    }
    void run(const float* a,float* out) {
        block_scan<<<blocks,256>>>(a,out,sums.p,n);
        CUDA_CHECK(cudaGetLastError());
        if(next) {
            next->run(sums.p,offsets.p);
            add_offsets<<<ceil_div(n,256),256>>>(out,offsets.p,n);
            CUDA_CHECK(cudaGetLastError());
        }
    }
};
int main(int argc, char**) {
    device_report();
    // >512^2 elements exercises three recursive levels; ragged sizes need padding.
    for(int n : {1,31,255,256,257,511,512,513,10007,262145,1<<20}) {
        auto a=input(n);std::vector<float> ref(n);double acc=0;
        for(int i=0;i<n;++i) { ref[i]=float(acc);acc+=a[i]; }
        Buffer da(n),db(n);da.upload(a);Scan scan(n);
        scan.run(da.p,db.p);check("exclusive scan",db.download(),ref,1e-3f,1e-4f);
        if(argc==1 && n==(1<<20)) benchmark("exclusive scan",[&]{scan.run(da.p,db.p);},8.0*n);
    }
}
