#include "runtime/engine/issue58_kv_cuda_copy.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using namespace ninfer::runtime::issue58;
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA_FAIL %s: %s\n",#x,cudaGetErrorString(e));return 2;}}while(0)
int main() {
    int devices=0;
    CHECK(cudaGetDeviceCount(&devices));
    if(devices<1)return 3;
    constexpr std::size_t bytes=4096;
    std::vector<std::uint8_t> original(bytes), output(bytes);
    for(std::size_t i=0;i<bytes;++i)original[i]=static_cast<std::uint8_t>((i*37+11)%251);
    void* device=nullptr;
    cudaStream_t stream=nullptr;
    CHECK(cudaMalloc(&device,bytes));
    CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    int failures=0;
    for (auto g : {
        KvPlaneGeometry{KvPageOrder::PageMajor,4,bytes,1024,0,0},
        KvPlaneGeometry{KvPageOrder::HeadMajor,4,bytes,128,512,8}
    }) {
        CHECK(cudaMemcpyAsync(device,original.data(),bytes,cudaMemcpyHostToDevice,stream));
        CHECK(cudaStreamSynchronize(stream));
        auto span=kv_page_copy_span(g,2);
        if(!span) return 4;
        auto count=kv_page_image_bytes(*span);
        if(!count)return 5;
        // cudaMallocHost gives independent pinned host storage for async D2H.
        void* host=nullptr;
        CHECK(cudaMallocHost(&host,*count));
        CHECK(kv_copy_page_to_host_async(device,bytes,host,*count,*span,stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemsetAsync(device,0,bytes,stream));
        CHECK(kv_copy_page_from_host_async(device,bytes,host,*count,*span,stream));
        CHECK(cudaMemcpyAsync(output.data(),device,bytes,cudaMemcpyDeviceToHost,stream));
        CHECK(cudaStreamSynchronize(stream));
        for(std::size_t i=0;i<bytes;++i) {
            bool within=false;
            for(std::size_t row=0;row<span->rows;++row){
                const auto start=span->offset+row*span->row_pitch;
                if(i>=start && i<start+span->bytes_per_row)within=true;
            }
            const auto expected=within?original[i]:0;
            if(output[i]!=expected)++failures;
        }
        CHECK(cudaFreeHost(host));
        std::printf("LAYOUT_%s=%s BYTES=%zu\n",
             g.order==KvPageOrder::PageMajor?"PAGE_MAJOR":"HEAD_MAJOR",
             failures?"FAIL":"PASS",*count);
    }
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(device));
    std::printf("GPU_KV_ROUNDTRIP=%s FAILURES=%d\n",failures?"FAIL":"PASS",failures);
    return failures?6:0;
}
