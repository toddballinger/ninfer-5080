#include "runtime/engine/issue58_kv_cuda_copy.h"
#include <cassert>
using namespace ninfer::runtime::issue58;
int main() {
    auto span=kv_page_copy_span(
        {KvPageOrder::HeadMajor,4,4096,128,512,8},2);
    assert(span.has_value());
    assert(kv_page_image_bytes(*span).value()==1024);
    // Invalid pointer checks short circuit before any CUDA operation.
    assert(kv_copy_page_to_host_async(nullptr,4096,nullptr,1024,*span,nullptr)
           ==cudaErrorInvalidValue);
    assert(kv_copy_page_from_host_async(nullptr,4096,nullptr,1024,*span,nullptr)
           ==cudaErrorInvalidValue);
}
