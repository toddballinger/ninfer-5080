// Compile-only API contract; does not allocate CUDA memory or execute GPU work.
// Build with nvcc or c++ plus CUDA include paths; do not run on production GPU.
#include "runtime/engine/issue58_kv_pinned_staging.h"
#include "runtime/engine/issue58_kv_cuda_copy.h"
#include <type_traits>
#include <utility>
using namespace ninfer::runtime::issue58;
static_assert(!std::is_copy_constructible_v<KvPinnedStaging>);
static_assert(!std::is_copy_assignable_v<KvPinnedStaging>);
static_assert(std::is_nothrow_move_constructible_v<KvPinnedStaging>);
static_assert(!std::is_move_assignable_v<KvPinnedStaging>);
static_assert(std::is_nothrow_destructible_v<KvPinnedStaging>);
static_assert(std::is_same_v<decltype(std::declval<KvPinnedStaging&>().allocate(1)),cudaError_t>);
static_assert(std::is_same_v<decltype(std::declval<KvPinnedStaging&>().reset()),cudaError_t>);
static_assert(std::is_same_v<decltype(kv_copy_page_to_host_async(
    static_cast<const void*>(nullptr),std::size_t{},
    std::declval<KvPinnedStaging&>().data(),std::declval<KvPinnedStaging&>().size(),
    std::declval<const KvPageCopySpan&>(),cudaStream_t{})),cudaError_t>);
int main(){return 0;}
