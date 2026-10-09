#pragma once
// Issue58 read-only allocator-facing capture plan. No CUDA copies or releases.
#include "core/paged_kv_cache.h"
#include "runtime/engine/issue58_kv_host_backup.h"
#include <stdexcept>
#include <vector>
namespace ninfer::runtime::issue58 {
struct KvAllocationCapturePlan {
    std::uint32_t entitlement=0;
    std::int32_t source_row=-1;
    std::vector<std::int32_t> page_ids;
    KvHostBackup images;
};
// The PagedKVAllocation and pool must stay alive and immutable until the
// eventual transfer completes. This function does not lock the pool or GPU.
[[nodiscard]] inline KvAllocationCapturePlan
plan_kv_allocation_capture(const PagedKVPool& pool,
                           const PagedKVAllocation& allocation,
                           KvPageOrder order) {
    if (!allocation.valid() || !allocation.belongs_to(pool) ||
        allocation.mapped_page_count()==0 ||
        allocation.mapped_page_count()!=allocation.page_ids().size())
        throw std::invalid_argument("KV allocation is not fully materialized in source pool");
    KvAllocationCapturePlan plan;
    plan.entitlement=allocation.page_entitlement();
    plan.source_row=allocation.bound_row();
    plan.page_ids.assign(allocation.page_ids().begin(),allocation.page_ids().end());
    if(plan.entitlement<plan.page_ids.size())
        throw std::invalid_argument("KV mapped count exceeds entitlement");
    for(std::size_t i=0;i<pool.plane_count();++i){
        const Tensor& tensor=pool.plane(i);
        if(tensor.nb[2]<=0 || tensor.nb[3]<=0 || tensor.ne[3]<=0)
            throw std::invalid_argument("KV tensor has invalid physical strides");
        const KvPlaneGeometry geom{
            order, pool.page_group_count(), static_cast<std::size_t>(tensor.nbytes()),
            order==KvPageOrder::PageMajor
                ?static_cast<std::size_t>(tensor.nb[3])
                :static_cast<std::size_t>(tensor.nb[2]),
            static_cast<std::size_t>(tensor.nb[3]),
            static_cast<std::size_t>(tensor.ne[3])
        };
        plan.images.append_plane(geom,plan.page_ids);
    }
    return plan;
}
} // namespace ninfer::runtime::issue58
