#pragma once
// Issue58: independent, host-owned physical KV image metadata and storage.
// Does not read/write GPU, reserve/release pool pages, or alter scheduling.
#include "runtime/engine/issue58_kv_page_geometry.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace ninfer::runtime::issue58 {
struct KvPlaneImage {
    KvPlaneGeometry geometry;
    std::vector<std::int32_t> physical_page_ids;
    std::vector<std::size_t> image_offsets;
    std::vector<std::uint8_t> data;
};
class KvHostBackup final {
public:
    KvHostBackup() = default;
    KvHostBackup(const KvHostBackup&)=delete;
    KvHostBackup& operator=(const KvHostBackup&)=delete;
    KvHostBackup(KvHostBackup&&) noexcept=default;
    KvHostBackup& operator=(KvHostBackup&&) noexcept=default;
    [[nodiscard]] const std::vector<KvPlaneImage>& planes() const noexcept {return planes_;}
    [[nodiscard]] std::size_t byte_count() const noexcept {return bytes_;}
    // No production allocator handles: only physical page IDs and independent bytes.
    // Caller guarantees page IDs correspond to the same logical-order mapping
    // when later reserving new physical pages for restoration.
    void append_plane(KvPlaneGeometry geometry,
                      std::vector<std::int32_t> physical_ids) {
        if(physical_ids.empty() || geometry.page_count==0)throw std::invalid_argument("empty KV image");
        KvPlaneImage image{geometry,std::move(physical_ids),{},{}};
        std::size_t plane_total=0;
        for(std::size_t i=0;i<image.physical_page_ids.size();++i){
            const auto id=image.physical_page_ids[i];
            if(id<0)throw std::invalid_argument("negative physical page");
            for(std::size_t j=0;j<i;++j)
                if(image.physical_page_ids[j]==id)throw std::invalid_argument("duplicate physical page");
            const auto span=kv_page_copy_span(geometry,static_cast<std::size_t>(id));
            if(!span || span->bytes_per_row==0 ||
               span->rows>std::numeric_limits<std::size_t>::max()/span->bytes_per_row)
                throw std::invalid_argument("invalid physical page geometry");
            const auto n=span->rows*span->bytes_per_row;
            if(n>std::numeric_limits<std::size_t>::max()-plane_total)
                throw std::overflow_error("KV plane byte overflow");
            image.image_offsets.push_back(plane_total);
            plane_total+=n;
        }
        if(plane_total>std::numeric_limits<std::size_t>::max()-bytes_)
            throw std::overflow_error("KV backup byte overflow");
        image.data.resize(plane_total);
        planes_.push_back(std::move(image));
        bytes_+=plane_total;
    }
private:
    std::vector<KvPlaneImage> planes_;
    std::size_t bytes_=0;
};
} // namespace ninfer::runtime::issue58
