#pragma once
// Issue58: checked byte geometry for exporting one physical KV page.
// No device pointers, CUDA operations, allocation release, or scheduler effects.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace ninfer::runtime::issue58 {
enum class KvPageOrder : std::uint8_t { PageMajor, HeadMajor };
struct KvPlaneGeometry {
    KvPageOrder order;
    std::size_t page_count;
    std::size_t plane_bytes;
    std::size_t page_stride; // Tensor nb[3] for PageMajor; nb[2] for HeadMajor
    std::size_t row_pitch;   // Tensor nb[3] for HeadMajor
    std::size_t rows;        // Tensor ne[3] for HeadMajor
};
struct KvPageCopySpan {
    std::size_t offset;
    std::size_t bytes_per_row;
    std::size_t row_pitch;
    std::size_t rows;
};
[[nodiscard]] constexpr std::optional<KvPageCopySpan>
kv_page_copy_span(const KvPlaneGeometry& g, std::size_t physical_page) noexcept {
    constexpr auto limit=std::numeric_limits<std::size_t>::max();
    if (g.page_count == 0 || physical_page >= g.page_count ||
        g.page_stride == 0 || g.plane_bytes == 0) return std::nullopt;
    if (physical_page > limit / g.page_stride) return std::nullopt;
    const auto offset=physical_page * g.page_stride;
    if (offset >= g.plane_bytes) return std::nullopt;
    if (g.order == KvPageOrder::PageMajor) {
        if (g.page_stride > g.plane_bytes - offset) return std::nullopt;
        return KvPageCopySpan{offset,g.page_stride,g.page_stride,1};
    }
    if (g.rows == 0 || g.row_pitch < g.page_stride ||
        g.page_count > g.row_pitch / g.page_stride) return std::nullopt;
    if (g.rows - 1 > (limit - offset) / g.row_pitch) return std::nullopt;
    const auto last=offset + (g.rows - 1)*g.row_pitch;
    if (g.page_stride > g.plane_bytes - (last < g.plane_bytes ? last : g.plane_bytes))
        return std::nullopt;
    return KvPageCopySpan{offset,g.page_stride,g.row_pitch,g.rows};
}
} // namespace ninfer::runtime::issue58
