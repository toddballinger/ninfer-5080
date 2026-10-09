#pragma once
// Issue58 experimental CUDA page-image transport. No allocator mutations.
// Caller must own the device/host buffers, maintain page ownership, and
// synchronize the stream BEFORE examining/releasing or reusing either buffer.
// For genuine asynchronous transfers host_image MUST refer to pinned/page-locked
// host storage; ordinary std::vector storage is NOT a validated async target.
// A future owning snapshot path must stage/copy into independent durable storage
// after CUDA completion, and keep the pinned staging allocation alive until then.
#include "runtime/engine/issue58_kv_page_geometry.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace ninfer::runtime::issue58 {
[[nodiscard]] constexpr std::optional<std::size_t>
kv_page_image_bytes(const KvPageCopySpan& s) noexcept {
    if (!s.rows || !s.bytes_per_row ||
        s.rows > std::numeric_limits<std::size_t>::max()/s.bytes_per_row)
        return std::nullopt;
    return s.rows*s.bytes_per_row;
}
// The device plane and the host image must have been allocated independently.
// Both copies are asynchronous; these functions never synchronize or free.
[[nodiscard]] inline cudaError_t kv_copy_page_to_host_async(
    const void* device_plane, std::size_t device_plane_bytes,
    void* host_image, std::size_t host_image_bytes,
    const KvPageCopySpan& span, cudaStream_t stream) noexcept {
    const auto bytes=kv_page_image_bytes(span);
    if (!device_plane || !host_image || !bytes || *bytes>host_image_bytes ||
        span.offset>=device_plane_bytes || span.row_pitch<span.bytes_per_row)
        return cudaErrorInvalidValue;
    if (span.rows-1 > (device_plane_bytes-1-span.offset)/span.row_pitch ||
        span.bytes_per_row>device_plane_bytes-span.offset-(span.rows-1)*span.row_pitch)
        return cudaErrorInvalidValue;
    return cudaMemcpy2DAsync(host_image,span.bytes_per_row,
        static_cast<const std::uint8_t*>(device_plane)+span.offset,
        span.row_pitch,span.bytes_per_row,span.rows,
        cudaMemcpyDeviceToHost,stream);
}
[[nodiscard]] inline cudaError_t kv_copy_page_from_host_async(
    void* device_plane, std::size_t device_plane_bytes,
    const void* host_image, std::size_t host_image_bytes,
    const KvPageCopySpan& span, cudaStream_t stream) noexcept {
    const auto bytes=kv_page_image_bytes(span);
    if (!device_plane || !host_image || !bytes || *bytes>host_image_bytes ||
        span.offset>=device_plane_bytes || span.row_pitch<span.bytes_per_row)
        return cudaErrorInvalidValue;
    if (span.rows-1 > (device_plane_bytes-1-span.offset)/span.row_pitch ||
        span.bytes_per_row>device_plane_bytes-span.offset-(span.rows-1)*span.row_pitch)
        return cudaErrorInvalidValue;
    return cudaMemcpy2DAsync(
        static_cast<std::uint8_t*>(device_plane)+span.offset,span.row_pitch,
        host_image,span.bytes_per_row,span.bytes_per_row,span.rows,
        cudaMemcpyHostToDevice,stream);
}
} // namespace ninfer::runtime::issue58
