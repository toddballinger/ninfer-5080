#pragma once
// Issue #58 experimental CUDA pinned staging. No scheduler or allocator integration.
// A caller MUST synchronize its CUDA stream after an enqueued copy BEFORE destroying,
// resetting, moving-assigning or reusing the buffer. This class cannot discover
// in-flight stream users and intentionally does not claim automatic synchronization.
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::runtime::issue58 {
// Ownership of a cudaMallocHost allocation, **not** ownership of an in-flight copy.
// Construct empty; call allocate only at a quiescent boundary.
// Destruction requires caller-proven stream completion.
class KvPinnedStaging final {
public:
    KvPinnedStaging() noexcept = default;
    KvPinnedStaging(const KvPinnedStaging&) = delete;
    KvPinnedStaging& operator=(const KvPinnedStaging&) = delete;
    KvPinnedStaging(KvPinnedStaging&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}
    KvPinnedStaging& operator=(KvPinnedStaging&& other) noexcept {
        if (this != &other) {
            // Contract: caller synchronizes any stream using *this* first.
            if (data_ != nullptr) (void)cudaFreeHost(data_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    ~KvPinnedStaging() noexcept {
        // Contract: caller synchronizes every user stream before destruction.
        if (data_ != nullptr) (void)cudaFreeHost(data_);
    }

    // Never reallocates an existing allocation; fail closed if not empty.
    [[nodiscard]] cudaError_t allocate(std::size_t bytes) noexcept {
        if (data_ != nullptr || bytes == 0)
            return cudaErrorInvalidValue;
        void* out = nullptr;
        const auto error = cudaMallocHost(&out, bytes);
        if (error != cudaSuccess) return error;
        data_ = static_cast<std::uint8_t*>(out);
        size_ = bytes;
        return cudaSuccess;
    }
    // Caller guarantees no outstanding asynchronous operations.
    [[nodiscard]] cudaError_t reset() noexcept {
        if (data_ == nullptr) return cudaSuccess;
        const auto result = cudaFreeHost(data_);
        if (result == cudaSuccess) {
            data_ = nullptr;
            size_ = 0;
        }
        return result;
    }
    [[nodiscard]] std::uint8_t* data() noexcept { return data_; }
    [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return data_ == nullptr; }
private:
    std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};
} // namespace ninfer::runtime::issue58
