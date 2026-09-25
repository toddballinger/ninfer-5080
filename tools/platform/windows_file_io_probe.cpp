// Probe: verify Windows file-mapping + positioned-read semantics used by the
// NInfer artifact reader port (src/artifact/reader.cpp).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

int main() {
    const std::filesystem::path path = std::filesystem::current_path() / L"rwtest.bin";
    const std::size_t total = 65536;

    std::vector<std::uint8_t> src(total);
    for (std::size_t i = 0; i < total; ++i) {
        src[i] = static_cast<std::uint8_t>((i * 7 + 11) & 0xFF);
    }

    {
        HANDLE f = ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) {
            std::printf("create_failed err=%lu\n", ::GetLastError());
            return 1;
        }
        DWORD written = 0;
        ::WriteFile(f, src.data(), static_cast<DWORD>(total), &written, nullptr);
        ::CloseHandle(f);
    }

    // Open exactly the way the port does: SEQUENTIAL_SCAN (no NO_BUFFERING, which
    // would make CreateFileMapping fail).
    HANDLE f = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        std::printf("open_failed err=%lu\n", ::GetLastError());
        return 1;
    }

    HANDLE mapping = ::CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        std::printf("create_file_mapping_failed err=%lu\n", ::GetLastError());
        return 1;
    }

    const std::uint8_t* view =
        static_cast<const std::uint8_t*>(::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
    if (view == nullptr) {
        std::printf("map_view_of_file_failed err=%lu\n", ::GetLastError());
        return 1;
    }

    int mmap_ok = 1;
    for (std::size_t i = 0; i < total; ++i) {
        if (view[i] != src[i]) {
            std::printf("mmap_mismatch_at=%zu\n", i);
            mmap_ok = 0;
            break;
        }
    }
    std::printf("mmap_view_ok=%d\n", mmap_ok);

    // Positioned read on a SYNCHRONOUS handle via OVERLAPPED offset.
    std::vector<std::uint8_t> dst(4096, 0);
    const std::uint64_t off = 4096;
    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(off & 0xFFFFFFFFull);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);

    DWORD got = 0;
    const BOOL ok = ::ReadFile(f, dst.data(), 4096, &got, &ov);
    const int positioned_ok =
        (ok != 0) && got == 4096 && std::memcmp(dst.data(), src.data() + off, 4096) == 0;
    std::printf("positioned_read_ok=%d ok=%d got=%lu got_first=%u expected_first=%u\n",
                positioned_ok, static_cast<int>(ok), got, dst[0], src[off]);

    // Second positioned read back at offset 0, to catch a sticky file pointer.
    std::vector<std::uint8_t> dst0(4096, 0);
    OVERLAPPED ov0{};
    DWORD got0 = 0;
    const BOOL ok0 = ::ReadFile(f, dst0.data(), 4096, &got0, &ov0);
    const int rewind_ok = (ok0 != 0) && got0 == 4096 && std::memcmp(dst0.data(), src.data(), 4096) == 0;
    std::printf("positioned_read_offset0_ok=%d ok=%d got=%lu\n", rewind_ok, static_cast<int>(ok0),
                got0);

    std::printf("RESULT=%s\n",
                (mmap_ok && positioned_ok && rewind_ok) ? "ALL_OK" : "FAILED");

    ::UnmapViewOfFile(view);
    ::CloseHandle(mapping);
    ::CloseHandle(f);
    ::DeleteFileW(path.c_str());
    return (mmap_ok && positioned_ok && rewind_ok) ? 0 : 2;
}
