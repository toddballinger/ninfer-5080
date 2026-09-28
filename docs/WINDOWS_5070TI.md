# Windows + RTX 5070 Ti build guide

This fork's upstream and the `ninfer-5080` reference are validated on **64-bit Linux
only**. Nothing about the 5070 Ti itself needs a port: it is GB203 / compute capability
12.0, so `sm_120a` is already the correct architecture, and it reports the same
16303 MiB of VRAM as the RTX 5080, so the `groupwise-int-5080` quantization recipe and
its 131072-token KV allocation fit identically. What this document covers is the
**host platform** port.

Status: the full build, end-to-end inference, and `/v1/chat/completions` serving are
verified on this machine (RTX 5070 Ti, MSVC 19.44.35227, CUDA 13.4.59). See section 6.

This port builds on [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080)
and upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) — thanks to both for the
validated 128K / Q4-KV / MTP-3 / Vision runtime that this extends to a native Windows build.

## 1. Prerequisites

### CUDA Toolkit >= 13.1

`CMakeLists.txt` hard-fails below 13.1. Install a Windows toolkit from the
[CUDA archive](https://developer.nvidia.com/cuda-toolkit-archive).

- Preferred: **13.3.x** — matches the upstream validated toolkit (13.3.73).
- Used here: **13.4.1** (nvcc 13.4.59). Installs cleanly side by side with 13.0.

Since CUDA 13.1 the Windows GPU driver is no longer bundled with the toolkit, so a
silent install (`cuda_13.4.1_windows_x86_64.exe -s`) cannot disturb an existing driver.
Driver 591.86 was verified unchanged after the install.

> **Correction to an earlier note in this file.** It previously claimed that CUDA 13.0
> fails on the NVFP4 TMA translation units with MSVC `error C2719`, and that this proved
> the >= 13.1 minimum was load-bearing. **That was wrong.** Measured with 13.4.59 the
> same `C2719` still occurs, so it is a source-level issue, not a toolkit-version issue.
> The gate is a project-declared requirement; no Windows-side evidence was found that
> 13.0 would fail for a version reason. See "C2719" below for the real cause and fix.

### `error C2719` on the NVFP4 TMA kernels — over-aligned by-value parameter

Symptom (both `nvfp4_w4a4_tma.cu` and `nvfp4_linear_swiglu_w4a4_tma.cu`, on any toolkit
including 13.4):

```text
nvfp4_w4a4_tma.cuh(185): error C2719: 'descriptors': a parameter requiring 128 alignment
  will not be aligned
.../tmpxft_..._w4a4_tma.cudafe1.stub.c(44): error C2719: 'unnamed-parameter': ...
```

Cause: `Nvfp4W4a4TmaDescriptors` was passed **by value** into a `__global__` function as a
`__grid_constant__` parameter. The struct aggregates four `CUtensorMap`s, and
`CUtensorMap` carries a 128-byte alignment requirement, so the aggregate is 128-aligned
whatever the struct itself is annotated with. MSVC's ABI accepts 64-byte-aligned
by-value parameters but rejects 128-byte ones. GCC accepts it, which is why the upstream
Linux build never saw this.

Measured matrix (nvcc 13.4.59 + MSVC 19.44.35227, `-gencode=arch=compute_120a,code=sm_120a`,
inside the real translation units):

| Form of the parameter | Result |
|---|---|
| aggregate by value, `alignas(128)` | **error C2719** |
| aggregate by value, `alignas(64)` | nvcc: "alignment cannot be set to less than the default alignment" |
| aggregate by value, no annotation | **error C2719** |
| aggregate by reference, `__grid_constant__` | nvcc: "must not have reference type" |
| **a single `CUtensorMap` by value** | **error C2719** — `CUtensorMap` itself requires 128-byte alignment |
| pointer to the aggregate | compiles |

So none of the obvious fixes is available: the alignment cannot be lowered (it comes from
`CUtensorMap`), `__grid_constant__` forbids references, and splitting the aggregate into
per-descriptor parameters does **not** help because each individual `CUtensorMap` is
already over-aligned. Only the pointer form compiles.

**Resolution on this tree: the two translation units are excluded on MSVC and replaced by
a reporting stub.**

`src/CMakeLists.txt` builds `ninfer_nvfp4_tma` from
`ops/linear/nvfp4/nvfp4_tma_windows_stub.cpp` under MSVC, and from the two real CUDA
translation units everywhere else. The stub implements all five exported entry points —
`launch_nvfp4_w4a4_tma_linear`, `..._attention`, `..._gdn`, `..._linear_add` and
`launch_nvfp4_linear_swiglu_w4a4_tma` — and throws with an explanatory message. That
keeps NVFP4 artifacts failing loudly at the point of use rather than the build silently
dropping the feature, and it unblocks `ninfer_ops -> ninfer_engine -> ninfer-serve`.

Consequence: **NVFP4 weight artifacts are unsupported in Windows builds.** They are not
affected for `groupwise-int-5080`, whose recipe is Q3/Q4/Q5.

If NVFP4 support is needed on Windows later, the pointer-based route is the way:

1. Change the kernels to take `const CUtensorMap*` (or keep the aggregate and take
   `const Nvfp4W4a4TmaDescriptors*`).
2. Put the descriptors in a `__constant__` symbol filled with `cudaMemcpyToSymbolAsync`,
   or in a cached device buffer filled with `cudaMemcpyAsync`, before each launch.
   `cp.async.bulk.tensor` accepts a tensormap in global or const space, so this is
   semantically valid.
3. Note that a `__constant__` symbol is shared state, so concurrent launches carrying
   different descriptors would race. That is safe for the single-stream
   `--max-concurrency 1` configuration this project targets, but it is an assumption and
   should be enforced rather than hoped for.

That route cannot be validated without an NVFP4 artifact to run, which is why the stub is
the shipped behaviour for now.

Do **not** try to suppress the error with `/wd2719`. C2719 says the 128-byte alignment
requirement cannot be honoured; suppressing it would let a misaligned tensormap reach
`cp.async.bulk.tensor`, which faults or reads garbage.

> Beware of standalone micro-probes for this. A minimal file declaring
> `__global__ void k(__grid_constant__ const CUtensorMap m) {}` was observed to compile
> in isolation while the same construct fails inside the real translation units. Trust
> the real build, not the probe.

### `__builtin_memcpy` is not available under MSVC

`ops/kernel/gqa_attention_kv_quant.cuh` used `__builtin_memcpy` (a GCC/Clang builtin):

```text
gqa_attention_kv_quant.cuh(170): error: identifier "__builtin_memcpy" is undefined
```

Replaced with `memcpy` on a compile-time-constant size, which lowers to the same single
load/store pair; `#include <cstring>` added. This was the only use of a `__builtin_*`
intrinsic in the tree.

### A note on `-arch=sm_120a` vs `-gencode`

Hand-written nvcc invocations must use the `-gencode` spelling:

```text
-arch=sm_120a                       -> emits BOTH a non-'a' compute_120 image and a
                                       compute_120a image; the plain compute_120 PTX
                                       cannot contain setmaxnreg or block-scaled MMA,
                                       so ptxas rejects it with
                                       "Instruction 'setmaxnreg.dec' not supported on
                                        .target 'sm_120'"
-gencode=arch=compute_120a,code=sm_120a   -> only the sm_120a image; correct
```

`CMAKE_CUDA_ARCHITECTURES=120a` already expands to the `-gencode` form, so the CMake
build is unaffected — this only bites manual compile checks.

### MSVC portability fixes compiled into the tree

Everything below was found by actually running a full `cmake` configure and build on
Windows/MSVC; the earlier "compile a couple of changed files" check missed all of it.

| Symptom | Cause | Fix |
|---|---|---|
| `fatal error C1189: MSVC/cl.exe with traditional preprocessor is used` | CUDA 13.4's bundled CCCL requires the conforming preprocessor | `/Zc:preprocessor` added to the MSVC flag list in the root `CMakeLists.txt`, mirrored onto CUDA via `-Xcompiler` |
| `error C2589: '(' : illegal token on right side of '::'`, then `C2059`/`C2143` cascades across `sparse_moe.cpp`, `acquire.cpp`, `decision_resources.h`, `program_impl.h`, `request_plan_impl.h`, `text_context_impl.h`, `vision_context_impl.h`, `mtp_alignment.h` | `<windows.h>` defines `min`/`max` as macros, breaking every `std::min` / `::max` | `add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)` under MSVC |
| `error C2491: definition of dllimport function not allowed` (18x, `third_party/utf8proc/utf8proc.c`) | `utf8proc.h` defaults to `__declspec(dllimport)` on `_WIN32` | define `UTF8PROC_STATIC` |
| `error C3861: 'localtime_r': identifier not found` (`src/serve/console_log.cpp`) | `localtime_r` is POSIX | `_WIN32` branch uses `localtime_s(&tm, &t)` — note the swapped argument order |
| `error C2665: std::wstring::starts_with: no overload` (`acquire.cpp`) | `path::native()` is `std::wstring` on Windows, so the narrow `".."` literal has no matching overload | compare against `parent_marker.native()` |
| see the `__builtin_memcpy` section above | GCC/Clang builtin | use `memcpy` |
| see the `C2719` section above | 128-byte aligned `CUtensorMap` passed by value | NVFP4 TMA translation units replaced by a stub on MSVC |
| see the `PkgConfig::LIBCURL` section above | imported-target name mismatch | `TARGET LIBCURL` |

Two of these only surface at **link** time:

| Symptom | Cause | Fix |
|---|---|---|
| `LNK2019: unresolved external symbol __imp_WSAStartup / __imp_getaddrinfo / __imp_freeaddrinfo / __imp_ntohl / inet_ntop` | `acquire.cpp` uses the Winsock API directly, and `WIN32_LEAN_AND_MEAN` keeps the legacy `winsock.h` (and its implicit `wsock32.lib`) out of `windows.h`, so nothing pulled `ws2_32` onto the link line | `target_link_libraries(ninfer_media_acquire PRIVATE ws2_32)` under `WIN32` |
| `LNK2019: unresolved external symbol RequestBasePlan<Variant>::RequestBasePlan(RequestBasePlan&&)`, likewise `RequestPlan` and `SequencePlan` | `api_impl.h` defines these as out-of-line **`template <>` explicit specialisations with `= default`**, and MSVC does not emit a symbol for that shape. GCC does, so Linux never noticed; `engine.cpp` and `registry.cpp` only see the declarations | write the bodies out explicitly (`: impl_(std::move(other.impl_)) {}`), which is what the defaulted versions did |

The two executables also need the vcpkg runtime DLLs beside them —
`avformat-63.dll`, `avcodec-63.dll`, `avutil-61.dll`, `swscale-10.dll`, `libcurl.dll`.
Nothing on the default search path provides them, and without them both apps abort before
`main()` with `0xC0000135` (`STATUS_DLL_NOT_FOUND`) and print nothing at all.
`apps/CMakeLists.txt` now copies `<prefix>/bin` into the executable directory as a
`POST_BUILD` step, so a build tree is directly runnable.

### Visual Studio: use the VS 2022 toolset, not VS 2026

**Do not build with the installed Visual Studio 2026 Build Tools (18.x, MSVC
14.50.35717).** CUDA documentation lists MSVC 195x / VS 2026 18.x as supported, but
nvcc rejects this toolchain in practice with:

```text
nvcc fatal : Host compiler targets unsupported OS.
```

That failure is reproducible on this exact MSVC 14.50.35717 + SDK 10.0.26100.0
combination. Use the explicitly supported row instead:

```text
MSVC Version 193x | Visual Studio 2022 17.x | native x86_64 | C++14, C++17, C++20
```

`Microsoft Visual Studio\2022\Community` is already installed but is missing the C++
workload. Add it via the Visual Studio Installer:

```text
Workload:  使用 C++ 的桌面开发 (Desktop development with C++)
Components: MSVC v143 build tools, Windows 11 SDK
```

Non-interactive equivalent (elevated). Note the argument list must be passed as **one**
quoted string — an array makes `Start-Process` join with spaces without quoting, which
truncates the install path at `C:\Program` and fails with
"no installed product matches installPath":

```powershell
$argStr = 'modify --installPath "C:\Program Files\Microsoft Visual Studio\2022\Community" ' +
          '--add Microsoft.VisualStudio.Workload.NativeDesktop ' +
          '--includeRecommended --quiet --norestart'
Start-Process -FilePath 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe' `
              -ArgumentList $argStr
```

The elevated installer must be kept alive for the whole run: if it is launched from a
short-lived parent, it is torn down with that parent's job object roughly 45 s in, after
the download phase and before any compiler lands on disk. Launch it from a process that
then blocks on it.

If VS 2026 must be used as a fallback, the flag `--allow-unsupported-compiler` can be
appended to `CMAKE_CUDA_FLAGS`. It is untested here and is not the default.

### FFmpeg and libcurl (prebuilt, no pkg-config)

The Windows build path takes explicit prefixes instead of pkg-config:

| Variable | Content |
|---|---|
| `NINFER_FFMPEG_ROOT` | tree containing `include/` and `lib/` for avformat, avcodec, avutil, swscale |
| `NINFER_CURL_ROOT` | tree containing `include/` and `lib/libcurl.lib` |

Any distribution that ships `include/` + `lib/` works. A prebuilt FFmpeg 6.x or 7.x
build satisfies the `libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7`
requirement. `vcpkg install ffmpeg:x64-windows curl:x64-windows` also works; the
CMake helper resolves `libcurl.lib`, `libcurld.lib` or `libcurl_a.lib`.

In practice gyan.dev and BtbN builds are MinGW/UCRT64, so their development files are
`.dll.a`, which the MSVC linker cannot consume. **Use vcpkg** — it produces proper MSVC
import libraries:

```text
vcpkg install ffmpeg:x64-windows curl:x64-windows
# -> <vcpkg>/installed/x64-windows/{include,lib}; one prefix serves both roots
-DNINFER_FFMPEG_ROOT=<vcpkg>/installed/x64-windows
-DNINFER_CURL_ROOT=<vcpkg>/installed/x64-windows
```

#### Imported-target name mismatch on the Windows path

The first successful CMake configure on Windows failed with:

```text
CMake Error at src/CMakeLists.txt:324 (target_link_libraries):
  Target "ninfer_media_acquire" links to:
      PkgConfig::LIBCURL
  but the target was not found.
```

`ninfer_import_prebuilt_target()` synthesises `PkgConfig::${name}` from its first
argument, and the Windows branch passed `CURL` — producing `PkgConfig::CURL`, while
`src/CMakeLists.txt` links `PkgConfig::LIBCURL`, matching the pkg-config module name used
on Linux. Any of the `${NINFER_CURL_ROOT}` plumbing was fine; only the target name was
wrong.

Fixed by giving the helper an explicit `TARGET` argument:

```cmake
ninfer_import_prebuilt_target(CURL
  TARGET LIBCURL            # not CURL: downstream links PkgConfig::LIBCURL
  ROOT "${NINFER_CURL_ROOT}"
  HEADERS "curl/curl.h"
  LIBS "libcurl")
```

This is the kind of defect that only a real `cmake` configure can surface; the earlier
platform port was verified by compiling individual translation units, which never
evaluated `target_link_libraries`.

## 2. Build

Open a **Developer Command Prompt for VS 2022** (so `cl.exe` and Ninja are on PATH):

```bat
cmake -S . -B build-windows -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DNINFER_FFMPEG_ROOT=C:/deps/ffmpeg ^
  -DNINFER_CURL_ROOT=C:/deps/curl

cmake --build build-windows -j
```

Targets produced: `build-windows/apps/ninfer.exe` and `build-windows/apps/ninfer-serve.exe`.

No operator, target or scheduling source is modified by the platform port, so a build
that links is expected to produce numerically identical output to Linux.

## 3. Free the VRAM first

This configuration is a near-capacity fit. It is not "can the model load" but
"is there tens of MiB left", so the GPU must be clean:

```text
weights             ~13220 MiB
runtime reservation  ~2577 MiB
startup edge           44.56 MiB   (text-only, measured on the 5080)
```

Minimum free device memory before launch, derived from the reference envelope:

| Profile | Required free |
|---|---|
| text-only | ~15797 MiB |
| Vision 1792 | ~15814 MiB |
| Vision 2048 | ~15832 MiB |

Check before every launch:

```bat
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

On this machine the desktop plus browsers plus NVIDIA Broadcast, Wallpaper Engine,
ComfyUI and the SD WebUI launcher hold several GiB. All must be closed.

The cleanest fix on Windows is to take the desktop off the discrete GPU entirely:
the installed **Intel Core Ultra 7 265K has integrated Xe graphics that is currently
not enabled**. Connecting the display to the motherboard output and enabling the iGPU
in firmware removes the compositor's VRAM footprint from the 5070 Ti, which is what
makes the reference envelope reachable.

## 4. Launch

Text-only profile (milestone 1):

```bat
build-windows\apps\ninfer-serve.exe C:\models\qwen3_8_27b.ninfer ^
  --host 0.0.0.0 ^
  --port 8080 ^
  --model-id qwen3.8-27b ^
  --max-context 131072 ^
  --kv-capacity 131072 ^
  --prefill-chunk 896 ^
  --kv-dtype q4 ^
  --spec mtp ^
  --draft-tokens 3 ^
  --no-cuda-graph ^
  --max-concurrency 1
```

Add `--vision --vision-max-tokens 1792` only after the text path is green and the
free-memory table above has been met.

### Fallback ladder if startup fails on reservation

Apply in order and stop as soon as it starts:

1. `--vision-max-tokens 1024` — Vision workspace 132 MiB -> 66 MiB
2. drop `--vision` entirely — recovers the full 132 MiB
3. `--kv-capacity 114688` (112K) — about 289 MiB, at roughly 18.1 MiB per 1024 tokens

## 5. Expected performance

The 5070 Ti has 70 SMs against the 5080's 84, and 896 GB/s against 960 GB/s. Decode is
bandwidth-bound and prefill is compute-bound, so against the 5080 reference
(1378.85 tok/s prefill, 71.44 tok/s decode, 44.74% MTP acceptance):

| Metric | Expect | **Measured on this machine** |
|---|---|---|
| prefill | ~1100-1180 tok/s | **1300.95 tok/s** |
| decode | ~65-68 tok/s | **65.01 tok/s** |
| MTP acceptance | unchanged (model-determined) | 39.03% @ 8k ctx, 51.33% @ 64k ctx (reference 44.74%) |

Measured with 565 prompt tokens, `--embedding-host`, on `groupwise-int-5080`:

| Context | CUDA graph | MTP | prefill | decode | Notes |
|---|---|---|---|---|---|
| 8192 | off | off | 1277.04 | 37.27 | baseline |
| **8192** | **on** | **on** | **1300.95** | **65.01** | **the recommended configuration** |
| 65536 | off | on | 596.45 | 38.22 | `--kv-dtype q4`, fits in ~1.8 GiB free |
| 65536 | on | on | 561.13 | 41.73 | `--kv-dtype q4`, graph enabled after recalibration |
| 131072 | - | - | - | - | does not fit: needs 2.62 GiB runtime reservation |
| 98304 | - | - | - | - | does not fit: needs 2.21 GiB |

So the recommended launch is the documented one *minus* the context size:

```bat
build-windows\apps\ninfer.exe C:\models\qwen3_8_27b.ninfer ^
  --prompt-file prompt.txt ^
  --max-context 8192 ^
  --spec mtp --draft-tokens 3 ^
  --embedding-host
```

To reach `--max-context 131072` the machine needs roughly 800 MiB more free VRAM, which
is what moving the desktop to the iGPU (section 3) buys.

### `graph_allowance_bytes` was calibrated too tightly for this device

`src/targets/qwen3_6/impl/runtime/layouts_impl.h` sized the CUDA graph planning allowance
as `12 MiB x max_concurrency` without speculation, and `12` or `82 MiB x max_concurrency`
with MTP depending on whether the final visible window exceeds 4096. It did **not** scale
with context length, and the measured capture-transient cost exceeded it:

| Configuration | Measured transient | Old allowance | Over by |
|---|---|---|---|
| 2048 ctx, no speculation | 31.9 MiB | 12 MiB | 2.7x |
| 65536 ctx, MTP | 182.9 MiB | 82 MiB | 2.2x |

The check is a planning assertion, not a true capacity limit — the final captured graph is
only ~62.5 MiB at 65536 ctx, but the *capture transient* peaks far higher, which is what the
allowance must cover. The constants have been re-calibrated for this device:

| Branch | Old | New |
|---|---|---|
| no speculation | 12 MiB | 48 MiB |
| MTP (`final_visible <= 4096`) | 12 MiB | 48 MiB |
| MTP (`final_visible > 4096`) | 82 MiB | 256 MiB |
| DFlash (`final_visible > 4096`) | 82 MiB | 96 MiB |

Verified: at 65536 ctx + MTP + `--kv-dtype q4` the graph now captures (62.5 MiB observed /
256 MiB allowance) and decode rises from 38 -> 41.7 tok/s. The allowance is subtracted
during capacity planning, so the larger figure leaves slightly less room for
`--kv-capacity auto`; with the explicit `--max-context` this tree uses, that is not hit.

## 6. Status

Verified on this machine (RTX 5070 Ti, 16303 MiB, driver 591.86, MSVC 19.44.35227,
CUDA 13.4.59):

- `cmake` configure + full Ninja build succeeds; `apps/ninfer.exe` (228.5 MB) and
  `apps/ninfer-serve.exe` (229.2 MB) link clean.
- Both executables start and print their usage.
- `ninfer.exe <model> --prompt-file <file> --embedding-host` loads the artifact
  (`target qwen3_8_27b`, `weights groupwise-int-5080`), transfers 11.60 GiB of weights,
  runs every prewarm kernel, and generates. Peak prewarm left roughly 1760 MiB free.
- `ninfer-serve.exe` serves the OpenAI-compatible `/v1/chat/completions` endpoint
  end-to-end: a request returns HTTP 200 with `choices[].message.content` (the answer)
  plus `reasoning_content`, `usage`, and `id`. Tested with `--spec mtp --draft-tokens 3
  --embedding-host`, response in ~0.9 s for 64 tokens. The public model id defaults to the
  artifact identity and can be pinned with `--model-id`.

Two Windows-specific runtime notes:

- **Pass non-ASCII prompts with `--prompt-file`, not `--prompt`.** A Windows process
  receives `argv` in the ANSI code page rather than UTF-8, so an inline CJK prompt is
  rejected with `failed to normalize UTF-8 text as NFC: Invalid UTF-8 string`.
- `--embedding-host` keeps 795.70 MiB of token-embedding table in pinned host memory.
  That is what brings the machine inside the free-memory envelope in section 3.

Still open:

- **NVFP4 weight artifacts are not supported in Windows builds.** The two NVFP4 TMA
  translation units cannot be compiled by MSVC; see the `C2719` section. The stub makes
  that explicit at run time instead of failing the build. `groupwise-int-5080` is
  unaffected.
- **Long context past 65536 still needs more free VRAM.** 131072 requires ~800 MiB more
  (2.62 GiB runtime reservation vs ~1.83 GiB available); 98304 is ~300 MiB short. Moving
  the desktop to the iGPU (section 3) is the fix. The `graph_allowance_bytes` planning
  constants that previously also blocked 65536 ctx are now recalibrated (section 5), so
  graph mode works up to 65536 ctx — it is no longer a blocker.
- Kernel launch-threshold bands in `src/ops/linear/q4/q4_dispatch.cpp` and
  `src/ops/linear/q5/q5_dispatch.cpp` were swept on 84 SMs. They still run on 70 SMs;
  they are simply not optimal. Re-sweeping needs a working build, which now exists.
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp` already adapts to the real
  SM count via `cudaGetDeviceProperties`, and is the pattern the linear dispatch tables
  should follow.
- `ninfer-serve.exe` has not been exercised yet — only the one-shot `ninfer.exe`. Its
  long-context launch line in section 4 inherits the same two limits.
