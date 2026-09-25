cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# --- Host toolchain hardening ------------------------------------------------
# MSVC needs switches that GCC/Clang enable by default. Registering them here keeps
# individual targets clean, and they are mirrored onto the CUDA line because nvcc
# forwards host flags through -Xcompiler rather than inheriting them.
#
# /bigobj is not optional: nvcc emits host translation units that exceed the default
# section limit on the larger operator files.
#
# /Zc:preprocessor is not optional either: CUDA 13.4's bundled CCCL hard-errors out of
# cuda/std/__cccl/preprocessor.h when cl.exe uses its traditional preprocessor.
if(MSVC)
  foreach(_flag /utf-8 /Zc:__cplusplus /Zc:preprocessor /bigobj)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${_flag}>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>)
  endforeach()

  # <windows.h> defines min/max as macros, which breaks every std::min / ::max in the
  # tree on the Windows path (symptoms: C2589 "'(' : illegal token on right side of
  # '::'", then a cascade of C2059/C2143 parse errors). WIN32_LEAN_AND_MEAN additionally
  # keeps the legacy winsock.h out of windows.h, which otherwise collides with the
  # winsock2.h that src/product/media_acquire/acquire.cpp includes.
  #
  # UTF8PROC_STATIC: third_party/utf8proc/utf8proc.h defaults to __declspec(dllimport)
  # on _WIN32, which makes it illegal to define those functions in the .c we compile
  # into ninfer_text (MSVC C2491).
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN UTF8PROC_STATIC)
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# --- Media dependencies ------------------------------------------------------
# The Linux path locates FFmpeg and libcurl through pkg-config. Windows has no
# pkg-config, so a prebuilt prefix is supplied explicitly and the *same* imported
# target names are synthesised. Every downstream target_include_directories /
# target_link_libraries line therefore stays byte-identical across platforms.
#
# Any distribution that ships include/ and lib/ works: vcpkg, gyan.dev or a BtbN
# FFmpeg build, and the official curl-for-Windows build.
# The Linux path names its imported targets after the pkg_check_modules prefix, and the
# downstream sources link PkgConfig::LIBCURL (not ...::CURL). Windows must therefore be
# able to synthesise a target whose name differs from the ROOT variable it reads.
function(ninfer_import_prebuilt_target name)
  cmake_parse_arguments(ARG "" "ROOT;TARGET" "HEADERS;LIBS" ${ARGN})

  if(NOT ARG_TARGET)
    set(ARG_TARGET "${name}")
  endif()

  if(NOT ARG_ROOT)
    message(FATAL_ERROR
      "Windows build requires -DNINFER_${name}_ROOT=<prefix> pointing at a prebuilt "
      "${name} tree that contains include/ and lib/.")
  endif()

  foreach(_header IN LISTS ARG_HEADERS)
    if(NOT EXISTS "${ARG_ROOT}/include/${_header}")
      message(FATAL_ERROR
        "NINFER_${name}_ROOT='${ARG_ROOT}' does not contain include/${_header}.")
    endif()
  endforeach()

  set(_libraries "")
  foreach(_lib IN LISTS ARG_LIBS)
    set(_suffixes ".lib" "d.lib" "_a.lib")
    set(_resolved "")
    foreach(_suffix IN LISTS _suffixes)
      if(EXISTS "${ARG_ROOT}/lib/${_lib}${_suffix}")
        set(_resolved "${ARG_ROOT}/lib/${_lib}${_suffix}")
        break()
      endif()
    endforeach()
    if(NOT _resolved)
      message(FATAL_ERROR
        "NINFER_${name}_ROOT='${ARG_ROOT}' has no lib/${_lib}{.lib} under lib/.")
    endif()
    list(APPEND _libraries "${_resolved}")
  endforeach()

  add_library(PkgConfig::${ARG_TARGET} INTERFACE IMPORTED)
  set_target_properties(PkgConfig::${ARG_TARGET} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ARG_ROOT}/include"
    INTERFACE_LINK_LIBRARIES "${_libraries}")
  message(STATUS "Using prebuilt ${name} at ${ARG_ROOT} as PkgConfig::${ARG_TARGET}")
endfunction()

if(WIN32)
  ninfer_import_prebuilt_target(FFMPEG
    ROOT "${NINFER_FFMPEG_ROOT}"
    HEADERS "libavformat/avformat.h;libavcodec/avcodec.h;libavutil/avutil.h;libswscale/swscale.h"
    LIBS "avformat;avcodec;avutil;swscale")

  if(NINFER_BUILD_MEDIA_ACQUIRE)
    # TARGET LIBCURL, not CURL: src/CMakeLists.txt links PkgConfig::LIBCURL, matching the
    # pkg-config module name used on Linux.
    ninfer_import_prebuilt_target(CURL
      TARGET LIBCURL
      ROOT "${NINFER_CURL_ROOT}"
      HEADERS "curl/curl.h"
      LIBS "libcurl")
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
