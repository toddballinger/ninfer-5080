add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)

# --- Runtime dependencies beside the executables ---------------------------------------
# Both apps link the FFmpeg and libcurl import libraries, so the matching DLLs must sit
# next to the executable or be on PATH when it starts. Without this the process dies
# before main() with 0xC0000135 (STATUS_DLL_NOT_FOUND), reporting no output at all:
#   avformat-63.dll, avcodec-63.dll, avutil-61.dll, swscale-10.dll, libcurl.dll
# The imported-target helper points the linker at <prefix>/lib, so the DLLs are in
# <prefix>/bin. NINFER_FFMPEG_ROOT and NINFER_CURL_ROOT normally name the same vcpkg
# prefix, so deduplicate before copying.
if(WIN32)
  set(_ninfer_runtime_prefixes "")
  foreach(_r "${NINFER_FFMPEG_ROOT}" "${NINFER_CURL_ROOT}")
    if(_r AND NOT _r IN_LIST _ninfer_runtime_prefixes)
      list(APPEND _ninfer_runtime_prefixes "${_r}")
    endif()
  endforeach()

  foreach(_app ninfer ninfer-serve)
    foreach(_prefix IN LISTS _ninfer_runtime_prefixes)
      if(EXISTS "${_prefix}/bin")
        add_custom_command(TARGET ${_app} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_directory
                  "${_prefix}/bin" "$<TARGET_FILE_DIR:${_app}>"
          COMMENT "Copying runtime DLLs from ${_prefix}/bin next to ${_app}")
      endif()
    endforeach()
  endforeach()
endif()
