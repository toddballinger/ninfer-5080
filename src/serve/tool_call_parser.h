#pragma once

#include "serve/request.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length);

// Separate fail-closed candidate parser for future validated emission.
// Unlike the legacy characterization parser, duplicate XML parameter names
// never become a callable tool, and malformed tool markup is not returned
// as user-visible raw assistant text. Not wired to generation yet.
ParsedToolCallOutput parse_qwen_tool_call_output_strict(const std::string& text,
                                                        std::size_t max_tool_name_length);

// Incrementally publishes text that is provably outside a possible Qwen
// <tool_call> suffix. At terminal time, a valid tool response discards the
// buffered tool region; malformed/non-tool output flushes it verbatim.
class ToolCallStreamFilter {
public:
    std::string feed(std::string_view text);
    std::string finish(bool is_tool_call_response);

    [[nodiscard]] std::size_t emitted_bytes() const noexcept { return emitted_bytes_; }

private:
    std::string pending_;
    std::string tool_region_;
    std::size_t emitted_bytes_ = 0;
    bool saw_tool_marker_      = false;
    bool finished_             = false;
};

} // namespace ninfer::serve
