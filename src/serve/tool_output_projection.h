#pragma once

#include "serve/request.h"
#include "serve/tool_call_parser.h"
#include "serve/tool_call_validation.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {

// Immutable request-time policy snapshot for the response-projection path.
// Callers copy it into the projection component at request setup; the
// component never reads engine or model state.
struct ToolOutputPolicy {
    std::vector<ToolDefinition> declared_tools;
    ToolChoice declared_tool_choice;
    std::size_t tool_name_max_length = 64;

    [[nodiscard]] bool tool_capable() const {
        return !declared_tools.empty() && declared_tool_choice.mode != ToolChoiceMode::None;
    }
};

// The single production projection of generated assistant content: strict
// markup parsing plus caller-schema validation, fail-closed. Both the live
// service and the CPU tests must call this exact component.
struct ProjectedToolOutput {
    bool is_tool_call_response = false;
    std::string visible_text;
    std::vector<ToolCall> validated_calls;
    // Safe diagnostics: why each parsed call was rejected, if any.
    std::vector<std::string> rejection_codes;
};

ProjectedToolOutput project_tool_output(const std::string& generated_content,
                                        const ToolOutputPolicy& policy);

// Streaming publication of generated content deltas. Only text that is
// provably outside a possible <tool_call> suffix is
// released while streaming; the validated projection is released once, at
// finalisation. Cancellation or failure before finalisation discards the
// buffer: no unsanitised content is ever flushed to a sink.
class ProjectedContentStream final {
public:
    explicit ProjectedContentStream(ToolOutputPolicy policy)
        : policy_(std::move(policy)), filter_(std::make_unique<ToolCallStreamFilter>()) {}

    ProjectedContentStream(const ProjectedContentStream&)            = delete;
    ProjectedContentStream& operator=(const ProjectedContentStream&) = delete;

    // Consume one generated content delta. Returns the safe visible delta to
    // publish, or the empty string when the content is buffered. After
    // finalise()/abandon() deltas are ignored and nothing is released.
    std::string feed(std::string_view delta);

    // Authoritative full generated content (the engine's concatenated
    // result). Call after the final feed(); finalise() projects it.
    void final_content(std::string content);

    // Finalise: release the validated projection exactly once (tool region
    // discarded for tool responses; safe prose released for text responses).
    ProjectedToolOutput finalise();

    // Discard everything buffered (cancellation, error, disconnect): no
    // unsanitised content is flushed to any sink. Safe after finalise().
    void abandon();

    [[nodiscard]] const ToolOutputPolicy& policy() const { return policy_; }

    // Total content bytes released through feed() before finalisation.
    [[nodiscard]] std::size_t streamed_content_bytes() const noexcept {
        return streamed_content_bytes_;
    }

private:
    std::string feed_visible(std::string_view delta);

    ToolOutputPolicy policy_;
    std::unique_ptr<ToolCallStreamFilter> filter_;
    bool finished_ = false;
    std::string final_content_;
    std::size_t streamed_content_bytes_ = 0;
    std::string buffered_content_;
};

// Engine-side sink adapter for ProjectedContentStream. Reasoning deltas are
// forwarded to on_reasoning; content deltas pass through the projection
// filter. The sink holds no other state.
class ProjectedContentSink final : public ninfer::OutputSink {
public:
    ProjectedContentStream* stream = nullptr;
    std::function<void(const std::string&)> on_content;
    std::function<void(const std::string&)> on_reasoning;

    // Routes reasoning deltas to on_reasoning and content deltas through the
    // projection filter; the projection releases the validated output at
    // finalisation, never raw deltas.
    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (on_reasoning) { on_reasoning(delta.text); }
            return;
        }
        if (stream == nullptr) { return; }
        const std::string visible = stream->feed(delta.text);
        if (!visible.empty() && on_content) { on_content(visible); }
    }
};

} // namespace ninfer::serve