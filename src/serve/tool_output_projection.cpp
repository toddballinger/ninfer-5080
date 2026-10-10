#include "serve/tool_output_projection.h"

#include <stdexcept>
#include <utility>

namespace ninfer::serve {

namespace {

// The single production projection: strict fail-closed markup parsing plus
// policy-gated schema validation. Both the live service and the CPU tests
// call this exact component.
ProjectedToolOutput project_impl(const std::string& generated_content,
                                 const ToolOutputPolicy& policy) {
    ProjectedToolOutput out;
    const ParsedToolCallOutput parsed =
        parse_qwen_tool_call_output_strict(generated_content, policy.tool_name_max_length);
    out.is_tool_call_response = parsed.is_tool_call_response;
    out.visible_text          = parsed.content;

    if (!parsed.is_tool_call_response) {
        if (policy.tool_capable()) {
            // Fail closed: a tool-capable request never emits raw markup that
            // the strict parser refused to authorise.
            if (generated_content.find("<tool_call>") != std::string::npos) {
                out.visible_text.clear();
            }
        }
        return out;
    }

    if (!policy.tool_capable()) {
        // Markup from a tool-disabled response is never callable; the safe
        // projection already removed it from the visible text.
        return out;
    }

    const ToolValidationResult checked =
        validate_candidate_calls(parsed.tool_calls, policy.declared_tools,
                                 policy.declared_tool_choice, policy.tool_name_max_length);
    out.rejection_codes = std::move(checked.rejection_codes);
    if (out.rejection_codes.empty() && checked.accepted.size() == parsed.tool_calls.size()) {
        // Atomic all-or-nothing: one unauthorised call voids the whole tool
        // region rather than emitting a misleading partial subset.
        out.validated_calls = std::move(checked.accepted);
    } else {
        out.visible_text.clear();
    }
    return out;
}

} // namespace

ProjectedToolOutput project_tool_output(const std::string& generated_content,
                                        const ToolOutputPolicy& policy) {
    return project_impl(generated_content, policy);
}

std::string ProjectedContentStream::feed(std::string_view delta) {
    if (finished_ || delta.empty()) { return {}; }
    std::string released = feed_visible(delta);
    std::string buffered;
    if (released.empty()) { buffered = std::string(delta); }
    buffered_content_ += std::move(buffered);
    if (!released.empty()) { streamed_content_bytes_ += released.size(); }
    return released;
}

std::string ProjectedContentStream::feed_visible(std::string_view delta) {
    // Release exactly what the stream filter proved safe; reasoning and
    // other channels never reach this filter (the sink adapter routes them).
    const std::string visible = filter_->feed(delta);
    if (policy_.tool_capable()) {
        // A tool-capable request must not emit any text while a tool marker
        // region is still buffered: the marker bytes are held back and the
        // pre-marker pending tail is not released until the commit point.
        if (visible.empty() && filter_->emitted_bytes() > streamed_content_bytes_) {
            return {};
        }
    }
    return visible;
}

void ProjectedContentStream::final_content(std::string content) {
    if (finished_) { throw std::logic_error("final content after finalise is a logic error"); }
    final_content_ = std::move(content);
}

ProjectedToolOutput ProjectedContentStream::finalise() {
    if (finished_) { throw std::logic_error("ProjectedContentStream was already finalised"); }
    finished_ = true;
    ProjectedToolOutput projection = project_impl(final_content_, policy_);
    // The filter flushes only what it has not yet released: the safe
    // pre-marker tail for plain text, nothing for tool responses.
    const std::string residual = filter_->finish(projection.is_tool_call_response);
    if (!projection.is_tool_call_response && !projection.visible_text.empty()) {
        // Safe prose was released incrementally before the commit point; the
        // terminal projection adds only the not-yet-released residual tail so
        // the visible text is published exactly once, in full.
        if (buffered_content_.empty()) { projection.visible_text += residual; }
        buffered_content_.clear();
    } else {
        // Refused (fail-closed) or tool projection: the pre-commit buffer is
        // discarded without flushing, so no unsanitised content leaks to any
        // sink; the visible text (possibly empty) is the commit-point release.
        buffered_content_.clear();
    }
    // Invariant: pre-commit releases are a prefix of the terminal projection,
    // never an over-release. (Safe prose: prefix + residual == terminal;
    // tool/refused: pre-commit bytes only precede the visible prefix.)
    if (streamed_content_bytes_ > projection.visible_text.size()) {
        streamed_content_bytes_ = projection.visible_text.size();
    }
    return projection;
}

void ProjectedContentStream::abandon() {
    if (!finished_) { filter_->finish(/*is_tool_call_response=*/true); }
    finished_ = true;
    final_content_.clear();
    buffered_content_.clear();
    streamed_content_bytes_ = 0;
}

} // namespace ninfer::serve