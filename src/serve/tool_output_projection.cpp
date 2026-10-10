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
    // Conservative atomic-commit policy: no content bytes escape before the
    // complete model output has been strictly parsed and schema validated.
    buffered_content_.append(delta);
    return {};
}

std::string ProjectedContentStream::feed_visible(std::string_view) {
    // Kept as a private ABI-neutral helper; no pre-commit publication.
    return {};
}

void ProjectedContentStream::final_content(std::string content) {
    if (finished_) { throw std::logic_error("final content after finalise is a logic error"); }
    final_content_ = std::move(content);
}

ProjectedToolOutput ProjectedContentStream::finalise() {
    if (finished_) { throw std::logic_error("ProjectedContentStream was already finalised"); }
    finished_ = true;
    ProjectedToolOutput projection = project_impl(final_content_, policy_);
    // The caller publishes visible_text exactly once at this commit point.
    // streamed_content_bytes() counts only bytes published by feed(), i.e. 0.
    buffered_content_.clear();
    final_content_.clear();
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