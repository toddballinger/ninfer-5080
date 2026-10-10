#include "serve/tool_output_projection.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer::serve;
namespace {
int failures = 0;
void expect(bool condition, const std::string& label) {
    if (!condition) { std::cerr << "FAIL: " << label << '\n'; ++failures; }
}
ToolOutputPolicy policy(bool allow, bool none = false) {
    ToolOutputPolicy p;
    p.declared_tool_choice.mode = none ? ToolChoiceMode::None : ToolChoiceMode::Auto;
    if (allow) {
        ToolDefinition d;
        d.name = "get_weather";
        d.parameters_json =
            R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})";
        p.declared_tools.push_back(std::move(d));
    }
    return p;
}
std::string tool(const std::string& city = "Paris") {
    return "<tool_call>\n<function=get_weather>\n<parameter=city>" + city +
           "</parameter>\n</function>\n</tool_call>";
}
void run_case(const std::string& label, const std::string& raw,
              const ToolOutputPolicy& p, const std::string& expected_visible,
              std::size_t expected_calls) {
    const auto direct = project_tool_output(raw, p);
    expect(direct.visible_text == expected_visible, label + " direct visible");
    expect(direct.validated_calls.size() == expected_calls, label + " direct calls");
    for (std::size_t split = 0; split <= raw.size(); ++split) {
        ProjectedContentStream stream(p);
        const std::string early = stream.feed(std::string_view(raw).substr(0, split)) +
                                  stream.feed(std::string_view(raw).substr(split));
        stream.final_content(raw);
        const auto result = stream.finalise();
        expect(result.validated_calls.size() == direct.validated_calls.size(),
               label + " split call parity " + std::to_string(split));
        expect(result.visible_text == direct.visible_text,
               label + " split final projection " + std::to_string(split));
        expect(early.size() <= result.visible_text.size() &&
                   result.visible_text.compare(0, early.size(), early) == 0,
               label + " split safe prefix " + std::to_string(split));
        // Production callers must not send any part of the validated content twice.
        // Early content plus terminal residual must reconstruct the visible result.
        const std::string terminal =
            result.visible_text.substr(std::min(early.size(), result.visible_text.size()));
        expect(early + terminal == direct.visible_text,
               label + " split no duplicate output " + std::to_string(split));
    }
    ProjectedContentStream bytewise(p);
    std::string early;
    for (char c : raw) { early += bytewise.feed(std::string_view(&c, 1)); }
    bytewise.final_content(raw);
    const auto completed = bytewise.finalise();
    expect(completed.visible_text == direct.visible_text, label + " bytewise projection");
    expect(early.size() <= completed.visible_text.size() &&
               completed.visible_text.compare(0, early.size(), early) == 0,
           label + " bytewise safe prefix");
}
} // namespace

int main() {
    const auto allowed = policy(true);
    const auto disabled = policy(false);
    const auto choice_none = policy(true, true);
    run_case("normal prose", "A simple answer.", disabled, "A simple answer.", 0);
    run_case("accepted call", tool(), allowed, "", 1);
    run_case("allowed preface", "Checking weather.\n" + tool(), allowed,
             "Checking weather.", 1);
    run_case("undeclared no-tools", tool(), disabled, "", 0);
    run_case("tool-choice none", tool(), choice_none, "", 0);
    run_case("schema invalid", tool("42"), allowed, "", 0);
    run_case("xml duplicate", "<tool_call><function=get_weather>"
             "<parameter=city>42</parameter><parameter=city>Paris</parameter>"
             "</function></tool_call>", allowed, "", 0);
    run_case("valid followed by duplicate", tool() + "\n" +
             "<tool_call><function=get_weather><parameter=city>42</parameter>"
             "<parameter=city>Paris</parameter></function></tool_call>", allowed, "", 0);
    run_case("truncated tool", "Safe <tool_call><function=get_weather>",
             allowed, "", 0);
    run_case("truncated disabled", "Safe <tool_call><function=get_weather>",
             disabled, "", 0);
    run_case("partial marker", "Safe <tool_", disabled, "Safe <tool_", 0);

    ProjectedContentStream abandoned(allowed);
    abandoned.feed("<tool_call><function=get_weather>");
    abandoned.abandon();
    expect(abandoned.feed("</function></tool_call>").empty(),
           "abandoned stream cannot release bytes");
    expect(abandoned.streamed_content_bytes() == 0,
           "abandoned stream count reset");
    std::cout << "projection assertions complete; failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
