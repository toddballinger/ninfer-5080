#include "serve/tool_call_validation.h"
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer::serve;
namespace {
int fails = 0;
void expect(bool ok, const char* label) {
    if (!ok) { std::cerr << "FAIL: " << label << '\n'; ++fails; }
}
ToolDefinition definition(std::string name, std::string schema) {
    ToolDefinition d;
    d.name = std::move(name);
    d.parameters_json = std::move(schema);
    return d;
}
ToolCall call(std::string name, std::string args) {
    ToolCall c;
    c.id = "call_0123456789abcdef";
    c.name = std::move(name);
    c.arguments_json = std::move(args);
    return c;
}
void case_check(const char* label, std::vector<ToolDefinition> declarations,
                ToolChoice choice, ToolCall candidate, bool accepted,
                const char* expected_error = nullptr, std::size_t limit = 64) {
    auto result = validate_candidate_calls({candidate}, declarations, choice, limit);
    expect((result.accepted.size() == 1) == accepted, label);
    if (accepted) {
        expect(result.accepted[0].id == candidate.id &&
               result.accepted[0].name == candidate.name &&
               result.accepted[0].arguments_json == candidate.arguments_json,
               "accepted output unchanged");
        expect(result.rejection_codes.empty(), "no rejection on accepted");
    } else {
        expect(result.accepted.empty(), "failed candidate not callable");
        expect(result.rejection_codes.size() == 1, "one rejection code");
        if (expected_error && !result.rejection_codes.empty()) {
            expect(result.rejection_codes[0] == expected_error, "expected rejection reason");
        }
    }
}
} // namespace

int main() {
    const std::string weather =
        R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})";
    const auto w = definition("get_weather", weather);
    const auto e = definition("echo", R"({"type":"object"})");
    const ToolChoice automatic{};
    case_check("declared accepted", {w}, automatic,
               call("get_weather", R"({"city":"Paris"})"), true);
    case_check("undeclared rejected", {e}, automatic,
               call("get_weather", R"({"city":"Paris"})"), false, "UNDECLARED_TOOL");
    case_check("empty declarations", {}, automatic,
               call("get_weather", "{}"), false, "UNDECLARED_TOOL");
    ToolChoice none; none.mode = ToolChoiceMode::None;
    case_check("none mode", {w}, none, call("get_weather", "{}"), false,
               "TOOL_CHOICE_DENIED");
    ToolChoice named; named.mode = ToolChoiceMode::Named; named.name = "echo";
    case_check("named mismatch", {w,e}, named,
               call("get_weather", R"({"city":"Paris"})"), false, "TOOL_CHOICE_DENIED");
    case_check("case mismatch", {w}, automatic,
               call("Get_Weather", R"({"city":"Paris"})"), false, "UNDECLARED_TOOL");
    case_check("duplicate declarations", {w,w}, automatic,
               call("get_weather", R"({"city":"Paris"})"), false, "AMBIGUOUS_DECLARATIONS");
    case_check("missing required", {w}, automatic,
               call("get_weather", "{}"), false, "ARGUMENT_SCHEMA_INVALID");
    case_check("wrong type", {w}, automatic,
               call("get_weather", R"({"city":42})"), false, "ARGUMENT_SCHEMA_INVALID");
    case_check("extra forbidden", {w}, automatic,
               call("get_weather", R"({"city":"Paris","admin":true})"), false,
               "ARGUMENT_SCHEMA_INVALID");
    case_check("malformed JSON", {w}, automatic,
               call("get_weather", R"({"city":)"), false, "ARGUMENT_JSON_INVALID");
    case_check("nonobject JSON", {w}, automatic,
               call("get_weather", "[]"), false, "ARGUMENT_JSON_INVALID");
    case_check("unsupported constraint",
               {definition("get_weather", R"({"type":"object","unevaluatedProperties":false})")},
               automatic, call("get_weather", "{}"), false, "SCHEMA_UNSUPPORTED");
    case_check("invalid schema", {definition("get_weather","{")},
               automatic, call("get_weather","{}"), false, "SCHEMA_UNSUPPORTED");
    case_check("empty schema", {definition("get_weather","")},
               automatic, call("get_weather","{}"), false, "SCHEMA_UNSUPPORTED");
    case_check("long client name", {definition(std::string(65,'a'),R"({"type":"object"})")},
               automatic, call(std::string(65,'a'),"{}"), false, "NAME_LENGTH_OR_SYNTAX");
    case_check("128 client name", {definition(std::string(128,'a'),R"({"type":"object"})")},
               automatic, call(std::string(128,'a'),"{}"), true, nullptr, 128);
    const std::string nested =
        R"({"type":"object","required":["payload"],"properties":{"payload":{"type":"object","required":["tags"],"properties":{"tags":{"type":"array","items":{"type":"string"}}},"additionalProperties":false}},"additionalProperties":false})";
    case_check("nested schema valid", {definition("nested",nested)}, automatic,
               call("nested",R"({"payload":{"tags":["a","b"]}})"), true);
    case_check("nested schema invalid", {definition("nested",nested)}, automatic,
               call("nested",R"({"payload":{"tags":[1]}})"), false,
               "ARGUMENT_SCHEMA_INVALID");
    case_check("unsupported nested keyword",
               {definition("nested", R"({"type":"object","properties":{"x":{"type":"string","pattern":"a"}}})")},
               automatic, call("nested",R"({"x":"a"})"), false, "SCHEMA_UNSUPPORTED");

    if (fails == 0) { std::cout << "ok: standalone declared-tool validation\n"; }
    return fails == 0 ? 0 : 1;
}
