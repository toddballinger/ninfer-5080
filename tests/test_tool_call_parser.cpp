#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

int test_single_call() {
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output("Calling weather.\n"
                                                   "<tool_call>\n"
                                                   "<function=get_weather>\n"
                                                   "<parameter=city>\nParis\n</parameter>\n"
                                                   "<parameter=days>\n2\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].id.rfind("call_", 0) == 0, "generated call id prefix");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "number parameter parsed");
    return failures;
}

int test_multiple_calls_and_json_values() {
    const ninfer::serve::ParsedToolCallOutput parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=first>\n"
        "<parameter=payload>\n{\"ok\":true,\"items\":[1,2]}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=second>\n"
        "<parameter=value>\nplain text\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multiple calls parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed calls");
    failures += check(parsed.tool_calls[0].name == "first", "first call name");
    failures += check(parsed.tool_calls[1].name == "second", "second call name");
    const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(first.at("payload").at("ok") == true, "object parameter bool");
    failures += check(first.at("payload").at("items").at(1) == 2, "object parameter array");
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(second.at("value") == "plain text", "plain text parameter string");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "<tool_call>\n<function=get_weather>\n";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content == text, "malformed xml preserved as text");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_falls_back_to_text() {
    const std::string text = "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "non-whitespace suffix falls back to text");
    failures += check(parsed.content == text, "suffix fallback preserves text");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const ninfer::serve::ParsedToolCallOutput anthropic =
        ninfer::serve::parse_qwen_tool_call_output(text, 128);
    const ninfer::serve::ParsedToolCallOutput openai =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const ninfer::serve::ParsedToolCallOutput too_long =
        ninfer::serve::parse_qwen_tool_call_output(too_long_text, 128);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1 &&
                          anthropic.tool_calls[0].name == name,
                      "128-character name accepted with Anthropic limit");
    failures +=
        check(!openai.is_tool_call_response, "128-character name rejected with OpenAI limit");
    failures +=
        check(!too_long.is_tool_call_response, "129-character name rejected with Anthropic limit");
    return failures;
}

int test_incremental_filter_valid_tool() {
    ninfer::serve::ToolCallStreamFilter filter;
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    visible += filter.finish(true);
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "valid tool filter did not stream the trimmed content prefix");
    failures +=
        check(filter.emitted_bytes() == visible.size(), "valid tool filter byte count mismatch");
    return failures;
}

int test_incremental_filter_fallback() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    ninfer::serve::ToolCallStreamFilter malformed;
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    restored += malformed.finish(false);

    ninfer::serve::ToolCallStreamFilter normal;
    std::string ordinary;
    ordinary += normal.feed("ordinary text  ");
    ordinary += normal.finish(false);

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    return failures;
}

// Provenance: independently hand-authored minimal examples, not parser-generated
// golden output or captured model responses. Format basis: the Qwen wire examples
// in test_single_call/test_multiple_calls_and_json_values above; stream basis:
// ToolCallStreamFilter's public header contract. No upstream capture is claimed.
// Policy expectations below intentionally pin current behavior, including quoted
// marker fallback, accepting undeclared names, and last duplicate parameter wins.
enum class CorpusCategory { quoted, later_call, name, duplicate, stream, client_limit };

struct ExpectedCall {
    std::string name;
    Json arguments;
};

struct RegressionCase {
    const char* id;
    CorpusCategory category;
    std::string text;
    std::size_t name_limit;
    bool is_tool_response;
    std::string content;
    std::vector<ExpectedCall> calls;
    std::string before_finish;
};

std::string wire_call(const std::string& name, const std::string& parameters = {}) {
    return "<tool_call>\n<function=" + name + ">\n" + parameters +
           "</function>\n</tool_call>";
}

// One oracle is shared by execution and the explicit partial-success
// counterexample. It never derives expected results from the parser under test.
bool matches_case(const RegressionCase& fixture,
                  const ninfer::serve::ParsedToolCallOutput& actual) {
    if (actual.is_tool_call_response != fixture.is_tool_response ||
        actual.content != fixture.content || actual.tool_calls.size() != fixture.calls.size()) {
        return false;
    }
    for (std::size_t i = 0; i < fixture.calls.size(); ++i) {
        const auto& call = actual.tool_calls[i];
        if (call.name != fixture.calls[i].name || call.id.size() != 21 ||
            call.id.rfind("call_", 0) != 0 ||
            call.id.find_first_not_of("0123456789abcdef", 5) != std::string::npos ||
            Json::parse(call.arguments_json, nullptr, false) != fixture.calls[i].arguments) {
            return false;
        }
    }
    return true;
}

int test_regression_corpus() {
    const std::string first = wire_call("first", "<parameter=value>\n1\n</parameter>\n");
    std::vector<RegressionCase> corpus;
    const auto fallback = [&](const char* id, CorpusCategory category, const std::string& text,
                              std::size_t limit = 64, const std::string& before_finish = "") {
        corpus.push_back({id, category, text, limit, false, text, {}, before_finish});
    };
    const auto accepted = [&](const char* id, CorpusCategory category, const std::string& text,
                              std::size_t limit, const std::string& content,
                              std::vector<ExpectedCall> calls) {
        corpus.push_back({id, category, text, limit, true, content, std::move(calls), content});
    };

    accepted("valid-first-control", CorpusCategory::later_call, first, 64, "",
             {{"first", Json{{"value", 1}}}});
    fallback("later-missing-tool-close", CorpusCategory::later_call,
             first + "\n<tool_call>\n<function=second>\n</function>");
    fallback("later-missing-function-close", CorpusCategory::later_call,
             first + "\n<tool_call>\n<function=second>\n</tool_call>");
    fallback("later-missing-parameter-close", CorpusCategory::later_call,
             first + "\n<tool_call>\n<function=second>\n<parameter=x>2\n"
                     "</function>\n</tool_call>");
    fallback("later-invalid-name", CorpusCategory::later_call,
             first + "\n" + wire_call("bad.name"));
    fallback("later-truncated-opener", CorpusCategory::later_call, first + "\n<tool_");

    const std::string quoted_prefix = "<think>The literal \"";
    fallback("quoted-opening-marker-before-call", CorpusCategory::quoted,
             quoted_prefix + "<tool_call>\" is documentation.</think>\n" + first,
             64, quoted_prefix);
    fallback("quoted-complete-call-in-reasoning", CorpusCategory::quoted,
             quoted_prefix + wire_call("example") + "\" is not a request.</think>\n" + first,
             64, quoted_prefix);
    const std::string harmless_reasoning =
        "<think>Quoted \"<function=example>\" and \"<parameter=x>\".</think>";
    accepted("quoted-non-tool-markers", CorpusCategory::quoted,
             harmless_reasoning + "\n" + first, 64, harmless_reasoning,
             {{"first", Json{{"value", 1}}}});

    fallback("empty-name", CorpusCategory::name, wire_call(""));
    fallback("space-in-name", CorpusCategory::name, wire_call("bad name"));
    fallback("punctuation-in-name", CorpusCategory::name, wire_call("bad.name"));
    fallback("missing-function-name-delimiter", CorpusCategory::name,
             "<tool_call>\n<function=broken\n</function>\n</tool_call>");
    // This parser accepts only a length limit, not a declared-tool registry.
    accepted("undeclared-syntactically-valid-name", CorpusCategory::name,
             wire_call("undeclared_7-tool"), 64, "", {{"undeclared_7-tool", Json::object()}});

    accepted("duplicate-string-last-wins", CorpusCategory::duplicate,
             wire_call("duplicate", "<parameter=x>first</parameter>\n"
                                    "<parameter=x>second</parameter>\n"),
             64, "", {{"duplicate", Json{{"x", "second"}}}});
    accepted("duplicate-type-last-wins", CorpusCategory::duplicate,
             wire_call("duplicate", "<parameter=x>plain</parameter>\n"
                                    "<parameter=keep>true</parameter>\n"
                                    "<parameter=x>{\"n\":2}</parameter>\n"),
             64, "", {{"duplicate", Json{{"x", Json{{"n", 2}}}, {"keep", true}}}});

    for (const std::size_t length : {std::size_t{64}, std::size_t{65}, std::size_t{128},
                                     std::size_t{129}}) {
        const std::string name(length, 'a');
        const std::string text = wire_call(name);
        // Stable row IDs identify every client boundary without random data.
        const char* openai_id = length == 64 ? "openai-64" : length == 65 ? "openai-65" :
                                length == 128 ? "openai-128" : "openai-129";
        const char* anthropic_id = length == 64 ? "anthropic-64" : length == 65 ? "anthropic-65" :
                                   length == 128 ? "anthropic-128" : "anthropic-129";
        if (length <= 64) {
            accepted(openai_id, CorpusCategory::client_limit, text, 64, "",
                     {{name, Json::object()}});
        } else {
            fallback(openai_id, CorpusCategory::client_limit, text, 64);
        }
        if (length <= 128) {
            accepted(anthropic_id, CorpusCategory::client_limit, text, 128, "",
                     {{name, Json::object()}});
        } else {
            fallback(anthropic_id, CorpusCategory::client_limit, text, 128);
        }
    }

    accepted("stream-trim-prefix", CorpusCategory::stream, "prefix \t\r\n" + first + "\n\t",
             64, "prefix", {{"first", Json{{"value", 1}}}});
    fallback("stream-restore-malformed", CorpusCategory::stream,
             "prefix \t\r\n<tool_call>\n<function=broken>", 64, "prefix");
    fallback("stream-partial-marker-at-eof", CorpusCategory::stream, "text \n<tool_", 64,
             "text");
    fallback("stream-near-marker", CorpusCategory::stream, "text <tool_X> \t", 64,
             "text <tool_X>");
    fallback("stream-whitespace-only", CorpusCategory::stream, " \t\r\n");
    fallback("stream-empty", CorpusCategory::stream, "");

    int failures = 0;
    std::size_t stream_runs = 0;
    std::size_t counterexamples = 0;
    const auto partial_success = ninfer::serve::parse_qwen_tool_call_output(first, 64);
    failures += check(partial_success.is_tool_call_response && partial_success.tool_calls.size() == 1,
                      "partial-success counterexample has a valid first call");
    for (const auto& fixture : corpus) {
        const std::string label = std::string("corpus/") + fixture.id;
        const auto parsed =
            ninfer::serve::parse_qwen_tool_call_output(fixture.text, fixture.name_limit);
        failures += check(matches_case(fixture, parsed), label + ": exact parser oracle");
        if (fixture.category == CorpusCategory::later_call && !fixture.is_tool_response) {
            failures += check(!matches_case(fixture, partial_success),
                              label + ": rejects valid-first partial-success counterexample");
            ++counterexamples;
        }
        const auto exercise_stream = [&](std::size_t split, bool bytewise) {
            ninfer::serve::ToolCallStreamFilter filter;
            std::string visible;
            const std::string run = label + (bytewise ? "/bytewise" : "/split-" + std::to_string(split));
            const auto feed = [&](std::string_view chunk) {
                visible += filter.feed(chunk);
                failures += check(filter.emitted_bytes() == visible.size(), run + ": feed byte count");
                failures += check(visible == fixture.text.substr(0, visible.size()),
                                  run + ": emitted bytes remain an exact input prefix");
            };
            feed("");
            if (bytewise) {
                for (std::size_t i = 0; i < fixture.text.size(); ++i) {
                    feed(std::string_view(fixture.text).substr(i, 1));
                }
            } else {
                feed(std::string_view(fixture.text).substr(0, split));
                feed("");
                feed(std::string_view(fixture.text).substr(split));
            }
            failures += check(visible == fixture.before_finish, run + ": pre-finish suppression");
            // Explicit table policy, not the parser's response flag, drives finish.
            visible += filter.finish(fixture.is_tool_response);
            failures += check(visible == fixture.content, run + ": exact terminal bytes");
            failures += check(filter.emitted_bytes() == visible.size(), run + ": terminal byte count");
            ++stream_runs;
        };
        for (std::size_t split = 0; split <= fixture.text.size(); ++split) {
            exercise_stream(split, false);
        }
        exercise_stream(0, true);
    }
    failures += check(counterexamples == 5, "five malformed-later-call counterexamples exercised");
    if (failures == 0) {
        std::cout << "corpus: " << corpus.size() << " rows; " << stream_runs
                  << " stream partitions; " << counterexamples
                  << " partial-success counterexamples rejected\n";
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_falls_back_to_text();
    failures += test_configured_name_limit();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    failures += test_regression_corpus();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
