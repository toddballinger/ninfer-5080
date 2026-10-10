#pragma once

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {

// Pure host-side candidate validator. No emission-path integration yet.
// Unsupported schema constraints fail closed rather than being ignored.
struct ToolValidationResult {
    std::vector<ToolCall> accepted;
    std::vector<std::string> rejection_codes;
};

namespace detail {
using Json = nlohmann::json;

// nlohmann::json normally overwrites duplicate object members. Detect them
// during parsing, before a last-key-wins object can pass schema validation.
// Object scopes are independent, including objects nested inside arrays.
inline Json parse_unique_keys(const std::string& raw, bool& duplicated) {
    duplicated = false;
    std::vector<std::set<std::string>> object_scopes;
    const auto callback = [&](int, Json::parse_event_t event, Json& value) {
        switch (event) {
        case Json::parse_event_t::object_start:
            object_scopes.emplace_back();
            break;
        case Json::parse_event_t::key:
            if (object_scopes.empty() ||
                !object_scopes.back().insert(value.get<std::string>()).second) {
                duplicated = true;
            }
            break;
        case Json::parse_event_t::object_end:
            if (!object_scopes.empty()) { object_scopes.pop_back(); }
            break;
        default:
            break;
        }
        return true; // do not discard keys before detection
    };
    return Json::parse(raw, callback, false);
}

inline bool supported_schema(const Json& schema, unsigned depth = 0) {
    if (!schema.is_object() || depth > 16) { return false; }
    static const std::set<std::string> keys = {
        "type", "properties", "required", "additionalProperties", "items",
        "description", "title"
    };
    for (auto it = schema.begin(); it != schema.end(); ++it) {
        if (!keys.count(it.key())) { return false; }
    }
    if (schema.contains("type")) {
        if (!schema["type"].is_string()) { return false; }
        const std::string t = schema["type"].get<std::string>();
        if (t != "object" && t != "array" && t != "string" &&
            t != "integer" && t != "number" && t != "boolean" && t != "null") {
            return false;
        }
    } else {
        return false;
    }
    const std::string type = schema["type"].get<std::string>();
    if (schema.contains("description") && !schema["description"].is_string()) { return false; }
    if (schema.contains("title") && !schema["title"].is_string()) { return false; }
    if (schema.contains("properties")) {
        if (type != "object" || !schema["properties"].is_object()) { return false; }
        for (auto it = schema["properties"].begin(); it != schema["properties"].end(); ++it) {
            if (!supported_schema(it.value(), depth + 1)) { return false; }
        }
    }
    if (schema.contains("required")) {
        if (type != "object" || !schema["required"].is_array()) { return false; }
        std::set<std::string> names;
        for (const auto& name : schema["required"]) {
            if (!name.is_string() || !names.insert(name.get<std::string>()).second) { return false; }
        }
    }
    if (schema.contains("additionalProperties") &&
        (type != "object" || !schema["additionalProperties"].is_boolean())) { return false; }
    if (schema.contains("items") &&
        (type != "array" || !supported_schema(schema["items"], depth + 1))) { return false; }
    return true;
}

inline bool value_matches(const Json& value, const Json& schema, unsigned depth = 0) {
    if (depth > 16) { return false; }
    const std::string type = schema["type"].get<std::string>();
    if (type == "object") {
        if (!value.is_object()) { return false; }
        if (schema.contains("required")) {
            for (const auto& name : schema["required"]) {
                if (!value.contains(name.get<std::string>())) { return false; }
            }
        }
        const Json props = schema.value("properties", Json::object());
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (props.contains(it.key())) {
                if (!value_matches(it.value(), props[it.key()], depth + 1)) { return false; }
            } else if (schema.value("additionalProperties", true) == false) {
                return false;
            }
        }
        return true;
    }
    if (type == "array") {
        if (!value.is_array()) { return false; }
        if (schema.contains("items")) {
            for (const auto& item : value) {
                if (!value_matches(item, schema["items"], depth + 1)) { return false; }
            }
        }
        return true;
    }
    if (type == "string") { return value.is_string(); }
    if (type == "number") { return value.is_number(); }
    if (type == "integer") { return value.is_number_integer() || value.is_number_unsigned(); }
    if (type == "boolean") { return value.is_boolean(); }
    return value.is_null();
}

inline bool syntactically_valid_name(const std::string& name, std::size_t limit) {
    if (name.empty() || name.size() > limit) { return false; }
    for (unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}
} // namespace detail

// Reject all candidates if request definitions are ambiguous.
// Each call is independently authorized; invalid calls are never emitted.
inline ToolValidationResult validate_candidate_calls(
    const std::vector<ToolCall>& candidates, const std::vector<ToolDefinition>& declared,
    const ToolChoice& choice, std::size_t name_limit) {
    ToolValidationResult result;
    std::set<std::string> names;
    bool duplicate_definition = false;
    for (const auto& def : declared) {
        if (!names.insert(def.name).second) { duplicate_definition = true; }
    }
    for (const ToolCall& call : candidates) {
        auto reject = [&](const char* reason) {
            result.rejection_codes.emplace_back(reason);
        };
        if (duplicate_definition) { reject("AMBIGUOUS_DECLARATIONS"); continue; }
        if (!detail::syntactically_valid_name(call.name, name_limit)) {
            reject("NAME_LENGTH_OR_SYNTAX"); continue;
        }
        if (choice.mode == ToolChoiceMode::None ||
            (choice.mode == ToolChoiceMode::Named && choice.name != call.name)) {
            reject("TOOL_CHOICE_DENIED"); continue;
        }
        const ToolDefinition* def = nullptr;
        for (const auto& candidate : declared) {
            if (candidate.name == call.name) { def = &candidate; break; }
        }
        if (def == nullptr) { reject("UNDECLARED_TOOL"); continue; }
        bool duplicate_args = false;
        const auto args = detail::parse_unique_keys(call.arguments_json, duplicate_args);
        if (duplicate_args) { reject("DUPLICATE_PARAMETER"); continue; }
        if (args.is_discarded() || !args.is_object()) {
            reject("ARGUMENT_JSON_INVALID"); continue;
        }
        bool duplicate_schema_keys = false;
        const auto schema = detail::parse_unique_keys(def->parameters_json,
                                                       duplicate_schema_keys);
        if (duplicate_schema_keys || schema.is_discarded() ||
            !detail::supported_schema(schema)) {
            reject("SCHEMA_UNSUPPORTED"); continue;
        }
        if (!detail::value_matches(args, schema)) {
            reject("ARGUMENT_SCHEMA_INVALID"); continue;
        }
        result.accepted.push_back(call);
    }
    return result;
}
} // namespace ninfer::serve
