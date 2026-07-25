#include "oracle/runtime/qwen35_chat.hpp"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace oracle::runtime {
namespace {

[[nodiscard]] std::string_view role_name(ChatRole role) {
    switch (role) {
        case ChatRole::system:
            return "system";
        case ChatRole::user:
            return "user";
        case ChatRole::assistant:
            return "assistant";
        case ChatRole::tool:
            return "tool";
    }
    throw std::runtime_error("unknown chat role");
}

[[nodiscard]] std::string trim_newlines(std::string value) {
    while (!value.empty() && value.front() == '\n') {
        value.erase(value.begin());
    }
    while (!value.empty() && value.back() == '\n') {
        value.pop_back();
    }
    return value;
}

[[nodiscard]] std::string ltrim_newlines(std::string value) {
    while (!value.empty() && value.front() == '\n') {
        value.erase(value.begin());
    }
    return value;
}

[[nodiscard]] std::string rtrim_newlines(std::string value) {
    while (!value.empty() && value.back() == '\n') {
        value.pop_back();
    }
    return value;
}

[[nodiscard]] bool is_embedded_tool_response(std::string_view content) noexcept {
    constexpr std::string_view prefix = "<tool_response>";
    constexpr std::string_view suffix = "</tool_response>";
    return content.starts_with(prefix) && content.ends_with(suffix);
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    constexpr char digits[] = "0123456789abcdef";
                    output << "\\u00" << digits[(character >> 4U) & 0x0FU]
                           << digits[character & 0x0FU];
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

}  // namespace

std::string format_qwen35_chat(const Qwen35ChatRequest& request) {
    if (request.messages.empty()) {
        throw std::invalid_argument("Qwen35 chat requires at least one message");
    }

    std::ostringstream output;
    const bool first_is_system = request.messages.front().role == ChatRole::system;
    if (!request.tools_json.empty()) {
        output << "<|im_start|>system\n";
        if (first_is_system) {
            output << request.messages.front().content << "\n\n";
        }
        output << "# Tools\n\n"
                  "You may call one or more functions to assist with the user query.\n\n"
                  "You are provided with function signatures within <tools></tools> XML tags:\n"
                  "<tools>";
        for (const std::string& tool : request.tools_json) {
            output << '\n' << tool;
        }
        output << "\n</tools>\n\n"
                  "For each function call, return a json object with function name and arguments "
                  "within <tool_call></tool_call> XML tags:\n"
                  "<tool_call>\n"
                  "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                  "</tool_call><|im_end|>\n";
    } else if (first_is_system) {
        output << "<|im_start|>system\n" << request.messages.front().content
               << "<|im_end|>\n";
    }

    std::size_t last_query_index = request.messages.size() - 1;
    for (std::size_t reverse = request.messages.size(); reverse > 0; --reverse) {
        const std::size_t index = reverse - 1;
        const ChatMessage& message = request.messages[index];
        if (message.role == ChatRole::user &&
            !is_embedded_tool_response(message.content)) {
            last_query_index = index;
            break;
        }
    }

    for (std::size_t index = 0; index < request.messages.size(); ++index) {
        const ChatMessage& message = request.messages[index];
        const bool first = index == 0;
        const bool last = index + 1 == request.messages.size();

        if (message.role == ChatRole::user ||
            (message.role == ChatRole::system && !first)) {
            output << "<|im_start|>" << role_name(message.role) << '\n'
                   << message.content << "<|im_end|>\n";
            continue;
        }
        if (message.role == ChatRole::system) {
            continue;
        }

        if (message.role == ChatRole::assistant) {
            std::string content = message.content;
            std::string reasoning = message.reasoning_content;
            if (reasoning.empty()) {
                const std::size_t close = content.find("</think>");
                if (close != std::string::npos) {
                    std::string before = rtrim_newlines(content.substr(0, close));
                    const std::size_t open = before.rfind("<think>");
                    reasoning = ltrim_newlines(
                        open == std::string::npos ? before : before.substr(open + 7));
                    const std::size_t last_close = content.rfind("</think>");
                    content = ltrim_newlines(content.substr(last_close + 8));
                }
            }

            if (index > last_query_index && (last || !reasoning.empty())) {
                output << "<|im_start|>assistant\n<think>\n"
                       << trim_newlines(reasoning) << "\n</think>\n\n"
                       << ltrim_newlines(content);
            } else {
                output << "<|im_start|>assistant\n" << content;
            }

            for (std::size_t call_index = 0; call_index < message.tool_calls.size();
                 ++call_index) {
                if ((call_index == 0 && !content.empty()) || call_index != 0) {
                    output << '\n';
                }
                const ToolCall& call = message.tool_calls[call_index];
                if (call.name.empty() || call.arguments_json.empty()) {
                    throw std::invalid_argument(
                        "tool calls require a name and JSON arguments");
                }
                output << "<tool_call>\n{\"name\": \"" << json_escape(call.name)
                       << "\", \"arguments\": " << call.arguments_json
                       << "}\n</tool_call>";
            }
            output << "<|im_end|>\n";
            continue;
        }

        if (message.role == ChatRole::tool) {
            const bool previous_is_tool =
                index > 0 && request.messages[index - 1].role == ChatRole::tool;
            const bool next_is_tool =
                index + 1 < request.messages.size() &&
                request.messages[index + 1].role == ChatRole::tool;
            if (!previous_is_tool) {
                output << "<|im_start|>user";
            }
            output << "\n<tool_response>\n" << message.content
                   << "\n</tool_response>";
            if (!next_is_tool) {
                output << "<|im_end|>\n";
            }
        }
    }

    if (request.add_generation_prompt) {
        output << "<|im_start|>assistant\n<think>\n";
    }
    return output.str();
}

}  // namespace oracle::runtime
