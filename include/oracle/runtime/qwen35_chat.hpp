#pragma once

#include <string>
#include <vector>

namespace oracle::runtime {

enum class ChatRole {
    system,
    user,
    assistant,
    tool,
};

struct ToolCall {
    std::string name;
    std::string arguments_json;
};

struct ChatMessage {
    ChatRole role{ChatRole::user};
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
};

struct Qwen35ChatRequest {
    std::vector<ChatMessage> messages;
    std::vector<std::string> tools_json;
    bool add_generation_prompt{true};
};

[[nodiscard]] std::string format_qwen35_chat(const Qwen35ChatRequest& request);

}  // namespace oracle::runtime
