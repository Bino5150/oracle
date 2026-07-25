#include "oracle/model/gguf.hpp"
#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/runtime/qwen35_chat.hpp"
#include "oracle/tokenizer/qwen35_tokenizer.hpp"

#include <charconv>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class Mode {
    none,
    manifest,
    encode,
    decode,
    chat,
};

struct Options {
    std::string path;
    Mode mode{Mode::none};
    std::string value;
    std::optional<std::string> system_message;
    bool json{false};
    bool parse_special{false};
    bool skip_special{false};
    bool add_generation_prompt{true};
    bool tokenize_chat{false};
};

[[nodiscard]] std::string usage() {
    return
        "usage:\n"
        "  oracle-tokenize <model.gguf> --manifest [--json]\n"
        "  oracle-tokenize <model.gguf> --encode <text> [--parse-special] [--json]\n"
        "  oracle-tokenize <model.gguf> --decode <id,id,...> [--skip-special] [--json]\n"
        "  oracle-tokenize <model.gguf> --chat-user <text> [--chat-system <text>] "
        "[--no-generation-prompt] [--tokenize-chat] [--json]";
}

void set_mode(Options& options, Mode mode, std::string value = {}) {
    if (options.mode != Mode::none) {
        throw std::invalid_argument("choose exactly one operation\n" + usage());
    }
    options.mode = mode;
    options.value = std::move(value);
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    if (argc < 3) {
        throw std::invalid_argument(usage());
    }
    Options options;
    options.path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto require_value = [argc, argv, &index, argument]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a value");
            }
            return argv[++index];
        };
        if (argument == "--manifest") {
            set_mode(options, Mode::manifest);
        } else if (argument == "--encode") {
            set_mode(options, Mode::encode, require_value());
        } else if (argument == "--decode") {
            set_mode(options, Mode::decode, require_value());
        } else if (argument == "--chat-user") {
            set_mode(options, Mode::chat, require_value());
        } else if (argument == "--chat-system") {
            options.system_message = require_value();
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--parse-special") {
            options.parse_special = true;
        } else if (argument == "--skip-special") {
            options.skip_special = true;
        } else if (argument == "--no-generation-prompt") {
            options.add_generation_prompt = false;
        } else if (argument == "--tokenize-chat") {
            options.tokenize_chat = true;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument) +
                                        "\n" + usage());
        }
    }
    if (options.mode == Mode::none) {
        throw std::invalid_argument(usage());
    }
    if (options.system_message && options.mode != Mode::chat) {
        throw std::invalid_argument("--chat-system requires --chat-user");
    }
    return options;
}

[[nodiscard]] std::vector<oracle::tokenizer::TokenId> parse_token_ids(
    std::string_view text) {
    std::vector<oracle::tokenizer::TokenId> ids;
    std::size_t position = 0;
    while (position < text.size()) {
        while (position < text.size() &&
               (text[position] == ',' || text[position] == ' ' || text[position] == '\t')) {
            ++position;
        }
        if (position == text.size()) {
            break;
        }
        const std::size_t begin = position;
        while (position < text.size() && text[position] >= '0' && text[position] <= '9') {
            ++position;
        }
        if (position == begin) {
            throw std::invalid_argument("invalid token-id list");
        }
        std::uint32_t value = 0;
        const char* first = text.data() + begin;
        const char* last = text.data() + position;
        const auto [pointer, error] = std::from_chars(first, last, value);
        if (error != std::errc{} || pointer != last) {
            throw std::invalid_argument("invalid token id");
        }
        ids.push_back(value);
        if (position < text.size() && text[position] != ',' && text[position] != ' ' &&
            text[position] != '\t') {
            throw std::invalid_argument("invalid token-id separator");
        }
    }
    return ids;
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 16);
    constexpr char digits[] = "0123456789abcdef";
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"':
                output += "\\\"";
                break;
            case '\\':
                output += "\\\\";
                break;
            case '\b':
                output += "\\b";
                break;
            case '\f':
                output += "\\f";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                output += "\\r";
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                if (character < 0x20U) {
                    output += "\\u00";
                    output.push_back(digits[(character >> 4U) & 0x0FU]);
                    output.push_back(digits[character & 0x0FU]);
                } else {
                    output.push_back(static_cast<char>(character));
                }
        }
    }
    return output;
}

void print_manifest(const oracle::model::Qwen35Manifest& manifest, bool json) {
    if (json) {
        std::cout << oracle::model::qwen35_manifest_json(manifest) << '\n';
        return;
    }
    std::cout << "Oracle Qwen35 manifest\n"
              << "architecture=" << manifest.architecture
              << " model=" << manifest.model_name << '\n'
              << "blocks=" << manifest.total_block_count
              << " backbone=" << manifest.backbone_block_count
              << " nextn=" << manifest.nextn_predict_layers
              << " mtp=" << (manifest.has_mtp() ? "yes" : "no") << '\n'
              << "context=" << manifest.context_length
              << " embedding=" << manifest.embedding_length
              << " ffn=" << manifest.feed_forward_length << '\n'
              << "heads=" << manifest.attention_head_count
              << " kv_heads=" << manifest.attention_head_count_kv
              << " key_length=" << manifest.attention_key_length
              << " value_length=" << manifest.attention_value_length << '\n'
              << "rope_dims=" << manifest.rope_dimension_count
              << " rope_sections=[" << manifest.rope_dimension_sections[0] << ','
              << manifest.rope_dimension_sections[1] << ','
              << manifest.rope_dimension_sections[2] << ','
              << manifest.rope_dimension_sections[3] << "]"
              << " rope_base=" << manifest.rope_frequency_base << '\n'
              << "ssm_inner=" << manifest.ssm_inner_size
              << " ssm_state=" << manifest.ssm_state_size
              << " ssm_groups=" << manifest.ssm_group_count
              << " ssm_rank=" << manifest.ssm_time_step_rank
              << " conv_kernel=" << manifest.ssm_convolution_kernel
              << " full_attention_interval=" << manifest.full_attention_interval << '\n'
              << "vocabulary=" << manifest.vocabulary_size
              << " bos=";
    if (manifest.bos_token_id) {
        std::cout << *manifest.bos_token_id;
    } else {
        std::cout << "none";
    }
    std::cout << " eos=";
    if (manifest.eos_token_id) {
        std::cout << *manifest.eos_token_id;
    } else {
        std::cout << "none";
    }
    std::cout << " padding=";
    if (manifest.padding_token_id) {
        std::cout << *manifest.padding_token_id;
    } else {
        std::cout << "none";
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const oracle::model::GgufFile file = oracle::model::GgufReader::read(options.path);
        const oracle::model::Qwen35Manifest manifest =
            oracle::model::load_qwen35_manifest(file);

        if (options.mode == Mode::manifest) {
            print_manifest(manifest, options.json);
            return 0;
        }

        const oracle::tokenizer::Qwen35Tokenizer tokenizer(file);
        if (options.mode == Mode::encode) {
            const std::vector<oracle::tokenizer::TokenId> ids = tokenizer.encode(
                options.value,
                oracle::tokenizer::EncodeOptions{
                    .parse_special_tokens = options.parse_special});
            if (options.json) {
                std::cout << "{\"text\":\"" << json_escape(options.value)
                          << "\",\"token_count\":" << ids.size()
                          << ",\"token_ids\":" << oracle::tokenizer::token_ids_json(ids)
                          << "}\n";
            } else {
                std::cout << "token_count=" << ids.size() << "\nids=";
                for (std::size_t index = 0; index < ids.size(); ++index) {
                    if (index != 0) {
                        std::cout << ',';
                    }
                    std::cout << ids[index];
                }
                std::cout << '\n';
            }
            return 0;
        }

        if (options.mode == Mode::decode) {
            const std::vector<oracle::tokenizer::TokenId> ids =
                parse_token_ids(options.value);
            const std::string text = tokenizer.decode(
                ids,
                oracle::tokenizer::DecodeOptions{
                    .skip_special_tokens = options.skip_special});
            if (options.json) {
                std::cout << "{\"token_ids\":" << oracle::tokenizer::token_ids_json(ids)
                          << ",\"text\":\"" << json_escape(text) << "\"}\n";
            } else {
                std::cout << text << '\n';
            }
            return 0;
        }

        oracle::runtime::Qwen35ChatRequest request;
        if (options.system_message) {
            request.messages.push_back(
                {oracle::runtime::ChatRole::system, *options.system_message, {}, {}});
        }
        request.messages.push_back(
            {oracle::runtime::ChatRole::user, options.value, {}, {}});
        request.add_generation_prompt = options.add_generation_prompt;
        const std::string prompt = oracle::runtime::format_qwen35_chat(request);
        if (options.tokenize_chat) {
            const std::vector<oracle::tokenizer::TokenId> ids = tokenizer.encode(
                prompt,
                oracle::tokenizer::EncodeOptions{.parse_special_tokens = true});
            if (options.json) {
                std::cout << "{\"prompt\":\"" << json_escape(prompt)
                          << "\",\"token_count\":" << ids.size()
                          << ",\"token_ids\":" << oracle::tokenizer::token_ids_json(ids)
                          << "}\n";
            } else {
                std::cout << prompt << "\n---\ntoken_count=" << ids.size() << "\nids=";
                for (std::size_t index = 0; index < ids.size(); ++index) {
                    if (index != 0) {
                        std::cout << ',';
                    }
                    std::cout << ids[index];
                }
                std::cout << '\n';
            }
        } else if (options.json) {
            std::cout << "{\"prompt\":\"" << json_escape(prompt) << "\"}\n";
        } else {
            std::cout << prompt;
        }
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Oracle tokenization failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
