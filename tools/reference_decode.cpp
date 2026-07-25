#include "oracle/runtime/sampler.hpp"
#include "oracle/runtime/tiny_hybrid_model.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::vector<std::uint32_t> prompt{1, 5, 2};
    std::size_t new_tokens{4};
    float temperature{0.0F};
    std::size_t top_k{0};
    float top_p{1.0F};
    std::uint64_t seed{5150};
    bool json{false};
};

[[nodiscard]] std::uint64_t parse_u64(std::string_view text, const char* field) {
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + std::string(text));
    }
    return value;
}

[[nodiscard]] float parse_float(std::string_view text, const char* field) {
    std::string owned(text);
    std::size_t consumed = 0;
    const float value = std::stof(owned, &consumed);
    if (consumed != owned.size() || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + owned);
    }
    return value;
}

[[nodiscard]] std::vector<std::uint32_t> parse_tokens(std::string_view text) {
    if (text.empty()) {
        throw std::invalid_argument("--tokens requires a comma-separated token list");
    }
    std::vector<std::uint32_t> tokens;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string_view part = text.substr(
            begin, end == std::string_view::npos ? text.size() - begin : end - begin);
        const std::uint64_t value = parse_u64(part, "token id");
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("token id exceeds uint32 range");
        }
        tokens.push_back(static_cast<std::uint32_t>(value));
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return tokens;
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto require_value = [&]() -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a value");
            }
            return argv[++index];
        };
        if (argument == "--tokens") {
            options.prompt = parse_tokens(require_value());
        } else if (argument == "--steps") {
            options.new_tokens = static_cast<std::size_t>(parse_u64(require_value(), "step count"));
        } else if (argument == "--temperature") {
            options.temperature = parse_float(require_value(), "temperature");
        } else if (argument == "--top-k") {
            options.top_k = static_cast<std::size_t>(parse_u64(require_value(), "top-k"));
        } else if (argument == "--top-p") {
            options.top_p = parse_float(require_value(), "top-p");
        } else if (argument == "--seed") {
            options.seed = parse_u64(require_value(), "seed");
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "usage: oracle-reference-decode [--tokens 1,5,2] [--steps 4] "
                   "[--temperature 0] [--top-k 0] [--top-p 1] [--seed 5150] [--json]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    if (options.prompt.empty()) {
        throw std::invalid_argument("prompt token list cannot be empty");
    }
    return options;
}

void print_tokens(std::span<const std::uint32_t> tokens) {
    std::cout << '[';
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << tokens[index];
    }
    std::cout << ']';
}

[[nodiscard]] float maximum_difference(std::span<const float> left,
                                       std::span<const float> right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("logit vectors have different sizes");
    }
    float maximum = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        oracle::runtime::TinyHybridModel model({}, options.seed);
        const std::size_t capacity = options.prompt.size() + options.new_tokens + 1;
        oracle::runtime::TinyHybridState state(model.config(), capacity);
        oracle::runtime::Sampler sampler({options.temperature,
                                          options.top_k,
                                          options.top_p,
                                          options.seed});

        const auto generated = model.generate(options.prompt,
                                              options.new_tokens,
                                              state,
                                              sampler);

        oracle::runtime::TinyHybridState replay_state(model.config(), capacity);
        auto replay_logits = model.prefill(options.prompt, replay_state);
        for (const std::uint32_t token : generated) {
            replay_logits = model.forward_token(token, replay_state);
        }

        oracle::runtime::TinyHybridState second_replay(model.config(), capacity);
        auto second_logits = model.prefill(options.prompt, second_replay);
        for (const std::uint32_t token : generated) {
            second_logits = model.forward_token(token, second_replay);
        }
        const float replay_error = maximum_difference(replay_logits.data(),
                                                      second_logits.data());

        if (options.json) {
            std::cout << "{\"model\":\"tiny-hybrid-reference\",\"prompt\":";
            print_tokens(options.prompt);
            std::cout << ",\"generated\":";
            print_tokens(generated);
            std::cout << ",\"sequence_length\":" << state.sequence_length()
                      << ",\"state_capacity\":" << state.capacity()
                      << ",\"state_bytes\":" << state.byte_size()
                      << ",\"replay_max_error\":" << std::setprecision(9)
                      << replay_error << "}\n";
        } else {
            std::cout << "Oracle tiny hybrid reference decode\n"
                      << "prompt=";
            print_tokens(options.prompt);
            std::cout << " generated=";
            print_tokens(generated);
            std::cout << "\nsequence_length=" << state.sequence_length()
                      << " state_capacity=" << state.capacity()
                      << " state_bytes=" << state.byte_size()
                      << " replay_max_error=" << std::setprecision(9)
                      << replay_error << '\n';
        }
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "reference decode failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
