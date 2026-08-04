#include "oracle/model/mapped_gguf.hpp"
#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/qwen35_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct OverrideSpecification {
    std::uint32_t block_index{0};
    std::string name;
    std::filesystem::path path;
};

struct Options {
    std::filesystem::path model_path;
    std::uint32_t token_id{0};
    std::uint64_t position{0};
    std::optional<std::filesystem::path> logits_path;
    std::optional<std::filesystem::path> trace_path;
    std::vector<OverrideSpecification> overrides;
    std::optional<std::filesystem::path> logits_override_path;
    std::string override_projection_contract{"q5-k/q6-k-x-q8-k"};
    std::string override_cache_contract{"f16-pinned"};
    bool json{false};
    bool full_json{false};
};

[[noreturn]] void usage() {
    throw std::invalid_argument(
        "usage: oracle-qwen35-forward <model.gguf> --token-id <id> [--position 0] "
        "[--logits-f32 <path>] [--trace-json <path>] "
        "[--override-f32 <block>:<name>=<path>]... [--override-logits-f32 <path>] "
        "[--override-projection-contract <label>] [--override-cache-contract <label>] "
        "[--json | --full-json]");
}

[[nodiscard]] std::uint64_t parse_u64(std::string_view text, std::string_view name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(std::string(text), &consumed, 10);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string(name) + " contains trailing characters");
    }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::uint32_t parse_u32(std::string_view text, std::string_view name) {
    const std::uint64_t value = parse_u64(text, name);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::out_of_range(std::string(name) + " exceeds uint32 range");
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] OverrideSpecification parse_override(std::string_view specification) {
    const std::size_t equals = specification.find('=');
    const std::size_t colon = specification.find(':');
    if (colon == std::string_view::npos || equals == std::string_view::npos ||
        colon == 0 || colon + 1 >= equals || equals + 1 >= specification.size()) {
        throw std::invalid_argument(
            "--override-f32 requires <block-index>:<trace-name>=<path>");
    }
    OverrideSpecification result;
    result.block_index = parse_u32(specification.substr(0, colon), "override block index");
    result.name = std::string(specification.substr(colon + 1, equals - colon - 1));
    result.path = std::filesystem::path(specification.substr(equals + 1));
    return result;
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    if (argc < 4) usage();
    Options options;
    options.model_path = argv[1];
    bool saw_token = false;

    for (int index = 2; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--token-id") {
            if (++index >= argc) usage();
            options.token_id = parse_u32(argv[index], "token id");
            saw_token = true;
        } else if (argument == "--position") {
            if (++index >= argc) usage();
            options.position = parse_u64(argv[index], "position");
        } else if (argument == "--logits-f32") {
            if (++index >= argc) usage();
            options.logits_path = std::filesystem::path(argv[index]);
        } else if (argument == "--trace-json") {
            if (++index >= argc) usage();
            options.trace_path = std::filesystem::path(argv[index]);
        } else if (argument == "--override-f32") {
            if (++index >= argc) usage();
            options.overrides.push_back(parse_override(argv[index]));
        } else if (argument == "--override-logits-f32") {
            if (++index >= argc) usage();
            options.logits_override_path = std::filesystem::path(argv[index]);
        } else if (argument == "--override-projection-contract") {
            if (++index >= argc) usage();
            options.override_projection_contract = argv[index];
        } else if (argument == "--override-cache-contract") {
            if (++index >= argc) usage();
            options.override_cache_contract = argv[index];
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--full-json") {
            options.json = true;
            options.full_json = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (!saw_token) {
        throw std::invalid_argument("--token-id is required");
    }
    if (options.position != 0U) {
        throw std::invalid_argument(
            "the Phase 2D CLI starts with fresh state and therefore supports position 0 only");
    }
    if (options.override_projection_contract.empty() || options.override_cache_contract.empty()) {
        throw std::invalid_argument("override contract labels must not be empty");
    }
    return options;
}

[[nodiscard]] std::vector<float> read_f32(const std::filesystem::path& path,
                                          std::string_view role) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open " + std::string(role) + " file: " + path.string());
    }
    std::vector<float> values;
    float value = 0.0F;
    while (input >> value) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite value in " + std::string(role) + " file: " +
                                     path.string());
        }
        values.push_back(value);
    }
    if (!input.eof()) {
        throw std::runtime_error("invalid floating-point value in " + std::string(role) +
                                 " file: " + path.string());
    }
    if (values.empty()) {
        throw std::runtime_error(std::string(role) + " file is empty: " + path.string());
    }
    return values;
}

void write_f32(const std::filesystem::path& path,
               std::span<const float> values,
               std::string_view role) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("unable to open " + std::string(role) + " output: " + path.string());
    }
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("refusing to write non-finite " + std::string(role));
        }
        output << value << '\n';
    }
    if (!output) {
        throw std::runtime_error("failed while writing " + std::string(role) + " output: " +
                                 path.string());
    }
}

void write_text(const std::filesystem::path& path, std::string_view text) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("unable to open trace output: " + path.string());
    }
    output << text << '\n';
    if (!output) {
        throw std::runtime_error("failed while writing trace output: " + path.string());
    }
}

[[nodiscard]] oracle::runtime::Qwen35ForwardOverrides build_overrides(
    const Options& options) {
    oracle::runtime::Qwen35ForwardOverrides result;
    result.projection_source_contract = options.override_projection_contract;
    result.attention_cache_source_contract = options.override_cache_contract;

    for (const OverrideSpecification& specification : options.overrides) {
        auto iterator = std::find_if(result.blocks.begin(), result.blocks.end(), [&](const auto& entry) {
            return entry.block_index == specification.block_index;
        });
        if (iterator == result.blocks.end()) {
            result.blocks.push_back({specification.block_index, {}});
            iterator = std::prev(result.blocks.end());
        }
        iterator->matvecs.tensors.push_back(
            {specification.name, read_f32(specification.path, "matvec override")});
    }
    if (options.logits_override_path.has_value()) {
        result.logits = oracle::runtime::Qwen35TraceTensor{
            "logits", read_f32(*options.logits_override_path, "logits override")};
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const oracle::model::MappedGgufModel model(options.model_path);
        const oracle::model::Qwen35Manifest manifest =
            oracle::model::load_qwen35_manifest(model.file());
        const oracle::model::Qwen35Weights weights =
            oracle::model::bind_qwen35_weights(model, manifest);

        if (manifest.backbone_block_count != 32U ||
            weights.report.recurrent_block_count != 24U ||
            weights.report.attention_block_count != 8U) {
            throw std::runtime_error(
                "Phase 2D requires the production 32-block, 24-recurrent/8-attention backbone");
        }
        oracle::runtime::HybridCache state(manifest, 1);
        oracle::runtime::Qwen35ForwardOverrides overrides = build_overrides(options);
        const bool has_overrides = !overrides.blocks.empty() || overrides.logits.has_value();
        const oracle::runtime::Qwen35ForwardResult result =
            oracle::runtime::execute_qwen35_reference_token(
                manifest,
                weights,
                options.token_id,
                state,
                oracle::runtime::RopePosition::text(options.position),
                true,
                has_overrides ? &overrides : nullptr);

        if (options.logits_path.has_value()) {
            write_f32(*options.logits_path, result.logits, "logits");
        }
        const std::string json = oracle::runtime::qwen35_forward_trace_json(
            result, options.full_json);
        if (options.trace_path.has_value()) {
            write_text(*options.trace_path, json);
        }
        if (options.json) {
            std::cout << json << '\n';
        } else {
            std::cout << oracle::runtime::qwen35_forward_trace_text(result);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "oracle-qwen35-forward: " << error.what() << '\n';
        return 1;
    }
}
