#include "oracle/model/mapped_gguf.hpp"
#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/model/storage_decode.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/qwen35_block.hpp"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::filesystem::path model_path;
    std::uint32_t block_index{0};
    std::optional<std::uint32_t> token_id;
    std::optional<std::filesystem::path> input_path;
    std::optional<std::filesystem::path> output_path;
    std::vector<std::pair<std::string, std::filesystem::path>> overrides;
    std::uint64_t position{0};
    bool json{false};
};

[[noreturn]] void usage() {
    throw std::invalid_argument(
        "usage: oracle-qwen35-block <model.gguf> <block-index> "
        "(--token-id <id> | --input-f32 <path>) [--position <n>] "
        "[--output-f32 <path>] [--override-f32 <name>=<path>]... [--json]");
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

[[nodiscard]] Options parse_options(int argc, char** argv) {
    if (argc < 5) usage();
    Options options;
    options.model_path = argv[1];
    options.block_index = parse_u32(argv[2], "block index");

    for (int index = 3; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--json") {
            options.json = true;
        } else if (argument == "--token-id") {
            if (++index >= argc) usage();
            options.token_id = parse_u32(argv[index], "token id");
        } else if (argument == "--input-f32") {
            if (++index >= argc) usage();
            options.input_path = std::filesystem::path(argv[index]);
        } else if (argument == "--output-f32") {
            if (++index >= argc) usage();
            options.output_path = std::filesystem::path(argv[index]);
        } else if (argument == "--override-f32") {
            if (++index >= argc) usage();
            const std::string specification = argv[index];
            const std::size_t separator = specification.find('=');
            if (separator == std::string::npos || separator == 0 ||
                separator + 1 >= specification.size()) {
                throw std::invalid_argument(
                    "--override-f32 requires <trace-name>=<path>");
            }
            options.overrides.emplace_back(
                specification.substr(0, separator),
                std::filesystem::path(specification.substr(separator + 1)));
        } else if (argument == "--position") {
            if (++index >= argc) usage();
            options.position = parse_u64(argv[index], "position");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }

    if (options.token_id.has_value() == options.input_path.has_value()) {
        throw std::invalid_argument("choose exactly one of --token-id or --input-f32");
    }
    return options;
}

[[nodiscard]] std::vector<float> read_input(const std::filesystem::path& path,
                                            std::size_t expected) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open F32 input file: " + path.string());
    }
    std::vector<float> values;
    values.reserve(expected);
    float value = 0.0F;
    while (input >> value) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite floating-point value in input file: " +
                                     path.string());
        }
        values.push_back(value);
    }
    if (!input.eof()) {
        throw std::runtime_error("invalid floating-point value in input file: " + path.string());
    }
    if (values.size() != expected) {
        throw std::runtime_error("F32 input count mismatch: expected " +
                                 std::to_string(expected) + ", received " +
                                 std::to_string(values.size()));
    }
    return values;
}

[[nodiscard]] std::vector<float> read_override(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open matvec override file: " + path.string());
    }
    std::vector<float> values;
    float value = 0.0F;
    while (input >> value) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite floating-point value in override file: " +
                                     path.string());
        }
        values.push_back(value);
    }
    if (!input.eof()) {
        throw std::runtime_error("invalid floating-point value in override file: " +
                                 path.string());
    }
    if (values.empty()) {
        throw std::runtime_error("matvec override file is empty: " + path.string());
    }
    return values;
}

[[nodiscard]] std::vector<float> embedding_input(const oracle::model::GgufTensorView& embedding,
                                                 std::uint32_t token_id,
                                                 std::size_t expected_width) {
    const std::size_t rows = oracle::model::gguf_tensor_row_count(embedding);
    if (token_id >= rows) {
        throw std::out_of_range("token id exceeds embedding row count");
    }
    const oracle::model::StorageRowView row =
        oracle::model::make_storage_row_view(embedding, token_id);
    if (row.element_count != expected_width) {
        throw std::runtime_error("embedding row width does not match the manifest");
    }
    std::vector<float> values(expected_width);
    oracle::model::decode_storage_row(row, values);
    return values;
}

void write_output(const std::filesystem::path& path, std::span<const float> values) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("unable to open output file: " + path.string());
    }
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("refusing to write non-finite block output");
        }
        output << value << '\n';
    }
    if (!output) {
        throw std::runtime_error("failed while writing output file: " + path.string());
    }
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

        if (options.block_index >= weights.blocks.size()) {
            throw std::out_of_range("block index exceeds the Qwen3.5 backbone");
        }
        const oracle::model::Qwen35BackboneBlockWeights& block =
            weights.blocks[options.block_index];
        if (options.position != 0U) {
            throw std::invalid_argument(
                "the isolated Phase 2C CLI currently supports position 0 only; "
                "nonzero positions require restored prior block state");
        }

        std::vector<float> input;
        if (options.token_id.has_value()) {
            if (weights.token_embedding == nullptr) {
                throw std::runtime_error("bound Qwen3.5 model has no token embedding");
            }
            input = embedding_input(*weights.token_embedding,
                                    *options.token_id,
                                    manifest.embedding_length);
        } else {
            input = read_input(*options.input_path, manifest.embedding_length);
        }

        oracle::runtime::Qwen35MatvecOverrides matvec_overrides;
        matvec_overrides.tensors.reserve(options.overrides.size());
        for (const auto& [name, path] : options.overrides) {
            matvec_overrides.tensors.push_back({name, read_override(path)});
        }
        const oracle::runtime::Qwen35MatvecOverrides* override_pointer =
            matvec_overrides.tensors.empty() ? nullptr : &matvec_overrides;

        oracle::runtime::Qwen35BlockResult result;
        if (block.kind == oracle::model::Qwen35BlockKind::recurrent) {
            const oracle::runtime::Qwen35SsmLayout layout =
                oracle::runtime::qwen35_ssm_layout(manifest);
            oracle::runtime::SsmState state(layout.convolution_channels,
                                             layout.convolution_kernel,
                                             layout.value_heads,
                                             layout.key_head_dimension,
                                             layout.value_head_dimension);
            result = oracle::runtime::execute_qwen35_recurrent_block_reference(
                manifest, block, input, state, true, override_pointer);
        } else {
            oracle::runtime::KvCache cache(1,
                                            manifest.attention_head_count_kv,
                                            manifest.attention_key_length,
                                            manifest.attention_value_length);
            result = oracle::runtime::execute_qwen35_attention_block_reference(
                manifest,
                block,
                input,
                cache,
                oracle::runtime::RopePosition::text(options.position),
                true,
                override_pointer);
        }

        if (options.output_path.has_value()) {
            write_output(*options.output_path, result.output);
        }
        if (options.json) {
            std::cout << oracle::runtime::qwen35_block_trace_json(result.trace) << '\n';
        } else {
            std::cout << oracle::runtime::qwen35_block_trace_text(result.trace);
            if (options.token_id.has_value() && options.block_index != 0U) {
                std::cout << "note: --token-id feeds the embedding directly into an isolated block; "
                             "it is not a complete backbone prefix\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "oracle-qwen35-block: " << error.what() << '\n';
        return 1;
    }
}
