#include "oracle/runtime/qwen35_forward.hpp"

#include "oracle/backend/cpu/quantized_reference.hpp"
#include "oracle/model/storage_decode.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace oracle::runtime {
namespace {

[[nodiscard]] std::size_t checked_size(std::uint64_t value, std::string_view context) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(std::string(context) + " exceeds addressable size");
    }
    return static_cast<std::size_t>(value);
}

void require_tensor(const model::GgufTensorView* tensor, std::string_view role) {
    if (tensor == nullptr) {
        throw std::invalid_argument("Qwen3.5 forward weight is missing: " + std::string(role));
    }
}

void require_finite(std::span<const float> values, std::string_view name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error("Qwen3.5 forward contains a non-finite value in " +
                                     std::string(name) + " at index " +
                                     std::to_string(index));
        }
    }
}

[[nodiscard]] std::vector<float> decode_tensor(const model::GgufTensorView& tensor) {
    const std::size_t elements = checked_size(tensor.element_count(), "mapped tensor element count");
    std::vector<float> result(elements);
    backend::cpu::reference_decode_mapped_tensor(tensor, result);
    return result;
}

[[nodiscard]] std::vector<float> embedding_row(const model::GgufTensorView& embedding,
                                               std::uint32_t token_id,
                                               std::size_t expected_width,
                                               std::size_t expected_rows) {
    const std::size_t rows = model::gguf_tensor_row_count(embedding);
    if (rows != expected_rows) {
        throw std::runtime_error("Qwen3.5 embedding row count does not match the manifest");
    }
    if (static_cast<std::size_t>(token_id) >= rows) {
        throw std::out_of_range("Qwen3.5 token id exceeds the embedding row count");
    }
    const model::StorageRowView row = model::make_storage_row_view(embedding, token_id);
    if (row.element_count != expected_width) {
        throw std::runtime_error("Qwen3.5 embedding row width does not match the manifest");
    }
    std::vector<float> values(expected_width);
    model::decode_storage_row(row, values);
    require_finite(values, "token_embedding");
    return values;
}

[[nodiscard]] std::vector<float> rms_norm(std::span<const float> input,
                                          std::span<const float> weight,
                                          float epsilon) {
    if (input.empty() || input.size() != weight.size()) {
        throw std::invalid_argument("Qwen3.5 final RMSNorm dimensions are incompatible");
    }
    if (!(epsilon > 0.0F) || !std::isfinite(epsilon)) {
        throw std::invalid_argument("Qwen3.5 final RMSNorm epsilon must be positive and finite");
    }

    double sum_squares = 0.0;
    for (const float value : input) {
        const double widened = static_cast<double>(value);
        sum_squares += widened * widened;
    }
    const double mean_square = sum_squares / static_cast<double>(input.size());
    const float inverse_rms =
        static_cast<float>(1.0 / std::sqrt(mean_square + static_cast<double>(epsilon)));

    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index] * inverse_rms * weight[index];
    }
    require_finite(output, "final_norm");
    return output;
}

[[nodiscard]] std::size_t matrix_rows(const model::GgufTensorView& tensor,
                                      std::string_view role) {
    const auto dimensions = tensor.dimensions();
    if (dimensions.size() != 2) {
        throw std::invalid_argument("Qwen3.5 " + std::string(role) + " must be rank 2");
    }
    return checked_size(dimensions[1], role);
}

[[nodiscard]] const model::GgufTensorView* projection_for_name(
    const model::Qwen35BackboneBlockWeights& block,
    std::string_view name) {
    if (name == "ffn_gate") return block.mlp.gate;
    if (name == "ffn_up") return block.mlp.up;
    if (name == "ffn_out") return block.mlp.down;

    if (block.kind == model::Qwen35BlockKind::recurrent && block.recurrent.has_value()) {
        const model::Qwen35RecurrentWeights& weights = *block.recurrent;
        if (name == "linear_attn_qkv_mixed") return weights.qkv;
        if (name == "z") return weights.gate;
        if (name == "beta") return weights.beta;
        if (name == "alpha") return weights.alpha;
        if (name == "linear_attn_out") return weights.output;
        return nullptr;
    }

    if (block.kind == model::Qwen35BlockKind::full_attention && block.attention.has_value()) {
        const model::Qwen35AttentionWeights& weights = *block.attention;
        if (name == "Qcur_full") return weights.query;
        if (name == "Kcur_projected") return weights.key;
        if (name == "Vcur_projected") return weights.value;
        if (name == "attn_output") return weights.output;
    }
    return nullptr;
}

void validate_overrides(const model::Qwen35Manifest& manifest,
                        const model::Qwen35Weights& weights,
                        const Qwen35ForwardOverrides* overrides) {
    if (overrides == nullptr) return;

    if ((!overrides->blocks.empty() || overrides->logits.has_value()) &&
        overrides->projection_source_contract.empty()) {
        throw std::invalid_argument("Qwen3.5 override projection contract label must not be empty");
    }
    if (!overrides->blocks.empty() && overrides->attention_cache_source_contract.empty()) {
        throw std::invalid_argument("Qwen3.5 override cache contract label must not be empty");
    }

    for (std::size_t index = 0; index < overrides->blocks.size(); ++index) {
        const Qwen35ForwardBlockOverrides& entry = overrides->blocks[index];
        if (entry.block_index >= weights.blocks.size()) {
            throw std::out_of_range("Qwen3.5 forward override block index exceeds the backbone");
        }
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (overrides->blocks[prior].block_index == entry.block_index) {
                throw std::invalid_argument("duplicate Qwen3.5 forward override block index");
            }
        }

        const model::Qwen35BackboneBlockWeights& block = weights.blocks[entry.block_index];
        for (std::size_t tensor_index = 0; tensor_index < entry.matvecs.tensors.size();
             ++tensor_index) {
            const Qwen35TraceTensor& replacement = entry.matvecs.tensors[tensor_index];
            if (replacement.name.empty()) {
                throw std::invalid_argument("Qwen3.5 forward override name must not be empty");
            }
            for (std::size_t prior = 0; prior < tensor_index; ++prior) {
                if (entry.matvecs.tensors[prior].name == replacement.name) {
                    throw std::invalid_argument("duplicate Qwen3.5 forward override name in block " +
                                                std::to_string(entry.block_index));
                }
            }
            const model::GgufTensorView* projection = projection_for_name(block, replacement.name);
            if (projection == nullptr) {
                throw std::invalid_argument("unknown Qwen3.5 forward override for block " +
                                            std::to_string(entry.block_index) + ": " +
                                            replacement.name);
            }
            const std::size_t expected = matrix_rows(*projection, "override projection");
            if (replacement.values.size() != expected) {
                throw std::invalid_argument("Qwen3.5 forward override count mismatch for block " +
                                            std::to_string(entry.block_index) + " tensor " +
                                            replacement.name + ": expected " +
                                            std::to_string(expected) + ", received " +
                                            std::to_string(replacement.values.size()));
            }
            require_finite(replacement.values, replacement.name);
        }
    }

    if (overrides->logits.has_value()) {
        const Qwen35TraceTensor& logits = *overrides->logits;
        if (logits.name != "logits") {
            throw std::invalid_argument("Qwen3.5 logits override must be named logits");
        }
        if (logits.values.size() != manifest.vocabulary_size) {
            throw std::invalid_argument("Qwen3.5 logits override count mismatch");
        }
        require_finite(logits.values, "logits override");
    }
}

void validate_forward_contract(const model::Qwen35Manifest& manifest,
                               const model::Qwen35Weights& weights,
                               const HybridCache& state,
                               std::uint32_t token_id,
                               RopePosition position) {
    if (manifest.architecture != "qwen35") {
        throw std::invalid_argument("Qwen3.5 forward requires architecture qwen35");
    }
    if (manifest.embedding_length == 0 || manifest.vocabulary_size == 0 ||
        manifest.backbone_block_count == 0) {
        throw std::invalid_argument("Qwen3.5 forward manifest dimensions must be nonzero");
    }
    if (weights.blocks.size() != manifest.backbone_block_count ||
        state.block_count() != manifest.backbone_block_count) {
        throw std::invalid_argument("Qwen3.5 forward backbone/state count mismatch");
    }
    if (token_id >= manifest.vocabulary_size) {
        throw std::out_of_range("Qwen3.5 token id exceeds the manifest vocabulary");
    }
    if (position.temporal != position.height || position.temporal != position.width) {
        throw std::invalid_argument("Phase 2D forward accepts text positions only");
    }
    if (position.temporal != state.sequence_length()) {
        throw std::invalid_argument("Qwen3.5 forward position does not match the hybrid state");
    }
    require_tensor(weights.token_embedding, "token embedding");
    require_tensor(weights.output_norm, "output norm");
    if (!weights.output_is_tied && weights.output == nullptr) {
        throw std::invalid_argument("Qwen3.5 explicit output projection is missing");
    }

    std::size_t recurrent = 0;
    std::size_t attention = 0;
    for (std::size_t index = 0; index < weights.blocks.size(); ++index) {
        const model::Qwen35BackboneBlockWeights& block = weights.blocks[index];
        if (block.index != index) {
            throw std::invalid_argument("Qwen3.5 forward block indices are not contiguous");
        }
        const bool expected_attention = manifest.is_full_attention_block(
            static_cast<std::uint32_t>(index));
        const bool actual_attention = block.kind == model::Qwen35BlockKind::full_attention;
        if (actual_attention != expected_attention || state.is_attention_block(index) != expected_attention) {
            throw std::invalid_argument("Qwen3.5 forward block/state family mismatch");
        }
        if (actual_attention) {
            ++attention;
        } else {
            ++recurrent;
        }
    }
    if (recurrent != weights.report.recurrent_block_count ||
        attention != weights.report.attention_block_count) {
        throw std::invalid_argument("Qwen3.5 forward binding-report block counts are inconsistent");
    }
}

[[nodiscard]] std::uint64_t fingerprint(std::span<const float> values) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const float value : values) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0; shift < 32U; shift += 8U) {
            hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xFFU);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

struct VectorSummary {
    float minimum{0.0F};
    float maximum{0.0F};
    double rms{0.0};
    std::uint64_t hash{0};
};

[[nodiscard]] VectorSummary summarize(std::span<const float> values) {
    require_finite(values, "trace");
    VectorSummary summary;
    summary.hash = fingerprint(values);
    if (values.empty()) return summary;

    summary.minimum = std::numeric_limits<float>::infinity();
    summary.maximum = -std::numeric_limits<float>::infinity();
    double sum_squares = 0.0;
    for (const float value : values) {
        summary.minimum = std::min(summary.minimum, value);
        summary.maximum = std::max(summary.maximum, value);
        const double widened = static_cast<double>(value);
        sum_squares += widened * widened;
    }
    summary.rms = std::sqrt(sum_squares / static_cast<double>(values.size()));
    return summary;
}

[[nodiscard]] std::vector<std::size_t> top_indices(std::span<const float> values,
                                                   std::size_t count) {
    std::vector<std::size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    const std::size_t retained = std::min(count, indices.size());
    std::partial_sort(indices.begin(),
                      indices.begin() + static_cast<std::ptrdiff_t>(retained),
                      indices.end(),
                      [&](std::size_t left, std::size_t right) {
                          if (values[left] != values[right]) return values[left] > values[right];
                          return left < right;
                      });
    indices.resize(retained);
    return indices;
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const char raw : value) {
        const auto character = static_cast<unsigned char>(raw);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned int>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

void append_json_values(std::ostringstream& output, std::span<const float> values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        output << values[index];
    }
    output << ']';
}

void append_json_vector(std::ostringstream& output,
                        std::string_view name,
                        std::span<const float> values,
                        bool include_values) {
    const VectorSummary summary = summarize(values);
    output << '{'
           << "\"name\":\"" << json_escape(name) << "\","
           << "\"count\":" << values.size() << ','
           << "\"min\":" << summary.minimum << ','
           << "\"max\":" << summary.maximum << ','
           << "\"rms\":" << summary.rms << ','
           << "\"fnv1a64\":\"0x" << std::hex << summary.hash << std::dec << '"';
    if (include_values) {
        output << ",\"values\":";
        append_json_values(output, values);
    }
    output << '}';
}

void append_text_summary(std::ostringstream& output,
                         std::string_view name,
                         std::span<const float> values) {
    const VectorSummary summary = summarize(values);
    output << name << ": count=" << values.size()
           << " min=" << summary.minimum
           << " max=" << summary.maximum
           << " rms=" << summary.rms
           << " fnv1a64=0x" << std::hex << summary.hash << std::dec << '\n';
}

}  // namespace

const Qwen35MatvecOverrides* Qwen35ForwardOverrides::find_block(
    std::uint32_t block_index) const noexcept {
    const auto iterator = std::find_if(blocks.begin(), blocks.end(), [&](const auto& entry) {
        return entry.block_index == block_index;
    });
    return iterator == blocks.end() ? nullptr : &iterator->matvecs;
}

Qwen35ForwardResult execute_qwen35_reference_token(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35Weights& weights,
    std::uint32_t token_id,
    HybridCache& state,
    RopePosition position,
    bool capture_block_outputs,
    const Qwen35ForwardOverrides* overrides) {
    validate_forward_contract(manifest, weights, state, token_id, position);
    validate_overrides(manifest, weights, overrides);

    Qwen35ForwardTrace trace;
    trace.token_id = token_id;
    trace.position = position.temporal;
    trace.output_is_tied = weights.output_is_tied;
    trace.mtp_executed = false;
    if (overrides != nullptr && (!overrides->blocks.empty() || overrides->logits.has_value())) {
        trace.contracts.execution_projection = "diagnostic-external-overrides";
        trace.contracts.override_projection_source = overrides->projection_source_contract;
        trace.contracts.override_attention_cache_source = overrides->attention_cache_source_contract;
    }

    std::vector<float> hidden = embedding_row(*weights.token_embedding,
                                               token_id,
                                               manifest.embedding_length,
                                               manifest.vocabulary_size);
    trace.embedding = hidden;
    trace.blocks.reserve(weights.blocks.size());

    const std::size_t initial_position = state.sequence_length();
    for (const model::Qwen35BackboneBlockWeights& block : weights.blocks) {
        const Qwen35MatvecOverrides* block_overrides =
            overrides == nullptr ? nullptr : overrides->find_block(block.index);
        Qwen35BlockResult block_result;
        if (block.kind == model::Qwen35BlockKind::recurrent) {
            block_result = execute_qwen35_recurrent_block_reference(
                manifest, block, hidden, state.ssm(block.index), false, block_overrides);
        } else {
            block_result = execute_qwen35_attention_block_reference(
                manifest, block, hidden, state.attention(block.index), position, false, block_overrides);
        }
        if (block_result.output.size() != manifest.embedding_length) {
            throw std::runtime_error("Qwen3.5 block output width does not match the manifest");
        }
        require_finite(block_result.output, "block output");
        hidden = std::move(block_result.output);
        Qwen35ForwardBlockTrace block_trace;
        block_trace.block_index = block.index;
        block_trace.kind = block.kind;
        if (capture_block_outputs) block_trace.output = hidden;
        trace.blocks.push_back(std::move(block_trace));
    }

    if (state.sequence_length() != initial_position + 1) {
        throw std::runtime_error("Qwen3.5 hybrid state did not advance exactly one token");
    }

    const std::vector<float> output_norm_weight = decode_tensor(*weights.output_norm);
    if (output_norm_weight.size() != manifest.embedding_length) {
        throw std::runtime_error("Qwen3.5 output norm width does not match the manifest");
    }
    trace.final_norm = rms_norm(hidden,
                                output_norm_weight,
                                manifest.attention_rms_epsilon);

    std::vector<float> logits;
    if (overrides != nullptr && overrides->logits.has_value()) {
        logits = overrides->logits->values;
    } else {
        const model::GgufTensorView* output_projection =
            weights.output_is_tied ? weights.token_embedding : weights.output;
        require_tensor(output_projection, "output projection");
        const auto dimensions = output_projection->dimensions();
        if (dimensions.size() != 2 || dimensions[0] != manifest.embedding_length ||
            dimensions[1] != manifest.vocabulary_size) {
            throw std::runtime_error("Qwen3.5 output projection dimensions do not match the manifest");
        }
        logits.resize(manifest.vocabulary_size);
        backend::cpu::reference_mapped_tensor_matvec(
            *output_projection, trace.final_norm, logits);
    }
    if (logits.size() != manifest.vocabulary_size) {
        throw std::runtime_error("Qwen3.5 logits count does not match the manifest");
    }
    require_finite(logits, "logits");
    return {std::move(logits), std::move(trace)};
}

std::string qwen35_forward_trace_text(const Qwen35ForwardResult& result) {
    std::ostringstream output;
    output << std::setprecision(9);
    output << "Qwen3.5 full reference forward\n"
           << "token_id: " << result.trace.token_id << '\n'
           << "position: " << result.trace.position << '\n'
           << "blocks: " << result.trace.blocks.size() << '\n'
           << "output_head: " << (result.trace.output_is_tied ? "tied" : "explicit") << '\n'
           << "mtp_executed: " << (result.trace.mtp_executed ? "yes" : "no") << '\n'
           << "execution_projection_contract: "
           << result.trace.contracts.execution_projection << '\n'
           << "execution_attention_cache_contract: "
           << result.trace.contracts.execution_attention_cache << '\n'
           << "independent_projection_reference: "
           << result.trace.contracts.independent_projection_reference << '\n'
           << "independent_attention_cache_reference: "
           << result.trace.contracts.independent_attention_cache_reference << '\n';
    if (!result.trace.contracts.override_projection_source.empty()) {
        output << "override_projection_source: "
               << result.trace.contracts.override_projection_source << '\n'
               << "override_attention_cache_source: "
               << result.trace.contracts.override_attention_cache_source << '\n';
    }

    append_text_summary(output, "embedding", result.trace.embedding);
    for (const Qwen35ForwardBlockTrace& block : result.trace.blocks) {
        if (!block.output.empty()) {
            append_text_summary(output,
                                "block." + std::to_string(block.block_index) + ".output",
                                block.output);
        }
    }
    append_text_summary(output, "final_norm", result.trace.final_norm);
    append_text_summary(output, "logits", result.logits);

    const std::vector<std::size_t> top = top_indices(result.logits, 20);
    output << "top20:";
    for (const std::size_t index : top) {
        output << ' ' << index << ':' << result.logits[index];
    }
    output << '\n';
    return output.str();
}

std::string qwen35_forward_trace_json(const Qwen35ForwardResult& result,
                                      bool include_logits) {
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << '{'
           << "\"token_id\":" << result.trace.token_id << ','
           << "\"position\":" << result.trace.position << ','
           << "\"contracts\":{"
           << "\"execution_projection\":\""
           << json_escape(result.trace.contracts.execution_projection) << "\","
           << "\"execution_attention_cache\":\""
           << json_escape(result.trace.contracts.execution_attention_cache) << "\","
           << "\"independent_projection_reference\":\""
           << json_escape(result.trace.contracts.independent_projection_reference) << "\","
           << "\"independent_attention_cache_reference\":\""
           << json_escape(result.trace.contracts.independent_attention_cache_reference) << "\","
           << "\"override_projection_source\":\""
           << json_escape(result.trace.contracts.override_projection_source) << "\","
           << "\"override_attention_cache_source\":\""
           << json_escape(result.trace.contracts.override_attention_cache_source) << "\"},"
           << "\"output_head\":\""
           << (result.trace.output_is_tied ? "tied" : "explicit") << "\","
           << "\"mtp_executed\":" << (result.trace.mtp_executed ? "true" : "false") << ','
           << "\"embedding\":";
    append_json_vector(output, "embedding", result.trace.embedding, true);
    output << ",\"blocks\":[";
    for (std::size_t index = 0; index < result.trace.blocks.size(); ++index) {
        if (index != 0) output << ',';
        const Qwen35ForwardBlockTrace& block = result.trace.blocks[index];
        output << '{'
               << "\"block_index\":" << block.block_index << ','
               << "\"kind\":\"" << model::qwen35_block_kind_name(block.kind) << "\","
               << "\"output\":";
        append_json_vector(output, "output", block.output, true);
        output << '}';
    }
    output << "],\"final_norm\":";
    append_json_vector(output, "final_norm", result.trace.final_norm, true);
    output << ",\"logits\":";
    append_json_vector(output, "logits", result.logits, include_logits);

    const std::vector<std::size_t> top = top_indices(result.logits, 20);
    output << ",\"top20\":[";
    for (std::size_t index = 0; index < top.size(); ++index) {
        if (index != 0) output << ',';
        output << "{\"token_id\":" << top[index]
               << ",\"value\":" << result.logits[top[index]] << '}';
    }
    output << "]}";
    return output.str();
}

}  // namespace oracle::runtime
