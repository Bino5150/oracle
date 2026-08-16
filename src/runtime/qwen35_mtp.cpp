#include "oracle/runtime/qwen35_mtp.hpp"

#include "oracle/backend/cpu/quantized_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace oracle::runtime {
namespace {

void require_finite(std::span<const float> values, std::string_view label) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument("Qwen3.5 MTP " + std::string(label) +
                                        " contains a non-finite value at index " +
                                        std::to_string(index));
        }
    }
}

void require_tensor(const model::GgufTensorView* tensor, std::string_view role) {
    if (tensor == nullptr) {
        throw std::invalid_argument("Qwen3.5 MTP weight is missing: " + std::string(role));
    }
}

[[nodiscard]] std::vector<float> decode_tensor(const model::GgufTensorView& tensor) {
    std::vector<float> result(tensor.element_count());
    backend::cpu::reference_decode_mapped_tensor(tensor, result);
    return result;
}

// Mirrors qwen35_forward.cpp's private rms_norm() formula exactly (double-precision
// sum of squares, epsilon inside the sqrt, scale by weight): the same operation,
// applied here to enorm/hnorm/shared_head_norm rather than the backbone's final norm.
[[nodiscard]] std::vector<float> rms_norm(std::span<const float> input,
                                          std::span<const float> weight,
                                          float epsilon,
                                          std::string_view label) {
    if (input.empty() || input.size() != weight.size()) {
        throw std::invalid_argument("Qwen3.5 MTP RMSNorm dimensions are incompatible: " +
                                    std::string(label));
    }
    if (!(epsilon > 0.0F) || !std::isfinite(epsilon)) {
        throw std::invalid_argument("Qwen3.5 MTP RMSNorm epsilon must be positive and finite");
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
    require_finite(output, label);
    return output;
}

// Mirrors qwen35_forward.cpp's private embedding_row() exactly: decode only the one
// requested row via the mapped storage-row adapter, never the whole embedding matrix.
[[nodiscard]] std::vector<float> embedding_row(const model::GgufTensorView& embedding,
                                               std::uint32_t token_id,
                                               std::size_t expected_width,
                                               std::size_t expected_rows) {
    const std::size_t rows = model::gguf_tensor_row_count(embedding);
    if (rows != expected_rows) {
        throw std::runtime_error("Qwen3.5 MTP token embedding row count does not match the manifest");
    }
    if (static_cast<std::size_t>(token_id) >= rows) {
        throw std::out_of_range("Qwen3.5 MTP candidate token id exceeds the embedding row count");
    }
    const model::StorageRowView row = model::make_storage_row_view(embedding, token_id);
    if (row.element_count != expected_width) {
        throw std::runtime_error("Qwen3.5 MTP token embedding row width does not match the manifest");
    }
    std::vector<float> values(expected_width);
    model::decode_storage_row(row, values);
    require_finite(values, "candidate token embedding");
    return values;
}

void capture(Qwen35MtpTrace& trace, bool enabled, std::string name, std::span<const float> values) {
    if (!enabled) {
        return;
    }
    trace.tensors.push_back({std::move(name), std::vector<float>(values.begin(), values.end())});
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

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const char raw : value) {
        const auto character = static_cast<unsigned char>(raw);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
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

}  // namespace

const Qwen35TraceTensor* Qwen35MtpTrace::find(std::string_view name) const noexcept {
    for (const Qwen35TraceTensor& tensor : tensors) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return block_trace.find(name);
}

Qwen35MtpStepOutput execute_qwen35_reference_mtp_step(const model::Qwen35Manifest& manifest,
                                                       const model::Qwen35Weights& weights,
                                                       const Qwen35MtpInput& input,
                                                       KvCache& mtp_state,
                                                       bool capture_intermediates) {
    if (!manifest.has_mtp()) {
        throw std::invalid_argument("Qwen3.5 MTP execution requires manifest.has_mtp()");
    }
    if (!weights.mtp.has_value()) {
        throw std::invalid_argument("Qwen3.5 MTP execution requires bound MTP weights");
    }
    const model::Qwen35MtpWeights& mtp = *weights.mtp;

    require_tensor(mtp.input_norm, "MTP attn_norm");
    require_tensor(mtp.post_attention_norm, "MTP post_attention_norm");
    require_tensor(mtp.attention.query, "MTP attention query");
    require_tensor(mtp.attention.key, "MTP attention key");
    require_tensor(mtp.attention.value, "MTP attention value");
    require_tensor(mtp.attention.output, "MTP attention output");
    require_tensor(mtp.attention.query_norm, "MTP attention query norm");
    require_tensor(mtp.attention.key_norm, "MTP attention key norm");
    require_tensor(mtp.mlp.gate, "MTP FFN gate");
    require_tensor(mtp.mlp.up, "MTP FFN up");
    require_tensor(mtp.mlp.down, "MTP FFN down");
    require_tensor(mtp.embedding_hidden_projection, "MTP eh_proj");
    require_tensor(mtp.embedding_norm, "MTP enorm");
    require_tensor(mtp.hidden_norm, "MTP hnorm");
    require_tensor(weights.token_embedding, "token embedding");
    require_tensor(weights.output, "output projection");

    if (input.target_hidden_state.size() != manifest.embedding_length) {
        throw std::invalid_argument("Qwen3.5 MTP target hidden-state width mismatch");
    }
    require_finite(input.target_hidden_state, "target hidden state");
    if (static_cast<std::size_t>(input.candidate_token_id) >= manifest.vocabulary_size) {
        throw std::out_of_range("Qwen3.5 MTP candidate token id exceeds vocabulary size");
    }
    if (input.position.temporal >= manifest.context_length ||
        input.position.height >= manifest.context_length ||
        input.position.width >= manifest.context_length) {
        throw std::invalid_argument("Qwen3.5 MTP position exceeds the manifest context length");
    }

    Qwen35MtpResult result;
    Qwen35MtpTrace& trace = result.trace;
    capture(trace, capture_intermediates, "target_hidden_state", input.target_hidden_state);

    // Token embedding: mtp.embed_tokens if present, else tied to the backbone's own
    // token_embd (source-verified: graph_mtp.cpp:694,
    // `tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd`).
    const model::GgufTensorView& embed_source =
        mtp.embed_tokens != nullptr ? *mtp.embed_tokens : *weights.token_embedding;
    const std::vector<float> token_embedding = embedding_row(
        embed_source, input.candidate_token_id, manifest.embedding_length, manifest.vocabulary_size);
    capture(trace, capture_intermediates, "mtp_tok_embd", token_embedding);

    // h_norm = RMSNorm(hnorm, h); e_norm = RMSNorm(enorm, token_embedding)
    // (graph_mtp.cpp:715-719).
    const std::vector<float> hnorm_weight = decode_tensor(*mtp.hidden_norm);
    const std::vector<float> h_norm =
        rms_norm(input.target_hidden_state, hnorm_weight, manifest.attention_rms_epsilon, "hnorm");
    capture(trace, capture_intermediates, "mtp_hnorm", h_norm);

    const std::vector<float> enorm_weight = decode_tensor(*mtp.embedding_norm);
    const std::vector<float> e_norm =
        rms_norm(token_embedding, enorm_weight, manifest.attention_rms_epsilon, "enorm");
    capture(trace, capture_intermediates, "mtp_enorm", e_norm);

    // concat = [e_norm; h_norm], e_norm FIRST along the feature dimension
    // (source-verified: graph_mtp.cpp:721, `ggml_concat(ctx0, e_norm, h_norm, dim=0)`).
    std::vector<float> concat(e_norm.size() + h_norm.size());
    std::copy(e_norm.begin(), e_norm.end(), concat.begin());
    std::copy(h_norm.begin(), h_norm.end(), concat.begin() + static_cast<std::ptrdiff_t>(e_norm.size()));
    capture(trace, capture_intermediates, "mtp_concat", concat);

    // eh_proj: Oracle's generic scalar reference matvec (Q8_0-capable since Slice 1B;
    // this is the real checkpoint's actual storage type for this tensor). No
    // MTP-specific kernel -- the same decode-to-F32-then-scalar-matvec path used for
    // every other Qwen3.5 matrix.
    const auto eh_proj_dims = mtp.embedding_hidden_projection->dimensions();
    if (eh_proj_dims.size() != 2 || eh_proj_dims[0] != concat.size() ||
        eh_proj_dims[1] != manifest.embedding_length) {
        throw std::runtime_error("Qwen3.5 MTP eh_proj dimensions do not match the manifest");
    }
    std::vector<float> eh_proj_output(manifest.embedding_length);
    backend::cpu::reference_mapped_tensor_matvec(*mtp.embedding_hidden_projection, concat,
                                                 eh_proj_output);
    require_finite(eh_proj_output, "eh_proj output");
    capture(trace, capture_intermediates, "mtp_eh_proj", eh_proj_output);

    // The appended MTP block's own attention+FFN sub-computation, residualed against
    // eh_proj_output, is source-verified architecturally identical to an ordinary
    // Qwen3.5 full-attention backbone block (fused Q+gate projection, per-head Q/K
    // RMSNorm, MRoPE, sigmoid-gated scaled-dot-product attention, output projection,
    // residual, post-attention norm, SwiGLU FFN, residual -- graph_mtp.cpp:727-795 is
    // line-for-line the same sequence as build_layer_attn/finish_block). Reused here
    // rather than duplicated, per Phase 2F Slice 2's "shared math stays shared" rule.
    model::Qwen35BackboneBlockWeights mtp_block;
    mtp_block.index = mtp.block_index;
    mtp_block.kind = model::Qwen35BlockKind::full_attention;
    mtp_block.input_norm = mtp.input_norm;
    mtp_block.post_attention_norm = mtp.post_attention_norm;
    mtp_block.mlp = mtp.mlp;
    mtp_block.attention = mtp.attention;

    // mtp_state is caller-owned: execute_qwen35_reference_mtp (below) passes a fresh,
    // ephemeral capacity-1 cache (Slice 2's isolated single-prediction contract,
    // unchanged); Slice 4's draft chain passes the same growing cache across
    // multiple calls to reproduce the pinned reference's chained MTP attention.
    // Either way this state is never the canonical 24-block HybridCache.
    if (mtp_state.key_value_heads() != manifest.attention_head_count_kv ||
        mtp_state.key_dimension() != manifest.attention_key_length ||
        mtp_state.value_dimension() != manifest.attention_value_length) {
        throw std::invalid_argument("Qwen3.5 MTP state cache dimensions do not match the manifest");
    }

    Qwen35BlockResult block_result = execute_qwen35_attention_block_reference(
        manifest, mtp_block, eh_proj_output, mtp_state, input.position, capture_intermediates);
    trace.block_trace = std::move(block_result.trace);
    const std::vector<float>& block_output = block_result.output;

    // shared_head_norm: mtp.shared_head_norm if present, else the backbone's own
    // output_norm (source-verified: graph_mtp.cpp:797-801). The real checkpoint
    // supplies shared_head_norm, so this fallback is exercised only by synthetic
    // tests, never by the real fixture.
    const model::GgufTensorView* head_norm_tensor =
        mtp.shared_head_norm != nullptr ? mtp.shared_head_norm : weights.output_norm;
    require_tensor(head_norm_tensor, "MTP shared_head_norm (and no output_norm fallback)");
    const std::vector<float> head_norm_weight = decode_tensor(*head_norm_tensor);
    const std::vector<float> shared_head_norm_output =
        rms_norm(block_output, head_norm_weight, manifest.attention_rms_epsilon, "shared_head_norm");
    capture(trace, capture_intermediates, "mtp_shared_head_norm", shared_head_norm_output);

    // Tied/shared output head: mtp.shared_head if present, else weights.output --
    // which is itself already tied to weights.token_embedding when the checkpoint has
    // no explicit output.weight (source-verified: graph_mtp.cpp:809-811,
    // `head_w = layer.nextn.shared_head_head ? ... : model.output`, and model.output
    // in beellama is duplicated from tok_embd under the same condition). The real
    // checkpoint has neither shared_head_head nor an explicit output.weight, so this
    // call resolves to the ordinary tied token embedding, exactly as section 12 of
    // the Slice 2 taskblock requires confirming.
    const model::GgufTensorView* head_projection =
        mtp.shared_head != nullptr ? mtp.shared_head : weights.output;
    const auto head_dims = head_projection->dimensions();
    if (head_dims.size() != 2 || head_dims[0] != manifest.embedding_length ||
        head_dims[1] != manifest.vocabulary_size) {
        throw std::runtime_error("Qwen3.5 MTP output projection dimensions do not match the manifest");
    }
    result.logits.resize(manifest.vocabulary_size);
    backend::cpu::reference_mapped_tensor_matvec(*head_projection, shared_head_norm_output,
                                                 result.logits);
    if (result.logits.size() != manifest.vocabulary_size) {
        throw std::runtime_error("Qwen3.5 MTP draft logits count does not match the manifest");
    }
    require_finite(result.logits, "MTP draft logits");
    capture(trace, capture_intermediates, "result_output", result.logits);

    std::size_t best_index = 0;
    float best_value = result.logits.front();
    for (std::size_t index = 1; index < result.logits.size(); ++index) {
        if (result.logits[index] > best_value) {
            best_value = result.logits[index];
            best_index = index;
        }
    }
    result.argmax_token_id = static_cast<std::uint32_t>(best_index);

    Qwen35MtpStepOutput step_output;
    step_output.result = std::move(result);
    step_output.chained_hidden_state = shared_head_norm_output;
    return step_output;
}

Qwen35MtpResult execute_qwen35_reference_mtp(const model::Qwen35Manifest& manifest,
                                             const model::Qwen35Weights& weights,
                                             const Qwen35MtpInput& input,
                                             bool capture_intermediates) {
    // Isolated, ephemeral single-position attention state: the MTP block has never
    // seen any token besides this one through its own attention, and this state is
    // discarded when this function returns. It is never the canonical 24-block
    // HybridCache and never persists or chains across calls (Phase 2F Slice 2 scope
    // is exactly one NextN prediction; no draft loop, no rollback). Behavior and
    // signature are unchanged from Slice 2 -- this is now a thin wrapper around
    // execute_qwen35_reference_mtp_step().
    KvCache mtp_state(1, manifest.attention_head_count_kv, manifest.attention_key_length,
                      manifest.attention_value_length);
    return execute_qwen35_reference_mtp_step(manifest, weights, input, mtp_state,
                                             capture_intermediates)
        .result;
}

std::string qwen35_mtp_trace_text(const Qwen35MtpResult& result) {
    std::ostringstream output;
    output << "Qwen3.5 MTP reference prediction\n"
           << "contract: projection=" << result.contracts.execution_projection
           << " cache=" << result.contracts.execution_attention_cache
           << " state=" << result.contracts.state_ownership << '\n'
           << "argmax_token_id: " << result.argmax_token_id << '\n'
           << "logits count: " << result.logits.size() << '\n';
    output << std::setprecision(9);
    for (const Qwen35TraceTensor& tensor : result.trace.tensors) {
        output << "- " << tensor.name << ": count=" << tensor.values.size()
               << " fnv1a64=0x" << std::hex << fingerprint(tensor.values) << std::dec << '\n';
    }
    for (const Qwen35TraceTensor& tensor : result.trace.block_trace.tensors) {
        output << "- block." << tensor.name << ": count=" << tensor.values.size()
               << " fnv1a64=0x" << std::hex << fingerprint(tensor.values) << std::dec << '\n';
    }
    return output.str();
}

std::string qwen35_mtp_trace_json(const Qwen35MtpResult& result, bool include_logits) {
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << '{' << "\"argmax_token_id\":" << result.argmax_token_id << ','
           << "\"logits_count\":" << result.logits.size() << ','
           << "\"contracts\":{" << "\"execution_projection\":\""
           << json_escape(result.contracts.execution_projection) << "\","
           << "\"execution_attention_cache\":\""
           << json_escape(result.contracts.execution_attention_cache) << "\","
           << "\"state_ownership\":\"" << json_escape(result.contracts.state_ownership) << "\"},";
    if (include_logits) {
        output << "\"logits\":[";
        for (std::size_t index = 0; index < result.logits.size(); ++index) {
            if (index != 0) output << ',';
            output << result.logits[index];
        }
        output << "],";
    }
    output << "\"tensors\":[";
    bool first = true;
    for (const Qwen35TraceTensor& tensor : result.trace.tensors) {
        if (!first) output << ',';
        first = false;
        output << '{' << "\"name\":\"" << json_escape(tensor.name) << "\","
               << "\"count\":" << tensor.values.size() << ','
               << "\"fnv1a64\":\"0x" << std::hex << fingerprint(tensor.values) << std::dec << "\"}";
    }
    for (const Qwen35TraceTensor& tensor : result.trace.block_trace.tensors) {
        if (!first) output << ',';
        first = false;
        output << '{' << "\"name\":\"block." << json_escape(tensor.name) << "\","
               << "\"count\":" << tensor.values.size() << ','
               << "\"fnv1a64\":\"0x" << std::hex << fingerprint(tensor.values) << std::dec << "\"}";
    }
    output << "]}";
    return output.str();
}

}  // namespace oracle::runtime
