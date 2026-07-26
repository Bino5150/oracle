#include "oracle/runtime/qwen35_block.hpp"

#include "oracle/backend/cpu/quantized_reference.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
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

[[nodiscard]] std::size_t checked_multiply(std::size_t left,
                                           std::size_t right,
                                           std::string_view context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left,
                                      std::size_t right,
                                      std::string_view context) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left + right;
}

void require_tensor(const model::GgufTensorView* tensor, std::string_view role) {
    if (tensor == nullptr) {
        throw std::invalid_argument("Qwen3.5 block weight is missing: " + std::string(role));
    }
}

[[nodiscard]] std::vector<float> decode_tensor(const model::GgufTensorView& tensor) {
    const std::size_t elements = checked_size(tensor.element_count(), "mapped tensor element count");
    std::vector<float> result(elements);
    backend::cpu::reference_decode_mapped_tensor(tensor, result);
    return result;
}

class MatvecOverrideCursor {
public:
    MatvecOverrideCursor(const Qwen35MatvecOverrides* overrides,
                         std::span<const std::string_view> allowed_names)
        : overrides_(overrides) {
        if (overrides_ == nullptr) {
            return;
        }
        for (std::size_t index = 0; index < overrides_->tensors.size(); ++index) {
            const Qwen35TraceTensor& tensor = overrides_->tensors[index];
            if (tensor.name.empty()) {
                throw std::invalid_argument("Qwen3.5 matvec override name must not be empty");
            }
            const bool allowed = std::find(allowed_names.begin(), allowed_names.end(), tensor.name) !=
                                 allowed_names.end();
            if (!allowed) {
                throw std::invalid_argument("unknown Qwen3.5 matvec override: " + tensor.name);
            }
            for (std::size_t prior = 0; prior < index; ++prior) {
                if (overrides_->tensors[prior].name == tensor.name) {
                    throw std::invalid_argument("duplicate Qwen3.5 matvec override: " + tensor.name);
                }
            }
            for (const float value : tensor.values) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("non-finite Qwen3.5 matvec override: " + tensor.name);
                }
            }
        }
    }

    [[nodiscard]] const Qwen35TraceTensor* find(std::string_view name,
                                                std::size_t expected_count) const {
        if (overrides_ == nullptr) {
            return nullptr;
        }
        const Qwen35TraceTensor* tensor = overrides_->find(name);
        if (tensor != nullptr && tensor->values.size() != expected_count) {
            throw std::invalid_argument("Qwen3.5 matvec override count mismatch for " +
                                        std::string(name) + ": expected " +
                                        std::to_string(expected_count) + ", received " +
                                        std::to_string(tensor->values.size()));
        }
        return tensor;
    }

private:
    const Qwen35MatvecOverrides* overrides_{nullptr};
};

[[nodiscard]] std::vector<float> mapped_matvec(const model::GgufTensorView& tensor,
                                               std::span<const float> input) {
    const auto dimensions = tensor.dimensions();
    if (dimensions.size() != 2) {
        throw std::invalid_argument("Qwen3.5 block matvec requires rank-2 weights: " +
                                    std::string(tensor.name()));
    }
    const std::size_t rows = checked_size(dimensions[1], "Qwen3.5 matvec row count");
    std::vector<float> output(rows);
    backend::cpu::reference_mapped_tensor_matvec(tensor, input, output);
    return output;
}

[[nodiscard]] std::vector<float> mapped_matvec_or_override(
    std::string_view trace_name,
    const model::GgufTensorView& tensor,
    std::span<const float> input,
    const MatvecOverrideCursor& overrides) {
    const auto dimensions = tensor.dimensions();
    if (dimensions.size() != 2) {
        throw std::invalid_argument("Qwen3.5 block matvec requires rank-2 weights: " +
                                    std::string(tensor.name()));
    }
    const std::size_t expected = checked_size(dimensions[1], "Qwen3.5 matvec row count");
    if (const Qwen35TraceTensor* replacement = overrides.find(trace_name, expected)) {
        return replacement->values;
    }
    return mapped_matvec(tensor, input);
}

[[nodiscard]] float sigmoid(float value) noexcept {
    if (value >= 0.0F) {
        const float exponent = std::exp(-value);
        return 1.0F / (1.0F + exponent);
    }
    const float exponent = std::exp(value);
    return exponent / (1.0F + exponent);
}

[[nodiscard]] float softplus(float value) noexcept {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return std::exp(value);
    }
    return std::log1p(std::exp(value));
}

[[nodiscard]] float silu(float value) noexcept {
    return value * sigmoid(value);
}

[[nodiscard]] std::vector<float> rms_norm_rows(std::span<const float> input,
                                               std::span<const float> weight,
                                               std::size_t row_width,
                                               float epsilon) {
    if (row_width == 0 || input.size() % row_width != 0 || weight.size() != row_width) {
        throw std::invalid_argument("Qwen3.5 RMSNorm dimensions are incompatible");
    }
    if (!(epsilon > 0.0F) || !std::isfinite(epsilon)) {
        throw std::invalid_argument("Qwen3.5 RMSNorm epsilon must be positive and finite");
    }

    std::vector<float> output(input.size());
    const std::size_t rows = input.size() / row_width;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * row_width;
        double sum_squares = 0.0;
        for (std::size_t column = 0; column < row_width; ++column) {
            const double value = static_cast<double>(input[base + column]);
            sum_squares += value * value;
        }
        const double mean_square = sum_squares / static_cast<double>(row_width);
        const float inverse_rms =
            static_cast<float>(1.0 / std::sqrt(mean_square + static_cast<double>(epsilon)));
        for (std::size_t column = 0; column < row_width; ++column) {
            output[base + column] = input[base + column] * inverse_rms * weight[column];
        }
    }
    return output;
}

[[nodiscard]] std::vector<float> add_vectors(std::span<const float> left,
                                             std::span<const float> right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("Qwen3.5 residual vector length mismatch");
    }
    std::vector<float> output(left.size());
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = left[index] + right[index];
    }
    return output;
}

[[nodiscard]] std::vector<float> l2_normalize_rows(std::span<const float> input,
                                                   std::size_t row_width,
                                                   float epsilon) {
    if (row_width == 0 || input.size() % row_width != 0) {
        throw std::invalid_argument("Qwen3.5 L2 normalization dimensions are incompatible");
    }
    std::vector<float> output(input.size());
    const std::size_t rows = input.size() / row_width;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * row_width;
        double sum_squares = 0.0;
        for (std::size_t column = 0; column < row_width; ++column) {
            const double value = static_cast<double>(input[base + column]);
            sum_squares += value * value;
        }
        const double norm = std::sqrt(sum_squares);
        const float inverse_norm = static_cast<float>(
            1.0 / std::max(norm, static_cast<double>(epsilon)));
        for (std::size_t column = 0; column < row_width; ++column) {
            output[base + column] = input[base + column] * inverse_norm;
        }
    }
    return output;
}

[[nodiscard]] std::vector<float> repeat_heads_tiled(std::span<const float> input,
                                                    std::size_t source_heads,
                                                    std::size_t target_heads,
                                                    std::size_t head_width) {
    if (source_heads == 0 || target_heads == 0 || target_heads % source_heads != 0 ||
        input.size() != checked_multiply(source_heads, head_width, "head input size")) {
        throw std::invalid_argument("Qwen3.5 repeated-head dimensions are incompatible");
    }
    std::vector<float> output(checked_multiply(target_heads, head_width, "head output size"));
    for (std::size_t target = 0; target < target_heads; ++target) {
        const std::size_t source = target % source_heads;
        std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(source * head_width),
                    static_cast<std::ptrdiff_t>(head_width),
                    output.begin() + static_cast<std::ptrdiff_t>(target * head_width));
    }
    return output;
}

[[nodiscard]] std::vector<float> canonicalize_recurrent_state_for_trace(
    std::span<const float> state,
    std::size_t heads,
    std::size_t key_dimension,
    std::size_t value_dimension) {
    const std::size_t head_size = checked_multiply(
        key_dimension, value_dimension, "Qwen3.5 recurrent trace head size");
    if (state.size() != checked_multiply(
                            heads, head_size, "Qwen3.5 recurrent trace state size")) {
        throw std::invalid_argument("Qwen3.5 recurrent trace state dimensions are incompatible");
    }

    // Oracle stores [head][key][value] with value contiguous. GGML traces the
    // logical tensor [key, value, head], where key is the fastest dimension.
    std::vector<float> output(state.size());
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t head_base = head * head_size;
        for (std::size_t key = 0; key < key_dimension; ++key) {
            for (std::size_t value = 0; value < value_dimension; ++value) {
                const std::size_t source = head_base + key * value_dimension + value;
                const std::size_t destination = head_base + value * key_dimension + key;
                output[destination] = state[source];
            }
        }
    }
    return output;
}

[[nodiscard]] core::Tensor make_tensor(std::span<const float> values,
                                       std::vector<std::size_t> shape) {
    core::Tensor tensor(std::move(shape));
    if (tensor.element_count() != values.size()) {
        throw std::invalid_argument("Qwen3.5 tensor shape does not match value count");
    }
    std::copy(values.begin(), values.end(), tensor.data().begin());
    return tensor;
}

[[nodiscard]] std::vector<float> copy_tensor(const core::Tensor& tensor) {
    return std::vector<float>(tensor.data().begin(), tensor.data().end());
}

void capture(Qwen35BlockTrace& trace,
             bool enabled,
             std::string name,
             std::span<const float> values) {
    if (!enabled) {
        return;
    }
    trace.tensors.push_back({std::move(name), std::vector<float>(values.begin(), values.end())});
}

[[nodiscard]] std::vector<float> execute_mlp(const model::Qwen35MlpWeights& weights,
                                             std::span<const float> input,
                                             Qwen35BlockTrace& trace,
                                             bool capture_intermediates,
                                             const MatvecOverrideCursor& overrides) {
    require_tensor(weights.gate, "MLP gate");
    require_tensor(weights.up, "MLP up");
    require_tensor(weights.down, "MLP down");

    std::vector<float> gate = mapped_matvec_or_override("ffn_gate", *weights.gate, input, overrides);
    std::vector<float> up = mapped_matvec_or_override("ffn_up", *weights.up, input, overrides);
    if (gate.size() != up.size()) {
        throw std::runtime_error("Qwen3.5 MLP gate/up widths differ");
    }
    capture(trace, capture_intermediates, "ffn_gate", gate);
    capture(trace, capture_intermediates, "ffn_up", up);

    std::vector<float> activated(gate.size());
    for (std::size_t index = 0; index < activated.size(); ++index) {
        activated[index] = silu(gate[index]) * up[index];
    }
    capture(trace, capture_intermediates, "ffn_swiglu", activated);

    std::vector<float> output = mapped_matvec_or_override("ffn_out", *weights.down, activated, overrides);
    capture(trace, capture_intermediates, "ffn_out", output);
    return output;
}

[[nodiscard]] std::vector<float> finish_block(const model::Qwen35Manifest& manifest,
                                              const model::Qwen35BackboneBlockWeights& block,
                                              std::span<const float> input,
                                              std::span<const float> branch_output,
                                              Qwen35BlockTrace& trace,
                                              bool capture_intermediates,
                                              const MatvecOverrideCursor& overrides) {
    require_tensor(block.post_attention_norm, "post-attention norm");
    std::vector<float> attention_residual = add_vectors(input, branch_output);
    capture(trace, capture_intermediates, "attn_residual", attention_residual);

    const std::vector<float> post_norm_weight = decode_tensor(*block.post_attention_norm);
    std::vector<float> post_norm = rms_norm_rows(attention_residual,
                                                 post_norm_weight,
                                                 manifest.embedding_length,
                                                 manifest.attention_rms_epsilon);
    capture(trace, capture_intermediates, "attn_post_norm", post_norm);

    std::vector<float> ffn_output = execute_mlp(block.mlp,
                                                post_norm,
                                                trace,
                                                capture_intermediates,
                                                overrides);
    std::vector<float> output = add_vectors(attention_residual, ffn_output);
    capture(trace, true, "post_ffn", output);
    return output;
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

}  // namespace

const Qwen35TraceTensor* Qwen35MatvecOverrides::find(std::string_view name) const noexcept {
    const auto iterator = std::find_if(tensors.begin(), tensors.end(), [&](const auto& tensor) {
        return tensor.name == name;
    });
    return iterator == tensors.end() ? nullptr : &*iterator;
}

const Qwen35TraceTensor* Qwen35BlockTrace::find(std::string_view name) const noexcept {
    const auto iterator = std::find_if(tensors.begin(), tensors.end(), [&](const auto& tensor) {
        return tensor.name == name;
    });
    return iterator == tensors.end() ? nullptr : &*iterator;
}

Qwen35BlockResult execute_qwen35_recurrent_block_reference(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35BackboneBlockWeights& block,
    std::span<const float> input,
    SsmState& state,
    bool capture_intermediates,
    const Qwen35MatvecOverrides* matvec_overrides) {
    if (block.kind != model::Qwen35BlockKind::recurrent || !block.recurrent.has_value()) {
        throw std::invalid_argument("Qwen3.5 recurrent executor received a non-recurrent block");
    }
    if (input.size() != manifest.embedding_length) {
        throw std::invalid_argument("Qwen3.5 recurrent block input width mismatch");
    }
    require_tensor(block.input_norm, "input norm");

    static constexpr std::array<std::string_view, 8> allowed_overrides{
        "linear_attn_qkv_mixed", "z", "beta", "alpha",
        "linear_attn_out", "ffn_gate", "ffn_up", "ffn_out",
    };
    const MatvecOverrideCursor overrides(matvec_overrides, allowed_overrides);

    const Qwen35SsmLayout layout = qwen35_ssm_layout(manifest);
    if (state.convolution_channels() != layout.convolution_channels ||
        state.convolution_kernel() != layout.convolution_kernel ||
        state.value_heads() != layout.value_heads ||
        state.key_dimension() != layout.key_head_dimension ||
        state.value_dimension() != layout.value_head_dimension) {
        throw std::invalid_argument("Qwen3.5 recurrent state dimensions do not match the manifest");
    }

    Qwen35BlockTrace trace{block.index,
                           model::Qwen35BlockKind::recurrent,
                           state.sequence_length(),
                           {}};
    capture(trace, capture_intermediates, "input", input);

    const std::vector<float> input_norm_weight = decode_tensor(*block.input_norm);
    std::vector<float> normalized = rms_norm_rows(input,
                                                  input_norm_weight,
                                                  manifest.embedding_length,
                                                  manifest.attention_rms_epsilon);
    capture(trace, capture_intermediates, "attn_norm", normalized);

    const model::Qwen35RecurrentWeights& weights = *block.recurrent;
    require_tensor(weights.qkv, "recurrent QKV");
    require_tensor(weights.gate, "recurrent output gate");
    require_tensor(weights.convolution, "recurrent convolution");
    require_tensor(weights.time_step_bias, "recurrent time-step bias");
    require_tensor(weights.decay, "recurrent decay");
    require_tensor(weights.beta, "recurrent beta");
    require_tensor(weights.alpha, "recurrent alpha");
    require_tensor(weights.norm, "recurrent output norm");
    require_tensor(weights.output, "recurrent output projection");

    std::vector<float> qkv = mapped_matvec_or_override("linear_attn_qkv_mixed", *weights.qkv, normalized, overrides);
    std::vector<float> z = mapped_matvec_or_override("z", *weights.gate, normalized, overrides);
    std::vector<float> beta = mapped_matvec_or_override("beta", *weights.beta, normalized, overrides);
    std::vector<float> alpha = mapped_matvec_or_override("alpha", *weights.alpha, normalized, overrides);
    capture(trace, capture_intermediates, "linear_attn_qkv_mixed", qkv);
    capture(trace, capture_intermediates, "z", z);
    capture(trace, capture_intermediates, "beta", beta);
    capture(trace, capture_intermediates, "alpha", alpha);

    if (beta.size() != layout.value_heads || alpha.size() != layout.value_heads) {
        throw std::runtime_error("Qwen3.5 recurrent alpha/beta width mismatch");
    }
    std::vector<float> beta_sigmoid(beta.size());
    for (std::size_t index = 0; index < beta.size(); ++index) {
        beta_sigmoid[index] = sigmoid(beta[index]);
    }
    capture(trace, capture_intermediates, "beta_sigmoid", beta_sigmoid);

    const std::vector<float> time_step_bias = decode_tensor(*weights.time_step_bias);
    const std::vector<float> decay = decode_tensor(*weights.decay);
    if (time_step_bias.size() != layout.value_heads || decay.size() != layout.value_heads) {
        throw std::runtime_error("Qwen3.5 recurrent time-step/decay width mismatch");
    }
    std::vector<float> alpha_softplus(alpha.size());
    std::vector<float> log_decay(alpha.size());
    for (std::size_t index = 0; index < alpha.size(); ++index) {
        alpha_softplus[index] = softplus(alpha[index] + time_step_bias[index]);
        log_decay[index] = alpha_softplus[index] * decay[index];
    }
    capture(trace, capture_intermediates, "a_softplus", alpha_softplus);
    capture(trace, capture_intermediates, "gate", log_decay);

    const std::size_t expected_qkv = layout.convolution_channels;
    if (qkv.size() != expected_qkv) {
        throw std::runtime_error("Qwen3.5 recurrent QKV width mismatch");
    }
    const std::vector<float> convolution_weights = decode_tensor(*weights.convolution);
    std::vector<float> convolved(qkv.size());
    causal_depthwise_convolution_step(qkv,
                                      convolution_weights,
                                      state,
                                      convolved,
                                      false);
    capture(trace, capture_intermediates, "conv_output_raw", convolved);
    for (float& value : convolved) {
        value = silu(value);
    }
    capture(trace, capture_intermediates, "conv_output_silu", convolved);

    const std::size_t key_width = checked_multiply(layout.key_heads,
                                                   layout.key_head_dimension,
                                                   "Qwen3.5 recurrent key width");
    const std::size_t value_width = checked_multiply(layout.value_heads,
                                                     layout.value_head_dimension,
                                                     "Qwen3.5 recurrent value width");
    const std::size_t qk_width = checked_multiply(key_width,
                                                    2,
                                                    "Qwen3.5 recurrent QK width");
    if (checked_add(qk_width, value_width, "Qwen3.5 recurrent QKV width") !=
        convolved.size()) {
        throw std::runtime_error("Qwen3.5 recurrent convolution split width mismatch");
    }

    const auto query_narrow = std::span<const float>(convolved).first(key_width);
    const auto key_narrow = std::span<const float>(convolved).subspan(key_width, key_width);
    const auto value = std::span<const float>(convolved).subspan(qk_width, value_width);
    capture(trace, capture_intermediates, "q_conv", query_narrow);
    capture(trace, capture_intermediates, "k_conv", key_narrow);
    capture(trace, capture_intermediates, "v_conv", value);

    std::vector<float> query = repeat_heads_tiled(query_narrow,
                                            layout.key_heads,
                                            layout.value_heads,
                                            layout.key_head_dimension);
    std::vector<float> key = repeat_heads_tiled(key_narrow,
                                          layout.key_heads,
                                          layout.value_heads,
                                          layout.key_head_dimension);
    std::vector<float> query_normalized = l2_normalize_rows(query,
                                                            layout.key_head_dimension,
                                                            manifest.attention_rms_epsilon);
    std::vector<float> key_normalized = l2_normalize_rows(key,
                                                          layout.key_head_dimension,
                                                          manifest.attention_rms_epsilon);
    capture(trace, capture_intermediates, "q_conv_predelta", query_normalized);
    capture(trace, capture_intermediates, "k_conv_predelta", key_normalized);
    capture(trace, capture_intermediates, "v_conv_predelta", value);

    const std::vector<float> state_predelta_trace = canonicalize_recurrent_state_for_trace(
        state.recurrent(),
        layout.value_heads,
        layout.key_head_dimension,
        layout.value_head_dimension);
    capture(trace, capture_intermediates, "state_predelta", state_predelta_trace);

    core::Tensor query_tensor = make_tensor(query,
                                            {layout.value_heads,
                                             layout.key_head_dimension});
    core::Tensor key_tensor = make_tensor(key,
                                          {layout.value_heads,
                                           layout.key_head_dimension});
    core::Tensor value_tensor = make_tensor(value,
                                            {layout.value_heads,
                                             layout.value_head_dimension});
    core::Tensor delta_output({layout.value_heads, layout.value_head_dimension});
    gated_delta_step(query_tensor,
                     key_tensor,
                     value_tensor,
                     beta_sigmoid,
                     log_decay,
                     state,
                     delta_output,
                     manifest.attention_rms_epsilon);
    std::vector<float> recurrent_output = copy_tensor(delta_output);
    capture(trace, capture_intermediates, "recurrent_output", recurrent_output);
    const std::vector<float> state_postdelta_trace = canonicalize_recurrent_state_for_trace(
        state.recurrent(),
        layout.value_heads,
        layout.key_head_dimension,
        layout.value_head_dimension);
    capture(trace, capture_intermediates, "state_postdelta", state_postdelta_trace);

    const std::vector<float> recurrent_norm_weight = decode_tensor(*weights.norm);
    std::vector<float> gated = rms_norm_rows(recurrent_output,
                                             recurrent_norm_weight,
                                             layout.value_head_dimension,
                                             manifest.attention_rms_epsilon);
    if (gated.size() != z.size()) {
        throw std::runtime_error("Qwen3.5 recurrent gate width mismatch");
    }
    for (std::size_t index = 0; index < gated.size(); ++index) {
        gated[index] *= silu(z[index]);
    }
    capture(trace, capture_intermediates, "attn_out_norm", gated);

    std::vector<float> branch_output = mapped_matvec_or_override("linear_attn_out", *weights.output, gated, overrides);
    capture(trace, capture_intermediates, "linear_attn_out", branch_output);
    std::vector<float> output = finish_block(manifest,
                                             block,
                                             input,
                                             branch_output,
                                             trace,
                                             capture_intermediates,
                                             overrides);
    return {std::move(output), std::move(trace)};
}

Qwen35BlockResult execute_qwen35_attention_block_reference(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35BackboneBlockWeights& block,
    std::span<const float> input,
    KvCache& cache,
    RopePosition position,
    bool capture_intermediates,
    const Qwen35MatvecOverrides* matvec_overrides) {
    if (block.kind != model::Qwen35BlockKind::full_attention || !block.attention.has_value()) {
        throw std::invalid_argument("Qwen3.5 attention executor received a non-attention block");
    }
    if (input.size() != manifest.embedding_length) {
        throw std::invalid_argument("Qwen3.5 attention block input width mismatch");
    }
    if (cache.key_value_heads() != manifest.attention_head_count_kv ||
        cache.key_dimension() != manifest.attention_key_length ||
        cache.value_dimension() != manifest.attention_value_length) {
        throw std::invalid_argument("Qwen3.5 attention cache dimensions do not match the manifest");
    }
    require_tensor(block.input_norm, "input norm");

    static constexpr std::array<std::string_view, 7> allowed_overrides{
        "Qcur_full", "Kcur_projected", "Vcur_projected", "attn_output",
        "ffn_gate", "ffn_up", "ffn_out",
    };
    const MatvecOverrideCursor overrides(matvec_overrides, allowed_overrides);

    Qwen35BlockTrace trace{block.index,
                           model::Qwen35BlockKind::full_attention,
                           position.temporal,
                           {}};
    capture(trace, capture_intermediates, "input", input);

    const std::vector<float> input_norm_weight = decode_tensor(*block.input_norm);
    std::vector<float> normalized = rms_norm_rows(input,
                                                  input_norm_weight,
                                                  manifest.embedding_length,
                                                  manifest.attention_rms_epsilon);
    capture(trace, capture_intermediates, "attn_norm", normalized);

    const model::Qwen35AttentionWeights& weights = *block.attention;
    require_tensor(weights.query, "attention query");
    require_tensor(weights.key, "attention key");
    require_tensor(weights.value, "attention value");
    require_tensor(weights.output, "attention output");
    require_tensor(weights.query_norm, "attention query norm");
    require_tensor(weights.key_norm, "attention key norm");

    std::vector<float> query_full = mapped_matvec_or_override("Qcur_full", *weights.query, normalized, overrides);
    std::vector<float> key = mapped_matvec_or_override("Kcur_projected", *weights.key, normalized, overrides);
    std::vector<float> value = mapped_matvec_or_override("Vcur_projected", *weights.value, normalized, overrides);
    capture(trace, capture_intermediates, "Qcur_full", query_full);
    capture(trace, capture_intermediates, "Kcur_projected", key);
    capture(trace, capture_intermediates, "Vcur_projected", value);

    const std::size_t query_head_width = manifest.attention_key_length;
    const std::size_t query_heads = manifest.attention_head_count;
    const std::size_t expected_query_full = checked_multiply(
        checked_multiply(query_heads, query_head_width, "Qwen3.5 query width"),
        2,
        "Qwen3.5 joint query/gate width");
    if (query_full.size() != expected_query_full) {
        throw std::runtime_error("Qwen3.5 joint query/gate projection width mismatch");
    }

    std::vector<float> query(checked_multiply(query_heads,
                                              query_head_width,
                                              "Qwen3.5 query values"));
    std::vector<float> gate(query.size());
    for (std::size_t head = 0; head < query_heads; ++head) {
        const std::size_t source = head * query_head_width * 2;
        const std::size_t destination = head * query_head_width;
        std::copy_n(query_full.begin() + static_cast<std::ptrdiff_t>(source),
                    static_cast<std::ptrdiff_t>(query_head_width),
                    query.begin() + static_cast<std::ptrdiff_t>(destination));
        std::copy_n(query_full.begin() + static_cast<std::ptrdiff_t>(source + query_head_width),
                    static_cast<std::ptrdiff_t>(query_head_width),
                    gate.begin() + static_cast<std::ptrdiff_t>(destination));
    }
    capture(trace, capture_intermediates, "Qcur_reshaped", query);
    capture(trace, capture_intermediates, "gate_reshaped", gate);

    const std::vector<float> query_norm_weight = decode_tensor(*weights.query_norm);
    const std::vector<float> key_norm_weight = decode_tensor(*weights.key_norm);
    query = rms_norm_rows(query,
                          query_norm_weight,
                          query_head_width,
                          manifest.attention_rms_epsilon);
    key = rms_norm_rows(key,
                        key_norm_weight,
                        manifest.attention_key_length,
                        manifest.attention_rms_epsilon);
    capture(trace, capture_intermediates, "Qcur_normed", query);
    capture(trace, capture_intermediates, "Kcur_normed", key);

    core::Tensor query_tensor = make_tensor(query,
                                            {manifest.attention_head_count,
                                             manifest.attention_key_length});
    core::Tensor key_tensor = make_tensor(key,
                                          {manifest.attention_head_count_kv,
                                           manifest.attention_key_length});
    core::Tensor value_tensor = make_tensor(value,
                                            {manifest.attention_head_count_kv,
                                             manifest.attention_value_length});
    apply_qwen35_rope(query_tensor,
                      position,
                      manifest.rope_dimension_count,
                      manifest.rope_dimension_sections,
                      manifest.rope_frequency_base);
    apply_qwen35_rope(key_tensor,
                      position,
                      manifest.rope_dimension_count,
                      manifest.rope_dimension_sections,
                      manifest.rope_frequency_base);
    capture(trace, capture_intermediates, "Qcur", query_tensor.data());
    capture(trace, capture_intermediates, "Kcur", key_tensor.data());
    capture(trace, capture_intermediates, "Vcur", value_tensor.data());

    core::Tensor attention_output({manifest.attention_head_count,
                                   manifest.attention_value_length});
    causal_attention_step(query_tensor,
                          key_tensor,
                          value_tensor,
                          cache,
                          attention_output);
    std::vector<float> attention = copy_tensor(attention_output);
    capture(trace, capture_intermediates, "attn_pregate", attention);

    if (attention.size() != gate.size()) {
        throw std::runtime_error("Qwen3.5 attention gate width mismatch");
    }
    std::vector<float> gate_sigmoid(gate.size());
    for (std::size_t index = 0; index < gate.size(); ++index) {
        gate_sigmoid[index] = sigmoid(gate[index]);
        attention[index] *= gate_sigmoid[index];
    }
    capture(trace, capture_intermediates, "gate_sigmoid", gate_sigmoid);
    capture(trace, capture_intermediates, "attn_gated", attention);

    std::vector<float> branch_output = mapped_matvec_or_override("attn_output", *weights.output, attention, overrides);
    capture(trace, capture_intermediates, "attn_output", branch_output);
    std::vector<float> output = finish_block(manifest,
                                             block,
                                             input,
                                             branch_output,
                                             trace,
                                             capture_intermediates,
                                             overrides);
    return {std::move(output), std::move(trace)};
}

void require_finite_trace(const Qwen35BlockTrace& trace) {
    for (const Qwen35TraceTensor& tensor : trace.tensors) {
        for (std::size_t index = 0; index < tensor.values.size(); ++index) {
            if (!std::isfinite(tensor.values[index])) {
                throw std::runtime_error("Qwen3.5 block trace contains a non-finite value in " +
                                         tensor.name + " at index " + std::to_string(index));
            }
        }
    }
}

std::string qwen35_block_trace_text(const Qwen35BlockTrace& trace) {
    require_finite_trace(trace);
    std::ostringstream output;
    output << "Qwen3.5 block " << trace.block_index << " ("
           << model::qwen35_block_kind_name(trace.kind) << ")\n"
           << "position: " << trace.position << '\n'
           << "captured tensors: " << trace.tensors.size() << '\n';

    output << std::setprecision(9);
    for (const Qwen35TraceTensor& tensor : trace.tensors) {
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        double sum_squares = 0.0;
        for (const float value : tensor.values) {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            sum_squares += static_cast<double>(value) * static_cast<double>(value);
        }
        const double rms = tensor.values.empty()
                               ? 0.0
                               : std::sqrt(sum_squares / static_cast<double>(tensor.values.size()));
        output << "- " << tensor.name
               << ": count=" << tensor.values.size()
               << " min=" << (tensor.values.empty() ? 0.0F : minimum)
               << " max=" << (tensor.values.empty() ? 0.0F : maximum)
               << " rms=" << rms
               << " fnv1a64=0x" << std::hex << fingerprint(tensor.values) << std::dec
               << " preview=[";
        const std::size_t preview = std::min<std::size_t>(tensor.values.size(), 8);
        for (std::size_t index = 0; index < preview; ++index) {
            if (index != 0) output << ',';
            output << tensor.values[index];
        }
        if (preview < tensor.values.size()) output << ",...";
        output << "]\n";
    }
    return output.str();
}

std::string qwen35_block_trace_json(const Qwen35BlockTrace& trace) {
    require_finite_trace(trace);
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << '{'
           << "\"block_index\":" << trace.block_index << ','
           << "\"kind\":\"" << model::qwen35_block_kind_name(trace.kind) << "\","
           << "\"position\":" << trace.position << ','
           << "\"tensors\":[";
    for (std::size_t tensor_index = 0; tensor_index < trace.tensors.size(); ++tensor_index) {
        if (tensor_index != 0) output << ',';
        const Qwen35TraceTensor& tensor = trace.tensors[tensor_index];
        output << '{'
               << "\"name\":\"" << json_escape(tensor.name) << "\","
               << "\"count\":" << tensor.values.size() << ','
               << "\"fnv1a64\":\"0x" << std::hex << fingerprint(tensor.values) << std::dec
               << "\",\"values\":[";
        for (std::size_t value_index = 0; value_index < tensor.values.size(); ++value_index) {
            if (value_index != 0) output << ',';
            output << tensor.values[value_index];
        }
        output << "]}";
    }
    output << "]}";
    return output.str();
}

}  // namespace oracle::runtime
