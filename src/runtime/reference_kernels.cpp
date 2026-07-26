#include "oracle/runtime/reference_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace oracle::runtime {
namespace {

void require_f32_contiguous(const core::Tensor& tensor, const char* operation) {
    if (tensor.dtype() != core::DataType::f32 || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(operation) + " requires contiguous f32 tensors");
    }
}

[[nodiscard]] float sigmoid(float value) noexcept {
    if (value >= 0.0F) {
        const float exponent = std::exp(-value);
        return 1.0F / (1.0F + exponent);
    }
    const float exponent = std::exp(value);
    return exponent / (1.0F + exponent);
}

[[nodiscard]] std::uint64_t position_for_frequency(std::size_t frequency_index,
                                                   RopePosition position,
                                                   std::array<std::int32_t, 4> sections) {
    const std::size_t height_count = static_cast<std::size_t>(sections[1]);
    const std::size_t width_count = static_cast<std::size_t>(sections[2]);
    if (frequency_index % 3U == 1U && frequency_index / 3U < height_count) {
        return position.height;
    }
    if (frequency_index % 3U == 2U && frequency_index / 3U < width_count) {
        return position.width;
    }
    return position.temporal;
}

}  // namespace

void embedding_lookup(const core::Tensor& embedding,
                      std::span<const std::uint32_t> token_ids,
                      core::Tensor& output) {
    require_f32_contiguous(embedding, "embedding_lookup");
    require_f32_contiguous(output, "embedding_lookup");
    if (embedding.rank() != 2 || output.rank() != 2) {
        throw std::invalid_argument("embedding_lookup expects rank-2 tensors");
    }
    if (output.shape()[0] != token_ids.size() ||
        output.shape()[1] != embedding.shape()[1]) {
        throw std::invalid_argument("embedding_lookup output dimensions are incompatible");
    }

    const std::size_t vocabulary = embedding.shape()[0];
    const std::size_t width = embedding.shape()[1];
    const auto table = embedding.data();
    auto result = output.data();
    for (std::size_t row = 0; row < token_ids.size(); ++row) {
        const std::uint32_t token = token_ids[row];
        if (token >= vocabulary) {
            throw std::out_of_range("embedding token id is outside the vocabulary");
        }
        const std::size_t source = static_cast<std::size_t>(token) * width;
        const std::size_t destination = row * width;
        std::copy_n(table.begin() + static_cast<std::ptrdiff_t>(source),
                    static_cast<std::ptrdiff_t>(width),
                    result.begin() + static_cast<std::ptrdiff_t>(destination));
    }
}

void apply_qwen35_rope(core::Tensor& heads,
                       RopePosition position,
                       std::size_t rotary_dimension,
                       std::array<std::int32_t, 4> dimension_sections,
                       float frequency_base) {
    require_f32_contiguous(heads, "apply_qwen35_rope");
    if (heads.rank() != 2) {
        throw std::invalid_argument("apply_qwen35_rope expects [heads, head_dimension]");
    }
    if (rotary_dimension == 0 || rotary_dimension % 2 != 0 ||
        rotary_dimension > heads.shape()[1]) {
        throw std::invalid_argument("RoPE dimension must be non-zero, even, and within head width");
    }
    if (!(frequency_base > 0.0F) || !std::isfinite(frequency_base)) {
        throw std::invalid_argument("RoPE frequency base must be positive and finite");
    }

    std::int64_t section_sum = 0;
    for (const std::int32_t section : dimension_sections) {
        if (section < 0) {
            throw std::invalid_argument("RoPE dimension sections must be non-negative");
        }
        section_sum += section;
    }
    if (section_sum != static_cast<std::int64_t>(rotary_dimension / 2)) {
        throw std::invalid_argument("RoPE dimension sections must sum to half the rotary dimension");
    }

    auto values = heads.data();
    const std::size_t head_dimension = heads.shape()[1];
    const std::size_t half = rotary_dimension / 2;
    for (std::size_t head = 0; head < heads.shape()[0]; ++head) {
        const std::size_t base = head * head_dimension;
        for (std::size_t frequency = 0; frequency < half; ++frequency) {
            const double exponent =
                (2.0 * static_cast<double>(frequency)) /
                static_cast<double>(rotary_dimension);
            const double inverse_frequency = 1.0 / std::pow(frequency_base, exponent);
            const std::uint64_t coordinate =
                position_for_frequency(frequency, position, dimension_sections);
            const double angle = static_cast<double>(coordinate) * inverse_frequency;
            const float cosine = static_cast<float>(std::cos(angle));
            const float sine = static_cast<float>(std::sin(angle));
            const std::size_t first = base + frequency;
            const std::size_t second = base + half + frequency;
            const float first_value = values[first];
            const float second_value = values[second];
            values[first] = first_value * cosine - second_value * sine;
            values[second] = second_value * cosine + first_value * sine;
        }
    }
}

void causal_attention_step(const core::Tensor& query,
                           const core::Tensor& key,
                           const core::Tensor& value,
                           KvCache& cache,
                           core::Tensor& output) {
    require_f32_contiguous(query, "causal_attention_step");
    require_f32_contiguous(key, "causal_attention_step");
    require_f32_contiguous(value, "causal_attention_step");
    require_f32_contiguous(output, "causal_attention_step");
    if (query.rank() != 2 || key.rank() != 2 || value.rank() != 2 || output.rank() != 2) {
        throw std::invalid_argument("causal_attention_step expects rank-2 head tensors");
    }

    const std::size_t query_heads = query.shape()[0];
    const std::size_t key_value_heads = key.shape()[0];
    if (key_value_heads == 0 || query_heads % key_value_heads != 0 ||
        value.shape()[0] != key_value_heads ||
        query.shape()[1] != cache.key_dimension() ||
        key.shape()[1] != cache.key_dimension() ||
        value.shape()[1] != cache.value_dimension() ||
        cache.key_value_heads() != key_value_heads ||
        output.shape()[0] != query_heads ||
        output.shape()[1] != cache.value_dimension()) {
        throw std::invalid_argument("causal_attention_step dimensions are incompatible");
    }

    cache.append(key.data(), value.data());
    const std::size_t repetitions = query_heads / key_value_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(cache.key_dimension()));
    const auto queries = query.data();
    auto result = output.data();
    std::ranges::fill(result, 0.0F);
    std::vector<float> scores(cache.size());

    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
        const std::size_t kv_head = query_head / repetitions;
        const std::size_t query_offset = query_head * cache.key_dimension();
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < cache.size(); ++token) {
            const auto cached_key = cache.key(token, kv_head);
            double dot = 0.0;
            for (std::size_t dimension = 0; dimension < cache.key_dimension(); ++dimension) {
                dot += static_cast<double>(queries[query_offset + dimension]) *
                       static_cast<double>(cached_key[dimension]);
            }
            scores[token] = static_cast<float>(dot) * scale;
            maximum = std::max(maximum, scores[token]);
        }

        double denominator = 0.0;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += static_cast<double>(score);
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator)) {
            throw std::runtime_error("causal attention softmax normalization failed");
        }

        const std::size_t output_offset = query_head * cache.value_dimension();
        for (std::size_t token = 0; token < cache.size(); ++token) {
            const float probability =
                static_cast<float>(static_cast<double>(scores[token]) / denominator);
            const auto cached_value = cache.value(token, kv_head);
            for (std::size_t dimension = 0; dimension < cache.value_dimension(); ++dimension) {
                result[output_offset + dimension] += probability * cached_value[dimension];
            }
        }
    }
}

void causal_depthwise_convolution_step(std::span<const float> input,
                                       std::span<const float> weights,
                                       SsmState& state,
                                       std::span<float> output,
                                       bool apply_silu) {
    const std::size_t channels = state.convolution_channels();
    const std::size_t kernel = state.convolution_kernel();
    if (input.size() != channels || output.size() != channels ||
        weights.size() != channels * kernel) {
        throw std::invalid_argument("depthwise convolution dimensions do not match SSM state");
    }

    auto history = state.convolution_history();
    for (std::size_t channel = 0; channel < channels; ++channel) {
        const std::size_t base = channel * kernel;
        for (std::size_t index = 1; index < kernel; ++index) {
            history[base + index - 1] = history[base + index];
        }
        history[base + kernel - 1] = input[channel];

        double sum = 0.0;
        for (std::size_t index = 0; index < kernel; ++index) {
            sum += static_cast<double>(history[base + index]) *
                   static_cast<double>(weights[base + index]);
        }
        const float value = static_cast<float>(sum);
        output[channel] = apply_silu ? value * sigmoid(value) : value;
    }
}

void gated_delta_step(const core::Tensor& query,
                      const core::Tensor& key,
                      const core::Tensor& value,
                      std::span<const float> beta,
                      std::span<const float> log_decay,
                      SsmState& state,
                      core::Tensor& output,
                      float normalization_epsilon) {
    require_f32_contiguous(query, "gated_delta_step");
    require_f32_contiguous(key, "gated_delta_step");
    require_f32_contiguous(value, "gated_delta_step");
    require_f32_contiguous(output, "gated_delta_step");
    if (query.rank() != 2 || key.rank() != 2 || value.rank() != 2 || output.rank() != 2) {
        throw std::invalid_argument("gated_delta_step expects rank-2 head tensors");
    }
    if (!(normalization_epsilon > 0.0F)) {
        throw std::invalid_argument("gated delta normalization epsilon must be positive");
    }

    const std::size_t heads = state.value_heads();
    const std::size_t key_dimension = state.key_dimension();
    const std::size_t value_dimension = state.value_dimension();
    if (query.shape() != std::vector<std::size_t>{heads, key_dimension} ||
        key.shape() != std::vector<std::size_t>{heads, key_dimension} ||
        value.shape() != std::vector<std::size_t>{heads, value_dimension} ||
        output.shape() != std::vector<std::size_t>{heads, value_dimension} ||
        beta.size() != heads || log_decay.size() != heads) {
        throw std::invalid_argument("gated_delta_step dimensions do not match SSM state");
    }

    const auto q_values = query.data();
    const auto k_values = key.data();
    const auto v_values = value.data();
    auto result = output.data();
    auto recurrent = state.recurrent();
    const float query_scale = 1.0F / std::sqrt(static_cast<float>(key_dimension));

    std::vector<float> normalized_query(key_dimension);
    std::vector<float> normalized_key(key_dimension);
    std::vector<float> memory(value_dimension);
    std::vector<float> delta(value_dimension);

    for (std::size_t head = 0; head < heads; ++head) {
        if (!(beta[head] >= 0.0F && beta[head] <= 1.0F) ||
            !std::isfinite(beta[head]) || !std::isfinite(log_decay[head])) {
            throw std::invalid_argument("gated delta beta/decay values are invalid");
        }
        const std::size_t qk_base = head * key_dimension;
        const std::size_t value_base = head * value_dimension;
        double q_norm = 0.0;
        double k_norm = 0.0;
        for (std::size_t dimension = 0; dimension < key_dimension; ++dimension) {
            const double q = q_values[qk_base + dimension];
            const double k = k_values[qk_base + dimension];
            q_norm += q * q;
            k_norm += k * k;
        }
        const double q_length = std::sqrt(q_norm);
        const double k_length = std::sqrt(k_norm);
        const float inverse_q = static_cast<float>(
            1.0 / std::max(q_length, static_cast<double>(normalization_epsilon)));
        const float inverse_k = static_cast<float>(
            1.0 / std::max(k_length, static_cast<double>(normalization_epsilon)));
        for (std::size_t dimension = 0; dimension < key_dimension; ++dimension) {
            normalized_query[dimension] =
                q_values[qk_base + dimension] * inverse_q * query_scale;
            normalized_key[dimension] = k_values[qk_base + dimension] * inverse_k;
        }

        const float decay = std::exp(log_decay[head]);
        const std::size_t state_head_base = head * key_dimension * value_dimension;
        for (std::size_t index = 0; index < key_dimension * value_dimension; ++index) {
            recurrent[state_head_base + index] *= decay;
        }

        std::ranges::fill(memory, 0.0F);
        for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
            const std::size_t row =
                state_head_base + key_index * value_dimension;
            for (std::size_t value_index = 0; value_index < value_dimension; ++value_index) {
                memory[value_index] += recurrent[row + value_index] *
                                       normalized_key[key_index];
            }
        }
        for (std::size_t value_index = 0; value_index < value_dimension; ++value_index) {
            delta[value_index] =
                (v_values[value_base + value_index] - memory[value_index]) * beta[head];
        }
        for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
            const std::size_t row =
                state_head_base + key_index * value_dimension;
            for (std::size_t value_index = 0; value_index < value_dimension; ++value_index) {
                recurrent[row + value_index] +=
                    normalized_key[key_index] * delta[value_index];
            }
        }
        for (std::size_t value_index = 0; value_index < value_dimension; ++value_index) {
            double sum = 0.0;
            for (std::size_t key_index = 0; key_index < key_dimension; ++key_index) {
                const std::size_t row =
                    state_head_base + key_index * value_dimension;
                sum += static_cast<double>(recurrent[row + value_index]) *
                       static_cast<double>(normalized_query[key_index]);
            }
            result[value_base + value_index] = static_cast<float>(sum);
        }
    }
    state.advance();
}

}  // namespace oracle::runtime
