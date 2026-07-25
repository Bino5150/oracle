#include "oracle/runtime/tiny_hybrid_model.hpp"

#include "oracle/runtime/reference_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace oracle::runtime {
namespace {

[[nodiscard]] std::size_t checked_multiply(std::size_t left,
                                           std::size_t right,
                                           const char* context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left,
                                      std::size_t right,
                                      const char* context) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left + right;
}

[[nodiscard]] std::size_t ssm_key_width(const TinyHybridConfig& config) {
    return checked_multiply(config.ssm_key_heads,
                            config.ssm_key_dimension,
                            "tiny SSM key width");
}

[[nodiscard]] std::size_t ssm_value_width(const TinyHybridConfig& config) {
    return checked_multiply(config.ssm_value_heads,
                            config.ssm_value_dimension,
                            "tiny SSM value width");
}

[[nodiscard]] std::size_t ssm_convolution_channels(const TinyHybridConfig& config) {
    return checked_add(checked_multiply(2, ssm_key_width(config), "tiny SSM QK width"),
                       ssm_value_width(config),
                       "tiny SSM convolution width");
}

[[nodiscard]] std::size_t attention_query_width(const TinyHybridConfig& config) {
    return checked_multiply(config.attention_heads,
                            config.attention_head_dimension,
                            "tiny attention query width");
}

[[nodiscard]] std::size_t attention_key_value_width(const TinyHybridConfig& config) {
    return checked_multiply(config.attention_key_value_heads,
                            config.attention_head_dimension,
                            "tiny attention KV width");
}

void dense(std::span<const float> input,
           const core::Tensor& weights,
           std::span<float> output) {
    if (weights.rank() != 2 || weights.shape()[0] != input.size() ||
        weights.shape()[1] != output.size()) {
        throw std::invalid_argument("tiny model dense projection dimensions are incompatible");
    }
    const auto matrix = weights.data();
    std::ranges::fill(output, 0.0F);
    for (std::size_t row = 0; row < input.size(); ++row) {
        const std::size_t base = row * output.size();
        for (std::size_t column = 0; column < output.size(); ++column) {
            output[column] += input[row] * matrix[base + column];
        }
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

[[nodiscard]] float softplus(float value) noexcept {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return std::exp(value);
    }
    return std::log1p(std::exp(value));
}

void fill_random(core::Tensor& tensor,
                 std::mt19937_64& random,
                 float scale) {
    std::uniform_real_distribution<float> distribution(-scale, scale);
    for (float& value : tensor.data()) {
        value = distribution(random);
    }
}

void add_in_place(std::span<float> destination, std::span<const float> source) {
    if (destination.size() != source.size()) {
        throw std::invalid_argument("tiny model residual dimensions are incompatible");
    }
    for (std::size_t index = 0; index < destination.size(); ++index) {
        destination[index] += source[index];
    }
}

}  // namespace

TinyHybridState::TinyHybridState(const TinyHybridConfig& config,
                                 std::size_t maximum_tokens)
    : ssm_(ssm_convolution_channels(config),
           config.ssm_convolution_kernel,
           config.ssm_value_heads,
           config.ssm_key_dimension,
           config.ssm_value_dimension),
      attention_(maximum_tokens,
                 config.attention_key_value_heads,
                 config.attention_head_dimension,
                 config.attention_head_dimension) {
    if (maximum_tokens == 0) {
        throw std::invalid_argument("tiny hybrid state capacity must be non-zero");
    }
}

void TinyHybridState::reset() noexcept {
    position_ = 0;
    ssm_.reset();
    attention_.reset();
}

std::size_t TinyHybridState::sequence_length() const noexcept { return position_; }
std::size_t TinyHybridState::capacity() const noexcept { return attention_.capacity(); }
std::size_t TinyHybridState::byte_size() const noexcept {
    return ssm_.byte_size() + attention_.byte_size();
}

TinyHybridConfig TinyHybridModel::validate_config(TinyHybridConfig config) {
    const auto require_positive = [](std::size_t value, const char* field) {
        if (value == 0) {
            throw std::invalid_argument(std::string(field) + " must be non-zero");
        }
    };
    require_positive(config.vocabulary_size, "tiny vocabulary size");
    require_positive(config.hidden_size, "tiny hidden size");
    require_positive(config.ssm_key_heads, "tiny SSM key heads");
    require_positive(config.ssm_value_heads, "tiny SSM value heads");
    require_positive(config.ssm_key_dimension, "tiny SSM key dimension");
    require_positive(config.ssm_value_dimension, "tiny SSM value dimension");
    require_positive(config.ssm_convolution_kernel, "tiny SSM convolution kernel");
    require_positive(config.attention_heads, "tiny attention heads");
    require_positive(config.attention_key_value_heads, "tiny attention KV heads");
    require_positive(config.attention_head_dimension, "tiny attention head dimension");
    require_positive(config.rotary_dimension, "tiny rotary dimension");
    if (config.ssm_value_heads % config.ssm_key_heads != 0) {
        throw std::invalid_argument("tiny SSM value heads must be divisible by key heads");
    }
    if (config.attention_heads % config.attention_key_value_heads != 0) {
        throw std::invalid_argument("tiny attention heads must be divisible by KV heads");
    }
    if (config.rotary_dimension % 2 != 0 ||
        config.rotary_dimension > config.attention_head_dimension) {
        throw std::invalid_argument("tiny rotary dimension is invalid");
    }
    if (!(config.rope_frequency_base > 0.0F) ||
        !(config.rms_epsilon > 0.0F)) {
        throw std::invalid_argument("tiny RoPE base and RMS epsilon must be positive");
    }
    return config;
}

TinyHybridModel::TinyHybridModel(TinyHybridConfig config, std::uint64_t seed)
    : config_(validate_config(config)),
      backend_(backend::make_cpu_backend()),
      embedding_({config_.vocabulary_size, config_.hidden_size}),
      norm_({config_.hidden_size}),
      ssm_q_projection_({config_.hidden_size, ssm_key_width(config_)}),
      ssm_k_projection_({config_.hidden_size, ssm_key_width(config_)}),
      ssm_v_projection_({config_.hidden_size, ssm_value_width(config_)}),
      ssm_z_projection_({config_.hidden_size, ssm_value_width(config_)}),
      ssm_beta_projection_({config_.hidden_size, config_.ssm_value_heads}),
      ssm_decay_projection_({config_.hidden_size, config_.ssm_value_heads}),
      ssm_convolution_weights_({ssm_convolution_channels(config_),
                                config_.ssm_convolution_kernel}),
      ssm_output_projection_({ssm_value_width(config_), config_.hidden_size}),
      attention_q_projection_({config_.hidden_size, attention_query_width(config_)}),
      attention_k_projection_({config_.hidden_size,
                               attention_key_value_width(config_)}),
      attention_v_projection_({config_.hidden_size,
                               attention_key_value_width(config_)}),
      attention_gate_projection_({config_.hidden_size,
                                  attention_query_width(config_)}),
      attention_output_projection_({attention_query_width(config_),
                                    config_.hidden_size}) {
    initialize_weights(seed);
}

void TinyHybridModel::initialize_weights(std::uint64_t seed) {
    std::mt19937_64 random(seed);
    fill_random(embedding_, random, 0.35F);
    norm_.fill(1.0F);
    fill_random(ssm_q_projection_, random, 0.22F);
    fill_random(ssm_k_projection_, random, 0.22F);
    fill_random(ssm_v_projection_, random, 0.22F);
    fill_random(ssm_z_projection_, random, 0.22F);
    fill_random(ssm_beta_projection_, random, 0.18F);
    fill_random(ssm_decay_projection_, random, 0.18F);
    fill_random(ssm_convolution_weights_, random, 0.25F);
    fill_random(ssm_output_projection_, random, 0.22F);
    fill_random(attention_q_projection_, random, 0.22F);
    fill_random(attention_k_projection_, random, 0.22F);
    fill_random(attention_v_projection_, random, 0.22F);
    fill_random(attention_gate_projection_, random, 0.18F);
    fill_random(attention_output_projection_, random, 0.22F);
}

const TinyHybridConfig& TinyHybridModel::config() const noexcept { return config_; }

core::Tensor TinyHybridModel::forward_token(std::uint32_t token_id,
                                            TinyHybridState& state) {
    if (token_id >= config_.vocabulary_size) {
        throw std::out_of_range("tiny model token id is outside the vocabulary");
    }
    if (state.position_ >= state.capacity()) {
        throw std::overflow_error("tiny model state capacity exceeded");
    }
    if (state.ssm_.convolution_channels() != ssm_convolution_channels(config_) ||
        state.ssm_.convolution_kernel() != config_.ssm_convolution_kernel ||
        state.ssm_.value_heads() != config_.ssm_value_heads ||
        state.ssm_.key_dimension() != config_.ssm_key_dimension ||
        state.ssm_.value_dimension() != config_.ssm_value_dimension ||
        state.attention_.key_value_heads() != config_.attention_key_value_heads ||
        state.attention_.key_dimension() != config_.attention_head_dimension ||
        state.attention_.value_dimension() != config_.attention_head_dimension) {
        throw std::invalid_argument("tiny model state was created for a different configuration");
    }

    core::Tensor hidden({1, config_.hidden_size});
    const std::array<std::uint32_t, 1> token{token_id};
    embedding_lookup(embedding_, token, hidden);

    core::Tensor normalized({1, config_.hidden_size});
    backend_->rms_norm(hidden, norm_, config_.rms_epsilon, normalized);

    core::Tensor ssm_q_flat({1, ssm_key_width(config_)});
    core::Tensor ssm_k_flat({1, ssm_key_width(config_)});
    core::Tensor ssm_v_flat({1, ssm_value_width(config_)});
    core::Tensor ssm_z_flat({1, ssm_value_width(config_)});
    core::Tensor beta_raw({1, config_.ssm_value_heads});
    core::Tensor decay_raw({1, config_.ssm_value_heads});
    dense(normalized.data(), ssm_q_projection_, ssm_q_flat.data());
    dense(normalized.data(), ssm_k_projection_, ssm_k_flat.data());
    dense(normalized.data(), ssm_v_projection_, ssm_v_flat.data());
    dense(normalized.data(), ssm_z_projection_, ssm_z_flat.data());
    dense(normalized.data(), ssm_beta_projection_, beta_raw.data());
    dense(normalized.data(), ssm_decay_projection_, decay_raw.data());

    std::vector<float> convolution_input(ssm_convolution_channels(config_));
    std::copy(ssm_q_flat.data().begin(),
              ssm_q_flat.data().end(),
              convolution_input.begin());
    std::copy(ssm_k_flat.data().begin(),
              ssm_k_flat.data().end(),
              convolution_input.begin() + static_cast<std::ptrdiff_t>(ssm_key_width(config_)));
    std::copy(ssm_v_flat.data().begin(),
              ssm_v_flat.data().end(),
              convolution_input.begin() +
                  static_cast<std::ptrdiff_t>(2 * ssm_key_width(config_)));
    std::vector<float> convolution_output(convolution_input.size());
    causal_depthwise_convolution_step(convolution_input,
                                      ssm_convolution_weights_.data(),
                                      state.ssm_,
                                      convolution_output,
                                      true);

    const std::size_t head_repetitions =
        config_.ssm_value_heads / config_.ssm_key_heads;
    core::Tensor ssm_query({config_.ssm_value_heads, config_.ssm_key_dimension});
    core::Tensor ssm_key({config_.ssm_value_heads, config_.ssm_key_dimension});
    core::Tensor ssm_value({config_.ssm_value_heads, config_.ssm_value_dimension});
    for (std::size_t value_head = 0; value_head < config_.ssm_value_heads; ++value_head) {
        const std::size_t key_head = value_head / head_repetitions;
        for (std::size_t dimension = 0; dimension < config_.ssm_key_dimension; ++dimension) {
            const std::size_t source = key_head * config_.ssm_key_dimension + dimension;
            ssm_query.data()[value_head * config_.ssm_key_dimension + dimension] =
                convolution_output[source];
            ssm_key.data()[value_head * config_.ssm_key_dimension + dimension] =
                convolution_output[ssm_key_width(config_) + source];
        }
        for (std::size_t dimension = 0; dimension < config_.ssm_value_dimension; ++dimension) {
            ssm_value.data()[value_head * config_.ssm_value_dimension + dimension] =
                convolution_output[2 * ssm_key_width(config_) +
                                   value_head * config_.ssm_value_dimension + dimension];
        }
    }

    std::vector<float> beta(config_.ssm_value_heads);
    std::vector<float> log_decay(config_.ssm_value_heads);
    for (std::size_t head = 0; head < config_.ssm_value_heads; ++head) {
        beta[head] = sigmoid(beta_raw.data()[head]);
        log_decay[head] = -softplus(decay_raw.data()[head]);
    }
    core::Tensor ssm_mixed({config_.ssm_value_heads, config_.ssm_value_dimension});
    gated_delta_step(ssm_query,
                     ssm_key,
                     ssm_value,
                     beta,
                     log_decay,
                     state.ssm_,
                     ssm_mixed,
                     config_.rms_epsilon);
    for (std::size_t index = 0; index < ssm_mixed.element_count(); ++index) {
        const float gate = ssm_z_flat.data()[index];
        ssm_mixed.data()[index] *= gate * sigmoid(gate);
    }
    core::Tensor ssm_output({1, config_.hidden_size});
    dense(ssm_mixed.data(), ssm_output_projection_, ssm_output.data());
    add_in_place(hidden.data(), ssm_output.data());

    backend_->rms_norm(hidden, norm_, config_.rms_epsilon, normalized);
    core::Tensor query_flat({1, attention_query_width(config_)});
    core::Tensor key_flat({1, attention_key_value_width(config_)});
    core::Tensor value_flat({1, attention_key_value_width(config_)});
    core::Tensor attention_gate({1, attention_query_width(config_)});
    dense(normalized.data(), attention_q_projection_, query_flat.data());
    dense(normalized.data(), attention_k_projection_, key_flat.data());
    dense(normalized.data(), attention_v_projection_, value_flat.data());
    dense(normalized.data(), attention_gate_projection_, attention_gate.data());

    core::Tensor query = query_flat.reshape(
        {config_.attention_heads, config_.attention_head_dimension});
    core::Tensor key = key_flat.reshape(
        {config_.attention_key_value_heads, config_.attention_head_dimension});
    core::Tensor value = value_flat.reshape(
        {config_.attention_key_value_heads, config_.attention_head_dimension});
    const std::array<std::int32_t, 4> rope_sections{
        static_cast<std::int32_t>(config_.rotary_dimension / 2), 0, 0, 0};
    apply_qwen35_rope(query,
                      RopePosition::text(state.position_),
                      config_.rotary_dimension,
                      rope_sections,
                      config_.rope_frequency_base);
    apply_qwen35_rope(key,
                      RopePosition::text(state.position_),
                      config_.rotary_dimension,
                      rope_sections,
                      config_.rope_frequency_base);

    core::Tensor attended({config_.attention_heads,
                           config_.attention_head_dimension});
    causal_attention_step(query, key, value, state.attention_, attended);
    for (std::size_t index = 0; index < attended.element_count(); ++index) {
        attended.data()[index] *= sigmoid(attention_gate.data()[index]);
    }
    core::Tensor attention_output({1, config_.hidden_size});
    dense(attended.data(), attention_output_projection_, attention_output.data());
    add_in_place(hidden.data(), attention_output.data());

    backend_->rms_norm(hidden, norm_, config_.rms_epsilon, normalized);
    core::Tensor logits({1, config_.vocabulary_size});
    const auto embeddings = embedding_.data();
    for (std::size_t token_index = 0; token_index < config_.vocabulary_size; ++token_index) {
        double dot = 0.0;
        const std::size_t embedding_offset = token_index * config_.hidden_size;
        for (std::size_t dimension = 0; dimension < config_.hidden_size; ++dimension) {
            dot += static_cast<double>(normalized.data()[dimension]) *
                   static_cast<double>(embeddings[embedding_offset + dimension]);
        }
        logits.data()[token_index] = static_cast<float>(dot);
    }

    ++state.position_;
    if (state.position_ != state.ssm_.sequence_length() ||
        state.position_ != state.attention_.size()) {
        throw std::logic_error("tiny model cache components advanced inconsistently");
    }
    return logits;
}

core::Tensor TinyHybridModel::prefill(std::span<const std::uint32_t> token_ids,
                                      TinyHybridState& state) {
    if (token_ids.empty()) {
        throw std::invalid_argument("tiny model prefill requires at least one token");
    }
    if (token_ids.size() > state.capacity() - state.sequence_length()) {
        throw std::overflow_error("tiny model prefill exceeds state capacity");
    }
    core::Tensor logits({1, config_.vocabulary_size});
    for (const std::uint32_t token : token_ids) {
        logits = forward_token(token, state);
    }
    return logits;
}

std::vector<std::uint32_t> TinyHybridModel::generate(
    std::span<const std::uint32_t> prompt,
    std::size_t new_tokens,
    TinyHybridState& state,
    Sampler& sampler) {
    if (prompt.empty()) {
        throw std::invalid_argument("tiny model generation requires a prompt");
    }
    if (prompt.size() + new_tokens > state.capacity() - state.sequence_length()) {
        throw std::overflow_error("tiny model generation exceeds state capacity");
    }
    core::Tensor logits = prefill(prompt, state);
    std::vector<std::uint32_t> generated;
    generated.reserve(new_tokens);
    for (std::size_t index = 0; index < new_tokens; ++index) {
        const SampleResult result = sampler.sample(logits.data());
        generated.push_back(result.token_id);
        logits = forward_token(result.token_id, state);
    }
    return generated;
}

}  // namespace oracle::runtime
