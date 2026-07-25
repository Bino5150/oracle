#pragma once

#include "oracle/backend/backend.hpp"
#include "oracle/core/tensor.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oracle::runtime {

struct TinyHybridConfig {
    std::size_t vocabulary_size{16};
    std::size_t hidden_size{8};

    std::size_t ssm_key_heads{2};
    std::size_t ssm_value_heads{2};
    std::size_t ssm_key_dimension{2};
    std::size_t ssm_value_dimension{2};
    std::size_t ssm_convolution_kernel{3};

    std::size_t attention_heads{2};
    std::size_t attention_key_value_heads{1};
    std::size_t attention_head_dimension{4};
    std::size_t rotary_dimension{4};
    float rope_frequency_base{10000.0F};
    float rms_epsilon{1.0e-6F};
};

class TinyHybridState {
public:
    TinyHybridState(const TinyHybridConfig& config, std::size_t maximum_tokens);

    void reset() noexcept;
    [[nodiscard]] std::size_t sequence_length() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t byte_size() const noexcept;

private:
    friend class TinyHybridModel;

    SsmState ssm_;
    KvCache attention_;
    std::size_t position_{0};
};

class TinyHybridModel {
public:
    explicit TinyHybridModel(TinyHybridConfig config = {}, std::uint64_t seed = 5150);

    TinyHybridModel(const TinyHybridModel&) = delete;
    TinyHybridModel& operator=(const TinyHybridModel&) = delete;
    TinyHybridModel(TinyHybridModel&&) noexcept = default;
    TinyHybridModel& operator=(TinyHybridModel&&) noexcept = default;
    ~TinyHybridModel() = default;

    [[nodiscard]] const TinyHybridConfig& config() const noexcept;
    [[nodiscard]] core::Tensor forward_token(std::uint32_t token_id,
                                             TinyHybridState& state);
    [[nodiscard]] core::Tensor prefill(std::span<const std::uint32_t> token_ids,
                                      TinyHybridState& state);
    [[nodiscard]] std::vector<std::uint32_t> generate(
        std::span<const std::uint32_t> prompt,
        std::size_t new_tokens,
        TinyHybridState& state,
        Sampler& sampler);

private:
    [[nodiscard]] static TinyHybridConfig validate_config(TinyHybridConfig config);
    void initialize_weights(std::uint64_t seed);

    TinyHybridConfig config_;
    std::unique_ptr<backend::IBackend> backend_;

    core::Tensor embedding_;
    core::Tensor norm_;

    core::Tensor ssm_q_projection_;
    core::Tensor ssm_k_projection_;
    core::Tensor ssm_v_projection_;
    core::Tensor ssm_z_projection_;
    core::Tensor ssm_beta_projection_;
    core::Tensor ssm_decay_projection_;
    core::Tensor ssm_convolution_weights_;
    core::Tensor ssm_output_projection_;

    core::Tensor attention_q_projection_;
    core::Tensor attention_k_projection_;
    core::Tensor attention_v_projection_;
    core::Tensor attention_gate_projection_;
    core::Tensor attention_output_projection_;
};

}  // namespace oracle::runtime
