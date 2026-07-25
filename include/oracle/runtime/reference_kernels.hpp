#pragma once

#include "oracle/core/tensor.hpp"
#include "oracle/runtime/hybrid_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace oracle::runtime {

struct RopePosition {
    std::uint64_t temporal{0};
    std::uint64_t height{0};
    std::uint64_t width{0};

    [[nodiscard]] static RopePosition text(std::uint64_t position) noexcept {
        return {position, position, position};
    }
};

void embedding_lookup(const core::Tensor& embedding,
                      std::span<const std::uint32_t> token_ids,
                      core::Tensor& output);

void apply_qwen35_rope(core::Tensor& heads,
                       RopePosition position,
                       std::size_t rotary_dimension,
                       std::array<std::int32_t, 4> dimension_sections,
                       float frequency_base);

void causal_attention_step(const core::Tensor& query,
                           const core::Tensor& key,
                           const core::Tensor& value,
                           KvCache& cache,
                           core::Tensor& output);

void causal_depthwise_convolution_step(std::span<const float> input,
                                       std::span<const float> weights,
                                       SsmState& state,
                                       std::span<float> output,
                                       bool apply_silu = true);

void gated_delta_step(const core::Tensor& query,
                      const core::Tensor& key,
                      const core::Tensor& value,
                      std::span<const float> beta,
                      std::span<const float> log_decay,
                      SsmState& state,
                      core::Tensor& output,
                      float normalization_epsilon = 1.0e-6F);

}  // namespace oracle::runtime
