#pragma once

#include "oracle/model/qwen35_weights.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/reference_kernels.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oracle::runtime {

struct Qwen35TraceTensor {
    std::string name;
    std::vector<float> values;
};

struct Qwen35MatvecOverrides {
    std::vector<Qwen35TraceTensor> tensors;

    [[nodiscard]] const Qwen35TraceTensor* find(std::string_view name) const noexcept;
};

struct Qwen35BlockTrace {
    std::uint32_t block_index{0};
    model::Qwen35BlockKind kind{model::Qwen35BlockKind::recurrent};
    std::uint64_t position{0};
    std::vector<Qwen35TraceTensor> tensors;

    [[nodiscard]] const Qwen35TraceTensor* find(std::string_view name) const noexcept;
};

struct Qwen35BlockResult {
    std::vector<float> output;
    Qwen35BlockTrace trace;
};

[[nodiscard]] Qwen35BlockResult execute_qwen35_recurrent_block_reference(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35BackboneBlockWeights& block,
    std::span<const float> input,
    SsmState& state,
    bool capture_intermediates = true,
    const Qwen35MatvecOverrides* matvec_overrides = nullptr);

[[nodiscard]] Qwen35BlockResult execute_qwen35_attention_block_reference(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35BackboneBlockWeights& block,
    std::span<const float> input,
    KvCache& cache,
    RopePosition position,
    bool capture_intermediates = true,
    const Qwen35MatvecOverrides* matvec_overrides = nullptr);

[[nodiscard]] std::string qwen35_block_trace_text(const Qwen35BlockTrace& trace);
[[nodiscard]] std::string qwen35_block_trace_json(const Qwen35BlockTrace& trace);

}  // namespace oracle::runtime
