#pragma once

#include "oracle/model/qwen35_weights.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/qwen35_block.hpp"
#include "oracle/runtime/reference_kernels.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oracle::runtime {

struct Qwen35ForwardContractLabels {
    std::string execution_projection{"decoded-f32-scalar"};
    std::string execution_attention_cache{"f32-semantic"};
    std::string independent_projection_reference{"q5-k/q6-k-x-q8-k"};
    std::string independent_attention_cache_reference{"f16-pinned"};
    std::string override_projection_source;
    std::string override_attention_cache_source;
};

struct Qwen35ForwardBlockOverrides {
    std::uint32_t block_index{0};
    Qwen35MatvecOverrides matvecs;
};

struct Qwen35ForwardOverrides {
    std::vector<Qwen35ForwardBlockOverrides> blocks;
    std::optional<Qwen35TraceTensor> logits;
    std::string projection_source_contract{"external-captured-projections"};
    std::string attention_cache_source_contract{"external-cache-unspecified"};

    [[nodiscard]] const Qwen35MatvecOverrides* find_block(std::uint32_t block_index) const noexcept;
};

struct Qwen35ForwardBlockTrace {
    std::uint32_t block_index{0};
    model::Qwen35BlockKind kind{model::Qwen35BlockKind::recurrent};
    std::vector<float> output;
};

struct Qwen35ForwardTrace {
    std::uint32_t token_id{0};
    std::uint64_t position{0};
    Qwen35ForwardContractLabels contracts;
    std::vector<float> embedding;
    std::vector<Qwen35ForwardBlockTrace> blocks;
    std::vector<float> final_norm;
    bool output_is_tied{false};
    bool mtp_executed{false};
};

struct Qwen35ForwardResult {
    std::vector<float> logits;
    Qwen35ForwardTrace trace;
};

[[nodiscard]] Qwen35ForwardResult execute_qwen35_reference_token(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35Weights& weights,
    std::uint32_t token_id,
    HybridCache& state,
    RopePosition position,
    bool capture_block_outputs = true,
    const Qwen35ForwardOverrides* overrides = nullptr);

[[nodiscard]] std::string qwen35_forward_trace_text(const Qwen35ForwardResult& result);
[[nodiscard]] std::string qwen35_forward_trace_json(const Qwen35ForwardResult& result,
                                                    bool include_logits = false);

}  // namespace oracle::runtime
