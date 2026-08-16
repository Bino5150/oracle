#include "oracle/runtime/qwen35_target_multi.hpp"

#include "oracle/runtime/qwen35_forward.hpp"

namespace oracle::runtime {

Qwen35TargetMultiTokenResult execute_qwen35_reference_target_multi(
    const model::Qwen35Manifest& manifest, const model::Qwen35Weights& weights,
    std::span<const std::uint32_t> tokens, std::uint64_t start_position, HybridCache& state,
    bool capture_block_outputs) {
    Qwen35TargetMultiTokenResult result;
    result.steps.reserve(tokens.size());

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint64_t position = start_position + index;
        const Qwen35ForwardResult forward = execute_qwen35_reference_token(
            manifest, weights, tokens[index], state, RopePosition::text(position),
            capture_block_outputs);

        Qwen35TargetMultiTokenStep step;
        step.token_id = tokens[index];
        step.position = position;
        step.logits = forward.logits;
        result.steps.push_back(std::move(step));
    }

    return result;
}

}  // namespace oracle::runtime
