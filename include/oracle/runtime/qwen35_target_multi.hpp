#pragma once

#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/runtime/hybrid_cache.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oracle::runtime {

// One target forward step within a multi-token target evaluation: the token that
// was decoded, the RoPE position it was decoded at, and the logits that decode
// produced -- which verify whatever draft token is proposed for position + 1
// (the same one-token shift Slice 3/4 established for single-step verification;
// source-verified again for the multi-token case from beellama's speculative
// batch construction, see docs/PHASE_2F.md Slice 5: spec_i_batch[0] indexes the
// anchor token's own logits row to verify draft[0], spec_i_batch[i+1] indexes
// draft[i]'s logits row to verify draft[i+1]).
struct Qwen35TargetMultiTokenStep {
    std::uint32_t token_id{0};
    std::uint64_t position{0};
    std::vector<float> logits;
};

struct Qwen35TargetMultiTokenResult {
    std::vector<Qwen35TargetMultiTokenStep> steps;  // one per input token, in order
};

// Executes `tokens` through the target model at consecutive RoPE positions
// start_position, start_position+1, ..., advancing `state` by exactly one
// position per token -- the same state-advance semantics as ordinary
// autoregressive decoding (execute_qwen35_reference_token), because that is
// literally what this function calls, once per token, in order. This is
// deliberately NOT a true batched/parallel target evaluation: Oracle has no
// batched compute backend, so this loop only models a single speculative
// target batch's *logical* contract (one ordered set of verification logits
// per proposed token) for the caller -- it carries no batched-performance
// implication and none should be inferred from it. Numerically, looping over
// the already-validated single-token path is exact by construction: each call
// mutates `state` exactly as one more token of ordinary generation would.
[[nodiscard]] Qwen35TargetMultiTokenResult execute_qwen35_reference_target_multi(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35Weights& weights,
    std::span<const std::uint32_t> tokens,
    std::uint64_t start_position,
    HybridCache& state,
    bool capture_block_outputs = false);

}  // namespace oracle::runtime
