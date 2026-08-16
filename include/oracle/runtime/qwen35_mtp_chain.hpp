#pragma once

#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/runtime/qwen35_mtp.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oracle::runtime {

// One draft in a chained MTP proposal sequence. draft_index is 0-based (D0, D1, ...);
// position is the canonical position this draft *predicts* -- matching
// Qwen35MtpDecision's target_position/draft_position convention from Slice 3, so a
// Qwen35MtpDraft can be verified directly against the target's logits at `position`.
// The RoPE position the MTP block's own attention actually used to compute this
// draft is `position - 1` (source-verified: beellama's draft() loop decodes D_k at
// RoPE position `dp.n_past + k`, which predicts the token at `dp.n_past + k + 1`).
struct Qwen35MtpDraft {
    std::uint32_t draft_index{0};
    std::uint64_t position{0};
    std::uint32_t input_token{0};
    std::uint32_t draft_token{0};
    std::vector<float> logits;
    std::uint64_t input_hidden_state_fingerprint{0};
    std::uint64_t mtp_state_fingerprint{0};
};

struct Qwen35MtpDraftChain {
    std::vector<Qwen35MtpDraft> drafts;  // ordered D0..Dn-1, n == requested_depth
    std::uint32_t requested_depth{0};
    std::uint64_t seed_position{0};
    std::uint32_t seed_token{0};
};

// Generates a bounded chain of MTP proposals D0..D(max_depth-1), reproducing the
// pinned reference's chaining contract exactly (source-verified, beellama
// common/speculative.cpp draft(), commit a620cbd48):
//   - D0's input token is seed_token (the canonical anchor/context token) and its
//     input hidden state is seed_hidden_state (the TARGET's own h_nextn at
//     seed_position -- not an MTP output);
//   - D_k (k>0)'s input token is D(k-1)'s draft_token, and its input hidden state
//     is D(k-1)'s *own* chained output hidden state (execute_qwen35_reference_mtp_step's
//     post-shared_head_norm value) -- never the target's;
//   - the MTP block's attention KV state accumulates across the whole chain: one
//     shared, growing KvCache is used for all max_depth steps (each step causally
//     attends to every prior step's K/V in the same chain), never reset mid-chain
//     and never the canonical 24-block HybridCache.
// Draft depth is fixed and explicit -- no confidence gating, no adaptive depth (the
// pinned reference's p_min early-stop is a policy heuristic, not an architectural
// requirement, and is out of scope here). Greedy only, matching Slice 2/3.
[[nodiscard]] Qwen35MtpDraftChain generate_qwen35_mtp_draft_chain(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35Weights& weights,
    std::uint32_t seed_token,
    std::span<const float> seed_hidden_state,
    std::uint64_t seed_position,
    std::uint32_t max_depth,
    bool capture_intermediates = false);

}  // namespace oracle::runtime
