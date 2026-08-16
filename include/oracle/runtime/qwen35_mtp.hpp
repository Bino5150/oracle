#pragma once

#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/qwen35_block.hpp"
#include "oracle/runtime/reference_kernels.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oracle::runtime {

// Numerical/state contract labels for one MTP/NextN reference prediction. Separate
// from Qwen35ForwardContractLabels: MTP is a distinct, separately-labelled numerical
// contract from the ordinary backbone (see docs/PHASE_2F.md), and its state is
// intentionally isolated from the canonical HybridCache (see state_ownership).
struct Qwen35MtpContractLabels {
    std::string execution_projection{"decoded-f32-scalar"};
    std::string execution_attention_cache{"f32-semantic"};
    std::string state_ownership{"isolated-ephemeral-mtp-state"};
};

// Everything one real NextN/MTP prediction needs beyond the bound manifest/weights.
// target_hidden_state is the target backbone's h_nextn row (its last hidden state,
// taken after the backbone's own final output_norm) at the position that produced
// candidate_token_id -- source-verified from beellama's graph_mtp
// (src/models/qwen35.cpp:285-290): `cur = build_norm(inpL, model.output_norm, ...);
// res->t_h_nextn = cur;`.
struct Qwen35MtpInput {
    std::uint32_t candidate_token_id{0};
    std::vector<float> target_hidden_state;
    RopePosition position;
};

struct Qwen35MtpTrace {
    std::vector<Qwen35TraceTensor> tensors;
    Qwen35BlockTrace block_trace;

    [[nodiscard]] const Qwen35TraceTensor* find(std::string_view name) const noexcept;
};

struct Qwen35MtpResult {
    std::vector<float> logits;
    std::uint32_t argmax_token_id{0};
    Qwen35MtpContractLabels contracts;
    Qwen35MtpTrace trace;
};

// Executes exactly one real Qwen3.5 NextN/MTP prediction through Oracle's decoded-F32
// scalar reference path: enorm/hnorm -> fusion -> eh_proj -> the appended MTP block's
// attention+FFN (reusing execute_qwen35_attention_block_reference, since Slice 2's
// source forensics established that sub-computation is architecturally identical to
// an ordinary Qwen3.5 full-attention backbone block) -> shared_head_norm -> the tied
// output head -> complete vocabulary-wide draft logits. Requires manifest.has_mtp()
// and bound MTP weights; the ordinary 24-block backbone is untouched by this call.
//
// State is never accepted as a parameter and never shared with a HybridCache: the
// MTP block's single-position attention KV state is created fresh inside this call
// and discarded when it returns, matching the appended block's real state ownership
// (its own isolated attention KV, never the canonical target's cache) and Phase 2F
// Slice 2's requirement that speculative state stay isolated from canonical state.
//
// Implements exactly one NextN prediction. No draft loop, no acceptance/rejection,
// no rollback, no speculative cache branching -- those remain a later slice.
[[nodiscard]] Qwen35MtpResult execute_qwen35_reference_mtp(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35Weights& weights,
    const Qwen35MtpInput& input,
    bool capture_intermediates = true);

// One MTP step's result, plus the value chaining actually needs to seed the next
// step: the block's own post-shared_head_norm hidden state (source-verified:
// graph_mtp.cpp:797-804 -- `res->t_h_nextn = cur` is captured immediately after
// shared_head_norm and *before* the final output-row selection/head projection).
// Phase 2F Slice 2's `Qwen35MtpResult` intentionally does not carry this value (a
// single depth-1 prediction never needs it); Slice 4's draft chain does.
struct Qwen35MtpStepOutput {
    Qwen35MtpResult result;
    std::vector<float> chained_hidden_state;
};

// Same computation as execute_qwen35_reference_mtp, factored out so a caller can
// supply its own KvCache instead of a fresh single-entry one -- this is what makes
// chaining possible: calling this repeatedly with the *same*, growing mtp_state
// reproduces the pinned reference's own MTP draft context, whose attention KV
// cache accumulates one entry per chained draft step (source-verified: beellama
// common/speculative.cpp's draft() loop runs one llama_decode(ctx_dft, ...) per
// step on the same ctx_dft, so each step's causal attention sees every prior
// step's K/V in that same chain). execute_qwen35_reference_mtp(...) is exactly
// this function called with a fresh capacity-1 KvCache -- its behavior and
// signature are unchanged from Slice 2.
[[nodiscard]] Qwen35MtpStepOutput execute_qwen35_reference_mtp_step(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35Weights& weights,
    const Qwen35MtpInput& input,
    KvCache& mtp_state,
    bool capture_intermediates = true);

[[nodiscard]] std::string qwen35_mtp_trace_text(const Qwen35MtpResult& result);
[[nodiscard]] std::string qwen35_mtp_trace_json(const Qwen35MtpResult& result,
                                                bool include_logits = false);

}  // namespace oracle::runtime
