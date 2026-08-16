#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oracle::runtime {

// One MTP draft proposal for exactly one position. draft_logits is the full
// vocabulary-wide draft distribution (e.g. Qwen35MtpResult.logits from
// execute_qwen35_reference_mtp); draft_token is the caller's already-computed
// greedy choice from it (e.g. Qwen35MtpResult.argmax_token_id) and must be
// self-consistent with draft_logits -- this catches a caller passing a stale or
// mismatched token alongside fresh logits, which is otherwise an easy silent bug.
struct Qwen35MtpProposal {
    std::uint32_t draft_token{0};
    std::vector<float> draft_logits;
    std::uint64_t draft_position{0};
};

// Depth-1 accounting: for exactly one verification event, drafts_proposed == 1 and
// drafts_accepted + drafts_rejected == 1 and verification_count == 1. No policy,
// no scheduler -- this is a pure tally of one event, generalized only so the same
// type composes forward if a later slice chains verification events.
struct Qwen35MtpVerificationAccounting {
    std::uint32_t drafts_proposed{0};
    std::uint32_t drafts_accepted{0};
    std::uint32_t drafts_rejected{0};
    std::uint32_t verification_count{0};
};

// The canonical accept/reject decision for one proposal against one target
// distribution. The target model is always authoritative (source-verified: the
// pinned reference's verification step is itself just a comparison against the
// target's own greedy choice at the anchor position -- it requires no additional
// target forward beyond logits the target already produced from canonical state).
// Greedy only: no temperature, no stochastic sampling, no rejection-sampling
// residual-distribution correction.
struct Qwen35MtpDecision {
    std::uint32_t draft_token{0};
    std::uint32_t target_token{0};
    std::uint32_t canonical_token{0};
    bool accepted{false};
    std::uint64_t target_position{0};
    std::uint64_t draft_position{0};
    Qwen35MtpVerificationAccounting accounting;
};

// Verifies one MTP proposal against the target's own already-produced logits at
// target_position. Ties resolve via ordinary argmax convention (lowest index wins).
// Pure function: reads no state, mutates no state, performs no target forward --
// forwarding the resulting canonical_token through the target model (advancing
// canonical HybridCache by exactly one position) is the caller's separate,
// subsequent responsibility using Oracle's existing execute_qwen35_reference_token,
// exactly as it already advances state for ordinary autoregressive generation.
[[nodiscard]] Qwen35MtpDecision verify_qwen35_mtp_proposal(const Qwen35MtpProposal& proposal,
                                                            std::span<const float> target_logits,
                                                            std::uint64_t target_position);

[[nodiscard]] std::string qwen35_mtp_decision_text(const Qwen35MtpDecision& decision);
[[nodiscard]] std::string qwen35_mtp_decision_json(const Qwen35MtpDecision& decision);

// Concise structured trace for one verification event, pairing a decision with
// deterministic before/after canonical-state fingerprints (see
// oracle/runtime/qwen35_state_fingerprint.hpp for how these are computed -- this
// header intentionally does not depend on HybridCache; the caller fingerprints its
// own state and passes the two combined hashes in). Full state dumps remain
// evidence-directory artifacts, generated separately; this trace is sized for
// ordinary structured logging.
struct Qwen35MtpVerificationTrace {
    Qwen35MtpDecision decision;
    std::uint64_t target_state_before_fingerprint{0};
    std::uint64_t target_state_after_fingerprint{0};
};

[[nodiscard]] std::string qwen35_mtp_verification_trace_text(const Qwen35MtpVerificationTrace& trace);
[[nodiscard]] std::string qwen35_mtp_verification_trace_json(const Qwen35MtpVerificationTrace& trace);

}  // namespace oracle::runtime
