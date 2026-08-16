#pragma once

#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/qwen35_mtp_chain.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace oracle::runtime {

// Batched-verification counterpart of Slice 4's Qwen35MtpChainAccounting. Carries
// the identical accounting surface (proposed/accepted/rejected/unused_suffix/
// verification_count/canonical_tokens/first_rejection_index/full_chain_accepted --
// required to be bit-for-bit equal to Slice 4's result for the same chain, see
// docs/PHASE_2F.md Slice 5's Lane A/B comparisons) plus Slice-5-specific
// speculative-state observability fields (section 11/18 of the Slice 5 taskblock).
struct Qwen35MtpChainAccountingBatched {
    std::uint32_t proposed{0};
    std::uint32_t accepted{0};
    std::uint32_t rejected{0};
    std::uint32_t unused_suffix{0};
    std::uint32_t verification_count{0};
    std::vector<std::uint32_t> canonical_tokens;
    std::int32_t first_rejection_index{-1};
    bool full_chain_accepted{false};

    // How many draft tokens were spuriously fed to the target during the
    // speculative "exploration" batch, regardless of how many survived as
    // canonical (== proposed, always -- the whole chain is always evaluated
    // speculatively before any acceptance decision is made).
    std::uint32_t speculative_target_tokens_evaluated{0};

    // fingerprint_qwen35_state(canonical_state) at three explicitly labelled
    // points: before any speculative write; after the full speculative batch
    // (this "dirty" mid-state may include rejected/unused-suffix tokens that
    // never become canonical); and after the final canonical commit (this MUST
    // equal Slice 4's sequential result's final state fingerprint).
    std::uint64_t target_state_before_fingerprint{0};
    std::uint64_t target_speculative_state_fingerprint{0};
    std::uint64_t target_state_after_fingerprint{0};

    // Logits produced by decoding the last committed canonical token -- the
    // target's own next-token distribution immediately after this verification,
    // exposed for direct comparison against Slice 4's equivalent (see the bonus-
    // token forensics section: on full acceptance this is numerically the same
    // "bonus" observation the pinned reference gets for free from its batch).
    std::vector<float> final_target_logits;
};

// Batched counterpart of Slice 4's verify_qwen35_mtp_draft_chain. Speculatively
// evaluates the *entire* chain through the target in one exploratory pass
// (execute_qwen35_reference_target_multi), determines the accepted prefix from
// that pass's logits, and either keeps the speculative write (full acceptance --
// nothing to undo) or rolls canonical_state back to its pre-speculative boundary
// and re-commits exactly the confirmed canonical sequence (accepted drafts plus
// the one rejection-correcting token, if any) via a second, short forward pass.
// This is an independent implementation of the same contract Slice 4 proves
// sequentially -- it does not call, wrap, or delegate to
// verify_qwen35_mtp_draft_chain, and Slice 4's function is unmodified and remains
// the permanent correctness oracle both lanes are compared against.
//
// No production-performance claim is made: Oracle has no batched compute
// backend, so the "speculative batch" is itself a loop over the same validated
// single-token target forward Slice 3/4 already use (see
// execute_qwen35_reference_target_multi). What this function demonstrates is the
// *algorithm* -- speculative multi-token evaluation, accepted-prefix
// determination, and exact rollback/recommit of non-canonical state -- producing
// results provably identical to Slice 4's sequential reference.
[[nodiscard]] Qwen35MtpChainAccountingBatched verify_qwen35_mtp_draft_chain_batched(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35Weights& weights,
    const Qwen35MtpDraftChain& chain,
    std::span<const float> anchor_target_logits,
    HybridCache& canonical_state,
    std::uint32_t configured_max_depth);

[[nodiscard]] std::string qwen35_mtp_chain_accounting_batched_text(
    const Qwen35MtpChainAccountingBatched& accounting);

}  // namespace oracle::runtime
