#include "oracle/runtime/qwen35_mtp_chain_verify_batched.hpp"

#include "oracle/runtime/qwen35_state_fingerprint.hpp"
#include "oracle/runtime/qwen35_target_multi.hpp"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace oracle::runtime {
namespace {

void require_finite(std::span<const float> values, std::string_view label) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument("Qwen3.5 MTP batched chain verification " +
                                        std::string(label) + " contains a non-finite value at index " +
                                        std::to_string(index));
        }
    }
}

[[nodiscard]] std::uint32_t argmax(std::span<const float> logits) {
    std::size_t best_index = 0;
    float best_value = logits.front();
    for (std::size_t index = 1; index < logits.size(); ++index) {
        if (logits[index] > best_value) {
            best_value = logits[index];
            best_index = index;
        }
    }
    return static_cast<std::uint32_t>(best_index);
}

}  // namespace

Qwen35MtpChainAccountingBatched verify_qwen35_mtp_draft_chain_batched(
    const model::Qwen35Manifest& manifest, const model::Qwen35Weights& weights,
    const Qwen35MtpDraftChain& chain, std::span<const float> anchor_target_logits,
    HybridCache& canonical_state, std::uint32_t configured_max_depth) {
    // --- Independent validation (mirrors, but does not call, Slice 4's checks). ---
    if (chain.drafts.empty()) {
        throw std::invalid_argument(
            "Qwen3.5 MTP batched chain verification requires a non-empty draft chain "
            "(depth zero is not valid)");
    }
    if (chain.drafts.size() > configured_max_depth) {
        throw std::invalid_argument(
            "Qwen3.5 MTP batched chain verification received more drafts than the configured "
            "maximum depth: proposed=" +
            std::to_string(chain.drafts.size()) +
            " configured_max_depth=" + std::to_string(configured_max_depth));
    }
    if (anchor_target_logits.empty()) {
        throw std::invalid_argument(
            "Qwen3.5 MTP batched chain verification requires a non-empty vocabulary (invalid "
            "vocabulary size)");
    }
    require_finite(anchor_target_logits, "anchor target logits");

    for (std::size_t index = 0; index < chain.drafts.size(); ++index) {
        const Qwen35MtpDraft& draft = chain.drafts[index];
        if (draft.position != chain.seed_position + index + 1) {
            throw std::invalid_argument(
                "Qwen3.5 MTP batched chain verification received a non-contiguous or invalid "
                "proposal position at draft index " +
                std::to_string(index) + ": position=" + std::to_string(draft.position) +
                " expected=" + std::to_string(chain.seed_position + index + 1));
        }
        if (draft.logits.size() != anchor_target_logits.size()) {
            throw std::invalid_argument(
                "Qwen3.5 MTP batched chain verification draft/target logits width mismatch at "
                "draft index " +
                std::to_string(index));
        }
        if (static_cast<std::size_t>(draft.draft_token) >= anchor_target_logits.size()) {
            throw std::invalid_argument(
                "Qwen3.5 MTP batched chain verification received an invalid draft token at draft "
                "index " +
                std::to_string(index));
        }
        require_finite(draft.logits, "draft logits");
    }

    Qwen35MtpChainAccountingBatched result;
    result.proposed = static_cast<std::uint32_t>(chain.drafts.size());
    result.speculative_target_tokens_evaluated = result.proposed;
    result.target_state_before_fingerprint =
        fingerprint_qwen35_state(canonical_state).combined;

    // --- Speculative exploration: the whole chain, in one batch-shaped pass. ---
    const HybridCacheBoundary boundary = canonical_state.mark_boundary();

    std::vector<std::uint32_t> draft_tokens;
    draft_tokens.reserve(chain.drafts.size());
    for (const Qwen35MtpDraft& draft : chain.drafts) {
        draft_tokens.push_back(draft.draft_token);
    }

    const Qwen35TargetMultiTokenResult speculative =
        execute_qwen35_reference_target_multi(manifest, weights, draft_tokens,
                                              chain.seed_position + 1, canonical_state, false);
    result.target_speculative_state_fingerprint =
        fingerprint_qwen35_state(canonical_state).combined;

    // --- Determine the accepted prefix from the speculative batch's own logits. ---
    // Verification logit alignment (source-verified, beellama spec_i_batch
    // construction): draft[0] is verified by the *already-produced* anchor
    // logits (the row that decoded the anchor token, before any speculative
    // write); draft[i>0] is verified by the logits produced when the
    // speculative batch decoded draft[i-1] -- i.e. speculative.steps[i-1].logits.
    for (std::size_t index = 0; index < chain.drafts.size(); ++index) {
        const Qwen35MtpDraft& draft = chain.drafts[index];
        const std::span<const float> verify_logits =
            index == 0 ? anchor_target_logits : std::span<const float>(speculative.steps[index - 1].logits);
        const std::uint32_t target_token = argmax(verify_logits);
        ++result.verification_count;

        const bool accept = draft.draft_token == target_token;
        if (accept) {
            result.canonical_tokens.push_back(draft.draft_token);
            ++result.accepted;
        } else {
            result.canonical_tokens.push_back(target_token);
            result.rejected = 1;
            result.first_rejection_index = static_cast<std::int32_t>(index);
            break;
        }
    }

    result.unused_suffix = result.proposed - result.accepted - result.rejected;
    result.full_chain_accepted = result.accepted == result.proposed && result.rejected == 0;

    if (result.full_chain_accepted) {
        // The speculative batch WAS exactly the canonical sequence -- nothing to
        // roll back or recommit. The "bonus" observation is the logits produced
        // by decoding the last (confirmed-canonical) draft token.
        result.target_state_after_fingerprint = result.target_speculative_state_fingerprint;
        result.final_target_logits = speculative.steps.back().logits;
        return result;
    }

    // --- Partial or zero acceptance: discard the speculative write entirely and
    // recommit exactly the confirmed canonical sequence (accepted drafts, plus
    // the one rejection-correcting token). This reproduces Slice 4's sequential
    // result exactly -- same tokens, same forward calls, same order -- at the
    // cost of recomputing the (already-correct) accepted prefix's attention KV,
    // a deliberate simplicity/correctness tradeoff (see docs/PHASE_2F.md Slice 5,
    // "target state commit model"): no production-performance claim is made.
    canonical_state.truncate_to(boundary);
    const Qwen35TargetMultiTokenResult commit = execute_qwen35_reference_target_multi(
        manifest, weights, result.canonical_tokens, chain.seed_position + 1, canonical_state, false);
    result.target_state_after_fingerprint = fingerprint_qwen35_state(canonical_state).combined;
    result.final_target_logits = commit.steps.back().logits;

    return result;
}

std::string qwen35_mtp_chain_accounting_batched_text(const Qwen35MtpChainAccountingBatched& accounting) {
    std::ostringstream output;
    output << "Qwen3.5 MTP batched chain verification\n"
           << "proposed: " << accounting.proposed << '\n'
           << "accepted: " << accounting.accepted << '\n'
           << "rejected: " << accounting.rejected << '\n'
           << "unused_suffix: " << accounting.unused_suffix << '\n'
           << "verification_count: " << accounting.verification_count << '\n'
           << "first_rejection_index: " << accounting.first_rejection_index << '\n'
           << "full_chain_accepted: " << (accounting.full_chain_accepted ? "true" : "false") << '\n'
           << "speculative_target_tokens_evaluated: " << accounting.speculative_target_tokens_evaluated
           << '\n'
           << "target_state_before_fingerprint: 0x" << std::hex
           << accounting.target_state_before_fingerprint << std::dec << '\n'
           << "target_speculative_state_fingerprint: 0x" << std::hex
           << accounting.target_speculative_state_fingerprint << std::dec << '\n'
           << "target_state_after_fingerprint: 0x" << std::hex
           << accounting.target_state_after_fingerprint << std::dec << '\n'
           << "canonical_tokens: [";
    for (std::size_t index = 0; index < accounting.canonical_tokens.size(); ++index) {
        if (index != 0) output << ',';
        output << accounting.canonical_tokens[index];
    }
    output << "]\n";
    return output.str();
}

}  // namespace oracle::runtime
