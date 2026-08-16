#include "oracle/runtime/qwen35_mtp_chain_verify.hpp"

#include "oracle/runtime/qwen35_forward.hpp"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace oracle::runtime {
namespace {

void require_finite(std::span<const float> values, std::string_view label) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument("Qwen3.5 MTP chain verification " + std::string(label) +
                                        " contains a non-finite value at index " +
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

Qwen35MtpChainAccounting verify_qwen35_mtp_draft_chain(const model::Qwen35Manifest& manifest,
                                                        const model::Qwen35Weights& weights,
                                                        const Qwen35MtpDraftChain& chain,
                                                        std::span<const float> anchor_target_logits,
                                                        HybridCache& canonical_state,
                                                        std::uint32_t configured_max_depth) {
    if (chain.drafts.empty()) {
        throw std::invalid_argument("Qwen3.5 MTP chain verification requires a non-empty draft "
                                    "chain (depth zero is not valid)");
    }
    if (chain.drafts.size() > configured_max_depth) {
        throw std::invalid_argument(
            "Qwen3.5 MTP chain verification received more drafts than the configured maximum "
            "depth: proposed=" +
            std::to_string(chain.drafts.size()) +
            " configured_max_depth=" + std::to_string(configured_max_depth));
    }
    if (anchor_target_logits.empty()) {
        throw std::invalid_argument(
            "Qwen3.5 MTP chain verification requires a non-empty vocabulary (invalid vocabulary "
            "size)");
    }
    require_finite(anchor_target_logits, "anchor target logits");

    for (std::size_t index = 0; index < chain.drafts.size(); ++index) {
        const Qwen35MtpDraft& draft = chain.drafts[index];
        if (draft.position != chain.seed_position + index + 1) {
            throw std::invalid_argument(
                "Qwen3.5 MTP chain verification received a non-contiguous or invalid proposal "
                "position at draft index " +
                std::to_string(index) + ": position=" + std::to_string(draft.position) +
                " expected=" + std::to_string(chain.seed_position + index + 1));
        }
        if (draft.logits.size() != anchor_target_logits.size()) {
            throw std::invalid_argument(
                "Qwen3.5 MTP chain verification draft/target logits width mismatch at draft "
                "index " +
                std::to_string(index));
        }
        if (static_cast<std::size_t>(draft.draft_token) >= anchor_target_logits.size()) {
            throw std::invalid_argument(
                "Qwen3.5 MTP chain verification received an invalid draft token at draft index " +
                std::to_string(index));
        }
        require_finite(draft.logits, "draft logits");
    }

    Qwen35MtpChainAccounting result;
    result.proposed = static_cast<std::uint32_t>(chain.drafts.size());

    std::vector<float> current_target_logits(anchor_target_logits.begin(), anchor_target_logits.end());
    std::uint64_t current_position = chain.seed_position;

    for (std::size_t index = 0; index < chain.drafts.size(); ++index) {
        const Qwen35MtpDraft& draft = chain.drafts[index];
        const std::uint32_t target_token = argmax(current_target_logits);
        ++result.verification_count;

        std::uint32_t canonical_token = 0;
        const bool accept = draft.draft_token == target_token;
        if (accept) {
            canonical_token = draft.draft_token;
            ++result.accepted;
        } else {
            canonical_token = target_token;
            result.rejected = 1;
            result.first_rejection_index = static_cast<std::int32_t>(index);
        }
        result.canonical_tokens.push_back(canonical_token);

        // Forward the canonical token through the target exactly once -- the only
        // way a token ever reaches canonical state, accepted or not. No canonical
        // speculative write happened before this decision, and no rollback is used:
        // state only ever advances.
        const Qwen35ForwardResult forward = execute_qwen35_reference_token(
            manifest, weights, canonical_token, canonical_state,
            RopePosition::text(current_position + 1), false);
        current_position += 1;

        if (!accept) {
            // Discard the unused draft suffix and stop verifying -- first-rejection
            // semantics, matching the pinned reference's sample-and-accept-n contract.
            break;
        }
        current_target_logits = forward.logits;
    }

    result.unused_suffix = result.proposed - result.accepted - result.rejected;
    result.full_chain_accepted = result.accepted == result.proposed && result.rejected == 0;
    return result;
}

std::string qwen35_mtp_chain_accounting_text(const Qwen35MtpChainAccounting& accounting) {
    std::ostringstream output;
    output << "Qwen3.5 MTP chain verification\n"
           << "proposed: " << accounting.proposed << '\n'
           << "accepted: " << accounting.accepted << '\n'
           << "rejected: " << accounting.rejected << '\n'
           << "unused_suffix: " << accounting.unused_suffix << '\n'
           << "verification_count: " << accounting.verification_count << '\n'
           << "first_rejection_index: " << accounting.first_rejection_index << '\n'
           << "full_chain_accepted: " << (accounting.full_chain_accepted ? "true" : "false") << '\n'
           << "canonical_tokens: [";
    for (std::size_t index = 0; index < accounting.canonical_tokens.size(); ++index) {
        if (index != 0) output << ',';
        output << accounting.canonical_tokens[index];
    }
    output << "]\n";
    return output.str();
}

std::string qwen35_mtp_chain_accounting_json(const Qwen35MtpChainAccounting& accounting) {
    std::ostringstream output;
    output << '{' << "\"proposed\":" << accounting.proposed << ','
           << "\"accepted\":" << accounting.accepted << ',' << "\"rejected\":" << accounting.rejected
           << ',' << "\"unused_suffix\":" << accounting.unused_suffix << ','
           << "\"verification_count\":" << accounting.verification_count << ','
           << "\"first_rejection_index\":" << accounting.first_rejection_index << ','
           << "\"full_chain_accepted\":" << (accounting.full_chain_accepted ? "true" : "false") << ','
           << "\"canonical_tokens\":[";
    for (std::size_t index = 0; index < accounting.canonical_tokens.size(); ++index) {
        if (index != 0) output << ',';
        output << accounting.canonical_tokens[index];
    }
    output << "]}";
    return output.str();
}

}  // namespace oracle::runtime
