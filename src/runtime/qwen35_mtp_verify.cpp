#include "oracle/runtime/qwen35_mtp_verify.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace oracle::runtime {
namespace {

void require_finite(std::span<const float> values, std::string_view label) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument("Qwen3.5 MTP verification " + std::string(label) +
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

Qwen35MtpDecision verify_qwen35_mtp_proposal(const Qwen35MtpProposal& proposal,
                                             std::span<const float> target_logits,
                                             std::uint64_t target_position) {
    if (target_logits.empty()) {
        throw std::invalid_argument("Qwen3.5 MTP verification requires a non-empty vocabulary "
                                    "(invalid vocabulary size)");
    }
    if (proposal.draft_logits.size() != target_logits.size()) {
        throw std::invalid_argument(
            "Qwen3.5 MTP verification draft/target logits width mismatch: draft=" +
            std::to_string(proposal.draft_logits.size()) +
            " target=" + std::to_string(target_logits.size()));
    }
    require_finite(target_logits, "target logits");
    require_finite(proposal.draft_logits, "draft logits");

    const std::uint32_t computed_draft_token = argmax(proposal.draft_logits);
    if (computed_draft_token != proposal.draft_token) {
        throw std::invalid_argument(
            "Qwen3.5 MTP verification received an invalid draft token: proposal.draft_token=" +
            std::to_string(proposal.draft_token) + " does not match argmax(draft_logits)=" +
            std::to_string(computed_draft_token));
    }
    if (proposal.draft_token >= target_logits.size()) {
        throw std::invalid_argument("Qwen3.5 MTP verification received an invalid draft token: " +
                                    std::to_string(proposal.draft_token) +
                                    " exceeds the vocabulary size");
    }
    if (proposal.draft_position != target_position) {
        throw std::invalid_argument(
            "Qwen3.5 MTP verification position mismatch: draft_position=" +
            std::to_string(proposal.draft_position) +
            " target_position=" + std::to_string(target_position));
    }

    Qwen35MtpDecision decision;
    decision.draft_token = proposal.draft_token;
    decision.target_token = argmax(target_logits);
    decision.accepted = decision.draft_token == decision.target_token;
    decision.canonical_token = decision.accepted ? decision.draft_token : decision.target_token;
    decision.target_position = target_position;
    decision.draft_position = proposal.draft_position;
    decision.accounting.drafts_proposed = 1;
    decision.accounting.drafts_accepted = decision.accepted ? 1U : 0U;
    decision.accounting.drafts_rejected = decision.accepted ? 0U : 1U;
    decision.accounting.verification_count = 1;
    return decision;
}

std::string qwen35_mtp_decision_text(const Qwen35MtpDecision& decision) {
    std::ostringstream output;
    output << "Qwen3.5 MTP depth-1 verification\n"
           << "target_position: " << decision.target_position << '\n'
           << "draft_position: " << decision.draft_position << '\n'
           << "draft_token: " << decision.draft_token << '\n'
           << "target_token: " << decision.target_token << '\n'
           << "accepted: " << (decision.accepted ? "true" : "false") << '\n'
           << "canonical_token: " << decision.canonical_token << '\n'
           << "drafts_proposed: " << decision.accounting.drafts_proposed << '\n'
           << "drafts_accepted: " << decision.accounting.drafts_accepted << '\n'
           << "drafts_rejected: " << decision.accounting.drafts_rejected << '\n'
           << "verification_count: " << decision.accounting.verification_count << '\n';
    return output.str();
}

std::string qwen35_mtp_decision_json(const Qwen35MtpDecision& decision) {
    std::ostringstream output;
    output << '{' << "\"target_position\":" << decision.target_position << ','
           << "\"draft_position\":" << decision.draft_position << ','
           << "\"draft_token\":" << decision.draft_token << ','
           << "\"target_token\":" << decision.target_token << ','
           << "\"accepted\":" << (decision.accepted ? "true" : "false") << ','
           << "\"canonical_token\":" << decision.canonical_token << ','
           << "\"drafts_proposed\":" << decision.accounting.drafts_proposed << ','
           << "\"drafts_accepted\":" << decision.accounting.drafts_accepted << ','
           << "\"drafts_rejected\":" << decision.accounting.drafts_rejected << ','
           << "\"verification_count\":" << decision.accounting.verification_count << '}';
    return output.str();
}

std::string qwen35_mtp_verification_trace_text(const Qwen35MtpVerificationTrace& trace) {
    std::ostringstream output;
    output << qwen35_mtp_decision_text(trace.decision) << "target_state_before: 0x" << std::hex
           << trace.target_state_before_fingerprint << '\n'
           << "target_state_after: 0x" << trace.target_state_after_fingerprint << std::dec << '\n';
    return output.str();
}

std::string qwen35_mtp_verification_trace_json(const Qwen35MtpVerificationTrace& trace) {
    std::ostringstream output;
    output << '{' << "\"decision\":" << qwen35_mtp_decision_json(trace.decision) << ','
           << "\"target_state_before_fingerprint\":\"0x" << std::hex
           << trace.target_state_before_fingerprint << "\","
           << "\"target_state_after_fingerprint\":\"0x" << trace.target_state_after_fingerprint
           << std::dec << "\"}";
    return output.str();
}

}  // namespace oracle::runtime
