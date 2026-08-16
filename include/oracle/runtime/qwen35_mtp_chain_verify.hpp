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

// Partial-acceptance accounting for one bounded draft chain. Invariants (see
// verify_qwen35_mtp_draft_chain, which is the only producer of this type):
//   accepted + rejected <= proposed
//   rejected is 0 or 1 (first-rejection semantics: verification stops at the first
//     mismatch, so at most one draft is ever rejected per chain)
//   unused_suffix == proposed - accepted - rejected
//   full_chain_accepted <=> (accepted == proposed && rejected == 0)
struct Qwen35MtpChainAccounting {
    std::uint32_t proposed{0};
    std::uint32_t accepted{0};
    std::uint32_t rejected{0};
    std::uint32_t unused_suffix{0};
    std::uint32_t verification_count{0};
    std::vector<std::uint32_t> canonical_tokens;
    std::int32_t first_rejection_index{-1};  // -1 when no draft was rejected
    bool full_chain_accepted{false};
};

// Sequentially verifies a bounded MTP draft chain against the target model,
// advancing canonical_state exactly once per verified draft (source-verified
// pseudo-contract, per the Slice 4 taskblock and beellama's sample-and-accept-n
// semantics extended token-by-token rather than batched):
//   for each draft, in order:
//     target_token = argmax(the target's current logits)
//     if draft.draft_token == target_token: ACCEPT, canonical_token = draft_token
//     else: REJECT, canonical_token = target_token, stop verifying the rest
//     forward canonical_token through the target exactly once (execute_qwen35_reference_token),
//       advancing canonical_state by exactly one position and producing the logits
//       used to verify the *next* draft (this is why, unlike Slice 3's single-proposal
//       verify_qwen35_mtp_proposal, this function is not a pure function: depth>1
//       verification genuinely requires a fresh target forward between drafts --
//       there is no way to obtain draft[1]'s target comparison logits without first
//       having forwarded whatever canonical token resulted from draft[0]'s decision).
// No canonical speculative write happens before a decision, and no target rollback
// is used or needed: state only ever advances forward, one decision at a time.
//
// anchor_target_logits are the target's already-produced logits at chain.seed_position
// (the anchor's own next-token distribution -- exactly what Slice 3 used for depth 1).
// configured_max_depth bounds chain.drafts.size() defensively (independent of
// whatever depth the chain generator recorded on itself).
[[nodiscard]] Qwen35MtpChainAccounting verify_qwen35_mtp_draft_chain(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35Weights& weights,
    const Qwen35MtpDraftChain& chain,
    std::span<const float> anchor_target_logits,
    HybridCache& canonical_state,
    std::uint32_t configured_max_depth);

[[nodiscard]] std::string qwen35_mtp_chain_accounting_text(const Qwen35MtpChainAccounting& accounting);
[[nodiscard]] std::string qwen35_mtp_chain_accounting_json(const Qwen35MtpChainAccounting& accounting);

}  // namespace oracle::runtime
