#include "oracle/runtime/qwen35_mtp_chain.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace oracle::runtime {
namespace {

// Mirrors qwen35_state_fingerprint.cpp's fingerprint_attention_block() exactly
// (same FNV1a64 algorithm, same field order), applied here to a standalone
// MTP-local KvCache rather than one block of a HybridCache. Diagnostic-only.
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

void hash_floats(std::uint64_t& hash, std::span<const float> values) noexcept {
    hash_bytes(hash, values.data(), values.size() * sizeof(float));
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    hash_bytes(hash, &value, sizeof(value));
}

[[nodiscard]] std::uint64_t fingerprint_kv_cache(const KvCache& cache) {
    std::uint64_t hash = kFnvOffset;
    hash_u64(hash, cache.size());
    hash_u64(hash, cache.key_value_heads());
    hash_u64(hash, cache.key_dimension());
    hash_u64(hash, cache.value_dimension());
    for (std::size_t token = 0; token < cache.size(); ++token) {
        for (std::size_t head = 0; head < cache.key_value_heads(); ++head) {
            hash_floats(hash, cache.key(token, head));
            hash_floats(hash, cache.value(token, head));
        }
    }
    return hash;
}

[[nodiscard]] std::uint64_t fingerprint_floats(std::span<const float> values) noexcept {
    std::uint64_t hash = kFnvOffset;
    hash_floats(hash, values);
    return hash;
}

}  // namespace

Qwen35MtpDraftChain generate_qwen35_mtp_draft_chain(const model::Qwen35Manifest& manifest,
                                                     const model::Qwen35Weights& weights,
                                                     std::uint32_t seed_token,
                                                     std::span<const float> seed_hidden_state,
                                                     std::uint64_t seed_position,
                                                     std::uint32_t max_depth,
                                                     bool capture_intermediates) {
    if (max_depth == 0) {
        throw std::invalid_argument("Qwen3.5 MTP draft chain requires max_depth >= 1");
    }
    if (!manifest.has_mtp()) {
        throw std::invalid_argument("Qwen3.5 MTP draft chain requires manifest.has_mtp()");
    }

    Qwen35MtpDraftChain chain;
    chain.requested_depth = max_depth;
    chain.seed_position = seed_position;
    chain.seed_token = seed_token;
    chain.drafts.reserve(max_depth);

    // One shared, growing KvCache for the whole chain (source-verified: the pinned
    // reference decodes every draft step on the same ctx_dft, so the MTP block's
    // attention causally accumulates state across the chain). Never the canonical
    // HybridCache; discarded when this function returns.
    KvCache mtp_state(max_depth, manifest.attention_head_count_kv, manifest.attention_key_length,
                      manifest.attention_value_length);

    std::uint32_t current_token = seed_token;
    std::vector<float> current_hidden(seed_hidden_state.begin(), seed_hidden_state.end());

    for (std::uint32_t index = 0; index < max_depth; ++index) {
        Qwen35MtpInput step_input;
        step_input.candidate_token_id = current_token;
        step_input.target_hidden_state = current_hidden;
        // RoPE position for computing D_index is seed_position + index (source-verified:
        // beellama draft() decodes D_k at RoPE position dp.n_past + k).
        step_input.position = RopePosition::text(seed_position + index);

        const Qwen35MtpStepOutput step = execute_qwen35_reference_mtp_step(
            manifest, weights, step_input, mtp_state, capture_intermediates);

        Qwen35MtpDraft draft;
        draft.draft_index = index;
        // Outward-facing position matches Slice 3's verification convention: the
        // *predicted* canonical position, one past the RoPE position just used.
        draft.position = seed_position + index + 1;
        draft.input_token = current_token;
        draft.draft_token = step.result.argmax_token_id;
        draft.logits = step.result.logits;
        draft.input_hidden_state_fingerprint = fingerprint_floats(current_hidden);
        draft.mtp_state_fingerprint = fingerprint_kv_cache(mtp_state);
        chain.drafts.push_back(std::move(draft));

        current_token = step.result.argmax_token_id;
        current_hidden = step.chained_hidden_state;
    }

    return chain;
}

}  // namespace oracle::runtime
