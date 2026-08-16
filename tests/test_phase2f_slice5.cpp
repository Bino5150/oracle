// Phase 2F Slice 5: batched target verification + canonical prefix rollback
// (oracle/runtime/qwen35_mtp_chain_verify_batched.hpp,
// oracle/runtime/qwen35_target_multi.hpp, HybridCache::mark_boundary/truncate_to).
// Hand-controlled synthetic tests that do not require the real checkpoint; the
// real full-accept/zero-accept fixtures and bonus-token forensics are
// evidence-only, recorded in
// model-reports/phase2f-slice1-mtp-baseline-20260816/slice5-batched/ (not
// committed).
//
// Slice 4's verify_qwen35_mtp_draft_chain is the permanent correctness oracle
// and is NOT modified, weakened, or called from the batched implementation.
// Every Lane A/B test here runs both verifiers independently, on separate
// HybridCache instances seeded identically, and compares their results.

#include "oracle/runtime/qwen35_mtp_chain.hpp"
#include "oracle/runtime/qwen35_mtp_chain_verify.hpp"
#include "oracle/runtime/qwen35_mtp_chain_verify_batched.hpp"
#include "oracle/runtime/qwen35_target_multi.hpp"
#include "oracle/runtime/qwen35_state_fingerprint.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/qwen35_forward.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/model/ggml_type.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Function>
void require_throws(Function&& function, std::string_view needle) {
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string_view(error.what()).find(needle) != std::string_view::npos,
                "exception message did not contain expected text: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

// --- synthetic Qwen3.5 fixture (all-zero weights by construction), same pattern
// established in tests/test_phase2f_slice3.cpp / test_phase2f_slice4.cpp ---

oracle::model::Qwen35Manifest manifest(bool mtp = false, std::uint32_t block_count = 2,
                                       std::uint32_t vocabulary_size = 32,
                                       std::uint32_t embedding_length = 64) {
    oracle::model::Qwen35Manifest value;
    value.architecture = "qwen35";
    value.model_name = "phase2f-slice5-fixture";
    value.backbone_block_count = block_count;
    value.nextn_predict_layers = mtp ? 1U : 0U;
    value.total_block_count = value.backbone_block_count + value.nextn_predict_layers;
    value.context_length = 1024;
    value.embedding_length = embedding_length;
    value.feed_forward_length = 128;
    value.attention_head_count = 2;
    value.attention_head_count_kv = 1;
    value.attention_key_length = 32;
    value.attention_value_length = 32;
    value.attention_rms_epsilon = 1.0e-6F;
    value.rope_dimension_count = 32;
    value.rope_dimension_sections = {6, 6, 4, 0};
    value.rope_frequency_base = 10000000.0F;
    value.ssm_convolution_kernel = 4;
    value.ssm_state_size = 16;
    value.ssm_group_count = 2;
    value.ssm_time_step_rank = 4;
    value.ssm_inner_size = 64;
    value.full_attention_interval = 2;
    value.vocabulary_size = vocabulary_size;
    return value;
}

struct Fixture {
    std::vector<oracle::model::GgufTensorInfo> infos;
    std::vector<std::vector<std::byte>> storage;
    std::vector<oracle::model::GgufTensorView> views;

    Fixture() {
        infos.reserve(128);
        storage.reserve(128);
        views.reserve(128);
    }

    void add(std::string name, std::initializer_list<std::uint64_t> dimensions,
             std::uint32_t type = 0) {
        infos.push_back({std::move(name), dimensions, type, 0});
        const auto* layout = oracle::model::ggml_type_layout(type);
        require(layout != nullptr, "fixture type layout missing");
        const std::uint64_t bytes = oracle::model::ggml_tensor_byte_size(infos.back().dimensions, type);
        storage.emplace_back(static_cast<std::size_t>(bytes));
        views.emplace_back(&infos.back(), layout, storage.back().data(), storage.back().size(), 0);
    }

    // Overwrites an already-added tensor's storage with the given row-major F32
    // values (used only by the identity fixture, below).
    void set_f32(const std::string& name, const std::vector<float>& values) {
        for (std::size_t index = 0; index < infos.size(); ++index) {
            if (infos[index].name != name) continue;
            require(storage[index].size() == values.size() * sizeof(float),
                    "fixture set_f32 size mismatch for " + name);
            std::memcpy(storage[index].data(), values.data(), storage[index].size());
            return;
        }
        throw std::runtime_error("fixture set_f32: tensor not found: " + name);
    }
};

void add_attention(Fixture& fixture, const oracle::model::Qwen35Manifest& model, std::uint32_t block) {
    const std::string prefix = "blk." + std::to_string(block) + ".";
    const std::uint64_t q_width = model.attention_key_length * model.attention_head_count * 2U;
    const std::uint64_t k_width = model.attention_key_length * model.attention_head_count_kv;
    const std::uint64_t v_width = model.attention_value_length * model.attention_head_count_kv;
    const std::uint64_t out_width = model.attention_key_length * model.attention_head_count;
    fixture.add(prefix + "attn_q.weight", {model.embedding_length, q_width});
    fixture.add(prefix + "attn_k.weight", {model.embedding_length, k_width});
    fixture.add(prefix + "attn_v.weight", {model.embedding_length, v_width});
    fixture.add(prefix + "attn_output.weight", {out_width, model.embedding_length});
    fixture.add(prefix + "attn_q_norm.weight", {model.attention_key_length});
    fixture.add(prefix + "attn_k_norm.weight", {model.attention_key_length});
}

void add_recurrent(Fixture& fixture, const oracle::model::Qwen35Manifest& model, std::uint32_t block) {
    const std::string prefix = "blk." + std::to_string(block) + ".";
    const std::uint64_t key_width = model.ssm_state_size * model.ssm_group_count;
    const std::uint64_t value_width = model.ssm_state_size * model.ssm_time_step_rank;
    const std::uint64_t conv_width = key_width * 2U + value_width;
    fixture.add(prefix + "attn_qkv.weight", {model.embedding_length, conv_width});
    fixture.add(prefix + "attn_gate.weight", {model.embedding_length, value_width});
    fixture.add(prefix + "ssm_conv1d.weight", {model.ssm_convolution_kernel, conv_width});
    fixture.add(prefix + "ssm_dt.bias", {model.ssm_time_step_rank});
    fixture.add(prefix + "ssm_a", {model.ssm_time_step_rank});
    fixture.add(prefix + "ssm_beta.weight", {model.embedding_length, model.ssm_time_step_rank});
    fixture.add(prefix + "ssm_alpha.weight", {model.embedding_length, model.ssm_time_step_rank});
    fixture.add(prefix + "ssm_norm.weight", {model.ssm_state_size});
    fixture.add(prefix + "ssm_out.weight", {value_width, model.embedding_length});
}

void add_common(Fixture& fixture, const oracle::model::Qwen35Manifest& model, std::uint32_t block) {
    const std::string prefix = "blk." + std::to_string(block) + ".";
    fixture.add(prefix + "attn_norm.weight", {model.embedding_length});
    fixture.add(prefix + "post_attention_norm.weight", {model.embedding_length});
    fixture.add(prefix + "ffn_gate.weight", {model.embedding_length, model.feed_forward_length});
    fixture.add(prefix + "ffn_down.weight", {model.feed_forward_length, model.embedding_length});
    fixture.add(prefix + "ffn_up.weight", {model.embedding_length, model.feed_forward_length});
}

Fixture make_fixture(const oracle::model::Qwen35Manifest& model) {
    Fixture fixture;
    fixture.add("token_embd.weight", {model.embedding_length, model.vocabulary_size});
    fixture.add("output_norm.weight", {model.embedding_length});
    for (std::uint32_t block = 0; block < model.backbone_block_count; ++block) {
        add_common(fixture, model, block);
        if (model.is_full_attention_block(block)) add_attention(fixture, model, block);
        else add_recurrent(fixture, model, block);
    }
    return fixture;
}

// Identity fixture: every non-output-projection weight stays zero (attn_output.weight
// and ffn_down.weight are zero, so every block contributes exactly zero to the
// residual stream regardless of input -- same reasoning as the all-zero fixture
// above, just not applied to token_embd/output_norm). token_embd is one-hot per
// token (embedding_length == vocabulary_size, row T is 1.0 at dimension T and 0
// elsewhere) and output_norm.weight is uniform (1.0 everywhere), so RMSNorm
// rescales a one-hot vector without changing its shape. Since the tied output
// head computes dot(normed_hidden, token_embd[V]) and both vectors are one-hot,
// the result is nonzero *only* at V == (whichever token was just decoded) --
// i.e. decoding token T always argmaxes to T. This gives genuine, distinct,
// input-dependent argmaxes at every position (unlike the all-zero fixture, which
// is constant everywhere), needed to make an off-by-one verification-alignment
// bug immediately visible rather than accidentally masked.
Fixture make_identity_fixture(const oracle::model::Qwen35Manifest& model) {
    require(model.embedding_length == model.vocabulary_size,
            "identity fixture requires embedding_length == vocabulary_size");
    Fixture fixture = make_fixture(model);

    std::vector<float> embed(static_cast<std::size_t>(model.embedding_length) * model.vocabulary_size, 0.0F);
    for (std::uint32_t token = 0; token < model.vocabulary_size; ++token) {
        embed[static_cast<std::size_t>(token) * model.embedding_length + token] = 1.0F;
    }
    fixture.set_f32("token_embd.weight", embed);

    const std::vector<float> norm_weight(model.embedding_length, 1.0F);
    fixture.set_f32("output_norm.weight", norm_weight);
    return fixture;
}

std::vector<float> make_logits(std::size_t vocab, std::size_t argmax_index, float peak = 10.0F) {
    std::vector<float> logits(vocab, 0.0F);
    logits[argmax_index] = peak;
    return logits;
}

// verify_qwen35_mtp_draft_chain[_batched] requires cache.sequence_length() ==
// seed_position + 1 on entry. Forwards dummy token 0 through every position up to
// and including seed_position.
void seed_cache_to_position(const oracle::model::Qwen35Manifest& model,
                            const oracle::model::Qwen35Weights& weights,
                            oracle::runtime::HybridCache& cache, std::uint64_t seed_position,
                            std::uint32_t seed_token = 0) {
    for (std::uint64_t position = 0; position <= seed_position; ++position) {
        static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
            model, weights, seed_token, cache, oracle::runtime::RopePosition::text(position), false));
    }
}

oracle::runtime::Qwen35MtpDraft make_draft(std::uint32_t index, std::uint64_t position,
                                           std::uint32_t input_token, std::uint32_t draft_token,
                                           std::size_t vocab) {
    oracle::runtime::Qwen35MtpDraft draft;
    draft.draft_index = index;
    draft.position = position;
    draft.input_token = input_token;
    draft.draft_token = draft_token;
    draft.logits = make_logits(vocab, draft_token);
    return draft;
}

// --- Lane A/B comparison helper --------------------------------------------

struct LaneComparisonCase {
    std::string name;
    oracle::runtime::Qwen35MtpDraftChain chain;
    std::vector<float> anchor_logits;
};

void compare_lanes(const oracle::model::Qwen35Manifest& model, const oracle::model::Qwen35Weights& weights,
                   const LaneComparisonCase& test_case, std::uint32_t configured_max_depth) {
    oracle::runtime::HybridCache cache_a(model, 20);
    oracle::runtime::HybridCache cache_b(model, 20);
    seed_cache_to_position(model, weights, cache_a, test_case.chain.seed_position);
    seed_cache_to_position(model, weights, cache_b, test_case.chain.seed_position);

    const auto lane_a = oracle::runtime::verify_qwen35_mtp_draft_chain(
        model, weights, test_case.chain, test_case.anchor_logits, cache_a, configured_max_depth);
    const auto lane_b = oracle::runtime::verify_qwen35_mtp_draft_chain_batched(
        model, weights, test_case.chain, test_case.anchor_logits, cache_b, configured_max_depth);

    const std::string label = test_case.name + ": ";
    require(lane_a.proposed == lane_b.proposed, label + "proposed mismatch");
    require(lane_a.accepted == lane_b.accepted, label + "accepted mismatch");
    require(lane_a.rejected == lane_b.rejected, label + "rejected mismatch");
    require(lane_a.unused_suffix == lane_b.unused_suffix, label + "unused_suffix mismatch");
    require(lane_a.verification_count == lane_b.verification_count,
            label + "verification_count mismatch");
    require(lane_a.first_rejection_index == lane_b.first_rejection_index,
            label + "first_rejection_index mismatch");
    require(lane_a.full_chain_accepted == lane_b.full_chain_accepted,
            label + "full_chain_accepted mismatch");
    require(lane_a.canonical_tokens == lane_b.canonical_tokens, label + "canonical_tokens mismatch");
    require(cache_a.sequence_length() == cache_b.sequence_length(),
            label + "final sequence_length mismatch");

    const auto fp_a = oracle::runtime::fingerprint_qwen35_state(cache_a);
    const auto fp_b = oracle::runtime::fingerprint_qwen35_state(cache_b);
    require(fp_a == fp_b, label + "final HybridCache fingerprint mismatch");
    require(lane_b.target_state_after_fingerprint == fp_b.combined,
            label + "batched target_state_after_fingerprint does not match cache_b's own fingerprint");

    // Peek at "what comes next" from independent copies of each lane's final
    // state (a dummy forward, discarded after use) -- required equal per the
    // taskblock's "next-target logits" comparison. Both lanes' HybridCache
    // states are already proven bit-identical above, so this necessarily
    // agrees, but we check it explicitly and literally rather than only
    // inferring it from the fingerprint match.
    oracle::runtime::HybridCache peek_a = cache_a;
    const auto next_a = oracle::runtime::execute_qwen35_reference_token(
        model, weights, 0, peek_a, oracle::runtime::RopePosition::text(cache_a.sequence_length()), false);
    require(next_a.logits == lane_b.final_target_logits,
            label + "next-target logits mismatch between Lane A peek and Lane B's final_target_logits");
}

// Item A: ALL ACCEPT.
void test_lane_ab_all_accept() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    LaneComparisonCase test_case;
    test_case.name = "A(all-accept)";
    test_case.chain.requested_depth = 3;
    test_case.chain.seed_position = 0;
    test_case.chain.seed_token = 0;
    test_case.chain.drafts = {
        make_draft(0, 1, 0, 0, model.vocabulary_size),
        make_draft(1, 2, 0, 0, model.vocabulary_size),
        make_draft(2, 3, 0, 0, model.vocabulary_size),
    };
    test_case.anchor_logits = make_logits(model.vocabulary_size, 0);
    compare_lanes(model, weights, test_case, 3);
}

// Item B: REJECT FIRST.
void test_lane_ab_reject_first() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    LaneComparisonCase test_case;
    test_case.name = "B(reject-first)";
    test_case.chain.requested_depth = 3;
    test_case.chain.seed_position = 10;
    test_case.chain.seed_token = 5;
    test_case.chain.drafts = {
        make_draft(0, 11, 5, 7, model.vocabulary_size),
        make_draft(1, 12, 7, 8, model.vocabulary_size),
        make_draft(2, 13, 8, 9, model.vocabulary_size),
    };
    test_case.anchor_logits = make_logits(model.vocabulary_size, 11);  // real choice is 11, not 7
    compare_lanes(model, weights, test_case, 3);
}

// Item C: PARTIAL ACCEPT.
void test_lane_ab_partial_accept() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    LaneComparisonCase test_case;
    test_case.name = "C(partial-accept)";
    test_case.chain.requested_depth = 3;
    test_case.chain.seed_position = 0;
    test_case.chain.seed_token = 0;
    test_case.chain.drafts = {
        make_draft(0, 1, 0, 0, model.vocabulary_size),
        make_draft(1, 2, 0, 0, model.vocabulary_size),
        make_draft(2, 3, 0, 11, model.vocabulary_size),
    };
    test_case.anchor_logits = make_logits(model.vocabulary_size, 0);
    compare_lanes(model, weights, test_case, 3);
}

// Item D: REJECT MIDDLE WITH UNUSED SUFFIX.
void test_lane_ab_reject_middle_with_suffix() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    LaneComparisonCase test_case;
    test_case.name = "D(reject-middle-with-suffix)";
    test_case.chain.requested_depth = 4;
    test_case.chain.seed_position = 0;
    test_case.chain.seed_token = 0;
    test_case.chain.drafts = {
        make_draft(0, 1, 0, 0, model.vocabulary_size),
        make_draft(1, 2, 0, 11, model.vocabulary_size),
        make_draft(2, 3, 11, 5, model.vocabulary_size),
        make_draft(3, 4, 5, 6, model.vocabulary_size),
    };
    test_case.anchor_logits = make_logits(model.vocabulary_size, 0);
    compare_lanes(model, weights, test_case, 4);
}

// --- Verification logit alignment: an off-by-one row mapping must be visible --

void test_verification_logit_alignment() {
    const auto model = manifest(false, 2, /*vocabulary_size=*/8, /*embedding_length=*/8);
    Fixture fixture = make_identity_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    // Sanity: confirm the identity property directly -- decoding token T really
    // does argmax to T, for several distinct T, proving the fixture gives
    // genuinely position/input-dependent verification logits (unlike the
    // all-zero fixture used elsewhere, which is constant everywhere).
    {
        oracle::runtime::HybridCache probe(model, 8);
        for (const std::uint32_t token : {1U, 5U, 3U}) {
            const auto forward = oracle::runtime::execute_qwen35_reference_token(
                model, weights, token, probe, oracle::runtime::RopePosition::text(probe.sequence_length()), false);
            std::size_t best = 0;
            for (std::size_t index = 1; index < forward.logits.size(); ++index) {
                if (forward.logits[index] > forward.logits[best]) best = index;
            }
            require(best == token, "identity fixture sanity check failed for token " + std::to_string(token));
        }
    }

    // Chain: seed token 1 (its real next-token argmax is 1, source-verified
    // above). draft[0] proposes 1 (matches anchor -> ACCEPT). draft[1] proposes 1
    // (matches the real forward of draft[0]=1, whose argmax is 1 -> ACCEPT).
    // draft[2] proposes 5 (does NOT match the real forward of draft[1]=1, whose
    // argmax is 1, not 5 -> REJECT). If verification were off-by-one (draft[i]
    // compared against speculative.steps[i].logits instead of
    // speculative.steps[i-1].logits), draft[1] would instead be checked against
    // the *real forward of itself* (token 1 -> argmax 1) which still matches, but
    // draft[2] would be checked against the real forward of token 5 (argmax 5),
    // which *would* then incorrectly ACCEPT instead of reject -- exactly the
    // kind of chain where an off-by-one alignment bug silently flips a decision.
    oracle::runtime::HybridCache cache_a(model, 20);
    oracle::runtime::HybridCache cache_b(model, 20);
    seed_cache_to_position(model, weights, cache_a, 0, /*seed_token=*/1);
    seed_cache_to_position(model, weights, cache_b, 0, /*seed_token=*/1);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 3;
    chain.seed_position = 0;
    chain.seed_token = 1;
    chain.drafts = {
        make_draft(0, 1, 1, 1, model.vocabulary_size),
        make_draft(1, 2, 1, 1, model.vocabulary_size),
        make_draft(2, 3, 1, 5, model.vocabulary_size),
    };
    const auto anchor_logits = make_logits(model.vocabulary_size, 1);  // real argmax after seed token 1

    const auto lane_a = oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, anchor_logits,
                                                                       cache_a, 3);
    const auto lane_b = oracle::runtime::verify_qwen35_mtp_draft_chain_batched(
        model, weights, chain, anchor_logits, cache_b, 3);

    require(lane_a.accepted == 2, "alignment: sequential lane must accept exactly 2");
    require(lane_a.rejected == 1, "alignment: sequential lane must reject at index 2");
    require(lane_a.first_rejection_index == 2, "alignment: sequential lane rejection index must be 2");
    require(lane_b.accepted == 2, "alignment: batched lane must accept exactly 2");
    require(lane_b.rejected == 1, "alignment: batched lane must reject at index 2");
    require(lane_b.first_rejection_index == 2, "alignment: batched lane rejection index must be 2");
    require(lane_a.canonical_tokens == lane_b.canonical_tokens, "alignment: canonical tokens must match");

    const auto fp_a = oracle::runtime::fingerprint_qwen35_state(cache_a);
    const auto fp_b = oracle::runtime::fingerprint_qwen35_state(cache_b);
    require(fp_a == fp_b, "alignment: final HybridCache fingerprints must match");
}

// --- Rollback-specific coverage (taskblock section 13) ----------------------

// Rollback after zero accepted: reuses case B's chain, but asserts specifically
// on the rollback mechanics (speculative vs after fingerprints must differ, since
// something WAS rolled back; sequence_length must equal boundary + 1, i.e. only
// the one rejection-correcting token survives).
void test_rollback_after_zero_accepted() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 20);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 3;
    chain.seed_position = 10;
    chain.seed_token = 5;
    chain.drafts = {
        make_draft(0, 11, 5, 7, model.vocabulary_size),
        make_draft(1, 12, 7, 8, model.vocabulary_size),
        make_draft(2, 13, 8, 9, model.vocabulary_size),
    };
    const auto anchor_logits = make_logits(model.vocabulary_size, 11);
    seed_cache_to_position(model, weights, cache, chain.seed_position);
    const std::size_t length_before = cache.sequence_length();

    const auto result =
        oracle::runtime::verify_qwen35_mtp_draft_chain_batched(model, weights, chain, anchor_logits, cache, 3);
    require(result.accepted == 0, "zero-accepted: accepted must be 0");
    require(result.speculative_target_tokens_evaluated == 3,
            "zero-accepted: whole chain must be speculatively evaluated regardless of outcome");
    require(result.target_state_before_fingerprint != result.target_speculative_state_fingerprint,
            "zero-accepted: speculative write must have changed state from the boundary");
    require(result.target_speculative_state_fingerprint != result.target_state_after_fingerprint,
            "zero-accepted: rollback must have changed state from the (wrong) speculative write");
    require(cache.sequence_length() == length_before + 1,
            "zero-accepted: only the one rejection-correcting token may survive");
}

void test_rollback_after_one_accepted() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 20);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 4;
    chain.seed_position = 0;
    chain.seed_token = 0;
    chain.drafts = {
        make_draft(0, 1, 0, 0, model.vocabulary_size),
        make_draft(1, 2, 0, 11, model.vocabulary_size),
        make_draft(2, 3, 11, 5, model.vocabulary_size),
        make_draft(3, 4, 5, 6, model.vocabulary_size),
    };
    const auto anchor_logits = make_logits(model.vocabulary_size, 0);
    seed_cache_to_position(model, weights, cache, chain.seed_position);
    const std::size_t length_before = cache.sequence_length();

    const auto result =
        oracle::runtime::verify_qwen35_mtp_draft_chain_batched(model, weights, chain, anchor_logits, cache, 4);
    require(result.accepted == 1, "one-accepted: accepted must be 1");
    require(result.speculative_target_tokens_evaluated == 4,
            "one-accepted: whole chain must be speculatively evaluated regardless of outcome");
    require(cache.sequence_length() == length_before + 2,
            "one-accepted: exactly accepted+1 tokens (the accepted draft plus the rejection "
            "correction) may survive");
}

void test_rollback_after_multiple_accepted() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 20);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 3;
    chain.seed_position = 0;
    chain.seed_token = 0;
    chain.drafts = {
        make_draft(0, 1, 0, 0, model.vocabulary_size),
        make_draft(1, 2, 0, 0, model.vocabulary_size),
        make_draft(2, 3, 0, 11, model.vocabulary_size),
    };
    const auto anchor_logits = make_logits(model.vocabulary_size, 0);
    seed_cache_to_position(model, weights, cache, chain.seed_position);
    const std::size_t length_before = cache.sequence_length();

    const auto result =
        oracle::runtime::verify_qwen35_mtp_draft_chain_batched(model, weights, chain, anchor_logits, cache, 3);
    require(result.accepted == 2, "multiple-accepted: accepted must be 2");
    require(cache.sequence_length() == length_before + 3,
            "multiple-accepted: accepted(2)+1 correction token must survive");
}

// No rollback required on full acceptance: the speculative and after-commit
// fingerprints must be *identical* (proving truncate_to/recommit never ran).
void test_no_rollback_on_full_acceptance() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 20);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 3;
    chain.seed_position = 0;
    chain.seed_token = 0;
    chain.drafts = {
        make_draft(0, 1, 0, 0, model.vocabulary_size),
        make_draft(1, 2, 0, 0, model.vocabulary_size),
        make_draft(2, 3, 0, 0, model.vocabulary_size),
    };
    const auto anchor_logits = make_logits(model.vocabulary_size, 0);
    seed_cache_to_position(model, weights, cache, chain.seed_position);

    const auto result =
        oracle::runtime::verify_qwen35_mtp_draft_chain_batched(model, weights, chain, anchor_logits, cache, 3);
    require(result.full_chain_accepted, "full-acceptance: full_chain_accepted must be true");
    require(result.target_speculative_state_fingerprint == result.target_state_after_fingerprint,
            "full-acceptance: speculative and after-commit fingerprints must be identical "
            "(no rollback/recommit should have run)");
}

// HybridCache::truncate_to specific coverage: no-op, future-length rejection,
// shape-mismatch rejection, determinism, and exact prefix preservation.
void test_hybrid_cache_truncate_to_direct() {
    const auto model = manifest(false, 2);
    oracle::runtime::HybridCache cache(model, 20);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    for (std::uint64_t position = 0; position < 5; ++position) {
        static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
            model, weights, static_cast<std::uint32_t>(position), cache,
            oracle::runtime::RopePosition::text(position), false));
    }
    require(cache.sequence_length() == 5, "truncate_to setup: expected length 5");

    // Truncation to current length is a no-op.
    const auto boundary_now = cache.mark_boundary();
    const auto fp_before_noop = oracle::runtime::fingerprint_qwen35_state(cache);
    cache.truncate_to(boundary_now);
    const auto fp_after_noop = oracle::runtime::fingerprint_qwen35_state(cache);
    require(fp_before_noop == fp_after_noop, "truncate_to(current length) must be a no-op");
    require(cache.sequence_length() == 5, "truncate_to(current length) must not change length");

    // Mark a boundary at length 5, extend further, then attempt to truncate an
    // *earlier*, shorter cache back to this now-future boundary: rejected.
    {
        oracle::runtime::HybridCache short_cache(model, 20);
        for (std::uint64_t position = 0; position < 2; ++position) {
            static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
                model, weights, 0, short_cache, oracle::runtime::RopePosition::text(position), false));
        }
        require_throws(
            [&] { short_cache.truncate_to(boundary_now); },
            "must not grow");
    }

    // Boundary/cache SSM-layer shape mismatch is rejected (different manifest).
    {
        const auto other_model = manifest(false, 3);  // different block_count -> different SSM layer count
        Fixture other_fixture = make_fixture(other_model);
        const auto other_weights = oracle::model::bind_qwen35_weights(other_fixture.views, other_model);
        oracle::runtime::HybridCache other_cache(other_model, 20);
        for (std::uint64_t position = 0; position < 5; ++position) {
            static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
                other_model, other_weights, 0, other_cache, oracle::runtime::RopePosition::text(position),
                false));
        }
        require_throws([&] { other_cache.truncate_to(boundary_now); }, "SSM layer count");
    }

    // Retained prefix (positions 0..2) is bit-identical before/after truncating
    // away positions 3..4, and repeated truncation to the same boundary is
    // deterministic.
    const auto boundary_at_3 = [&] {
        oracle::runtime::HybridCache probe(model, 20);
        for (std::uint64_t position = 0; position < 3; ++position) {
            static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
                model, weights, static_cast<std::uint32_t>(position), probe,
                oracle::runtime::RopePosition::text(position), false));
        }
        return probe.mark_boundary();
    }();

    std::vector<std::uint8_t> attention_row_before;
    for (std::size_t block = 0; block < cache.block_count(); ++block) {
        if (!cache.is_attention_block(block)) continue;
        const auto row = cache.attention(block).key(0, 0);
        attention_row_before.assign(reinterpret_cast<const std::uint8_t*>(row.data()),
                                    reinterpret_cast<const std::uint8_t*>(row.data() + row.size()));
        break;
    }

    cache.truncate_to(boundary_at_3);
    require(cache.sequence_length() == 3, "truncate_to(3): length must become 3");
    const auto fp_after_first_truncate = oracle::runtime::fingerprint_qwen35_state(cache);

    for (std::size_t block = 0; block < cache.block_count(); ++block) {
        if (!cache.is_attention_block(block)) continue;
        const auto row = cache.attention(block).key(0, 0);
        const std::vector<std::uint8_t> after(reinterpret_cast<const std::uint8_t*>(row.data()),
                                              reinterpret_cast<const std::uint8_t*>(row.data() + row.size()));
        require(after == attention_row_before,
                "truncate_to: retained attention prefix (position 0) must be byte-identical");
        break;
    }

    // Repeated truncation to a boundary at the CURRENT length is deterministic
    // (idempotent no-op the second time).
    cache.truncate_to(boundary_at_3);
    const auto fp_after_second_truncate = oracle::runtime::fingerprint_qwen35_state(cache);
    require(fp_after_first_truncate == fp_after_second_truncate,
            "truncate_to: repeated truncation to the same (now-current) boundary must be deterministic");
}

// Unused speculative suffix leaves zero canonical-state residue: after a
// reject-middle-with-suffix batched run, the surviving sequence length accounts
// for exactly accepted+rejected tokens -- the two unused-suffix drafts never
// permanently exist in canonical state at any length the caller can observe.
void test_unused_suffix_leaves_zero_residue() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 20);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 4;
    chain.seed_position = 0;
    chain.seed_token = 0;
    chain.drafts = {
        make_draft(0, 1, 0, 0, model.vocabulary_size),
        make_draft(1, 2, 0, 11, model.vocabulary_size),
        make_draft(2, 3, 11, 5, model.vocabulary_size),
        make_draft(3, 4, 5, 6, model.vocabulary_size),
    };
    const auto anchor_logits = make_logits(model.vocabulary_size, 0);
    seed_cache_to_position(model, weights, cache, chain.seed_position);
    const std::size_t length_before = cache.sequence_length();

    const auto result =
        oracle::runtime::verify_qwen35_mtp_draft_chain_batched(model, weights, chain, anchor_logits, cache, 4);
    require(result.unused_suffix == 2, "unused-suffix: expected 2 unused drafts");
    require(cache.sequence_length() == length_before + result.accepted + result.rejected,
            "unused-suffix: surviving length must equal exactly accepted+rejected, never "
            "including any unused-suffix draft");
    require(result.canonical_tokens.size() == result.accepted + result.rejected,
            "unused-suffix: canonical_tokens must contain exactly accepted+rejected entries");
}

}  // namespace

int main() {
    try {
        test_lane_ab_all_accept();
        test_lane_ab_reject_first();
        test_lane_ab_partial_accept();
        test_lane_ab_reject_middle_with_suffix();
        test_verification_logit_alignment();
        test_rollback_after_zero_accepted();
        test_rollback_after_one_accepted();
        test_rollback_after_multiple_accepted();
        test_no_rollback_on_full_acceptance();
        test_hybrid_cache_truncate_to_direct();
        test_unused_suffix_leaves_zero_residue();
        std::cout << "Phase 2F Slice 5 batched verification tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 2F Slice 5 test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
