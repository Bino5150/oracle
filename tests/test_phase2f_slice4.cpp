// Phase 2F Slice 4: multi-draft MTP chaining + partial acceptance + sequential
// target verification (oracle/runtime/qwen35_mtp_chain.hpp,
// oracle/runtime/qwen35_mtp_chain_verify.hpp). Hand-controlled synthetic tests that
// do not require the real checkpoint; the real depth-3 chain capture (matched
// against the pinned beellama reference) and real full-accept/partial-reject
// fixtures are evidence-only, recorded in
// model-reports/phase2f-slice1-mtp-baseline-20260816/slice4-chain/ (not
// committed -- the 1.4GB checkpoint is not available in CI).
//
// verify_qwen35_mtp_draft_chain() genuinely calls execute_qwen35_reference_token()
// between drafts (depth>1 verification requires a fresh target forward to obtain
// the next draft's comparison logits -- see the header comment), so unlike Slice
// 3's pure verify_qwen35_mtp_proposal(), these tests cannot simply inject arbitrary
// "target" logits for every step. Instead they use a fixture whose every weight
// (including every RMSNorm weight) is zero-initialized: RMSNorm's output is scaled
// by its weight, so with an all-zero weight every norm output is exactly zero
// regardless of input, which cascades through the whole backbone (zero attention
// Q/K/V, zero FFN activation, zero final-norm output, zero logits) for *any* input
// token or position. The target's real argmax is therefore deterministically 0 at
// every step, for any history -- giving full, honest control over multi-step
// accept/reject scenarios without mocking the forward pass.

#include "oracle/runtime/qwen35_mtp_chain.hpp"
#include "oracle/runtime/qwen35_mtp_chain_verify.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/qwen35_forward.hpp"
#include "oracle/runtime/qwen35_state_fingerprint.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/model/ggml_type.hpp"

#include <cstddef>
#include <cstdint>
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
// established in tests/test_phase2f_slice3.cpp ---

oracle::model::Qwen35Manifest manifest(bool mtp = false, std::uint32_t block_count = 2) {
    oracle::model::Qwen35Manifest value;
    value.architecture = "qwen35";
    value.model_name = "phase2f-slice4-fixture";
    value.backbone_block_count = block_count;
    value.nextn_predict_layers = mtp ? 1U : 0U;
    value.total_block_count = value.backbone_block_count + value.nextn_predict_layers;
    value.context_length = 1024;
    value.embedding_length = 64;
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
    value.vocabulary_size = 32;
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

std::vector<float> make_logits(std::size_t vocab, std::size_t argmax_index, float peak = 10.0F) {
    std::vector<float> logits(vocab, 0.0F);
    logits[argmax_index] = peak;
    return logits;
}

// verify_qwen35_mtp_draft_chain requires cache.sequence_length() == seed_position + 1
// on entry (the anchor token must already be canonical -- its own logits are what
// produced anchor_target_logits in the first place). Forwards dummy token 0
// through every position up to and including seed_position.
void seed_cache_to_position(const oracle::model::Qwen35Manifest& model,
                            const oracle::model::Qwen35Weights& weights,
                            oracle::runtime::HybridCache& cache, std::uint64_t seed_position) {
    for (std::uint64_t position = 0; position <= seed_position; ++position) {
        static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
            model, weights, 0, cache, oracle::runtime::RopePosition::text(position), false));
    }
}

// A hand-built draft with a chosen draft_token, matching the manifest's vocab
// width, at the given chain position.
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

// Item A: ALL ACCEPT. All-zero-weight fixture always argmaxes to 0 at every
// position for any history, so a chain proposing 0,0,0 is genuinely, honestly
// all-accepted by a real sequential target forward at each step -- not mocked.
void test_all_accept() {
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
    const auto anchor_logits = make_logits(model.vocabulary_size, 0);  // target's real choice is 0
    seed_cache_to_position(model, weights, cache, chain.seed_position);
    const std::size_t length_before = cache.sequence_length();

    const auto accounting =
        oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, anchor_logits, cache, 3);
    require(accounting.proposed == 3, "A: proposed must be 3");
    require(accounting.accepted == 3, "A: accepted must be 3");
    require(accounting.rejected == 0, "A: rejected must be 0");
    require(accounting.unused_suffix == 0, "A: unused_suffix must be 0");
    require(accounting.verification_count == 3, "A: verification_count must be 3");
    require(accounting.first_rejection_index == -1, "A: no rejection index");
    require(accounting.full_chain_accepted, "A: full_chain_accepted must be true");
    require(accounting.canonical_tokens == std::vector<std::uint32_t>({0, 0, 0}),
            "A: canonical tokens must be [0,0,0]");
    require(cache.sequence_length() == length_before + 3,
            "A: canonical state must advance by exactly 3 (one per accepted draft)");
}

// Item B: REJECT FIRST. anchor_target_logits is fully caller-controlled, so the
// very first comparison can be made to disagree deterministically.
void test_reject_first() {
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
    const auto anchor_logits = make_logits(model.vocabulary_size, 11);  // target's real choice is 11, not 7
    seed_cache_to_position(model, weights, cache, chain.seed_position);
    const std::size_t length_before = cache.sequence_length();

    const auto accounting =
        oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, anchor_logits, cache, 3);
    require(accounting.proposed == 3, "B: proposed must be 3");
    require(accounting.accepted == 0, "B: accepted must be 0");
    require(accounting.rejected == 1, "B: rejected must be 1");
    require(accounting.unused_suffix == 2, "B: unused_suffix must be 2");
    require(accounting.verification_count == 1, "B: verification_count must be 1 (stop at first)");
    require(accounting.first_rejection_index == 0, "B: rejection at index 0");
    require(!accounting.full_chain_accepted, "B: full_chain_accepted must be false");
    require(accounting.canonical_tokens == std::vector<std::uint32_t>({11}),
            "B: canonical tokens must be [11]");
    require(cache.sequence_length() == length_before + 1,
            "B: canonical state must advance by exactly 1");
}

// Item C: PARTIAL ACCEPT. draft[0]=0 and draft[1]=0 match the fixture's natural
// argmax (0) at every step; draft[2]=11 does not, so it rejects.
void test_partial_accept() {
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

    const auto accounting =
        oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, anchor_logits, cache, 3);
    require(accounting.proposed == 3, "C: proposed must be 3");
    require(accounting.accepted == 2, "C: accepted must be 2");
    require(accounting.rejected == 1, "C: rejected must be 1");
    require(accounting.unused_suffix == 0, "C: unused_suffix must be 0");
    require(accounting.verification_count == 3, "C: verification_count must be 3");
    require(accounting.first_rejection_index == 2, "C: rejection at index 2");
    require(!accounting.full_chain_accepted, "C: full_chain_accepted must be false");
    require(accounting.canonical_tokens == std::vector<std::uint32_t>({0, 0, 0}),
            "C: canonical tokens must be [0,0,target's real choice(0)]");
    require(cache.sequence_length() == length_before + 3,
            "C: canonical state must advance by exactly 3 (accepted+rejected)");
    require(accounting.accepted + accounting.rejected <= accounting.proposed,
            "C: accepted+rejected must not exceed proposed");
}

// Item D: REJECT MIDDLE WITH SUFFIX. draft[0]=0 accepts; draft[1]=11 rejects;
// draft[2],draft[3] are discarded unused.
void test_reject_middle_with_suffix() {
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

    const auto accounting =
        oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, anchor_logits, cache, 4);
    require(accounting.proposed == 4, "D: proposed must be 4");
    require(accounting.accepted == 1, "D: accepted must be 1");
    require(accounting.rejected == 1, "D: rejected must be 1");
    require(accounting.unused_suffix == 2, "D: unused_suffix must be 2");
    require(accounting.verification_count == 2, "D: verification_count must be 2");
    require(accounting.first_rejection_index == 1, "D: rejection at index 1");
    require(accounting.canonical_tokens == std::vector<std::uint32_t>({0, 0}),
            "D: canonical tokens must be [0, target's real choice(0)]");
    require(cache.sequence_length() == length_before + 2,
            "D: canonical state must advance by exactly 2");
}

// Canonical state equivalence, Lane A/B, for chained/partial-acceptance cases
// (task 53 -- extends Slice 3's single-draft Lane A/B finding to multi-draft
// chains). Lane A runs verify_qwen35_mtp_draft_chain, which advances a HybridCache
// by forwarding each verified draft's canonical_token exactly once. Lane B is a
// second, independently constructed HybridCache seeded identically, then manually
// forwarded through the SAME canonical_tokens sequence returned by Lane A, calling
// execute_qwen35_reference_token directly -- with no chain verification involved at
// all. If chain verification does anything to canonical state beyond "forward each
// decided token once, in order," the two final fingerprints will diverge.
void test_canonical_state_equivalence_chained_full_accept() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    oracle::runtime::HybridCache cache_a(model, 20);
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
    seed_cache_to_position(model, weights, cache_a, chain.seed_position);
    const auto accounting =
        oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, anchor_logits, cache_a, 3);
    require(accounting.full_chain_accepted, "Lane A/B full-accept: chain must fully accept");

    oracle::runtime::HybridCache cache_b(model, 20);
    seed_cache_to_position(model, weights, cache_b, chain.seed_position);
    std::uint64_t position = chain.seed_position;
    for (const std::uint32_t canonical_token : accounting.canonical_tokens) {
        ++position;
        static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
            model, weights, canonical_token, cache_b, oracle::runtime::RopePosition::text(position), false));
    }

    require(cache_a.sequence_length() == cache_b.sequence_length(),
            "Lane A/B full-accept: sequence_length must match");
    const auto fp_a = oracle::runtime::fingerprint_qwen35_state(cache_a);
    const auto fp_b = oracle::runtime::fingerprint_qwen35_state(cache_b);
    require(fp_a == fp_b, "Lane A/B full-accept: state fingerprints must be bit-identical");
}

void test_canonical_state_equivalence_chained_partial_reject() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    oracle::runtime::HybridCache cache_a(model, 20);
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
    seed_cache_to_position(model, weights, cache_a, chain.seed_position);
    const auto accounting =
        oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, anchor_logits, cache_a, 4);
    require(accounting.accepted == 1 && accounting.rejected == 1,
            "Lane A/B partial-reject: expected accepted=1 rejected=1");

    oracle::runtime::HybridCache cache_b(model, 20);
    seed_cache_to_position(model, weights, cache_b, chain.seed_position);
    std::uint64_t position = chain.seed_position;
    for (const std::uint32_t canonical_token : accounting.canonical_tokens) {
        ++position;
        static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
            model, weights, canonical_token, cache_b, oracle::runtime::RopePosition::text(position), false));
    }

    require(cache_a.sequence_length() == cache_b.sequence_length(),
            "Lane A/B partial-reject: sequence_length must match");
    const auto fp_a = oracle::runtime::fingerprint_qwen35_state(cache_a);
    const auto fp_b = oracle::runtime::fingerprint_qwen35_state(cache_b);
    require(fp_a == fp_b, "Lane A/B partial-reject: state fingerprints must be bit-identical");

    // No unused-suffix drafts (5, 6) ever reached canonical state: only two
    // positions were ever forwarded (accounting.canonical_tokens.size() == 2), and
    // Lane B's manual replay used only those two tokens.
    require(accounting.canonical_tokens.size() == 2,
            "Lane A/B partial-reject: exactly 2 tokens should have reached canonical state");
}

// Depth zero rejected.
void test_depth_zero_rejected() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 20);

    oracle::runtime::Qwen35MtpDraftChain empty_chain;
    empty_chain.requested_depth = 0;
    empty_chain.seed_position = 0;
    empty_chain.seed_token = 0;
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::verify_qwen35_mtp_draft_chain(
                model, weights, empty_chain, make_logits(model.vocabulary_size, 0), cache, 3));
        },
        "depth zero");
}

// Depth above configured maximum rejected.
void test_depth_above_configured_maximum_rejected() {
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
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::verify_qwen35_mtp_draft_chain(
                model, weights, chain, make_logits(model.vocabulary_size, 0), cache,
                /*configured_max_depth=*/2));
        },
        "configured maximum depth");
}

// Invalid / non-contiguous proposal positions rejected.
void test_invalid_and_noncontiguous_positions_rejected() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 20);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 2;
    chain.seed_position = 0;
    chain.seed_token = 0;
    chain.drafts = {
        make_draft(0, 1, 0, 0, model.vocabulary_size),
        make_draft(1, 3 /* should be 2 */, 0, 0, model.vocabulary_size),
    };
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::verify_qwen35_mtp_draft_chain(
                model, weights, chain, make_logits(model.vocabulary_size, 0), cache, 3));
        },
        "non-contiguous");
}

// Invalid token IDs rejected.
void test_invalid_token_id_rejected() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 20);

    oracle::runtime::Qwen35MtpDraft bad_draft;
    bad_draft.draft_index = 0;
    bad_draft.position = 1;
    bad_draft.input_token = 0;
    bad_draft.draft_token = 9999;  // exceeds vocabulary_size (32)
    bad_draft.logits = make_logits(model.vocabulary_size, 0);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 1;
    chain.seed_position = 0;
    chain.seed_token = 0;
    chain.drafts = {bad_draft};
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::verify_qwen35_mtp_draft_chain(
                model, weights, chain, make_logits(model.vocabulary_size, 0), cache, 3));
        },
        "invalid draft token");
}

// Non-finite logits (anchor and draft) rejected.
void test_non_finite_logits_rejected() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 20);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 1;
    chain.seed_position = 0;
    chain.seed_token = 0;
    chain.drafts = {make_draft(0, 1, 0, 0, model.vocabulary_size)};

    auto bad_anchor = make_logits(model.vocabulary_size, 0);
    bad_anchor[5] = std::numeric_limits<float>::infinity();
    require_throws(
        [&] {
            static_cast<void>(
                oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, bad_anchor, cache, 3));
        },
        "non-finite value");

    oracle::runtime::HybridCache cache2(model, 20);
    oracle::runtime::Qwen35MtpDraftChain bad_chain = chain;
    bad_chain.drafts[0].logits[3] = std::numeric_limits<float>::quiet_NaN();
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::verify_qwen35_mtp_draft_chain(
                model, weights, bad_chain, make_logits(model.vocabulary_size, 0), cache2, 3));
        },
        "non-finite value");
}

// Deterministic repeated execution: same chain, same starting state shape, same
// accounting result.
void test_deterministic_repeated_execution() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    oracle::runtime::Qwen35MtpDraftChain chain;
    chain.requested_depth = 2;
    chain.seed_position = 0;
    chain.seed_token = 0;
    chain.drafts = {
        make_draft(0, 1, 0, 0, model.vocabulary_size),
        make_draft(1, 2, 0, 0, model.vocabulary_size),
    };
    const auto anchor_logits = make_logits(model.vocabulary_size, 0);

    oracle::runtime::HybridCache cache_a(model, 20);
    oracle::runtime::HybridCache cache_b(model, 20);
    seed_cache_to_position(model, weights, cache_a, chain.seed_position);
    seed_cache_to_position(model, weights, cache_b, chain.seed_position);
    const auto first = oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, anchor_logits, cache_a, 3);
    const auto second = oracle::runtime::verify_qwen35_mtp_draft_chain(model, weights, chain, anchor_logits, cache_b, 3);
    require(first.accepted == second.accepted && first.rejected == second.rejected &&
                first.canonical_tokens == second.canonical_tokens,
            "repeated verification of identical chains must be deterministic");
}

// generate_qwen35_mtp_draft_chain: depth 1/2/3 supported, chain metadata correct.
void test_chain_generation_depths() {
    const auto model = manifest(true, 2);
    Fixture fixture = make_fixture(model);
    const std::uint32_t block = model.backbone_block_count;
    const std::string prefix = "blk." + std::to_string(block) + ".";
    fixture.add(prefix + "attn_norm.weight", {model.embedding_length});
    fixture.add(prefix + "post_attention_norm.weight", {model.embedding_length});
    fixture.add(prefix + "attn_q.weight",
               {model.embedding_length, model.attention_key_length * model.attention_head_count * 2U});
    fixture.add(prefix + "attn_k.weight",
               {model.embedding_length, model.attention_key_length * model.attention_head_count_kv});
    fixture.add(prefix + "attn_v.weight",
               {model.embedding_length, model.attention_value_length * model.attention_head_count_kv});
    fixture.add(prefix + "attn_output.weight",
               {model.attention_key_length * model.attention_head_count, model.embedding_length});
    fixture.add(prefix + "attn_q_norm.weight", {model.attention_key_length});
    fixture.add(prefix + "attn_k_norm.weight", {model.attention_key_length});
    fixture.add(prefix + "ffn_gate.weight", {model.embedding_length, model.feed_forward_length});
    fixture.add(prefix + "ffn_down.weight", {model.feed_forward_length, model.embedding_length});
    fixture.add(prefix + "ffn_up.weight", {model.embedding_length, model.feed_forward_length});
    const std::string nextn = prefix + "nextn.";
    fixture.add(nextn + "eh_proj.weight", {model.embedding_length * 2U, model.embedding_length});
    fixture.add(nextn + "enorm.weight", {model.embedding_length});
    fixture.add(nextn + "hnorm.weight", {model.embedding_length});
    fixture.add(nextn + "shared_head_norm.weight", {model.embedding_length});

    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(weights.mtp.has_value(), "fixture must bind MTP weights");

    const std::vector<float> seed_hidden(model.embedding_length, 0.0F);
    for (const std::uint32_t depth : {1U, 2U, 3U}) {
        const auto chain = oracle::runtime::generate_qwen35_mtp_draft_chain(
            model, weights, /*seed_token=*/0, seed_hidden, /*seed_position=*/10, depth, false);
        require(chain.drafts.size() == depth, "chain must contain exactly `depth` drafts");
        require(chain.requested_depth == depth, "requested_depth must be recorded");
        require(chain.seed_position == 10, "seed_position must be recorded");
        for (std::uint32_t i = 0; i < depth; ++i) {
            require(chain.drafts[i].draft_index == i, "draft_index must be sequential");
            require(chain.drafts[i].position == 10 + i + 1, "position must be seed_position+i+1");
        }
        // D_k's input token must be D_{k-1}'s draft token (or the seed for D0) --
        // proves chaining, not independent re-seeding.
        require(chain.drafts[0].input_token == 0, "D0 input token must be the seed token");
        for (std::uint32_t i = 1; i < depth; ++i) {
            require(chain.drafts[i].input_token == chain.drafts[i - 1].draft_token,
                    "D_k input token must equal D_{k-1}'s draft token");
        }
    }

    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::generate_qwen35_mtp_draft_chain(
                model, weights, 0, seed_hidden, 10, /*max_depth=*/0, false));
        },
        "max_depth >= 1");
}

}  // namespace

int main() {
    try {
        test_all_accept();
        test_reject_first();
        test_partial_accept();
        test_reject_middle_with_suffix();
        test_canonical_state_equivalence_chained_full_accept();
        test_canonical_state_equivalence_chained_partial_reject();
        test_depth_zero_rejected();
        test_depth_above_configured_maximum_rejected();
        test_invalid_and_noncontiguous_positions_rejected();
        test_invalid_token_id_rejected();
        test_non_finite_logits_rejected();
        test_deterministic_repeated_execution();
        test_chain_generation_depths();
        std::cout << "Phase 2F Slice 4 multi-draft chain tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 2F Slice 4 test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
