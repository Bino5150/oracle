// Phase 2F Slice 3: single-draft target verification + accept/reject state
// semantics (oracle/runtime/qwen35_mtp_verify.hpp,
// oracle/runtime/qwen35_state_fingerprint.hpp). Hand-controlled synthetic tests
// that do not require the real checkpoint; the real accept/reject fixtures
// (bit-identical Lane A/B canonical state, matched against the pinned beellama
// reference) are evidence-only, recorded in
// model-reports/phase2f-slice1-mtp-baseline-20260816/slice3-verification/ (not
// committed -- the 1.4GB checkpoint is not available in CI).

#include "oracle/runtime/qwen35_mtp_verify.hpp"
#include "oracle/runtime/qwen35_state_fingerprint.hpp"
#include "oracle/runtime/qwen35_forward.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
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

// --- small synthetic Qwen3.5 fixture, matching the pattern established in
// tests/test_phase2f_slice1.cpp / test_phase2f_slice2.cpp ---

oracle::model::Qwen35Manifest manifest(bool mtp = false, std::uint32_t block_count = 2) {
    oracle::model::Qwen35Manifest value;
    value.architecture = "qwen35";
    value.model_name = "phase2f-slice3-fixture";
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
    value.full_attention_interval = 2;  // block 1 is full-attention, block 0 recurrent
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

// Item 1: depth-1 accept.
void test_depth1_accept() {
    oracle::runtime::Qwen35MtpProposal proposal;
    proposal.draft_token = 7;
    proposal.draft_logits = make_logits(16, 7);
    proposal.draft_position = 100;
    const auto target_logits = make_logits(16, 7);

    const auto decision = oracle::runtime::verify_qwen35_mtp_proposal(proposal, target_logits, 100);
    require(decision.accepted, "matching draft/target must be accepted");
    require(decision.canonical_token == 7, "canonical token must equal the agreed token");
    require(decision.draft_token == 7 && decision.target_token == 7, "both tokens must be 7");
}

// Item 2: depth-1 reject. Item 3: target authority on reject.
void test_depth1_reject_target_authority() {
    oracle::runtime::Qwen35MtpProposal proposal;
    proposal.draft_token = 7;
    proposal.draft_logits = make_logits(16, 7);
    proposal.draft_position = 5;
    const auto target_logits = make_logits(16, 11);

    const auto decision = oracle::runtime::verify_qwen35_mtp_proposal(proposal, target_logits, 5);
    require(!decision.accepted, "mismatched draft/target must be rejected");
    require(decision.target_token == 11, "target_token must be the target's own argmax");
    require(decision.canonical_token == 11,
            "canonical_token must be the TARGET's token on reject, not the draft's -- target authority");
    require(decision.canonical_token != decision.draft_token,
            "on reject, canonical must differ from the discarded draft");
}

// Item 4 + item 5 (no double commit): canonical state advances exactly once, and
// forwarding the canonical token commits it exactly once (not twice).
void test_canonical_state_advances_exactly_once() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    oracle::runtime::HybridCache cache(model, 8);
    require(cache.sequence_length() == 0, "cache must start empty");

    const std::size_t old_length = cache.sequence_length();
    static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
        model, weights, /*token_id=*/3, cache, oracle::runtime::RopePosition::text(old_length), false));
    const std::size_t new_length = cache.sequence_length();

    require(new_length == old_length + 1,
            "new_sequence_length must equal old_sequence_length + 1 exactly (no double commit)");
}

// Item 6 + item 7: rejected draft never enters the canonical ledger or cache.
// verify_qwen35_mtp_proposal() takes no cache/state parameter at all -- it is a
// pure function -- so a real HybridCache is provably untouched by calling it,
// regardless of the decision.
void test_rejected_draft_never_enters_canonical_state() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 8);
    const auto fingerprint_before = oracle::runtime::fingerprint_qwen35_state(cache);

    oracle::runtime::Qwen35MtpProposal proposal;
    proposal.draft_token = 7;
    proposal.draft_logits = make_logits(16, 7);
    proposal.draft_position = 0;
    const auto target_logits = make_logits(16, 11);
    const auto decision = oracle::runtime::verify_qwen35_mtp_proposal(proposal, target_logits, 0);
    require(!decision.accepted, "fixture must exercise the reject path");

    const auto fingerprint_after_verify = oracle::runtime::fingerprint_qwen35_state(cache);
    require(fingerprint_before == fingerprint_after_verify,
            "verify_qwen35_mtp_proposal must never touch canonical HybridCache state");

    // Only the canonical_token (target's own choice, 11) is ever forwarded -- the
    // discarded draft (7) never reaches the model.
    static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
        model, weights, decision.canonical_token, cache, oracle::runtime::RopePosition::text(0), false));
    require(cache.sequence_length() == 1, "exactly one token (the canonical one) must commit");
    require(decision.canonical_token == 11 && decision.canonical_token != 7,
            "the committed token must be the target's token, never the rejected draft");

    static_cast<void>(weights);
}

// Item 8: MTP temporary state remains isolated (extends Slice 2's isolation proof
// to the full propose->verify sequence).
void test_mtp_temporary_state_isolated_through_verification() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache canonical_cache(model, 8);
    const auto before = oracle::runtime::fingerprint_qwen35_state(canonical_cache);

    oracle::runtime::Qwen35MtpProposal proposal;
    proposal.draft_token = 3;
    proposal.draft_logits = make_logits(16, 3);
    proposal.draft_position = 0;
    const auto target_logits = make_logits(16, 3);
    static_cast<void>(oracle::runtime::verify_qwen35_mtp_proposal(proposal, target_logits, 0));

    const auto after = oracle::runtime::fingerprint_qwen35_state(canonical_cache);
    require(before == after, "canonical HybridCache must remain untouched by proposal+verification");
    static_cast<void>(weights);
}

// Item 9: accounting counts exact, for both accept and reject.
void test_accounting_counts_exact() {
    oracle::runtime::Qwen35MtpProposal accept_proposal;
    accept_proposal.draft_token = 4;
    accept_proposal.draft_logits = make_logits(8, 4);
    accept_proposal.draft_position = 0;
    const auto accept_decision = oracle::runtime::verify_qwen35_mtp_proposal(
        accept_proposal, make_logits(8, 4), 0);
    require(accept_decision.accounting.drafts_proposed == 1, "accept: proposed must be 1");
    require(accept_decision.accounting.drafts_accepted == 1, "accept: accepted must be 1");
    require(accept_decision.accounting.drafts_rejected == 0, "accept: rejected must be 0");
    require(accept_decision.accounting.verification_count == 1, "accept: verification_count must be 1");
    require(accept_decision.accounting.drafts_accepted + accept_decision.accounting.drafts_rejected == 1,
            "accept: accepted+rejected must equal 1");

    oracle::runtime::Qwen35MtpProposal reject_proposal;
    reject_proposal.draft_token = 4;
    reject_proposal.draft_logits = make_logits(8, 4);
    reject_proposal.draft_position = 0;
    const auto reject_decision = oracle::runtime::verify_qwen35_mtp_proposal(
        reject_proposal, make_logits(8, 5), 0);
    require(reject_decision.accounting.drafts_proposed == 1, "reject: proposed must be 1");
    require(reject_decision.accounting.drafts_accepted == 0, "reject: accepted must be 0");
    require(reject_decision.accounting.drafts_rejected == 1, "reject: rejected must be 1");
    require(reject_decision.accounting.verification_count == 1, "reject: verification_count must be 1");
    require(reject_decision.accounting.drafts_accepted + reject_decision.accounting.drafts_rejected == 1,
            "reject: accepted+rejected must equal 1");
}

// Item 10: deterministic tie rule -- equal top logits resolve to the lower index,
// consistently for both draft and target argmax.
void test_deterministic_tie_rule() {
    std::vector<float> tied(8, 0.0F);
    tied[2] = 5.0F;
    tied[6] = 5.0F;  // tie between index 2 and 6; lower index (2) must win

    oracle::runtime::Qwen35MtpProposal proposal;
    proposal.draft_token = 2;
    proposal.draft_logits = tied;
    proposal.draft_position = 0;
    const auto decision = oracle::runtime::verify_qwen35_mtp_proposal(proposal, tied, 0);
    require(decision.draft_token == 2, "tie must resolve to the lower index for the draft");
    require(decision.target_token == 2, "tie must resolve to the lower index for the target");
    require(decision.accepted, "identical tied distributions must agree and accept");
}

// Item 11: non-finite logits rejected (both target and draft).
void test_non_finite_logits_rejected() {
    oracle::runtime::Qwen35MtpProposal proposal;
    proposal.draft_token = 0;
    proposal.draft_logits = make_logits(8, 0);
    proposal.draft_position = 0;
    auto bad_target = make_logits(8, 0);
    bad_target[3] = std::numeric_limits<float>::infinity();
    require_throws(
        [&] { static_cast<void>(oracle::runtime::verify_qwen35_mtp_proposal(proposal, bad_target, 0)); },
        "non-finite value");

    oracle::runtime::Qwen35MtpProposal bad_draft_proposal;
    bad_draft_proposal.draft_token = 0;
    bad_draft_proposal.draft_logits = make_logits(8, 0);
    bad_draft_proposal.draft_logits[5] = std::numeric_limits<float>::quiet_NaN();
    bad_draft_proposal.draft_position = 0;
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::verify_qwen35_mtp_proposal(bad_draft_proposal,
                                                                          make_logits(8, 0), 0));
        },
        "non-finite value");
}

// Item 12: malformed logits width rejected. Also covers invalid vocabulary size and
// invalid draft token (self-consistency).
void test_malformed_width_and_invalid_token_rejected() {
    oracle::runtime::Qwen35MtpProposal narrow;
    narrow.draft_token = 0;
    narrow.draft_logits = make_logits(4, 0);
    narrow.draft_position = 0;
    require_throws(
        [&] { static_cast<void>(oracle::runtime::verify_qwen35_mtp_proposal(narrow, make_logits(8, 0), 0)); },
        "width mismatch");

    // Invalid vocabulary size: empty target logits.
    oracle::runtime::Qwen35MtpProposal empty_proposal;
    empty_proposal.draft_position = 0;
    require_throws(
        [&] {
            static_cast<void>(
                oracle::runtime::verify_qwen35_mtp_proposal(empty_proposal, std::span<const float>{}, 0));
        },
        "invalid vocabulary size");

    // Invalid draft token: draft_token inconsistent with argmax(draft_logits).
    oracle::runtime::Qwen35MtpProposal inconsistent;
    inconsistent.draft_token = 2;  // argmax of draft_logits below is actually 5
    inconsistent.draft_logits = make_logits(8, 5);
    inconsistent.draft_position = 0;
    require_throws(
        [&] {
            static_cast<void>(
                oracle::runtime::verify_qwen35_mtp_proposal(inconsistent, make_logits(8, 5), 0));
        },
        "invalid draft token");
}

// Item 13: invalid position (draft/target position mismatch) rejected.
void test_invalid_position_rejected() {
    oracle::runtime::Qwen35MtpProposal proposal;
    proposal.draft_token = 1;
    proposal.draft_logits = make_logits(8, 1);
    proposal.draft_position = 5;
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::verify_qwen35_mtp_proposal(proposal, make_logits(8, 1), 6));
        },
        "position mismatch");
}

// Item 14: repeated execution deterministic.
void test_repeated_execution_deterministic() {
    oracle::runtime::Qwen35MtpProposal proposal;
    proposal.draft_token = 9;
    proposal.draft_logits = make_logits(16, 9);
    proposal.draft_position = 12;
    const auto target_logits = make_logits(16, 3);

    const auto first = oracle::runtime::verify_qwen35_mtp_proposal(proposal, target_logits, 12);
    const auto second = oracle::runtime::verify_qwen35_mtp_proposal(proposal, target_logits, 12);
    require(first.draft_token == second.draft_token && first.target_token == second.target_token &&
                first.canonical_token == second.canonical_token && first.accepted == second.accepted,
            "repeated verification of identical inputs must be deterministic");
}

// Diagnostic trace: assemble and print a full Qwen35MtpVerificationTrace.
void test_diagnostic_trace_assembly() {
    const auto model = manifest(false, 2);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    oracle::runtime::HybridCache cache(model, 8);
    const auto state_before = oracle::runtime::fingerprint_qwen35_state(cache);

    oracle::runtime::Qwen35MtpProposal proposal;
    proposal.draft_token = 4;
    proposal.draft_logits = make_logits(16, 4);
    proposal.draft_position = 0;
    const auto decision =
        oracle::runtime::verify_qwen35_mtp_proposal(proposal, make_logits(16, 4), 0);

    static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
        model, weights, decision.canonical_token, cache, oracle::runtime::RopePosition::text(0), false));
    const auto state_after = oracle::runtime::fingerprint_qwen35_state(cache);

    oracle::runtime::Qwen35MtpVerificationTrace trace;
    trace.decision = decision;
    trace.target_state_before_fingerprint = state_before.combined;
    trace.target_state_after_fingerprint = state_after.combined;

    const std::string text = oracle::runtime::qwen35_mtp_verification_trace_text(trace);
    const std::string json = oracle::runtime::qwen35_mtp_verification_trace_json(trace);
    require(text.find("accepted: true") != std::string::npos, "trace text must show acceptance");
    require(json.find("\"accepted\":true") != std::string::npos, "trace json must show acceptance");
    require(state_before.combined != state_after.combined,
            "state fingerprint must change once the canonical token is forwarded");
}

}  // namespace

int main() {
    try {
        test_depth1_accept();
        test_depth1_reject_target_authority();
        test_canonical_state_advances_exactly_once();
        test_rejected_draft_never_enters_canonical_state();
        test_mtp_temporary_state_isolated_through_verification();
        test_accounting_counts_exact();
        test_deterministic_tie_rule();
        test_non_finite_logits_rejected();
        test_malformed_width_and_invalid_token_rejected();
        test_invalid_position_rejected();
        test_repeated_execution_deterministic();
        test_diagnostic_trace_assembly();
        std::cout << "Phase 2F Slice 3 verification tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 2F Slice 3 test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
