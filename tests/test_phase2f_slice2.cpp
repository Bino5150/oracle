// Phase 2F Slice 2: one real Qwen3.5 NextN/MTP prediction through Oracle's
// decoded-F32 scalar reference path (oracle/runtime/qwen35_mtp.hpp,
// execute_qwen35_reference_mtp). These are hand-controlled synthetic tests that do
// not require the real checkpoint; the real-checkpoint validation (stage-by-stage
// comparison against a real graph_mtp capture, argmax agreement, top-N overlap) is
// evidence-only and recorded in
// model-reports/phase2f-slice1-mtp-baseline-20260816/slice2-mtp-forward-capture/
// (not committed -- the 1.4GB checkpoint is not available in CI).

#include "oracle/runtime/qwen35_mtp.hpp"
#include "oracle/backend/cpu/quantized_reference.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/model/ggml_type.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

void require_near(float actual, float expected, float tolerance, std::string_view message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(message) + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
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

oracle::model::Qwen35Manifest manifest(bool mtp = false, std::uint32_t block_count = 4) {
    oracle::model::Qwen35Manifest value;
    value.architecture = "qwen35";
    value.model_name = "phase2f-slice2-fixture";
    value.backbone_block_count = block_count;
    value.nextn_predict_layers = mtp ? 1U : 0U;
    value.total_block_count = value.backbone_block_count + value.nextn_predict_layers;
    value.context_length = 1024;
    value.embedding_length = 256;
    value.feed_forward_length = 512;
    value.attention_head_count = 4;
    value.attention_head_count_kv = 2;
    value.attention_key_length = 64;
    value.attention_value_length = 64;
    value.attention_rms_epsilon = 1.0e-6F;
    value.rope_dimension_count = 64;
    value.rope_dimension_sections = {11, 11, 10, 0};
    value.rope_frequency_base = 10000000.0F;
    value.ssm_convolution_kernel = 4;
    value.ssm_state_size = 32;
    value.ssm_group_count = 2;
    value.ssm_time_step_rank = 4;
    value.ssm_inner_size = 128;
    value.full_attention_interval = 4;
    value.vocabulary_size = 256;
    return value;
}

struct Fixture {
    std::vector<oracle::model::GgufTensorInfo> infos;
    std::vector<std::vector<std::byte>> storage;
    std::vector<oracle::model::GgufTensorView> views;

    Fixture() {
        infos.reserve(512);
        storage.reserve(512);
        views.reserve(512);
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

    [[nodiscard]] std::size_t find(std::string_view name) const {
        for (std::size_t index = 0; index < infos.size(); ++index) {
            if (infos[index].name == name) return index;
        }
        throw std::runtime_error("fixture tensor not found: " + std::string(name));
    }

    // Writes explicit F32 values (raw little-endian bytes, matching type=0) into a
    // fixture tensor's storage, for hand-computable known-answer tests. Requires the
    // value count to match the tensor exactly.
    void set_f32(std::string_view name, std::span<const float> values) {
        const std::size_t index = find(name);
        require(storage[index].size() == values.size() * sizeof(float),
                "set_f32 value count does not match tensor byte size");
        std::memcpy(storage[index].data(), values.data(), storage[index].size());
    }

    // Writes explicit F32 values into one row of a multi-row (matrix) tensor, e.g.
    // one token's row within token_embd.weight, leaving all other rows untouched
    // (zero-initialized).
    void set_f32_row(std::string_view name, std::size_t row_index, std::span<const float> values) {
        const std::size_t index = find(name);
        const std::size_t row_bytes = values.size() * sizeof(float);
        const std::size_t offset = row_index * row_bytes;
        require(offset + row_bytes <= storage[index].size(), "set_f32_row is out of bounds");
        std::memcpy(storage[index].data() + offset, values.data(), row_bytes);
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
    if (model.has_mtp()) {
        const std::uint32_t block = model.backbone_block_count;
        add_common(fixture, model, block);
        add_attention(fixture, model, block);
        const std::string nextn = "blk." + std::to_string(block) + ".nextn.";
        fixture.add(nextn + "eh_proj.weight", {model.embedding_length * 2U, model.embedding_length});
        fixture.add(nextn + "enorm.weight", {model.embedding_length});
        fixture.add(nextn + "hnorm.weight", {model.embedding_length});
        fixture.add(nextn + "shared_head_norm.weight", {model.embedding_length});
    }
    return fixture;
}

oracle::runtime::Qwen35MtpInput valid_input(const oracle::model::Qwen35Manifest& model) {
    oracle::runtime::Qwen35MtpInput input;
    input.candidate_token_id = 1;
    input.target_hidden_state.assign(model.embedding_length, 0.1F);
    input.position = oracle::runtime::RopePosition::text(0);
    return input;
}

// Item 1: MTP absent -> execution rejected.
void test_mtp_absent_rejected() {
    const auto model = manifest(false);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::execute_qwen35_reference_mtp(model, weights, valid_input(model)));
        },
        "requires manifest.has_mtp()");
}

// Item 2: incomplete MTP weights rejected (tests execute_qwen35_reference_mtp's own
// validation directly, independent of the binder's already-tested rejection paths).
void test_incomplete_mtp_weights_rejected() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(weights.mtp.has_value(), "fixture must bind MTP weights");
    weights.mtp->attention.query = nullptr;
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::execute_qwen35_reference_mtp(model, weights, valid_input(model)));
        },
        "MTP attention query");
}

// Item 3: input width mismatch.
void test_input_width_mismatch_rejected() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    auto input = valid_input(model);
    input.target_hidden_state.resize(model.embedding_length - 1);
    require_throws(
        [&] { static_cast<void>(oracle::runtime::execute_qwen35_reference_mtp(model, weights, input)); },
        "target hidden-state width mismatch");
}

// Item 4: invalid position.
void test_invalid_position_rejected() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    auto input = valid_input(model);
    input.position = oracle::runtime::RopePosition::text(model.context_length);
    require_throws(
        [&] { static_cast<void>(oracle::runtime::execute_qwen35_reference_mtp(model, weights, input)); },
        "position exceeds the manifest context length");
}

// Item 5: non-finite input rejection.
void test_non_finite_input_rejected() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    auto input = valid_input(model);
    input.target_hidden_state[3] = std::numeric_limits<float>::infinity();
    require_throws(
        [&] { static_cast<void>(oracle::runtime::execute_qwen35_reference_mtp(model, weights, input)); },
        "non-finite value");
}

// Item 6: known small eh_proj calculation, computed independently of the
// implementation under test. With enorm/hnorm weights all 1.0 and a candidate-token
// embedding row and target hidden state both constant at 4.0, RMSNorm reduces (up to
// the negligible 1e-6 epsilon) to sign(4.0) == 1.0 for every element, so
// concat == [1,1,...,1] (2 * embedding_length elements). With eh_proj's weight matrix
// set so every output column sums its 2*embedding_length input weights to exactly
// 1.0 (each entry == 1/(2*embedding_length)), eh_proj's output is 1.0 * 1.0 * (2*n)
// == 1.0 for every one of its embedding_length elements -- a closed-form expected
// value never computed by calling Oracle's own decoder/matvec.
void test_known_small_eh_proj_calculation() {
    auto model = manifest(true, 1);
    Fixture fixture = make_fixture(model);

    const std::size_t n = model.embedding_length;
    std::vector<float> ones_row(n, 1.0F);
    fixture.set_f32("blk.1.nextn.enorm.weight", ones_row);
    fixture.set_f32("blk.1.nextn.hnorm.weight", ones_row);

    std::vector<float> token_row(n, 4.0F);
    // token_embd.weight is [embedding_length, vocabulary_size]; row-major per token
    // means token id 0's row occupies the first `n` floats.
    fixture.set_f32_row("token_embd.weight", 0, token_row);

    const float column_weight = 1.0F / static_cast<float>(2 * n);
    std::vector<float> eh_proj_flat(2 * n * n, column_weight);
    fixture.set_f32("blk.1.nextn.eh_proj.weight", eh_proj_flat);

    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    oracle::runtime::Qwen35MtpInput input;
    input.candidate_token_id = 0;
    input.target_hidden_state.assign(n, 4.0F);
    input.position = oracle::runtime::RopePosition::text(0);

    const auto result = oracle::runtime::execute_qwen35_reference_mtp(model, weights, input, true);
    const auto* eh_proj = result.trace.find("mtp_eh_proj");
    require(eh_proj != nullptr, "mtp_eh_proj must be captured");
    require(eh_proj->values.size() == n, "eh_proj output width mismatch");
    for (std::size_t i = 0; i < n; ++i) {
        require_near(eh_proj->values[i], 1.0F, 1.0e-3F, "known small eh_proj calculation mismatch");
    }
}

// Item 7: norm/fusion ordering -- e_norm occupies the FIRST half of mtp_concat and
// h_norm the SECOND half (source-verified: graph_mtp.cpp:721,
// ggml_concat(ctx0, e_norm, h_norm, dim=0)), using distinguishable constant values.
void test_norm_fusion_ordering() {
    auto model = manifest(true, 1);
    Fixture fixture = make_fixture(model);
    const std::size_t n = model.embedding_length;

    // RMSNorm is scale-invariant in its input (any positive constant input normalizes
    // to the same unit-RMS shape), so distinguishing the two halves requires distinct
    // *weights*, not just distinct input magnitudes.
    fixture.set_f32("blk.1.nextn.enorm.weight", std::vector<float>(n, 1.0F));
    fixture.set_f32("blk.1.nextn.hnorm.weight", std::vector<float>(n, 2.0F));
    fixture.set_f32_row("token_embd.weight", 0, std::vector<float>(n, 2.0F));

    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    oracle::runtime::Qwen35MtpInput input;
    input.candidate_token_id = 0;
    input.target_hidden_state.assign(n, 6.0F);
    input.position = oracle::runtime::RopePosition::text(0);

    const auto result = oracle::runtime::execute_qwen35_reference_mtp(model, weights, input, true);
    const auto* concat = result.trace.find("mtp_concat");
    const auto* enorm = result.trace.find("mtp_enorm");
    const auto* hnorm = result.trace.find("mtp_hnorm");
    require(concat != nullptr && enorm != nullptr && hnorm != nullptr, "required trace tensors missing");
    require(concat->values.size() == 2 * n, "concat width mismatch");
    for (std::size_t i = 0; i < n; ++i) {
        require_near(concat->values[i], enorm->values[i], 0.0F, "concat first half must equal e_norm");
        require_near(concat->values[n + i], hnorm->values[i], 0.0F, "concat second half must equal h_norm");
    }
    // e_norm (from token embedding 2.0) and h_norm (from hidden state 6.0) must be
    // distinguishable -- proves the two halves are not accidentally swapped/aliased.
    require(std::abs(enorm->values[0] - hnorm->values[0]) > 1.0e-3F,
            "e_norm and h_norm must differ for distinct inputs (ordering check would be vacuous otherwise)");
}

// Item 8: shared-head tied output behavior. The fixture never adds
// nextn.shared_head_head, so the projection must fall back to weights.output (which
// is itself tied to token_embd for this fixture, since no explicit output.weight is
// added) -- matching graph_mtp.cpp:809-811.
void test_shared_head_tied_output_behavior() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(weights.mtp->shared_head == nullptr,
            "fixture must omit shared_head_head to exercise the tied fallback");
    require(weights.output_is_tied && weights.output == weights.token_embedding,
            "backbone output must itself be tied for this fixture");

    const auto result = oracle::runtime::execute_qwen35_reference_mtp(model, weights, valid_input(model), true);
    require(result.logits.size() == model.vocabulary_size, "logits width mismatch");

    // Independently recompute the final projection from the captured
    // mtp_shared_head_norm output using weights.output directly, and confirm it
    // matches Oracle's own logits exactly -- proving the tied fallback was actually
    // used, not merely that *some* projection ran.
    const auto* head_norm = result.trace.find("mtp_shared_head_norm");
    require(head_norm != nullptr, "mtp_shared_head_norm must be captured");
    std::vector<float> expected(model.vocabulary_size);
    oracle::backend::cpu::reference_mapped_tensor_matvec(*weights.output, head_norm->values, expected);
    require(expected.size() == result.logits.size(), "recomputed logits width mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require_near(result.logits[i], expected[i], 0.0F,
                     "logits must exactly match tied-output recomputation");
    }
}

// Item 9: state isolation. A canonical HybridCache is constructed and never touched
// by execute_qwen35_reference_mtp (whose signature does not even accept one) --
// confirms MTP execution cannot mutate canonical backbone state.
void test_state_isolation_from_canonical_cache() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);

    oracle::runtime::HybridCache canonical_cache(model, /*maximum_tokens=*/16);
    require(canonical_cache.sequence_length() == 0, "canonical cache must start empty");

    static_cast<void>(oracle::runtime::execute_qwen35_reference_mtp(model, weights, valid_input(model), true));

    require(canonical_cache.sequence_length() == 0,
            "canonical HybridCache must remain untouched by MTP execution");
}

// Item 10: complete logits count.
void test_complete_logits_count() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    const auto result = oracle::runtime::execute_qwen35_reference_mtp(model, weights, valid_input(model), false);
    require(result.logits.size() == model.vocabulary_size, "logits count must equal vocabulary_size exactly");
}

// Item 11: deterministic repeated execution (greedy argmax only, no stochastic
// sampling anywhere in this path).
void test_deterministic_repeated_execution() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    const auto input = valid_input(model);

    const auto first = oracle::runtime::execute_qwen35_reference_mtp(model, weights, input, true);
    const auto second = oracle::runtime::execute_qwen35_reference_mtp(model, weights, input, true);

    require(first.argmax_token_id == second.argmax_token_id, "argmax must be deterministic");
    require(first.logits.size() == second.logits.size(), "logits size must be deterministic");
    for (std::size_t i = 0; i < first.logits.size(); ++i) {
        require(first.logits[i] == second.logits[i], "logits must be bit-identical across repeated calls");
    }
}

}  // namespace

int main() {
    try {
        test_mtp_absent_rejected();
        test_incomplete_mtp_weights_rejected();
        test_input_width_mismatch_rejected();
        test_invalid_position_rejected();
        test_non_finite_input_rejected();
        test_known_small_eh_proj_calculation();
        test_norm_fusion_ordering();
        test_shared_head_tied_output_behavior();
        test_state_isolation_from_canonical_cache();
        test_complete_logits_count();
        test_deterministic_repeated_execution();
        std::cout << "Phase 2F Slice 2 MTP execution tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 2F Slice 2 test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
