// Phase 2F Slice 1: real Qwen3.5 MTP/NextN checkpoint forensics closed the loop back
// onto the Phase 2B contract (oracle/model/qwen35_manifest.hpp, qwen35_weights.hpp).
// These tests are additive: they do not change qwen35_manifest.cpp/qwen35_weights.cpp,
// which already implement the MTP capability contract and tensor binding validated here.
//
// Motivating real-world facts, captured against the real
// unsloth/Qwen3.5-2B-MTP-GGUF checkpoint (Q5_K_M, SHA-256
// b8d558161010664c469b59efeed318e8a64267b1aecdaabeb21a1c44e21aac22):
//   - qwen35.block_count=25, qwen35.nextn_predict_layers=1 -> backbone=24, MTP block at blk.24
//   - blk.24.nextn.* carries eh_proj/enorm/hnorm/shared_head_norm (shared_head_head and
//     embed_tokens are absent -> tied to the main output/token_embd, as the optional-tensor
//     design already allows)
//   - blk.24.nextn.eh_proj.weight is stored as ggml type Q8_0 (type id 8), which
//     oracle::model::storage_decode.cpp does not decode; Oracle's existing binder already
//     rejects this deterministically via allowed_storage_type()
//   - the real file has exactly 335 tensors, of which exactly 1 (shared_head_norm) is optional

#include "oracle/model/qwen35_weights.hpp"
#include "oracle/model/ggml_type.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
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

oracle::model::Qwen35Manifest manifest(bool mtp = false, std::uint32_t block_count = 4) {
    oracle::model::Qwen35Manifest value;
    value.architecture = "qwen35";
    value.model_name = "phase2f-slice1-fixture";
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

    void add(std::string name,
             std::initializer_list<std::uint64_t> dimensions,
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

    void replace_type(std::string_view name, std::uint32_t type) {
        const std::size_t index = find(name);
        infos[index].ggml_type = type;
        const auto* layout = oracle::model::ggml_type_layout(type);
        require(layout != nullptr, "replacement type layout missing");
        const std::uint64_t bytes = oracle::model::ggml_tensor_byte_size(infos[index].dimensions, type);
        storage[index].assign(static_cast<std::size_t>(bytes), std::byte{0});
        views[index] = oracle::model::GgufTensorView(
            &infos[index], layout, storage[index].data(), storage[index].size(), 0);
    }
};

void add_attention(Fixture& fixture,
                   const oracle::model::Qwen35Manifest& model,
                   std::uint32_t block) {
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

void add_recurrent(Fixture& fixture,
                   const oracle::model::Qwen35Manifest& model,
                   std::uint32_t block) {
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

void add_common(Fixture& fixture,
                const oracle::model::Qwen35Manifest& model,
                std::uint32_t block) {
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

// Real-checkpoint fact: blk.24.nextn.eh_proj.weight is a required MTP tensor. Removing
// any required MTP tensor must be rejected the same way a missing backbone tensor is.
void test_missing_required_mtp_tensor_rejected() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const std::size_t index = fixture.find("blk.4.nextn.eh_proj.weight");
    fixture.views.erase(fixture.views.begin() + static_cast<std::ptrdiff_t>(index));
    require_throws(
        [&] { static_cast<void>(oracle::model::bind_qwen35_weights(fixture.views, model)); },
        "missing required Qwen3.5 tensor: blk.4.nextn.eh_proj.weight");
}

// A malformed MTP tensor dimension (e.g. a corrupted GGUF or an incompatible export)
// must be rejected with the same precision as a malformed backbone tensor.
void test_malformed_mtp_dimension_rejected() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    fixture.infos[fixture.find("blk.4.nextn.hnorm.weight")].dimensions[0] = 999;
    require_throws(
        [&] { static_cast<void>(oracle::model::bind_qwen35_weights(fixture.views, model)); },
        "expected dimensions [256], found [999]");
}

// The Phase 2B binder currently supports exactly one NextN layer; a manifest claiming
// more must be rejected before any tensor is touched.
void test_mtp_layer_count_mismatch_rejected() {
    auto model = manifest(true);
    model.nextn_predict_layers = 2;
    model.total_block_count = model.backbone_block_count + model.nextn_predict_layers;
    Fixture fixture = make_fixture(model);
    require_throws(
        [&] { static_cast<void>(oracle::model::bind_qwen35_weights(fixture.views, model)); },
        "currently supports exactly one NextN layer");
}

// Phase 2F Slice 1B superseded the original finding here: Slice 1 found that the real
// unsloth Qwen3.5-2B-MTP-GGUF (Q5_K_M) stores blk.24.nextn.eh_proj.weight as Q8_0, a
// type Oracle's decoder did not support at the time, and this test asserted that the
// binder rejected it. Slice 1B added reviewed Q8_0 scalar storage decoding
// (storage_decode.cpp) and extended the matrix-weight policy
// (qwen35_weights.cpp::allowed_storage_type) to accept it, closing that gap. This test
// now asserts the opposite of its Slice 1 form: eh_proj as Q8_0 must bind successfully.
// The original test's *intent* — that the binder correctly discriminates supported from
// unsupported matrix storage types — is preserved by
// test_mtp_eh_proj_still_rejects_genuinely_unsupported_type() below, using a type that
// remains unsupported (Q4_K) rather than deleting the coverage outright.
void test_mtp_eh_proj_q8_0_now_accepted() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    constexpr std::uint32_t ggml_type_q8_0 = 8;
    fixture.replace_type("blk.4.nextn.eh_proj.weight", ggml_type_q8_0);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(weights.mtp.has_value(), "MTP weights must bind when eh_proj is valid Q8_0");
    require(weights.mtp->embedding_hidden_projection != nullptr,
            "eh_proj must be bound");
    require(weights.mtp->embedding_hidden_projection->layout().type == ggml_type_q8_0,
            "bound eh_proj must retain its Q8_0 storage type");
}

// Preserves the Slice 1 test's original intent: a genuinely unsupported matrix storage
// type on an MTP tensor must still be rejected deterministically and by name. Q4_K (12)
// is not in Oracle's matrix allow-list ({F32, F16, BF16, Q5_K, Q6_K, Q8_0}).
void test_mtp_eh_proj_still_rejects_genuinely_unsupported_type() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    constexpr std::uint32_t ggml_type_q4_k = 12;
    fixture.replace_type("blk.4.nextn.eh_proj.weight", ggml_type_q4_k);
    require_throws(
        [&] { static_cast<void>(oracle::model::bind_qwen35_weights(fixture.views, model)); },
        "blk.4.nextn.eh_proj.weight: unsupported storage type q4_K for matrix weight");
}

// Q8_0 is deliberately matrix-only in Oracle's contract (Slice 1B): no known Qwen3.5
// vector weight requires it, so vector policy is intentionally unchanged. A vector
// tensor (enorm) declared as Q8_0 must still be rejected.
void test_mtp_vector_tensor_still_rejects_q8_0() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    constexpr std::uint32_t ggml_type_q8_0 = 8;
    fixture.replace_type("blk.4.nextn.enorm.weight", ggml_type_q8_0);
    require_throws(
        [&] { static_cast<void>(oracle::model::bind_qwen35_weights(fixture.views, model)); },
        "blk.4.nextn.enorm.weight: unsupported storage type q8_0 for vector weight");
}

// A malformed Q8_0 MTP tensor (byte size inconsistent with its declared shape, e.g. a
// truncated/corrupted export) must still be rejected even though Q8_0 itself is now a
// supported matrix type.
void test_mtp_eh_proj_malformed_q8_0_rejected() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    constexpr std::uint32_t ggml_type_q8_0 = 8;
    fixture.replace_type("blk.4.nextn.eh_proj.weight", ggml_type_q8_0);
    const std::size_t index = fixture.find("blk.4.nextn.eh_proj.weight");
    const auto& original = fixture.views[index];
    fixture.views[index] = oracle::model::GgufTensorView(
        &fixture.infos[index], &original.layout(), fixture.storage[index].data(),
        fixture.storage[index].size() - 1, 0);
    require_throws(
        [&] { static_cast<void>(oracle::model::bind_qwen35_weights(fixture.views, model)); },
        "mapped byte size does not match tensor geometry");
}

// MTP weights must never be reachable through the backbone block list, and vice versa.
void test_mtp_weights_kept_separate_from_backbone() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(weights.blocks.size() == model.backbone_block_count,
            "backbone block list must exclude the appended MTP block");
    for (const auto& block : weights.blocks) {
        require(block.index != model.backbone_block_count,
                "MTP block index leaked into the backbone block list");
    }
    require(weights.mtp.has_value(), "MTP weights must be present when manifest.has_mtp()");
    require(weights.mtp->block_index == model.backbone_block_count,
            "MTP weights must be indexed at the appended block");
}

// The optional MTP capability must default to safely absent.
void test_default_manifest_has_no_mtp() {
    const oracle::model::Qwen35Manifest empty{};
    require(!empty.has_mtp(), "default-constructed manifest must report has_mtp() == false");
    require(empty.nextn_predict_layers == 0, "default nextn_predict_layers must be zero");

    const auto model = manifest(false);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(!weights.mtp.has_value(), "non-MTP manifest must bind with weights.mtp absent");
}

// This mirrors the real unsloth/Qwen3.5-2B-MTP-GGUF geometry: 24 backbone blocks with
// full_attention_interval=4 (18 recurrent + 6 attention), one appended MTP block whose
// only optional tensor is shared_head_norm (shared_head_head/embed_tokens are absent
// and tie to the main output/token embedding, matching what the real checkpoint does).
// The real file has exactly 335 tensors; this fixture predicts that count structurally,
// independent of the real file's absolute dimensions or quantization.
void test_real_qwen35_2b_mtp_topology() {
    const auto model = manifest(true, 24);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(weights.report.recurrent_block_count == 18, "2B MTP recurrent block count mismatch");
    require(weights.report.attention_block_count == 6, "2B MTP attention block count mismatch");
    require(weights.report.required_tensor_count == 334,
            "2B MTP required tensor count must match real checkpoint topology");
    require(weights.report.optional_tensor_count == 1,
            "2B MTP optional tensor count must match real checkpoint topology "
            "(only shared_head_norm is optionally present)");
    require(weights.report.bound_tensor_count == 335,
            "2B MTP topology must predict the real checkpoint's 335 tensors exactly");
    require(weights.mtp && weights.mtp->block_index == 24,
            "2B MTP block must sit at blk.24, matching qwen35.block_count=25 minus nextn_predict_layers=1");
    require(weights.mtp->embed_tokens == nullptr && weights.mtp->shared_head == nullptr,
            "2B MTP embed_tokens/shared_head_head must remain untied placeholders "
            "when absent, matching the real checkpoint");
}

// Phase 2F Slice 1B: the same real-checkpoint topology as above, but with eh_proj
// stored as Q8_0 -- exactly the real unsloth/Qwen3.5-2B-MTP-GGUF's actual encoding.
// Slice 1 could only prove this fixture's *shape* was correct (binding failed at
// eh_proj's unsupported type); Slice 1B closes the loop end-to-end.
void test_real_qwen35_2b_mtp_topology_with_q8_0_eh_proj() {
    const auto model = manifest(true, 24);
    Fixture fixture = make_fixture(model);
    constexpr std::uint32_t ggml_type_q8_0 = 8;
    fixture.replace_type("blk.24.nextn.eh_proj.weight", ggml_type_q8_0);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(weights.report.required_tensor_count == 334,
            "2B MTP+Q8_0 required tensor count must still match real checkpoint topology");
    require(weights.report.optional_tensor_count == 1,
            "2B MTP+Q8_0 optional tensor count must still match real checkpoint topology");
    require(weights.report.bound_tensor_count == 335,
            "2B MTP+Q8_0 topology must still predict the real checkpoint's 335 tensors exactly");
    require(weights.report.unexpected_tensor_count == 0,
            "2B MTP+Q8_0 topology must have zero unexpected tensors, matching the real checkpoint");
    require(weights.mtp->embedding_hidden_projection->layout().type == ggml_type_q8_0,
            "eh_proj must bind as Q8_0, exactly as the real checkpoint stores it");
}

}  // namespace

int main() {
    try {
        test_missing_required_mtp_tensor_rejected();
        test_malformed_mtp_dimension_rejected();
        test_mtp_layer_count_mismatch_rejected();
        test_mtp_eh_proj_q8_0_now_accepted();
        test_mtp_eh_proj_still_rejects_genuinely_unsupported_type();
        test_mtp_vector_tensor_still_rejects_q8_0();
        test_mtp_eh_proj_malformed_q8_0_rejected();
        test_mtp_weights_kept_separate_from_backbone();
        test_default_manifest_has_no_mtp();
        test_real_qwen35_2b_mtp_topology();
        test_real_qwen35_2b_mtp_topology_with_q8_0_eh_proj();
        std::cout << "Phase 2F Slice 1/1B MTP forensics-informed tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 2F Slice 1 test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
