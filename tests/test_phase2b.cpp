#include "oracle/model/qwen35_weights.hpp"
#include "oracle/model/ggml_type.hpp"

#include <algorithm>
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
                "exception message did not contain expected text");
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

oracle::model::Qwen35Manifest manifest(bool mtp = false, std::uint32_t block_count = 4) {
    oracle::model::Qwen35Manifest value;
    value.architecture = "qwen35";
    value.model_name = "phase2b-fixture";
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
        throw std::runtime_error("fixture tensor not found");
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

Fixture make_fixture(const oracle::model::Qwen35Manifest& model,
                     bool explicit_output = false) {
    Fixture fixture;
    fixture.add("token_embd.weight", {model.embedding_length, model.vocabulary_size});
    fixture.add("output_norm.weight", {model.embedding_length});
    if (explicit_output) {
        fixture.add("output.weight", {model.embedding_length, model.vocabulary_size});
    }
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

void test_base_binding() {
    const auto model = manifest();
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(weights.output_is_tied, "base fixture should tie output to embedding");
    require(weights.output == weights.token_embedding, "tied output pointer mismatch");
    require(weights.blocks.size() == 4, "backbone block count mismatch");
    require(weights.report.recurrent_block_count == 3, "recurrent count mismatch");
    require(weights.report.attention_block_count == 1, "attention count mismatch");
    require(weights.report.required_tensor_count == 55, "base required tensor count mismatch");
    require(weights.report.bound_tensor_count == 55, "base bound tensor count mismatch");
    require(weights.report.unexpected_tensor_count == 0, "base unexpected tensor count mismatch");
    require(weights.blocks[3].attention.has_value(), "block 3 must be attention");
    require(weights.blocks[0].recurrent.has_value(), "block 0 must be recurrent");
}

void test_explicit_output_and_unexpected() {
    const auto model = manifest();
    Fixture fixture = make_fixture(model, true);
    fixture.add("vendor.extra.weight", {model.embedding_length});
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(!weights.output_is_tied, "explicit output was not selected");
    require(weights.report.optional_tensor_count == 1, "explicit output optional count mismatch");
    require(weights.report.unexpected_tensor_count == 1, "unexpected tensor was not reported");
    require(weights.report.unexpected_tensors.front() == "vendor.extra.weight",
            "unexpected tensor name mismatch");
}

void test_mtp_binding() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    require(weights.mtp.has_value(), "MTP block was not bound");
    require(weights.mtp->block_index == 4, "MTP block index mismatch");
    require(weights.mtp->shared_head_norm != nullptr, "shared head norm was not bound");
    require(weights.report.required_tensor_count == 69, "MTP required count mismatch");
    require(weights.report.optional_tensor_count == 1, "MTP optional count mismatch");
    require(weights.report.bound_tensor_count == 70, "MTP total count mismatch");
}

void test_missing_and_shape_errors() {
    const auto model = manifest();
    Fixture missing = make_fixture(model);
    const std::size_t missing_index = missing.find("blk.0.ssm_a");
    missing.views.erase(missing.views.begin() + static_cast<std::ptrdiff_t>(missing_index));
    require_throws([&] { static_cast<void>(oracle::model::bind_qwen35_weights(missing.views, model)); },
                   "missing required Qwen3.5 tensor: blk.0.ssm_a");

    Fixture wrong_shape = make_fixture(model);
    wrong_shape.infos[wrong_shape.find("blk.3.attn_q_norm.weight")].dimensions[0] = 32;
    require_throws([&] { static_cast<void>(oracle::model::bind_qwen35_weights(wrong_shape.views, model)); },
                   "expected dimensions [64], found [32]");
}

void test_type_and_duplicate_errors() {
    const auto model = manifest();
    Fixture wrong_type = make_fixture(model);
    wrong_type.replace_type("output_norm.weight", 12);
    require_throws([&] { static_cast<void>(oracle::model::bind_qwen35_weights(wrong_type.views, model)); },
                   "unsupported storage type q4_K for vector weight");

    Fixture duplicate = make_fixture(model);
    duplicate.views.push_back(duplicate.views.front());
    require_throws([&] { static_cast<void>(oracle::model::bind_qwen35_weights(duplicate.views, model)); },
                   "duplicate tensor in Qwen3.5 binding input");
}


void test_real_family_tensor_counts() {
    {
        const auto model = manifest(false, 32);
        Fixture fixture = make_fixture(model);
        const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
        require(weights.report.recurrent_block_count == 24,
                "Qwopus recurrent block count mismatch");
        require(weights.report.attention_block_count == 8,
                "Qwopus attention block count mismatch");
        require(weights.report.required_tensor_count == 426,
                "Qwopus base tensor schema must bind 426 tensors");
        require(weights.report.bound_tensor_count == 426,
                "Qwopus base bound tensor count mismatch");
    }
    {
        const auto model = manifest(true, 32);
        Fixture fixture = make_fixture(model);
        const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
        require(weights.report.required_tensor_count == 440,
                "Qwopus MTP required tensor count mismatch");
        require(weights.report.optional_tensor_count == 1,
                "Qwopus MTP optional tensor count mismatch");
        require(weights.report.bound_tensor_count == 441,
                "Qwopus MTP schema must bind 441 tensors");
        require(weights.mtp && weights.mtp->block_index == 32,
                "Qwopus MTP must remain separate at blk.32");
    }
}

void test_quantized_matrix_types_and_byte_geometry() {
    const auto model = manifest();
    Fixture quantized = make_fixture(model);
    quantized.replace_type("blk.0.ffn_gate.weight", 13);
    quantized.replace_type("blk.0.ffn_down.weight", 14);
    const auto weights = oracle::model::bind_qwen35_weights(quantized.views, model);
    require(weights.blocks[0].mlp.gate->layout().type == 13,
            "Q5_K matrix was not accepted");
    require(weights.blocks[0].mlp.down->layout().type == 14,
            "Q6_K matrix was not accepted");

    Fixture malformed = make_fixture(model);
    const std::size_t index = malformed.find("blk.0.ffn_gate.weight");
    const auto& original = malformed.views[index];
    malformed.views[index] = oracle::model::GgufTensorView(
        &malformed.infos[index],
        &original.layout(),
        malformed.storage[index].data(),
        malformed.storage[index].size() - 1,
        0);
    require_throws(
        [&] { static_cast<void>(oracle::model::bind_qwen35_weights(malformed.views, model)); },
        "mapped byte size does not match tensor geometry");
}

void test_manifest_consistency_errors() {
    auto total = manifest();
    ++total.total_block_count;
    Fixture total_fixture = make_fixture(manifest());
    require_throws(
        [&] { static_cast<void>(oracle::model::bind_qwen35_weights(total_fixture.views, total)); },
        "total block count does not equal backbone plus NextN layers");

    auto inner = manifest();
    ++inner.ssm_inner_size;
    Fixture inner_fixture = make_fixture(manifest());
    require_throws(
        [&] { static_cast<void>(oracle::model::bind_qwen35_weights(inner_fixture.views, inner)); },
        "SSM inner size does not match state size times time-step rank");
}

void test_reports() {
    const auto model = manifest(true);
    Fixture fixture = make_fixture(model);
    const auto weights = oracle::model::bind_qwen35_weights(fixture.views, model);
    const std::string text = oracle::model::qwen35_binding_report_text(weights);
    const std::string json = oracle::model::qwen35_binding_report_json(weights);
    require(text.find("blk.3  attention") != std::string::npos,
            "text report missing attention classification");
    require(json.find("\"bound_tensor_count\":70") != std::string::npos,
            "JSON report missing bound count");
    require(json.find("\"mtp_block_index\":4") != std::string::npos,
            "JSON report missing MTP block");
}

}  // namespace

int main() {
    try {
        test_base_binding();
        test_explicit_output_and_unexpected();
        test_mtp_binding();
        test_missing_and_shape_errors();
        test_type_and_duplicate_errors();
        test_quantized_matrix_types_and_byte_geometry();
        test_manifest_consistency_errors();
        test_real_family_tensor_counts();
        test_reports();
        std::cout << "Phase 2B Qwen3.5 binding tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 2B test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
