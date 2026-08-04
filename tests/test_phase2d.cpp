#include "oracle/model/ggml_type.hpp"
#include "oracle/runtime/qwen35_forward.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
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

void require_close(float actual, float expected, float tolerance, std::string_view message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(message) + ": expected " +
                                 std::to_string(expected) + ", received " +
                                 std::to_string(actual));
    }
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

oracle::model::Qwen35Manifest manifest() {
    oracle::model::Qwen35Manifest value;
    value.architecture = "qwen35";
    value.model_name = "phase2d-fixture";
    value.total_block_count = 2;
    value.backbone_block_count = 2;
    value.context_length = 32;
    value.embedding_length = 4;
    value.feed_forward_length = 4;
    value.attention_head_count = 2;
    value.attention_head_count_kv = 1;
    value.attention_key_length = 2;
    value.attention_value_length = 2;
    value.attention_rms_epsilon = 1.0e-6F;
    value.rope_dimension_count = 2;
    value.rope_dimension_sections = {1, 0, 0, 0};
    value.rope_frequency_base = 10000.0F;
    value.ssm_convolution_kernel = 2;
    value.ssm_state_size = 1;
    value.ssm_group_count = 2;
    value.ssm_time_step_rank = 4;
    value.ssm_inner_size = 4;
    value.full_attention_interval = 2;
    value.vocabulary_size = 8;
    return value;
}

struct Fixture {
    std::vector<oracle::model::GgufTensorInfo> infos;
    std::vector<std::vector<std::byte>> storage;
    std::vector<oracle::model::GgufTensorView> views;

    Fixture() {
        infos.reserve(64);
        storage.reserve(64);
        views.reserve(64);
    }

    const oracle::model::GgufTensorView* add(std::string name,
                                             std::vector<std::uint64_t> dimensions,
                                             std::vector<float> values) {
        std::size_t expected = 1;
        for (const std::uint64_t dimension : dimensions) {
            expected *= static_cast<std::size_t>(dimension);
        }
        require(values.size() == expected, "fixture value count mismatch for " + name);

        infos.push_back({std::move(name), std::move(dimensions), 0, 0});
        storage.emplace_back(values.size() * sizeof(float));
        for (std::size_t index = 0; index < values.size(); ++index) {
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(values[index]);
            const std::size_t offset = index * sizeof(float);
            storage.back()[offset] = static_cast<std::byte>(bits & 0xFFU);
            storage.back()[offset + 1] = static_cast<std::byte>((bits >> 8U) & 0xFFU);
            storage.back()[offset + 2] = static_cast<std::byte>((bits >> 16U) & 0xFFU);
            storage.back()[offset + 3] = static_cast<std::byte>((bits >> 24U) & 0xFFU);
        }
        const auto* layout = oracle::model::ggml_type_layout(0);
        require(layout != nullptr, "F32 layout missing");
        views.emplace_back(&infos.back(),
                           layout,
                           storage.back().data(),
                           storage.back().size(),
                           0);
        return &views.back();
    }

    const oracle::model::GgufTensorView* vector(std::string name,
                                                std::vector<float> values) {
        const std::size_t size = values.size();
        return add(std::move(name), {size}, std::move(values));
    }

    const oracle::model::GgufTensorView* matrix(std::string name,
                                                std::size_t columns,
                                                std::size_t rows,
                                                std::vector<float> values) {
        return add(std::move(name), {columns, rows}, std::move(values));
    }
};

std::vector<float> zeros(std::size_t size) {
    return std::vector<float>(size, 0.0F);
}

std::vector<float> identity(std::size_t size) {
    std::vector<float> values(size * size, 0.0F);
    for (std::size_t index = 0; index < size; ++index) {
        values[index * size + index] = 1.0F;
    }
    return values;
}

oracle::model::Qwen35MlpWeights add_zero_mlp(Fixture& fixture, std::string_view prefix) {
    oracle::model::Qwen35MlpWeights weights;
    weights.gate = fixture.matrix(std::string(prefix) + "ffn_gate", 4, 4, zeros(16));
    weights.up = fixture.matrix(std::string(prefix) + "ffn_up", 4, 4, zeros(16));
    weights.down = fixture.matrix(std::string(prefix) + "ffn_down", 4, 4, zeros(16));
    return weights;
}

oracle::model::Qwen35BackboneBlockWeights recurrent_block(Fixture& fixture) {
    oracle::model::Qwen35BackboneBlockWeights block;
    block.index = 0;
    block.kind = oracle::model::Qwen35BlockKind::recurrent;
    block.input_norm = fixture.vector("r.input_norm", {1, 1, 1, 1});
    block.post_attention_norm = fixture.vector("r.post_norm", {1, 1, 1, 1});
    block.mlp = add_zero_mlp(fixture, "r.");

    std::vector<float> qkv(8 * 4, 0.0F);
    qkv[0 * 4 + 0] = 1.0F;
    qkv[1 * 4 + 1] = 1.0F;
    qkv[2 * 4 + 0] = 1.0F;
    qkv[3 * 4 + 1] = 1.0F;
    for (std::size_t row = 0; row < 4; ++row) {
        qkv[(row + 4) * 4 + row] = 1.0F;
    }

    std::vector<float> convolution(8 * 2, 0.0F);
    for (std::size_t channel = 0; channel < 8; ++channel) {
        convolution[channel * 2 + 1] = 1.0F;
    }

    oracle::model::Qwen35RecurrentWeights weights;
    weights.qkv = fixture.matrix("r.qkv", 4, 8, qkv);
    weights.gate = fixture.matrix("r.z", 4, 4, zeros(16));
    weights.convolution = fixture.matrix("r.conv", 2, 8, convolution);
    weights.time_step_bias = fixture.vector("r.dt", {0, 0, 0, 0});
    weights.decay = fixture.vector("r.a", {-1, -1, -1, -1});
    weights.beta = fixture.matrix("r.beta", 4, 4, zeros(16));
    weights.alpha = fixture.matrix("r.alpha", 4, 4, zeros(16));
    weights.norm = fixture.vector("r.norm", {1});
    weights.output = fixture.matrix("r.out", 4, 4, identity(4));
    block.recurrent = weights;
    return block;
}

oracle::model::Qwen35BackboneBlockWeights attention_block(Fixture& fixture) {
    oracle::model::Qwen35BackboneBlockWeights block;
    block.index = 1;
    block.kind = oracle::model::Qwen35BlockKind::full_attention;
    block.input_norm = fixture.vector("a.input_norm", {1, 1, 1, 1});
    block.post_attention_norm = fixture.vector("a.post_norm", {1, 1, 1, 1});
    block.mlp = add_zero_mlp(fixture, "a.");

    std::vector<float> query(8 * 4, 0.0F);
    query[0 * 4 + 0] = 1.0F;
    query[1 * 4 + 1] = 1.0F;
    query[4 * 4 + 2] = 1.0F;
    query[5 * 4 + 3] = 1.0F;

    std::vector<float> key(2 * 4, 0.0F);
    key[0 * 4 + 0] = 1.0F;
    key[1 * 4 + 1] = 1.0F;

    std::vector<float> value(2 * 4, 0.0F);
    value[0 * 4 + 2] = 1.0F;
    value[1 * 4 + 3] = 1.0F;

    oracle::model::Qwen35AttentionWeights weights;
    weights.query = fixture.matrix("a.q", 4, 8, query);
    weights.key = fixture.matrix("a.k", 4, 2, key);
    weights.value = fixture.matrix("a.v", 4, 2, value);
    weights.output = fixture.matrix("a.out", 4, 4, identity(4));
    weights.query_norm = fixture.vector("a.qnorm", {1, 1});
    weights.key_norm = fixture.vector("a.knorm", {1, 1});
    block.attention = weights;
    return block;
}

struct ModelFixture {
    Fixture tensors;
    oracle::model::Qwen35Weights weights;
    std::vector<float> embedding_values;

    ModelFixture() {
        embedding_values = {
            1, 2, 3, 4,
            2, 1, 0, -1,
            -1, 0, 1, 2,
            0.5F, 0.25F, -0.25F, -0.5F,
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        };
        weights.token_embedding = tensors.matrix("token_embd.weight", 4, 8, embedding_values);
        weights.output_norm = tensors.vector("output_norm.weight", {1, 1, 1, 1});
        weights.output = nullptr;
        weights.output_is_tied = true;
        weights.blocks.push_back(recurrent_block(tensors));
        weights.blocks.push_back(attention_block(tensors));
        weights.report.backbone_block_count = 2;
        weights.report.recurrent_block_count = 1;
        weights.report.attention_block_count = 1;
        weights.report.tied_output = true;
    }
};

void test_complete_forward() {
    ModelFixture fixture;
    const auto model = manifest();
    oracle::runtime::HybridCache state(model, 1);
    const auto result = oracle::runtime::execute_qwen35_reference_token(
        model,
        fixture.weights,
        0,
        state,
        oracle::runtime::RopePosition::text(0),
        true);

    require(state.sequence_length() == 1, "hybrid state did not advance");
    require(result.trace.blocks.size() == 2, "forward trace block count mismatch");
    require(result.trace.blocks[0].kind == oracle::model::Qwen35BlockKind::recurrent,
            "first state owner must be recurrent");
    require(result.trace.blocks[1].kind == oracle::model::Qwen35BlockKind::full_attention,
            "second state owner must be attention");
    require(result.trace.final_norm.size() == 4, "final norm width mismatch");
    require(result.logits.size() == 8, "complete logits count mismatch");
    require(result.trace.output_is_tied, "fixture output projection should be tied");
    require(!result.trace.mtp_executed, "Phase 2D must not execute MTP");
    require(result.trace.contracts.execution_projection == "decoded-f32-scalar",
            "default projection contract label mismatch");
    require(result.trace.contracts.execution_attention_cache == "f32-semantic",
            "default attention cache contract label mismatch");

    for (std::size_t row = 0; row < 8; ++row) {
        float expected = 0.0F;
        for (std::size_t column = 0; column < 4; ++column) {
            expected += fixture.embedding_values[row * 4 + column] *
                        result.trace.final_norm[column];
        }
        require_close(result.logits[row], expected, 1.0e-6F,
                      "tied output projection mismatch");
    }

    const std::string text = oracle::runtime::qwen35_forward_trace_text(result);
    const std::string json = oracle::runtime::qwen35_forward_trace_json(result, false);
    const std::string full_json = oracle::runtime::qwen35_forward_trace_json(result, true);
    require(text.find("decoded-f32-scalar") != std::string::npos,
            "text trace omitted the projection contract");
    require(json.find("\"count\":8") != std::string::npos,
            "JSON trace omitted the logits count");
    require(json.find("\"logits\":{\"name\":\"logits\"") != std::string::npos,
            "JSON trace omitted the logits summary");
    require(full_json.size() > json.size(), "full JSON should include complete logits values");
}

void test_diagnostic_overrides() {
    ModelFixture fixture;
    const auto model = manifest();
    oracle::runtime::HybridCache state(model, 1);
    oracle::runtime::Qwen35ForwardOverrides overrides;
    overrides.projection_source_contract = "q5-k/q6-k-x-q8-k";
    overrides.attention_cache_source_contract = "f16-pinned";
    overrides.logits = oracle::runtime::Qwen35TraceTensor{
        "logits", {0, 1, 2, 3, 4, 5, 6, 7}};

    const auto result = oracle::runtime::execute_qwen35_reference_token(
        model,
        fixture.weights,
        0,
        state,
        oracle::runtime::RopePosition::text(0),
        true,
        &overrides);
    require(result.logits == overrides.logits->values, "logits override was not applied");
    require(result.trace.contracts.execution_projection == "diagnostic-external-overrides",
            "override execution contract was not labeled");
    require(result.trace.contracts.override_projection_source == "q5-k/q6-k-x-q8-k",
            "override source projection contract was not preserved");
    require(result.trace.contracts.override_attention_cache_source == "f16-pinned",
            "override source cache contract was not preserved");
}

void test_override_preflight_is_strict() {
    ModelFixture fixture;
    const auto model = manifest();
    oracle::runtime::HybridCache state(model, 1);
    oracle::runtime::Qwen35ForwardOverrides overrides;
    overrides.blocks.push_back({0, {{{"not_a_projection", {1.0F}}}}});

    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
                model,
                fixture.weights,
                0,
                state,
                oracle::runtime::RopePosition::text(0),
                true,
                &overrides));
        },
        "unknown Qwen3.5 forward override");
    require(state.sequence_length() == 0,
            "invalid overrides must be rejected before state mutation");

    oracle::runtime::Qwen35ForwardOverrides bad_logits;
    bad_logits.logits = oracle::runtime::Qwen35TraceTensor{"logits", {1.0F}};
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
                model,
                fixture.weights,
                0,
                state,
                oracle::runtime::RopePosition::text(0),
                true,
                &bad_logits));
        },
        "logits override count mismatch");
    require(state.sequence_length() == 0,
            "invalid logits override must be rejected before state mutation");
}

void test_position_and_family_validation() {
    ModelFixture fixture;
    const auto model = manifest();
    oracle::runtime::HybridCache state(model, 1);
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
                model,
                fixture.weights,
                0,
                state,
                oracle::runtime::RopePosition::text(1),
                false));
        },
        "position does not match");

    fixture.weights.blocks[0].kind = oracle::model::Qwen35BlockKind::full_attention;
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::execute_qwen35_reference_token(
                model,
                fixture.weights,
                0,
                state,
                oracle::runtime::RopePosition::text(0),
                false));
        },
        "block/state family mismatch");
}

}  // namespace

int main() {
    try {
        test_complete_forward();
        test_diagnostic_overrides();
        test_override_preflight_is_strict();
        test_position_and_family_validation();
        std::cout << "Phase 2D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Phase 2D test failure: " << error.what() << '\n';
        return 1;
    }
}
