#include "oracle/model/ggml_type.hpp"
#include "oracle/runtime/qwen35_block.hpp"
#include "oracle/runtime/reference_kernels.hpp"

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
    value.model_name = "phase2c-fixture";
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

void test_recurrent_block() {
    Fixture fixture;
    const auto model = manifest();
    const auto block = recurrent_block(fixture);
    const auto layout = oracle::runtime::qwen35_ssm_layout(model);
    oracle::runtime::SsmState state(layout.convolution_channels,
                                     layout.convolution_kernel,
                                     layout.value_heads,
                                     layout.key_head_dimension,
                                     layout.value_head_dimension);
    const std::vector<float> input{1, -2, 3, 4};
    const auto result = oracle::runtime::execute_qwen35_recurrent_block_reference(
        model, block, input, state, true);

    require(result.output == input, "zero recurrent output gate should preserve the residual");
    require(state.sequence_length() == 1, "recurrent state did not advance");
    require(std::any_of(state.recurrent().begin(), state.recurrent().end(),
                        [](float value) { return value != 0.0F; }),
            "recurrent state should contain a non-zero update");

    const auto* beta = result.trace.find("beta_sigmoid");
    const auto* gate = result.trace.find("gate");
    require(beta != nullptr && gate != nullptr, "recurrent trace is missing alpha/beta tensors");
    require_close(beta->values[0], 0.5F, 1.0e-6F, "beta sigmoid mismatch");
    require_close(gate->values[0], -std::log(2.0F), 1.0e-6F, "decay gate mismatch");
    require(result.trace.find("post_ffn") != nullptr, "recurrent trace is missing final output");
    const auto* q_conv = result.trace.find("q_conv");
    const auto* q_predelta = result.trace.find("q_conv_predelta");
    const auto* state_before = result.trace.find("state_predelta");
    const auto* state_after = result.trace.find("state_postdelta");
    require(q_conv != nullptr && q_conv->values.size() == 2,
            "recurrent trace must capture narrow Q before head repetition");
    require(q_predelta != nullptr && q_predelta->values.size() == 4,
            "recurrent trace must capture repeated normalized Q");
    require(q_predelta->values[0] > 0.0F && q_predelta->values[1] < 0.0F &&
                q_predelta->values[2] > 0.0F && q_predelta->values[3] < 0.0F,
            "Q/K repetition must use tiled GGUF head order");
    require(state_before != nullptr && state_after != nullptr,
            "recurrent trace is missing state snapshots");
    require(std::all_of(state_before->values.begin(), state_before->values.end(),
                        [](float value) { return value == 0.0F; }),
            "initial recurrent state snapshot should be zero");
    require(std::any_of(state_after->values.begin(), state_after->values.end(),
                        [](float value) { return value != 0.0F; }),
            "final recurrent state snapshot should contain the update");
}

void test_gated_delta_uses_ggml_l2_epsilon_contract() {
    oracle::runtime::SsmState state(1, 1, 1, 2, 1);
    oracle::core::Tensor query({1, 2});
    oracle::core::Tensor key({1, 2});
    oracle::core::Tensor value({1, 1});
    oracle::core::Tensor output({1, 1});

    query.fill(0.0F);
    key.fill(0.0F);
    query.data()[0] = 1.0e-4F;
    key.data()[0] = 1.0e-4F;
    value.data()[0] = 2.0F;

    const std::vector<float> beta{1.0F};
    const std::vector<float> no_decay{0.0F};
    oracle::runtime::gated_delta_step(
        query, key, value, beta, no_decay, state, output, 1.0e-6F);

    require_close(output.data()[0], std::sqrt(2.0F), 2.0e-6F,
                  "gated delta must use max(L2 norm, epsilon)");
}

void test_attention_block() {
    Fixture fixture;
    const auto model = manifest();
    const auto block = attention_block(fixture);
    oracle::runtime::KvCache cache(1,
                                    model.attention_head_count_kv,
                                    model.attention_key_length,
                                    model.attention_value_length);
    const std::vector<float> input{1, 2, 3, 4};
    const auto result = oracle::runtime::execute_qwen35_attention_block_reference(
        model, block, input, cache, oracle::runtime::RopePosition::text(0), true);

    const float inverse_rms = 1.0F / std::sqrt(7.5F + model.attention_rms_epsilon);
    const std::vector<float> expected{
        input[0] + 0.5F * input[2] * inverse_rms,
        input[1] + 0.5F * input[3] * inverse_rms,
        input[2] + 0.5F * input[2] * inverse_rms,
        input[3] + 0.5F * input[3] * inverse_rms,
    };
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_close(result.output[index], expected[index], 1.0e-5F,
                      "attention block output mismatch");
    }
    require(cache.size() == 1, "attention cache did not append the token");
    const auto* gated = result.trace.find("attn_gated");
    require(gated != nullptr && gated->values.size() == 4,
            "attention trace is missing the gated output");
}

void test_matvec_overrides() {
    Fixture fixture;
    const auto model = manifest();
    const auto block = attention_block(fixture);
    oracle::runtime::KvCache cache(1,
                                    model.attention_head_count_kv,
                                    model.attention_key_length,
                                    model.attention_value_length);
    const std::vector<float> input{1, 2, 3, 4};

    oracle::runtime::Qwen35MatvecOverrides overrides;
    overrides.tensors.push_back({"attn_output", {0.25F, 0.5F, 0.75F, 1.0F}});
    overrides.tensors.push_back({"ffn_gate", zeros(4)});
    overrides.tensors.push_back({"ffn_up", zeros(4)});
    overrides.tensors.push_back({"ffn_out", zeros(4)});

    const auto result = oracle::runtime::execute_qwen35_attention_block_reference(
        model,
        block,
        input,
        cache,
        oracle::runtime::RopePosition::text(0),
        true,
        &overrides);

    const std::vector<float> expected{1.25F, 2.5F, 3.75F, 5.0F};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_close(result.output[index], expected[index], 1.0e-6F,
                      "matvec override output mismatch");
    }
    const auto* branch = result.trace.find("attn_output");
    require(branch != nullptr && branch->values == overrides.tensors[0].values,
            "attention output override was not captured exactly");

    oracle::runtime::Qwen35MatvecOverrides wrong_count;
    wrong_count.tensors.push_back({"attn_output", {1.0F}});
    oracle::runtime::KvCache fresh_cache(1, 1, 2, 2);
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::execute_qwen35_attention_block_reference(
                model,
                block,
                input,
                fresh_cache,
                oracle::runtime::RopePosition::text(0),
                false,
                &wrong_count));
        },
        "override count mismatch");

    oracle::runtime::Qwen35MatvecOverrides unknown;
    unknown.tensors.push_back({"not_a_projection", {1.0F}});
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::execute_qwen35_attention_block_reference(
                model,
                block,
                input,
                fresh_cache,
                oracle::runtime::RopePosition::text(0),
                false,
                &unknown));
        },
        "unknown Qwen3.5 matvec override");
}

void test_trace_reports_and_errors() {
    Fixture fixture;
    const auto model = manifest();
    const auto block = attention_block(fixture);
    oracle::runtime::KvCache cache(1, 1, 2, 2);
    const auto result = oracle::runtime::execute_qwen35_attention_block_reference(
        model,
        block,
        std::vector<float>{1, 2, 3, 4},
        cache,
        oracle::runtime::RopePosition::text(0),
        true);
    const std::string text = oracle::runtime::qwen35_block_trace_text(result.trace);
    const std::string json = oracle::runtime::qwen35_block_trace_json(result.trace);
    require(text.find("attn_gated") != std::string::npos, "text trace omitted a tensor");
    require(json.find("\"kind\":\"attention\"") != std::string::npos,
            "JSON trace kind mismatch");
    require(json.find("\"values\"") != std::string::npos, "JSON trace omitted values");

    const auto recurrent = recurrent_block(fixture);
    oracle::runtime::SsmState wrong_state(1, 1, 1, 1, 1);
    require_throws(
        [&] {
            static_cast<void>(oracle::runtime::execute_qwen35_recurrent_block_reference(
                model, recurrent, std::vector<float>{1, 2, 3, 4}, wrong_state, false));
        },
        "state dimensions");

    oracle::runtime::Qwen35BlockTrace non_finite;
    non_finite.tensors.push_back({"broken", {std::numeric_limits<float>::infinity()}});
    require_throws(
        [&] { static_cast<void>(oracle::runtime::qwen35_block_trace_json(non_finite)); },
        "non-finite");
}

}  // namespace

int main() {
    try {
        test_recurrent_block();
        test_gated_delta_uses_ggml_l2_epsilon_contract();
        test_attention_block();
        test_matvec_overrides();
        test_trace_reports_and_errors();
        std::cout << "Phase 2C tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Phase 2C test failure: " << error.what() << '\n';
        return 1;
    }
}
