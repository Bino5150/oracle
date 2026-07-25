#include "oracle/core/tensor.hpp"
#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/reference_kernels.hpp"
#include "oracle/runtime/sampler.hpp"
#include "oracle/runtime/tiny_hybrid_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(float actual, float expected, float tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

[[nodiscard]] oracle::model::Qwen35Manifest real_shape_manifest() {
    oracle::model::Qwen35Manifest manifest;
    manifest.architecture = "qwen35";
    manifest.total_block_count = 32;
    manifest.backbone_block_count = 32;
    manifest.context_length = 262144;
    manifest.embedding_length = 2560;
    manifest.feed_forward_length = 9216;
    manifest.attention_head_count = 16;
    manifest.attention_head_count_kv = 4;
    manifest.attention_key_length = 256;
    manifest.attention_value_length = 256;
    manifest.rope_dimension_count = 64;
    manifest.rope_dimension_sections = {11, 11, 10, 0};
    manifest.rope_frequency_base = 10000000.0F;
    manifest.attention_rms_epsilon = 1.0e-6F;
    manifest.ssm_convolution_kernel = 4;
    manifest.ssm_state_size = 128;
    manifest.ssm_group_count = 16;
    manifest.ssm_time_step_rank = 32;
    manifest.ssm_inner_size = 4096;
    manifest.full_attention_interval = 4;
    manifest.vocabulary_size = 248320;
    return manifest;
}

void test_embedding_lookup() {
    oracle::core::Tensor embedding({3, 4});
    for (std::size_t index = 0; index < embedding.element_count(); ++index) {
        embedding.data()[index] = static_cast<float>(index);
    }
    oracle::core::Tensor output({2, 4});
    const std::array<std::uint32_t, 2> tokens{2, 0};
    oracle::runtime::embedding_lookup(embedding, tokens, output);
    const std::array<float, 8> expected{8, 9, 10, 11, 0, 1, 2, 3};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_near(output.data()[index], expected[index], 0.0F,
                     "embedding lookup mismatch");
    }
    const std::array<std::uint32_t, 1> invalid{3};
    oracle::core::Tensor one({1, 4});
    require_throws([&] { oracle::runtime::embedding_lookup(embedding, invalid, one); },
                   "embedding lookup must reject out-of-range token ids");
}

void test_rope() {
    oracle::core::Tensor head({1, 4});
    head.data()[0] = 1.0F;
    head.data()[1] = 2.0F;
    head.data()[2] = 3.0F;
    head.data()[3] = 4.0F;
    oracle::runtime::apply_qwen35_rope(head,
                                       oracle::runtime::RopePosition::text(1),
                                       4,
                                       {2, 0, 0, 0},
                                       1.0F);
    const float cosine = std::cos(1.0F);
    const float sine = std::sin(1.0F);
    require_near(head.data()[0], 1.0F * cosine - 3.0F * sine, 1.0e-6F,
                 "RoPE first half mismatch");
    require_near(head.data()[1], 2.0F * cosine - 4.0F * sine, 1.0e-6F,
                 "RoPE first half mismatch");
    require_near(head.data()[2], 3.0F * cosine + 1.0F * sine, 1.0e-6F,
                 "RoPE second half mismatch");
    require_near(head.data()[3], 4.0F * cosine + 2.0F * sine, 1.0e-6F,
                 "RoPE second half mismatch");
    double norm = 0.0;
    for (const float value : head.data()) {
        norm += static_cast<double>(value) * value;
    }
    require_near(static_cast<float>(norm), 30.0F, 1.0e-5F,
                 "RoPE must preserve vector norm");

    oracle::core::Tensor multimodal({1, 6});
    multimodal.fill(0.0F);
    multimodal.data()[0] = 1.0F;
    multimodal.data()[1] = 1.0F;
    multimodal.data()[2] = 1.0F;
    oracle::runtime::apply_qwen35_rope(multimodal,
                                       {1, 2, 3},
                                       6,
                                       {1, 1, 1, 0},
                                       1.0F);
    require_near(multimodal.data()[0], std::cos(1.0F), 1.0e-6F,
                 "MRoPE temporal frequency mismatch");
    require_near(multimodal.data()[1], std::cos(2.0F), 1.0e-6F,
                 "MRoPE height frequency mismatch");
    require_near(multimodal.data()[2], std::cos(3.0F), 1.0e-6F,
                 "MRoPE width frequency mismatch");
    require_near(multimodal.data()[3], std::sin(1.0F), 1.0e-6F,
                 "MRoPE temporal sine mismatch");
    require_near(multimodal.data()[4], std::sin(2.0F), 1.0e-6F,
                 "MRoPE height sine mismatch");
    require_near(multimodal.data()[5], std::sin(3.0F), 1.0e-6F,
                 "MRoPE width sine mismatch");
}

void test_kv_cache_and_attention() {
    oracle::runtime::KvCache cache(2, 1, 2, 2);
    oracle::core::Tensor query({2, 2});
    query.fill(0.0F);
    query.data()[0] = 1.0F;
    query.data()[2] = 1.0F;
    oracle::core::Tensor key({1, 2});
    key.fill(0.0F);
    key.data()[0] = 1.0F;
    oracle::core::Tensor value({1, 2});
    value.data()[0] = 2.0F;
    value.data()[1] = 3.0F;
    oracle::core::Tensor output({2, 2});
    oracle::runtime::causal_attention_step(query, key, value, cache, output);
    require(cache.size() == 1, "attention step must append to KV cache");
    for (std::size_t head = 0; head < 2; ++head) {
        require_near(output.data()[head * 2], 2.0F, 1.0e-6F,
                     "single-token attention value mismatch");
        require_near(output.data()[head * 2 + 1], 3.0F, 1.0e-6F,
                     "single-token attention value mismatch");
    }

    query.fill(0.0F);
    query.data()[1] = 1.0F;
    query.data()[3] = 1.0F;
    key.fill(0.0F);
    key.data()[1] = 1.0F;
    value.data()[0] = 4.0F;
    value.data()[1] = 5.0F;
    oracle::runtime::causal_attention_step(query, key, value, cache, output);
    const float new_score = 1.0F / std::sqrt(2.0F);
    const float new_probability = std::exp(new_score) / (1.0F + std::exp(new_score));
    const float old_probability = 1.0F - new_probability;
    require_near(output.data()[0], old_probability * 2.0F + new_probability * 4.0F,
                 1.0e-6F, "cached attention weighted value mismatch");
    require_near(output.data()[1], old_probability * 3.0F + new_probability * 5.0F,
                 1.0e-6F, "cached attention weighted value mismatch");
    require_throws([&] { oracle::runtime::causal_attention_step(query, key, value, cache, output); },
                   "KV cache must enforce its capacity");
    cache.reset();
    require(cache.size() == 0, "KV cache reset must clear logical length");
}

void test_ssm_primitives() {
    oracle::runtime::SsmState convolution_state(1, 3, 1, 2, 1);
    const std::array<float, 3> weights{1.0F, 2.0F, 3.0F};
    std::array<float, 1> output{};
    const std::array<float, 1> first{1.0F};
    oracle::runtime::causal_depthwise_convolution_step(first,
                                                        weights,
                                                        convolution_state,
                                                        output,
                                                        false);
    require_near(output[0], 3.0F, 1.0e-6F,
                 "causal convolution first step mismatch");
    const std::array<float, 1> second{2.0F};
    oracle::runtime::causal_depthwise_convolution_step(second,
                                                        weights,
                                                        convolution_state,
                                                        output,
                                                        false);
    require_near(output[0], 8.0F, 1.0e-6F,
                 "causal convolution second step mismatch");

    oracle::runtime::SsmState state(1, 1, 1, 2, 1);
    oracle::core::Tensor query({1, 2});
    oracle::core::Tensor key({1, 2});
    oracle::core::Tensor value({1, 1});
    oracle::core::Tensor mixed({1, 1});
    query.fill(0.0F);
    key.fill(0.0F);
    query.data()[0] = 1.0F;
    key.data()[0] = 1.0F;
    value.data()[0] = 2.0F;
    const std::array<float, 1> beta_first{0.5F};
    const std::array<float, 1> no_decay{0.0F};
    oracle::runtime::gated_delta_step(query,
                                      key,
                                      value,
                                      beta_first,
                                      no_decay,
                                      state,
                                      mixed,
                                      1.0e-12F);
    require_near(mixed.data()[0], 1.0F / std::sqrt(2.0F), 2.0e-6F,
                 "gated delta first step mismatch");

    const std::array<float, 1> beta_second{1.0F};
    const std::array<float, 1> half_decay{std::log(0.5F)};
    oracle::runtime::gated_delta_step(query,
                                      key,
                                      value,
                                      beta_second,
                                      half_decay,
                                      state,
                                      mixed,
                                      1.0e-12F);
    require_near(mixed.data()[0], std::sqrt(2.0F), 3.0e-6F,
                 "gated delta recurrent step mismatch");
    require(state.sequence_length() == 2,
            "gated delta state must track sequence length");
}

void test_cache_plan() {
    const auto manifest = real_shape_manifest();
    const auto layout = oracle::runtime::qwen35_ssm_layout(manifest);
    require(layout.key_heads == 16 && layout.value_heads == 32,
            "Qwen3.5 SSM head layout mismatch");
    require(layout.key_head_dimension == 128 && layout.value_head_dimension == 128,
            "Qwen3.5 SSM head dimensions mismatch");
    require(layout.convolution_channels == 8192 && layout.convolution_kernel == 4,
            "Qwen3.5 SSM convolution layout mismatch");

    const auto plan = oracle::runtime::plan_qwen35_cache(manifest, 4096);
    require(plan.attention_layers == 8 && plan.ssm_layers == 24,
            "hybrid cache block counts mismatch");
    require(plan.kv_bytes_per_attention_layer == 33554432,
            "KV bytes per attention layer mismatch");
    require(plan.convolution_bytes_per_ssm_layer == 131072,
            "SSM convolution bytes mismatch");
    require(plan.recurrent_bytes_per_ssm_layer == 2097152,
            "SSM recurrent bytes mismatch");
    require(plan.total_bytes == 321912832,
            "total hybrid cache plan mismatch");
    require_throws([&] { static_cast<void>(oracle::runtime::plan_qwen35_cache(manifest, 262145)); },
                   "cache planning must enforce context length");

    auto tiny = manifest;
    tiny.total_block_count = 4;
    tiny.backbone_block_count = 4;
    tiny.context_length = 16;
    tiny.attention_head_count = 2;
    tiny.attention_head_count_kv = 1;
    tiny.attention_key_length = 2;
    tiny.attention_value_length = 2;
    tiny.ssm_group_count = 1;
    tiny.ssm_time_step_rank = 1;
    tiny.ssm_state_size = 2;
    tiny.ssm_inner_size = 2;
    tiny.ssm_convolution_kernel = 2;
    oracle::runtime::HybridCache cache(tiny, 8);
    require(cache.block_count() == 4 && cache.plan().attention_layers == 1,
            "hybrid cache construction mismatch");
    require(!cache.is_attention_block(0) && cache.is_attention_block(3),
            "hybrid cache block typing mismatch");
    static_cast<void>(cache.ssm(0));
    static_cast<void>(cache.attention(3));
    require_throws([&] { static_cast<void>(cache.attention(0)); },
                   "hybrid cache must reject wrong block-state access");
}

void test_sampler() {
    const std::array<float, 4> logits{0.0F, 1.0F, 3.0F, 2.0F};
    oracle::runtime::Sampler greedy;
    const auto greedy_result = greedy.sample(logits);
    require(greedy_result.token_id == 2 && greedy_result.candidate_count == 1,
            "greedy sampler mismatch");

    oracle::runtime::Sampler first({1.0F, 2, 1.0F, 5150});
    oracle::runtime::Sampler second({1.0F, 2, 1.0F, 5150});
    const auto first_result = first.sample(logits);
    const auto second_result = second.sample(logits);
    require(first_result.token_id == second_result.token_id &&
                first_result.probability == second_result.probability &&
                first_result.candidate_count == 2,
            "seeded sampler must be deterministic");
    require(first_result.token_id == 2 || first_result.token_id == 3,
            "top-k sampler selected a filtered token");

    oracle::runtime::Sampler nucleus({1.0F, 0, 0.5F, 5150});
    const auto nucleus_result = nucleus.sample(logits);
    require(nucleus_result.token_id == 2 && nucleus_result.candidate_count == 1,
            "top-p sampler mismatch");
}

[[nodiscard]] float maximum_difference(std::span<const float> left,
                                       std::span<const float> right) {
    require(left.size() == right.size(), "logit vector sizes differ");
    float maximum = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

void test_tiny_hybrid_model() {
    oracle::runtime::TinyHybridModel model({}, 5150);
    oracle::runtime::TinyHybridState cached_state(model.config(), 16);
    const std::array<std::uint32_t, 3> prompt{1, 5, 2};
    auto cached_logits = model.prefill(prompt, cached_state);
    require(cached_state.sequence_length() == prompt.size(),
            "tiny prefill state length mismatch");
    for (const float logit : cached_logits.data()) {
        require(std::isfinite(logit), "tiny model produced non-finite logits");
    }

    oracle::runtime::TinyHybridState replay_state(model.config(), 16);
    auto replay_logits = model.prefill(prompt, replay_state);
    require(maximum_difference(cached_logits.data(), replay_logits.data()) == 0.0F,
            "tiny cached and replay prefill diverged");

    oracle::runtime::Sampler greedy;
    const std::uint32_t next = greedy.sample(cached_logits.data()).token_id;
    cached_logits = model.forward_token(next, cached_state);
    const std::array<std::uint32_t, 4> extended{1, 5, 2, next};
    oracle::runtime::TinyHybridState full_replay_state(model.config(), 16);
    replay_logits = model.prefill(extended, full_replay_state);
    require(maximum_difference(cached_logits.data(), replay_logits.data()) == 0.0F,
            "tiny cached decode and full replay diverged");

    oracle::runtime::TinyHybridState generation_a(model.config(), 16);
    oracle::runtime::TinyHybridState generation_b(model.config(), 16);
    oracle::runtime::Sampler sampler_a({0.8F, 4, 0.9F, 5150});
    oracle::runtime::Sampler sampler_b({0.8F, 4, 0.9F, 5150});
    const auto generated_a = model.generate(prompt, 4, generation_a, sampler_a);
    const auto generated_b = model.generate(prompt, 4, generation_b, sampler_b);
    require(generated_a == generated_b,
            "tiny seeded generation must be deterministic");
    require(generation_a.sequence_length() == prompt.size() + generated_a.size(),
            "tiny generation state length mismatch");
    require(generation_a.byte_size() > 0,
            "tiny hybrid state must report allocated bytes");
}

}  // namespace

int main() {
    try {
        test_embedding_lookup();
        test_rope();
        test_kv_cache_and_attention();
        test_ssm_primitives();
        test_cache_plan();
        test_sampler();
        test_tiny_hybrid_model();
        std::cout << "Phase 1E tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 1E test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
