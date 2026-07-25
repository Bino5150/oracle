#pragma once

#include "oracle/model/mapped_gguf.hpp"
#include "oracle/model/qwen35_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oracle::model {

enum class Qwen35BlockKind {
    recurrent,
    full_attention,
};

[[nodiscard]] std::string_view qwen35_block_kind_name(Qwen35BlockKind kind) noexcept;

struct Qwen35MlpWeights {
    const GgufTensorView* gate{nullptr};
    const GgufTensorView* down{nullptr};
    const GgufTensorView* up{nullptr};
};

struct Qwen35AttentionWeights {
    const GgufTensorView* query{nullptr};
    const GgufTensorView* key{nullptr};
    const GgufTensorView* value{nullptr};
    const GgufTensorView* output{nullptr};
    const GgufTensorView* query_norm{nullptr};
    const GgufTensorView* key_norm{nullptr};
};

struct Qwen35RecurrentWeights {
    const GgufTensorView* qkv{nullptr};
    const GgufTensorView* gate{nullptr};
    const GgufTensorView* convolution{nullptr};
    const GgufTensorView* time_step_bias{nullptr};
    const GgufTensorView* decay{nullptr};
    const GgufTensorView* beta{nullptr};
    const GgufTensorView* alpha{nullptr};
    const GgufTensorView* norm{nullptr};
    const GgufTensorView* output{nullptr};
};

struct Qwen35BackboneBlockWeights {
    std::uint32_t index{0};
    Qwen35BlockKind kind{Qwen35BlockKind::recurrent};
    const GgufTensorView* input_norm{nullptr};
    const GgufTensorView* post_attention_norm{nullptr};
    Qwen35MlpWeights mlp;
    std::optional<Qwen35AttentionWeights> attention;
    std::optional<Qwen35RecurrentWeights> recurrent;
};

struct Qwen35MtpWeights {
    std::uint32_t block_index{0};
    const GgufTensorView* input_norm{nullptr};
    const GgufTensorView* post_attention_norm{nullptr};
    Qwen35AttentionWeights attention;
    Qwen35MlpWeights mlp;
    const GgufTensorView* embedding_hidden_projection{nullptr};
    const GgufTensorView* embedding_norm{nullptr};
    const GgufTensorView* hidden_norm{nullptr};
    const GgufTensorView* embed_tokens{nullptr};
    const GgufTensorView* shared_head{nullptr};
    const GgufTensorView* shared_head_norm{nullptr};
};

struct Qwen35BlockBindingSummary {
    std::uint32_t index{0};
    Qwen35BlockKind kind{Qwen35BlockKind::recurrent};
    std::size_t tensor_count{0};
    std::uint64_t mapped_bytes{0};
};

struct Qwen35BindingReport {
    std::uint32_t backbone_block_count{0};
    std::uint32_t recurrent_block_count{0};
    std::uint32_t attention_block_count{0};
    std::uint32_t nextn_predict_layers{0};
    std::size_t required_tensor_count{0};
    std::size_t optional_tensor_count{0};
    std::size_t bound_tensor_count{0};
    std::size_t unexpected_tensor_count{0};
    std::uint64_t bound_bytes{0};
    bool tied_output{false};
    std::vector<std::string> unexpected_tensors;
    std::vector<Qwen35BlockBindingSummary> blocks;
};

struct Qwen35Weights {
    const GgufTensorView* token_embedding{nullptr};
    const GgufTensorView* output_norm{nullptr};
    const GgufTensorView* output{nullptr};
    bool output_is_tied{false};
    std::vector<Qwen35BackboneBlockWeights> blocks;
    std::optional<Qwen35MtpWeights> mtp;
    Qwen35BindingReport report;
};

[[nodiscard]] Qwen35Weights bind_qwen35_weights(std::span<const GgufTensorView> tensors,
                                                const Qwen35Manifest& manifest);
[[nodiscard]] Qwen35Weights bind_qwen35_weights(const MappedGgufModel& model,
                                                const Qwen35Manifest& manifest);

[[nodiscard]] std::string qwen35_binding_report_text(const Qwen35Weights& weights);
[[nodiscard]] std::string qwen35_binding_report_json(const Qwen35Weights& weights);

}  // namespace oracle::model
