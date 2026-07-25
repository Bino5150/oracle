#pragma once

#include "oracle/model/gguf.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace oracle::model {

struct Qwen35Manifest {
    std::string architecture;
    std::string model_name;

    std::uint32_t total_block_count{0};
    std::uint32_t backbone_block_count{0};
    std::uint32_t nextn_predict_layers{0};
    std::uint32_t context_length{0};
    std::uint32_t embedding_length{0};
    std::uint32_t feed_forward_length{0};

    std::uint32_t attention_head_count{0};
    std::uint32_t attention_head_count_kv{0};
    std::uint32_t attention_key_length{0};
    std::uint32_t attention_value_length{0};
    float attention_rms_epsilon{0.0F};

    std::uint32_t rope_dimension_count{0};
    std::array<std::int32_t, 4> rope_dimension_sections{};
    float rope_frequency_base{0.0F};

    std::uint32_t ssm_convolution_kernel{0};
    std::uint32_t ssm_state_size{0};
    std::uint32_t ssm_group_count{0};
    std::uint32_t ssm_time_step_rank{0};
    std::uint32_t ssm_inner_size{0};
    std::uint32_t full_attention_interval{0};

    std::uint32_t vocabulary_size{0};
    std::optional<std::uint32_t> bos_token_id;
    std::optional<std::uint32_t> eos_token_id;
    std::optional<std::uint32_t> padding_token_id;

    [[nodiscard]] bool has_mtp() const noexcept;
    [[nodiscard]] bool is_full_attention_block(std::uint32_t block_index) const noexcept;
};

[[nodiscard]] Qwen35Manifest load_qwen35_manifest(const GgufFile& file);
[[nodiscard]] std::string qwen35_manifest_json(const Qwen35Manifest& manifest);

}  // namespace oracle::model
