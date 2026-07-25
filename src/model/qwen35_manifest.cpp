#include "oracle/model/qwen35_manifest.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace oracle::model {
namespace {

[[nodiscard]] const GgufValue& require_value(const GgufFile& file, std::string_view key) {
    const GgufMetadataEntry* entry = file.find_metadata(key);
    if (entry == nullptr) {
        throw std::runtime_error("missing required GGUF metadata: " + std::string(key));
    }
    return entry->value;
}

[[nodiscard]] std::uint32_t require_u32(const GgufFile& file, std::string_view key) {
    const GgufValue& value = require_value(file, key);
    const auto* typed = value.get_if<std::uint32_t>();
    if (typed == nullptr) {
        throw std::runtime_error("GGUF metadata must be uint32: " + std::string(key));
    }
    return *typed;
}

[[nodiscard]] std::optional<std::uint32_t> optional_u32(const GgufFile& file,
                                                         std::string_view key) {
    const GgufMetadataEntry* entry = file.find_metadata(key);
    if (entry == nullptr) {
        return std::nullopt;
    }
    const auto* typed = entry->value.get_if<std::uint32_t>();
    if (typed == nullptr) {
        throw std::runtime_error("GGUF metadata must be uint32: " + std::string(key));
    }
    return *typed;
}

[[nodiscard]] float require_f32(const GgufFile& file, std::string_view key) {
    const GgufValue& value = require_value(file, key);
    const auto* typed = value.get_if<float>();
    if (typed == nullptr) {
        throw std::runtime_error("GGUF metadata must be float32: " + std::string(key));
    }
    return *typed;
}

[[nodiscard]] std::string require_string(const GgufFile& file, std::string_view key) {
    const GgufValue& value = require_value(file, key);
    const auto* typed = value.get_if<std::string>();
    if (typed == nullptr) {
        throw std::runtime_error("GGUF metadata must be string: " + std::string(key));
    }
    return *typed;
}

[[nodiscard]] std::string optional_string(const GgufFile& file,
                                          std::string_view key,
                                          std::string fallback = {}) {
    const GgufMetadataEntry* entry = file.find_metadata(key);
    if (entry == nullptr) {
        return fallback;
    }
    const auto* typed = entry->value.get_if<std::string>();
    if (typed == nullptr) {
        throw std::runtime_error("GGUF metadata must be string: " + std::string(key));
    }
    return *typed;
}

[[nodiscard]] const GgufArray& require_array(const GgufFile& file,
                                             std::string_view key,
                                             GgufMetadataType element_type) {
    const GgufValue& value = require_value(file, key);
    const auto* array_ptr = value.get_if<std::shared_ptr<GgufArray>>();
    if (array_ptr == nullptr || *array_ptr == nullptr) {
        throw std::runtime_error("GGUF metadata must be an array: " + std::string(key));
    }
    if ((*array_ptr)->element_type != element_type) {
        throw std::runtime_error("GGUF metadata array has wrong element type: " +
                                 std::string(key));
    }
    return **array_ptr;
}

[[nodiscard]] std::uint32_t checked_size_to_u32(std::size_t size,
                                                std::string_view context) {
    if (size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(context) + " exceeds uint32 range");
    }
    return static_cast<std::uint32_t>(size);
}

void require_positive(std::uint32_t value, std::string_view key) {
    if (value == 0) {
        throw std::runtime_error("GGUF metadata must be non-zero: " + std::string(key));
    }
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned int>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

}  // namespace

bool Qwen35Manifest::has_mtp() const noexcept { return nextn_predict_layers != 0; }

bool Qwen35Manifest::is_full_attention_block(std::uint32_t block_index) const noexcept {
    return block_index < backbone_block_count && full_attention_interval != 0 &&
           ((block_index + 1U) % full_attention_interval == 0U);
}

Qwen35Manifest load_qwen35_manifest(const GgufFile& file) {
    Qwen35Manifest manifest;
    manifest.architecture = require_string(file, "general.architecture");
    if (manifest.architecture != "qwen35") {
        throw std::runtime_error("unsupported architecture for Qwen35 manifest: " +
                                 manifest.architecture);
    }
    manifest.model_name = optional_string(file, "general.name", file.path.filename().string());

    manifest.total_block_count = require_u32(file, "qwen35.block_count");
    manifest.nextn_predict_layers = optional_u32(file, "qwen35.nextn_predict_layers").value_or(0);
    if (manifest.nextn_predict_layers > manifest.total_block_count) {
        throw std::runtime_error("qwen35.nextn_predict_layers exceeds qwen35.block_count");
    }
    manifest.backbone_block_count =
        manifest.total_block_count - manifest.nextn_predict_layers;

    manifest.context_length = require_u32(file, "qwen35.context_length");
    manifest.embedding_length = require_u32(file, "qwen35.embedding_length");
    manifest.feed_forward_length = require_u32(file, "qwen35.feed_forward_length");
    manifest.attention_head_count = require_u32(file, "qwen35.attention.head_count");
    manifest.attention_head_count_kv =
        require_u32(file, "qwen35.attention.head_count_kv");
    manifest.attention_key_length = require_u32(file, "qwen35.attention.key_length");
    manifest.attention_value_length = require_u32(file, "qwen35.attention.value_length");
    manifest.attention_rms_epsilon =
        require_f32(file, "qwen35.attention.layer_norm_rms_epsilon");

    manifest.rope_dimension_count = require_u32(file, "qwen35.rope.dimension_count");
    manifest.rope_frequency_base = require_f32(file, "qwen35.rope.freq_base");
    const GgufArray& sections = require_array(
        file, "qwen35.rope.dimension_sections", GgufMetadataType::int32);
    if (sections.values.size() != manifest.rope_dimension_sections.size()) {
        throw std::runtime_error("qwen35.rope.dimension_sections must contain four values");
    }
    std::int64_t section_sum = 0;
    for (std::size_t index = 0; index < sections.values.size(); ++index) {
        const auto* value = sections.values[index].get_if<std::int32_t>();
        if (value == nullptr || *value < 0) {
            throw std::runtime_error(
                "qwen35.rope.dimension_sections must contain non-negative int32 values");
        }
        manifest.rope_dimension_sections[index] = *value;
        section_sum += *value;
    }
    if (section_sum * 2 != static_cast<std::int64_t>(manifest.rope_dimension_count)) {
        throw std::runtime_error(
            "qwen35.rope.dimension_sections must sum to half rope.dimension_count");
    }

    manifest.ssm_convolution_kernel = require_u32(file, "qwen35.ssm.conv_kernel");
    manifest.ssm_state_size = require_u32(file, "qwen35.ssm.state_size");
    manifest.ssm_group_count = require_u32(file, "qwen35.ssm.group_count");
    manifest.ssm_time_step_rank = require_u32(file, "qwen35.ssm.time_step_rank");
    manifest.ssm_inner_size = require_u32(file, "qwen35.ssm.inner_size");
    manifest.full_attention_interval =
        require_u32(file, "qwen35.full_attention_interval");

    for (const auto& [value, key] : {
             std::pair{manifest.total_block_count, "qwen35.block_count"},
             std::pair{manifest.backbone_block_count, "Qwen35 backbone block count"},
             std::pair{manifest.context_length, "qwen35.context_length"},
             std::pair{manifest.embedding_length, "qwen35.embedding_length"},
             std::pair{manifest.feed_forward_length, "qwen35.feed_forward_length"},
             std::pair{manifest.attention_head_count, "qwen35.attention.head_count"},
             std::pair{manifest.attention_head_count_kv,
                       "qwen35.attention.head_count_kv"},
             std::pair{manifest.attention_key_length, "qwen35.attention.key_length"},
             std::pair{manifest.attention_value_length, "qwen35.attention.value_length"},
             std::pair{manifest.rope_dimension_count, "qwen35.rope.dimension_count"},
             std::pair{manifest.ssm_convolution_kernel, "qwen35.ssm.conv_kernel"},
             std::pair{manifest.ssm_state_size, "qwen35.ssm.state_size"},
             std::pair{manifest.ssm_group_count, "qwen35.ssm.group_count"},
             std::pair{manifest.ssm_time_step_rank, "qwen35.ssm.time_step_rank"},
             std::pair{manifest.ssm_inner_size, "qwen35.ssm.inner_size"},
             std::pair{manifest.full_attention_interval,
                       "qwen35.full_attention_interval"},
         }) {
        require_positive(value, key);
    }
    if (manifest.attention_head_count % manifest.attention_head_count_kv != 0) {
        throw std::runtime_error(
            "qwen35.attention.head_count must be divisible by head_count_kv");
    }
    if (!(manifest.attention_rms_epsilon > 0.0F)) {
        throw std::runtime_error("qwen35 RMS epsilon must be positive");
    }
    if (!(manifest.rope_frequency_base > 0.0F)) {
        throw std::runtime_error("qwen35 RoPE frequency base must be positive");
    }

    const GgufArray& tokens =
        require_array(file, "tokenizer.ggml.tokens", GgufMetadataType::string);
    const GgufArray& token_types =
        require_array(file, "tokenizer.ggml.token_type", GgufMetadataType::int32);
    if (tokens.values.empty() || tokens.values.size() != token_types.values.size()) {
        throw std::runtime_error(
            "tokenizer token and token-type arrays must be non-empty and equal length");
    }
    manifest.vocabulary_size =
        checked_size_to_u32(tokens.values.size(), "tokenizer vocabulary");
    manifest.bos_token_id = optional_u32(file, "tokenizer.ggml.bos_token_id");
    manifest.eos_token_id = optional_u32(file, "tokenizer.ggml.eos_token_id");
    manifest.padding_token_id = optional_u32(file, "tokenizer.ggml.padding_token_id");

    for (const auto& [token_id, key] : {
             std::pair{manifest.bos_token_id, "tokenizer.ggml.bos_token_id"},
             std::pair{manifest.eos_token_id, "tokenizer.ggml.eos_token_id"},
             std::pair{manifest.padding_token_id, "tokenizer.ggml.padding_token_id"},
         }) {
        if (token_id && *token_id >= manifest.vocabulary_size) {
            throw std::runtime_error(std::string(key) + " is outside the vocabulary");
        }
    }

    if (const GgufTensorInfo* embedding = file.find_tensor("token_embd.weight")) {
        if (embedding->dimensions.size() != 2 ||
            embedding->dimensions[0] != manifest.embedding_length ||
            embedding->dimensions[1] != manifest.vocabulary_size) {
            throw std::runtime_error(
                "token_embd.weight dimensions do not match manifest and vocabulary");
        }
    }

    if (manifest.has_mtp()) {
        const std::string prefix =
            "blk." + std::to_string(manifest.backbone_block_count) + ".nextn.";
        const bool found_nextn = std::ranges::any_of(
            file.tensors, [&prefix](const GgufTensorInfo& tensor) {
                return tensor.name.starts_with(prefix);
            });
        if (!file.tensors.empty() && !found_nextn) {
            throw std::runtime_error(
                "MTP metadata is present but no nextn tensors were found at the appended block");
        }
    }

    return manifest;
}

std::string qwen35_manifest_json(const Qwen35Manifest& manifest) {
    std::ostringstream output;
    output << "{\"architecture\":\"" << json_escape(manifest.architecture)
           << "\",\"model_name\":\"" << json_escape(manifest.model_name)
           << "\",\"total_block_count\":" << manifest.total_block_count
           << ",\"backbone_block_count\":" << manifest.backbone_block_count
           << ",\"nextn_predict_layers\":" << manifest.nextn_predict_layers
           << ",\"has_mtp\":" << (manifest.has_mtp() ? "true" : "false")
           << ",\"context_length\":" << manifest.context_length
           << ",\"embedding_length\":" << manifest.embedding_length
           << ",\"feed_forward_length\":" << manifest.feed_forward_length
           << ",\"attention_head_count\":" << manifest.attention_head_count
           << ",\"attention_head_count_kv\":" << manifest.attention_head_count_kv
           << ",\"attention_key_length\":" << manifest.attention_key_length
           << ",\"attention_value_length\":" << manifest.attention_value_length
           << ",\"attention_rms_epsilon\":" << std::setprecision(9)
           << manifest.attention_rms_epsilon << ",\"rope_dimension_count\":"
           << manifest.rope_dimension_count << ",\"rope_dimension_sections\":[";
    for (std::size_t index = 0; index < manifest.rope_dimension_sections.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << manifest.rope_dimension_sections[index];
    }
    output << "],\"rope_frequency_base\":" << std::setprecision(9)
           << manifest.rope_frequency_base
           << ",\"ssm_convolution_kernel\":" << manifest.ssm_convolution_kernel
           << ",\"ssm_state_size\":" << manifest.ssm_state_size
           << ",\"ssm_group_count\":" << manifest.ssm_group_count
           << ",\"ssm_time_step_rank\":" << manifest.ssm_time_step_rank
           << ",\"ssm_inner_size\":" << manifest.ssm_inner_size
           << ",\"full_attention_interval\":" << manifest.full_attention_interval
           << ",\"vocabulary_size\":" << manifest.vocabulary_size;
    const auto append_optional = [&output](std::string_view key,
                                           std::optional<std::uint32_t> value) {
        output << ",\"" << key << "\":";
        if (value) {
            output << *value;
        } else {
            output << "null";
        }
    };
    append_optional("bos_token_id", manifest.bos_token_id);
    append_optional("eos_token_id", manifest.eos_token_id);
    append_optional("padding_token_id", manifest.padding_token_id);
    output << '}';
    return output.str();
}

}  // namespace oracle::model
