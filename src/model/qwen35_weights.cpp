#include "oracle/model/qwen35_weights.hpp"

#include "oracle/model/ggml_type.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace oracle::model {
namespace {

inline constexpr std::uint32_t ggml_type_f32 = 0;
inline constexpr std::uint32_t ggml_type_f16 = 1;
inline constexpr std::uint32_t ggml_type_q8_0 = 8;
inline constexpr std::uint32_t ggml_type_q5_k = 13;
inline constexpr std::uint32_t ggml_type_q6_k = 14;
inline constexpr std::uint32_t ggml_type_bf16 = 30;

enum class TensorRole {
    vector,
    matrix,
};

[[nodiscard]] std::uint64_t checked_multiply(std::uint64_t left,
                                             std::uint64_t right,
                                             std::string_view context) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left * right;
}

[[nodiscard]] std::uint64_t checked_add(std::uint64_t left,
                                        std::uint64_t right,
                                        std::string_view context) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left + right;
}

[[nodiscard]] bool allowed_storage_type(std::uint32_t type, TensorRole role) noexcept {
    if (type == ggml_type_f32 || type == ggml_type_f16 || type == ggml_type_bf16) {
        return true;
    }
    // Q8_0 is a matrix-only storage encoding here (Phase 2F Slice 1B): it entered
    // Oracle's support matrix specifically because the real Qwen3.5-2B-MTP-GGUF
    // checkpoint stores blk.<n>.nextn.eh_proj.weight (a matrix) as Q8_0. No known
    // Qwen3.5 vector weight requires it, so vector policy is left unchanged.
    return role == TensorRole::matrix &&
           (type == ggml_type_q5_k || type == ggml_type_q6_k || type == ggml_type_q8_0);
}

[[nodiscard]] std::string shape_string(std::span<const std::uint64_t> shape) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << shape[index];
    }
    output << ']';
    return output.str();
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
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

class Binder {
public:
    Binder(std::span<const GgufTensorView> tensors, const Qwen35Manifest& manifest)
        : tensors_(tensors), manifest_(manifest) {
        index_.reserve(tensors_.size());
        consumed_.reserve(tensors_.size());
        for (const GgufTensorView& tensor : tensors_) {
            const auto [iterator, inserted] = index_.emplace(std::string(tensor.name()), &tensor);
            static_cast<void>(iterator);
            if (!inserted) {
                throw std::runtime_error("duplicate tensor in Qwen3.5 binding input: " +
                                         std::string(tensor.name()));
            }
        }
    }

    [[nodiscard]] Qwen35Weights bind() {
        validate_manifest();

        Qwen35Weights weights;
        weights.token_embedding = require("token_embd.weight",
                                          {manifest_.embedding_length,
                                           manifest_.vocabulary_size},
                                          TensorRole::matrix);
        weights.output_norm = require("output_norm.weight",
                                      {manifest_.embedding_length},
                                      TensorRole::vector);

        if (const GgufTensorView* explicit_output = optional(
                "output.weight",
                {manifest_.embedding_length, manifest_.vocabulary_size},
                TensorRole::matrix)) {
            weights.output = explicit_output;
        } else {
            weights.output = weights.token_embedding;
            weights.output_is_tied = true;
        }

        weights.blocks.reserve(manifest_.backbone_block_count);
        for (std::uint32_t index = 0; index < manifest_.backbone_block_count; ++index) {
            weights.blocks.push_back(bind_backbone_block(index));
        }

        if (manifest_.has_mtp()) {
            if (manifest_.nextn_predict_layers != 1U) {
                throw std::runtime_error(
                    "Phase 2B Qwen3.5 binding currently supports exactly one NextN layer");
            }
            weights.mtp = bind_mtp(manifest_.backbone_block_count);
        }

        weights.report = make_report(weights);
        return weights;
    }

private:
    void validate_manifest() const {
        if (manifest_.architecture != "qwen35") {
            throw std::runtime_error("Qwen3.5 weight binder received architecture: " +
                                     manifest_.architecture);
        }
        if (manifest_.backbone_block_count == 0 || manifest_.embedding_length == 0 ||
            manifest_.feed_forward_length == 0 || manifest_.vocabulary_size == 0) {
            throw std::runtime_error("Qwen3.5 manifest is incomplete for weight binding");
        }
        const std::uint64_t expected_total_blocks = checked_add(
            manifest_.backbone_block_count,
            manifest_.nextn_predict_layers,
            "Qwen3.5 total block count");
        if (expected_total_blocks != manifest_.total_block_count) {
            throw std::runtime_error(
                "Qwen3.5 total block count does not equal backbone plus NextN layers");
        }
        if (manifest_.full_attention_interval == 0) {
            throw std::runtime_error("Qwen3.5 full-attention interval is zero");
        }
        if (manifest_.attention_head_count == 0 || manifest_.attention_head_count_kv == 0 ||
            manifest_.attention_key_length == 0 || manifest_.attention_value_length == 0) {
            throw std::runtime_error("Qwen3.5 attention dimensions are incomplete");
        }
        if (manifest_.attention_key_length != manifest_.attention_value_length) {
            throw std::runtime_error(
                "Phase 2B Qwen3.5 binding requires equal attention key and value widths");
        }
        if (manifest_.ssm_state_size == 0 || manifest_.ssm_group_count == 0 ||
            manifest_.ssm_time_step_rank == 0 || manifest_.ssm_convolution_kernel == 0 ||
            manifest_.ssm_inner_size == 0) {
            throw std::runtime_error("Qwen3.5 recurrent dimensions are incomplete");
        }
        const std::uint64_t recurrent_value_width = checked_multiply(
            manifest_.ssm_state_size,
            manifest_.ssm_time_step_rank,
            "Qwen3.5 recurrent value width");
        if (recurrent_value_width != manifest_.ssm_inner_size) {
            throw std::runtime_error(
                "Qwen3.5 SSM inner size does not match state size times time-step rank");
        }
    }

    [[nodiscard]] const GgufTensorView* lookup(std::string_view name) const noexcept {
        const auto iterator = index_.find(std::string(name));
        return iterator == index_.end() ? nullptr : iterator->second;
    }

    [[nodiscard]] const GgufTensorView* require(
        std::string name,
        std::initializer_list<std::uint64_t> expected_shape,
        TensorRole role) {
        const GgufTensorView* tensor = lookup(name);
        if (tensor == nullptr) {
            throw std::runtime_error("missing required Qwen3.5 tensor: " + name);
        }
        validate_tensor(*tensor, expected_shape, role);
        mark_consumed(*tensor, false);
        return tensor;
    }

    [[nodiscard]] const GgufTensorView* optional(
        std::string name,
        std::initializer_list<std::uint64_t> expected_shape,
        TensorRole role) {
        const GgufTensorView* tensor = lookup(name);
        if (tensor == nullptr) {
            return nullptr;
        }
        validate_tensor(*tensor, expected_shape, role);
        mark_consumed(*tensor, true);
        return tensor;
    }

    void validate_tensor(const GgufTensorView& tensor,
                         std::initializer_list<std::uint64_t> expected_shape,
                         TensorRole role) const {
        const std::span<const std::uint64_t> actual = tensor.dimensions();
        const std::span<const std::uint64_t> expected(expected_shape.begin(),
                                                      expected_shape.size());
        if (!std::ranges::equal(actual, expected)) {
            throw std::runtime_error(std::string(tensor.name()) + ": expected dimensions " +
                                     shape_string(expected) + ", found " +
                                     shape_string(actual));
        }
        if (!allowed_storage_type(tensor.layout().type, role)) {
            throw std::runtime_error(std::string(tensor.name()) +
                                     ": unsupported storage type " +
                                     std::string(tensor.layout().name) + " for " +
                                     (role == TensorRole::vector ? "vector" : "matrix") +
                                     " weight");
        }
        const std::uint64_t expected_bytes =
            ggml_tensor_byte_size(tensor.dimensions(), tensor.layout().type);
        if (expected_bytes != tensor.bytes().size()) {
            throw std::runtime_error(std::string(tensor.name()) +
                                     ": mapped byte size does not match tensor geometry");
        }
    }

    void mark_consumed(const GgufTensorView& tensor, bool optional_tensor) {
        const auto [iterator, inserted] = consumed_.insert(std::string(tensor.name()));
        static_cast<void>(iterator);
        if (!inserted) {
            throw std::logic_error("Qwen3.5 tensor assigned to more than one role: " +
                                   std::string(tensor.name()));
        }
        bound_bytes_ = checked_add(bound_bytes_, tensor.bytes().size(),
                                   "Qwen3.5 bound tensor bytes");
        if (optional_tensor) {
            ++optional_tensor_count_;
        } else {
            ++required_tensor_count_;
        }
    }

    [[nodiscard]] Qwen35MlpWeights bind_mlp(std::uint32_t index) {
        const std::string prefix = "blk." + std::to_string(index) + ".";
        return {
            require(prefix + "ffn_gate.weight",
                    {manifest_.embedding_length, manifest_.feed_forward_length},
                    TensorRole::matrix),
            require(prefix + "ffn_down.weight",
                    {manifest_.feed_forward_length, manifest_.embedding_length},
                    TensorRole::matrix),
            require(prefix + "ffn_up.weight",
                    {manifest_.embedding_length, manifest_.feed_forward_length},
                    TensorRole::matrix),
        };
    }

    [[nodiscard]] Qwen35AttentionWeights bind_attention(std::uint32_t index) {
        const std::string prefix = "blk." + std::to_string(index) + ".";
        const std::uint64_t query_width = checked_multiply(
            checked_multiply(manifest_.attention_key_length,
                             manifest_.attention_head_count,
                             "Qwen3.5 query width"),
            2,
            "Qwen3.5 gated query width");
        const std::uint64_t key_width = checked_multiply(
            manifest_.attention_key_length,
            manifest_.attention_head_count_kv,
            "Qwen3.5 key width");
        const std::uint64_t value_width = checked_multiply(
            manifest_.attention_value_length,
            manifest_.attention_head_count_kv,
            "Qwen3.5 value width");
        const std::uint64_t attention_output_width = checked_multiply(
            manifest_.attention_key_length,
            manifest_.attention_head_count,
            "Qwen3.5 attention output width");

        return {
            require(prefix + "attn_q.weight",
                    {manifest_.embedding_length, query_width},
                    TensorRole::matrix),
            require(prefix + "attn_k.weight",
                    {manifest_.embedding_length, key_width},
                    TensorRole::matrix),
            require(prefix + "attn_v.weight",
                    {manifest_.embedding_length, value_width},
                    TensorRole::matrix),
            require(prefix + "attn_output.weight",
                    {attention_output_width, manifest_.embedding_length},
                    TensorRole::matrix),
            require(prefix + "attn_q_norm.weight",
                    {manifest_.attention_key_length},
                    TensorRole::vector),
            require(prefix + "attn_k_norm.weight",
                    {manifest_.attention_key_length},
                    TensorRole::vector),
        };
    }

    [[nodiscard]] Qwen35RecurrentWeights bind_recurrent(std::uint32_t index) {
        const std::string prefix = "blk." + std::to_string(index) + ".";
        const std::uint64_t key_width = checked_multiply(
            manifest_.ssm_state_size,
            manifest_.ssm_group_count,
            "Qwen3.5 recurrent key width");
        const std::uint64_t value_width = checked_multiply(
            manifest_.ssm_state_size,
            manifest_.ssm_time_step_rank,
            "Qwen3.5 recurrent value width");
        const std::uint64_t convolution_width = checked_add(
            checked_multiply(key_width, 2, "Qwen3.5 recurrent doubled key width"),
            value_width,
            "Qwen3.5 recurrent convolution width");

        return {
            require(prefix + "attn_qkv.weight",
                    {manifest_.embedding_length, convolution_width},
                    TensorRole::matrix),
            require(prefix + "attn_gate.weight",
                    {manifest_.embedding_length, value_width},
                    TensorRole::matrix),
            require(prefix + "ssm_conv1d.weight",
                    {manifest_.ssm_convolution_kernel, convolution_width},
                    TensorRole::matrix),
            require(prefix + "ssm_dt.bias",
                    {manifest_.ssm_time_step_rank},
                    TensorRole::vector),
            require(prefix + "ssm_a",
                    {manifest_.ssm_time_step_rank},
                    TensorRole::vector),
            require(prefix + "ssm_beta.weight",
                    {manifest_.embedding_length, manifest_.ssm_time_step_rank},
                    TensorRole::matrix),
            require(prefix + "ssm_alpha.weight",
                    {manifest_.embedding_length, manifest_.ssm_time_step_rank},
                    TensorRole::matrix),
            require(prefix + "ssm_norm.weight",
                    {manifest_.ssm_state_size},
                    TensorRole::vector),
            require(prefix + "ssm_out.weight",
                    {value_width, manifest_.embedding_length},
                    TensorRole::matrix),
        };
    }

    [[nodiscard]] Qwen35BackboneBlockWeights bind_backbone_block(std::uint32_t index) {
        const std::string prefix = "blk." + std::to_string(index) + ".";
        Qwen35BackboneBlockWeights block;
        block.index = index;
        block.kind = manifest_.is_full_attention_block(index)
                         ? Qwen35BlockKind::full_attention
                         : Qwen35BlockKind::recurrent;
        block.input_norm = require(prefix + "attn_norm.weight",
                                   {manifest_.embedding_length},
                                   TensorRole::vector);
        block.post_attention_norm = require(prefix + "post_attention_norm.weight",
                                            {manifest_.embedding_length},
                                            TensorRole::vector);
        if (block.kind == Qwen35BlockKind::full_attention) {
            block.attention = bind_attention(index);
        } else {
            block.recurrent = bind_recurrent(index);
        }
        block.mlp = bind_mlp(index);
        return block;
    }

    [[nodiscard]] Qwen35MtpWeights bind_mtp(std::uint32_t index) {
        const std::string prefix = "blk." + std::to_string(index) + ".";
        const std::string nextn = prefix + "nextn.";
        Qwen35MtpWeights mtp;
        mtp.block_index = index;
        mtp.input_norm = require(prefix + "attn_norm.weight",
                                 {manifest_.embedding_length},
                                 TensorRole::vector);
        mtp.post_attention_norm = require(prefix + "post_attention_norm.weight",
                                          {manifest_.embedding_length},
                                          TensorRole::vector);
        mtp.attention = bind_attention(index);
        mtp.mlp = bind_mlp(index);
        mtp.embedding_hidden_projection = require(
            nextn + "eh_proj.weight",
            {checked_multiply(manifest_.embedding_length,
                              2,
                              "Qwen3.5 NextN projection input width"),
             manifest_.embedding_length},
            TensorRole::matrix);
        mtp.embedding_norm = require(nextn + "enorm.weight",
                                     {manifest_.embedding_length},
                                     TensorRole::vector);
        mtp.hidden_norm = require(nextn + "hnorm.weight",
                                  {manifest_.embedding_length},
                                  TensorRole::vector);
        mtp.embed_tokens = optional(nextn + "embed_tokens.weight",
                                    {manifest_.embedding_length,
                                     manifest_.vocabulary_size},
                                    TensorRole::matrix);
        mtp.shared_head = optional(nextn + "shared_head_head.weight",
                                   {manifest_.embedding_length,
                                    manifest_.vocabulary_size},
                                   TensorRole::matrix);
        mtp.shared_head_norm = optional(nextn + "shared_head_norm.weight",
                                        {manifest_.embedding_length},
                                        TensorRole::vector);
        return mtp;
    }

    [[nodiscard]] Qwen35BindingReport make_report(const Qwen35Weights& weights) const {
        Qwen35BindingReport report;
        report.backbone_block_count = manifest_.backbone_block_count;
        report.nextn_predict_layers = manifest_.nextn_predict_layers;
        report.required_tensor_count = required_tensor_count_;
        report.optional_tensor_count = optional_tensor_count_;
        report.bound_tensor_count = consumed_.size();
        report.bound_bytes = bound_bytes_;
        report.tied_output = weights.output_is_tied;
        report.blocks.reserve(weights.blocks.size());

        for (const Qwen35BackboneBlockWeights& block : weights.blocks) {
            if (block.kind == Qwen35BlockKind::full_attention) {
                ++report.attention_block_count;
            } else {
                ++report.recurrent_block_count;
            }
            const std::string prefix = "blk." + std::to_string(block.index) + ".";
            Qwen35BlockBindingSummary summary;
            summary.index = block.index;
            summary.kind = block.kind;
            for (const GgufTensorView& tensor : tensors_) {
                if (tensor.name().starts_with(prefix) &&
                    consumed_.contains(std::string(tensor.name()))) {
                    ++summary.tensor_count;
                    summary.mapped_bytes = checked_add(summary.mapped_bytes,
                                                       tensor.bytes().size(),
                                                       "Qwen3.5 block mapped bytes");
                }
            }
            report.blocks.push_back(summary);
        }

        for (const GgufTensorView& tensor : tensors_) {
            if (!consumed_.contains(std::string(tensor.name()))) {
                report.unexpected_tensors.emplace_back(tensor.name());
            }
        }
        std::ranges::sort(report.unexpected_tensors);
        report.unexpected_tensor_count = report.unexpected_tensors.size();
        return report;
    }

    std::span<const GgufTensorView> tensors_;
    const Qwen35Manifest& manifest_;
    std::unordered_map<std::string, const GgufTensorView*> index_;
    std::unordered_set<std::string> consumed_;
    std::size_t required_tensor_count_{0};
    std::size_t optional_tensor_count_{0};
    std::uint64_t bound_bytes_{0};
};

}  // namespace

std::string_view qwen35_block_kind_name(Qwen35BlockKind kind) noexcept {
    switch (kind) {
        case Qwen35BlockKind::recurrent: return "recurrent";
        case Qwen35BlockKind::full_attention: return "attention";
    }
    return "unknown";
}

Qwen35Weights bind_qwen35_weights(std::span<const GgufTensorView> tensors,
                                  const Qwen35Manifest& manifest) {
    return Binder(tensors, manifest).bind();
}

Qwen35Weights bind_qwen35_weights(const MappedGgufModel& model,
                                  const Qwen35Manifest& manifest) {
    return bind_qwen35_weights(model.tensors(), manifest);
}

std::string qwen35_binding_report_text(const Qwen35Weights& weights) {
    const Qwen35BindingReport& report = weights.report;
    std::ostringstream output;
    output << "architecture: qwen35\n"
           << "backbone blocks: " << report.backbone_block_count << '\n'
           << "recurrent blocks: " << report.recurrent_block_count << '\n'
           << "attention blocks: " << report.attention_block_count << '\n'
           << "NextN layers: " << report.nextn_predict_layers << '\n'
           << "required tensors bound: " << report.required_tensor_count << '\n'
           << "optional tensors bound: " << report.optional_tensor_count << '\n'
           << "total tensors bound: " << report.bound_tensor_count << '\n'
           << "unexpected tensors: " << report.unexpected_tensor_count << '\n'
           << "mapped weight bytes: " << report.bound_bytes << '\n'
           << "output head: " << (report.tied_output ? "tied token embedding" : "explicit")
           << '\n';
    for (const Qwen35BlockBindingSummary& block : report.blocks) {
        output << "blk." << block.index << "  " << qwen35_block_kind_name(block.kind)
               << "  tensors=" << block.tensor_count
               << "  bytes=" << block.mapped_bytes << '\n';
    }
    if (weights.mtp) {
        output << "blk." << weights.mtp->block_index << "  mtp  bound\n";
    }
    if (!report.unexpected_tensors.empty()) {
        output << "unexpected:\n";
        for (const std::string& tensor : report.unexpected_tensors) {
            output << "  " << tensor << '\n';
        }
    }
    return output.str();
}

std::string qwen35_binding_report_json(const Qwen35Weights& weights) {
    const Qwen35BindingReport& report = weights.report;
    std::ostringstream output;
    output << "{\"architecture\":\"qwen35\",\"backbone_block_count\":"
           << report.backbone_block_count << ",\"recurrent_block_count\":"
           << report.recurrent_block_count << ",\"attention_block_count\":"
           << report.attention_block_count << ",\"nextn_predict_layers\":"
           << report.nextn_predict_layers << ",\"required_tensor_count\":"
           << report.required_tensor_count << ",\"optional_tensor_count\":"
           << report.optional_tensor_count << ",\"bound_tensor_count\":"
           << report.bound_tensor_count << ",\"unexpected_tensor_count\":"
           << report.unexpected_tensor_count << ",\"bound_bytes\":"
           << report.bound_bytes << ",\"tied_output\":"
           << (report.tied_output ? "true" : "false") << ",\"blocks\":[";
    for (std::size_t index = 0; index < report.blocks.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        const Qwen35BlockBindingSummary& block = report.blocks[index];
        output << "{\"index\":" << block.index << ",\"kind\":\""
               << qwen35_block_kind_name(block.kind) << "\",\"tensor_count\":"
               << block.tensor_count << ",\"mapped_bytes\":" << block.mapped_bytes << '}';
    }
    output << "],\"mtp_block_index\":";
    if (weights.mtp) {
        output << weights.mtp->block_index;
    } else {
        output << "null";
    }
    output << ",\"unexpected_tensors\":[";
    for (std::size_t index = 0; index < report.unexpected_tensors.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << '"' << json_escape(report.unexpected_tensors[index]) << '"';
    }
    output << "]}";
    return output.str();
}

}  // namespace oracle::model
