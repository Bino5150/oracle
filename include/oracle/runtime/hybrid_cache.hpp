#pragma once

#include "oracle/model/qwen35_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace oracle::runtime {

struct Qwen35SsmLayout {
    std::size_t key_heads{0};
    std::size_t value_heads{0};
    std::size_t key_head_dimension{0};
    std::size_t value_head_dimension{0};
    std::size_t convolution_channels{0};
    std::size_t convolution_kernel{0};
};

[[nodiscard]] Qwen35SsmLayout qwen35_ssm_layout(const model::Qwen35Manifest& manifest);

struct HybridCachePlan {
    std::size_t maximum_tokens{0};
    std::size_t attention_layers{0};
    std::size_t ssm_layers{0};
    std::size_t kv_bytes_per_attention_layer{0};
    std::size_t convolution_bytes_per_ssm_layer{0};
    std::size_t recurrent_bytes_per_ssm_layer{0};
    std::size_t total_bytes{0};
};

[[nodiscard]] HybridCachePlan plan_qwen35_cache(const model::Qwen35Manifest& manifest,
                                                 std::size_t maximum_tokens);

class KvCache {
public:
    KvCache(std::size_t maximum_tokens,
            std::size_t key_value_heads,
            std::size_t key_dimension,
            std::size_t value_dimension);

    void append(std::span<const float> key, std::span<const float> value);
    void reset() noexcept;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t key_value_heads() const noexcept;
    [[nodiscard]] std::size_t key_dimension() const noexcept;
    [[nodiscard]] std::size_t value_dimension() const noexcept;
    [[nodiscard]] std::size_t byte_size() const noexcept;

    [[nodiscard]] std::span<const float> key(std::size_t token,
                                             std::size_t head) const;
    [[nodiscard]] std::span<const float> value(std::size_t token,
                                               std::size_t head) const;

private:
    std::size_t maximum_tokens_{0};
    std::size_t key_value_heads_{0};
    std::size_t key_dimension_{0};
    std::size_t value_dimension_{0};
    std::size_t length_{0};
    std::vector<float> keys_;
    std::vector<float> values_;
};

class SsmState {
public:
    SsmState(std::size_t convolution_channels,
             std::size_t convolution_kernel,
             std::size_t value_heads,
             std::size_t key_dimension,
             std::size_t value_dimension);

    void reset() noexcept;
    void advance() noexcept;

    [[nodiscard]] std::size_t sequence_length() const noexcept;
    [[nodiscard]] std::size_t convolution_channels() const noexcept;
    [[nodiscard]] std::size_t convolution_kernel() const noexcept;
    [[nodiscard]] std::size_t value_heads() const noexcept;
    [[nodiscard]] std::size_t key_dimension() const noexcept;
    [[nodiscard]] std::size_t value_dimension() const noexcept;
    [[nodiscard]] std::size_t byte_size() const noexcept;

    [[nodiscard]] std::span<float> convolution_history() noexcept;
    [[nodiscard]] std::span<const float> convolution_history() const noexcept;
    [[nodiscard]] std::span<float> recurrent() noexcept;
    [[nodiscard]] std::span<const float> recurrent() const noexcept;

private:
    std::size_t convolution_channels_{0};
    std::size_t convolution_kernel_{0};
    std::size_t value_heads_{0};
    std::size_t key_dimension_{0};
    std::size_t value_dimension_{0};
    std::size_t sequence_length_{0};
    std::vector<float> convolution_history_;
    std::vector<float> recurrent_;
};

using HybridLayerState = std::variant<SsmState, KvCache>;

class HybridCache {
public:
    HybridCache(const model::Qwen35Manifest& manifest, std::size_t maximum_tokens);

    [[nodiscard]] std::size_t block_count() const noexcept;
    [[nodiscard]] std::size_t sequence_length() const noexcept;
    [[nodiscard]] std::size_t byte_size() const noexcept;
    [[nodiscard]] const HybridCachePlan& plan() const noexcept;

    [[nodiscard]] bool is_attention_block(std::size_t block_index) const;
    [[nodiscard]] KvCache& attention(std::size_t block_index);
    [[nodiscard]] const KvCache& attention(std::size_t block_index) const;
    [[nodiscard]] SsmState& ssm(std::size_t block_index);
    [[nodiscard]] const SsmState& ssm(std::size_t block_index) const;

    void reset() noexcept;

private:
    HybridCachePlan plan_;
    std::vector<HybridLayerState> layers_;
};

}  // namespace oracle::runtime
