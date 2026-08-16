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

    // Truncates to new_length (<= size()). Exact and lossless: append() only ever
    // writes at the current length and advances forward, so every row below
    // new_length has never been overwritten and remains bit-identical to what it
    // held before growing past new_length in the first place. Throws if
    // new_length > size() (state must never grow via truncation).
    void truncate(std::size_t new_length);

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

    // Restores sequence_length() to a previously observed value without touching
    // recurrent()/convolution_history() -- the caller is responsible for having
    // already restored those buffers' contents first (see HybridCache::truncate_to,
    // the only intended caller). Not a general-purpose setter: recurrent state has
    // no per-token history, so this by itself does not "undo" any recurrence step.
    void restore_sequence_length(std::size_t value) noexcept;

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

// A single, lightweight recovery point for one HybridCache, sufficient to restore
// it back to exactly the sequence_length() it had when mark_boundary() was called.
// Attention (KvCache) layers need no stored data at all: their rows are append-only
// and never overwritten in place, so plain length truncation is already exact.
// Recurrent (SsmState) layers are captured in full (their small, fixed-size
// recurrent/convolution buffers) because Gated DeltaNet state has no per-token
// history to truncate back to -- it is mutated strictly in place (see
// reference_kernels.cpp's gated_delta_step/causal_depthwise_convolution_step) --
// exactly the same constraint the pinned beellama reference's own recurrent memory
// module documents ("models like Mamba or RWKV can't have a state partially erased
// ... their state isn't preserved for previous tokens", llama-memory-recurrent.cpp)
// and works around via bounded per-token snapshots. This type intentionally holds
// exactly one boundary -- not a stack, not an arbitrary-depth history -- because
// Oracle's speculative verification only ever needs to recover to the single
// canonical position that existed before a speculative batch was evaluated.
class HybridCacheBoundary {
public:
    [[nodiscard]] std::size_t sequence_length() const noexcept { return sequence_length_; }

private:
    friend class HybridCache;

    std::size_t sequence_length_{0};
    std::vector<std::vector<float>> recurrent_snapshots_;    // one per SSM block, block order
    std::vector<std::vector<float>> convolution_snapshots_;  // one per SSM block, block order
};

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

    // Captures a recovery point at the current sequence_length(), for later exact,
    // lossless recovery via truncate_to(). See HybridCacheBoundary.
    [[nodiscard]] HybridCacheBoundary mark_boundary() const;

    // Truncates every layer back to boundary.sequence_length(), exactly and
    // losslessly: attention layers via direct length truncation; recurrent layers
    // by restoring the exact buffers captured in the boundary. Throws if
    // boundary.sequence_length() > sequence_length() (state must never grow via
    // truncation) or if the boundary's shape does not match this cache's own
    // block/SSM layout (e.g. a boundary taken from a differently-shaped cache).
    // A no-op if boundary.sequence_length() == sequence_length() already.
    void truncate_to(const HybridCacheBoundary& boundary);

private:
    HybridCachePlan plan_;
    std::vector<HybridLayerState> layers_;
};

}  // namespace oracle::runtime
