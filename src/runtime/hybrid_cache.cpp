#include "oracle/runtime/hybrid_cache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace oracle::runtime {
namespace {

[[nodiscard]] std::size_t checked_multiply(std::size_t left,
                                           std::size_t right,
                                           const char* context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left,
                                      std::size_t right,
                                      const char* context) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left + right;
}

void require_positive(std::size_t value, const char* name) {
    if (value == 0) {
        throw std::invalid_argument(std::string(name) + " must be non-zero");
    }
}

}  // namespace

Qwen35SsmLayout qwen35_ssm_layout(const model::Qwen35Manifest& manifest) {
    Qwen35SsmLayout layout;
    layout.key_heads = manifest.ssm_group_count;
    layout.value_heads = manifest.ssm_time_step_rank;
    layout.key_head_dimension = manifest.ssm_state_size;
    layout.convolution_kernel = manifest.ssm_convolution_kernel;

    require_positive(layout.key_heads, "SSM key-head count");
    require_positive(layout.value_heads, "SSM value-head count");
    require_positive(layout.key_head_dimension, "SSM key-head dimension");
    require_positive(layout.convolution_kernel, "SSM convolution kernel");
    if (layout.value_heads % layout.key_heads != 0) {
        throw std::runtime_error("Qwen3.5 SSM value-head count must be divisible by key-head count");
    }
    if (manifest.ssm_inner_size % layout.value_heads != 0) {
        throw std::runtime_error("Qwen3.5 SSM inner size must be divisible by value-head count");
    }
    layout.value_head_dimension = manifest.ssm_inner_size / layout.value_heads;

    const std::size_t key_width = checked_multiply(
        layout.key_heads, layout.key_head_dimension, "Qwen3.5 SSM key width");
    const std::size_t value_width = checked_multiply(
        layout.value_heads, layout.value_head_dimension, "Qwen3.5 SSM value width");
    layout.convolution_channels = checked_add(
        checked_multiply(key_width, 2, "Qwen3.5 SSM QK width"),
        value_width,
        "Qwen3.5 SSM convolution width");
    return layout;
}

HybridCachePlan plan_qwen35_cache(const model::Qwen35Manifest& manifest,
                                  std::size_t maximum_tokens) {
    require_positive(maximum_tokens, "maximum token count");
    if (maximum_tokens > manifest.context_length) {
        throw std::invalid_argument("maximum token count exceeds model context length");
    }

    require_positive(manifest.backbone_block_count, "Qwen3.5 backbone block count");
    require_positive(manifest.attention_head_count_kv, "attention KV head count");
    require_positive(manifest.attention_key_length, "attention key dimension");
    require_positive(manifest.attention_value_length, "attention value dimension");

    HybridCachePlan plan;
    plan.maximum_tokens = maximum_tokens;
    const Qwen35SsmLayout ssm_layout = qwen35_ssm_layout(manifest);

    for (std::uint32_t block = 0; block < manifest.backbone_block_count; ++block) {
        if (manifest.is_full_attention_block(block)) {
            ++plan.attention_layers;
        } else {
            ++plan.ssm_layers;
        }
    }

    const std::size_t kv_elements_per_token = checked_multiply(
        manifest.attention_head_count_kv,
        checked_add(manifest.attention_key_length,
                    manifest.attention_value_length,
                    "KV dimensions"),
        "KV elements per token");
    plan.kv_bytes_per_attention_layer = checked_multiply(
        checked_multiply(maximum_tokens, kv_elements_per_token, "KV elements"),
        sizeof(float),
        "KV bytes");

    plan.convolution_bytes_per_ssm_layer = checked_multiply(
        checked_multiply(ssm_layout.convolution_channels,
                         ssm_layout.convolution_kernel,
                         "SSM convolution state elements"),
        sizeof(float),
        "SSM convolution state bytes");
    plan.recurrent_bytes_per_ssm_layer = checked_multiply(
        checked_multiply(
            checked_multiply(ssm_layout.value_heads,
                             ssm_layout.key_head_dimension,
                             "SSM recurrent key elements"),
            ssm_layout.value_head_dimension,
            "SSM recurrent elements"),
        sizeof(float),
        "SSM recurrent bytes");

    const std::size_t all_kv = checked_multiply(
        plan.attention_layers, plan.kv_bytes_per_attention_layer, "all KV cache bytes");
    const std::size_t ssm_per_layer = checked_add(plan.convolution_bytes_per_ssm_layer,
                                                  plan.recurrent_bytes_per_ssm_layer,
                                                  "SSM bytes per layer");
    const std::size_t all_ssm = checked_multiply(plan.ssm_layers,
                                                 ssm_per_layer,
                                                 "all SSM cache bytes");
    plan.total_bytes = checked_add(all_kv, all_ssm, "hybrid cache bytes");
    return plan;
}

KvCache::KvCache(std::size_t maximum_tokens,
                 std::size_t key_value_heads,
                 std::size_t key_dimension,
                 std::size_t value_dimension)
    : maximum_tokens_(maximum_tokens),
      key_value_heads_(key_value_heads),
      key_dimension_(key_dimension),
      value_dimension_(value_dimension) {
    require_positive(maximum_tokens_, "KV cache capacity");
    require_positive(key_value_heads_, "KV head count");
    require_positive(key_dimension_, "KV key dimension");
    require_positive(value_dimension_, "KV value dimension");

    keys_.resize(checked_multiply(
        checked_multiply(maximum_tokens_, key_value_heads_, "KV key rows"),
        key_dimension_,
        "KV key elements"));
    values_.resize(checked_multiply(
        checked_multiply(maximum_tokens_, key_value_heads_, "KV value rows"),
        value_dimension_,
        "KV value elements"));
}

void KvCache::append(std::span<const float> key_values,
                     std::span<const float> value_values) {
    if (length_ >= maximum_tokens_) {
        throw std::overflow_error("KV cache capacity exceeded");
    }
    const std::size_t expected_keys = key_value_heads_ * key_dimension_;
    const std::size_t expected_values = key_value_heads_ * value_dimension_;
    if (key_values.size() != expected_keys || value_values.size() != expected_values) {
        throw std::invalid_argument("KV cache append dimensions do not match cache layout");
    }
    const std::size_t key_offset = length_ * expected_keys;
    const std::size_t value_offset = length_ * expected_values;
    std::copy(key_values.begin(), key_values.end(), keys_.begin() + static_cast<std::ptrdiff_t>(key_offset));
    std::copy(value_values.begin(), value_values.end(), values_.begin() + static_cast<std::ptrdiff_t>(value_offset));
    ++length_;
}

void KvCache::reset() noexcept {
    length_ = 0;
    std::ranges::fill(keys_, 0.0F);
    std::ranges::fill(values_, 0.0F);
}

std::size_t KvCache::size() const noexcept { return length_; }
std::size_t KvCache::capacity() const noexcept { return maximum_tokens_; }
std::size_t KvCache::key_value_heads() const noexcept { return key_value_heads_; }
std::size_t KvCache::key_dimension() const noexcept { return key_dimension_; }
std::size_t KvCache::value_dimension() const noexcept { return value_dimension_; }
std::size_t KvCache::byte_size() const noexcept {
    return (keys_.size() + values_.size()) * sizeof(float);
}

std::span<const float> KvCache::key(std::size_t token, std::size_t head) const {
    if (token >= length_ || head >= key_value_heads_) {
        throw std::out_of_range("KV key lookup is out of range");
    }
    const std::size_t offset = (token * key_value_heads_ + head) * key_dimension_;
    return {keys_.data() + offset, key_dimension_};
}

std::span<const float> KvCache::value(std::size_t token, std::size_t head) const {
    if (token >= length_ || head >= key_value_heads_) {
        throw std::out_of_range("KV value lookup is out of range");
    }
    const std::size_t offset = (token * key_value_heads_ + head) * value_dimension_;
    return {values_.data() + offset, value_dimension_};
}

SsmState::SsmState(std::size_t convolution_channels,
                   std::size_t convolution_kernel,
                   std::size_t value_heads,
                   std::size_t key_dimension,
                   std::size_t value_dimension)
    : convolution_channels_(convolution_channels),
      convolution_kernel_(convolution_kernel),
      value_heads_(value_heads),
      key_dimension_(key_dimension),
      value_dimension_(value_dimension) {
    require_positive(convolution_channels_, "SSM convolution channels");
    require_positive(convolution_kernel_, "SSM convolution kernel");
    require_positive(value_heads_, "SSM value heads");
    require_positive(key_dimension_, "SSM key dimension");
    require_positive(value_dimension_, "SSM value dimension");

    convolution_history_.resize(checked_multiply(convolution_channels_,
                                                 convolution_kernel_,
                                                 "SSM convolution history"));
    recurrent_.resize(checked_multiply(
        checked_multiply(value_heads_, key_dimension_, "SSM recurrent rows"),
        value_dimension_,
        "SSM recurrent state"));
}

void SsmState::reset() noexcept {
    sequence_length_ = 0;
    std::ranges::fill(convolution_history_, 0.0F);
    std::ranges::fill(recurrent_, 0.0F);
}

void SsmState::advance() noexcept { ++sequence_length_; }
std::size_t SsmState::sequence_length() const noexcept { return sequence_length_; }
std::size_t SsmState::convolution_channels() const noexcept { return convolution_channels_; }
std::size_t SsmState::convolution_kernel() const noexcept { return convolution_kernel_; }
std::size_t SsmState::value_heads() const noexcept { return value_heads_; }
std::size_t SsmState::key_dimension() const noexcept { return key_dimension_; }
std::size_t SsmState::value_dimension() const noexcept { return value_dimension_; }
std::size_t SsmState::byte_size() const noexcept {
    return (convolution_history_.size() + recurrent_.size()) * sizeof(float);
}
std::span<float> SsmState::convolution_history() noexcept { return convolution_history_; }
std::span<const float> SsmState::convolution_history() const noexcept {
    return convolution_history_;
}
std::span<float> SsmState::recurrent() noexcept { return recurrent_; }
std::span<const float> SsmState::recurrent() const noexcept { return recurrent_; }

HybridCache::HybridCache(const model::Qwen35Manifest& manifest,
                         std::size_t maximum_tokens)
    : plan_(plan_qwen35_cache(manifest, maximum_tokens)) {
    layers_.reserve(manifest.backbone_block_count);
    const Qwen35SsmLayout ssm_layout = qwen35_ssm_layout(manifest);
    for (std::uint32_t block = 0; block < manifest.backbone_block_count; ++block) {
        if (manifest.is_full_attention_block(block)) {
            layers_.emplace_back(std::in_place_type<KvCache>,
                                 maximum_tokens,
                                 manifest.attention_head_count_kv,
                                 manifest.attention_key_length,
                                 manifest.attention_value_length);
        } else {
            layers_.emplace_back(std::in_place_type<SsmState>,
                                 ssm_layout.convolution_channels,
                                 ssm_layout.convolution_kernel,
                                 ssm_layout.value_heads,
                                 ssm_layout.key_head_dimension,
                                 ssm_layout.value_head_dimension);
        }
    }
}

std::size_t HybridCache::block_count() const noexcept { return layers_.size(); }

std::size_t HybridCache::sequence_length() const noexcept {
    if (layers_.empty()) {
        return 0;
    }
    return std::visit(
        [](const auto& state) -> std::size_t {
            using State = std::decay_t<decltype(state)>;
            if constexpr (std::is_same_v<State, KvCache>) {
                return state.size();
            } else {
                return state.sequence_length();
            }
        },
        layers_.front());
}

std::size_t HybridCache::byte_size() const noexcept { return plan_.total_bytes; }
const HybridCachePlan& HybridCache::plan() const noexcept { return plan_; }

bool HybridCache::is_attention_block(std::size_t block_index) const {
    if (block_index >= layers_.size()) {
        throw std::out_of_range("hybrid cache block index is out of range");
    }
    return std::holds_alternative<KvCache>(layers_[block_index]);
}

KvCache& HybridCache::attention(std::size_t block_index) {
    if (block_index >= layers_.size()) {
        throw std::out_of_range("hybrid cache block index is out of range");
    }
    auto* cache = std::get_if<KvCache>(&layers_[block_index]);
    if (cache == nullptr) {
        throw std::logic_error("requested attention cache for an SSM block");
    }
    return *cache;
}

const KvCache& HybridCache::attention(std::size_t block_index) const {
    if (block_index >= layers_.size()) {
        throw std::out_of_range("hybrid cache block index is out of range");
    }
    const auto* cache = std::get_if<KvCache>(&layers_[block_index]);
    if (cache == nullptr) {
        throw std::logic_error("requested attention cache for an SSM block");
    }
    return *cache;
}

SsmState& HybridCache::ssm(std::size_t block_index) {
    if (block_index >= layers_.size()) {
        throw std::out_of_range("hybrid cache block index is out of range");
    }
    auto* state = std::get_if<SsmState>(&layers_[block_index]);
    if (state == nullptr) {
        throw std::logic_error("requested SSM state for an attention block");
    }
    return *state;
}

const SsmState& HybridCache::ssm(std::size_t block_index) const {
    if (block_index >= layers_.size()) {
        throw std::out_of_range("hybrid cache block index is out of range");
    }
    const auto* state = std::get_if<SsmState>(&layers_[block_index]);
    if (state == nullptr) {
        throw std::logic_error("requested SSM state for an attention block");
    }
    return *state;
}

void HybridCache::reset() noexcept {
    for (HybridLayerState& layer : layers_) {
        std::visit([](auto& state) { state.reset(); }, layer);
    }
}

}  // namespace oracle::runtime
