#include "oracle/runtime/qwen35_state_fingerprint.hpp"

#include <iomanip>
#include <sstream>

namespace oracle::runtime {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

void hash_floats(std::uint64_t& hash, std::span<const float> values) noexcept {
    hash_bytes(hash, values.data(), values.size() * sizeof(float));
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    hash_bytes(hash, &value, sizeof(value));
}

[[nodiscard]] std::uint64_t fingerprint_attention_block(const KvCache& cache) {
    std::uint64_t hash = kFnvOffset;
    hash_u64(hash, cache.size());
    hash_u64(hash, cache.key_value_heads());
    hash_u64(hash, cache.key_dimension());
    hash_u64(hash, cache.value_dimension());
    for (std::size_t token = 0; token < cache.size(); ++token) {
        for (std::size_t head = 0; head < cache.key_value_heads(); ++head) {
            hash_floats(hash, cache.key(token, head));
            hash_floats(hash, cache.value(token, head));
        }
    }
    return hash;
}

[[nodiscard]] std::uint64_t fingerprint_ssm_block(const SsmState& state) {
    std::uint64_t hash = kFnvOffset;
    hash_u64(hash, state.sequence_length());
    hash_u64(hash, state.convolution_channels());
    hash_u64(hash, state.convolution_kernel());
    hash_u64(hash, state.value_heads());
    hash_floats(hash, state.convolution_history());
    hash_floats(hash, state.recurrent());
    return hash;
}

}  // namespace

Qwen35StateFingerprint fingerprint_qwen35_state(const HybridCache& cache) {
    Qwen35StateFingerprint result;
    result.sequence_length = cache.sequence_length();
    result.block_count = cache.block_count();
    result.per_block.reserve(result.block_count);

    std::uint64_t combined = kFnvOffset;
    hash_u64(combined, result.sequence_length);
    hash_u64(combined, result.block_count);

    for (std::size_t block = 0; block < result.block_count; ++block) {
        const std::uint64_t block_hash = cache.is_attention_block(block)
                                             ? fingerprint_attention_block(cache.attention(block))
                                             : fingerprint_ssm_block(cache.ssm(block));
        result.per_block.push_back(block_hash);
        hash_u64(combined, block_hash);
    }
    result.combined = combined;
    return result;
}

bool Qwen35StateFingerprint::operator==(const Qwen35StateFingerprint& other) const noexcept {
    return sequence_length == other.sequence_length && block_count == other.block_count &&
           combined == other.combined && per_block == other.per_block;
}

std::string qwen35_state_fingerprint_text(const Qwen35StateFingerprint& fingerprint) {
    std::ostringstream output;
    output << "sequence_length=" << fingerprint.sequence_length
           << " block_count=" << fingerprint.block_count << " combined=0x" << std::hex
           << fingerprint.combined << std::dec;
    return output.str();
}

}  // namespace oracle::runtime
