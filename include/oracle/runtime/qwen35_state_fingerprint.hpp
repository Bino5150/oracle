#pragma once

#include "oracle/runtime/hybrid_cache.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace oracle::runtime {

// Diagnostic/test-only deterministic fingerprint of a HybridCache's complete
// canonical state: every attention block's K/V entries and every recurrent block's
// convolution history and recurrent state, in block order. Not a serialization
// format and not an ABI -- it exists solely to let Slice 3's Lane A/B tests prove
// two independently-constructed HybridCache instances hold bit-identical state
// without dumping their full contents into ordinary logs (full dumps remain
// evidence-directory artifacts, generated separately, not part of this facility).
struct Qwen35StateFingerprint {
    std::uint64_t sequence_length{0};
    std::uint64_t block_count{0};
    std::vector<std::uint64_t> per_block;  // one FNV1a64 hash per backbone block, in block order
    std::uint64_t combined{0};             // FNV1a64 over sequence_length, block_count, and per_block

    [[nodiscard]] bool operator==(const Qwen35StateFingerprint& other) const noexcept;
};

[[nodiscard]] Qwen35StateFingerprint fingerprint_qwen35_state(const HybridCache& cache);
[[nodiscard]] std::string qwen35_state_fingerprint_text(const Qwen35StateFingerprint& fingerprint);

}  // namespace oracle::runtime
