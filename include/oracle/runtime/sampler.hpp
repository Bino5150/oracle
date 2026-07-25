#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace oracle::runtime {

struct SamplingConfig {
    float temperature{0.0F};
    std::size_t top_k{0};
    float top_p{1.0F};
    std::uint64_t seed{5150};
};

struct SampleResult {
    std::uint32_t token_id{0};
    float probability{0.0F};
    std::size_t candidate_count{0};
};

class Sampler {
public:
    explicit Sampler(SamplingConfig config = {});

    [[nodiscard]] const SamplingConfig& config() const noexcept;
    [[nodiscard]] SampleResult sample(std::span<const float> logits);
    void reseed(std::uint64_t seed);

private:
    struct Candidate {
        std::uint32_t token_id{0};
        double probability{0.0};
    };

    SamplingConfig config_;
    std::mt19937_64 random_;
    std::vector<Candidate> candidates_;
};

}  // namespace oracle::runtime
