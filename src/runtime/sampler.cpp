#include "oracle/runtime/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace oracle::runtime {

Sampler::Sampler(SamplingConfig config) : config_(config), random_(config.seed) {
    if (config_.temperature < 0.0F || !std::isfinite(config_.temperature)) {
        throw std::invalid_argument("sampling temperature must be finite and non-negative");
    }
    if (!(config_.top_p > 0.0F && config_.top_p <= 1.0F) ||
        !std::isfinite(config_.top_p)) {
        throw std::invalid_argument("sampling top_p must be in (0, 1]");
    }
}

const SamplingConfig& Sampler::config() const noexcept { return config_; }

SampleResult Sampler::sample(std::span<const float> logits) {
    if (logits.empty()) {
        throw std::invalid_argument("cannot sample an empty logits vector");
    }
    if (logits.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("logits vector exceeds token-id range");
    }

    std::uint32_t greedy_token = 0;
    float greedy_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (std::isnan(logits[index])) {
            throw std::invalid_argument("logits contain NaN");
        }
        if (logits[index] > greedy_logit) {
            greedy_logit = logits[index];
            greedy_token = static_cast<std::uint32_t>(index);
        }
    }
    if (!std::isfinite(greedy_logit)) {
        throw std::invalid_argument("logits contain no finite candidates");
    }
    if (config_.temperature == 0.0F) {
        return {greedy_token, 1.0F, 1};
    }

    candidates_.clear();
    candidates_.reserve(logits.size());
    const double inverse_temperature = 1.0 / static_cast<double>(config_.temperature);
    double denominator = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (!std::isfinite(logits[index])) {
            continue;
        }
        const double probability = std::exp(
            (static_cast<double>(logits[index]) - static_cast<double>(greedy_logit)) *
            inverse_temperature);
        candidates_.push_back({static_cast<std::uint32_t>(index), probability});
        denominator += probability;
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
        throw std::runtime_error("sampling softmax normalization failed");
    }
    for (Candidate& candidate : candidates_) {
        candidate.probability /= denominator;
    }
    std::ranges::sort(candidates_, [](const Candidate& left, const Candidate& right) {
        if (left.probability != right.probability) {
            return left.probability > right.probability;
        }
        return left.token_id < right.token_id;
    });

    if (config_.top_k != 0 && candidates_.size() > config_.top_k) {
        candidates_.resize(config_.top_k);
    }
    if (config_.top_p < 1.0F) {
        double cumulative = 0.0;
        std::size_t keep = 0;
        for (; keep < candidates_.size(); ++keep) {
            cumulative += candidates_[keep].probability;
            if (cumulative >= static_cast<double>(config_.top_p)) {
                ++keep;
                break;
            }
        }
        candidates_.resize(std::max<std::size_t>(1, keep));
    }

    double retained = 0.0;
    for (const Candidate& candidate : candidates_) {
        retained += candidate.probability;
    }
    for (Candidate& candidate : candidates_) {
        candidate.probability /= retained;
    }

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const double draw = distribution(random_);
    double cumulative = 0.0;
    for (const Candidate& candidate : candidates_) {
        cumulative += candidate.probability;
        if (draw <= cumulative) {
            return {candidate.token_id,
                    static_cast<float>(candidate.probability),
                    candidates_.size()};
        }
    }
    const Candidate& fallback = candidates_.back();
    return {fallback.token_id,
            static_cast<float>(fallback.probability),
            candidates_.size()};
}

void Sampler::reseed(std::uint64_t seed) {
    config_.seed = seed;
    random_.seed(seed);
}

}  // namespace oracle::runtime
