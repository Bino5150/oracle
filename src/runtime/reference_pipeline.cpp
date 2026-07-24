#include "oracle/runtime/reference_pipeline.hpp"

#include <stdexcept>

namespace oracle::runtime {

core::Tensor run_reference_pipeline(backend::IBackend& backend,
                                    const core::Tensor& input,
                                    const ReferencePipelineWeights& weights,
                                    float rms_epsilon) {
    if (input.rank() != 2 || weights.norm.rank() != 1 ||
        weights.up_projection.rank() != 2 || weights.down_projection.rank() != 2) {
        throw std::invalid_argument("reference pipeline expects rank-2 activations and projections");
    }

    const std::size_t batch = input.shape()[0];
    const std::size_t hidden = input.shape()[1];
    const std::size_t feed_forward = weights.up_projection.shape()[1];
    if (weights.norm.shape()[0] != hidden || weights.up_projection.shape()[0] != hidden ||
        weights.down_projection.shape()[0] != feed_forward ||
        weights.down_projection.shape()[1] != hidden) {
        throw std::invalid_argument("reference pipeline weight dimensions are incompatible");
    }

    core::Tensor normalized({batch, hidden});
    core::Tensor projected({batch, feed_forward});
    core::Tensor activated({batch, feed_forward});
    core::Tensor restored({batch, hidden});
    core::Tensor residual({batch, hidden});
    core::Tensor probabilities({batch, hidden});

    backend.rms_norm(input, weights.norm, rms_epsilon, normalized);
    backend.matmul(normalized, weights.up_projection, projected);
    backend.silu(projected, activated);
    backend.matmul(activated, weights.down_projection, restored);
    backend.add(input, restored, residual);
    backend.softmax(residual, probabilities);
    return probabilities;
}

}  // namespace oracle::runtime
