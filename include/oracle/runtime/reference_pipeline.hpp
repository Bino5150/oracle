#pragma once

#include "oracle/backend/backend.hpp"
#include "oracle/core/tensor.hpp"

namespace oracle::runtime {

struct ReferencePipelineWeights {
    core::Tensor norm;
    core::Tensor up_projection;
    core::Tensor down_projection;
};

[[nodiscard]] core::Tensor run_reference_pipeline(
    backend::IBackend& backend,
    const core::Tensor& input,
    const ReferencePipelineWeights& weights,
    float rms_epsilon = 1.0e-5F);

}  // namespace oracle::runtime
