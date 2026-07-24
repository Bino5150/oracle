#pragma once

#include "oracle/core/tensor.hpp"

#include <memory>
#include <string_view>

namespace oracle::backend {

class IBackend {
public:
    virtual ~IBackend() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    virtual void add(const core::Tensor& lhs, const core::Tensor& rhs, core::Tensor& out) = 0;
    virtual void multiply(const core::Tensor& lhs,
                          const core::Tensor& rhs,
                          core::Tensor& out) = 0;
    virtual void matmul(const core::Tensor& lhs,
                        const core::Tensor& rhs,
                        core::Tensor& out) = 0;
    virtual void rms_norm(const core::Tensor& input,
                          const core::Tensor& weight,
                          float epsilon,
                          core::Tensor& out) = 0;
    virtual void silu(const core::Tensor& input, core::Tensor& out) = 0;
    virtual void softmax(const core::Tensor& input, core::Tensor& out) = 0;
};

[[nodiscard]] std::unique_ptr<IBackend> make_cpu_backend();

}  // namespace oracle::backend
