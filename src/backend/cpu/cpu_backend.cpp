#include "oracle/backend/backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace oracle::backend {
namespace {

void require_f32_contiguous(const core::Tensor& tensor, std::string_view operation) {
    if (tensor.dtype() != core::DataType::f32) {
        throw std::invalid_argument(std::string(operation) + " requires f32 tensors");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(operation) + " requires contiguous tensors");
    }
}

void require_matching_shapes(const core::Tensor& lhs,
                             const core::Tensor& rhs,
                             const core::Tensor& out,
                             std::string_view operation) {
    require_f32_contiguous(lhs, operation);
    require_f32_contiguous(rhs, operation);
    require_f32_contiguous(out, operation);
    if (lhs.shape() != rhs.shape() || lhs.shape() != out.shape()) {
        throw std::invalid_argument(std::string(operation) + " tensor shapes must match");
    }
}

class CpuBackend final : public IBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "cpu"; }

    void add(const core::Tensor& lhs, const core::Tensor& rhs, core::Tensor& out) override {
        require_matching_shapes(lhs, rhs, out, "add");
        const auto a = lhs.data();
        const auto b = rhs.data();
        auto c = out.data();
        for (std::size_t index = 0; index < c.size(); ++index) {
            c[index] = a[index] + b[index];
        }
    }

    void multiply(const core::Tensor& lhs,
                  const core::Tensor& rhs,
                  core::Tensor& out) override {
        require_matching_shapes(lhs, rhs, out, "multiply");
        const auto a = lhs.data();
        const auto b = rhs.data();
        auto c = out.data();
        for (std::size_t index = 0; index < c.size(); ++index) {
            c[index] = a[index] * b[index];
        }
    }

    void matmul(const core::Tensor& lhs,
                const core::Tensor& rhs,
                core::Tensor& out) override {
        require_f32_contiguous(lhs, "matmul");
        require_f32_contiguous(rhs, "matmul");
        require_f32_contiguous(out, "matmul");
        if (lhs.rank() != 2 || rhs.rank() != 2 || out.rank() != 2) {
            throw std::invalid_argument("matmul currently requires rank-2 tensors");
        }

        const std::size_t rows = lhs.shape()[0];
        const std::size_t inner = lhs.shape()[1];
        const std::size_t columns = rhs.shape()[1];
        if (rhs.shape()[0] != inner || out.shape()[0] != rows || out.shape()[1] != columns) {
            throw std::invalid_argument("matmul tensor dimensions are incompatible");
        }

        const auto a = lhs.data();
        const auto b = rhs.data();
        auto c = out.data();
        std::ranges::fill(c, 0.0F);

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t pivot = 0; pivot < inner; ++pivot) {
                const float lhs_value = a[row * inner + pivot];
                const std::size_t rhs_base = pivot * columns;
                const std::size_t out_base = row * columns;
                for (std::size_t column = 0; column < columns; ++column) {
                    c[out_base + column] += lhs_value * b[rhs_base + column];
                }
            }
        }
    }

    void rms_norm(const core::Tensor& input,
                  const core::Tensor& weight,
                  float epsilon,
                  core::Tensor& out) override {
        require_f32_contiguous(input, "rms_norm");
        require_f32_contiguous(weight, "rms_norm");
        require_f32_contiguous(out, "rms_norm");
        if (input.shape() != out.shape() || weight.rank() != 1 ||
            weight.shape()[0] != input.shape().back()) {
            throw std::invalid_argument("rms_norm tensor dimensions are incompatible");
        }
        if (!(epsilon > 0.0F)) {
            throw std::invalid_argument("rms_norm epsilon must be positive");
        }

        const std::size_t width = input.shape().back();
        const std::size_t rows = input.element_count() / width;
        const auto x = input.data();
        const auto scale = weight.data();
        auto y = out.data();

        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t base = row * width;
            double sum_squares = 0.0;
            for (std::size_t column = 0; column < width; ++column) {
                const double value = static_cast<double>(x[base + column]);
                sum_squares += value * value;
            }
            const double mean_square = sum_squares / static_cast<double>(width);
            const float inverse_rms =
                static_cast<float>(1.0 / std::sqrt(mean_square + static_cast<double>(epsilon)));
            for (std::size_t column = 0; column < width; ++column) {
                y[base + column] = x[base + column] * inverse_rms * scale[column];
            }
        }
    }

    void silu(const core::Tensor& input, core::Tensor& out) override {
        require_f32_contiguous(input, "silu");
        require_f32_contiguous(out, "silu");
        if (input.shape() != out.shape()) {
            throw std::invalid_argument("silu tensor shapes must match");
        }

        const auto x = input.data();
        auto y = out.data();
        for (std::size_t index = 0; index < y.size(); ++index) {
            const float value = x[index];
            y[index] = value / (1.0F + std::exp(-value));
        }
    }

    void softmax(const core::Tensor& input, core::Tensor& out) override {
        require_f32_contiguous(input, "softmax");
        require_f32_contiguous(out, "softmax");
        if (input.shape() != out.shape()) {
            throw std::invalid_argument("softmax tensor shapes must match");
        }

        const std::size_t width = input.shape().back();
        const std::size_t rows = input.element_count() / width;
        const auto x = input.data();
        auto y = out.data();

        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t base = row * width;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t column = 0; column < width; ++column) {
                maximum = std::max(maximum, x[base + column]);
            }

            double sum = 0.0;
            for (std::size_t column = 0; column < width; ++column) {
                const float exponent = std::exp(x[base + column] - maximum);
                y[base + column] = exponent;
                sum += static_cast<double>(exponent);
            }

            const float inverse_sum = static_cast<float>(1.0 / sum);
            for (std::size_t column = 0; column < width; ++column) {
                y[base + column] *= inverse_sum;
            }
        }
    }
};

}  // namespace

std::unique_ptr<IBackend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

}  // namespace oracle::backend
