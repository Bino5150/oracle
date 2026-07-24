#include "oracle/core/config.hpp"
#include "oracle/core/tensor.hpp"
#include "oracle/runtime/engine.hpp"
#include "oracle/runtime/reference_pipeline.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

class TestRunner {
public:
    void expect(bool condition, std::string_view message) {
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    void expect_near(float actual, float expected, float tolerance, std::string_view message) {
        if (std::fabs(actual - expected) > tolerance) {
            ++failures_;
            std::cerr << "FAIL: " << message << " (actual=" << actual
                      << ", expected=" << expected << ")\n";
        }
    }

    template <typename Function>
    void expect_throws(Function&& function, std::string_view message) {
        try {
            std::invoke(std::forward<Function>(function));
            ++failures_;
            std::cerr << "FAIL: " << message << " (no exception)\n";
        } catch (const std::exception&) {
        }
    }

    [[nodiscard]] int result() const noexcept { return failures_ == 0 ? 0 : 1; }
    [[nodiscard]] int failures() const noexcept { return failures_; }

private:
    int failures_{0};
};

void test_tensor(TestRunner& runner) {
    oracle::core::Tensor tensor({2, 3});
    runner.expect(tensor.rank() == 2, "tensor rank");
    runner.expect(tensor.element_count() == 6, "tensor element count");
    runner.expect(tensor.strides() == std::vector<std::size_t>({3, 1}), "contiguous strides");
    runner.expect(tensor.is_contiguous(), "new tensor is contiguous");
    runner.expect(!tensor.is_view(), "new tensor owns its logical region");

    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data().data());
    runner.expect(address % oracle::core::Tensor::default_alignment == 0,
                  "tensor storage is 64-byte aligned");

    for (std::size_t index = 0; index < tensor.element_count(); ++index) {
        tensor.data()[index] = static_cast<float>(index);
    }

    auto slice = tensor.view({3}, 2);
    runner.expect(slice.is_view(), "slice reports view ownership");
    slice.data()[0] = 42.0F;
    runner.expect_near(tensor.data()[2], 42.0F, 0.0F, "view aliases source storage");

    auto reshaped = tensor.reshape({3, 2});
    reshaped.data()[5] = 77.0F;
    runner.expect_near(tensor.data()[5], 77.0F, 0.0F, "reshape is zero-copy");

    auto transposed = tensor.view({3, 2}, {1, 3});
    const std::array<std::size_t, 2> transposed_index{2, 1};
    runner.expect_near(transposed.at(transposed_index), 77.0F, 0.0F,
                       "strided view resolves indices");
    runner.expect(!transposed.is_contiguous(), "transpose-style view is non-contiguous");
    runner.expect_throws([&transposed] { static_cast<void>(transposed.data()); },
                         "raw span rejects non-contiguous tensor");

    const std::array<std::size_t, 2> invalid_index{2, 0};
    runner.expect_throws([&tensor, &invalid_index] { static_cast<void>(tensor.at(invalid_index)); },
                         "bounds-checked tensor access");
    runner.expect_throws([&tensor] { static_cast<void>(tensor.view({5}, 3)); },
                         "view cannot exceed source bounds");
}

void test_elementwise(oracle::backend::IBackend& backend, TestRunner& runner) {
    oracle::core::Tensor lhs({3});
    oracle::core::Tensor rhs({3});
    oracle::core::Tensor added({3});
    oracle::core::Tensor multiplied({3});
    lhs.data()[0] = 1.0F;
    lhs.data()[1] = 2.0F;
    lhs.data()[2] = 3.0F;
    rhs.data()[0] = 4.0F;
    rhs.data()[1] = 5.0F;
    rhs.data()[2] = 6.0F;

    backend.add(lhs, rhs, added);
    backend.multiply(lhs, rhs, multiplied);

    const float expected_add[] = {5.0F, 7.0F, 9.0F};
    const float expected_multiply[] = {4.0F, 10.0F, 18.0F};
    for (std::size_t index = 0; index < 3; ++index) {
        runner.expect_near(added.data()[index], expected_add[index], 1.0e-6F,
                           "elementwise add");
        runner.expect_near(multiplied.data()[index], expected_multiply[index], 1.0e-6F,
                           "elementwise multiply");
    }
}

void test_matmul(oracle::backend::IBackend& backend, TestRunner& runner) {
    oracle::core::Tensor lhs({2, 3});
    oracle::core::Tensor rhs({3, 2});
    oracle::core::Tensor out({2, 2});
    const float lhs_values[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const float rhs_values[] = {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    for (std::size_t index = 0; index < lhs.element_count(); ++index) {
        lhs.data()[index] = lhs_values[index];
    }
    for (std::size_t index = 0; index < rhs.element_count(); ++index) {
        rhs.data()[index] = rhs_values[index];
    }

    backend.matmul(lhs, rhs, out);
    const float expected[] = {58.0F, 64.0F, 139.0F, 154.0F};
    for (std::size_t index = 0; index < out.element_count(); ++index) {
        runner.expect_near(out.data()[index], expected[index], 1.0e-5F, "reference matmul");
    }
}

void test_normalization_and_activations(oracle::backend::IBackend& backend,
                                        TestRunner& runner) {
    oracle::core::Tensor input({1, 2});
    oracle::core::Tensor weight({2});
    oracle::core::Tensor normalized({1, 2});
    input.data()[0] = 3.0F;
    input.data()[1] = 4.0F;
    weight.fill(1.0F);
    constexpr float epsilon = 1.0e-5F;
    backend.rms_norm(input, weight, epsilon, normalized);
    const float inverse_rms = 1.0F / std::sqrt(12.5F + epsilon);
    runner.expect_near(normalized.data()[0], 3.0F * inverse_rms, 1.0e-6F, "RMSNorm x0");
    runner.expect_near(normalized.data()[1], 4.0F * inverse_rms, 1.0e-6F, "RMSNorm x1");

    oracle::core::Tensor activation_input({3});
    oracle::core::Tensor activation_output({3});
    activation_input.data()[0] = -1.0F;
    activation_input.data()[1] = 0.0F;
    activation_input.data()[2] = 1.0F;
    backend.silu(activation_input, activation_output);
    runner.expect_near(activation_output.data()[0], -1.0F / (1.0F + std::exp(1.0F)),
                       1.0e-6F, "SiLU negative");
    runner.expect_near(activation_output.data()[1], 0.0F, 1.0e-6F, "SiLU zero");
    runner.expect_near(activation_output.data()[2], 1.0F / (1.0F + std::exp(-1.0F)),
                       1.0e-6F, "SiLU positive");

    oracle::core::Tensor logits({1, 3});
    oracle::core::Tensor probabilities({1, 3});
    logits.data()[0] = 1.0F;
    logits.data()[1] = 2.0F;
    logits.data()[2] = 3.0F;
    backend.softmax(logits, probabilities);
    const float denominator = std::exp(-2.0F) + std::exp(-1.0F) + 1.0F;
    runner.expect_near(probabilities.data()[0], std::exp(-2.0F) / denominator, 1.0e-6F,
                       "softmax first");
    runner.expect_near(probabilities.data()[1], std::exp(-1.0F) / denominator, 1.0e-6F,
                       "softmax second");
    runner.expect_near(probabilities.data()[2], 1.0F / denominator, 1.0e-6F,
                       "softmax third");
}

void test_reference_pipeline(oracle::backend::IBackend& backend, TestRunner& runner) {
    oracle::core::Tensor input({2, 4});
    for (std::size_t index = 0; index < input.element_count(); ++index) {
        input.data()[index] = static_cast<float>(index + 1) * 0.1F;
    }

    oracle::runtime::ReferencePipelineWeights weights{
        oracle::core::Tensor({4}),
        oracle::core::Tensor({4, 5}),
        oracle::core::Tensor({5, 4}),
    };
    weights.norm.fill(1.0F);
    for (std::size_t index = 0; index < weights.up_projection.element_count(); ++index) {
        weights.up_projection.data()[index] =
            static_cast<float>(static_cast<int>(index % 7) - 3) * 0.025F;
    }
    for (std::size_t index = 0; index < weights.down_projection.element_count(); ++index) {
        weights.down_projection.data()[index] =
            static_cast<float>(static_cast<int>(index % 5) - 2) * 0.020F;
    }

    const auto output = oracle::runtime::run_reference_pipeline(backend, input, weights);
    runner.expect(output.shape() == std::vector<std::size_t>({2, 4}),
                  "reference pipeline output shape");
    for (std::size_t row = 0; row < 2; ++row) {
        float sum = 0.0F;
        for (std::size_t column = 0; column < 4; ++column) {
            const float value = output.data()[row * 4 + column];
            runner.expect(value > 0.0F && value < 1.0F,
                          "reference pipeline emits probabilities");
            sum += value;
        }
        runner.expect_near(sum, 1.0F, 1.0e-5F, "reference pipeline row sums to one");
    }
}

void test_engine_status(TestRunner& runner) {
    oracle::runtime::Engine engine(oracle::core::EngineConfig{});
    runner.expect(engine.config().server_port == 5150, "Oracle defaults to port 5150");
    runner.expect(engine.status().find("127.0.0.1:5150") != std::string::npos,
                  "text status includes server endpoint");
    runner.expect(engine.status_json().find("\"port\":5150") != std::string::npos,
                  "JSON status includes server port");
    runner.expect(engine.status_json().find("\"ready\":true") != std::string::npos,
                  "JSON status reports ready state");

    auto invalid = oracle::core::EngineConfig{};
    invalid.server_port = 0;
    runner.expect_throws([&invalid] { oracle::runtime::Engine rejected(invalid); },
                         "invalid server port is rejected");
}

}  // namespace

int main() {
    TestRunner runner;
    try {
        test_tensor(runner);
        oracle::runtime::Engine engine(oracle::core::EngineConfig{});
        test_elementwise(engine.backend(), runner);
        test_matmul(engine.backend(), runner);
        test_normalization_and_activations(engine.backend(), runner);
        test_reference_pipeline(engine.backend(), runner);
        test_engine_status(runner);
    } catch (const std::exception& error) {
        std::cerr << "UNCAUGHT TEST ERROR: " << error.what() << '\n';
        return 1;
    }

    if (runner.failures() == 0) {
        std::cout << "all Phase 1A tests passed\n";
    }
    return runner.result();
}
