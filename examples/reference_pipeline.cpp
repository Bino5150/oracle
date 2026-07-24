#include "oracle/core/config.hpp"
#include "oracle/runtime/engine.hpp"
#include "oracle/runtime/reference_pipeline.hpp"

#include <iomanip>
#include <iostream>

int main() {
    oracle::runtime::Engine engine(oracle::core::EngineConfig{});

    oracle::core::Tensor input({2, 4});
    oracle::runtime::ReferencePipelineWeights weights{
        oracle::core::Tensor({4}),
        oracle::core::Tensor({4, 6}),
        oracle::core::Tensor({6, 4}),
    };

    const float input_values[] = {0.25F, -0.50F, 0.75F, 1.00F,
                                  1.00F, 0.50F, -0.25F, -0.75F};
    for (std::size_t index = 0; index < input.element_count(); ++index) {
        input.data()[index] = input_values[index];
    }
    weights.norm.fill(1.0F);

    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 6; ++column) {
            const int pattern = static_cast<int>((row + column) % 5) - 2;
            weights.up_projection.data()[row * 6 + column] =
                0.075F * static_cast<float>(pattern);
        }
    }
    for (std::size_t row = 0; row < 6; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            const int pattern = static_cast<int>((row * 2 + column) % 7) - 3;
            weights.down_projection.data()[row * 4 + column] =
                0.050F * static_cast<float>(pattern);
        }
    }

    const auto output =
        oracle::runtime::run_reference_pipeline(engine.backend(), input, weights);

    std::cout << engine.status() << "\nreference pipeline probabilities:\n";
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t row = 0; row < output.shape()[0]; ++row) {
        std::cout << "  row " << row << ':';
        for (std::size_t column = 0; column < output.shape()[1]; ++column) {
            std::cout << ' ' << output.data()[row * output.shape()[1] + column];
        }
        std::cout << '\n';
    }
}
