#include "oracle/core/config.hpp"
#include "oracle/core/tensor.hpp"
#include "oracle/runtime/engine.hpp"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
    std::size_t rows{64};
    std::size_t inner{256};
    std::size_t columns{256};
    std::size_t iterations{10};
};

[[nodiscard]] std::size_t parse_size(std::string_view value, std::string_view name) {
    std::size_t parsed = 0;
    try {
        parsed = static_cast<std::size_t>(std::stoull(std::string(value)));
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + std::string(name));
    }
    if (parsed == 0) {
        throw std::invalid_argument(std::string(name) + " must be greater than zero");
    }
    return parsed;
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value after " + std::string(argument));
        }
        const std::string_view value(argv[++index]);
        if (argument == "--rows") {
            options.rows = parse_size(value, argument);
        } else if (argument == "--inner") {
            options.inner = parse_size(value, argument);
        } else if (argument == "--columns") {
            options.columns = parse_size(value, argument);
        } else if (argument == "--iterations") {
            options.iterations = parse_size(value, argument);
        } else {
            throw std::invalid_argument("unknown benchmark option: " + std::string(argument));
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        oracle::runtime::Engine engine(oracle::core::EngineConfig{});
        oracle::core::Tensor lhs({options.rows, options.inner});
        oracle::core::Tensor rhs({options.inner, options.columns});
        oracle::core::Tensor out({options.rows, options.columns});

        for (std::size_t index = 0; index < lhs.element_count(); ++index) {
            lhs.data()[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0F;
        }
        for (std::size_t index = 0; index < rhs.element_count(); ++index) {
            rhs.data()[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 13.0F;
        }

        engine.backend().matmul(lhs, rhs, out);
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
            engine.backend().matmul(lhs, rhs, out);
        }
        const auto end = std::chrono::steady_clock::now();

        const double milliseconds =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double average_ms = milliseconds / static_cast<double>(options.iterations);
        const double operations = 2.0 * static_cast<double>(options.rows) *
                                  static_cast<double>(options.inner) *
                                  static_cast<double>(options.columns);
        const double gflops = operations / (average_ms * 1.0e6);

        double checksum = 0.0;
        for (const float value : out.data()) {
            checksum += static_cast<double>(value);
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "{\"benchmark\":\"f32_matmul\",\"backend\":\"" << engine.backend().name()
                  << "\",\"shape\":[" << options.rows << ',' << options.inner << ','
                  << options.columns << "],\"iterations\":" << options.iterations
                  << ",\"average_ms\":" << average_ms << ",\"gflops\":" << gflops
                  << ",\"checksum\":" << checksum << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "oracle-bench: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
