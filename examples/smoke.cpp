#include "oracle/core/config.hpp"
#include "oracle/core/tensor.hpp"
#include "oracle/runtime/engine.hpp"

#include <iostream>

int main() {
    oracle::runtime::Engine engine(oracle::core::EngineConfig{});
    oracle::core::Tensor lhs({4});
    oracle::core::Tensor rhs({4});
    oracle::core::Tensor out({4});

    for (std::size_t index = 0; index < 4; ++index) {
        lhs.data()[index] = static_cast<float>(index);
        rhs.data()[index] = 10.0F;
    }

    engine.backend().add(lhs, rhs, out);
    std::cout << engine.status() << "\nresult:";
    for (const float value : out.data()) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}
