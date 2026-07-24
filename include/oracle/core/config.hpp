#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace oracle::core {

struct EngineConfig {
    std::string backend{"cpu"};
    std::size_t context_length{4096};
    std::size_t max_concurrent_requests{1};
    std::string server_host{"127.0.0.1"};
    std::uint16_t server_port{5150};
    bool verbose{false};

    [[nodiscard]] bool valid() const noexcept;
};

}  // namespace oracle::core
