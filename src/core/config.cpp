#include "oracle/core/config.hpp"

namespace oracle::core {

bool EngineConfig::valid() const noexcept {
    return !backend.empty() && context_length > 0 && max_concurrent_requests > 0 &&
           !server_host.empty() && server_port > 0;
}

}  // namespace oracle::core
