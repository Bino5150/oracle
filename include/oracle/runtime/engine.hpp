#pragma once

#include "oracle/backend/backend.hpp"
#include "oracle/core/config.hpp"
#include "oracle/scheduler/scheduler.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace oracle::runtime {

struct EngineStatus {
    bool ready{false};
    std::string_view version;
    std::string_view backend;
    std::size_t context_length{0};
    std::size_t max_concurrent_requests{0};
    std::string server_host;
    std::uint16_t server_port{0};

    [[nodiscard]] std::string to_text() const;
    [[nodiscard]] std::string to_json() const;
};

class Engine {
public:
    explicit Engine(core::EngineConfig config);

    [[nodiscard]] EngineStatus status_snapshot() const;
    [[nodiscard]] std::string status() const;
    [[nodiscard]] std::string status_json() const;
    [[nodiscard]] const core::EngineConfig& config() const noexcept;
    [[nodiscard]] backend::IBackend& backend() noexcept;
    [[nodiscard]] const backend::IBackend& backend() const noexcept;

private:
    core::EngineConfig config_;
    std::unique_ptr<backend::IBackend> backend_;
    scheduler::Scheduler scheduler_;
};

}  // namespace oracle::runtime
