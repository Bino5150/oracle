#include "oracle/runtime/engine.hpp"

#include "oracle/version.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace oracle::runtime {
namespace {

[[nodiscard]] std::string escape_json(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += character;
                break;
        }
    }
    return escaped;
}

}  // namespace

std::string EngineStatus::to_text() const {
    std::ostringstream out;
    out << "Oracle engine " << (ready ? "ready" : "not ready") << " | version=" << version
        << " | backend=" << backend << " | context=" << context_length
        << " | concurrency=" << max_concurrent_requests << " | server=" << server_host << ':'
        << server_port;
    return out.str();
}

std::string EngineStatus::to_json() const {
    std::ostringstream out;
    out << "{\"engine\":\"oracle\",\"ready\":" << (ready ? "true" : "false")
        << ",\"version\":\"" << escape_json(version) << "\",\"backend\":\""
        << escape_json(backend) << "\",\"context_length\":" << context_length
        << ",\"max_concurrent_requests\":" << max_concurrent_requests
        << ",\"server\":{\"host\":\"" << escape_json(server_host) << "\",\"port\":"
        << server_port << "}}";
    return out.str();
}

Engine::Engine(core::EngineConfig config)
    : config_(std::move(config)),
      backend_(config_.backend == "cpu" ? backend::make_cpu_backend() : nullptr),
      scheduler_(config_.max_concurrent_requests) {
    if (!config_.valid()) {
        throw std::invalid_argument("invalid engine configuration");
    }
    if (!backend_) {
        throw std::invalid_argument("requested backend is not available: " + config_.backend);
    }
}

EngineStatus Engine::status_snapshot() const {
    return EngineStatus{true,
                        oracle::version,
                        backend_->name(),
                        config_.context_length,
                        scheduler_.capacity(),
                        config_.server_host,
                        config_.server_port};
}

std::string Engine::status() const { return status_snapshot().to_text(); }
std::string Engine::status_json() const { return status_snapshot().to_json(); }
const core::EngineConfig& Engine::config() const noexcept { return config_; }
backend::IBackend& Engine::backend() noexcept { return *backend_; }
const backend::IBackend& Engine::backend() const noexcept { return *backend_; }

}  // namespace oracle::runtime
