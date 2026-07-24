#include "oracle/core/config.hpp"
#include "oracle/runtime/engine.hpp"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    try {
        oracle::core::EngineConfig config;
        oracle::runtime::Engine engine(config);
        const bool json = argc > 1 && std::string_view(argv[1]) == "--json";
        std::cout << (json ? engine.status_json() : engine.status()) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "oracle-cli: " << error.what() << '\n';
        return 1;
    }
}
