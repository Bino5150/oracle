#include "oracle/model/mapped_gguf.hpp"
#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/model/qwen35_weights.hpp"

#include <exception>
#include <iostream>
#include <string_view>

namespace {

void print_usage(std::string_view program) {
    std::cerr << "usage: " << program << " <model.gguf> [--json]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        print_usage(argv[0]);
        return 2;
    }
    const bool json = argc == 3 && std::string_view(argv[2]) == "--json";
    if (argc == 3 && !json) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const oracle::model::MappedGgufModel model(argv[1]);
        const oracle::model::Qwen35Manifest manifest =
            oracle::model::load_qwen35_manifest(model.file());
        const oracle::model::Qwen35Weights weights =
            oracle::model::bind_qwen35_weights(model, manifest);
        std::cout << (json ? oracle::model::qwen35_binding_report_json(weights)
                          : oracle::model::qwen35_binding_report_text(weights));
        if (json) {
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "oracle-qwen35-bind: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
