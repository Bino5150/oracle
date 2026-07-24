#include "oracle/model/gguf.hpp"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: oracle-gguf-inspect <model.gguf> [--json]\n";
        return 2;
    }

    const bool json = argc == 3 && std::string_view(argv[2]) == "--json";
    if (argc == 3 && !json) {
        std::cerr << "unknown option: " << argv[2] << '\n';
        return 2;
    }

    try {
        const oracle::model::GgufFile file = oracle::model::GgufReader::read(argv[1]);
        if (json) {
            std::cout << oracle::model::gguf_summary_json(file) << '\n';
            return 0;
        }

        std::cout << "Oracle GGUF inspector\n"
                  << "path=" << file.path << "\n"
                  << "version=" << file.version << " tensors=" << file.tensor_count
                  << " metadata=" << file.metadata_count << " alignment=" << file.alignment
                  << " data_offset=" << file.data_offset << " file_size=" << file.file_size
                  << "\n\nmetadata:\n";
        for (const auto& entry : file.metadata) {
            std::cout << "  " << entry.key << " ["
                      << oracle::model::gguf_metadata_type_name(entry.value.type())
                      << "] = " << oracle::model::gguf_value_to_string(entry.value) << '\n';
        }

        std::cout << "\ntensors:\n";
        for (const auto& tensor : file.tensors) {
            std::cout << "  " << tensor.name << " type=" << tensor.ggml_type << " dims=[";
            for (std::size_t index = 0; index < tensor.dimensions.size(); ++index) {
                if (index != 0) {
                    std::cout << ',';
                }
                std::cout << tensor.dimensions[index];
            }
            std::cout << "] relative_offset=" << tensor.relative_offset << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "GGUF inspection failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
