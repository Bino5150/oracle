#include "oracle/model/ggml_type.hpp"
#include "oracle/model/gguf.hpp"
#include "oracle/model/mapped_gguf.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
    std::string path;
    bool json{false};
    bool verify{false};
    std::optional<std::string> tensor_name;
};

[[nodiscard]] Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: oracle-gguf-inspect <model.gguf> [--verify] [--tensor <name>] [--json]");
    }

    Options options;
    options.path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--json") {
            options.json = true;
        } else if (argument == "--verify") {
            options.verify = true;
        } else if (argument == "--tensor") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--tensor requires a tensor name");
            }
            options.tensor_name = argv[++index];
            options.verify = true;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    return options;
}

void print_dimensions(std::span<const std::uint64_t> dimensions) {
    std::cout << '[';
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << dimensions[index];
    }
    std::cout << ']';
}

void print_tensor(const oracle::model::GgufTensorView& tensor) {
    std::cout << "  " << tensor.name() << " type=" << tensor.layout().name << '(' << tensor.layout().type
              << ") dims=";
    print_dimensions(tensor.dimensions());
    std::cout << " elements=" << tensor.element_count()
              << " relative_offset=" << tensor.relative_offset()
              << " absolute_offset=" << tensor.absolute_offset()
              << " bytes=" << tensor.bytes().size() << '\n';
}

void print_metadata_only(const oracle::model::GgufFile& file) {
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

    std::cout << "\ntensors (metadata only; use --verify for payload validation):\n";
    for (const auto& tensor : file.tensors) {
        const oracle::model::GgmlTypeLayout* layout =
            oracle::model::ggml_type_layout(tensor.ggml_type);
        std::cout << "  " << tensor.name << " type="
                  << (layout == nullptr ? "unsupported" : std::string(layout->name)) << '('
                  << tensor.ggml_type << ") dims=";
        print_dimensions(tensor.dimensions);
        std::cout << " relative_offset=" << tensor.relative_offset << '\n';
    }
}

void print_verified(const oracle::model::MappedGgufModel& model) {
    const auto& stats = model.stats();
    std::cout << "Oracle GGUF inspector — verified mapping\n"
              << "path=" << model.file().path << "\n"
              << "version=" << model.file().version << " tensors=" << stats.tensor_count
              << " metadata=" << model.file().metadata_count
              << " alignment=" << model.file().alignment << " file_size=" << stats.file_size
              << " data_offset=" << stats.data_offset << " mapped_bytes=" << stats.mapped_bytes
              << " validated_payload_bytes=" << stats.validated_payload_bytes
              << " tensor_span_bytes=" << stats.tensor_span_bytes << "\n\nmetadata:\n";
    for (const auto& entry : model.file().metadata) {
        std::cout << "  " << entry.key << " ["
                  << oracle::model::gguf_metadata_type_name(entry.value.type())
                  << "] = " << oracle::model::gguf_value_to_string(entry.value) << '\n';
    }
    std::cout << "\ntensors:\n";
    for (const auto& tensor : model.tensors()) {
        print_tensor(tensor);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.verify) {
            const oracle::model::MappedGgufModel model(options.path);
            if (options.tensor_name) {
                const auto& tensor = model.tensor(*options.tensor_name);
                if (options.json) {
                    std::cout << oracle::model::gguf_tensor_view_json(tensor) << '\n';
                } else {
                    std::cout << "Oracle GGUF tensor\n";
                    print_tensor(tensor);
                }
            } else if (options.json) {
                std::cout << oracle::model::mapped_gguf_summary_json(model) << '\n';
            } else {
                print_verified(model);
            }
            return 0;
        }

        const oracle::model::GgufFile file = oracle::model::GgufReader::read(options.path);
        if (options.json) {
            std::cout << oracle::model::gguf_summary_json(file) << '\n';
        } else {
            print_metadata_only(file);
        }
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "GGUF inspection failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
