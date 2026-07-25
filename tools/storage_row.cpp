#include "oracle/model/mapped_gguf.hpp"
#include "oracle/model/storage_decode.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] std::size_t parse_row_index(std::string_view text) {
    std::size_t value = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || position != end) {
        throw std::invalid_argument("row index must be a non-negative integer");
    }
    return value;
}

void print_usage(std::string_view program) {
    std::cerr << "usage: " << program
              << " <model.gguf> <tensor-name> <row-index> [--json]\n";
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned int>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

void print_json(const oracle::model::GgufTensorView& tensor,
                std::size_t row_index,
                std::span<const float> values) {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
              << "{\"tensor\":\"" << json_escape(tensor.name())
              << "\",\"ggml_type\":"
              << tensor.layout().type << ",\"type_name\":\"" << tensor.layout().name
              << "\",\"row_index\":" << row_index << ",\"row_count\":"
              << oracle::model::gguf_tensor_row_count(tensor) << ",\"element_count\":"
              << values.size() << ",\"values\":[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        if (std::isfinite(values[index])) {
            std::cout << values[index];
        } else {
            std::cout << "null";
        }
    }
    std::cout << "]}\n";
}

void print_text(const oracle::model::GgufTensorView& tensor,
                std::size_t row_index,
                std::span<const float> values) {
    constexpr std::size_t preview_limit = 16;
    std::cout << "tensor: " << tensor.name() << '\n'
              << "storage: " << tensor.layout().name << " (type " << tensor.layout().type
              << ")\n"
              << "row: " << row_index << " ("
              << oracle::model::gguf_tensor_row_count(tensor) << " rows)\n"
              << "elements: " << values.size() << '\n'
              << "preview:";
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    const std::size_t preview = std::min(preview_limit, values.size());
    for (std::size_t index = 0; index < preview; ++index) {
        std::cout << ' ' << values[index];
    }
    if (preview < values.size()) {
        std::cout << " ...";
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        print_usage(argv[0]);
        return 2;
    }
    const bool json = argc == 5 && std::string_view(argv[4]) == "--json";
    if (argc == 5 && !json) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const oracle::model::MappedGgufModel model(argv[1]);
        const oracle::model::GgufTensorView& tensor = model.tensor(argv[2]);
        const std::size_t row_index = parse_row_index(argv[3]);
        const oracle::model::StorageRowView row =
            oracle::model::make_storage_row_view(tensor, row_index);
        std::vector<float> values(row.element_count);
        oracle::model::decode_storage_row(row, values);

        if (json) {
            print_json(tensor, row_index, values);
        } else {
            print_text(tensor, row_index, values);
        }
    } catch (const std::exception& error) {
        std::cerr << "oracle-storage-row: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
