#include "oracle/model/ggml_type.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace oracle::model {
namespace {

// Keep this table synchronized with the upstream ggml_type enum and quant block layouts.
constexpr std::array<GgmlTypeLayout, 43> layouts{{
    {0, "f32", 1, 4, false},
    {1, "f16", 1, 2, false},
    {2, "q4_0", 32, 18, true},
    {3, "q4_1", 32, 20, true},
    {4, "deprecated_q4_2", 0, 0, false},
    {5, "deprecated_q4_3", 0, 0, false},
    {6, "q5_0", 32, 22, true},
    {7, "q5_1", 32, 24, true},
    {8, "q8_0", 32, 34, true},
    {9, "q8_1", 32, 36, true},
    {10, "q2_K", 256, 84, true},
    {11, "q3_K", 256, 110, true},
    {12, "q4_K", 256, 144, true},
    {13, "q5_K", 256, 176, true},
    {14, "q6_K", 256, 210, true},
    {15, "q8_K", 256, 292, true},
    {16, "iq2_xxs", 256, 66, true},
    {17, "iq2_xs", 256, 74, true},
    {18, "iq3_xxs", 256, 98, true},
    {19, "iq1_s", 256, 50, true},
    {20, "iq4_nl", 32, 18, true},
    {21, "iq3_s", 256, 110, true},
    {22, "iq2_s", 256, 82, true},
    {23, "iq4_xs", 256, 136, true},
    {24, "i8", 1, 1, false},
    {25, "i16", 1, 2, false},
    {26, "i32", 1, 4, false},
    {27, "i64", 1, 8, false},
    {28, "f64", 1, 8, false},
    {29, "iq1_m", 256, 56, true},
    {30, "bf16", 1, 2, false},
    {31, "removed_q4_0_4_4", 0, 0, false},
    {32, "removed_q4_0_4_8", 0, 0, false},
    {33, "removed_q4_0_8_8", 0, 0, false},
    {34, "tq1_0", 256, 54, true},
    {35, "tq2_0", 256, 66, true},
    {36, "removed_iq4_nl_4_4", 0, 0, false},
    {37, "removed_iq4_nl_4_8", 0, 0, false},
    {38, "removed_iq4_nl_8_8", 0, 0, false},
    {39, "mxfp4", 32, 17, true},
    {40, "nvfp4", 64, 36, true},
    {41, "q1_0", 128, 18, true},
    {42, "q2_0", 64, 18, true},
}};

[[nodiscard]] std::uint64_t checked_multiply(std::uint64_t left,
                                             std::uint64_t right,
                                             std::string_view context) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left * right;
}

}  // namespace

const GgmlTypeLayout* ggml_type_layout(std::uint32_t type) noexcept {
    if (type >= layouts.size()) {
        return nullptr;
    }
    const GgmlTypeLayout& layout = layouts[type];
    return layout.block_elements == 0 || layout.bytes_per_block == 0 ? nullptr : &layout;
}

std::uint64_t ggml_row_byte_size(std::uint64_t row_elements, std::uint32_t type) {
    const GgmlTypeLayout* layout = ggml_type_layout(type);
    if (layout == nullptr) {
        throw std::runtime_error("unsupported GGML tensor type: " + std::to_string(type));
    }
    if (row_elements == 0) {
        throw std::invalid_argument("GGML tensor row must contain at least one element");
    }
    if (row_elements % layout->block_elements != 0) {
        throw std::runtime_error("GGML tensor row length " + std::to_string(row_elements) +
                                 " is not divisible by block size " +
                                 std::to_string(layout->block_elements) + " for type " +
                                 std::string(layout->name));
    }
    return checked_multiply(row_elements / layout->block_elements,
                            layout->bytes_per_block,
                            "GGML row byte size");
}

std::uint64_t ggml_tensor_element_count(std::span<const std::uint64_t> dimensions) {
    if (dimensions.empty()) {
        throw std::invalid_argument("GGML tensor must have at least one dimension");
    }
    std::uint64_t elements = 1;
    for (const std::uint64_t dimension : dimensions) {
        if (dimension == 0) {
            throw std::invalid_argument("GGML tensor dimensions must be non-zero");
        }
        elements = checked_multiply(elements, dimension, "GGML tensor element count");
    }
    return elements;
}

std::uint64_t ggml_tensor_byte_size(std::span<const std::uint64_t> dimensions,
                                    std::uint32_t type) {
    static_cast<void>(ggml_tensor_element_count(dimensions));
    std::uint64_t bytes = ggml_row_byte_size(dimensions.front(), type);
    for (std::size_t dimension = 1; dimension < dimensions.size(); ++dimension) {
        bytes = checked_multiply(bytes, dimensions[dimension], "GGML tensor byte size");
    }
    return bytes;
}

}  // namespace oracle::model
