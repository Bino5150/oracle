#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace oracle::model {

struct GgmlTypeLayout {
    std::uint32_t type{0};
    std::string_view name;
    std::uint64_t block_elements{0};
    std::uint64_t bytes_per_block{0};
    bool quantized{false};
};

[[nodiscard]] const GgmlTypeLayout* ggml_type_layout(std::uint32_t type) noexcept;
[[nodiscard]] std::uint64_t ggml_row_byte_size(std::uint64_t row_elements,
                                               std::uint32_t type);
[[nodiscard]] std::uint64_t ggml_tensor_element_count(
    std::span<const std::uint64_t> dimensions);
[[nodiscard]] std::uint64_t ggml_tensor_byte_size(
    std::span<const std::uint64_t> dimensions,
    std::uint32_t type);

}  // namespace oracle::model
