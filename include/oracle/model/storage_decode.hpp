#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace oracle::model {

inline constexpr std::uint32_t ggml_type_f16 = 1;
inline constexpr std::uint32_t ggml_type_q5_k = 13;
inline constexpr std::uint32_t ggml_type_q6_k = 14;
inline constexpr std::uint32_t ggml_type_bf16 = 30;

inline constexpr std::size_t qk_k = 256;
inline constexpr std::size_t q5_k_block_bytes = 176;
inline constexpr std::size_t q6_k_block_bytes = 210;

struct StorageRowView {
    std::uint32_t type{0};
    std::size_t element_count{0};
    std::span<const std::byte> bytes;
};

[[nodiscard]] float f16_to_f32(std::uint16_t bits) noexcept;
[[nodiscard]] float bf16_to_f32(std::uint16_t bits) noexcept;

[[nodiscard]] StorageRowView make_storage_row_view(std::uint32_t type,
                                                   std::size_t element_count,
                                                   std::span<const std::byte> bytes);

void decode_storage_row(StorageRowView row, std::span<float> output);

}  // namespace oracle::model
