#include "oracle/model/storage_decode.hpp"

#include "oracle/model/ggml_type.hpp"
#include "oracle/model/mapped_gguf.hpp"

#include <bit>
#include <limits>
#include <stdexcept>
#include <string>

namespace oracle::model {
namespace {

[[nodiscard]] std::uint8_t byte_value(std::byte value) noexcept {
    return std::to_integer<std::uint8_t>(value);
}

[[nodiscard]] int signed_byte_value(std::byte value) noexcept {
    const std::uint8_t raw = byte_value(value);
    return raw < 0x80U ? static_cast<int>(raw) : static_cast<int>(raw) - 256;
}

[[nodiscard]] std::uint16_t load_u16_le(std::span<const std::byte> bytes,
                                        std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 2) {
        throw std::out_of_range("16-bit storage read exceeds row bounds");
    }
    return static_cast<std::uint16_t>(byte_value(bytes[offset])) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(byte_value(bytes[offset + 1]))
                                      << 8U);
}

void unpack_scale_min_k4(std::size_t index,
                         std::span<const std::byte, 12> packed,
                         std::uint8_t& scale,
                         std::uint8_t& minimum) noexcept {
    if (index < 4) {
        scale = static_cast<std::uint8_t>(byte_value(packed[index]) & 0x3FU);
        minimum = static_cast<std::uint8_t>(byte_value(packed[index + 4]) & 0x3FU);
        return;
    }
    const std::uint32_t scale_low = byte_value(packed[index + 4]) & 0x0FU;
    const std::uint32_t scale_high =
        static_cast<std::uint32_t>(byte_value(packed[index - 4]) >> 6U) << 4U;
    const std::uint32_t minimum_low = byte_value(packed[index + 4]) >> 4U;
    const std::uint32_t minimum_high =
        static_cast<std::uint32_t>(byte_value(packed[index]) >> 6U) << 4U;
    scale = static_cast<std::uint8_t>(scale_low | scale_high);
    minimum = static_cast<std::uint8_t>(minimum_low | minimum_high);
}

void decode_q5_k_block(std::span<const std::byte, q5_k_block_bytes> block,
                       std::span<float, qk_k> output) {
    const float d = f16_to_f32(load_u16_le(block, 0));
    const float dmin = f16_to_f32(load_u16_le(block, 2));
    const auto scales = block.subspan<4, 12>();
    const auto high = block.subspan<16, 32>();
    const auto low = block.subspan<48, 128>();

    for (std::size_t index = 0; index < qk_k; ++index) {
        const std::size_t group = index / 64;
        const std::size_t within_group = index % 64;
        const bool upper_nibble = within_group >= 32;
        const std::size_t packed_index = group * 32 + (within_group % 32);
        const std::uint8_t packed = byte_value(low[packed_index]);
        std::uint8_t quant = upper_nibble ? static_cast<std::uint8_t>(packed >> 4U)
                                          : static_cast<std::uint8_t>(packed & 0x0FU);
        const std::uint8_t high_mask = static_cast<std::uint8_t>(
            1U << static_cast<unsigned>(group * 2 + (upper_nibble ? 1 : 0)));
        if ((byte_value(high[within_group % 32]) & high_mask) != 0) {
            quant = static_cast<std::uint8_t>(quant + 16U);
        }

        std::uint8_t scale = 0;
        std::uint8_t minimum = 0;
        unpack_scale_min_k4(group * 2 + (upper_nibble ? 1U : 0U),
                            scales,
                            scale,
                            minimum);
        output[index] = d * static_cast<float>(scale) * static_cast<float>(quant) -
                        dmin * static_cast<float>(minimum);
    }
}

void decode_q6_k_block(std::span<const std::byte, q6_k_block_bytes> block,
                       std::span<float, qk_k> output) {
    const auto low = block.subspan<0, 128>();
    const auto high = block.subspan<128, 64>();
    const auto scales = block.subspan<192, 16>();
    const float d = f16_to_f32(load_u16_le(block, 208));

    for (std::size_t index = 0; index < qk_k; ++index) {
        const std::size_t half = index / 128;
        const std::size_t position = index % 128;
        const std::size_t segment = position / 32;
        const std::size_t lane = position % 32;
        const std::size_t low_base = half * 64;
        const std::size_t high_base = half * 32;

        const std::size_t low_offset = low_base + lane + ((segment == 1 || segment == 3) ? 32U : 0U);
        const std::uint8_t packed_low = byte_value(low[low_offset]);
        const std::uint8_t low_bits = segment >= 2
                                          ? static_cast<std::uint8_t>(packed_low >> 4U)
                                          : static_cast<std::uint8_t>(packed_low & 0x0FU);
        const unsigned high_shift = static_cast<unsigned>(segment * 2);
        const std::uint8_t high_bits = static_cast<std::uint8_t>(
            (byte_value(high[high_base + lane]) >> high_shift) & 0x03U);
        const int quant = static_cast<int>(low_bits | static_cast<std::uint8_t>(high_bits << 4U)) - 32;

        const std::size_t scale_index = half * 8 + lane / 16 + segment * 2;
        const int scale = signed_byte_value(scales[scale_index]);
        output[index] = d * static_cast<float>(scale) * static_cast<float>(quant);
    }
}

[[nodiscard]] std::size_t checked_size(std::uint64_t value, const char* context) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(std::string(context) + " exceeds addressable size");
    }
    return static_cast<std::size_t>(value);
}

}  // namespace

float f16_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    std::uint32_t exponent = (bits >> 10U) & 0x1FU;
    std::uint32_t mantissa = bits & 0x03FFU;

    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            int unbiased_exponent = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --unbiased_exponent;
            }
            mantissa &= 0x03FFU;
            const auto f32_exponent = static_cast<std::uint32_t>(unbiased_exponent + 127);
            result = sign | (f32_exponent << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        result = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        exponent += 112U;
        result = sign | (exponent << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(result);
}

float bf16_to_f32(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

StorageRowView make_storage_row_view(std::uint32_t type,
                                    std::size_t element_count,
                                    std::span<const std::byte> bytes) {
    const std::size_t expected = checked_size(ggml_row_byte_size(element_count, type),
                                              "GGML row byte size");
    if (bytes.size() != expected) {
        throw std::invalid_argument("storage row byte size mismatch: expected " +
                                    std::to_string(expected) + ", received " +
                                    std::to_string(bytes.size()));
    }
    if (type != ggml_type_f16 && type != ggml_type_bf16 && type != ggml_type_q5_k &&
        type != ggml_type_q6_k) {
        throw std::invalid_argument("storage decoder does not support GGML type " +
                                    std::to_string(type));
    }
    return StorageRowView{type, element_count, bytes};
}

std::size_t gguf_tensor_row_count(const GgufTensorView& tensor) {
    const auto dimensions = tensor.dimensions();
    if (dimensions.empty()) {
        throw std::invalid_argument("GGUF tensor must have at least one dimension");
    }

    const std::uint64_t row_elements = dimensions.front();
    const std::uint64_t total_elements = tensor.element_count();
    if (row_elements == 0 || total_elements % row_elements != 0) {
        throw std::logic_error("GGUF tensor has invalid row geometry");
    }
    return checked_size(total_elements / row_elements, "GGUF tensor row count");
}

StorageRowView make_storage_row_view(const GgufTensorView& tensor,
                                     std::size_t row_index) {
    const auto dimensions = tensor.dimensions();
    if (dimensions.empty()) {
        throw std::invalid_argument("GGUF tensor must have at least one dimension");
    }

    const std::size_t row_elements = checked_size(dimensions.front(), "GGUF tensor row width");
    const std::size_t row_count = gguf_tensor_row_count(tensor);
    if (row_index >= row_count) {
        throw std::out_of_range("GGUF tensor row index exceeds row count");
    }

    const std::size_t row_bytes = checked_size(
        ggml_row_byte_size(dimensions.front(), tensor.layout().type),
        "GGUF tensor row byte size");
    if (row_count != 0 && row_bytes > std::numeric_limits<std::size_t>::max() / row_count) {
        throw std::overflow_error("GGUF tensor byte size exceeds addressable size");
    }
    const std::size_t expected_tensor_bytes = row_bytes * row_count;
    if (tensor.bytes().size() != expected_tensor_bytes) {
        throw std::invalid_argument("GGUF tensor byte size does not match row geometry");
    }

    const std::size_t offset = row_index * row_bytes;
    return make_storage_row_view(tensor.layout().type,
                                 row_elements,
                                 tensor.bytes().subspan(offset, row_bytes));
}

void decode_storage_row(StorageRowView row, std::span<float> output) {
    const StorageRowView validated = make_storage_row_view(row.type, row.element_count, row.bytes);
    if (output.size() != validated.element_count) {
        throw std::invalid_argument("decoded row output size mismatch");
    }

    if (validated.type == ggml_type_f16 || validated.type == ggml_type_bf16) {
        for (std::size_t index = 0; index < validated.element_count; ++index) {
            const std::uint16_t bits = load_u16_le(validated.bytes, index * 2);
            output[index] = validated.type == ggml_type_f16 ? f16_to_f32(bits)
                                                            : bf16_to_f32(bits);
        }
        return;
    }

    const std::size_t block_bytes = validated.type == ggml_type_q5_k ? q5_k_block_bytes
                                                                     : q6_k_block_bytes;
    for (std::size_t offset = 0, element = 0; offset < validated.bytes.size();
         offset += block_bytes, element += qk_k) {
        if (validated.type == ggml_type_q5_k) {
            decode_q5_k_block(
                std::span<const std::byte, q5_k_block_bytes>(validated.bytes.data() + offset,
                                                            q5_k_block_bytes),
                std::span<float, qk_k>(output.data() + element, qk_k));
        } else {
            decode_q6_k_block(
                std::span<const std::byte, q6_k_block_bytes>(validated.bytes.data() + offset,
                                                            q6_k_block_bytes),
                std::span<float, qk_k>(output.data() + element, qk_k));
        }
    }
}

}  // namespace oracle::model
