#include "oracle/backend/cpu/quantized_reference.hpp"

#include "oracle/model/ggml_type.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace oracle::backend::cpu {
namespace {

[[nodiscard]] std::size_t checked_multiply(std::size_t left,
                                           std::size_t right,
                                           const char* context) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left * right;
}

}  // namespace

float reference_storage_dot(model::StorageRowView row, std::span<const float> vector) {
    const model::StorageRowView validated =
        model::make_storage_row_view(row.type, row.element_count, row.bytes);
    if (vector.size() != validated.element_count) {
        throw std::invalid_argument("reference dot vector length mismatch");
    }

    constexpr std::size_t block_elements = model::qk_k;
    std::array<float, block_elements> decoded{};
    float sum = 0.0F;

    if (validated.type == model::ggml_type_f32 || validated.type == model::ggml_type_f16 ||
        validated.type == model::ggml_type_bf16) {
        std::array<float, 1> scalar{};
        const std::size_t scalar_bytes_count =
            validated.type == model::ggml_type_f32 ? 4U : 2U;
        for (std::size_t index = 0; index < validated.element_count; ++index) {
            const auto scalar_bytes = validated.bytes.subspan(index * scalar_bytes_count,
                                                                scalar_bytes_count);
            model::decode_storage_row(
                model::make_storage_row_view(validated.type, 1, scalar_bytes), scalar);
            sum += scalar[0] * vector[index];
        }
        return sum;
    }

    const std::size_t block_bytes = validated.type == model::ggml_type_q5_k
                                        ? model::q5_k_block_bytes
                                        : model::q6_k_block_bytes;
    for (std::size_t offset = 0, element = 0; offset < validated.bytes.size();
         offset += block_bytes, element += block_elements) {
        model::decode_storage_row(
            model::make_storage_row_view(validated.type,
                                         block_elements,
                                         validated.bytes.subspan(offset, block_bytes)),
            decoded);
        for (std::size_t lane = 0; lane < block_elements; ++lane) {
            sum += decoded[lane] * vector[element + lane];
        }
    }
    return sum;
}


void reference_decode_mapped_tensor(const model::GgufTensorView& tensor,
                                    std::span<float> output) {
    const std::size_t row_count = model::gguf_tensor_row_count(tensor);
    const auto dimensions = tensor.dimensions();
    if (dimensions.empty()) {
        throw std::invalid_argument("mapped tensor must have at least one dimension");
    }
    if (dimensions.front() > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("mapped tensor row width exceeds addressable size");
    }
    const std::size_t row_width = static_cast<std::size_t>(dimensions.front());
    const std::size_t expected = checked_multiply(row_count, row_width,
                                                  "mapped tensor element count");
    if (output.size() != expected) {
        throw std::invalid_argument("mapped tensor decode output length mismatch");
    }
    for (std::size_t row = 0; row < row_count; ++row) {
        model::decode_storage_row(model::make_storage_row_view(tensor, row),
                                  output.subspan(row * row_width, row_width));
    }
}

void reference_mapped_tensor_matvec(const model::GgufTensorView& tensor,
                                    std::span<const float> vector,
                                    std::span<float> output) {
    const auto dimensions = tensor.dimensions();
    if (dimensions.size() != 2) {
        throw std::invalid_argument("mapped tensor matvec requires a rank-2 tensor");
    }
    if (dimensions[0] > std::numeric_limits<std::size_t>::max() ||
        dimensions[1] > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("mapped tensor matvec dimensions exceed addressable size");
    }
    const std::size_t columns = static_cast<std::size_t>(dimensions[0]);
    const std::size_t rows = static_cast<std::size_t>(dimensions[1]);
    reference_storage_matvec(tensor.layout().type,
                             rows,
                             columns,
                             tensor.bytes(),
                             vector,
                             output);
}

void reference_storage_matvec(std::uint32_t type,
                              std::size_t row_count,
                              std::size_t column_count,
                              std::span<const std::byte> matrix_bytes,
                              std::span<const float> vector,
                              std::span<float> output) {
    if (row_count == 0 || column_count == 0) {
        throw std::invalid_argument("reference matvec dimensions must be non-zero");
    }
    if (vector.size() != column_count) {
        throw std::invalid_argument("reference matvec vector length mismatch");
    }
    if (output.size() != row_count) {
        throw std::invalid_argument("reference matvec output length mismatch");
    }

    const std::uint64_t row_bytes_u64 = model::ggml_row_byte_size(column_count, type);
    if (row_bytes_u64 > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("reference matvec row size exceeds addressable size");
    }
    const std::size_t row_bytes = static_cast<std::size_t>(row_bytes_u64);
    const std::size_t expected_bytes = checked_multiply(row_count, row_bytes,
                                                        "reference matvec matrix size");
    if (matrix_bytes.size() != expected_bytes) {
        throw std::invalid_argument("reference matvec matrix byte size mismatch");
    }

    for (std::size_t row = 0; row < row_count; ++row) {
        output[row] = reference_storage_dot(
            model::make_storage_row_view(type,
                                         column_count,
                                         matrix_bytes.subspan(row * row_bytes, row_bytes)),
            vector);
    }
}

}  // namespace oracle::backend::cpu
