#pragma once

#include "oracle/model/mapped_gguf.hpp"
#include "oracle/model/storage_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oracle::backend::cpu {

[[nodiscard]] float reference_storage_dot(model::StorageRowView row,
                                          std::span<const float> vector);

void reference_decode_mapped_tensor(const model::GgufTensorView& tensor,
                                    std::span<float> output);

void reference_mapped_tensor_matvec(const model::GgufTensorView& tensor,
                                    std::span<const float> vector,
                                    std::span<float> output);

void reference_storage_matvec(std::uint32_t type,
                              std::size_t row_count,
                              std::size_t column_count,
                              std::span<const std::byte> matrix_bytes,
                              std::span<const float> vector,
                              std::span<float> output);

}  // namespace oracle::backend::cpu
