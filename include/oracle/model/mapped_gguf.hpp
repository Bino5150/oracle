#pragma once

#include "oracle/core/mapped_file.hpp"
#include "oracle/model/ggml_type.hpp"
#include "oracle/model/gguf.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace oracle::model {

class GgufTensorView {
public:
    GgufTensorView(const GgufTensorInfo* info,
                   const GgmlTypeLayout* layout,
                   const std::byte* data,
                   std::size_t byte_size,
                   std::uint64_t absolute_offset);

    [[nodiscard]] const GgufTensorInfo& info() const noexcept;
    [[nodiscard]] const GgmlTypeLayout& layout() const noexcept;
    [[nodiscard]] std::string_view name() const noexcept;
    [[nodiscard]] std::span<const std::uint64_t> dimensions() const noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] std::uint64_t absolute_offset() const noexcept;
    [[nodiscard]] std::uint64_t relative_offset() const noexcept;
    [[nodiscard]] std::uint64_t element_count() const;

private:
    const GgufTensorInfo* info_;
    const GgmlTypeLayout* layout_;
    const std::byte* data_;
    std::size_t byte_size_;
    std::uint64_t absolute_offset_;
};

struct GgufMappingStats {
    std::uint64_t file_size{0};
    std::uint64_t data_offset{0};
    std::uint64_t mapped_bytes{0};
    std::uint64_t validated_payload_bytes{0};
    std::uint64_t tensor_span_bytes{0};
    std::uint64_t tensor_count{0};
};

class MappedGgufModel {
public:
    explicit MappedGgufModel(const std::filesystem::path& path);

    MappedGgufModel(const MappedGgufModel&) = delete;
    MappedGgufModel& operator=(const MappedGgufModel&) = delete;
    MappedGgufModel(MappedGgufModel&&) = delete;
    MappedGgufModel& operator=(MappedGgufModel&&) = delete;

    [[nodiscard]] const GgufFile& file() const noexcept;
    [[nodiscard]] const core::MappedFile& mapping() const noexcept;
    [[nodiscard]] const std::vector<GgufTensorView>& tensors() const noexcept;
    [[nodiscard]] const GgufTensorView* find_tensor(std::string_view name) const noexcept;
    [[nodiscard]] const GgufTensorView& tensor(std::string_view name) const;
    [[nodiscard]] const GgufMappingStats& stats() const noexcept;

private:
    GgufFile file_;
    core::MappedFile mapping_;
    std::vector<GgufTensorView> tensors_;
    std::unordered_map<std::string, std::size_t> tensor_index_;
    GgufMappingStats stats_;
};

[[nodiscard]] std::string mapped_gguf_summary_json(const MappedGgufModel& model);
[[nodiscard]] std::string gguf_tensor_view_json(const GgufTensorView& tensor);

}  // namespace oracle::model
