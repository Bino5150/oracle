#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace oracle::model {

enum class GgufMetadataType : std::uint32_t {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

[[nodiscard]] std::string_view gguf_metadata_type_name(GgufMetadataType type) noexcept;

class GgufValue;

struct GgufArray {
    GgufMetadataType element_type{GgufMetadataType::uint8};
    std::vector<GgufValue> values;
};

using GgufValueStorage =
    std::variant<std::uint8_t,
                 std::int8_t,
                 std::uint16_t,
                 std::int16_t,
                 std::uint32_t,
                 std::int32_t,
                 float,
                 bool,
                 std::string,
                 std::shared_ptr<GgufArray>,
                 std::uint64_t,
                 std::int64_t,
                 double>;

class GgufValue {
public:
    GgufValue(GgufMetadataType type, GgufValueStorage storage);

    [[nodiscard]] GgufMetadataType type() const noexcept;
    [[nodiscard]] const GgufValueStorage& storage() const noexcept;

    template <typename T>
    [[nodiscard]] const T* get_if() const noexcept {
        return std::get_if<T>(&storage_);
    }

private:
    GgufMetadataType type_;
    GgufValueStorage storage_;
};

struct GgufMetadataEntry {
    std::string key;
    GgufValue value;
};

struct GgufTensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t ggml_type{0};
    std::uint64_t relative_offset{0};
};

class GgufFile {
public:
    std::filesystem::path path;
    std::uint32_t version{0};
    std::uint64_t tensor_count{0};
    std::uint64_t metadata_count{0};
    std::uint32_t alignment{32};
    std::uint64_t data_offset{0};
    std::uint64_t file_size{0};
    std::vector<GgufMetadataEntry> metadata;
    std::vector<GgufTensorInfo> tensors;

    [[nodiscard]] const GgufMetadataEntry* find_metadata(std::string_view key) const noexcept;
    [[nodiscard]] const GgufTensorInfo* find_tensor(std::string_view name) const noexcept;
};

class GgufReader {
public:
    [[nodiscard]] static GgufFile read(const std::filesystem::path& path);
};

struct GgufMetadataQuery {
    std::string_view exact_key{};
    std::string_view prefix{};
};

enum class GgufMetadataJsonMode {
    full,
    compact,
};

[[nodiscard]] bool gguf_metadata_matches(std::string_view key,
                                         GgufMetadataQuery query) noexcept;
[[nodiscard]] std::size_t gguf_metadata_match_count(const GgufFile& file,
                                                    GgufMetadataQuery query = {}) noexcept;
[[nodiscard]] std::string gguf_value_to_string(const GgufValue& value);
[[nodiscard]] std::string gguf_value_preview(const GgufValue& value,
                                             std::size_t max_array_items = 8,
                                             std::size_t max_string_bytes = 160);
[[nodiscard]] std::string gguf_metadata_entries_json(
    const GgufFile& file,
    GgufMetadataJsonMode mode = GgufMetadataJsonMode::full,
    GgufMetadataQuery query = {});
[[nodiscard]] std::string gguf_metadata_report_json(const GgufFile& file,
                                                    GgufMetadataQuery query = {});
[[nodiscard]] std::string gguf_summary_json(const GgufFile& file);

}  // namespace oracle::model
