#include "oracle/model/gguf.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace oracle::model {
namespace {

constexpr std::uint64_t max_entry_count = 1'000'000;
constexpr std::uint64_t max_string_bytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t max_array_depth = 32;

class BinaryReader {
public:
    explicit BinaryReader(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
        if (!stream_) {
            throw std::runtime_error("unable to open GGUF file: " + path.string());
        }
    }

    [[nodiscard]] std::uint64_t position() {
        const std::streampos position = stream_.tellg();
        if (position < 0) {
            throw std::runtime_error("unable to determine GGUF stream position");
        }
        return static_cast<std::uint64_t>(position);
    }

    void read_exact(void* destination, std::size_t byte_count) {
        if (byte_count == 0) {
            return;
        }
        stream_.read(static_cast<char*>(destination), static_cast<std::streamsize>(byte_count));
        if (!stream_) {
            throw std::runtime_error("unexpected end of GGUF file");
        }
    }

    template <typename Unsigned>
    [[nodiscard]] Unsigned read_unsigned_le() {
        static_assert(std::is_unsigned_v<Unsigned>);
        std::array<std::byte, sizeof(Unsigned)> bytes{};
        read_exact(bytes.data(), bytes.size());
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            value |= static_cast<std::uint64_t>(
                         std::to_integer<unsigned int>(bytes[index]))
                     << (index * 8U);
        }
        return static_cast<Unsigned>(value);
    }

    [[nodiscard]] std::uint8_t read_u8() { return read_unsigned_le<std::uint8_t>(); }
    [[nodiscard]] std::int8_t read_i8() {
        return static_cast<std::int8_t>(read_unsigned_le<std::uint8_t>());
    }
    [[nodiscard]] std::uint16_t read_u16() { return read_unsigned_le<std::uint16_t>(); }
    [[nodiscard]] std::int16_t read_i16() {
        return std::bit_cast<std::int16_t>(read_unsigned_le<std::uint16_t>());
    }
    [[nodiscard]] std::uint32_t read_u32() { return read_unsigned_le<std::uint32_t>(); }
    [[nodiscard]] std::int32_t read_i32() {
        return std::bit_cast<std::int32_t>(read_unsigned_le<std::uint32_t>());
    }
    [[nodiscard]] std::uint64_t read_u64() { return read_unsigned_le<std::uint64_t>(); }
    [[nodiscard]] std::int64_t read_i64() {
        return std::bit_cast<std::int64_t>(read_unsigned_le<std::uint64_t>());
    }
    [[nodiscard]] float read_f32() {
        return std::bit_cast<float>(read_unsigned_le<std::uint32_t>());
    }
    [[nodiscard]] double read_f64() {
        return std::bit_cast<double>(read_unsigned_le<std::uint64_t>());
    }

    [[nodiscard]] std::string read_string(std::uint64_t maximum = max_string_bytes) {
        const std::uint64_t length = read_u64();
        if (length > maximum || length > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("GGUF string length exceeds safety limit");
        }
        std::string value(static_cast<std::size_t>(length), '\0');
        read_exact(value.data(), value.size());
        return value;
    }

private:
    std::ifstream stream_;
};

[[nodiscard]] GgufMetadataType parse_metadata_type(std::uint32_t raw) {
    if (raw > static_cast<std::uint32_t>(GgufMetadataType::float64)) {
        throw std::runtime_error("unknown GGUF metadata type: " + std::to_string(raw));
    }
    return static_cast<GgufMetadataType>(raw);
}

[[nodiscard]] GgufValue read_value(BinaryReader& reader,
                                   GgufMetadataType type,
                                   std::size_t depth) {
    if (depth > max_array_depth) {
        throw std::runtime_error("GGUF metadata array nesting exceeds safety limit");
    }

    switch (type) {
        case GgufMetadataType::uint8:
            return {type, reader.read_u8()};
        case GgufMetadataType::int8:
            return {type, reader.read_i8()};
        case GgufMetadataType::uint16:
            return {type, reader.read_u16()};
        case GgufMetadataType::int16:
            return {type, reader.read_i16()};
        case GgufMetadataType::uint32:
            return {type, reader.read_u32()};
        case GgufMetadataType::int32:
            return {type, reader.read_i32()};
        case GgufMetadataType::float32:
            return {type, reader.read_f32()};
        case GgufMetadataType::boolean: {
            const std::uint8_t raw = reader.read_u8();
            if (raw > 1) {
                throw std::runtime_error("GGUF boolean metadata must be 0 or 1");
            }
            return {type, raw != 0};
        }
        case GgufMetadataType::string:
            return {type, reader.read_string()};
        case GgufMetadataType::array: {
            const GgufMetadataType element_type = parse_metadata_type(reader.read_u32());
            const std::uint64_t length = reader.read_u64();
            if (length > max_entry_count || length > std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error("GGUF metadata array exceeds safety limit");
            }
            auto array = std::make_shared<GgufArray>();
            array->element_type = element_type;
            array->values.reserve(static_cast<std::size_t>(length));
            for (std::uint64_t index = 0; index < length; ++index) {
                array->values.push_back(read_value(reader, element_type, depth + 1));
            }
            return {type, std::move(array)};
        }
        case GgufMetadataType::uint64:
            return {type, reader.read_u64()};
        case GgufMetadataType::int64:
            return {type, reader.read_i64()};
        case GgufMetadataType::float64:
            return {type, reader.read_f64()};
    }
    throw std::runtime_error("unreachable GGUF metadata type");
}

[[nodiscard]] std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || !std::has_single_bit(alignment)) {
        throw std::runtime_error("GGUF alignment must be a non-zero power of two");
    }
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw std::overflow_error("GGUF data offset overflow");
    }
    return (value + mask) & ~mask;
}

[[nodiscard]] std::size_t utf8_safe_prefix_length(std::string_view value,
                                                    std::size_t maximum) noexcept {
    if (value.size() <= maximum) {
        return value.size();
    }
    std::size_t length = maximum;
    while (length > 0 &&
           (static_cast<unsigned char>(value[length]) & 0xc0U) == 0x80U) {
        --length;
    }
    return length;
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

[[nodiscard]] std::string value_json(const GgufValue& value) {
    std::ostringstream output;
    switch (value.type()) {
        case GgufMetadataType::uint8:
            output << static_cast<unsigned int>(*value.get_if<std::uint8_t>());
            break;
        case GgufMetadataType::int8:
            output << static_cast<int>(*value.get_if<std::int8_t>());
            break;
        case GgufMetadataType::uint16:
            output << *value.get_if<std::uint16_t>();
            break;
        case GgufMetadataType::int16:
            output << *value.get_if<std::int16_t>();
            break;
        case GgufMetadataType::uint32:
            output << *value.get_if<std::uint32_t>();
            break;
        case GgufMetadataType::int32:
            output << *value.get_if<std::int32_t>();
            break;
        case GgufMetadataType::float32: {
            const float number = *value.get_if<float>();
            if (std::isfinite(number)) {
                output << std::setprecision(9) << number;
            } else if (std::isnan(number)) {
                output << "\"NaN\"";
            } else {
                output << (number > 0.0F ? "\"Infinity\"" : "\"-Infinity\"");
            }
            break;
        }
        case GgufMetadataType::boolean:
            output << (*value.get_if<bool>() ? "true" : "false");
            break;
        case GgufMetadataType::string:
            output << '"' << json_escape(*value.get_if<std::string>()) << '"';
            break;
        case GgufMetadataType::array: {
            output << '[';
            const auto& array = **value.get_if<std::shared_ptr<GgufArray>>();
            for (std::size_t index = 0; index < array.values.size(); ++index) {
                if (index != 0) {
                    output << ',';
                }
                output << value_json(array.values[index]);
            }
            output << ']';
            break;
        }
        case GgufMetadataType::uint64:
            output << *value.get_if<std::uint64_t>();
            break;
        case GgufMetadataType::int64:
            output << *value.get_if<std::int64_t>();
            break;
        case GgufMetadataType::float64: {
            const double number = *value.get_if<double>();
            if (std::isfinite(number)) {
                output << std::setprecision(17) << number;
            } else if (std::isnan(number)) {
                output << "\"NaN\"";
            } else {
                output << (number > 0.0 ? "\"Infinity\"" : "\"-Infinity\"");
            }
            break;
        }
    }
    return output.str();
}

[[nodiscard]] std::string value_preview_impl(const GgufValue& value,
                                             std::size_t max_array_items,
                                             std::size_t max_string_bytes) {
    if (value.type() == GgufMetadataType::string) {
        const std::string& text = *value.get_if<std::string>();
        if (text.size() <= max_string_bytes) {
            return value_json(value);
        }
        const std::size_t preview_bytes =
            utf8_safe_prefix_length(text, max_string_bytes);
        return "\"" + json_escape(std::string_view(text).substr(0, preview_bytes)) +
               "...\" (" + std::to_string(text.size()) + " bytes)";
    }

    if (value.type() != GgufMetadataType::array) {
        return value_json(value);
    }

    const auto& array = **value.get_if<std::shared_ptr<GgufArray>>();
    std::ostringstream output;
    output << "array<" << gguf_metadata_type_name(array.element_type) << ">["
           << array.values.size() << ']';
    if (array.values.empty() || max_array_items == 0) {
        return output.str();
    }

    output << " {";
    const std::size_t preview_count = std::min(array.values.size(), max_array_items);
    for (std::size_t index = 0; index < preview_count; ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << value_preview_impl(array.values[index], max_array_items, max_string_bytes);
    }
    if (preview_count < array.values.size()) {
        output << ", ...";
    }
    output << '}';
    return output.str();
}

void append_metadata_entry_json(std::ostringstream& output,
                                const GgufMetadataEntry& entry,
                                GgufMetadataJsonMode mode) {
    output << "{\"key\":\"" << json_escape(entry.key) << "\",\"type\":\""
           << gguf_metadata_type_name(entry.value.type()) << '"';

    if (entry.value.type() == GgufMetadataType::array) {
        const auto& array = **entry.value.get_if<std::shared_ptr<GgufArray>>();
        output << ",\"element_type\":\"" << gguf_metadata_type_name(array.element_type)
               << "\",\"length\":" << array.values.size();
        if (mode == GgufMetadataJsonMode::full) {
            output << ",\"value\":" << value_json(entry.value);
        }
    } else if (mode == GgufMetadataJsonMode::compact &&
               entry.value.type() == GgufMetadataType::string &&
               entry.value.get_if<std::string>()->size() > 512) {
        const std::string& text = *entry.value.get_if<std::string>();
        const std::size_t preview_bytes = utf8_safe_prefix_length(text, 512);
        output << ",\"length\":" << text.size() << ",\"preview\":\""
               << json_escape(std::string_view(text).substr(0, preview_bytes)) << "...\"";
    } else {
        output << ",\"value\":" << value_json(entry.value);
    }
    output << '}';
}

}  // namespace

std::string_view gguf_metadata_type_name(GgufMetadataType type) noexcept {
    switch (type) {
        case GgufMetadataType::uint8:
            return "uint8";
        case GgufMetadataType::int8:
            return "int8";
        case GgufMetadataType::uint16:
            return "uint16";
        case GgufMetadataType::int16:
            return "int16";
        case GgufMetadataType::uint32:
            return "uint32";
        case GgufMetadataType::int32:
            return "int32";
        case GgufMetadataType::float32:
            return "float32";
        case GgufMetadataType::boolean:
            return "bool";
        case GgufMetadataType::string:
            return "string";
        case GgufMetadataType::array:
            return "array";
        case GgufMetadataType::uint64:
            return "uint64";
        case GgufMetadataType::int64:
            return "int64";
        case GgufMetadataType::float64:
            return "float64";
    }
    return "unknown";
}

GgufValue::GgufValue(GgufMetadataType type, GgufValueStorage storage)
    : type_(type), storage_(std::move(storage)) {}

GgufMetadataType GgufValue::type() const noexcept { return type_; }
const GgufValueStorage& GgufValue::storage() const noexcept { return storage_; }

const GgufMetadataEntry* GgufFile::find_metadata(std::string_view key) const noexcept {
    const auto iterator = std::ranges::find_if(
        metadata, [key](const GgufMetadataEntry& entry) { return entry.key == key; });
    return iterator == metadata.end() ? nullptr : &*iterator;
}

const GgufTensorInfo* GgufFile::find_tensor(std::string_view name) const noexcept {
    const auto iterator = std::ranges::find_if(
        tensors, [name](const GgufTensorInfo& tensor) { return tensor.name == name; });
    return iterator == tensors.end() ? nullptr : &*iterator;
}

GgufFile GgufReader::read(const std::filesystem::path& path) {
    BinaryReader reader(path);
    std::array<char, 4> magic{};
    reader.read_exact(magic.data(), magic.size());
    if (magic != std::array<char, 4>{'G', 'G', 'U', 'F'}) {
        throw std::runtime_error("file is not GGUF: invalid magic");
    }

    GgufFile file;
    file.path = path;
    file.version = reader.read_u32();
    if (file.version != 2 && file.version != 3) {
        throw std::runtime_error("unsupported GGUF version: " + std::to_string(file.version));
    }
    file.tensor_count = reader.read_u64();
    file.metadata_count = reader.read_u64();
    if (file.tensor_count > max_entry_count || file.metadata_count > max_entry_count) {
        throw std::runtime_error("GGUF header count exceeds safety limit");
    }

    file.metadata.reserve(static_cast<std::size_t>(file.metadata_count));
    std::unordered_set<std::string> metadata_keys;
    for (std::uint64_t index = 0; index < file.metadata_count; ++index) {
        std::string key = reader.read_string(65'535);
        if (!metadata_keys.insert(key).second) {
            throw std::runtime_error("duplicate GGUF metadata key: " + key);
        }
        const GgufMetadataType type = parse_metadata_type(reader.read_u32());
        file.metadata.push_back(GgufMetadataEntry{std::move(key), read_value(reader, type, 0)});
    }

    if (const GgufMetadataEntry* alignment = file.find_metadata("general.alignment")) {
        if (const auto* value = alignment->value.get_if<std::uint32_t>()) {
            file.alignment = *value;
        } else {
            throw std::runtime_error("general.alignment must be uint32");
        }
    }
    if (file.alignment == 0 || !std::has_single_bit(file.alignment)) {
        throw std::runtime_error("GGUF alignment must be a non-zero power of two");
    }

    file.tensors.reserve(static_cast<std::size_t>(file.tensor_count));
    std::unordered_set<std::string> tensor_names;
    for (std::uint64_t index = 0; index < file.tensor_count; ++index) {
        GgufTensorInfo tensor;
        tensor.name = reader.read_string(64);
        if (!tensor_names.insert(tensor.name).second) {
            throw std::runtime_error("duplicate GGUF tensor name: " + tensor.name);
        }
        const std::uint32_t dimension_count = reader.read_u32();
        if (dimension_count == 0 || dimension_count > 64) {
            throw std::runtime_error("GGUF tensor dimension count is invalid");
        }
        tensor.dimensions.reserve(dimension_count);
        for (std::uint32_t dimension = 0; dimension < dimension_count; ++dimension) {
            const std::uint64_t size = reader.read_u64();
            if (size == 0) {
                throw std::runtime_error("GGUF tensor dimensions must be non-zero");
            }
            tensor.dimensions.push_back(size);
        }
        tensor.ggml_type = reader.read_u32();
        tensor.relative_offset = reader.read_u64();
        if (tensor.relative_offset % file.alignment != 0) {
            throw std::runtime_error("GGUF tensor offset is not aligned");
        }
        file.tensors.push_back(std::move(tensor));
    }

    file.data_offset = align_up(reader.position(), file.alignment);
    std::error_code error;
    file.file_size = std::filesystem::file_size(path, error);
    if (error) {
        throw std::runtime_error("unable to determine GGUF file size: " + error.message());
    }
    if (file.data_offset > file.file_size) {
        throw std::runtime_error("GGUF tensor data offset exceeds file size");
    }
    for (const GgufTensorInfo& tensor : file.tensors) {
        if (tensor.relative_offset > file.file_size - file.data_offset) {
            throw std::runtime_error("GGUF tensor offset exceeds file size: " + tensor.name);
        }
    }
    return file;
}

bool gguf_metadata_matches(std::string_view key, GgufMetadataQuery query) noexcept {
    if (!query.exact_key.empty()) {
        return key == query.exact_key;
    }
    if (!query.prefix.empty()) {
        return key.starts_with(query.prefix);
    }
    return true;
}

std::size_t gguf_metadata_match_count(const GgufFile& file,
                                      GgufMetadataQuery query) noexcept {
    return static_cast<std::size_t>(std::ranges::count_if(
        file.metadata,
        [query](const GgufMetadataEntry& entry) {
            return gguf_metadata_matches(entry.key, query);
        }));
}

std::string gguf_value_to_string(const GgufValue& value) { return value_json(value); }

std::string gguf_value_preview(const GgufValue& value,
                               std::size_t max_array_items,
                               std::size_t max_string_bytes) {
    return value_preview_impl(value, max_array_items, max_string_bytes);
}

std::string gguf_metadata_entries_json(const GgufFile& file,
                                       GgufMetadataJsonMode mode,
                                       GgufMetadataQuery query) {
    if (!query.exact_key.empty() && !query.prefix.empty()) {
        throw std::invalid_argument("GGUF metadata query cannot combine exact key and prefix");
    }

    std::ostringstream output;
    output << '[';
    bool first = true;
    for (const GgufMetadataEntry& entry : file.metadata) {
        if (!gguf_metadata_matches(entry.key, query)) {
            continue;
        }
        if (!first) {
            output << ',';
        }
        first = false;
        append_metadata_entry_json(output, entry, mode);
    }
    output << ']';
    return output.str();
}

std::string gguf_metadata_report_json(const GgufFile& file, GgufMetadataQuery query) {
    std::ostringstream output;
    output << "{\"path\":\"" << json_escape(file.path.string()) << "\",\"version\":"
           << file.version << ",\"metadata_count\":" << file.metadata_count
           << ",\"selected_metadata_count\":" << gguf_metadata_match_count(file, query)
           << ",\"metadata\":"
           << gguf_metadata_entries_json(file, GgufMetadataJsonMode::full, query) << '}';
    return output.str();
}

std::string gguf_summary_json(const GgufFile& file) {
    std::ostringstream output;
    output << "{\"path\":\"" << json_escape(file.path.string()) << "\",\"version\":"
           << file.version << ",\"tensor_count\":" << file.tensor_count
           << ",\"metadata_count\":" << file.metadata_count << ",\"alignment\":"
           << file.alignment << ",\"data_offset\":" << file.data_offset
           << ",\"file_size\":" << file.file_size << ",\"metadata\":"
           << gguf_metadata_entries_json(file, GgufMetadataJsonMode::compact)
           << ",\"tensors\":[";
    for (std::size_t index = 0; index < file.tensors.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        const GgufTensorInfo& tensor = file.tensors[index];
        output << "{\"name\":\"" << json_escape(tensor.name) << "\",\"dimensions\":[";
        for (std::size_t dimension = 0; dimension < tensor.dimensions.size(); ++dimension) {
            if (dimension != 0) {
                output << ',';
            }
            output << tensor.dimensions[dimension];
        }
        output << "],\"ggml_type\":" << tensor.ggml_type
               << ",\"relative_offset\":" << tensor.relative_offset << '}';
    }
    output << "]}";
    return output.str();
}

}  // namespace oracle::model
