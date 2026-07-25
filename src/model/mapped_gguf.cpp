#include "oracle/model/mapped_gguf.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace oracle::model {
namespace {

struct TensorRange {
    std::uint64_t begin{0};
    std::uint64_t end{0};
    std::size_t tensor_index{0};
};

[[nodiscard]] std::uint64_t checked_add(std::uint64_t left,
                                        std::uint64_t right,
                                        std::string_view context) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(context) + " overflow");
    }
    return left + right;
}

[[nodiscard]] std::uint64_t checked_add_payload(std::uint64_t total,
                                                std::uint64_t value) {
    return checked_add(total, value, "GGUF validated payload size");
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

void append_tensor_json(std::ostringstream& output, const GgufTensorView& tensor) {
    output << "{\"name\":\"" << json_escape(tensor.name()) << "\",\"dimensions\":[";
    for (std::size_t index = 0; index < tensor.dimensions().size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << tensor.dimensions()[index];
    }
    output << "],\"ggml_type\":" << tensor.layout().type << ",\"type_name\":\""
           << tensor.layout().name << "\",\"quantized\":"
           << (tensor.layout().quantized ? "true" : "false")
           << ",\"block_elements\":" << tensor.layout().block_elements
           << ",\"bytes_per_block\":" << tensor.layout().bytes_per_block
           << ",\"element_count\":" << tensor.element_count()
           << ",\"relative_offset\":" << tensor.relative_offset()
           << ",\"absolute_offset\":" << tensor.absolute_offset()
           << ",\"byte_size\":" << tensor.bytes().size() << '}';
}

}  // namespace

GgufTensorView::GgufTensorView(const GgufTensorInfo* info,
                               const GgmlTypeLayout* layout,
                               const std::byte* data,
                               std::size_t byte_size,
                               std::uint64_t absolute_offset)
    : info_(info),
      layout_(layout),
      data_(data),
      byte_size_(byte_size),
      absolute_offset_(absolute_offset) {
    if (info_ == nullptr || layout_ == nullptr || (data_ == nullptr && byte_size_ != 0)) {
        throw std::invalid_argument("invalid GGUF tensor view");
    }
}

const GgufTensorInfo& GgufTensorView::info() const noexcept { return *info_; }
const GgmlTypeLayout& GgufTensorView::layout() const noexcept { return *layout_; }
std::string_view GgufTensorView::name() const noexcept { return info_->name; }
std::span<const std::uint64_t> GgufTensorView::dimensions() const noexcept {
    return info_->dimensions;
}
std::span<const std::byte> GgufTensorView::bytes() const noexcept {
    return {data_, byte_size_};
}
std::uint64_t GgufTensorView::absolute_offset() const noexcept { return absolute_offset_; }
std::uint64_t GgufTensorView::relative_offset() const noexcept {
    return info_->relative_offset;
}
std::uint64_t GgufTensorView::element_count() const {
    return ggml_tensor_element_count(info_->dimensions);
}

MappedGgufModel::MappedGgufModel(const std::filesystem::path& path)
    : file_(GgufReader::read(path)), mapping_(path) {
    if (mapping_.size() != file_.file_size) {
        throw std::runtime_error("GGUF file changed while it was being opened");
    }
    if (file_.file_size > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("GGUF file exceeds addressable memory");
    }

    tensors_.reserve(file_.tensors.size());
    tensor_index_.reserve(file_.tensors.size());
    std::vector<TensorRange> ranges;
    ranges.reserve(file_.tensors.size());

    std::uint64_t validated_payload_bytes = 0;
    std::uint64_t maximum_end = file_.data_offset;

    for (std::size_t index = 0; index < file_.tensors.size(); ++index) {
        const GgufTensorInfo& tensor = file_.tensors[index];
        const GgmlTypeLayout* layout = ggml_type_layout(tensor.ggml_type);
        if (layout == nullptr) {
            throw std::runtime_error("unsupported GGML tensor type " +
                                     std::to_string(tensor.ggml_type) + " for tensor " +
                                     tensor.name);
        }

        const std::uint64_t byte_size =
            ggml_tensor_byte_size(tensor.dimensions, tensor.ggml_type);
        const std::uint64_t absolute_offset =
            checked_add(file_.data_offset, tensor.relative_offset, "GGUF tensor offset");
        const std::uint64_t end_offset =
            checked_add(absolute_offset, byte_size, "GGUF tensor end offset");
        if (end_offset > file_.file_size) {
            throw std::runtime_error("GGUF tensor payload exceeds file size: " + tensor.name);
        }
        if (byte_size > std::numeric_limits<std::size_t>::max() ||
            absolute_offset > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("GGUF tensor view exceeds addressable memory: " +
                                     tensor.name);
        }

        const auto [iterator, inserted] = tensor_index_.emplace(tensor.name, index);
        static_cast<void>(iterator);
        if (!inserted) {
            throw std::runtime_error("duplicate GGUF tensor name: " + tensor.name);
        }

        const std::byte* tensor_data =
            mapping_.data() + static_cast<std::size_t>(absolute_offset);
        tensors_.emplace_back(&tensor,
                              layout,
                              tensor_data,
                              static_cast<std::size_t>(byte_size),
                              absolute_offset);
        ranges.push_back({absolute_offset, end_offset, index});
        validated_payload_bytes = checked_add_payload(validated_payload_bytes, byte_size);
        maximum_end = std::max(maximum_end, end_offset);
    }

    std::ranges::sort(ranges, [](const TensorRange& left, const TensorRange& right) {
        if (left.begin != right.begin) {
            return left.begin < right.begin;
        }
        return left.end < right.end;
    });
    for (std::size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index].begin < ranges[index - 1].end) {
            throw std::runtime_error("GGUF tensor payloads overlap: " +
                                     file_.tensors[ranges[index - 1].tensor_index].name +
                                     " and " + file_.tensors[ranges[index].tensor_index].name);
        }
    }

    stats_ = {file_.file_size,
              file_.data_offset,
              static_cast<std::uint64_t>(mapping_.size()),
              validated_payload_bytes,
              maximum_end - file_.data_offset,
              static_cast<std::uint64_t>(tensors_.size())};
}

const GgufFile& MappedGgufModel::file() const noexcept { return file_; }
const core::MappedFile& MappedGgufModel::mapping() const noexcept { return mapping_; }
const std::vector<GgufTensorView>& MappedGgufModel::tensors() const noexcept {
    return tensors_;
}

const GgufTensorView* MappedGgufModel::find_tensor(std::string_view name) const noexcept {
    const auto iterator = tensor_index_.find(std::string(name));
    return iterator == tensor_index_.end() ? nullptr : &tensors_[iterator->second];
}

const GgufTensorView& MappedGgufModel::tensor(std::string_view name) const {
    const GgufTensorView* view = find_tensor(name);
    if (view == nullptr) {
        throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    }
    return *view;
}

const GgufMappingStats& MappedGgufModel::stats() const noexcept { return stats_; }

std::string mapped_gguf_summary_json(const MappedGgufModel& model) {
    const GgufMappingStats& stats = model.stats();
    std::ostringstream output;
    output << "{\"path\":\"" << json_escape(model.file().path.string())
           << "\",\"verified\":true,\"version\":" << model.file().version
           << ",\"tensor_count\":" << stats.tensor_count
           << ",\"metadata_count\":" << model.file().metadata_count
           << ",\"alignment\":" << model.file().alignment
           << ",\"file_size\":" << stats.file_size
           << ",\"data_offset\":" << stats.data_offset
           << ",\"mapped_bytes\":" << stats.mapped_bytes
           << ",\"validated_payload_bytes\":" << stats.validated_payload_bytes
           << ",\"tensor_span_bytes\":" << stats.tensor_span_bytes
           << ",\"metadata\":"
           << gguf_metadata_entries_json(model.file(), GgufMetadataJsonMode::compact)
           << ",\"tensors\":[";
    for (std::size_t index = 0; index < model.tensors().size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        append_tensor_json(output, model.tensors()[index]);
    }
    output << "]}";
    return output.str();
}

std::string gguf_tensor_view_json(const GgufTensorView& tensor) {
    std::ostringstream output;
    append_tensor_json(output, tensor);
    return output.str();
}

}  // namespace oracle::model
