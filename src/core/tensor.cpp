#include "oracle/core/tensor.hpp"

#include <algorithm>
#include <bit>
#include <functional>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace oracle::core {
namespace {

[[nodiscard]] std::size_t checked_element_count(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("tensor shape cannot be empty");
    }

    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument("tensor dimensions must be non-zero");
        }
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor element count overflow");
        }
        count *= dimension;
    }
    return count;
}

[[nodiscard]] std::vector<std::size_t> contiguous_strides(
    const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index > 1; --index) {
        const std::size_t current = index - 2;
        const std::size_t next = index - 1;
        if (strides[next] > std::numeric_limits<std::size_t>::max() / shape[next]) {
            throw std::overflow_error("tensor stride overflow");
        }
        strides[current] = strides[next] * shape[next];
    }
    return strides;
}

[[nodiscard]] bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && std::has_single_bit(value);
}

}  // namespace

struct Tensor::Storage {
    explicit Storage(std::size_t bytes, std::size_t requested_alignment)
        : byte_count(bytes), alignment(requested_alignment) {
        pointer = ::operator new(byte_count, std::align_val_t{alignment});
    }

    ~Storage() { ::operator delete(pointer, std::align_val_t{alignment}); }

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    void* pointer{nullptr};
    std::size_t byte_count{0};
    std::size_t alignment{Tensor::default_alignment};
};

Tensor::Tensor(std::vector<std::size_t> shape, DataType type, std::size_t alignment)
    : shape_(std::move(shape)), type_(type), element_count_(checked_element_count(shape_)) {
    if (!is_power_of_two(alignment) || alignment < alignof(float)) {
        throw std::invalid_argument("tensor alignment must be a power of two and at least alignof(float)");
    }

    strides_ = contiguous_strides(shape_);
    const std::size_t element_size = data_type_size(type_);
    if (element_size == 0 || element_count_ > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("tensor byte size overflow");
    }
    storage_ = std::make_shared<Storage>(element_count_ * element_size, alignment);
}

Tensor::Tensor(std::vector<std::size_t> shape,
               std::vector<std::size_t> strides,
               DataType type,
               std::shared_ptr<Storage> storage,
               std::size_t byte_offset,
               bool is_view)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      type_(type),
      storage_(std::move(storage)),
      byte_offset_(byte_offset),
      element_count_(checked_element_count(shape_)),
      is_view_(is_view) {
    validate_layout();
}

const std::vector<std::size_t>& Tensor::shape() const noexcept { return shape_; }
const std::vector<std::size_t>& Tensor::strides() const noexcept { return strides_; }
std::size_t Tensor::rank() const noexcept { return shape_.size(); }
std::size_t Tensor::element_count() const noexcept { return element_count_; }
std::size_t Tensor::byte_size() const noexcept { return element_count_ * data_type_size(type_); }
std::size_t Tensor::alignment() const noexcept { return storage_->alignment; }
DataType Tensor::dtype() const noexcept { return type_; }
bool Tensor::is_view() const noexcept { return is_view_; }

bool Tensor::is_contiguous() const noexcept {
    if (shape_.empty() || strides_.size() != shape_.size()) {
        return false;
    }
    std::size_t expected = 1;
    for (std::size_t index = shape_.size(); index > 0; --index) {
        const std::size_t current = index - 1;
        if (strides_[current] != expected) {
            return false;
        }
        expected *= shape_[current];
    }
    return true;
}

std::span<float> Tensor::data() {
    if (type_ != DataType::f32) {
        throw std::logic_error("Tensor::data currently supports only f32 tensors");
    }
    if (!is_contiguous()) {
        throw std::logic_error("Tensor::data requires contiguous storage");
    }
    auto* bytes = static_cast<std::byte*>(storage_->pointer) + byte_offset_;
    return {reinterpret_cast<float*>(bytes), element_count_};
}

std::span<const float> Tensor::data() const {
    if (type_ != DataType::f32) {
        throw std::logic_error("Tensor::data currently supports only f32 tensors");
    }
    if (!is_contiguous()) {
        throw std::logic_error("Tensor::data requires contiguous storage");
    }
    const auto* bytes = static_cast<const std::byte*>(storage_->pointer) + byte_offset_;
    return {reinterpret_cast<const float*>(bytes), element_count_};
}

std::size_t Tensor::offset_of(std::span<const std::size_t> indices) const {
    if (indices.size() != rank()) {
        throw std::out_of_range("tensor index rank does not match tensor rank");
    }

    std::size_t offset = 0;
    for (std::size_t dimension = 0; dimension < rank(); ++dimension) {
        if (indices[dimension] >= shape_[dimension]) {
            throw std::out_of_range("tensor index is out of bounds");
        }
        offset += indices[dimension] * strides_[dimension];
    }
    return offset;
}

float& Tensor::at(std::span<const std::size_t> indices) {
    if (type_ != DataType::f32) {
        throw std::logic_error("Tensor::at currently supports only f32 tensors");
    }
    auto* bytes = static_cast<std::byte*>(storage_->pointer) + byte_offset_;
    return reinterpret_cast<float*>(bytes)[offset_of(indices)];
}

const float& Tensor::at(std::span<const std::size_t> indices) const {
    if (type_ != DataType::f32) {
        throw std::logic_error("Tensor::at currently supports only f32 tensors");
    }
    const auto* bytes = static_cast<const std::byte*>(storage_->pointer) + byte_offset_;
    return reinterpret_cast<const float*>(bytes)[offset_of(indices)];
}

void Tensor::fill(float value) {
    if (is_contiguous()) {
        std::ranges::fill(data(), value);
        return;
    }

    std::vector<std::size_t> indices(rank(), 0);
    for (std::size_t linear = 0; linear < element_count_; ++linear) {
        std::size_t remainder = linear;
        for (std::size_t dimension = rank(); dimension > 0; --dimension) {
            const std::size_t current = dimension - 1;
            indices[current] = remainder % shape_[current];
            remainder /= shape_[current];
        }
        at(indices) = value;
    }
}

Tensor Tensor::view(std::vector<std::size_t> shape, std::size_t element_offset) const {
    if (!is_contiguous()) {
        throw std::logic_error("contiguous Tensor::view requires a contiguous source tensor");
    }
    const std::size_t count = checked_element_count(shape);
    if (element_offset > element_count_ || count > element_count_ - element_offset) {
        throw std::out_of_range("tensor view exceeds source tensor bounds");
    }
    const std::size_t offset_bytes = element_offset * data_type_size(type_);
    auto view_strides = contiguous_strides(shape);
    return Tensor(std::move(shape),
                  std::move(view_strides),
                  type_,
                  storage_,
                  byte_offset_ + offset_bytes,
                  true);
}

Tensor Tensor::view(std::vector<std::size_t> shape,
                    std::vector<std::size_t> strides,
                    std::size_t element_offset) const {
    if (element_offset > maximum_element_offset()) {
        throw std::out_of_range("strided tensor view offset exceeds source bounds");
    }
    const std::size_t offset_bytes = element_offset * data_type_size(type_);
    Tensor result(std::move(shape),
                  std::move(strides),
                  type_,
                  storage_,
                  byte_offset_ + offset_bytes,
                  true);
    if (result.maximum_element_offset() > maximum_element_offset() - element_offset) {
        throw std::out_of_range("strided tensor view exceeds source tensor bounds");
    }
    return result;
}

Tensor Tensor::reshape(std::vector<std::size_t> shape) const {
    if (!is_contiguous()) {
        throw std::logic_error("Tensor::reshape requires a contiguous tensor");
    }
    if (checked_element_count(shape) != element_count_) {
        throw std::invalid_argument("reshape must preserve tensor element count");
    }
    return view(std::move(shape));
}

std::size_t Tensor::maximum_element_offset() const noexcept {
    std::size_t maximum = 0;
    for (std::size_t dimension = 0; dimension < rank(); ++dimension) {
        maximum += (shape_[dimension] - 1) * strides_[dimension];
    }
    return maximum;
}

void Tensor::validate_layout() const {
    if (!storage_) {
        throw std::invalid_argument("tensor storage cannot be null");
    }
    if (strides_.size() != shape_.size()) {
        throw std::invalid_argument("tensor stride rank must match shape rank");
    }
    if (byte_offset_ % data_type_size(type_) != 0) {
        throw std::invalid_argument("tensor byte offset is not element aligned");
    }
    if (byte_offset_ >= storage_->byte_count) {
        throw std::out_of_range("tensor byte offset exceeds storage");
    }
    const std::size_t final_byte = byte_offset_ + maximum_element_offset() * data_type_size(type_);
    if (final_byte + data_type_size(type_) > storage_->byte_count) {
        throw std::out_of_range("tensor layout exceeds storage bounds");
    }
}

}  // namespace oracle::core
