#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace oracle::core {

enum class DataType { f32 };

[[nodiscard]] constexpr std::size_t data_type_size(DataType type) noexcept {
    switch (type) {
        case DataType::f32:
            return sizeof(float);
    }
    return 0;
}

[[nodiscard]] constexpr std::string_view data_type_name(DataType type) noexcept {
    switch (type) {
        case DataType::f32:
            return "f32";
    }
    return "unknown";
}

class Tensor {
public:
    static constexpr std::size_t default_alignment = 64;

    explicit Tensor(std::vector<std::size_t> shape,
                    DataType type = DataType::f32,
                    std::size_t alignment = default_alignment);

    Tensor(const Tensor&) noexcept = default;
    Tensor& operator=(const Tensor&) noexcept = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t element_count() const noexcept;
    [[nodiscard]] std::size_t byte_size() const noexcept;
    [[nodiscard]] std::size_t alignment() const noexcept;
    [[nodiscard]] DataType dtype() const noexcept;
    [[nodiscard]] bool is_contiguous() const noexcept;
    [[nodiscard]] bool is_view() const noexcept;

    [[nodiscard]] std::span<float> data();
    [[nodiscard]] std::span<const float> data() const;

    [[nodiscard]] float& at(std::span<const std::size_t> indices);
    [[nodiscard]] const float& at(std::span<const std::size_t> indices) const;

    void fill(float value);

    [[nodiscard]] Tensor view(std::vector<std::size_t> shape,
                              std::size_t element_offset = 0) const;
    [[nodiscard]] Tensor view(std::vector<std::size_t> shape,
                              std::vector<std::size_t> strides,
                              std::size_t element_offset = 0) const;
    [[nodiscard]] Tensor reshape(std::vector<std::size_t> shape) const;

private:
    struct Storage;

    Tensor(std::vector<std::size_t> shape,
           std::vector<std::size_t> strides,
           DataType type,
           std::shared_ptr<Storage> storage,
           std::size_t byte_offset,
           bool is_view);

    [[nodiscard]] std::size_t offset_of(std::span<const std::size_t> indices) const;
    [[nodiscard]] std::size_t maximum_element_offset() const noexcept;
    void validate_layout() const;

    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    DataType type_{DataType::f32};
    std::shared_ptr<Storage> storage_;
    std::size_t byte_offset_{0};
    std::size_t element_count_{0};
    bool is_view_{false};
};

}  // namespace oracle::core
