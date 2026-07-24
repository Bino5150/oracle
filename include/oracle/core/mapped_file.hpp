#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

namespace oracle::core {

class MappedFile {
public:
    explicit MappedFile(std::filesystem::path path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    [[nodiscard]] const std::filesystem::path& path() const noexcept;
    [[nodiscard]] const std::byte* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

private:
    void reset() noexcept;

    std::filesystem::path path_;
    void* mapping_{nullptr};
    std::size_t size_{0};
    std::vector<std::byte> fallback_storage_;
};

}  // namespace oracle::core
