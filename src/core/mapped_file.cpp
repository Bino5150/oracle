#include "oracle/core/mapped_file.hpp"

#include <cerrno>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace oracle::core {
namespace {

[[nodiscard]] std::runtime_error mapping_error(const std::filesystem::path& path,
                                               std::string_view operation,
                                               int error_number) {
    return std::runtime_error(std::string(operation) + " failed for " + path.string() +
                              ": " + std::error_code(error_number, std::generic_category()).message());
}

}  // namespace

MappedFile::MappedFile(std::filesystem::path path) : path_(std::move(path)) {
#if defined(_WIN32)
    std::ifstream input(path_, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("unable to open file for mapping: " + path_.string());
    }
    const std::streamoff end = input.tellg();
    if (end < 0 || static_cast<std::uintmax_t>(end) >
                       static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("mapped file size exceeds addressable memory: " + path_.string());
    }
    fallback_storage_.resize(static_cast<std::size_t>(end));
    input.seekg(0);
    if (!fallback_storage_.empty()) {
        input.read(reinterpret_cast<char*>(fallback_storage_.data()),
                   static_cast<std::streamsize>(fallback_storage_.size()));
        if (!input) {
            throw std::runtime_error("unable to read mapped-file fallback: " + path_.string());
        }
    }
    size_ = fallback_storage_.size();
#else
    int open_flags = O_RDONLY;
#ifdef O_CLOEXEC
    open_flags |= O_CLOEXEC;
#endif
    const int descriptor = ::open(path_.c_str(), open_flags);
    if (descriptor < 0) {
        throw mapping_error(path_, "open", errno);
    }

    struct stat status {};
    if (::fstat(descriptor, &status) != 0) {
        const int error_number = errno;
        ::close(descriptor);
        throw mapping_error(path_, "fstat", error_number);
    }
    if (status.st_size < 0 || static_cast<std::uintmax_t>(status.st_size) >
                                  static_cast<std::uintmax_t>(
                                      std::numeric_limits<std::size_t>::max())) {
        ::close(descriptor);
        throw std::runtime_error("mapped file size exceeds addressable memory: " + path_.string());
    }

    size_ = static_cast<std::size_t>(status.st_size);
    if (size_ != 0) {
        void* address = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
        if (address == MAP_FAILED) {
            const int error_number = errno;
            ::close(descriptor);
            size_ = 0;
            throw mapping_error(path_, "mmap", error_number);
        }
        mapping_ = address;
    }
    ::close(descriptor);
#endif
}

MappedFile::~MappedFile() { reset(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : path_(std::move(other.path_)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      fallback_storage_(std::move(other.fallback_storage_)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    reset();
    path_ = std::move(other.path_);
    mapping_ = std::exchange(other.mapping_, nullptr);
    size_ = std::exchange(other.size_, 0);
    fallback_storage_ = std::move(other.fallback_storage_);
    return *this;
}

const std::filesystem::path& MappedFile::path() const noexcept { return path_; }

const std::byte* MappedFile::data() const noexcept {
    if (mapping_ != nullptr) {
        return static_cast<const std::byte*>(mapping_);
    }
    return fallback_storage_.empty() ? nullptr : fallback_storage_.data();
}

std::size_t MappedFile::size() const noexcept { return size_; }
bool MappedFile::empty() const noexcept { return size_ == 0; }
std::span<const std::byte> MappedFile::bytes() const noexcept { return {data(), size_}; }

void MappedFile::reset() noexcept {
#if !defined(_WIN32)
    if (mapping_ != nullptr) {
        ::munmap(mapping_, size_);
    }
#endif
    mapping_ = nullptr;
    size_ = 0;
    fallback_storage_.clear();
}

}  // namespace oracle::core
