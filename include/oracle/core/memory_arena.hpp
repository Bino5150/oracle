#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace oracle::core {

struct ArenaStats {
    std::size_t capacity_bytes{0};
    std::size_t used_bytes{0};
    std::size_t peak_bytes{0};
    std::size_t allocation_count{0};
};

class MemoryArena {
public:
    static constexpr std::size_t default_alignment = 64;

    explicit MemoryArena(std::size_t capacity_bytes,
                         std::size_t alignment = default_alignment);

    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&) noexcept = default;
    MemoryArena& operator=(MemoryArena&&) noexcept = default;
    ~MemoryArena() = default;

    [[nodiscard]] void* allocate_bytes(
        std::size_t byte_count,
        std::size_t alignment = default_alignment);

    template <typename T>
    [[nodiscard]] T* allocate(std::size_t count = 1) {
        if (count == 0) {
            return nullptr;
        }
        if (count > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::overflow_error("arena typed allocation overflow");
        }
        return static_cast<T*>(allocate_bytes(count * sizeof(T), alignof(T)));
    }

    [[nodiscard]] std::size_t mark() const noexcept;
    void rewind(std::size_t mark);
    void reset() noexcept;
    void clear_peak() noexcept;

    [[nodiscard]] ArenaStats stats() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t used() const noexcept;
    [[nodiscard]] std::size_t remaining() const noexcept;
    [[nodiscard]] std::size_t alignment() const noexcept;
    [[nodiscard]] std::byte* data() noexcept;
    [[nodiscard]] const std::byte* data() const noexcept;

private:
    struct AlignedDelete {
        std::size_t alignment{default_alignment};
        void operator()(std::byte* pointer) const noexcept;
    };

    std::unique_ptr<std::byte, AlignedDelete> storage_;
    std::size_t capacity_bytes_{0};
    std::size_t alignment_{default_alignment};
    std::size_t cursor_{0};
    std::size_t peak_bytes_{0};
    std::size_t allocation_count_{0};
};

class ScopedArenaMark {
public:
    explicit ScopedArenaMark(MemoryArena& arena) noexcept;
    ScopedArenaMark(const ScopedArenaMark&) = delete;
    ScopedArenaMark& operator=(const ScopedArenaMark&) = delete;
    ScopedArenaMark(ScopedArenaMark&& other) noexcept;
    ScopedArenaMark& operator=(ScopedArenaMark&&) = delete;
    ~ScopedArenaMark();

    void release() noexcept;

private:
    MemoryArena* arena_{nullptr};
    std::size_t mark_{0};
};

}  // namespace oracle::core
