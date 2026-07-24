#include "oracle/core/memory_arena.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace oracle::core {
namespace {

[[nodiscard]] bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && std::has_single_bit(value);
}

[[nodiscard]] std::size_t align_up(std::size_t value, std::size_t alignment) {
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error("arena alignment overflow");
    }
    return (value + mask) & ~mask;
}

}  // namespace

void MemoryArena::AlignedDelete::operator()(std::byte* pointer) const noexcept {
    if (pointer != nullptr) {
        ::operator delete(pointer, std::align_val_t{alignment});
    }
}

MemoryArena::MemoryArena(std::size_t capacity_bytes, std::size_t alignment)
    : storage_(nullptr, AlignedDelete{alignment}),
      capacity_bytes_(capacity_bytes),
      alignment_(alignment) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("arena capacity must be non-zero");
    }
    if (!is_power_of_two(alignment) || alignment < alignof(std::max_align_t)) {
        throw std::invalid_argument(
            "arena alignment must be a power of two and at least alignof(max_align_t)");
    }
    storage_.reset(static_cast<std::byte*>(
        ::operator new(capacity_bytes_, std::align_val_t{alignment_})));
}

void* MemoryArena::allocate_bytes(std::size_t byte_count, std::size_t alignment) {
    if (byte_count == 0) {
        return nullptr;
    }
    if (!is_power_of_two(alignment) || alignment > alignment_) {
        throw std::invalid_argument(
            "allocation alignment must be a power of two no greater than arena alignment");
    }

    const std::size_t offset = align_up(cursor_, alignment);
    if (offset > capacity_bytes_ || byte_count > capacity_bytes_ - offset) {
        throw std::bad_alloc{};
    }

    cursor_ = offset + byte_count;
    peak_bytes_ = std::max(peak_bytes_, cursor_);
    ++allocation_count_;
    return storage_.get() + offset;
}

std::size_t MemoryArena::mark() const noexcept { return cursor_; }

void MemoryArena::rewind(std::size_t mark_value) {
    if (mark_value > cursor_) {
        throw std::out_of_range("arena rewind mark exceeds current cursor");
    }
    cursor_ = mark_value;
}

void MemoryArena::reset() noexcept { cursor_ = 0; }

void MemoryArena::clear_peak() noexcept {
    peak_bytes_ = cursor_;
    allocation_count_ = 0;
}

ArenaStats MemoryArena::stats() const noexcept {
    return ArenaStats{capacity_bytes_, cursor_, peak_bytes_, allocation_count_};
}

std::size_t MemoryArena::capacity() const noexcept { return capacity_bytes_; }
std::size_t MemoryArena::used() const noexcept { return cursor_; }
std::size_t MemoryArena::remaining() const noexcept { return capacity_bytes_ - cursor_; }
std::size_t MemoryArena::alignment() const noexcept { return alignment_; }
std::byte* MemoryArena::data() noexcept { return storage_.get(); }
const std::byte* MemoryArena::data() const noexcept { return storage_.get(); }

ScopedArenaMark::ScopedArenaMark(MemoryArena& arena) noexcept
    : arena_(&arena), mark_(arena.mark()) {}

ScopedArenaMark::ScopedArenaMark(ScopedArenaMark&& other) noexcept
    : arena_(std::exchange(other.arena_, nullptr)), mark_(other.mark_) {}

ScopedArenaMark::~ScopedArenaMark() {
    if (arena_ != nullptr) {
        arena_->rewind(mark_);
    }
}

void ScopedArenaMark::release() noexcept { arena_ = nullptr; }

}  // namespace oracle::core
