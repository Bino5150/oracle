#include "oracle/core/scratch_planner.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace oracle::core {
namespace {

[[nodiscard]] std::size_t align_up(std::size_t value, std::size_t alignment) {
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error("scratch offset alignment overflow");
    }
    return (value + mask) & ~mask;
}

[[nodiscard]] bool lifetimes_overlap(const ScratchAllocation& lhs,
                                     const ScratchRequest& rhs) noexcept {
    return !(lhs.last_step < rhs.first_step || rhs.last_step < lhs.first_step);
}

}  // namespace

ScratchPlan::ScratchPlan(std::vector<ScratchAllocation> allocations, std::size_t peak_bytes)
    : allocations_(std::move(allocations)), peak_bytes_(peak_bytes) {}

const std::vector<ScratchAllocation>& ScratchPlan::allocations() const noexcept {
    return allocations_;
}

std::size_t ScratchPlan::peak_bytes() const noexcept { return peak_bytes_; }

const ScratchAllocation* ScratchPlan::find(std::string_view name) const noexcept {
    const auto iterator = std::ranges::find_if(
        allocations_, [name](const ScratchAllocation& allocation) {
            return allocation.name == name;
        });
    return iterator == allocations_.end() ? nullptr : &*iterator;
}

void ScratchPlanner::add(ScratchRequest request) {
    if (request.name.empty()) {
        throw std::invalid_argument("scratch request name cannot be empty");
    }
    if (request.byte_count == 0) {
        throw std::invalid_argument("scratch request byte count must be non-zero");
    }
    if (!std::has_single_bit(request.alignment)) {
        throw std::invalid_argument("scratch request alignment must be a power of two");
    }
    if (request.first_step > request.last_step) {
        throw std::invalid_argument("scratch request lifetime is reversed");
    }
    if (std::ranges::any_of(requests_, [&request](const ScratchRequest& existing) {
            return existing.name == request.name;
        })) {
        throw std::invalid_argument("scratch request names must be unique");
    }
    requests_.push_back(std::move(request));
}

void ScratchPlanner::clear() noexcept { requests_.clear(); }

ScratchPlan ScratchPlanner::build() const {
    std::vector<ScratchRequest> ordered = requests_;
    std::ranges::sort(ordered, [](const ScratchRequest& lhs, const ScratchRequest& rhs) {
        if (lhs.first_step != rhs.first_step) {
            return lhs.first_step < rhs.first_step;
        }
        if (lhs.alignment != rhs.alignment) {
            return lhs.alignment > rhs.alignment;
        }
        if (lhs.byte_count != rhs.byte_count) {
            return lhs.byte_count > rhs.byte_count;
        }
        if (lhs.last_step != rhs.last_step) {
            return lhs.last_step < rhs.last_step;
        }
        return lhs.name < rhs.name;
    });

    std::vector<ScratchAllocation> allocations;
    allocations.reserve(ordered.size());
    std::size_t peak = 0;

    for (const ScratchRequest& request : ordered) {
        std::vector<const ScratchAllocation*> conflicts;
        for (const ScratchAllocation& allocation : allocations) {
            if (lifetimes_overlap(allocation, request)) {
                conflicts.push_back(&allocation);
            }
        }
        std::ranges::sort(conflicts, [](const auto* lhs, const auto* rhs) {
            return lhs->offset < rhs->offset;
        });

        std::size_t candidate = 0;
        for (const ScratchAllocation* conflict : conflicts) {
            candidate = align_up(candidate, request.alignment);
            if (candidate <= conflict->offset &&
                request.byte_count <= conflict->offset - candidate) {
                break;
            }
            if (conflict->offset > std::numeric_limits<std::size_t>::max() -
                                       conflict->byte_count) {
                throw std::overflow_error("scratch allocation range overflow");
            }
            candidate = std::max(candidate, conflict->offset + conflict->byte_count);
        }
        candidate = align_up(candidate, request.alignment);
        if (candidate > std::numeric_limits<std::size_t>::max() - request.byte_count) {
            throw std::overflow_error("scratch plan size overflow");
        }

        allocations.push_back(ScratchAllocation{request.name,
                                                 candidate,
                                                 request.byte_count,
                                                 request.alignment,
                                                 request.first_step,
                                                 request.last_step});
        peak = std::max(peak, candidate + request.byte_count);
    }

    std::ranges::sort(allocations, [](const ScratchAllocation& lhs,
                                     const ScratchAllocation& rhs) {
        return lhs.name < rhs.name;
    });
    return ScratchPlan(std::move(allocations), peak);
}

const std::vector<ScratchRequest>& ScratchPlanner::requests() const noexcept {
    return requests_;
}

}  // namespace oracle::core
