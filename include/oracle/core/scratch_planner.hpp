#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace oracle::core {

struct ScratchRequest {
    std::string name;
    std::size_t byte_count{0};
    std::size_t alignment{64};
    std::size_t first_step{0};
    std::size_t last_step{0};
};

struct ScratchAllocation {
    std::string name;
    std::size_t offset{0};
    std::size_t byte_count{0};
    std::size_t alignment{64};
    std::size_t first_step{0};
    std::size_t last_step{0};
};

class ScratchPlan {
public:
    ScratchPlan() = default;
    ScratchPlan(std::vector<ScratchAllocation> allocations, std::size_t peak_bytes);

    [[nodiscard]] const std::vector<ScratchAllocation>& allocations() const noexcept;
    [[nodiscard]] std::size_t peak_bytes() const noexcept;
    [[nodiscard]] const ScratchAllocation* find(std::string_view name) const noexcept;

private:
    std::vector<ScratchAllocation> allocations_;
    std::size_t peak_bytes_{0};
};

class ScratchPlanner {
public:
    void add(ScratchRequest request);
    void clear() noexcept;
    [[nodiscard]] ScratchPlan build() const;
    [[nodiscard]] const std::vector<ScratchRequest>& requests() const noexcept;

private:
    std::vector<ScratchRequest> requests_;
};

}  // namespace oracle::core
