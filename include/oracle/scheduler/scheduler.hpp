#pragma once

#include <cstddef>

namespace oracle::scheduler {

class Scheduler {
public:
    explicit Scheduler(std::size_t capacity);
    [[nodiscard]] std::size_t capacity() const noexcept;

private:
    std::size_t capacity_;
};

}  // namespace oracle::scheduler
