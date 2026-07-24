#include "oracle/scheduler/scheduler.hpp"

#include <stdexcept>

namespace oracle::scheduler {

Scheduler::Scheduler(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
        throw std::invalid_argument("scheduler capacity must be non-zero");
    }
}

std::size_t Scheduler::capacity() const noexcept { return capacity_; }

}  // namespace oracle::scheduler
