#include "oracle/model/model.hpp"

#include <stdexcept>
#include <utility>

namespace oracle::model {

Model::Model(std::string name) : name_(std::move(name)) {
    if (name_.empty()) {
        throw std::invalid_argument("model name cannot be empty");
    }
}

const std::string& Model::name() const noexcept { return name_; }

}  // namespace oracle::model
