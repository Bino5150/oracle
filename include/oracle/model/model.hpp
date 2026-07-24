#pragma once

#include <string>

namespace oracle::model {

class Model {
public:
    explicit Model(std::string name);
    [[nodiscard]] const std::string& name() const noexcept;

private:
    std::string name_;
};

}  // namespace oracle::model
