#pragma once

#include <memory>
#include <vector>

#include "mini_infer/operators/operator.h"

namespace mini_infer {
namespace operators {

struct SoftmaxParam {
    int axis{-1};
};

class Softmax : public Operator {
public:
    explicit Softmax(const SoftmaxParam& param = SoftmaxParam());
    ~Softmax() override = default;

    core::Status forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                         std::vector<std::shared_ptr<core::Tensor>>& outputs) override;

    core::Status infer_shape(const std::vector<core::Shape>& input_shapes,
                             std::vector<core::Shape>& output_shapes) override;

private:
    SoftmaxParam param_;
};

}  // namespace operators
}  // namespace mini_infer
