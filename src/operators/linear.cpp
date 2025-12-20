#include "mini_infer/operators/linear.h"

#include <algorithm>
#include <cstring>

#include "mini_infer/core/op_type.h"
#include "mini_infer/kernels/bias.h"
#include "mini_infer/kernels/gemm.h"

namespace mini_infer {
namespace operators {

Linear::Linear(const LinearParam& param)
    : Operator(core::op_names::kLinear, core::OpType::kGEMM), param_(param) {}

core::Status Linear::forward(const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                             std::vector<std::shared_ptr<core::Tensor>>& outputs) {
    // Input validation
    size_t expected_inputs = param_.use_bias ? 3 : 2;
    if (inputs.size() != expected_inputs) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input = inputs[0];   // [batch_size, in_features]
    const auto& weight = inputs[1];  // [out_features, in_features]

    if (!input || !weight) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Validate bias if needed
    if (param_.use_bias) {
        const auto& bias = inputs[2];
        if (!bias) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    // Get input shapes
    const auto& input_shape = input->shape();
    const auto& weight_shape = weight->shape();

    // Validate shapes
    if (input_shape.ndim() < 2 || weight_shape.ndim() != 2) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Calculate dimensions
    int batch_size = 1;
    for (size_t i = 0; i < input_shape.ndim() - 1; ++i) {
        batch_size *= static_cast<int>(input_shape[i]);
    }
    int in_features = static_cast<int>(input_shape[input_shape.ndim() - 1]);
    int weight_in_features = static_cast<int>(weight_shape[1]);

    // Validate feature dimensions match
    if (in_features != weight_in_features) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Get pre-allocated output tensor (Engine already did shape inference)
    if (outputs.empty() || !outputs[0]) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    auto output = outputs[0];

    // Extract output dimensions from pre-allocated tensor
    const auto& output_shape = output->shape();
    if (output_shape.ndim() < 2) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    int out_features = static_cast<int>(output_shape[output_shape.ndim() - 1]);

    // Update parameters if not set
    if (param_.in_features == 0) {
        param_.in_features = in_features;
    }
    if (param_.out_features == 0) {
        param_.out_features = out_features;
    }

    // Perform computation based on data type
    const auto dtype = input->dtype();

    if (dtype == core::DataType::FLOAT32) {
        const float* input_data = static_cast<const float*>(input->data());
        const float* weight_data = static_cast<const float*>(weight->data());
        float* output_data = static_cast<float*>(output->data());

        // Matrix multiplication: output = input @ weight^T
        kernels::GEMMKernel::gemm_nt<float>(input_data,   // [batch_size, in_features]
                                            weight_data,  // [out_features, in_features]
                                            output_data,  // [batch_size, out_features]
                                            batch_size, out_features, in_features);

        // Add bias if needed
        if (param_.use_bias) {
            const float* bias_data = static_cast<const float*>(inputs[2]->data());
            kernels::BiasKernel::add_channel_bias<float>(output_data, bias_data,
                                                         batch_size,    // batch_size
                                                         out_features,  // channels
                                                         1  // spatial_size (1 for Linear)
            );
        }

    } else if (dtype == core::DataType::INT32) {
        const int32_t* input_data = static_cast<const int32_t*>(input->data());
        const int32_t* weight_data = static_cast<const int32_t*>(weight->data());
        int32_t* output_data = static_cast<int32_t*>(output->data());

        // Matrix multiplication: output = input @ weight^T
        kernels::GEMMKernel::gemm_nt<int32_t>(input_data,   // [batch_size, in_features]
                                              weight_data,  // [out_features, in_features]
                                              output_data,  // [batch_size, out_features]
                                              batch_size, out_features, in_features);

        // Add bias if needed
        if (param_.use_bias) {
            const int32_t* bias_data = static_cast<const int32_t*>(inputs[2]->data());
            kernels::BiasKernel::add_channel_bias<int32_t>(output_data, bias_data,
                                                           batch_size,    // batch_size
                                                           out_features,  // channels
                                                           1  // spatial_size (1 for Linear)
            );
        }

    } else {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    return core::Status::SUCCESS;
}

core::Status Linear::infer_shape(const std::vector<core::Shape>& input_shapes,
                                 std::vector<core::Shape>& output_shapes) {
    // Validate input count
    size_t expected_inputs = param_.use_bias ? 3 : 2;
    if (input_shapes.size() != expected_inputs) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    const auto& input_shape = input_shapes[0];   // [..., in_features]
    const auto& weight_shape = input_shapes[1];  // [out_features, in_features]

    // Validate shapes
    if (input_shape.ndim() < 2 || weight_shape.ndim() != 2) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Get dimensions
    int in_features = static_cast<int>(input_shape[input_shape.ndim() - 1]);
    int out_features = static_cast<int>(weight_shape[0]);
    int weight_in_features = static_cast<int>(weight_shape[1]);

    // Validate feature dimensions
    if (param_.in_features > 0 && in_features != param_.in_features) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    if (param_.out_features > 0 && out_features != param_.out_features) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    if (in_features != weight_in_features) {
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Validate bias shape if needed
    if (param_.use_bias && input_shapes.size() > 2) {
        const auto& bias_shape = input_shapes[2];
        if (bias_shape.ndim() != 1 || bias_shape[0] != out_features) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
    }

    // Output shape: replace last dimension with out_features
    std::vector<int64_t> output_dims;
    for (size_t i = 0; i < input_shape.ndim() - 1; ++i) {
        output_dims.push_back(input_shape[i]);
    }
    output_dims.push_back(static_cast<int64_t>(out_features));

    output_shapes.clear();
    output_shapes.push_back(core::Shape(output_dims));

    return core::Status::SUCCESS;
}

// Register Linear operator
REGISTER_OPERATOR(Linear, Linear);

}  // namespace operators
}  // namespace mini_infer
