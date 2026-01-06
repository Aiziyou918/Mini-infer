#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/kernels/gemm.h"
#include "mini_infer/kernels/bias.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Linear (Fully Connected) CPU Plugin
 *
 * Performs: output = input @ weight^T + bias
 */
class LinearCPUPlugin : public CPUPlugin<LinearCPUPlugin, LinearParam> {
public:
    LinearCPUPlugin() {
        param_ = std::make_shared<LinearParam>();
    }
    ~LinearCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Gemm";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kGEMM;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return param_ && param_->use_bias ? 3 : 2;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.size() < 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        const auto& weight_shape = input_shapes[1];

        if (input_shape.ndim() < 1 || weight_shape.ndim() != 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        std::vector<int64_t> output_dims;
        for (size_t i = 0; i + 1 < input_shape.ndim(); ++i) {
            output_dims.push_back(input_shape[i]);
        }
        output_dims.push_back(weight_shape[0]);  // out_features

        output_shapes.clear();
        output_shapes.push_back(core::Shape(output_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        const size_t expected_inputs = (param_ && param_->use_bias) ? 3 : 2;
        if (inputs.size() != expected_inputs || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input = inputs[0];
        const auto& weight = inputs[1];
        auto& output = outputs[0];

        if (!input || !weight || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& input_shape = input->shape();
        const auto& weight_shape = weight->shape();
        const auto& output_shape = output->shape();

        int batch_size = 1;
        for (size_t i = 0; i + 1 < input_shape.ndim(); ++i) {
            batch_size *= static_cast<int>(input_shape[i]);
        }

        const int in_features = static_cast<int>(input_shape[input_shape.ndim() - 1]);
        const int out_features = static_cast<int>(output_shape[output_shape.ndim() - 1]);

        const float* input_data = static_cast<const float*>(input->data());
        const float* weight_data = static_cast<const float*>(weight->data());
        float* output_data = static_cast<float*>(output->data());

        // GEMM: output = input @ weight^T
        kernels::GEMMKernel::gemm_nt<float>(
            input_data, weight_data, output_data,
            batch_size, out_features, in_features,
            kernels::KernelBackend::CPU
        );

        // Add bias if present
        if (param_ && param_->use_bias && inputs.size() > 2 && inputs[2]) {
            const float* bias_data = static_cast<const float*>(inputs[2]->data());
            kernels::BiasKernel::add_channel_bias<float>(
                output_data, bias_data,
                batch_size, out_features, 1,
                kernels::KernelBackend::CPU
            );
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(LinearCPUPlugin, "Gemm", kGEMM, CPU)

}  // namespace operators
}  // namespace mini_infer
