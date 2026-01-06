#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cstring>

namespace mini_infer {
namespace operators {

/**
 * @brief Flatten CPU Plugin
 *
 * Flattens the input tensor into a 2D matrix.
 * For axis=1 (default): [N, C, H, W] -> [N, C*H*W]
 */
class FlattenCPUPlugin : public CPUPlugin<FlattenCPUPlugin, FlattenParam> {
public:
    FlattenCPUPlugin() {
        param_ = std::make_shared<FlattenParam>();
    }
    ~FlattenCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Flatten";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kFLATTEN;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 1;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        const int ndim = static_cast<int>(input_shape.ndim());

        int axis = param_ ? param_->axis : 1;
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis > ndim) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        int64_t outer_dim = 1;
        int64_t inner_dim = 1;

        for (int i = 0; i < axis; ++i) {
            outer_dim *= input_shape[i];
        }
        for (int i = axis; i < ndim; ++i) {
            inner_dim *= input_shape[i];
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape({outer_dim, inner_dim}));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        if (inputs.size() != 1 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input = inputs[0];
        auto& output = outputs[0];

        if (!input || !output) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const void* src = input->data();
        void* dst = output->data();

        if (src && dst && src != dst) {
            std::memcpy(dst, src, input->size_in_bytes());
        }

        return core::Status::SUCCESS;
    }
};

// Define creator and register plugin
REGISTER_PLUGIN_SIMPLE(FlattenCPUPlugin, "Flatten", kFLATTEN, CPU)

}  // namespace operators
}  // namespace mini_infer
