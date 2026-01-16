#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cstring>

namespace mini_infer {
namespace operators {

/**
 * @brief Cast CPU Plugin
 *
 * Casts the input tensor to the specified data type.
 */
class CastCPUPlugin : public CPUPlugin<CastCPUPlugin, CastParam> {
public:
    CastCPUPlugin() {
        param_ = std::make_shared<CastParam>();
    }
    ~CastCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Cast";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kCAST;
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

        if (input_shapes.empty()) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Output shape is same as input shape
        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
        return core::Status::SUCCESS;
    }

    core::Status infer_output_metadata(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& input_dtypes,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const override {
        (void)input_dtypes;

        auto status = infer_output_shapes(input_shapes, output_shapes);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        // Output dtype is from parameter
        output_dtypes.clear();
        output_dtypes.push_back(param_ ? param_->to_dtype : core::DataType::FLOAT32);
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

        if (!input || !output || !param_) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const int64_t total = input->shape().numel();
        const core::DataType src_dtype = input->dtype();
        const core::DataType dst_dtype = param_->to_dtype;

        // Same type - just copy
        if (src_dtype == dst_dtype) {
            std::memcpy(output->data(), input->data(), input->size_in_bytes());
            return core::Status::SUCCESS;
        }

        // FLOAT32 -> INT64
        if (src_dtype == core::DataType::FLOAT32 && dst_dtype == core::DataType::INT64) {
            const float* src = static_cast<const float*>(input->data());
            int64_t* dst = static_cast<int64_t*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<int64_t>(src[i]);
            }
            return core::Status::SUCCESS;
        }

        // INT64 -> FLOAT32
        if (src_dtype == core::DataType::INT64 && dst_dtype == core::DataType::FLOAT32) {
            const int64_t* src = static_cast<const int64_t*>(input->data());
            float* dst = static_cast<float*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
            return core::Status::SUCCESS;
        }

        // FLOAT32 -> INT32
        if (src_dtype == core::DataType::FLOAT32 && dst_dtype == core::DataType::INT32) {
            const float* src = static_cast<const float*>(input->data());
            int32_t* dst = static_cast<int32_t*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<int32_t>(src[i]);
            }
            return core::Status::SUCCESS;
        }

        // INT32 -> FLOAT32
        if (src_dtype == core::DataType::INT32 && dst_dtype == core::DataType::FLOAT32) {
            const int32_t* src = static_cast<const int32_t*>(input->data());
            float* dst = static_cast<float*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
            return core::Status::SUCCESS;
        }

        // INT32 -> INT64
        if (src_dtype == core::DataType::INT32 && dst_dtype == core::DataType::INT64) {
            const int32_t* src = static_cast<const int32_t*>(input->data());
            int64_t* dst = static_cast<int64_t*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<int64_t>(src[i]);
            }
            return core::Status::SUCCESS;
        }

        // INT64 -> INT32
        if (src_dtype == core::DataType::INT64 && dst_dtype == core::DataType::INT32) {
            const int64_t* src = static_cast<const int64_t*>(input->data());
            int32_t* dst = static_cast<int32_t*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<int32_t>(src[i]);
            }
            return core::Status::SUCCESS;
        }

        return core::Status::ERROR_NOT_IMPLEMENTED;
    }
};

REGISTER_PLUGIN_SIMPLE(CastCPUPlugin, "Cast", kCAST, CPU)

}  // namespace operators
}  // namespace mini_infer
