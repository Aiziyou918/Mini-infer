#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

namespace mini_infer {
namespace operators {

class CastCPUPlugin : public SimpleCPUPlugin<CastCPUPlugin> {
public:
    CastCPUPlugin() = default;
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

    void set_param(std::shared_ptr<PluginParam> param) override {
        param_ = std::dynamic_pointer_cast<CastParam>(param);
    }

    const void* get_param_ptr() const noexcept override {
        return param_.get();
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {
        if (input_shapes.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

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

        output_dtypes.clear();
        core::DataType target_dtype = core::DataType::FLOAT32;
        if (param_) {
            // ONNX TensorProto.DataType: 1=FLOAT, 6=INT32, 7=INT64, 9=BOOL, 11=DOUBLE
            switch (param_->to_dtype) {
                case 1: target_dtype = core::DataType::FLOAT32; break;
                case 6: target_dtype = core::DataType::INT32; break;
                case 7: target_dtype = core::DataType::INT64; break;
                default: target_dtype = core::DataType::FLOAT32; break;
            }
        }
        output_dtypes.push_back(target_dtype);
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

        const int64_t total = input->shape().numel();
        core::DataType src_dtype = input->dtype();
        core::DataType dst_dtype = output->dtype();

        // Fast path: same dtype, just copy
        if (src_dtype == dst_dtype) {
            std::memcpy(output->data(), input->data(), total * input->element_size());
            return core::Status::SUCCESS;
        }

        // Handle common conversions
        if (src_dtype == core::DataType::FLOAT32 && dst_dtype == core::DataType::FLOAT32) {
            std::memcpy(output->data(), input->data(), total * sizeof(float));
        } else if (src_dtype == core::DataType::FLOAT32 && dst_dtype == core::DataType::INT64) {
            const float* src = static_cast<const float*>(input->data());
            int64_t* dst = static_cast<int64_t*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<int64_t>(src[i]);
            }
        } else if (src_dtype == core::DataType::INT64 && dst_dtype == core::DataType::FLOAT32) {
            const int64_t* src = static_cast<const int64_t*>(input->data());
            float* dst = static_cast<float*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
        } else if (src_dtype == core::DataType::INT32 && dst_dtype == core::DataType::FLOAT32) {
            const int32_t* src = static_cast<const int32_t*>(input->data());
            float* dst = static_cast<float*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
        } else if (src_dtype == core::DataType::FLOAT32 && dst_dtype == core::DataType::INT32) {
            const float* src = static_cast<const float*>(input->data());
            int32_t* dst = static_cast<int32_t*>(output->data());
            for (int64_t i = 0; i < total; ++i) {
                dst[i] = static_cast<int32_t>(src[i]);
            }
        } else {
            // For unsupported conversions, just copy if same size
            if (input->element_size() == output->element_size()) {
                std::memcpy(output->data(), input->data(), total * input->element_size());
            } else {
                return core::Status::ERROR_NOT_IMPLEMENTED;
            }
        }

        return core::Status::SUCCESS;
    }

private:
    std::shared_ptr<CastParam> param_;
};

REGISTER_PLUGIN_SIMPLE(CastCPUPlugin, "Cast", kCAST, CPU)

}  // namespace operators
}  // namespace mini_infer
