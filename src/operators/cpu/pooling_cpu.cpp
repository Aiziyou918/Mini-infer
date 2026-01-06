#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <algorithm>
#include <limits>

namespace mini_infer {
namespace operators {

/**
 * @brief MaxPool CPU Plugin
 */
class MaxPoolCPUPlugin : public CPUPlugin<MaxPoolCPUPlugin, PoolingParam> {
public:
    MaxPoolCPUPlugin() {
        param_ = std::make_shared<PoolingParam>();
        param_->type = PoolingType::MAX;
    }
    ~MaxPoolCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "MaxPool";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kMAX_POOL;
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
        if (input_shapes.size() != 1 || input_shapes[0].ndim() != 4) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        int64_t N = input_shape[0];
        int64_t C = input_shape[1];
        int64_t H_in = input_shape[2];
        int64_t W_in = input_shape[3];

        int kh = param_ ? param_->kernel_h : 2;
        int kw = param_ ? param_->kernel_w : 2;
        int sh = param_ ? param_->stride_h : 2;
        int sw = param_ ? param_->stride_w : 2;
        int ph = param_ ? param_->padding_h : 0;
        int pw = param_ ? param_->padding_w : 0;

        int64_t H_out = (H_in + 2 * ph - kh) / sh + 1;
        int64_t W_out = (W_in + 2 * pw - kw) / sw + 1;

        output_shapes.clear();
        output_shapes.push_back(core::Shape({N, C, H_out, W_out}));
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

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& input_shape = input->shape();
        const auto& output_shape = output->shape();

        int N = static_cast<int>(input_shape[0]);
        int C = static_cast<int>(input_shape[1]);
        int H_in = static_cast<int>(input_shape[2]);
        int W_in = static_cast<int>(input_shape[3]);
        int H_out = static_cast<int>(output_shape[2]);
        int W_out = static_cast<int>(output_shape[3]);

        int kh = param_->kernel_h;
        int kw = param_->kernel_w;
        int sh = param_->stride_h;
        int sw = param_->stride_w;
        int ph = param_->padding_h;
        int pw = param_->padding_w;

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                const float* input_nc = in_data + (n * C + c) * H_in * W_in;
                float* output_nc = out_data + (n * C + c) * H_out * W_out;

                for (int h_out = 0; h_out < H_out; ++h_out) {
                    for (int w_out = 0; w_out < W_out; ++w_out) {
                        int h_start = h_out * sh - ph;
                        int w_start = w_out * sw - pw;
                        int h_end = std::min(h_start + kh, H_in);
                        int w_end = std::min(w_start + kw, W_in);
                        h_start = std::max(h_start, 0);
                        w_start = std::max(w_start, 0);

                    float max_val = std::numeric_limits<float>::lowest();
                        for (int h = h_start; h < h_end; ++h) {
                            for (int w = w_start; w < w_end; ++w) {
                                max_val = std::max(max_val, input_nc[h * W_in + w]);
                            }
                        }
                        output_nc[h_out * W_out + w_out] = max_val;
                    }
                }
            }
        }

        return core::Status::SUCCESS;
    }
};

/**
 * @brief AvgPool CPU Plugin
 */
class AvgPoolCPUPlugin : public CPUPlugin<AvgPoolCPUPlugin, PoolingParam> {
public:
    AvgPoolCPUPlugin() {
        param_ = std::make_shared<PoolingParam>();
        param_->type = PoolingType::AVERAGE;
    }
    ~AvgPoolCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "AveragePool";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kAVERAGE_POOL;
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
        if (input_shapes.size() != 1 || input_shapes[0].ndim() != 4) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& input_shape = input_shapes[0];
        int64_t N = input_shape[0];
        int64_t C = input_shape[1];
        int64_t H_in = input_shape[2];
        int64_t W_in = input_shape[3];

        int kh = param_ ? param_->kernel_h : 2;
        int kw = param_ ? param_->kernel_w : 2;
        int sh = param_ ? param_->stride_h : 2;
        int sw = param_ ? param_->stride_w : 2;
        int ph = param_ ? param_->padding_h : 0;
        int pw = param_ ? param_->padding_w : 0;

        int64_t H_out = (H_in + 2 * ph - kh) / sh + 1;
        int64_t W_out = (W_in + 2 * pw - kw) / sw + 1;

        output_shapes.clear();
        output_shapes.push_back(core::Shape({N, C, H_out, W_out}));
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

        if (input->dtype() != core::DataType::FLOAT32) {
            return core::Status::ERROR_NOT_IMPLEMENTED;
        }

        const auto& input_shape = input->shape();
        const auto& output_shape = output->shape();

        int N = static_cast<int>(input_shape[0]);
        int C = static_cast<int>(input_shape[1]);
        int H_in = static_cast<int>(input_shape[2]);
        int W_in = static_cast<int>(input_shape[3]);
        int H_out = static_cast<int>(output_shape[2]);
        int W_out = static_cast<int>(output_shape[3]);

        int kh = param_->kernel_h;
        int kw = param_->kernel_w;
        int sh = param_->stride_h;
        int sw = param_->stride_w;
        int ph = param_->padding_h;
        int pw = param_->padding_w;

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                const float* input_nc = in_data + (n * C + c) * H_in * W_in;
                float* output_nc = out_data + (n * C + c) * H_out * W_out;

                for (int h_out = 0; h_out < H_out; ++h_out) {
                    for (int w_out = 0; w_out < W_out; ++w_out) {
                        int h_start = h_out * sh - ph;
                        int w_start = w_out * sw - pw;
                        int h_end = std::min(h_start + kh, H_in);
                        int w_end = std::min(w_start + kw, W_in);
                        h_start = std::max(h_start, 0);
                        w_start = std::max(w_start, 0);

                        float sum = 0.0f;
                        int count = 0;
                        for (int h = h_start; h < h_end; ++h) {
                            for (int w = w_start; w < w_end; ++w) {
                                sum += input_nc[h * W_in + w];
                                ++count;
                            }
                        }
                output_nc[h_out * W_out + w_out] = (count > 0) ? (sum / count) : 0.0f;
                    }
                }
            }
        }

        return core::Status::SUCCESS;
    }
};

// Define creators and register plugins
REGISTER_PLUGIN_SIMPLE(MaxPoolCPUPlugin, "MaxPool", kMAX_POOL, CPU)
REGISTER_PLUGIN_SIMPLE(AvgPoolCPUPlugin, "AveragePool", kAVERAGE_POOL, CPU)

}  // namespace operators
}  // namespace mini_infer
