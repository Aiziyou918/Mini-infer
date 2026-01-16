#include "mini_infer/operators/cpu_plugin.h"
#include "mini_infer/operators/plugin_registry.h"

#include <cmath>
#include <algorithm>

namespace mini_infer {
namespace operators {

// =============================================================================
// Pow CPU Plugin
// =============================================================================

class PowCPUPlugin : public SimpleCPUPlugin<PowCPUPlugin> {
public:
    PowCPUPlugin() = default;
    ~PowCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Pow";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kPOW;
    }

    int32_t get_nb_outputs() const noexcept override {
        return 1;
    }

    int32_t get_nb_inputs() const noexcept override {
        return 2;
    }

    core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const override {

        if (input_shapes.size() != 2) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Broadcast shapes
        const auto& a = input_shapes[0];
        const auto& b = input_shapes[1];

        const size_t ndim_a = a.ndim();
        const size_t ndim_b = b.ndim();
        const size_t ndim_out = std::max(ndim_a, ndim_b);

        std::vector<int64_t> out_dims(ndim_out);
        for (size_t i = 0; i < ndim_out; ++i) {
            const int64_t dim_a = (i < ndim_a) ? a[ndim_a - 1 - i] : 1;
            const int64_t dim_b = (i < ndim_b) ? b[ndim_b - 1 - i] : 1;

            if (dim_a == dim_b) {
                out_dims[ndim_out - 1 - i] = dim_a;
            } else if (dim_a == 1) {
                out_dims[ndim_out - 1 - i] = dim_b;
            } else if (dim_b == 1) {
                out_dims[ndim_out - 1 - i] = dim_a;
            } else {
                return core::Status::ERROR_INVALID_ARGUMENT;
            }
        }

        output_shapes.clear();
        output_shapes.push_back(core::Shape(out_dims));
        return core::Status::SUCCESS;
    }

    core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) override {
        (void)context;

        if (inputs.size() != 2 || outputs.size() != 1) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        const auto& base = inputs[0];
        const auto& exp = inputs[1];
        auto& out = outputs[0];

        if (!base || !exp || !out) {
            return core::Status::ERROR_INVALID_ARGUMENT;
        }

        // Simple case: scalar exponent
        if (exp->shape().numel() == 1) {
            const float* base_data = static_cast<const float*>(base->data());
            const float exp_val = *static_cast<const float*>(exp->data());
            float* out_data = static_cast<float*>(out->data());

            const int64_t total = base->shape().numel();
            for (int64_t i = 0; i < total; ++i) {
                out_data[i] = std::pow(base_data[i], exp_val);
            }
            return core::Status::SUCCESS;
        }

        // General case with broadcasting (simplified)
        const float* base_data = static_cast<const float*>(base->data());
        const float* exp_data = static_cast<const float*>(exp->data());
        float* out_data = static_cast<float*>(out->data());

        const int64_t total = out->shape().numel();
        const int64_t base_total = base->shape().numel();
        const int64_t exp_total = exp->shape().numel();

        for (int64_t i = 0; i < total; ++i) {
            out_data[i] = std::pow(base_data[i % base_total], exp_data[i % exp_total]);
        }

        return core::Status::SUCCESS;
    }
};

// =============================================================================
// Sqrt CPU Plugin
// =============================================================================

class SqrtCPUPlugin : public SimpleCPUPlugin<SqrtCPUPlugin> {
public:
    SqrtCPUPlugin() = default;
    ~SqrtCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Sqrt";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kSQRT;
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

        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
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

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const int64_t total = input->shape().numel();
        for (int64_t i = 0; i < total; ++i) {
            out_data[i] = std::sqrt(in_data[i]);
        }

        return core::Status::SUCCESS;
    }
};

// =============================================================================
// Erf CPU Plugin
// =============================================================================

class ErfCPUPlugin : public SimpleCPUPlugin<ErfCPUPlugin> {
public:
    ErfCPUPlugin() = default;
    ~ErfCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Erf";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kERF;
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

        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
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

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const int64_t total = input->shape().numel();
        for (int64_t i = 0; i < total; ++i) {
            out_data[i] = std::erf(in_data[i]);
        }

        return core::Status::SUCCESS;
    }
};

// =============================================================================
// Tanh CPU Plugin (standalone, not activation)
// =============================================================================

class TanhCPUPlugin : public SimpleCPUPlugin<TanhCPUPlugin> {
public:
    TanhCPUPlugin() = default;
    ~TanhCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Tanh";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kTANH;
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

        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
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

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const int64_t total = input->shape().numel();
        for (int64_t i = 0; i < total; ++i) {
            out_data[i] = std::tanh(in_data[i]);
        }

        return core::Status::SUCCESS;
    }
};

// =============================================================================
// Neg CPU Plugin
// =============================================================================

class NegCPUPlugin : public SimpleCPUPlugin<NegCPUPlugin> {
public:
    NegCPUPlugin() = default;
    ~NegCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Neg";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kNEG;
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

        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
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

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const int64_t total = input->shape().numel();
        for (int64_t i = 0; i < total; ++i) {
            out_data[i] = -in_data[i];
        }

        return core::Status::SUCCESS;
    }
};

// =============================================================================
// Exp CPU Plugin
// =============================================================================

class ExpCPUPlugin : public SimpleCPUPlugin<ExpCPUPlugin> {
public:
    ExpCPUPlugin() = default;
    ~ExpCPUPlugin() override = default;

    const char* get_plugin_type() const noexcept override {
        return "Exp";
    }

    core::OpType get_op_type() const noexcept override {
        return core::OpType::kEXP;
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

        output_shapes.clear();
        output_shapes.push_back(input_shapes[0]);
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

        const float* in_data = static_cast<const float*>(input->data());
        float* out_data = static_cast<float*>(output->data());

        const int64_t total = input->shape().numel();
        for (int64_t i = 0; i < total; ++i) {
            out_data[i] = std::exp(in_data[i]);
        }

        return core::Status::SUCCESS;
    }
};

// Register plugins
REGISTER_PLUGIN_SIMPLE(PowCPUPlugin, "Pow", kPOW, CPU)
REGISTER_PLUGIN_SIMPLE(SqrtCPUPlugin, "Sqrt", kSQRT, CPU)
REGISTER_PLUGIN_SIMPLE(ErfCPUPlugin, "Erf", kERF, CPU)
REGISTER_PLUGIN_SIMPLE(TanhCPUPlugin, "Tanh", kTANH, CPU)
REGISTER_PLUGIN_SIMPLE(NegCPUPlugin, "Neg", kNEG, CPU)
REGISTER_PLUGIN_SIMPLE(ExpCPUPlugin, "Exp", kEXP, CPU)

}  // namespace operators
}  // namespace mini_infer
