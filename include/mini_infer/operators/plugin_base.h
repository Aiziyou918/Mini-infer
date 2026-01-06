#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mini_infer/backends/device_context.h"
#include "mini_infer/core/op_type.h"
#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"
#include "mini_infer/operators/activation_type.h"

namespace mini_infer {
namespace operators {

/**
 * @brief Plugin execution context
 *
 * Contains runtime information needed during plugin execution.
 * Similar to TensorRT's IPluginV2::enqueue() context parameters.
 */
struct PluginContext {
    backends::DeviceContext* device_context{nullptr};
    std::shared_ptr<void> workspace;
    size_t workspace_size{0};
};

/**
 * @brief Plugin parameter base class
 */
struct PluginParam {
    virtual ~PluginParam() = default;
};

// ============================================================================
// Operator Parameter Structures
// ============================================================================

/**
 * @brief Pooling type enumeration
 */
enum class PoolingType {
    MAX,
    AVERAGE
};

/**
 * @brief Conv2D operator parameters
 */
struct Conv2DParam : public PluginParam {
    int kernel_h{1};
    int kernel_w{1};
    int stride_h{1};
    int stride_w{1};
    int padding_h{0};
    int padding_w{0};
    int dilation_h{1};
    int dilation_w{1};
    int groups{1};
    bool use_bias{true};
    ActivationType activation{ActivationType::NONE};

    Conv2DParam() = default;
    Conv2DParam(int kh, int kw, int sh, int sw, int ph, int pw, int g, bool bias)
        : kernel_h(kh), kernel_w(kw), stride_h(sh), stride_w(sw),
          padding_h(ph), padding_w(pw), groups(g), use_bias(bias) {}
};

/**
 * @brief Linear operator parameters
 */
struct LinearParam : public PluginParam {
    int in_features{0};
    int out_features{0};
    bool use_bias{true};

    LinearParam() = default;
    LinearParam(int in_f, int out_f, bool bias)
        : in_features(in_f), out_features(out_f), use_bias(bias) {}
};

/**
 * @brief Pooling operator parameters
 */
struct PoolingParam : public PluginParam {
    PoolingType type{PoolingType::MAX};
    int kernel_h{2};
    int kernel_w{2};
    int stride_h{2};
    int stride_w{2};
    int padding_h{0};
    int padding_w{0};

    PoolingParam() = default;
    PoolingParam(PoolingType t, int kh, int kw, int sh, int sw, int ph, int pw)
        : type(t), kernel_h(kh), kernel_w(kw), stride_h(sh), stride_w(sw),
          padding_h(ph), padding_w(pw) {}
};

/**
 * @brief Flatten operator parameters
 */
struct FlattenParam : public PluginParam {
    int axis{1};

    FlattenParam() = default;
    explicit FlattenParam(int a) : axis(a) {}
};

/**
 * @brief Reshape operator parameters
 */
struct ReshapeParam : public PluginParam {
    std::vector<int64_t> shape;
    bool allowzero{false};

    ReshapeParam() = default;
    explicit ReshapeParam(const std::vector<int64_t>& s) : shape(s) {}
};

/**
 * @brief Softmax operator parameters
 */
struct SoftmaxParam : public PluginParam {
    int axis{-1};

    SoftmaxParam() = default;
    explicit SoftmaxParam(int a) : axis(a) {}
};

/**
 * @brief Abstract base class for all plugins
 *
 * Inspired by TensorRT's IPluginV2 interface.
 * Each plugin encapsulates both shape inference and execution logic.
 * Different backends (CPU/CUDA) are implemented through inheritance.
 */
class IPlugin {
public:
    virtual ~IPlugin() = default;

    // ========================================================================
    // Identity
    // ========================================================================

    /**
     * @brief Get the plugin type name
     * @return Plugin type name string (e.g., "ReLU", "Conv2D")
     */
    virtual const char* get_plugin_type() const noexcept = 0;

    /**
     * @brief Get the operator type enum
     * @return OpType enum value
     */
    virtual core::OpType get_op_type() const noexcept = 0;

    /**
     * @brief Get the device type this plugin runs on
     * @return DeviceType enum value
     */
    virtual core::DeviceType get_device_type() const noexcept = 0;

    // ========================================================================
    // I/O Configuration
    // ========================================================================

    /**
     * @brief Get the number of outputs
     * @return Number of output tensors
     */
    virtual int32_t get_nb_outputs() const noexcept { return 1; }

    /**
     * @brief Get the number of inputs
     * @return Number of input tensors
     */
    virtual int32_t get_nb_inputs() const noexcept { return 1; }

    // ========================================================================
    // Shape Inference
    // ========================================================================

    /**
     * @brief Infer output shapes from input shapes
     * @param input_shapes Input tensor shapes
     * @param output_shapes Output tensor shapes (to be filled)
     * @return Status code
     */
    virtual core::Status infer_output_shapes(
        const std::vector<core::Shape>& input_shapes,
        std::vector<core::Shape>& output_shapes) const = 0;

    /**
     * @brief Infer output shapes and dtypes
     * @param input_shapes Input tensor shapes
     * @param input_dtypes Input tensor dtypes
     * @param output_shapes Output tensor shapes (to be filled)
     * @param output_dtypes Output tensor dtypes (to be filled)
     * @return Status code
     *
     * Default behavior: infer shapes and propagate the first input dtype.
     */
    virtual core::Status infer_output_metadata(
        const std::vector<core::Shape>& input_shapes,
        const std::vector<core::DataType>& input_dtypes,
        std::vector<core::Shape>& output_shapes,
        std::vector<core::DataType>& output_dtypes) const {
        auto status = infer_output_shapes(input_shapes, output_shapes);
        if (status != core::Status::SUCCESS) {
            return status;
        }

        output_dtypes.clear();
        if (output_shapes.empty()) {
            return core::Status::SUCCESS;
        }

        const core::DataType inferred =
            input_dtypes.empty() ? core::DataType::FLOAT32 : input_dtypes[0];
        output_dtypes.assign(output_shapes.size(), inferred);
        return core::Status::SUCCESS;
    }

    // ========================================================================
    // Workspace
    // ========================================================================

    /**
     * @brief Get workspace size required for execution
     * @param input_shapes Input tensor shapes
     * @return Workspace size in bytes
     */
    virtual size_t get_workspace_size(
        const std::vector<core::Shape>& input_shapes) const noexcept {
        (void)input_shapes;
        return 0;
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /**
     * @brief Initialize the plugin
     *
     * Called once before first execution. Use this to allocate resources,
     * create cuDNN descriptors, etc.
     *
     * @return Status code
     */
    virtual core::Status initialize() { return core::Status::SUCCESS; }

    /**
     * @brief Terminate the plugin
     *
     * Called when the plugin is no longer needed. Use this to release
     * resources, destroy cuDNN descriptors, etc.
     */
    virtual void terminate() noexcept {}

    // ========================================================================
    // Execution
    // ========================================================================

    /**
     * @brief Execute the plugin
     *
     * This is the core method that performs the actual computation.
     *
     * @param inputs Input tensors
     * @param outputs Output tensors (pre-allocated)
     * @param context Execution context
     * @return Status code
     */
    virtual core::Status enqueue(
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        std::vector<std::shared_ptr<core::Tensor>>& outputs,
        const PluginContext& context) = 0;

    // ========================================================================
    // Clone
    // ========================================================================

    /**
     * @brief Clone the plugin
     *
     * Creates a deep copy of the plugin with all parameters.
     * Used when the same operator needs to be instantiated multiple times.
     *
     * @return Unique pointer to the cloned plugin
     */
    virtual std::unique_ptr<IPlugin> clone() const = 0;

    // ========================================================================
    // Parameters
    // ========================================================================

    /**
     * @brief Get pointer to plugin parameters
     * @return Pointer to parameters, or nullptr if no parameters
     */
    virtual const void* get_param_ptr() const noexcept { return nullptr; }

    /**
     * @brief Set plugin parameters
     * @param param Shared pointer to parameters
     */
    virtual void set_param(std::shared_ptr<PluginParam> param) {
        (void)param;
    }
};

/**
 * @brief Plugin creator interface
 *
 * Factory interface for creating plugin instances.
 * Each plugin type should have a corresponding creator.
 */
class IPluginCreator {
public:
    virtual ~IPluginCreator() = default;

    /**
     * @brief Get the plugin type name
     * @return Plugin type name string
     */
    virtual const char* get_plugin_type() const noexcept = 0;

    /**
     * @brief Get the operator type enum
     * @return OpType enum value
     */
    virtual core::OpType get_op_type() const noexcept = 0;

    /**
     * @brief Get the device type
     * @return DeviceType enum value
     */
    virtual core::DeviceType get_device_type() const noexcept = 0;

    /**
     * @brief Create a plugin instance
     * @return Unique pointer to the created plugin
     */
    virtual std::unique_ptr<IPlugin> create_plugin() const = 0;
};

}  // namespace operators
}  // namespace mini_infer
