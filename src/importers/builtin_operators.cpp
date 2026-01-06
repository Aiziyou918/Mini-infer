#ifdef MINI_INFER_ONNX_ENABLED

#include "importers/internal/builtin_operators.h"
#include "importers/internal/attribute_helper.h"
#include "importers/internal/weight_importer.h"
#include "mini_infer/utils/logger.h"
#include "mini_infer/operators/generic_operator.h"
#include "mini_infer/operators/plugin_base.h"
#include "onnx.pb.h"

namespace mini_infer {
namespace importers {

namespace {

void register_node_outputs(ImporterContext& ctx,
                           graph::Node& graph_node,
                           const onnx::NodeProto& node,
                           const std::string& node_name) {
    for (int i = 0; i < node.output_size(); ++i) {
        const std::string& output_name = node.output(i);
        auto output_tensor = ctx.get_tensor(output_name);
        if (!output_tensor) {
            output_tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(output_name, output_tensor);
        }
        graph_node.output_tensors().push_back(output_tensor);
        ctx.register_tensor_producer(output_name, node_name, i);
    }
}

void connect_input_ports(ImporterContext& ctx,
                         const onnx::NodeProto& node,
                         const std::string& node_name) {
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        if (input_name.empty() || ctx.is_weight(input_name)) {
            continue;
        }
        std::string src_node_name;
        int src_port = 0;
        if (ctx.get_tensor_producer(input_name, src_node_name, src_port)) {
            (void)ctx.get_graph()->connect(src_node_name, node_name, src_port, i);
        } else {
            ctx.get_graph()->create_node(input_name);
            (void)ctx.get_graph()->connect(input_name, node_name, 0, i);
        }
    }
}

} // namespace

// =============================================================================
// Conv Operator Importer
// =============================================================================

core::Status ConvImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    // Parse attributes
    auto kernel_shape = attrs.get_ints("kernel_shape");
    auto strides = to_int_vector(attrs.get_ints("strides"));
    auto pads = to_int_vector(attrs.get_ints("pads"));
    auto dilations = to_int_vector(attrs.get_ints("dilations"));
    int64_t group = attrs.get_int("group", 1);
    
    // Validate inputs
    if (node.input_size() < 2) {
        ctx.set_error("Conv requires at least 2 inputs (input, weight)");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Get kernel size
    int kernel_h = kernel_shape.size() > 0 ? static_cast<int>(kernel_shape[0]) : 1;
    int kernel_w = kernel_shape.size() > 1 ? static_cast<int>(kernel_shape[1]) : kernel_h;
    
    // Get strides (default: [1, 1])
    int stride_h = strides.size() > 0 ? strides[0] : 1;
    int stride_w = strides.size() > 1 ? strides[1] : stride_h;
    
    // Get padding (ONNX uses [top, left, bottom, right], we use [h, w])
    int padding_h = pads.size() > 0 ? pads[0] : 0;
    int padding_w = pads.size() > 1 ? pads[1] : padding_h;
    
    // Get dilations
    int dilation_h = dilations.size() > 0 ? static_cast<int>(dilations[0]) : 1;
    int dilation_w = dilations.size() > 1 ? static_cast<int>(dilations[1]) : dilation_h;
    
    // Check if bias exists
    bool use_bias = (node.input_size() >= 3);
    
    ctx.log_info("Conv operator - kernel: [" + std::to_string(kernel_h) + 
                 "x" + std::to_string(kernel_w) + "], stride: [" + 
                 std::to_string(stride_h) + "x" + std::to_string(stride_w) + "]");
    
    // Create Conv2D parameter
    auto param = std::make_shared<operators::Conv2DParam>();
    param->kernel_h = kernel_h;
    param->kernel_w = kernel_w;
    param->stride_h = stride_h;
    param->stride_w = stride_w;
    param->padding_h = padding_h;
    param->padding_w = padding_w;
    param->dilation_h = dilation_h;
    param->dilation_w = dilation_w;
    param->groups = static_cast<int>(group);
    param->use_bias = use_bias;
    
    // Create operator with plugin parameter
    auto op = std::make_shared<operators::GenericOperator>("Conv", core::OpType::kCONVOLUTION);
    op->set_plugin_param(param);
    
    // Create node
    const std::string& node_name = node.output(0);  // Use first output as node name
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);
    
    // Set input tensors
    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) {
            tensor = ctx.get_weight(input_name);  // Try to get from weights
        }
        if (!tensor) {
            // Create placeholder tensor
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);
    
    // Connect graph edges for non-weight inputs
    connect_input_ports(ctx, node, node_name);
    
    // Register output tensors
    register_node_outputs(ctx, *graph_node, node, node_name);
    
    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// Gemm Operator Importer
// =============================================================================

core::Status GemmImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    float alpha = attrs.get_float("alpha", 1.0f);
    float beta = attrs.get_float("beta", 1.0f);
    int64_t transA = attrs.get_int("transA", 0);
    int64_t transB = attrs.get_int("transB", 0);
    
    // Validate inputs: A, B, and optional C (bias)
    if (node.input_size() < 2) {
        ctx.set_error("Gemm requires at least 2 inputs (A, B)");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    bool use_bias = (node.input_size() >= 3);
    
    ctx.log_info("Gemm operator - alpha: " + std::to_string(alpha) + 
                 ", beta: " + std::to_string(beta) + 
                 ", transA: " + std::to_string(transA) + 
                 ", transB: " + std::to_string(transB));
    
    // For standard Gemm with transB=1, we can use Linear operator
    // Note: ONNX Gemm: Y = alpha * A @ B^T + beta * C (when transB=1)
    // Mini-Infer Linear: Y = A @ W^T + bias
    if (transA == 0 && transB == 1 && alpha == 1.0f && beta == 1.0f) {
        // Create Linear operator with plugin parameter
        auto param = std::make_shared<operators::LinearParam>();
        param->use_bias = use_bias;
        auto op = std::make_shared<operators::GenericOperator>("Gemm", core::OpType::kGEMM);
        op->set_plugin_param(param);
        
        const std::string& node_name = node.output(0);
        auto graph_node = ctx.get_graph()->create_node(node_name);
        graph_node->set_operator(op);
        
        // Set input tensors
        std::vector<std::shared_ptr<core::Tensor>> input_tensors;
        for (int i = 0; i < node.input_size(); ++i) {
            const std::string& input_name = node.input(i);
            auto tensor = ctx.get_tensor(input_name);
            if (!tensor) {
                tensor = ctx.get_weight(input_name);
            }
            if (!tensor) {
                tensor = std::make_shared<core::Tensor>();
                ctx.register_tensor(input_name, tensor);
            }
            input_tensors.push_back(tensor);
        }
        graph_node->set_input_tensors(input_tensors);
        
        // Connect graph edges for non-weight inputs
        connect_input_ports(ctx, node, node_name);
        
        // Register output tensors
        register_node_outputs(ctx, *graph_node, node, node_name);
        
        ctx.add_node(graph_node);
        return core::Status::SUCCESS;
    } else {
        // TODO: Support other parameters
        ctx.log_warning("Gemm with non-standard parameters not fully supported yet");
        return core::Status::ERROR_NOT_IMPLEMENTED;
    }
}

// =============================================================================
// MatMul Operator Importer
// =============================================================================

core::Status MatMulImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("MatMul operator");
    
    // Validate inputs
    if (node.input_size() != 2) {
        ctx.set_error("MatMul requires exactly 2 inputs");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // MatMul: Y = A @ B, can be implemented using Linear without bias
    auto param = std::make_shared<operators::LinearParam>();
    param->use_bias = false;
    auto op = std::make_shared<operators::GenericOperator>("MatMul", core::OpType::kLINEAR);
    op->set_plugin_param(param);
    
    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);
    
    // Set input tensors
    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) {
            tensor = ctx.get_weight(input_name);
        }
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);
    
    // Connect graph edges for non-weight inputs
    connect_input_ports(ctx, node, node_name);
    
    // Register output tensors
    register_node_outputs(ctx, *graph_node, node, node_name);
    
    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// Relu Operator Importer
// =============================================================================

core::Status ReluImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Relu operator");
    
    // Validate inputs
    if (node.input_size() != 1) {
        ctx.set_error("Relu requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Create ReLU operator
    auto op = std::make_shared<operators::GenericOperator>("Relu", core::OpType::kRELU);
    
    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);
    
    // Set input tensor
    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});
    
    // Connect graph edge: input -> relu
    connect_input_ports(ctx, node, node_name);
    
    // Register output tensor
    register_node_outputs(ctx, *graph_node, node, node_name);
    
    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// MaxPool Operator Importer
// =============================================================================

core::Status MaxPoolImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    auto kernel_shape = attrs.get_ints("kernel_shape");
    auto strides = to_int_vector(attrs.get_ints("strides"));
    auto pads = to_int_vector(attrs.get_ints("pads"));
    
    // Validate inputs
    if (node.input_size() != 1) {
        ctx.set_error("MaxPool requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Get kernel size
    int kernel_h = kernel_shape.size() > 0 ? static_cast<int>(kernel_shape[0]) : 1;
    int kernel_w = kernel_shape.size() > 1 ? static_cast<int>(kernel_shape[1]) : kernel_h;
    
    // Get strides (default: same as kernel)
    int stride_h = strides.size() > 0 ? strides[0] : kernel_h;
    int stride_w = strides.size() > 1 ? strides[1] : kernel_w;
    
    // Get padding
    int padding_h = pads.size() > 0 ? pads[0] : 0;
    int padding_w = pads.size() > 1 ? pads[1] : padding_h;
    
    ctx.log_info("MaxPool operator - kernel: [" + std::to_string(kernel_h) + 
                 "x" + std::to_string(kernel_w) + "], stride: [" + 
                 std::to_string(stride_h) + "x" + std::to_string(stride_w) + "]");
    
    // Create Pooling parameter
    auto param = std::make_shared<operators::PoolingParam>();
    param->type = operators::PoolingType::MAX;
    param->kernel_h = kernel_h;
    param->kernel_w = kernel_w;
    param->stride_h = stride_h;
    param->stride_w = stride_w;
    param->padding_h = padding_h;
    param->padding_w = padding_w;
    
    // Create operator with plugin parameter
    auto op = std::make_shared<operators::GenericOperator>("MaxPool", core::OpType::kMAX_POOL);
    op->set_plugin_param(param);
    
    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);
    
    // Set input tensor
    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});
    
    // Connect graph edge: input -> maxpool
    connect_input_ports(ctx, node, node_name);
    
    // Register output tensor
    register_node_outputs(ctx, *graph_node, node, node_name);
    
    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// AveragePool Operator Importer
// =============================================================================

core::Status AveragePoolImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    auto kernel_shape = attrs.get_ints("kernel_shape");
    auto strides = to_int_vector(attrs.get_ints("strides"));
    auto pads = to_int_vector(attrs.get_ints("pads"));
    
    // Validate inputs
    if (node.input_size() != 1) {
        ctx.set_error("AveragePool requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Get kernel size
    int kernel_h = kernel_shape.size() > 0 ? static_cast<int>(kernel_shape[0]) : 1;
    int kernel_w = kernel_shape.size() > 1 ? static_cast<int>(kernel_shape[1]) : kernel_h;
    
    // Get strides (default: same as kernel)
    int stride_h = strides.size() > 0 ? strides[0] : kernel_h;
    int stride_w = strides.size() > 1 ? strides[1] : kernel_w;
    
    // Get padding
    int padding_h = pads.size() > 0 ? pads[0] : 0;
    int padding_w = pads.size() > 1 ? pads[1] : padding_h;
    
    ctx.log_info("AveragePool operator - kernel: [" + std::to_string(kernel_h) + 
                 "x" + std::to_string(kernel_w) + "]");
    
    // Create Pooling parameter
    auto param = std::make_shared<operators::PoolingParam>();
    param->type = operators::PoolingType::AVERAGE;
    param->kernel_h = kernel_h;
    param->kernel_w = kernel_w;
    param->stride_h = stride_h;
    param->stride_w = stride_w;
    param->padding_h = padding_h;
    param->padding_w = padding_w;
    
    // Create operator with plugin parameter
    auto op = std::make_shared<operators::GenericOperator>("AveragePool", core::OpType::kAVERAGE_POOL);
    op->set_plugin_param(param);
    
    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);
    
    // Set input tensor
    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});
    
    // Connect graph edge: input -> averagepool
    connect_input_ports(ctx, node, node_name);
    
    // Register output tensor
    register_node_outputs(ctx, *graph_node, node, node_name);
    
    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// GlobalAveragePool Operator Importer
// =============================================================================

core::Status GlobalAveragePoolImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("GlobalAveragePool operator");
    
    // TODO: Create GlobalAveragePool operator and add to graph
    ctx.log_warning("GlobalAveragePool operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// BatchNormalization Operator Importer
// =============================================================================

core::Status BatchNormalizationImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    float epsilon = attrs.get_float("epsilon", 1e-5f);
    float momentum = attrs.get_float("momentum", 0.9f);
    
    ctx.log_info("BatchNormalization operator - epsilon: " + std::to_string(epsilon));
    
    // TODO: Create BatchNormalization operator and add to graph
    ctx.log_warning("BatchNormalization operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Add Operator Importer
// =============================================================================

core::Status AddImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Add operator");
    
    // TODO: Create Add operator and add to graph
    ctx.log_warning("Add operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Mul Operator Importer
// =============================================================================

core::Status MulImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Mul operator");
    
    // TODO: Create Mul operator and add to graph
    ctx.log_warning("Mul operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Reshape Operator Importer
// =============================================================================

core::Status ReshapeImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    int64_t allowzero = attrs.get_int("allowzero", 0);
    
    ctx.log_info("Reshape operator");
    
    // Validate inputs: data + shape
    if (node.input_size() < 2) {
        ctx.set_error("Reshape requires 2 inputs (data, shape)");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Create Reshape operator
    auto param = std::make_shared<operators::ReshapeParam>();
    param->allowzero = (allowzero != 0);
    
    // Try to get shape from initializer (constant shape)
    const std::string& shape_input_name = node.input(1);
    auto shape_weight = ctx.get_weight(shape_input_name);
    
    if (shape_weight) {
        // Shape is a constant tensor, extract the shape values
        if (shape_weight->dtype() != core::DataType::INT64) {
            ctx.set_error("Reshape shape tensor must be INT64");
            return core::Status::ERROR_INVALID_ARGUMENT;
        }
        
        const int64_t* shape_data = static_cast<const int64_t*>(shape_weight->data());
        size_t shape_size = static_cast<size_t>(shape_weight->shape().numel());
        param->shape.assign(shape_data, shape_data + shape_size);
        
        ctx.log_info("Reshape with constant shape: [" + 
                     [&]() {
                         std::string s;
                         for (size_t i = 0; i < param->shape.size(); ++i) {
                             if (i > 0) s += ", ";
                             s += std::to_string(param->shape[i]);
                         }
                         return s;
                     }() + "]");
    }
    
    auto op = std::make_shared<operators::GenericOperator>("Reshape", core::OpType::kRESHAPE);
    op->set_plugin_param(param);
    
    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);
    
    // Collect input tensors
    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    
    // Input 0: data tensor
    const std::string& data_input_name = node.input(0);
    auto data_tensor = ctx.get_tensor(data_input_name);
    if (!data_tensor) {
        data_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(data_input_name, data_tensor);
    }
    input_tensors.push_back(data_tensor);
    
    // Input 1: shape tensor (may be weight or another tensor)
    if (shape_weight) {
        // Shape is constant, add it as input
        input_tensors.push_back(shape_weight);
    } else {
        // Shape is dynamic (from another node's output)
        auto shape_tensor = ctx.get_tensor(shape_input_name);
        if (!shape_tensor) {
            shape_tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(shape_input_name, shape_tensor);
        }
        input_tensors.push_back(shape_tensor);
    }
    
    graph_node->set_input_tensors(input_tensors);
    
    // Connect graph edges for non-weight inputs
    connect_input_ports(ctx, node, node_name);
    
    // Register output tensor
    register_node_outputs(ctx, *graph_node, node, node_name);
    
    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// Flatten Operator Importer
// =============================================================================

core::Status FlattenImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    int64_t axis = attrs.get_int("axis", 1);
    
    ctx.log_info("Flatten operator - axis: " + std::to_string(axis));
    
    // Validate inputs
    if (node.input_size() != 1) {
        ctx.set_error("Flatten requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Create Flatten parameter
    auto param = std::make_shared<operators::FlattenParam>();
    param->axis = static_cast<int>(axis);
    
    // Create operator with plugin parameter
    auto op = std::make_shared<operators::GenericOperator>("Flatten", core::OpType::kFLATTEN);
    op->set_plugin_param(param);
    
    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);
    
    // Set input tensor
    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});
    
    // Connect graph edges for non-weight inputs
    connect_input_ports(ctx, node, node_name);
    
    // Register output tensor
    register_node_outputs(ctx, *graph_node, node, node_name);
    
    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// Concat Operator Importer
// =============================================================================

core::Status ConcatImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    int64_t axis = attrs.get_int("axis");
    
    ctx.log_info("Concat operator - axis: " + std::to_string(axis));
    
    // TODO: Create Concat operator and add to graph
    ctx.log_warning("Concat operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Softmax Operator Importer
// =============================================================================

core::Status SoftmaxImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    
    int64_t axis = attrs.get_int("axis", -1);
    
    ctx.log_info("Softmax operator - axis: " + std::to_string(axis));
    
    // TODO: Create Softmax operator and add to graph
    ctx.log_warning("Softmax operator import not fully implemented yet");
    
    return core::Status::SUCCESS;
}

// =============================================================================
// Constant Operator Importer
// =============================================================================

core::Status ConstantImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Constant operator");
    
    // Constant nodes contain their value in the "value" attribute (TensorProto)
    // They don't need to be graph nodes, just register the constant value
    
    // Find the "value" attribute
    const onnx::AttributeProto* value_attr = nullptr;
    for (const auto& attr : node.attribute()) {
        if (attr.name() == "value" && attr.has_t()) {
            value_attr = &attr;
            break;
        }
    }
    
    if (!value_attr) {
        ctx.set_error("Constant node missing 'value' attribute");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Import the constant tensor
    std::string error_message;
    auto constant_tensor = WeightImporter::import_tensor(value_attr->t(), error_message);
    if (!constant_tensor) {
        ctx.set_error("Failed to import constant tensor: " + error_message);
        return core::Status::ERROR_INVALID_ARGUMENT;
    }
    
    // Register as weight (constant) so it can be used by other nodes
    const std::string& output_name = node.output(0);
    ctx.register_weight(output_name, constant_tensor);
    
    ctx.log_info("Constant registered: " + output_name + " " + constant_tensor->shape().to_string());
    
    // Constant nodes don't create graph nodes - they just provide values
    return core::Status::SUCCESS;
}

// =============================================================================
// Register All Builtin Operators
// =============================================================================

void register_builtin_operators(OperatorRegistry& registry) {
    // Convolution operators
    REGISTER_ONNX_OPERATOR("Conv", ConvImporter);
    
    // Linear algebra operators
    REGISTER_ONNX_OPERATOR("Gemm", GemmImporter);
    REGISTER_ONNX_OPERATOR("MatMul", MatMulImporter);
    
    // Activation operators
    REGISTER_ONNX_OPERATOR("Relu", ReluImporter);
    
    // Pooling operators
    REGISTER_ONNX_OPERATOR("MaxPool", MaxPoolImporter);
    REGISTER_ONNX_OPERATOR("AveragePool", AveragePoolImporter);
    REGISTER_ONNX_OPERATOR("GlobalAveragePool", GlobalAveragePoolImporter);
    
    // Normalization operators
    REGISTER_ONNX_OPERATOR("BatchNormalization", BatchNormalizationImporter);
    
    // Element-wise operators
    REGISTER_ONNX_OPERATOR("Add", AddImporter);
    REGISTER_ONNX_OPERATOR("Mul", MulImporter);
    
    // Shape operators
    REGISTER_ONNX_OPERATOR("Reshape", ReshapeImporter);
    REGISTER_ONNX_OPERATOR("Flatten", FlattenImporter);
    REGISTER_ONNX_OPERATOR("Concat", ConcatImporter);
    
    // Other operators
    REGISTER_ONNX_OPERATOR("Softmax", SoftmaxImporter);
    REGISTER_ONNX_OPERATOR("Constant", ConstantImporter);
    
    MI_LOG_INFO("[BuiltinOperators] Registered 15 builtin ONNX operators");
}

} // namespace importers
} // namespace mini_infer

#endif // MINI_INFER_ONNX_ENABLED
