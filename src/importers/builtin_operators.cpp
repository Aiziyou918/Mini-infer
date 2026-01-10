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

    // MatMul: Y = A @ B
    auto op = std::make_shared<operators::GenericOperator>("MatMul", core::OpType::kMATMUL);

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

    if (node.input_size() != 2) {
        ctx.set_error("Add requires exactly 2 inputs");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Add", core::OpType::kADD);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// Mul Operator Importer
// =============================================================================

core::Status MulImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Mul operator");

    if (node.input_size() != 2) {
        ctx.set_error("Mul requires exactly 2 inputs");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Mul", core::OpType::kMUL);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
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

    if (node.input_size() < 1) {
        ctx.set_error("Concat requires at least 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto param = std::make_shared<operators::ConcatParam>();
    param->axis = axis;

    auto op = std::make_shared<operators::GenericOperator>("Concat", core::OpType::kCONCAT);
    op->set_plugin_param(param);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// Softmax Operator Importer
// =============================================================================

core::Status SoftmaxImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);

    int64_t axis = attrs.get_int("axis", -1);

    ctx.log_info("Softmax operator - axis: " + std::to_string(axis));

    if (node.input_size() != 1) {
        ctx.set_error("Softmax requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto param = std::make_shared<operators::SoftmaxParam>();
    param->axis = static_cast<int>(axis);

    auto op = std::make_shared<operators::GenericOperator>("Softmax", core::OpType::kSOFTMAX);
    op->set_plugin_param(param);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
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
// BERT-related Operator Importers
// =============================================================================

// Div Importer
core::Status DivImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Div operator");

    if (node.input_size() != 2) {
        ctx.set_error("Div requires exactly 2 inputs");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Div", core::OpType::kDIV);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// Sqrt Importer
core::Status SqrtImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Sqrt operator");

    if (node.input_size() != 1) {
        ctx.set_error("Sqrt requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Sqrt", core::OpType::kSQRT);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) input_tensor = ctx.get_weight(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// Gelu Importer
core::Status GeluImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Gelu operator");

    if (node.input_size() != 1) {
        ctx.set_error("Gelu requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Gelu", core::OpType::kGELU);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// Transpose Importer
core::Status TransposeImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    auto perm = attrs.get_ints("perm");

    ctx.log_info("Transpose operator");

    if (node.input_size() != 1) {
        ctx.set_error("Transpose requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto param = std::make_shared<operators::TransposeParam>();
    param->perm = perm;

    auto op = std::make_shared<operators::GenericOperator>("Transpose", core::OpType::kTRANSPOSE);
    op->set_plugin_param(param);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// Gather Importer
core::Status GatherImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    int64_t axis = attrs.get_int("axis", 0);

    ctx.log_info("Gather operator - axis: " + std::to_string(axis));

    if (node.input_size() != 2) {
        ctx.set_error("Gather requires exactly 2 inputs (data, indices)");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto param = std::make_shared<operators::GatherParam>();
    param->axis = axis;

    auto op = std::make_shared<operators::GenericOperator>("Gather", core::OpType::kGATHER);
    op->set_plugin_param(param);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// LayerNormalization Importer
core::Status LayerNormalizationImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    float epsilon = attrs.get_float("epsilon", 1e-5f);
    int64_t axis = attrs.get_int("axis", -1);

    ctx.log_info("LayerNormalization operator - epsilon: " + std::to_string(epsilon) + ", axis: " + std::to_string(axis));

    if (node.input_size() < 1) {
        ctx.set_error("LayerNormalization requires at least 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto param = std::make_shared<operators::LayerNormParam>();
    param->epsilon = epsilon;
    param->axis = axis;

    auto op = std::make_shared<operators::GenericOperator>("LayerNormalization", core::OpType::kLAYER_NORM);
    op->set_plugin_param(param);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// Squeeze Importer
core::Status SqueezeImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    auto axes = attrs.get_ints("axes");

    ctx.log_info("Squeeze operator");

    auto param = std::make_shared<operators::SqueezeParam>();
    param->axes = axes;

    auto op = std::make_shared<operators::GenericOperator>("Squeeze", core::OpType::kSQUEEZE);
    op->set_plugin_param(param);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// Unsqueeze Importer
core::Status UnsqueezeImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    AttributeHelper attrs(node);
    auto axes = attrs.get_ints("axes");

    ctx.log_info("Unsqueeze operator");

    auto param = std::make_shared<operators::UnsqueezeParam>();
    param->axes = axes;

    auto op = std::make_shared<operators::GenericOperator>("Unsqueeze", core::OpType::kUNSQUEEZE);
    op->set_plugin_param(param);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// Identity Operator Importer
// =============================================================================

core::Status IdentityImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Identity operator");

    if (node.input_size() != 1) {
        ctx.set_error("Identity requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Identity is a pass-through operation
    // We just need to register the output tensor as an alias to the input
    const std::string& input_name = node.input(0);
    const std::string& output_name = node.output(0);

    // Get or create input tensor
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = ctx.get_weight(input_name);
    }
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }

    // Register output as the same tensor (alias)
    ctx.register_tensor(output_name, input_tensor);

    // Copy producer info
    std::string src_node_name;
    int src_port = 0;
    if (ctx.get_tensor_producer(input_name, src_node_name, src_port)) {
        ctx.register_tensor_producer(output_name, src_node_name, src_port);
    }

    return core::Status::SUCCESS;
}

// =============================================================================
// Shape Operator Importer
// =============================================================================

core::Status ShapeImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Shape operator");

    if (node.input_size() != 1) {
        ctx.set_error("Shape requires exactly 1 input");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Shape", core::OpType::kSHAPE);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    const std::string& input_name = node.input(0);
    auto input_tensor = ctx.get_tensor(input_name);
    if (!input_tensor) {
        input_tensor = ctx.get_weight(input_name);
    }
    if (!input_tensor) {
        input_tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, input_tensor);
    }
    graph_node->set_input_tensors({input_tensor});

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// Slice Operator Importer
// =============================================================================

core::Status SliceImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Slice operator");

    // ONNX Slice (opset >= 10):
    // Inputs: data, starts, ends, [axes], [steps]
    if (node.input_size() < 3) {
        ctx.set_error("Slice requires at least 3 inputs (data, starts, ends)");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    // Create slice parameter
    auto param = std::make_shared<operators::SliceParam>();

    // Try to get starts from initializer
    const std::string& starts_name = node.input(1);
    auto starts_tensor = ctx.get_weight(starts_name);
    if (starts_tensor && starts_tensor->data()) {
        const int64_t* starts_data = static_cast<const int64_t*>(starts_tensor->data());
        int64_t num_starts = starts_tensor->shape().numel();
        param->starts.assign(starts_data, starts_data + num_starts);
    }

    // Try to get ends from initializer
    const std::string& ends_name = node.input(2);
    auto ends_tensor = ctx.get_weight(ends_name);
    if (ends_tensor && ends_tensor->data()) {
        const int64_t* ends_data = static_cast<const int64_t*>(ends_tensor->data());
        int64_t num_ends = ends_tensor->shape().numel();
        param->ends.assign(ends_data, ends_data + num_ends);
    }

    // Try to get axes from initializer (optional)
    if (node.input_size() > 3 && !node.input(3).empty()) {
        const std::string& axes_name = node.input(3);
        auto axes_tensor = ctx.get_weight(axes_name);
        if (axes_tensor && axes_tensor->data()) {
            const int64_t* axes_data = static_cast<const int64_t*>(axes_tensor->data());
            int64_t num_axes = axes_tensor->shape().numel();
            param->axes.assign(axes_data, axes_data + num_axes);
        }
    }

    // Try to get steps from initializer (optional)
    if (node.input_size() > 4 && !node.input(4).empty()) {
        const std::string& steps_name = node.input(4);
        auto steps_tensor = ctx.get_weight(steps_name);
        if (steps_tensor && steps_tensor->data()) {
            const int64_t* steps_data = static_cast<const int64_t*>(steps_tensor->data());
            int64_t num_steps = steps_tensor->shape().numel();
            param->steps.assign(steps_data, steps_data + num_steps);
        }
    }

    // Check if we have static slice parameters (from initializers)
    bool has_static_params = !param->starts.empty() && !param->ends.empty();

    // Create operator with correct OpType
    auto op = std::make_shared<operators::GenericOperator>("Slice", core::OpType::kSLICE);
    op->set_plugin_param(param);

    const std::string& output_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(output_name);
    graph_node->set_operator(op);

    // Add input tensors
    // If static params: only data input (index 0)
    // If dynamic params: all inputs (data, starts, ends, [axes], [steps])
    std::vector<std::shared_ptr<core::Tensor>> input_tensors;

    int num_inputs = has_static_params ? 1 : node.input_size();
    for (int i = 0; i < num_inputs; ++i) {
        if (i > 0 && node.input(i).empty()) {
            // Optional input not provided, add nullptr placeholder
            input_tensors.push_back(nullptr);
            continue;
        }
        const std::string& input_name = node.input(i);
        auto input_tensor = ctx.get_tensor(input_name);
        if (!input_tensor) input_tensor = ctx.get_weight(input_name);
        if (!input_tensor) {
            input_tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, input_tensor);
        }
        input_tensors.push_back(input_tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    // Connect input ports
    for (int i = 0; i < num_inputs; ++i) {
        if (i > 0 && node.input(i).empty()) continue;
        const std::string& input_name = node.input(i);
        auto src_node = ctx.get_graph()->get_node(input_name);
        if (src_node) {
            ctx.get_graph()->connect(src_node->name(), output_name, 0, i);
        }
    }

    register_node_outputs(ctx, *graph_node, node, output_name);
    ctx.add_node(graph_node);

    return core::Status::SUCCESS;
}

// =============================================================================
// Additional BERT Operators (Placeholder Implementations)
// =============================================================================

core::Status ReduceMeanImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("ReduceMean operator");

    AttributeHelper attrs(node);
    auto axes = attrs.get_ints("axes");
    int64_t keepdims = attrs.get_int("keepdims", 1);

    auto op = std::make_shared<operators::GenericOperator>("ReduceMean", core::OpType::kREDUCE_MEAN);

    // Store axes and keepdims in a custom way - the plugin will read from param
    auto param = std::make_shared<operators::ReduceMeanParam>();
    param->axes = axes;
    param->keepdims = (keepdims != 0);
    // Note: GenericOperator uses PluginParam, we need to cast
    op->set_plugin_param(param);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

core::Status SubImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Sub operator");

    if (node.input_size() != 2) {
        ctx.set_error("Sub requires exactly 2 inputs");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Sub", core::OpType::kSUB);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

core::Status PowImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Pow operator");

    if (node.input_size() != 2) {
        ctx.set_error("Pow requires exactly 2 inputs");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Pow", core::OpType::kPOW);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

core::Status CastImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Cast operator");

    AttributeHelper attrs(node);
    int64_t to_dtype = attrs.get_int("to", 1);  // Default to FLOAT

    auto op = std::make_shared<operators::GenericOperator>("Cast", core::OpType::kCAST);

    auto param = std::make_shared<operators::CastParam>();
    param->to_dtype = static_cast<int32_t>(to_dtype);
    op->set_plugin_param(param);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

core::Status ErfImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Erf operator");

    auto op = std::make_shared<operators::GenericOperator>("Erf", core::OpType::kERF);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

core::Status TanhImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Tanh operator");

    auto op = std::make_shared<operators::GenericOperator>("Tanh", core::OpType::kTANH);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

core::Status EqualImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Equal operator");

    if (node.input_size() != 2) {
        ctx.set_error("Equal requires exactly 2 inputs");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Equal", core::OpType::kEQUAL);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

core::Status WhereImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Where operator");

    if (node.input_size() != 3) {
        ctx.set_error("Where requires exactly 3 inputs (condition, X, Y)");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Where", core::OpType::kWHERE);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

core::Status ExpandImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("Expand operator");

    if (node.input_size() != 2) {
        ctx.set_error("Expand requires exactly 2 inputs (input, shape)");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("Expand", core::OpType::kEXPAND);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    for (int i = 0; i < node.input_size(); ++i) {
        const std::string& input_name = node.input(i);
        auto tensor = ctx.get_tensor(input_name);
        if (!tensor) tensor = ctx.get_weight(input_name);
        if (!tensor) {
            tensor = std::make_shared<core::Tensor>();
            ctx.register_tensor(input_name, tensor);
        }
        input_tensors.push_back(tensor);
    }
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

core::Status ConstantOfShapeImporter::import_operator(ImporterContext& ctx, const onnx::NodeProto& node) {
    ctx.log_info("ConstantOfShape operator");

    if (node.input_size() != 1) {
        ctx.set_error("ConstantOfShape requires exactly 1 input (shape)");
        return core::Status::ERROR_INVALID_ARGUMENT;
    }

    auto op = std::make_shared<operators::GenericOperator>("ConstantOfShape", core::OpType::kCONSTANT_OF_SHAPE);

    const std::string& node_name = node.output(0);
    auto graph_node = ctx.get_graph()->create_node(node_name);
    graph_node->set_operator(op);

    std::vector<std::shared_ptr<core::Tensor>> input_tensors;
    const std::string& input_name = node.input(0);
    auto tensor = ctx.get_tensor(input_name);
    if (!tensor) tensor = ctx.get_weight(input_name);
    if (!tensor) {
        tensor = std::make_shared<core::Tensor>();
        ctx.register_tensor(input_name, tensor);
    }
    input_tensors.push_back(tensor);
    graph_node->set_input_tensors(input_tensors);

    connect_input_ports(ctx, node, node_name);
    register_node_outputs(ctx, *graph_node, node, node_name);

    ctx.add_node(graph_node);
    return core::Status::SUCCESS;
}

// =============================================================================
// All Builtin Operators
// =============================================================================

void register_builtin_operators(OperatorRegistry& registry) {
    // Convolution operators
    REGISTER_ONNX_OPERATOR("Conv", ConvImporter);

    // Linear algebra operators
    REGISTER_ONNX_OPERATOR("Gemm", GemmImporter);
    REGISTER_ONNX_OPERATOR("MatMul", MatMulImporter);

    // Activation operators
    REGISTER_ONNX_OPERATOR("Relu", ReluImporter);
    REGISTER_ONNX_OPERATOR("Gelu", GeluImporter);

    // Pooling operators
    REGISTER_ONNX_OPERATOR("MaxPool", MaxPoolImporter);
    REGISTER_ONNX_OPERATOR("AveragePool", AveragePoolImporter);
    REGISTER_ONNX_OPERATOR("GlobalAveragePool", GlobalAveragePoolImporter);

    // Normalization operators
    REGISTER_ONNX_OPERATOR("BatchNormalization", BatchNormalizationImporter);
    REGISTER_ONNX_OPERATOR("LayerNormalization", LayerNormalizationImporter);

    // Element-wise operators
    REGISTER_ONNX_OPERATOR("Add", AddImporter);
    REGISTER_ONNX_OPERATOR("Mul", MulImporter);
    REGISTER_ONNX_OPERATOR("Div", DivImporter);
    REGISTER_ONNX_OPERATOR("Sqrt", SqrtImporter);

    // Shape operators
    REGISTER_ONNX_OPERATOR("Reshape", ReshapeImporter);
    REGISTER_ONNX_OPERATOR("Flatten", FlattenImporter);
    REGISTER_ONNX_OPERATOR("Concat", ConcatImporter);
    REGISTER_ONNX_OPERATOR("Transpose", TransposeImporter);
    REGISTER_ONNX_OPERATOR("Squeeze", SqueezeImporter);
    REGISTER_ONNX_OPERATOR("Unsqueeze", UnsqueezeImporter);
    REGISTER_ONNX_OPERATOR("Gather", GatherImporter);

    // Other operators
    REGISTER_ONNX_OPERATOR("Softmax", SoftmaxImporter);
    REGISTER_ONNX_OPERATOR("Constant", ConstantImporter);
    REGISTER_ONNX_OPERATOR("Identity", IdentityImporter);
    REGISTER_ONNX_OPERATOR("Shape", ShapeImporter);
    REGISTER_ONNX_OPERATOR("Slice", SliceImporter);
    REGISTER_ONNX_OPERATOR("ReduceMean", ReduceMeanImporter);
    REGISTER_ONNX_OPERATOR("Sub", SubImporter);
    REGISTER_ONNX_OPERATOR("Pow", PowImporter);
    REGISTER_ONNX_OPERATOR("Cast", CastImporter);
    REGISTER_ONNX_OPERATOR("Erf", ErfImporter);
    REGISTER_ONNX_OPERATOR("Tanh", TanhImporter);
    REGISTER_ONNX_OPERATOR("Equal", EqualImporter);
    REGISTER_ONNX_OPERATOR("Where", WhereImporter);
    REGISTER_ONNX_OPERATOR("Expand", ExpandImporter);
    REGISTER_ONNX_OPERATOR("ConstantOfShape", ConstantOfShapeImporter);

    MI_LOG_INFO("[BuiltinOperators] Registered 37 builtin ONNX operators");
}

} // namespace importers
} // namespace mini_infer

#endif // MINI_INFER_ONNX_ENABLED
