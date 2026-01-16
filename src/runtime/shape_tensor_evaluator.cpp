#include "mini_infer/runtime/shape_tensor_evaluator.h"
#include "mini_infer/utils/logger.h"
#include "mini_infer/operators/plugin_base.h"

namespace mini_infer {
namespace runtime {

void ShapeTensorEvaluator::initialize(const std::shared_ptr<graph::Graph>& graph) {
    graph_ = graph;
    sorted_nodes_.clear();
    shape_values_.clear();
    shape_tensor_names_.clear();

    if (!graph_) {
        return;
    }

    // Get topologically sorted nodes
    graph_->topological_sort(sorted_nodes_);

    // Resize shape_values_by_id_ to fit all node IDs
    shape_values_by_id_.clear();
    shape_values_by_id_.resize(graph_->node_capacity());
}

void ShapeTensorEvaluator::seed_from_initializers() {
    if (!graph_) {
        return;
    }

    // Iterate through all nodes and seed from their imported tensors (constants)
    for (const auto& node : sorted_nodes_) {
        if (!node) continue;

        const auto& imported_tensors = node->input_tensors();
        for (size_t i = 0; i < imported_tensors.size(); ++i) {
            const auto& tensor = imported_tensors[i];
            if (!tensor || !tensor->data()) continue;

            // Only seed integer tensors (shape tensors are typically int64/int32)
            if (tensor->dtype() != core::DataType::INT64 &&
                tensor->dtype() != core::DataType::INT32) {
                continue;
            }

            // Read values from tensor
            std::vector<int64_t> values;
            const int64_t numel = tensor->shape().numel();

            if (tensor->dtype() == core::DataType::INT64) {
                const int64_t* data = static_cast<const int64_t*>(tensor->data());
                values.assign(data, data + numel);
            } else if (tensor->dtype() == core::DataType::INT32) {
                const int32_t* data = static_cast<const int32_t*>(tensor->data());
                values.reserve(numel);
                for (int64_t j = 0; j < numel; ++j) {
                    values.push_back(static_cast<int64_t>(data[j]));
                }
            }

            // Store as shape value using a unique key for this input
            // Format: node_name:input_port
            std::string key = node->name() + ":input" + std::to_string(i);
            shape_values_[key] = ShapeValue(values, tensor->dtype());
        }
    }
}

void ShapeTensorEvaluator::set_known_output_shapes(
    const std::vector<std::vector<core::Shape>>* shapes_by_id) {
    known_output_shapes_by_id_ = shapes_by_id;
}

core::Status ShapeTensorEvaluator::evaluate(
    const std::unordered_map<std::string, core::Shape>& input_shapes) {

    input_shapes_ = input_shapes;

    // Store input shapes as shape values for input nodes
    for (const auto& [name, shape] : input_shapes) {
        auto node = graph_->get_node(name);
        if (node) {
            // Input nodes don't produce shape values directly,
            // but their shapes are available for Shape op
            if (node->id() < shape_values_by_id_.size()) {
                // Mark that this node's shape is known
                shape_values_by_id_[node->id()].is_valid = false;  // Not a shape tensor
            }
        }
    }

    int total_evaluated = 0;

    // Iteratively evaluate shape tensors until no more progress
    // This handles dependency chains like Shape -> Gather -> Unsqueeze -> Concat
    bool made_progress = true;
    int iteration = 0;
    const int max_iterations = 100;  // Prevent infinite loops

    while (made_progress && iteration < max_iterations) {
        made_progress = false;
        iteration++;

        // Evaluate shape tensors in topological order
        for (const auto& node : sorted_nodes_) {
            if (!node || !node->get_operator()) {
                continue;
            }

            // Skip if already evaluated
            if (node->id() < shape_values_by_id_.size() &&
                shape_values_by_id_[node->id()].is_valid) {
                continue;
            }

            core::OpType op_type = node->type();

            // Check if this is a shape-tensor-producing op
            if (!is_shape_tensor_op(op_type)) {
                continue;
            }

            // Check if we can evaluate this node
            if (!can_evaluate(node)) {
                continue;
            }

            // Evaluate and store result
            if (evaluate_node(node)) {
                total_evaluated++;
                shape_tensor_names_.insert(node->name());
                made_progress = true;
            }
        }
    }

    MI_LOG_INFO("[ShapeTensorEvaluator] Evaluated " + std::to_string(total_evaluated) +
                " shape tensor(s) in " + std::to_string(iteration) + " iteration(s)");

    return core::Status::SUCCESS;
}

const ShapeTensorEvaluator::ShapeValue* ShapeTensorEvaluator::get_shape_value(
    const std::string& tensor_name) const {

    auto it = shape_values_.find(tensor_name);
    if (it != shape_values_.end() && it->second.is_valid) {
        return &it->second;
    }
    return nullptr;
}

const ShapeTensorEvaluator::ShapeValue* ShapeTensorEvaluator::get_shape_value(
    size_t node_id) const {

    if (node_id < shape_values_by_id_.size() && shape_values_by_id_[node_id].is_valid) {
        return &shape_values_by_id_[node_id];
    }
    return nullptr;
}

bool ShapeTensorEvaluator::is_shape_tensor(const std::string& tensor_name) const {
    return shape_tensor_names_.count(tensor_name) > 0;
}

void ShapeTensorEvaluator::clear() {
    shape_values_.clear();
    shape_values_by_id_.clear();
    shape_tensor_names_.clear();
    input_shapes_.clear();
}

bool ShapeTensorEvaluator::is_shape_tensor_op(core::OpType op_type) const {
    switch (op_type) {
        case core::OpType::kSHAPE:
        case core::OpType::kGATHER:
        case core::OpType::kUNSQUEEZE:
        case core::OpType::kSQUEEZE:
        case core::OpType::kCONCAT:
        case core::OpType::kCAST:
        case core::OpType::kSLICE:
        case core::OpType::kCONSTANT_OF_SHAPE:
        case core::OpType::kMUL:
        case core::OpType::kADD:
        case core::OpType::kEQUAL:
        case core::OpType::kWHERE:
        case core::OpType::kRESHAPE:
            return true;
        default:
            return false;
    }
}

bool ShapeTensorEvaluator::can_evaluate(const std::shared_ptr<graph::Node>& node) const {
    core::OpType op_type = node->type();

    switch (op_type) {
        case core::OpType::kSHAPE: {
            // Shape op can be evaluated if input shape is known
            auto input_shape = get_input_shape(node, 0);
            return input_shape.ndim() > 0;
        }

        case core::OpType::kGATHER:
        case core::OpType::kUNSQUEEZE:
        case core::OpType::kSQUEEZE:
        case core::OpType::kCONCAT:
        case core::OpType::kCAST:
        case core::OpType::kMUL:
        case core::OpType::kADD:
        case core::OpType::kEQUAL:
        case core::OpType::kWHERE:
        case core::OpType::kRESHAPE: {
            // These ops need all input shape-values to be known
            const auto& input_edges = node->inputs();
            const auto& imported_tensors = node->input_tensors();

            // Determine number of inputs
            int max_port = -1;
            for (const auto& edge : input_edges) {
                max_port = std::max(max_port, edge.dst_port);
            }
            size_t num_inputs = std::max(
                max_port >= 0 ? static_cast<size_t>(max_port + 1) : 0,
                imported_tensors.size());

            // Check each input
            for (size_t i = 0; i < num_inputs; ++i) {
                // Check if we have a shape value for this input
                auto* sv = get_input_shape_value(node, static_cast<int>(i));
                if (!sv) {
                    // Check if it's from an imported tensor (constant)
                    std::string key = node->name() + ":input" + std::to_string(i);
                    if (shape_values_.find(key) == shape_values_.end()) {
                        // Also check if it's from a graph edge
                        bool found = false;
                        for (const auto& edge : input_edges) {
                            if (edge.dst_port == static_cast<int>(i) && edge.node) {
                                auto* edge_sv = get_shape_value(edge.node->id());
                                if (edge_sv) {
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (!found && i < imported_tensors.size() && imported_tensors[i] &&
                            imported_tensors[i]->data()) {
                            found = true;  // Has constant data
                        }
                        if (!found) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        case core::OpType::kSLICE: {
            // Slice needs: data shape known, starts/ends/axes/steps shape-values known
            auto input_shape = get_input_shape(node, 0);
            if (input_shape.ndim() == 0) {
                return false;
            }

            // Check starts (input 1) and ends (input 2) are available
            auto* starts = get_input_shape_value(node, 1);
            auto* ends = get_input_shape_value(node, 2);

            // If not from graph edges, check imported tensors
            const auto& imported_tensors = node->input_tensors();
            if (!starts && imported_tensors.size() > 1 && imported_tensors[1] &&
                imported_tensors[1]->data()) {
                starts = nullptr;  // Will read from tensor directly
            }
            if (!ends && imported_tensors.size() > 2 && imported_tensors[2] &&
                imported_tensors[2]->data()) {
                ends = nullptr;  // Will read from tensor directly
            }

            // For Slice, we need at least starts and ends to be computable
            // They can come from either shape_values or imported_tensors
            return true;  // We'll handle missing values in eval_slice_op
        }

        case core::OpType::kCONSTANT_OF_SHAPE: {
            // ConstantOfShape needs the shape input to be known
            auto* shape_input = get_input_shape_value(node, 0);
            if (shape_input) return true;

            // Check imported tensor
            const auto& imported_tensors = node->input_tensors();
            if (!imported_tensors.empty() && imported_tensors[0] &&
                imported_tensors[0]->data()) {
                return true;
            }
            return false;
        }

        default:
            return false;
    }
}

bool ShapeTensorEvaluator::evaluate_node(const std::shared_ptr<graph::Node>& node) {
    ShapeValue result;

    switch (node->type()) {
        case core::OpType::kSHAPE:
            result = eval_shape_op(node);
            break;
        case core::OpType::kGATHER:
            result = eval_gather_op(node);
            break;
        case core::OpType::kUNSQUEEZE:
            result = eval_unsqueeze_op(node);
            break;
        case core::OpType::kSQUEEZE:
            result = eval_squeeze_op(node);
            break;
        case core::OpType::kCONCAT:
            result = eval_concat_op(node);
            break;
        case core::OpType::kCAST:
            result = eval_cast_op(node);
            break;
        case core::OpType::kSLICE:
            result = eval_slice_op(node);
            break;
        case core::OpType::kCONSTANT_OF_SHAPE:
            result = eval_constant_of_shape_op(node);
            break;
        case core::OpType::kMUL:
            result = eval_mul_op(node);
            break;
        case core::OpType::kADD:
            result = eval_add_op(node);
            break;
        case core::OpType::kEQUAL:
            result = eval_equal_op(node);
            break;
        case core::OpType::kWHERE:
            result = eval_where_op(node);
            break;
        case core::OpType::kRESHAPE:
            result = eval_reshape_op(node);
            break;
        default:
            return false;
    }

    if (result.is_valid) {
        shape_values_[node->name()] = result;
        if (node->id() < shape_values_by_id_.size()) {
            shape_values_by_id_[node->id()] = result;
        }
        MI_LOG_INFO("[ShapeTensorEvaluator] Evaluated node '" + node->name() +
                    "' (id=" + std::to_string(node->id()) +
                    ") with " + std::to_string(result.data.size()) + " values");
        return true;
    }

    MI_LOG_INFO("[ShapeTensorEvaluator] Failed to evaluate node '" + node->name() +
                "' (id=" + std::to_string(node->id()) + ")");
    return false;
}

const ShapeTensorEvaluator::ShapeValue* ShapeTensorEvaluator::get_input_shape_value(
    const std::shared_ptr<graph::Node>& node, int port) const {

    // First check graph edges
    const auto& input_edges = node->inputs();
    for (const auto& edge : input_edges) {
        if (edge.dst_port == port && edge.node) {
            auto* sv = get_shape_value(edge.node->id());
            if (sv) return sv;
        }
    }

    // Check imported tensor key
    std::string key = node->name() + ":input" + std::to_string(port);
    auto it = shape_values_.find(key);
    if (it != shape_values_.end() && it->second.is_valid) {
        return &it->second;
    }

    return nullptr;
}

core::Shape ShapeTensorEvaluator::get_input_shape(
    const std::shared_ptr<graph::Node>& node, int port) const {

    // Check graph edges
    const auto& input_edges = node->inputs();
    for (const auto& edge : input_edges) {
        if (edge.dst_port == port && edge.node) {
            // Check if it's an input node
            if (graph_->is_input(edge.node->name())) {
                auto it = input_shapes_.find(edge.node->name());
                if (it != input_shapes_.end()) {
                    return it->second;
                }
            }
            // Prefer known runtime shapes if provided
            if (known_output_shapes_by_id_) {
                size_t node_id = edge.node->id();
                const size_t src_index = static_cast<size_t>(edge.src_port);
                if (node_id < known_output_shapes_by_id_->size()) {
                    const auto& shapes = (*known_output_shapes_by_id_)[node_id];
                    if (src_index < shapes.size() && shapes[src_index].ndim() > 0) {
                        return shapes[src_index];
                    }
                }
                return core::Shape();
            }
            // Check output tensors of the source node
            const auto& outputs = edge.node->output_tensors();
            if (static_cast<size_t>(edge.src_port) < outputs.size() && outputs[edge.src_port]) {
                return outputs[edge.src_port]->shape();
            }
        }
    }

    // Check imported tensors
    const auto& imported_tensors = node->input_tensors();
    if (static_cast<size_t>(port) < imported_tensors.size() && imported_tensors[port]) {
        return imported_tensors[port]->shape();
    }

    return core::Shape();
}

// Helper to read values from imported tensor or shape_value
std::vector<int64_t> read_input_values(
    const std::shared_ptr<graph::Node>& node,
    int port,
    const ShapeTensorEvaluator::ShapeValue* sv) {

    if (sv && sv->is_valid) {
        return sv->data;
    }

    // Read from imported tensor
    const auto& imported_tensors = node->input_tensors();
    if (static_cast<size_t>(port) < imported_tensors.size()) {
        const auto& tensor = imported_tensors[port];
        if (tensor && tensor->data()) {
            std::vector<int64_t> values;
            const int64_t numel = tensor->shape().numel();

            if (tensor->dtype() == core::DataType::INT64) {
                const int64_t* data = static_cast<const int64_t*>(tensor->data());
                values.assign(data, data + numel);
            } else if (tensor->dtype() == core::DataType::INT32) {
                const int32_t* data = static_cast<const int32_t*>(tensor->data());
                values.reserve(numel);
                for (int64_t i = 0; i < numel; ++i) {
                    values.push_back(static_cast<int64_t>(data[i]));
                }
            }
            return values;
        }
    }

    return {};
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_shape_op(
    const std::shared_ptr<graph::Node>& node) {

    auto input_shape = get_input_shape(node, 0);
    if (input_shape.ndim() == 0) {
        return ShapeValue();
    }

    return ShapeValue(input_shape.dims(), core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_gather_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* data_sv = get_input_shape_value(node, 0);
    auto* indices_sv = get_input_shape_value(node, 1);

    std::vector<int64_t> data_values = read_input_values(node, 0, data_sv);
    std::vector<int64_t> indices_values = read_input_values(node, 1, indices_sv);

    if (data_values.empty() || indices_values.empty()) {
        return ShapeValue();
    }

    // Get axis from operator params
    int64_t axis = 0;
    auto op = node->get_operator();
    if (op && op->param()) {
        auto* gather_param = dynamic_cast<const operators::GatherParam*>(op->param().get());
        if (gather_param) {
            axis = gather_param->axis;
        }
    }

    // For 1D data (shape tensor), gather is simple indexing
    std::vector<int64_t> result;
    result.reserve(indices_values.size());

    for (int64_t idx : indices_values) {
        if (idx < 0) idx += static_cast<int64_t>(data_values.size());
        if (idx >= 0 && idx < static_cast<int64_t>(data_values.size())) {
            result.push_back(data_values[idx]);
        }
    }

    return ShapeValue(result, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_unsqueeze_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* input_sv = get_input_shape_value(node, 0);
    std::vector<int64_t> input_values = read_input_values(node, 0, input_sv);

    if (input_values.empty()) {
        return ShapeValue();
    }

    // Unsqueeze just adds a dimension, values stay the same
    return ShapeValue(input_values, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_squeeze_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* input_sv = get_input_shape_value(node, 0);
    std::vector<int64_t> input_values = read_input_values(node, 0, input_sv);

    if (input_values.empty()) {
        return ShapeValue();
    }

    // Squeeze removes dimensions, values stay the same
    return ShapeValue(input_values, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_concat_op(
    const std::shared_ptr<graph::Node>& node) {

    std::vector<int64_t> result;

    // Determine number of inputs
    const auto& input_edges = node->inputs();
    const auto& imported_tensors = node->input_tensors();

    int max_port = -1;
    for (const auto& edge : input_edges) {
        max_port = std::max(max_port, edge.dst_port);
    }
    size_t num_inputs = std::max(
        max_port >= 0 ? static_cast<size_t>(max_port + 1) : 0,
        imported_tensors.size());

    // Concatenate all inputs
    for (size_t i = 0; i < num_inputs; ++i) {
        auto* sv = get_input_shape_value(node, static_cast<int>(i));
        std::vector<int64_t> values = read_input_values(node, static_cast<int>(i), sv);

        if (values.empty()) {
            // Try to get from imported tensor directly
            if (i < imported_tensors.size() && imported_tensors[i] &&
                imported_tensors[i]->data()) {
                const auto& tensor = imported_tensors[i];
                const int64_t numel = tensor->shape().numel();

                if (tensor->dtype() == core::DataType::INT64) {
                    const int64_t* data = static_cast<const int64_t*>(tensor->data());
                    values.assign(data, data + numel);
                } else if (tensor->dtype() == core::DataType::INT32) {
                    const int32_t* data = static_cast<const int32_t*>(tensor->data());
                    for (int64_t j = 0; j < numel; ++j) {
                        values.push_back(static_cast<int64_t>(data[j]));
                    }
                }
            }
        }

        result.insert(result.end(), values.begin(), values.end());
    }

    if (result.empty()) {
        return ShapeValue();
    }

    return ShapeValue(result, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_cast_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* input_sv = get_input_shape_value(node, 0);
    std::vector<int64_t> input_values = read_input_values(node, 0, input_sv);

    if (input_values.empty()) {
        return ShapeValue();
    }

    // Get target dtype from params
    core::DataType target_dtype = core::DataType::INT64;
    auto op = node->get_operator();
    if (op && op->param()) {
        auto* cast_param = dynamic_cast<const operators::CastParam*>(op->param().get());
        if (cast_param) {
            target_dtype = cast_param->to_dtype;
        }
    }

    // For shape tensors, cast doesn't change values (just dtype)
    return ShapeValue(input_values, target_dtype);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_slice_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* data_sv = get_input_shape_value(node, 0);
    std::vector<int64_t> data_values = read_input_values(node, 0, data_sv);

    if (data_values.empty()) {
        return ShapeValue();
    }

    // Get slice parameters
    auto* starts_sv = get_input_shape_value(node, 1);
    auto* ends_sv = get_input_shape_value(node, 2);
    auto* axes_sv = get_input_shape_value(node, 3);
    auto* steps_sv = get_input_shape_value(node, 4);

    std::vector<int64_t> starts = read_input_values(node, 1, starts_sv);
    std::vector<int64_t> ends = read_input_values(node, 2, ends_sv);
    std::vector<int64_t> axes = read_input_values(node, 3, axes_sv);
    std::vector<int64_t> steps = read_input_values(node, 4, steps_sv);

    if (starts.empty() || ends.empty()) {
        return ShapeValue();
    }

    // Default axes and steps
    if (axes.empty()) {
        axes.resize(starts.size());
        for (size_t i = 0; i < starts.size(); ++i) {
            axes[i] = static_cast<int64_t>(i);
        }
    }
    if (steps.empty()) {
        steps.resize(starts.size(), 1);
    }

    // For 1D shape tensor, perform slice
    int64_t dim_size = static_cast<int64_t>(data_values.size());
    int64_t start = starts[0];
    int64_t end = ends[0];
    int64_t step = steps.empty() ? 1 : steps[0];

    // Normalize
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;
    start = std::max(int64_t(0), std::min(start, dim_size));
    end = std::max(int64_t(0), std::min(end, dim_size));

    std::vector<int64_t> result;
    if (step > 0) {
        for (int64_t i = start; i < end; i += step) {
            result.push_back(data_values[i]);
        }
    } else if (step < 0) {
        for (int64_t i = start; i > end; i += step) {
            result.push_back(data_values[i]);
        }
    }

    return ShapeValue(result, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_constant_of_shape_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* shape_sv = get_input_shape_value(node, 0);
    std::vector<int64_t> shape_values = read_input_values(node, 0, shape_sv);

    if (shape_values.empty()) {
        return ShapeValue();
    }

    // Get the constant value from params
    int64_t fill_value = 1;  // Default
    auto op = node->get_operator();
    if (op && op->param()) {
        auto* cos_param = dynamic_cast<const operators::ConstantOfShapeParam*>(op->param().get());
        if (cos_param) {
            fill_value = static_cast<int64_t>(cos_param->value);
        }
    }

    // Create output filled with the constant value
    int64_t total = 1;
    for (int64_t dim : shape_values) {
        total *= dim;
    }

    std::vector<int64_t> result(total, fill_value);
    return ShapeValue(result, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_mul_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* a_sv = get_input_shape_value(node, 0);
    auto* b_sv = get_input_shape_value(node, 1);

    std::vector<int64_t> a_values = read_input_values(node, 0, a_sv);
    std::vector<int64_t> b_values = read_input_values(node, 1, b_sv);

    if (a_values.empty() || b_values.empty()) {
        return ShapeValue();
    }

    // Element-wise multiplication with broadcasting
    size_t out_size = std::max(a_values.size(), b_values.size());
    std::vector<int64_t> result(out_size);

    for (size_t i = 0; i < out_size; ++i) {
        int64_t a_val = a_values[i % a_values.size()];
        int64_t b_val = b_values[i % b_values.size()];
        result[i] = a_val * b_val;
    }

    return ShapeValue(result, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_add_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* a_sv = get_input_shape_value(node, 0);
    auto* b_sv = get_input_shape_value(node, 1);

    std::vector<int64_t> a_values = read_input_values(node, 0, a_sv);
    std::vector<int64_t> b_values = read_input_values(node, 1, b_sv);

    if (a_values.empty() || b_values.empty()) {
        return ShapeValue();
    }

    // Element-wise addition with broadcasting
    size_t out_size = std::max(a_values.size(), b_values.size());
    std::vector<int64_t> result(out_size);

    for (size_t i = 0; i < out_size; ++i) {
        int64_t a_val = a_values[i % a_values.size()];
        int64_t b_val = b_values[i % b_values.size()];
        result[i] = a_val + b_val;
    }

    return ShapeValue(result, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_equal_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* a_sv = get_input_shape_value(node, 0);
    auto* b_sv = get_input_shape_value(node, 1);

    std::vector<int64_t> a_values = read_input_values(node, 0, a_sv);
    std::vector<int64_t> b_values = read_input_values(node, 1, b_sv);

    if (a_values.empty() || b_values.empty()) {
        return ShapeValue();
    }

    // Element-wise equality comparison with broadcasting
    size_t out_size = std::max(a_values.size(), b_values.size());
    std::vector<int64_t> result(out_size);

    for (size_t i = 0; i < out_size; ++i) {
        int64_t a_val = a_values[i % a_values.size()];
        int64_t b_val = b_values[i % b_values.size()];
        result[i] = (a_val == b_val) ? 1 : 0;  // Boolean result as int64
    }

    return ShapeValue(result, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_where_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* condition_sv = get_input_shape_value(node, 0);
    auto* x_sv = get_input_shape_value(node, 1);
    auto* y_sv = get_input_shape_value(node, 2);

    std::vector<int64_t> condition_values = read_input_values(node, 0, condition_sv);
    std::vector<int64_t> x_values = read_input_values(node, 1, x_sv);
    std::vector<int64_t> y_values = read_input_values(node, 2, y_sv);

    if (condition_values.empty() || x_values.empty() || y_values.empty()) {
        return ShapeValue();
    }

    // Element-wise selection with broadcasting
    size_t out_size = std::max({condition_values.size(), x_values.size(), y_values.size()});
    std::vector<int64_t> result(out_size);

    for (size_t i = 0; i < out_size; ++i) {
        int64_t cond = condition_values[i % condition_values.size()];
        int64_t x_val = x_values[i % x_values.size()];
        int64_t y_val = y_values[i % y_values.size()];
        result[i] = (cond != 0) ? x_val : y_val;
    }

    return ShapeValue(result, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_reshape_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* data_sv = get_input_shape_value(node, 0);
    auto* shape_sv = get_input_shape_value(node, 1);

    std::vector<int64_t> data_values = read_input_values(node, 0, data_sv);
    std::vector<int64_t> target_shape = read_input_values(node, 1, shape_sv);

    if (data_values.empty() || target_shape.empty()) {
        return ShapeValue();
    }

    // Reshape just reorganizes the data, values stay the same
    // For shape tensors, we just return the data as-is
    return ShapeValue(data_values, core::DataType::INT64);
}

}  // namespace runtime
}  // namespace mini_infer
