#include "mini_infer/runtime/shape_tensor_evaluator.h"

#include <deque>
#include <unordered_set>

#include "mini_infer/operators/plugin_base.h"
#include "mini_infer/utils/logger.h"

namespace mini_infer {
namespace runtime {

void ShapeTensorEvaluator::initialize(const std::shared_ptr<graph::Graph>& graph) {
    graph_ = graph;
    shape_values_.clear();
    input_shapes_.clear();

    if (!graph_) {
        return;
    }

    // Resize shape_values_ to fit all node IDs
    shape_values_.resize(graph_->node_capacity());
}

void ShapeTensorEvaluator::seed_from_graph_constants() {
    int seeded_count = 0;

    // Reuse constants from Graph::constants_ (collected by ConstantFoldingPass)
    // This avoids duplicate traversal and collection
    for (const auto& [node_id, tensor] : graph_->constants()) {
        if (!tensor || !tensor->data()) {
            continue;
        }

        // Only seed integer tensors (shape tensors are typically int64/int32)
        if (tensor->dtype() != core::DataType::INT64 &&
            tensor->dtype() != core::DataType::INT32) {
            continue;
        }

        // Read values from tensor
        std::vector<int64_t> values = read_tensor_values(tensor);
        if (values.empty()) {
            continue;
        }

        // Store as shape value using node ID
        if (node_id < shape_values_.size()) {
            shape_values_[node_id] = ShapeValue(values, tensor->dtype());
            seeded_count++;
        }
    }

    if (seeded_count > 0) {
        MI_LOG_INFO("[ShapeTensorEvaluator] Seeded " + std::to_string(seeded_count) +
                    " shape value(s) from Graph::constants_");
    }
}

std::vector<int64_t> ShapeTensorEvaluator::read_tensor_values(
    const std::shared_ptr<core::Tensor>& tensor) const {

    if (!tensor || !tensor->data()) {
        return {};
    }

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

std::vector<std::vector<size_t>> ShapeTensorEvaluator::build_shape_consumers(
    const std::vector<std::shared_ptr<graph::Node>>& sorted_nodes,
    size_t capacity) const {
    std::vector<std::vector<size_t>> consumers(capacity);
    for (const auto& node : sorted_nodes) {
        if (!node || !node->get_operator()) {
            continue;
        }
        if (!is_shape_tensor_op(node->type())) {
            continue;
        }
        for (const auto& edge : node->inputs()) {
            if (!edge.node) {
                continue;
            }
            const size_t src_id = edge.node->id();
            if (src_id < capacity) {
                consumers[src_id].push_back(node->id());
            }
        }
    }
    return consumers;
}

int ShapeTensorEvaluator::count_unmet_inputs(
    const std::shared_ptr<graph::Node>& node) const {
    if (!node) {
        return 1;
    }
    const core::OpType op_type = node->type();
    if (op_type == core::OpType::kSHAPE) {
        return get_input_shape(node, 0).ndim() > 0 ? 0 : 1;
    }

    const auto& input_edges = node->inputs();
    const auto& imported_tensors = node->input_tensors();

    int max_port = -1;
    for (const auto& edge : input_edges) {
        max_port = std::max(max_port, edge.dst_port);
    }
    const size_t num_inputs = std::max(
        max_port >= 0 ? static_cast<size_t>(max_port + 1) : 0,
        imported_tensors.size());

    int unmet = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
        const auto* sv = get_input_shape_value(node, static_cast<int>(i));
        if (sv) {
            continue;
        }
        if (i < imported_tensors.size() && imported_tensors[i] &&
            imported_tensors[i]->data()) {
            continue;
        }
        unmet++;
    }
    return unmet;
}

void ShapeTensorEvaluator::seed_ready_queue(
    const std::vector<std::shared_ptr<graph::Node>>& sorted_nodes,
    size_t capacity,
    std::vector<int>& pending_inputs,
    std::deque<size_t>& ready) const {
    for (const auto& node : sorted_nodes) {
        if (!node || !node->get_operator()) {
            continue;
        }
        if (!is_shape_tensor_op(node->type())) {
            continue;
        }
        const size_t node_id = node->id();
        if (node_id >= capacity) {
            continue;
        }
        if (node_id < shape_values_.size() && shape_values_[node_id].is_valid) {
            pending_inputs[node_id] = 0;
            continue;
        }
        const int unmet = count_unmet_inputs(node);
        pending_inputs[node_id] = unmet;
        if (unmet == 0) {
            ready.push_back(node_id);
        }
    }
}

core::Status ShapeTensorEvaluator::evaluate(
    const std::unordered_map<size_t, core::Shape>& input_shapes) {

    input_shapes_ = input_shapes;

    int total_evaluated = 0;

    const auto& sorted_nodes = graph_->get_sorted_nodes();
    const size_t capacity = graph_->node_capacity();
    std::vector<int> pending_inputs(capacity, -1);

    // Build and cache consumers graph for incremental updates
    consumers_ = build_shape_consumers(sorted_nodes, capacity);

    std::deque<size_t> ready;
    seed_ready_queue(sorted_nodes, capacity, pending_inputs, ready);

    while (!ready.empty()) {
        const size_t node_id = ready.front();
        ready.pop_front();

        if (node_id >= shape_values_.size() || shape_values_[node_id].is_valid) {
            continue;
        }
        auto node = graph_->get_node(node_id);
        if (!node || !node->get_operator()) {
            continue;
        }
        if (!can_evaluate(node)) {
            continue;
        }

        if (!evaluate_node(node)) {
            continue;
        }
        total_evaluated++;

        // Notify downstream shape ops that depend on this node.
        if (node_id < consumers_.size()) {
            for (const auto& consumer_id : consumers_[node_id]) {
                if (consumer_id >= pending_inputs.size()) {
                    continue;
                }
                if (consumer_id < shape_values_.size() &&
                    shape_values_[consumer_id].is_valid) {
                    continue;
                }
                if (pending_inputs[consumer_id] > 0) {
                    pending_inputs[consumer_id]--;
                    if (pending_inputs[consumer_id] == 0) {
                        ready.push_back(consumer_id);
                    }
                }
            }
        }
    }

    MI_LOG_DEBUG("[ShapeTensorEvaluator] Evaluated " + std::to_string(total_evaluated) +
                " shape tensor(s)");

    return core::Status::SUCCESS;
}

core::Status ShapeTensorEvaluator::incremental_evaluate(
    size_t node_id, const core::Shape& shape) {

    // Update input_shapes_ map
    if (node_id < input_shapes_.size() || input_shapes_.find(node_id) != input_shapes_.end()) {
        input_shapes_[node_id] = shape;
    } else {
        input_shapes_[node_id] = shape;
    }

    // If consumers graph not built yet, nothing to do
    if (consumers_.empty()) {
        return core::Status::SUCCESS;
    }

    // Find all shape tensor nodes that depend on this node
    if (node_id >= consumers_.size()) {
        return core::Status::SUCCESS;
    }

    std::deque<size_t> ready;
    std::unordered_set<size_t> visited;

    // Check immediate consumers
    for (size_t consumer_id : consumers_[node_id]) {
        auto node = graph_->get_node(consumer_id);
        if (!node || !node->get_operator()) {
            continue;
        }
        if (!is_shape_tensor_op(node->type())) {
            continue;
        }
        // Skip if already evaluated
        if (consumer_id < shape_values_.size() && shape_values_[consumer_id].is_valid) {
            continue;
        }
        // Check if all inputs are ready
        if (count_unmet_inputs(node) == 0) {
            ready.push_back(consumer_id);
            visited.insert(consumer_id);
        }
    }

    int total_evaluated = 0;

    // Topological evaluation of affected nodes
    while (!ready.empty()) {
        const size_t current_id = ready.front();
        ready.pop_front();

        auto node = graph_->get_node(current_id);
        if (!node || !node->get_operator()) {
            continue;
        }

        if (!can_evaluate(node)) {
            continue;
        }

        if (!evaluate_node(node)) {
            continue;
        }
        total_evaluated++;

        // Notify downstream consumers
        if (current_id < consumers_.size()) {
            for (size_t downstream_id : consumers_[current_id]) {
                if (visited.count(downstream_id)) {
                    continue;  // Already in queue
                }
                if (downstream_id < shape_values_.size() &&
                    shape_values_[downstream_id].is_valid) {
                    continue;  // Already evaluated
                }

                auto downstream_node = graph_->get_node(downstream_id);
                if (!downstream_node || !downstream_node->get_operator()) {
                    continue;
                }
                if (!is_shape_tensor_op(downstream_node->type())) {
                    continue;
                }

                if (count_unmet_inputs(downstream_node) == 0) {
                    ready.push_back(downstream_id);
                    visited.insert(downstream_id);
                }
            }
        }
    }

    if (total_evaluated > 0) {
        MI_LOG_DEBUG("[ShapeTensorEvaluator] Incrementally evaluated " +
                    std::to_string(total_evaluated) + " shape tensor(s)");
    }

    return core::Status::SUCCESS;
}

const ShapeTensorEvaluator::ShapeValue* ShapeTensorEvaluator::get_shape_value(
    size_t node_id) const {

    if (node_id < shape_values_.size() && shape_values_[node_id].is_valid) {
        return &shape_values_[node_id];
    }
    return nullptr;
}

bool ShapeTensorEvaluator::has_shape_value(size_t node_id) const {
    return node_id < shape_values_.size() && shape_values_[node_id].is_valid;
}

void ShapeTensorEvaluator::clear() {
    shape_values_.clear();
    input_shapes_.clear();
}

bool ShapeTensorEvaluator::is_shape_tensor_op(core::OpType op_type) {
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
                    if (i < imported_tensors.size() && imported_tensors[i] &&
                        imported_tensors[i]->data()) {
                        continue;  // Has constant data, OK
                    }
                    return false;
                }
            }
            return true;
        }

        case core::OpType::kSLICE: {
            // Slice needs: data shape known, starts/ends available
            auto input_shape = get_input_shape(node, 0);
            if (input_shape.ndim() == 0) {
                return false;
            }
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
        if (node->id() < shape_values_.size()) {
            shape_values_[node->id()] = result;
        }
        MI_LOG_DEBUG("[ShapeTensorEvaluator] Evaluated node '" + node->name() +
                    "' (id=" + std::to_string(node->id()) +
                    ") with " + std::to_string(result.data.size()) + " values");
        return true;
    }

    return false;
}

const ShapeTensorEvaluator::ShapeValue* ShapeTensorEvaluator::get_input_shape_value(
    const std::shared_ptr<graph::Node>& node, int port) const {

    // Check graph edges first
    const auto& input_edges = node->inputs();
    for (const auto& edge : input_edges) {
        if (edge.dst_port == port && edge.node) {
            return get_shape_value(edge.node->id());
        }
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
                auto it = input_shapes_.find(edge.node->id());
                if (it != input_shapes_.end()) {
                    return it->second;
                }
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

// Helper to read values from shape_value or imported tensor
std::vector<int64_t> read_input_values(
    const ShapeTensorEvaluator::ShapeValue* sv,
    const std::shared_ptr<graph::Node>& node,
    int port) {

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

    std::vector<int64_t> data_values = read_input_values(data_sv, node, 0);
    std::vector<int64_t> indices_values = read_input_values(indices_sv, node, 1);

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
    std::vector<int64_t> input_values = read_input_values(input_sv, node, 0);

    if (input_values.empty()) {
        return ShapeValue();
    }

    // Unsqueeze just adds a dimension, values stay the same
    return ShapeValue(input_values, core::DataType::INT64);
}

ShapeTensorEvaluator::ShapeValue ShapeTensorEvaluator::eval_squeeze_op(
    const std::shared_ptr<graph::Node>& node) {

    auto* input_sv = get_input_shape_value(node, 0);
    std::vector<int64_t> input_values = read_input_values(input_sv, node, 0);

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
        std::vector<int64_t> values = read_input_values(sv, node, static_cast<int>(i));
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
    std::vector<int64_t> input_values = read_input_values(input_sv, node, 0);

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
    std::vector<int64_t> data_values = read_input_values(data_sv, node, 0);

    if (data_values.empty()) {
        return ShapeValue();
    }

    // Get slice parameters
    auto* starts_sv = get_input_shape_value(node, 1);
    auto* ends_sv = get_input_shape_value(node, 2);
    auto* axes_sv = get_input_shape_value(node, 3);
    auto* steps_sv = get_input_shape_value(node, 4);

    std::vector<int64_t> starts = read_input_values(starts_sv, node, 1);
    std::vector<int64_t> ends = read_input_values(ends_sv, node, 2);
    std::vector<int64_t> axes = read_input_values(axes_sv, node, 3);
    std::vector<int64_t> steps = read_input_values(steps_sv, node, 4);

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
    std::vector<int64_t> shape_values = read_input_values(shape_sv, node, 0);

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

    std::vector<int64_t> a_values = read_input_values(a_sv, node, 0);
    std::vector<int64_t> b_values = read_input_values(b_sv, node, 1);

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

    std::vector<int64_t> a_values = read_input_values(a_sv, node, 0);
    std::vector<int64_t> b_values = read_input_values(b_sv, node, 1);

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

    std::vector<int64_t> a_values = read_input_values(a_sv, node, 0);
    std::vector<int64_t> b_values = read_input_values(b_sv, node, 1);

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

    std::vector<int64_t> condition_values = read_input_values(condition_sv, node, 0);
    std::vector<int64_t> x_values = read_input_values(x_sv, node, 1);
    std::vector<int64_t> y_values = read_input_values(y_sv, node, 2);

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
    std::vector<int64_t> data_values = read_input_values(data_sv, node, 0);

    if (data_values.empty()) {
        return ShapeValue();
    }

    // Reshape just reorganizes the data, values stay the same
    // For shape tensors, we just return the data as-is
    return ShapeValue(data_values, core::DataType::INT64);
}

}  // namespace runtime
}  // namespace mini_infer
