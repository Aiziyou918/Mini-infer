/**
 * @file lenet5_optimized_with_memory_planning.cpp
 * @brief LeNet-5 optimized inference with graph optimization and memory planning
 *
 * Features:
 * - Graph optimization (operator fusion)
 * - Static memory planning (TensorRT-style)
 * - Memory usage comparison
 * - Accuracy validation
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mini_infer/core/tensor.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/runtime/execution_context.h"
#include "mini_infer/runtime/inference_plan.h"
#include "mini_infer/utils/logger.h"

namespace fs = std::filesystem;

struct InferenceResult {
    std::string filename;
    int predicted_label;
    std::vector<float> logits;
    std::vector<float> probabilities;
    int actual_label;
    bool is_correct;
};

struct MemoryStats {
    size_t original_memory;
    size_t optimized_memory;
    float saving_ratio;
    int num_pools;
};

/**
 * @brief Load binary sample file
 */
std::vector<float> load_sample(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        MI_LOG_ERROR("Failed to open sample file: " + filepath);
        return {};
    }

    // Read all data
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_floats = file_size / sizeof(float);
    std::vector<float> data(num_floats);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    MI_LOG_INFO("Loaded sample: " + filepath + " (" + std::to_string(num_floats) + " floats)");
    return data;
}

/**
 * @brief Extract label from filename (if present)
 */
int extract_label_from_filename(const std::string& filename) {
    // Format: sample_0000_label_7.bin
    size_t label_pos = filename.find("_label_");
    if (label_pos != std::string::npos) {
        size_t start = label_pos + 7;
        size_t end = filename.find(".bin", start);
        if (end != std::string::npos) {
            std::string label_str = filename.substr(start, end - start);
            return std::stoi(label_str);
        }
    }
    return -1;  // No label in filename
}

/**
 * @brief Softmax function
 */
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] /= sum;
    }

    return probs;
}

/**
 * @brief Run optimized inference with memory planning
 */
InferenceResult run_inference(mini_infer::runtime::InferencePlan& plan,
                              mini_infer::runtime::ExecutionContext& ctx,
                              const std::vector<float>& input_data, int actual_label,
                              const std::string& input_name, const std::string& output_name) {
    InferenceResult result;
    result.actual_label = actual_label;

    // Create input tensor
    auto input_tensor = std::make_shared<mini_infer::core::Tensor>(
        mini_infer::core::Shape({1, 1, 28, 28}), mini_infer::core::DataType::FLOAT32);

    // Copy input data
    if (input_data.size() == 784) {  // 28x28
        std::memcpy(input_tensor->data(), input_data.data(), input_data.size() * sizeof(float));
    } else {
        MI_LOG_ERROR("Invalid input data size: " + std::to_string(input_data.size()));
        return result;
    }

    // Prepare inputs
    std::unordered_map<std::string, std::shared_ptr<mini_infer::core::Tensor>> inputs;
    inputs[input_name] = input_tensor;

    // Execute inference
    auto status = ctx.set_inputs(inputs);
    if (status == mini_infer::core::Status::SUCCESS) {
        status = plan.execute(&ctx);
    }

    if (status != mini_infer::core::Status::SUCCESS) {
        MI_LOG_ERROR("Inference failed");
        return result;
    }

    // Get output tensor
    auto outputs = ctx.named_outputs();
    auto it = outputs.find(output_name);
    auto output_tensor = it != outputs.end() ? it->second : nullptr;
    if (!output_tensor) {
        MI_LOG_ERROR("Output tensor not found: " + output_name);
        return result;
    }

    // Extract logits
    const float* data = static_cast<const float*>(output_tensor->data());
    int64_t numel = output_tensor->shape().numel();
    result.logits.assign(data, data + numel);

    // Calculate probabilities
    result.probabilities = softmax(result.logits);

    // Get predicted label
    auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
    result.predicted_label = static_cast<int>(std::distance(result.probabilities.begin(), max_it));

    // Check correctness
    result.is_correct = (actual_label >= 0) && (result.predicted_label == actual_label);

    return result;
}

/**
 * @brief Test optimized inference with memory planning
 */
void test_optimized_inference(const std::string& model_path, const std::string& samples_dir,
                              bool enable_memory_planning, bool save_outputs,
                              const std::string& output_file) {
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("LeNet-5 Optimized Inference Test");
    MI_LOG_INFO("Memory Planning: " + std::string(enable_memory_planning ? "ENABLED" : "DISABLED"));
    MI_LOG_INFO("========================================");

    // Step 1: Load ONNX model
    MI_LOG_INFO("[Step 1] Loading ONNX model: " + model_path);
    mini_infer::importers::OnnxParser parser;
    auto graph_uptr = parser.parse_from_file(model_path);
    if (!graph_uptr) {
        MI_LOG_ERROR("Failed to parse ONNX model: " + parser.get_error());
        return;
    }
    auto graph = std::shared_ptr<mini_infer::graph::Graph>(std::move(graph_uptr));
    MI_LOG_INFO("Model loaded successfully");
    MI_LOG_INFO("");

    // Step 2: Build inference plan (TensorRT-style: includes optimization and memory planning)
    MI_LOG_INFO("[Step 2] Building inference plan...");
    mini_infer::runtime::EngineConfig config;
    config.device_type = mini_infer::core::DeviceType::CPU;
    config.enable_graph_optimization = true;
    config.enable_memory_planning = enable_memory_planning;
    config.memory_alignment = 256;
    config.enable_profiling = true;  // Enable verbose logging

    auto plan = std::make_shared<mini_infer::runtime::InferencePlan>(config);
    auto status = plan->build(graph);
    if (status != mini_infer::core::Status::SUCCESS) {
        MI_LOG_ERROR("Failed to build plan");
        return;
    }
    auto ctx = plan->create_execution_context();
    if (!ctx) {
        MI_LOG_ERROR("Failed to create execution context");
        return;
    }
    MI_LOG_INFO("");

    // Get statistics from engine
    const auto& memory_plan = plan->get_memory_plan();
    
    MemoryStats mem_stats = {0, 0, 0.0f, 0};
    if (enable_memory_planning) {
        mem_stats.original_memory = memory_plan.original_memory;
        mem_stats.optimized_memory = memory_plan.total_memory;
        mem_stats.saving_ratio = memory_plan.memory_saving_ratio;
        mem_stats.num_pools = static_cast<int>(memory_plan.pools.size());
    }

    // Step 3: Get input/output names
    auto input_names = plan->get_input_names();
    auto output_names = plan->get_output_names();

    if (input_names.empty() || output_names.empty()) {
        MI_LOG_ERROR("Failed to get input/output names from engine");
        return;
    }

    std::string input_name = input_names[0];
    std::string output_name = output_names[0];
    MI_LOG_INFO("Input name: " + input_name + ", Output name: " + output_name);

    // Prefer binary subdir if exists
    std::string actual_samples_dir = samples_dir;
    if (fs::exists(fs::path(samples_dir) / "binary")) {
        actual_samples_dir = (fs::path(samples_dir) / "binary").string();
    }

    // Step 4: Load test samples
    MI_LOG_INFO("[Step 6] Loading test samples from: " + actual_samples_dir);
    std::vector<std::string> sample_files;
    for (const auto& entry : fs::directory_iterator(actual_samples_dir)) {
        if (entry.path().extension() == ".bin") {
            sample_files.push_back(entry.path().string());
        }
    }
    std::sort(sample_files.begin(), sample_files.end());
    MI_LOG_INFO("Found " + std::to_string(sample_files.size()) + " sample(s)");

    // Step 5: Run inference
    MI_LOG_INFO("[Step 7] Running inference...");
    std::vector<InferenceResult> results;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& sample_file : sample_files) {
        auto input_data = load_sample(sample_file);
        if (input_data.empty()) {
            continue;
        }

        std::string filename = fs::path(sample_file).filename().string();
        int actual_label = extract_label_from_filename(filename);

        auto result = run_inference(*plan, *ctx, input_data, actual_label, input_name, output_name);
        result.filename = filename;
        results.push_back(result);

        MI_LOG_INFO("Sample: " + filename +
                    " | Predicted: " + std::to_string(result.predicted_label) +
                    " | Actual: " + (actual_label >= 0 ? std::to_string(actual_label) : "N/A") +
                    " | " + (result.is_correct ? "[SUCESS]" : (actual_label >= 0 ? "✗" : "-")));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Step 6: Calculate accuracy
    MI_LOG_INFO("[Step 8] Computing accuracy...");
    int total_samples = 0;
    int correct_predictions = 0;

    for (const auto& result : results) {
        if (result.actual_label >= 0) {
            total_samples++;
            if (result.is_correct) {
                correct_predictions++;
            }
        }
    }

    float accuracy = total_samples > 0 ? (100.0f * correct_predictions / total_samples) : 0.0f;

    // Step 7: Print summary
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("Inference Summary");
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("Total samples:        " + std::to_string(results.size()));
    MI_LOG_INFO("Samples with labels:  " + std::to_string(total_samples));
    MI_LOG_INFO("Correct predictions:  " + std::to_string(correct_predictions));
    MI_LOG_INFO("Accuracy:             " + std::to_string(accuracy) + " %");
    MI_LOG_INFO("Inference time:       " + std::to_string(duration.count()) + " ms");
    MI_LOG_INFO("Avg time per sample:  " +
                std::to_string(results.size() > 0 ? duration.count() / results.size() : 0) + " ms");

    if (enable_memory_planning) {
        MI_LOG_INFO("========================================");
        MI_LOG_INFO("Memory Planning Summary");
        MI_LOG_INFO("========================================");
        MI_LOG_INFO("Original memory:  " + std::to_string(mem_stats.original_memory / 1024.0) +
                    " KB");
        MI_LOG_INFO("Optimized memory: " + std::to_string(mem_stats.optimized_memory / 1024.0) +
                    " KB");
        MI_LOG_INFO("Memory saving:    " + std::to_string(mem_stats.saving_ratio * 100.0f) + "%");
        MI_LOG_INFO("Number of pools:  " + std::to_string(mem_stats.num_pools));
    }

    MI_LOG_INFO("========================================");

    // Step 8: Save outputs if requested
    if (save_outputs && !output_file.empty()) {
        MI_LOG_INFO("[Step 10] Saving outputs to: " + output_file);

        std::ofstream out(output_file);
        auto sanitize = [](const std::string& p) {
            std::string s = p;
            std::replace(s.begin(), s.end(), '\\', '/');
            return s;
        };
        out << "{\n";
        out << "  \"model\": \"" << sanitize(model_path) << "\",\n";
        out << "  \"samples_directory\": \"" << sanitize(actual_samples_dir) << "\",\n";
        out << "  \"memory_planning_enabled\": " << (enable_memory_planning ? "true" : "false")
            << ",\n";
        out << "  \"total_samples\": " << results.size() << ",\n";
        out << "  \"accuracy\": " << accuracy << ",\n";
        out << "  \"inference_time_ms\": " << duration.count() << ",\n";

        if (enable_memory_planning) {
            out << "  \"memory_stats\": {\n";
            out << "    \"original_memory_kb\": " << (mem_stats.original_memory / 1024.0) << ",\n";
            out << "    \"optimized_memory_kb\": " << (mem_stats.optimized_memory / 1024.0)
                << ",\n";
            out << "    \"saving_ratio\": " << mem_stats.saving_ratio << ",\n";
            out << "    \"num_pools\": " << mem_stats.num_pools << "\n";
            out << "  },\n";
        }

        out << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            out << "    {\n";
            out << "      \"filename\": \"" << r.filename << "\",\n";
            out << "      \"sample_index\": " << i << ",\n";
            out << "      \"label\": " << r.actual_label << ",\n";
            out << "      \"predicted\": " << r.predicted_label << ",\n";
            out << "      \"is_correct\": " << (r.is_correct ? "true" : "false") << ",\n";
            out << "      \"logits\": [";
            for (size_t j = 0; j < r.logits.size(); ++j) {
                out << r.logits[j];
                if (j < r.logits.size() - 1)
                    out << ", ";
            }
            out << "],\n";
            out << "      \"probabilities\": [";
            for (size_t j = 0; j < r.probabilities.size(); ++j) {
                out << r.probabilities[j];
                if (j < r.probabilities.size() - 1)
                    out << ", ";
            }
            out << "],\n";
            out << "      \"confidence\": "
                << (r.probabilities.empty() ? 0.0f : r.probabilities[r.predicted_label]) << "\n";
            out << "    }";
            if (i < results.size() - 1)
                out << ",";
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";
        out.close();

        MI_LOG_INFO("Outputs saved successfully");
    }

    // Validation
    if (total_samples > 0 && accuracy < 90.0f) {
        MI_LOG_ERROR("[FAIL] Accuracy is lower than expected. Accuracy=" +
                     std::to_string(accuracy) + "%, Total=" + std::to_string(total_samples));
    } else if (total_samples > 0) {
        MI_LOG_INFO("[PASS] Accuracy validation passed!");
    }
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/lenet5.onnx";
    std::string samples_dir = "models/python/lenet5/test_samples";
    bool enable_memory_planning = true;
    bool save_outputs = false;
    std::string output_file = "lenet5_optimized_outputs.json";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--samples" && i + 1 < argc) {
            samples_dir = argv[++i];
        } else if (arg == "--no-memory-planning") {
            enable_memory_planning = false;
        } else if (arg == "--save-outputs" && i + 1 < argc) {
            save_outputs = true;
            output_file = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout
                << "  --model <path>           Path to ONNX model (default: models/lenet5.onnx)\n";
            std::cout << "  --samples <dir>          Path to test samples directory\n";
            std::cout << "  --no-memory-planning     Disable memory planning\n";
            std::cout << "  --save-outputs <file>    Save outputs to JSON file\n";
            std::cout << "  --help                   Show this help message\n";
            return 0;
        }
    }

    test_optimized_inference(model_path, samples_dir, enable_memory_planning, save_outputs,
                             output_file);

    return 0;
}
