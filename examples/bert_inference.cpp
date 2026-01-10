/**
 * @file bert_inference.cpp
 * @brief BERT-tiny inference example using Mini-Infer
 *
 * This example demonstrates how to:
 * 1. Load a BERT-tiny ONNX model
 * 2. Prepare input tensors (input_ids, attention_mask, token_type_ids)
 * 3. Run inference
 * 4. Get the output (last_hidden_state, pooler_output)
 *
 * Usage:
 *   ./bert_inference <model_path> [test_data_dir]
 *
 * Example:
 *   ./bert_inference models/python/bert/bert_tiny.onnx models/python/bert/test_samples
 */

#ifdef MINI_INFER_ONNX_ENABLED

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>

#include "mini_infer/core/tensor.h"
#include "mini_infer/core/types.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/runtime/inference_plan.h"
#include "mini_infer/runtime/execution_context.h"
#include "mini_infer/runtime/optimization_profile.h"
#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/utils/logger.h"

using namespace mini_infer;
namespace fs = std::filesystem;

// Load binary file into vector
template<typename T>
std::vector<T> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<T> buffer(size / sizeof(T));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file: " + path);
    }

    return buffer;
}

// Compare two float arrays
float compute_max_diff(const float* a, const float* b, size_t size) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path> [test_data_dir]" << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  model_path     - BERT-tiny ONNX model file" << std::endl;
    std::cout << "  test_data_dir  - Directory containing test data (optional)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " models/python/bert/bert_tiny.onnx" << std::endl;
    std::cout << "  " << program_name << " models/python/bert/bert_tiny.onnx models/python/bert/test_samples" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "BERT-tiny Inference Example - Mini-Infer" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::endl;

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string test_data_dir = (argc > 2) ? argv[2] : "";

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Model: " << model_path << std::endl;
    if (!test_data_dir.empty()) {
        std::cout << "  Test data: " << test_data_dir << std::endl;
    }
    std::cout << std::endl;

    // Check if model file exists
    if (!fs::exists(model_path)) {
        std::cerr << "Error: Model file not found: " << model_path << std::endl;
        return 1;
    }

    try {
        // Step 1: Parse ONNX model
        std::cout << "Step 1: Parsing ONNX Model" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        importers::OnnxParser parser;
        parser.set_verbose(true);

        auto graph_uptr = parser.parse_from_file(model_path);

        if (!graph_uptr) {
            std::cerr << "Failed to parse ONNX model: " << parser.get_error() << std::endl;
            return 1;
        }
        std::shared_ptr<graph::Graph> graph = std::move(graph_uptr);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "[SUCCESS] Model parsed in " << duration.count() << " ms" << std::endl;
        std::cout << "  Graph nodes: " << graph->node_count() << std::endl;
        std::cout << std::endl;

        // Step 2: Build Inference Plan
        std::cout << "Step 2: Building Inference Plan" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        // Create optimization profile for dynamic shapes
        // BERT has 3 inputs: input_ids, attention_mask, token_type_ids
        // All have shape [batch_size, seq_length]
        auto profile = std::make_shared<runtime::OptimizationProfile>();
        profile->set_shape_range("input_ids",
            core::Shape({1, 1}),    // min: batch=1, seq=1
            core::Shape({1, 32}),   // opt: batch=1, seq=32
            core::Shape({4, 128})   // max: batch=4, seq=128
        );
        profile->set_shape_range("attention_mask",
            core::Shape({1, 1}),
            core::Shape({1, 32}),
            core::Shape({4, 128})
        );
        profile->set_shape_range("token_type_ids",
            core::Shape({1, 1}),
            core::Shape({1, 32}),
            core::Shape({4, 128})
        );

        runtime::EngineConfig config;
        config.device_type = core::DeviceType::CPU;
        config.enable_profiling = false;
        config.enable_graph_optimization = false;
        config.enable_memory_planning = false;
        config.enable_dynamic_shapes = true;
        config.optimization_profile = profile;

        auto plan = std::make_shared<runtime::InferencePlan>(config);
        auto status = plan->build(graph);

        if (status != core::Status::SUCCESS) {
            std::cerr << "Failed to build inference plan" << std::endl;
            return 1;
        }

        auto ctx = plan->create_execution_context();
        if (!ctx) {
            std::cerr << "Failed to create execution context" << std::endl;
            return 1;
        }

        std::cout << "[SUCCESS] Plan built successfully!" << std::endl;

        // Get input and output names
        auto input_names = plan->get_input_names();
        auto output_names = plan->get_output_names();

        std::cout << "  Inputs: ";
        for (const auto& name : input_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;

        std::cout << "  Outputs: ";
        for (const auto& name : output_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl << std::endl;

        // Step 3: Prepare input tensors
        std::cout << "Step 3: Preparing Input Tensors" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        const int batch_size = 1;
        const int seq_length = 32;

        // Create input tensors
        auto input_ids = core::Tensor::create(
            core::Shape({batch_size, seq_length}), core::DataType::INT64, core::DeviceType::CPU);
        auto attention_mask = core::Tensor::create(
            core::Shape({batch_size, seq_length}), core::DataType::INT64, core::DeviceType::CPU);
        auto token_type_ids = core::Tensor::create(
            core::Shape({batch_size, seq_length}), core::DataType::INT64, core::DeviceType::CPU);

        // Fill with test data or random data
        if (!test_data_dir.empty() && fs::exists(test_data_dir)) {
            std::cout << "Loading test data from: " << test_data_dir << std::endl;

            auto input_ids_data = load_binary_file<int64_t>(test_data_dir + "/input_ids.bin");
            auto attention_mask_data = load_binary_file<int64_t>(test_data_dir + "/attention_mask.bin");
            auto token_type_ids_data = load_binary_file<int64_t>(test_data_dir + "/token_type_ids.bin");

            std::memcpy(input_ids->data(), input_ids_data.data(),
                        input_ids_data.size() * sizeof(int64_t));
            std::memcpy(attention_mask->data(), attention_mask_data.data(),
                        attention_mask_data.size() * sizeof(int64_t));
            std::memcpy(token_type_ids->data(), token_type_ids_data.data(),
                        token_type_ids_data.size() * sizeof(int64_t));
        } else {
            std::cout << "Using random input data" << std::endl;

            // Fill with random token IDs (0-30521 for BERT vocabulary)
            int64_t* ids_ptr = static_cast<int64_t*>(input_ids->data());
            int64_t* mask_ptr = static_cast<int64_t*>(attention_mask->data());
            int64_t* type_ptr = static_cast<int64_t*>(token_type_ids->data());

            for (int i = 0; i < batch_size * seq_length; ++i) {
                ids_ptr[i] = rand() % 30522;
                mask_ptr[i] = 1;  // All tokens are valid
                type_ptr[i] = 0;  // Single sentence
            }
        }

        std::cout << "  input_ids: " << input_ids->shape().to_string() << std::endl;
        std::cout << "  attention_mask: " << attention_mask->shape().to_string() << std::endl;
        std::cout << "  token_type_ids: " << token_type_ids->shape().to_string() << std::endl;
        std::cout << std::endl;

        // Set inputs
        std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;
        inputs["input_ids"] = input_ids;
        inputs["attention_mask"] = attention_mask;
        inputs["token_type_ids"] = token_type_ids;

        status = ctx->set_inputs(inputs);
        if (status != core::Status::SUCCESS) {
            std::cerr << "Failed to set inputs" << std::endl;
            return 1;
        }

        // Step 4: Run inference
        std::cout << "Step 4: Running Inference" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        // Warmup
        std::cout << "Warmup run..." << std::endl;
        status = plan->execute(ctx.get());
        if (status != core::Status::SUCCESS) {
            std::cerr << "Warmup inference failed" << std::endl;
            return 1;
        }

        // Benchmark
        const int num_runs = 10;
        std::cout << "Benchmarking (" << num_runs << " runs)..." << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            status = plan->execute(ctx.get());
            if (status != core::Status::SUCCESS) {
                std::cerr << "Inference failed at run " << i << std::endl;
                return 1;
            }
        }
        end = std::chrono::high_resolution_clock::now();

        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_latency = total_duration.count() / static_cast<double>(num_runs) / 1000.0;

        std::cout << "[SUCCESS] Inference completed!" << std::endl;
        std::cout << "  Average latency: " << avg_latency << " ms" << std::endl;
        std::cout << "  Throughput: " << 1000.0 / avg_latency << " inferences/sec" << std::endl;
        std::cout << std::endl;

        // Step 5: Get outputs
        std::cout << "Step 5: Getting Outputs" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        auto outputs = ctx->named_outputs();

        for (const auto& [name, tensor] : outputs) {
            if (tensor) {
                std::cout << "  " << name << ": " << tensor->shape().to_string() << std::endl;

                // Print first few values
                if (tensor->dtype() == core::DataType::FLOAT32) {
                    const float* data = static_cast<const float*>(tensor->data());
                    std::cout << "    First 5 values: ";
                    for (int i = 0; i < 5 && i < tensor->shape().numel(); ++i) {
                        std::cout << data[i] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        std::cout << std::endl;

        // Step 6: Verify against reference (if test data provided)
        if (!test_data_dir.empty() && fs::exists(test_data_dir + "/last_hidden_state.bin")) {
            std::cout << "Step 6: Verifying Against Reference" << std::endl;
            std::cout << std::string(70, '-') << std::endl;

            try {
                auto ref_data = load_binary_file<float>(test_data_dir + "/last_hidden_state.bin");

                auto it = outputs.find("last_hidden_state");
                if (it != outputs.end() && it->second) {
                    const float* output_data = static_cast<const float*>(it->second->data());
                    float max_diff = compute_max_diff(output_data, ref_data.data(), ref_data.size());

                    std::cout << "  Max difference from reference: " << max_diff << std::endl;

                    if (max_diff < 1e-4f) {
                        std::cout << "  [PASS] Output matches reference!" << std::endl;
                    } else if (max_diff < 1e-3f) {
                        std::cout << "  [WARN] Output has small differences from reference" << std::endl;
                    } else {
                        std::cout << "  [FAIL] Output differs significantly from reference" << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cout << "  Could not load reference data: " << e.what() << std::endl;
            }
            std::cout << std::endl;
        }

        std::cout << std::string(70, '=') << std::endl;
        std::cout << "BERT-tiny Inference Complete!" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#else

#include <iostream>

int main() {
    std::cout << "ONNX support is not enabled." << std::endl;
    std::cout << "Please build with MINI_INFER_ENABLE_ONNX=ON" << std::endl;
    return 1;
}

#endif // MINI_INFER_ONNX_ENABLED
