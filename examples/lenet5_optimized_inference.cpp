/**
 * @file lenet5_optimized_inference.cpp
 * @brief LeNet-5 ONNX optimized inference example (TensorRT-style graph optimization)
 *
 * Features:
 * 1. Import LeNet-5 model from ONNX file
 * 2. Apply TensorRT-style graph optimization (operator fusion)
 * 3. Use Runtime Engine to build inference engine
 * 4. Load MNIST test samples
 * 5. Batch inference and calculate accuracy
 * 6. Compare performance before and after optimization
 *
 * Usage:
 *   lenet5_optimized_inference <model.onnx> <samples_dir> [num_samples] [OPTIONS]
 *
 * Example:
 *   lenet5_optimized_inference lenet5.onnx ../models/python/lenet5/test_samples/binary 100
 */

#ifdef MINI_INFER_ONNX_ENABLED

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_infer/graph/fusion_pass.h"
#include "mini_infer/graph/graph_optimizer.h"
#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/runtime/execution_context.h"
#include "mini_infer/runtime/inference_plan.h"
#include "utils/simple_loader.h"


using namespace mini_infer;
namespace fs = std::filesystem;

/**
 * @brief Calculate softmax
 */
std::vector<float> softmax(const std::shared_ptr<core::Tensor>& logits) {
    const float* data = static_cast<const float*>(logits->data());
    int64_t numel = logits->shape().numel();

    float max_val = data[0];
    for (int64_t i = 1; i < numel; ++i) {
        if (data[i] > max_val)
            max_val = data[i];
    }

    std::vector<float> exp_values(numel);
    float sum = 0.0f;
    for (int64_t i = 0; i < numel; ++i) {
        exp_values[i] = std::exp(data[i] - max_val);
        sum += exp_values[i];
    }

    for (int64_t i = 0; i < numel; ++i) {
        exp_values[i] /= sum;
    }

    return exp_values;
}

/**
 * @brief Test engine performance
 */
struct BenchmarkResult {
    int total_samples;
    int correct;
    double total_time_ms;
    double avg_time_ms;
    double accuracy;
};

struct SampleResult {
    int index;
    std::string filename;
    int label;
    int predicted;
    std::vector<float> logits;
    std::vector<float> probabilities;
    float confidence;
    bool correct;
};

BenchmarkResult benchmark_engine(std::shared_ptr<runtime::InferencePlan> plan,
                                 runtime::ExecutionContext& ctx,
                                 const std::string& input_name, const std::string& output_name,
                                 const std::vector<fs::path>& sample_files, bool verbose = false,
                                 std::vector<SampleResult>* collected_results = nullptr) {
    BenchmarkResult result = {0, 0, 0.0, 0.0, 0.0};

    auto start_time = std::chrono::high_resolution_clock::now();

    int sample_index = 0;
    for (const auto& filepath : sample_files) {
        try {
            if (verbose) {
                std::cout << "[Sample " << sample_index << "] " << filepath.filename().string()
                          << std::endl;
            }
            // Load sample
            auto input_tensor = utils::load_mnist_sample(filepath.string());

            // Prepare input
            std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;
            inputs[input_name] = input_tensor;

            // Execute inference
            auto status = ctx.set_inputs(inputs);
            if (status == core::Status::SUCCESS) {
                status = plan->execute(&ctx);
            }

            if (status != core::Status::SUCCESS) {
                if (verbose) {
                    std::cerr << "Error: Inference failed for " << filepath.filename() << std::endl;
                }
                continue;
            }
            if (verbose) {
                std::cout << "  -> Execution succeeded" << std::endl;
            }

            // Get output
            auto outputs = ctx.named_outputs();
            auto it = outputs.find(output_name);
            auto output_tensor = it != outputs.end() ? it->second : nullptr;
            if (!output_tensor) {
                if (verbose) {
                    std::cerr << "Error: Output tensor not found for " << filepath.filename()
                              << std::endl;
                }
                continue;
            }

            // Get prediction
            int predicted = utils::argmax(output_tensor);

            // 从文件名提取标签
            std::string filename = filepath.stem().string();
            int label = -1;

            size_t label_pos = filename.find("_label_");
            if (label_pos != std::string::npos) {
                try {
                    label = std::stoi(filename.substr(label_pos + 7));
                } catch (...) {
                    label = -1;
                }
            }

            result.total_samples++;
            const bool is_correct = (predicted == label && label != -1);
            if (is_correct) {
                result.correct++;
            }

            // Collect detailed results for JSON output
            if (collected_results) {
                SampleResult r;
                r.index = sample_index;
                r.filename = filepath.filename().string();
                r.label = label;
                r.predicted = predicted;
                const float* data = static_cast<const float*>(output_tensor->data());
                int64_t numel = output_tensor->shape().numel();
                r.logits.assign(data, data + numel);
                r.probabilities = softmax(output_tensor);
                r.confidence =
                    (predicted >= 0 && predicted < static_cast<int>(r.probabilities.size()))
                        ? r.probabilities[predicted]
                        : 0.0f;
                r.correct = is_correct;
                collected_results->push_back(std::move(r));
            }

            sample_index++;

        } catch (const std::exception& e) {
            if (verbose) {
                std::cerr << "Error processing " << filepath << ": " << e.what() << std::endl;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (result.total_samples > 0) {
        result.avg_time_ms = result.total_time_ms / result.total_samples;
        result.accuracy = 100.0 * result.correct / result.total_samples;
    }

    return result;
}

/**
 * @brief Print usage instructions
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model.onnx> <samples_dir> [num_samples] [OPTIONS]"
              << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  model.onnx   - LeNet-5 ONNX model file" << std::endl;
    std::cout << "  samples_dir  - Directory containing MNIST test samples (.bin files)"
              << std::endl;
    std::cout << "  num_samples  - Number of samples to test (optional, default: all)" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --save-outputs <file>  - Save inference results to JSON file" << std::endl;
    std::cout << "  --verbose              - Enable verbose logging" << std::endl;
    std::cout << "  --no-optimization      - Disable graph optimization" << std::endl;
    std::cout << "  -h, --help             - Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " lenet5.onnx samples/" << std::endl;
    std::cout << "  " << program_name << " lenet5.onnx samples/ 100 --save-outputs results.json"
              << std::endl;
    std::cout << "  " << program_name << " lenet5.onnx samples/ --verbose" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         LeNet-5 Optimized Inference - TensorRT-style Fusion               ║\n";
    std::cout << "║         Mini-Infer: Graph Optimization Performance Demo                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Parse command line arguments
    std::string model_path;
    std::string samples_dir;
    int num_samples = -1;
    bool verbose = false;
    bool enable_optimization = true;
    std::string output_json;

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--no-optimization") {
            enable_optimization = false;
        } else if (arg == "--save-outputs" && i + 1 < argc) {
            output_json = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (i == 1) {
            model_path = arg;
        } else if (i == 2) {
            samples_dir = arg;
        } else if (i == 3 && arg.find("--") != 0) {
            num_samples = std::atoi(arg.c_str());
        }
    }

    // Display configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  ONNX model: " << model_path << std::endl;
    std::cout << "  Samples directory: " << samples_dir << std::endl;
    if (num_samples > 0) {
        std::cout << "  Number of samples: " << num_samples << std::endl;
    } else {
        std::cout << "  Number of samples: all" << std::endl;
    }
    std::cout << "  Graph optimization: " << (enable_optimization ? "enabled" : "disabled")
              << std::endl;
    std::cout << "  Verbose: " << (verbose ? "enabled" : "disabled") << std::endl;
    if (!output_json.empty()) {
        std::cout << "  Save outputs: " << output_json << std::endl;
    }
    std::cout << std::endl;

    // Check if file exists
    if (!fs::exists(model_path)) {
        std::cerr << "Error: ONNX model file not found: " << model_path << std::endl;
        return 1;
    }

    if (!fs::exists(samples_dir)) {
        std::cerr << "Error: Samples directory not found: " << samples_dir << std::endl;
        return 1;
    }

    try {
        // Determine actual sample directory
        std::string actual_samples_dir = samples_dir;
        fs::path samples_path(samples_dir);

        if (fs::exists(samples_path / "binary")) {
            actual_samples_dir = (samples_path / "binary").string();
        }

        // Get all sample files
        std::vector<fs::path> sample_files;
        for (const auto& entry : fs::directory_iterator(actual_samples_dir)) {
            if (entry.path().extension() == ".bin") {
                sample_files.push_back(entry.path());
            }
        }

        std::sort(sample_files.begin(), sample_files.end());

        if (num_samples > 0 && static_cast<size_t>(num_samples) < sample_files.size()) {
            sample_files.resize(num_samples);
        }

        std::cout << "Found " << sample_files.size() << " test samples" << std::endl;
        std::cout << std::endl;

        // ========================================================================
        // Step 1: Parse ONNX model
        // ========================================================================
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Step 1: Parsing ONNX Model" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        importers::OnnxParser parser;
        parser.set_verbose(verbose);

        auto graph_uptr = parser.parse_from_file(model_path);

        if (!graph_uptr) {
            std::cerr << "Failed to parse ONNX model: " << parser.get_error() << std::endl;
            return 1;
        }
        std::shared_ptr<graph::Graph> graph = std::move(graph_uptr);

        std::cout << "[SUCCESS] Model parsed successfully!" << std::endl;
        std::cout << "  Graph nodes: " << graph->node_count() << std::endl;
        std::cout << std::endl;

        // ========================================================================
        // Step 2: Graph optimization (TensorRT-style)
        // ========================================================================
        if (enable_optimization) {
            std::cout << std::string(80, '=') << std::endl;
            std::cout << "Step 2: Graph Optimization (TensorRT-style)" << std::endl;
            std::cout << std::string(80, '=') << std::endl;

            size_t nodes_before = graph->node_count();

            graph::GraphOptimizer optimizer;
            optimizer.set_verbose(true);

            // Add fusion pass
            auto fusion_pass = std::make_shared<graph::FusionPass>();
            optimizer.add_pass(fusion_pass);

            // Apply optimization
            auto status = optimizer.optimize(graph.get());

            if (status != core::Status::SUCCESS) {
                std::cerr << "Warning: Graph optimization failed" << std::endl;
            } else {
                const auto& stats = optimizer.get_statistics();
                size_t nodes_after = graph->node_count();

                std::cout << "\n[OPTIMIZATION RESULTS]" << std::endl;
                std::cout << "  Total passes: " << stats.total_passes << std::endl;
                std::cout << "  Total modifications: " << stats.total_modifications << std::endl;
                std::cout << "  Original graph nodes: " << nodes_before << std::endl;
                std::cout << "  Optimized graph nodes: " << nodes_after << std::endl;
                std::cout << "  Nodes reduced: " << (nodes_before - nodes_after) << std::endl;
            }
            std::cout << std::endl;
        }

        // ========================================================================
        // Step 3: Build engine
        // ========================================================================
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Step 3: Building Inference Plan" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        runtime::EngineConfig config;
        config.device_type = core::DeviceType::CPU;
        config.enable_profiling = false;

        auto plan = std::make_shared<runtime::InferencePlan>(config);
        auto status = plan->build(graph);

        if (status != core::Status::SUCCESS) {
            std::cerr << "Failed to build plan" << std::endl;
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

        if (input_names.empty() || output_names.empty()) {
            std::cerr << "Error: Model has no inputs or outputs!" << std::endl;
            return 1;
        }

        // ========================================================================
        // Step 4: Performance test
        // ========================================================================
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Step 4: Performance Benchmark" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        std::cout << "Running inference on " << sample_files.size() << " samples..." << std::endl;
        std::cout << std::endl;

        std::vector<SampleResult> detailed_results;
        auto result =
            benchmark_engine(plan, *ctx, input_names[0], output_names[0], sample_files, verbose,
                             output_json.empty() ? nullptr : &detailed_results);

        // ========================================================================
        // Step 5: Results display
        // ========================================================================
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Results" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        std::cout << "Total samples: " << result.total_samples << std::endl;
        std::cout << "Correct predictions: " << result.correct << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << result.accuracy << " %"
                  << std::endl;
        std::cout << std::endl;

        std::cout << "Performance:" << std::endl;
        std::cout << "  Total time: " << std::fixed << std::setprecision(2) << result.total_time_ms
                  << " ms" << std::endl;
        std::cout << "  Average time per sample: " << std::fixed << std::setprecision(3)
                  << result.avg_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                  << (1000.0 / result.avg_time_ms) << " samples/sec" << std::endl;

        std::cout << std::string(80, '=') << std::endl;
        std::cout << "\n[SUCCESS] Optimized inference completed successfully!" << std::endl;

        if (enable_optimization) {
            std::cout << "\nNote: Graph optimization (TensorRT-style fusion) was applied."
                      << std::endl;
            std::cout << "      Conv + Activation layers were fused for better performance."
                      << std::endl;
        }

        // Optional: save outputs to JSON for comparison with PyTorch
        if (!output_json.empty() && !detailed_results.empty()) {
            std::ofstream json_file(output_json);
            if (!json_file.is_open()) {
                std::cerr << "Error: Could not open output file: " << output_json << std::endl;
                return 1;
            }

            // Convert Windows path to JSON-safe format (replace backslashes with forward slashes)
            std::string json_safe_path = actual_samples_dir;
            std::replace(json_safe_path.begin(), json_safe_path.end(), '\\', '/');

            json_file << "{\n";
            json_file << "  \"model_type\": \"Optimized\",\n";
            json_file << "  \"samples_directory\": \"" << json_safe_path << "\",\n";
            json_file << "  \"total_samples\": " << result.total_samples << ",\n";
            json_file << "  \"accuracy\": " << result.accuracy << ",\n";
            json_file << "  \"results\": [\n";
            for (size_t i = 0; i < detailed_results.size(); ++i) {
                const auto& r = detailed_results[i];
                json_file << "    {\n";
                json_file << "      \"index\": " << r.index << ",\n";
                json_file << "      \"filename\": \"" << r.filename << "\",\n";
                json_file << "      \"label\": " << r.label << ",\n";
                json_file << "      \"predicted\": " << r.predicted << ",\n";
                json_file << "      \"logits\": [";
                for (size_t j = 0; j < r.logits.size(); ++j) {
                    json_file << std::fixed << std::setprecision(6) << r.logits[j];
                    if (j + 1 < r.logits.size())
                        json_file << ", ";
                }
                json_file << "],\n";
                json_file << "      \"probabilities\": [";
                for (size_t j = 0; j < r.probabilities.size(); ++j) {
                    json_file << std::fixed << std::setprecision(6) << r.probabilities[j];
                    if (j + 1 < r.probabilities.size())
                        json_file << ", ";
                }
                json_file << "],\n";
                json_file << "      \"confidence\": " << std::fixed << std::setprecision(6)
                          << r.confidence << ",\n";
                json_file << "      \"correct\": " << (r.correct ? "true" : "false") << "\n";
                json_file << "    }";
                if (i + 1 < detailed_results.size())
                    json_file << ",";
                json_file << "\n";
            }
            json_file << "  ]\n";
            json_file << "}\n";
            json_file.close();
            std::cout << "\n[SUCCESS] Outputs saved to: " << output_json << std::endl;
        }

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

#endif  // MINI_INFER_ONNX_ENABLED
