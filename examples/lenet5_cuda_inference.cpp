/**
 * @file lenet5_cuda_inference.cpp
 * @brief LeNet-5 ONNX CUDA GPU inference example
 *
 * Features:
 * 1. Import LeNet-5 model from ONNX file
 * 2. Apply TensorRT-style graph optimization (operator fusion)
 * 3. Use CUDA GPU for inference
 * 4. Load MNIST test samples
 * 5. Batch inference and calculate accuracy
 * 6. Compare CPU vs GPU performance
 *
 * Usage:
 *   lenet5_cuda_inference <model.onnx> <samples_dir> [num_samples] [OPTIONS]
 *
 * Example:
 *   lenet5_cuda_inference lenet5.onnx ../models/python/lenet5/test_samples/binary 100
 */

#if defined(MINI_INFER_ONNX_ENABLED) && defined(MINI_INFER_USE_CUDA)

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
#include "mini_infer/backends/cuda/cuda_device_context.h"
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
 * @brief Benchmark result structure
 */
struct BenchmarkResult {
    int total_samples;
    int correct;
    double total_time_ms;
    double avg_time_ms;
    double accuracy;
};

/**
 * @brief Run benchmark on engine
 */
BenchmarkResult benchmark_engine(std::shared_ptr<runtime::InferencePlan> plan,
                                 runtime::ExecutionContext& ctx,
                                 const std::string& input_name,
                                 const std::string& output_name,
                                 const std::vector<fs::path>& sample_files,
                                 bool verbose = false) {
    BenchmarkResult result = {0, 0, 0.0, 0.0, 0.0};

    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& filepath : sample_files) {
        try {
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

            // Extract label from filename
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
            if (predicted == label && label != -1) {
                result.correct++;
            }

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
 * @brief Print CUDA device information
 */
void print_cuda_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::cout << "CUDA Device Information:" << std::endl;
    std::cout << "  Number of CUDA devices: " << device_count << std::endl;

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\n  Device " << i << ": " << prop.name << std::endl;
        std::cout << "    Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "    Total memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "    SM count: " << prop.multiProcessorCount << std::endl;
        std::cout << "    Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "    Warp size: " << prop.warpSize << std::endl;
    }
    std::cout << std::endl;
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
    std::cout << "  --verbose              - Enable verbose logging" << std::endl;
    std::cout << "  --no-optimization      - Disable graph optimization" << std::endl;
    std::cout << "  --compare-cpu          - Also run CPU inference for comparison" << std::endl;
    std::cout << "  -h, --help             - Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " lenet5.onnx samples/" << std::endl;
    std::cout << "  " << program_name << " lenet5.onnx samples/ 100 --compare-cpu" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "         LeNet-5 CUDA GPU Inference - Mini-Infer Demo                  \n";
    std::cout << "========================================================================\n";
    std::cout << "\n";

    // Parse command line arguments
    std::string model_path;
    std::string samples_dir;
    int num_samples = -1;
    bool verbose = false;
    bool enable_optimization = true;
    bool compare_cpu = false;

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
        } else if (arg == "--compare-cpu") {
            compare_cpu = true;
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

    // Print CUDA device information
    print_cuda_info();

    // Display configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  ONNX model: " << model_path << std::endl;
    std::cout << "  Samples directory: " << samples_dir << std::endl;
    if (num_samples > 0) {
        std::cout << "  Number of samples: " << num_samples << std::endl;
    } else {
        std::cout << "  Number of samples: all" << std::endl;
    }
    std::cout << "  Graph optimization: " << (enable_optimization ? "enabled" : "disabled") << std::endl;
    std::cout << "  Compare with CPU: " << (compare_cpu ? "yes" : "no") << std::endl;
    std::cout << std::endl;

    // Check if files exist
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
        // Step 2: Build CUDA GPU Engine
        // ========================================================================
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Step 2: Building CUDA GPU Inference Plan" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        runtime::EngineConfig cuda_config;
        cuda_config.device_type = core::DeviceType::CUDA;  // Use CUDA GPU
        cuda_config.device_id = 0;
        cuda_config.enable_profiling = false;

        auto cuda_plan = std::make_shared<runtime::InferencePlan>(cuda_config);
        auto status = cuda_plan->build(graph);

        if (status != core::Status::SUCCESS) {
            std::cerr << "Failed to build CUDA plan" << std::endl;
            return 1;
        }

        auto cuda_ctx = cuda_plan->create_execution_context();
        if (!cuda_ctx) {
            std::cerr << "Failed to create CUDA execution context" << std::endl;
            return 1;
        }

        std::cout << "[SUCCESS] CUDA GPU Plan built successfully!" << std::endl;

        // Get input and output names
        auto input_names = cuda_plan->get_input_names();
        auto output_names = cuda_plan->get_output_names();

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
        // Step 3: GPU Performance Benchmark
        // ========================================================================
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Step 3: CUDA GPU Performance Benchmark" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        std::cout << "Running GPU inference on " << sample_files.size() << " samples..." << std::endl;

        // Warmup run
        std::cout << "  Warming up GPU..." << std::endl;
        for (int i = 0; i < std::min(10, static_cast<int>(sample_files.size())); ++i) {
            auto input_tensor = utils::load_mnist_sample(sample_files[i].string());
            std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;
            inputs[input_names[0]] = input_tensor;
            cuda_ctx->set_inputs(inputs);
            cuda_plan->execute(cuda_ctx.get());
        }

        // Actual benchmark
        auto gpu_result = benchmark_engine(cuda_plan, *cuda_ctx, input_names[0], output_names[0],
                                           sample_files, verbose);

        std::cout << std::endl;

        // ========================================================================
        // Step 4: Optional CPU Comparison
        // ========================================================================
        BenchmarkResult cpu_result = {0, 0, 0.0, 0.0, 0.0};

        if (compare_cpu) {
            std::cout << std::string(80, '=') << std::endl;
            std::cout << "Step 4: CPU Performance Comparison" << std::endl;
            std::cout << std::string(80, '=') << std::endl;

            // Re-parse model for CPU (graph was modified by optimization)
            auto cpu_graph_uptr = parser.parse_from_file(model_path);
            std::shared_ptr<graph::Graph> cpu_graph = std::move(cpu_graph_uptr);

            runtime::EngineConfig cpu_config;
            cpu_config.device_type = core::DeviceType::CPU;
            cpu_config.enable_profiling = false;

            auto cpu_plan = std::make_shared<runtime::InferencePlan>(cpu_config);
            status = cpu_plan->build(cpu_graph);

            if (status == core::Status::SUCCESS) {
                auto cpu_ctx = cpu_plan->create_execution_context();
                if (cpu_ctx) {
                    std::cout << "Running CPU inference on " << sample_files.size() << " samples..." << std::endl;
                    cpu_result = benchmark_engine(cpu_plan, *cpu_ctx, input_names[0], output_names[0],
                                                  sample_files, verbose);
                }
            }
            std::cout << std::endl;
        }

        // ========================================================================
        // Step 5: Results Display
        // ========================================================================
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Results Summary" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        std::cout << "\n[CUDA GPU Results]" << std::endl;
        std::cout << "  Total samples: " << gpu_result.total_samples << std::endl;
        std::cout << "  Correct predictions: " << gpu_result.correct << std::endl;
        std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << gpu_result.accuracy << " %" << std::endl;
        std::cout << "  Total time: " << std::fixed << std::setprecision(2) << gpu_result.total_time_ms << " ms" << std::endl;
        std::cout << "  Average time per sample: " << std::fixed << std::setprecision(3) << gpu_result.avg_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << (1000.0 / gpu_result.avg_time_ms) << " samples/sec" << std::endl;

        if (compare_cpu && cpu_result.total_samples > 0) {
            std::cout << "\n[CPU Results]" << std::endl;
            std::cout << "  Total samples: " << cpu_result.total_samples << std::endl;
            std::cout << "  Correct predictions: " << cpu_result.correct << std::endl;
            std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << cpu_result.accuracy << " %" << std::endl;
            std::cout << "  Total time: " << std::fixed << std::setprecision(2) << cpu_result.total_time_ms << " ms" << std::endl;
            std::cout << "  Average time per sample: " << std::fixed << std::setprecision(3) << cpu_result.avg_time_ms << " ms" << std::endl;
            std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << (1000.0 / cpu_result.avg_time_ms) << " samples/sec" << std::endl;

            std::cout << "\n[Performance Comparison]" << std::endl;
            double speedup = cpu_result.avg_time_ms / gpu_result.avg_time_ms;
            std::cout << "  GPU Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;

            if (speedup > 1.0) {
                std::cout << "  GPU is " << std::fixed << std::setprecision(1)
                          << ((speedup - 1.0) * 100) << "% faster than CPU" << std::endl;
            } else {
                std::cout << "  Note: CPU is faster for this small model/batch size." << std::endl;
                std::cout << "        GPU benefits are more visible with larger models and batch sizes." << std::endl;
            }
        }

        std::cout << std::string(80, '=') << std::endl;
        std::cout << "\n[SUCCESS] CUDA GPU inference completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#else

#include <iostream>

int main() {
#ifndef MINI_INFER_ONNX_ENABLED
    std::cout << "ONNX support is not enabled." << std::endl;
    std::cout << "Please build with MINI_INFER_ENABLE_ONNX=ON" << std::endl;
#endif
#ifndef MINI_INFER_USE_CUDA
    std::cout << "CUDA support is not enabled." << std::endl;
    std::cout << "Please build with MINI_INFER_ENABLE_CUDA=ON" << std::endl;
#endif
    return 1;
}

#endif  // MINI_INFER_ONNX_ENABLED && MINI_INFER_USE_CUDA
