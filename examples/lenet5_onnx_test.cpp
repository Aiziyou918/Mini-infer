/**
 * @file lenet5_onnx_test.cpp
 * @brief LeNet-5 ONNX Complete Test Program
 * 
 * Features:
 * 1. Import LeNet-5 model from ONNX file
 * 2. Build inference engine using Runtime Engine
 * 3. Load MNIST test samples
 * 4. Batch inference and calculate accuracy
 * 5. Output detailed results and statistics
 * 
 * Usage:
 *   lenet5_onnx_test <model.onnx> <samples_dir> [num_samples] [OPTIONS]
 * 
 * Example:
 *   lenet5_onnx_test lenet5.onnx ../models/python/lenet5/test_samples/binary 100
 */

#ifdef MINI_INFER_ONNX_ENABLED

#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/runtime/execution_context.h"
#include "mini_infer/runtime/inference_plan.h"
#include "mini_infer/utils/logger.h"
#include "utils/simple_loader.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>

using namespace mini_infer;
namespace fs = std::filesystem;

/**
 * @brief Calculate softmax
 */
std::vector<float> softmax(const std::shared_ptr<core::Tensor>& logits) {
    const float* data = static_cast<const float*>(logits->data());
    int64_t numel = logits->shape().numel();
    
    // Find the maximum value (numerical stability)
    float max_val = data[0];
    for (int64_t i = 1; i < numel; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    // Calculate exp and sum
    std::vector<float> exp_values(numel);
    float sum = 0.0f;
    for (int64_t i = 0; i < numel; ++i) {
        exp_values[i] = std::exp(data[i] - max_val);
        sum += exp_values[i];
    }
    
    // Normalize
    for (int64_t i = 0; i < numel; ++i) {
        exp_values[i] /= sum;
    }
    
    return exp_values;
}

/**
 * @brief Test LeNet-5 ONNX model
 */
void test_lenet5_onnx(
    std::shared_ptr<runtime::InferencePlan> plan,
    runtime::ExecutionContext& ctx,
    const std::string& input_name,
    const std::string& output_name,
    const std::string& samples_dir,
    int num_samples = -1,
    const std::string& output_json = ""
) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Testing LeNet-5 ONNX Model on MNIST Samples" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Determine the actual sample directory
    std::string actual_samples_dir = samples_dir;
    fs::path samples_path(samples_dir);
    
    if (fs::exists(samples_path / "binary")) {
        actual_samples_dir = (samples_path / "binary").string();
        std::cout << "\nNote: Using binary subdirectory: " << actual_samples_dir << std::endl;
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
    
    std::cout << "\nTesting on " << sample_files.size() << " samples..." << std::endl;
    std::cout << "Sample directory: " << actual_samples_dir << std::endl << std::endl;
    
    // Store results
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
    std::vector<SampleResult> results;
    
    int correct = 0;
    int total = 0;
    
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
                std::cerr << "Error: Inference failed for " 
                          << filepath.filename() << std::endl;
                continue;
            }
            
            // Get output
            auto outputs = ctx.named_outputs();
            auto it = outputs.find(output_name);
            auto output_tensor = it != outputs.end() ? it->second : nullptr;
            if (!output_tensor) {
                std::cerr << "Error: Output tensor not found for " 
                          << filepath.filename() << std::endl;
                continue;
            }
            
            // Get logits
            const float* logits_data = static_cast<const float*>(output_tensor->data());
            std::vector<float> logits(logits_data, logits_data + 10);
            
            // Calculate probabilities
            auto probabilities = softmax(output_tensor);
            
            // Get prediction
            int predicted = utils::argmax(output_tensor);
            float confidence = probabilities[predicted];
            
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
            
            total++;
            bool is_correct = (predicted == label);
            if (is_correct && label != -1) correct++;
            
            // Store results
            results.push_back({
                total - 1,
                filepath.filename().string(),
                label,
                predicted,
                logits,
                probabilities,
                confidence,
                is_correct
            });
            
            // Print results
            std::cout << "Sample " << std::setw(4) << total << ": ";
            std::cout << filepath.filename().string();
            std::cout << " -> predicted=" << predicted;
            
            if (label != -1) {
                std::cout << ", label=" << label;
                std::cout << (is_correct ? " [SUCCESS]" : " [FAILED]");
            }
            
            std::cout << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing " << filepath << ": " 
                      << e.what() << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );
    
    // Print summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Total samples: " << total << std::endl;
    
    if (correct > 0) {
        float accuracy = 100.0f * correct / total;
        std::cout << "Correct: " << correct << " / " << total << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) 
                  << accuracy << "%" << std::endl;
    }
    
    float avg_time = duration.count() / static_cast<float>(total);
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average time per sample: " << std::fixed << std::setprecision(2)
              << avg_time << " ms" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Save JSON output (optional)
    if (!output_json.empty() && !results.empty()) {
        std::cout << "\nSaving outputs to JSON..." << std::endl;
        std::ofstream json_file(output_json);
        if (!json_file.is_open()) {
            std::cerr << "Error: Could not open output file: " << output_json << std::endl;
            return;
        }
        
        json_file << "{\n";
        json_file << "  \"model_type\": \"ONNX\",\n";
        json_file << "  \"samples_directory\": \"" << actual_samples_dir << "\",\n";
        json_file << "  \"total_samples\": " << total << ",\n";
        json_file << "  \"accuracy\": " << (correct * 100.0f / total) << ",\n";
        json_file << "  \"results\": [\n";
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            json_file << "    {\n";
            json_file << "      \"index\": " << r.index << ",\n";
            json_file << "      \"filename\": \"" << r.filename << "\",\n";
            json_file << "      \"label\": " << r.label << ",\n";
            json_file << "      \"predicted\": " << r.predicted << ",\n";
            
            // Logits
            json_file << "      \"logits\": [";
            for (size_t j = 0; j < r.logits.size(); ++j) {
                json_file << std::fixed << std::setprecision(6) << r.logits[j];
                if (j < r.logits.size() - 1) json_file << ", ";
            }
            json_file << "],\n";
            
            // Probabilities
            json_file << "      \"probabilities\": [";
            for (size_t j = 0; j < r.probabilities.size(); ++j) {
                json_file << std::fixed << std::setprecision(6) << r.probabilities[j];
                if (j < r.probabilities.size() - 1) json_file << ", ";
            }
            json_file << "],\n";
            
            json_file << "      \"confidence\": " << std::fixed << std::setprecision(6) 
                      << r.confidence << ",\n";
            json_file << "      \"correct\": " << (r.correct ? "true" : "false") << "\n";
            json_file << "    }";
            if (i < results.size() - 1) json_file << ",";
            json_file << "\n";
        }
        
        json_file << "  ]\n";
        json_file << "}\n";
        json_file.close();
        
        std::cout << "  [SUCCESS] Outputs saved to: " << output_json << std::endl;
    }
}

/**
 * @brief Print usage instructions
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name 
              << " <model.onnx> <samples_dir> [num_samples] [OPTIONS]" << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  model.onnx   - LeNet-5 ONNX model file" << std::endl;
    std::cout << "  samples_dir  - Directory containing MNIST test samples (.bin files)" << std::endl;
    std::cout << "  num_samples  - Number of samples to test (optional, default: all)" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --save-outputs <file>  - Save inference results to JSON file" << std::endl;
    std::cout << "  --verbose              - Enable verbose logging" << std::endl;
    std::cout << "  -h, --help             - Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " lenet5.onnx samples/" << std::endl;
    std::cout << "  " << program_name << " lenet5.onnx samples/ 100" << std::endl;
    std::cout << "  " << program_name << " lenet5.onnx samples/ --save-outputs results.json" << std::endl;
    std::cout << "  " << program_name << " lenet5.onnx samples/ --verbose" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "LeNet-5 ONNX Test - Mini-Infer" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    std::string model_path;
    std::string samples_dir;
    int num_samples = -1;
    std::string output_json;
    bool verbose = false;
    
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--save-outputs" && i + 1 < argc) {
            output_json = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
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
    std::cout << "  Verbose: " << (verbose ? "enabled" : "disabled") << std::endl;
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
        // Step 1: Parse ONNX model
        std::cout << "Step 1: Parsing ONNX Model" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        importers::OnnxParser parser;
        parser.set_verbose(verbose);
        
        // Parse to unique_ptr, need to convert to shared_ptr to match plan build interface
        auto graph_uptr = parser.parse_from_file(model_path);
        
        if (!graph_uptr) {
            std::cerr << "Failed to parse ONNX model: " << parser.get_error() << std::endl;
            return 1;
        }
        std::shared_ptr<graph::Graph> graph = std::move(graph_uptr);
        
        std::cout << "[SUCCESS] Model parsed successfully!" << std::endl;
        std::cout << "Graph has " << graph->nodes().size() << " nodes" << std::endl;
        std::cout << std::endl;
        
        // Step 2: Build Inference Plan
        std::cout << "Step 2: Building Inference Plan" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
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
        
        // Step 3: Run tests
        std::cout << "Step 3: Running Tests" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        test_lenet5_onnx(plan, *ctx, input_names[0], output_names[0], samples_dir, num_samples,
                         output_json);
        
        std::cout << "\n[SUCCESS] ONNX inference test completed successfully!" << std::endl;
        
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
