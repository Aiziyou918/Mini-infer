/**
 * @file lenet5_inference.cpp
 * @brief Complete LeNet-5 inference example using Mini-Infer
 * 
 * This example demonstrates:
 * 1. Loading binary weights exported from PyTorch
 * 2. Loading MNIST test samples
 * 3. Building LeNet-5 model with Mini-Infer operators
 * 4. Running inference and computing accuracy
 * 
 * Usage:
 *   lenet5_inference <weights_dir> <samples_dir> [num_samples]
 * 
 * Example:
 *   lenet5_inference ../models/python/lenet5/weights ../models/python/lenet5/test_samples/binary 10
 */

#include "mini_infer/mini_infer.h"
#include "mini_infer/operators/plugin_registry.h"
#include "mini_infer/operators/plugin_base.h"
#include "utils/simple_loader.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <cmath>

using namespace mini_infer;
namespace fs = std::filesystem;

/**
 * @brief LeNet-5 Model
 * 
 * Architecture:
 *   Input (1x28x28)
 *     ↓
 *   Conv1 (6 filters, 5x5) → ReLU → MaxPool(2x2)
 *     ↓ (6x12x12)
 *   Conv2 (16 filters, 5x5) → ReLU → MaxPool(2x2)
 *     ↓ (16x4x4 = 256)
 *   Flatten
 *     ↓
 *   FC1 (256→120) → ReLU
 *     ↓
 *   FC2 (120→84) → ReLU
 *     ↓
 *   FC3 (84→10)
 *     ↓
 *   Output (10 class scores)
 */
class LeNet5 {
public:
    LeNet5(const utils::LeNet5Weights& weights) {
        // Store weights
        weights_ = weights;

        // Create operators using new plugin architecture

        // Conv1: 1 → 6 channels, 5x5 kernel
        conv1_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kCONVOLUTION, core::DeviceType::CPU);
        if (!conv1_) {
            throw std::runtime_error("Failed to create Conv1 plugin");
        }
        auto conv1_param = std::make_shared<operators::Conv2DParam>();
        conv1_param->kernel_h = 5;
        conv1_param->kernel_w = 5;
        conv1_param->stride_h = 1;
        conv1_param->stride_w = 1;
        conv1_param->padding_h = 0;
        conv1_param->padding_w = 0;
        conv1_param->use_bias = true;
        conv1_->set_param(conv1_param);

        // Conv2: 6 → 16 channels, 5x5 kernel
        conv2_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kCONVOLUTION, core::DeviceType::CPU);
        if (!conv2_) {
            throw std::runtime_error("Failed to create Conv2 plugin");
        }
        auto conv2_param = std::make_shared<operators::Conv2DParam>();
        conv2_param->kernel_h = 5;
        conv2_param->kernel_w = 5;
        conv2_param->stride_h = 1;
        conv2_param->stride_w = 1;
        conv2_param->padding_h = 0;
        conv2_param->padding_w = 0;
        conv2_param->use_bias = true;
        conv2_->set_param(conv2_param);

        // Pooling: 2x2 max pooling
        pool_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kMAX_POOL, core::DeviceType::CPU);
        if (!pool_) {
            throw std::runtime_error("Failed to create Pooling plugin");
        }
        auto pool_param = std::make_shared<operators::PoolingParam>();
        pool_param->type = operators::PoolingType::MAX;
        pool_param->kernel_h = 2;
        pool_param->kernel_w = 2;
        pool_param->stride_h = 2;
        pool_param->stride_w = 2;
        pool_param->padding_h = 0;
        pool_param->padding_w = 0;
        pool_->set_param(pool_param);

        // ReLU activation
        relu_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kRELU, core::DeviceType::CPU);
        if (!relu_) {
            throw std::runtime_error("Failed to create ReLU plugin");
        }

        // Flatten
        flatten_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kFLATTEN, core::DeviceType::CPU);
        if (!flatten_) {
            throw std::runtime_error("Failed to create Flatten plugin");
        }
        auto flatten_param = std::make_shared<operators::FlattenParam>();
        flatten_param->axis = 1;
        flatten_->set_param(flatten_param);

        // Linear (Fully Connected) layers
        fc1_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kGEMM, core::DeviceType::CPU);
        if (!fc1_) {
            throw std::runtime_error("Failed to create FC1 plugin");
        }

        fc2_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kGEMM, core::DeviceType::CPU);
        if (!fc2_) {
            throw std::runtime_error("Failed to create FC2 plugin");
        }

        fc3_ = operators::PluginRegistry::instance().create_plugin(
            core::OpType::kGEMM, core::DeviceType::CPU);
        if (!fc3_) {
            throw std::runtime_error("Failed to create FC3 plugin");
        }

        std::cout << "LeNet-5 model created successfully (using plugin architecture)" << std::endl;
    }
    
    /**
     * @brief Helper function to execute a plugin
     */
    std::shared_ptr<core::Tensor> execute_plugin(
        std::shared_ptr<operators::IPlugin> plugin,
        const std::vector<std::shared_ptr<core::Tensor>>& inputs,
        const std::string& op_name
    ) {
        // Infer output shape
        std::vector<core::Shape> input_shapes;
        for (const auto& input : inputs) {
            input_shapes.push_back(input->shape());
        }

        std::vector<core::Shape> output_shapes;
        auto status = plugin->infer_output_shapes(input_shapes, output_shapes);
        if (status != core::Status::SUCCESS || output_shapes.empty()) {
            std::cerr << "Error: " << op_name << " shape inference failed" << std::endl;
            return nullptr;
        }

        // Create output tensor
        auto output = core::Tensor::create(output_shapes[0], core::DataType::FLOAT32);

        // Execute plugin
        std::vector<std::shared_ptr<core::Tensor>> outputs = {output};
        operators::PluginContext ctx;
        status = plugin->enqueue(inputs, outputs, ctx);
        if (status != core::Status::SUCCESS) {
            std::cerr << "Error: " << op_name << " forward failed" << std::endl;
            return nullptr;
        }

        return output;
    }

    /**
     * @brief Forward pass
     *
     * @param input Input tensor [batch_size, 1, 28, 28]
     * @return Output tensor [batch_size, 10]
     */
    std::shared_ptr<core::Tensor> forward(std::shared_ptr<core::Tensor> input) {
        auto x = input;

        // Conv1: [N, 1, 28, 28] -> [N, 6, 24, 24]
        x = execute_plugin(conv1_, {x, weights_.conv1_weight, weights_.conv1_bias}, "Conv1");
        if (!x) return nullptr;

        // ReLU1
        x = execute_plugin(relu_, {x}, "ReLU1");
        if (!x) return nullptr;

        // Pool1: [N, 6, 24, 24] -> [N, 6, 12, 12]
        x = execute_plugin(pool_, {x}, "Pool1");
        if (!x) return nullptr;

        // Conv2: [N, 6, 12, 12] -> [N, 16, 8, 8]
        x = execute_plugin(conv2_, {x, weights_.conv2_weight, weights_.conv2_bias}, "Conv2");
        if (!x) return nullptr;

        // ReLU2
        x = execute_plugin(relu_, {x}, "ReLU2");
        if (!x) return nullptr;

        // Pool2: [N, 16, 8, 8] -> [N, 16, 4, 4]
        x = execute_plugin(pool_, {x}, "Pool2");
        if (!x) return nullptr;

        // Flatten: [N, 16, 4, 4] → [N, 256]
        x = execute_plugin(flatten_, {x}, "Flatten");
        if (!x) return nullptr;

        // FC1: [N, 256] -> [N, 120]
        x = execute_plugin(fc1_, {x, weights_.fc1_weight, weights_.fc1_bias}, "FC1");
        if (!x) return nullptr;

        // ReLU3
        x = execute_plugin(relu_, {x}, "ReLU3");
        if (!x) return nullptr;

        // FC2: [N, 120] -> [N, 84]
        x = execute_plugin(fc2_, {x, weights_.fc2_weight, weights_.fc2_bias}, "FC2");
        if (!x) return nullptr;

        // ReLU4
        x = execute_plugin(relu_, {x}, "ReLU4");
        if (!x) return nullptr;

        // FC3: [N, 84] -> [N, 10] (no activation)
        x = execute_plugin(fc3_, {x, weights_.fc3_weight, weights_.fc3_bias}, "FC3");
        if (!x) return nullptr;

        return x;
    }
    
private:
    utils::LeNet5Weights weights_;

    // Operators (using new plugin architecture)
    std::shared_ptr<operators::IPlugin> conv1_;
    std::shared_ptr<operators::IPlugin> conv2_;
    std::shared_ptr<operators::IPlugin> pool_;
    std::shared_ptr<operators::IPlugin> relu_;
    std::shared_ptr<operators::IPlugin> fc1_;
    std::shared_ptr<operators::IPlugin> fc2_;
    std::shared_ptr<operators::IPlugin> fc3_;
    std::shared_ptr<operators::IPlugin> flatten_;
};

/**
 * @brief Compute softmax
 */
std::vector<float> softmax(const std::shared_ptr<core::Tensor>& logits) {
    const float* data = static_cast<const float*>(logits->data());
    int64_t numel = logits->shape().numel();
    
    // Find max for numerical stability
    float max_val = data[0];
    for (int64_t i = 1; i < numel; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    // Compute exp and sum
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
 * @brief Test LeNet-5 on MNIST samples
 */
void test_lenet5(LeNet5& model, 
                 const std::string& samples_dir, 
                 int num_samples = -1,
                 const std::string& output_json = "") {
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Testing LeNet-5 on MNIST Samples" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Determine the actual binary directory
    std::string actual_samples_dir = samples_dir;
    fs::path samples_path(samples_dir);
    
    // If user provided test_samples, look for binary subdirectory
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
    
    // Storage for JSON output
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
            auto input = utils::load_mnist_sample(filepath.string());
            
            // Run inference
            auto output = model.forward(input);
            
            if (!output) {
                std::cerr << "Error: Forward pass returned nullptr for " 
                          << filepath.filename() << std::endl;
                continue;
            }
            
            // Get logits
            const float* logits_data = static_cast<const float*>(output->data());
            std::vector<float> logits(logits_data, logits_data + 10);
            
            // Compute probabilities
            auto probabilities = softmax(output);
            
            // Get prediction
            int predicted = utils::argmax(output);
            float confidence = probabilities[predicted];
            
            // Extract label from filename if available
            std::string filename = filepath.stem().string();
            int label = -1;
            
            // Try to parse label from filename (sample_XXXX_label_Y format)
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
            
            // Store result
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
            
            // Print result
            std::cout << "Sample " << std::setw(4) << total << ": ";
            std::cout << filepath.filename().string();
            std::cout << " → predicted=" << predicted;
            
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
    
    // Save JSON output if requested
    if (!output_json.empty() && !results.empty()) {
        std::cout << "\nSaving outputs to JSON..." << std::endl;
        std::ofstream json_file(output_json);
        if (!json_file.is_open()) {
            std::cerr << "Error: Could not open output file: " << output_json << std::endl;
            return;
        }
        
        json_file << "{\n";
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
        
        std::cout << "  Outputs saved to: " << output_json << std::endl;
    }
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name 
              << " [weights_dir] [samples_dir] [num_samples] [OPTIONS]" << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  weights_dir  - Directory containing LeNet-5 binary weights" << std::endl;
    std::cout << "                 (default: ./models/python/lenet5/weights)" << std::endl;
    std::cout << "  samples_dir  - Directory containing MNIST test samples (.bin files)" << std::endl;
    std::cout << "                 (default: ./models/python/lenet5/test_samples/binary)" << std::endl;
    std::cout << "  num_samples  - Number of samples to test (optional, default: all)" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --save-outputs <file>  - Save inference results to JSON file" << std::endl;
    std::cout << "  -h, --help             - Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << std::endl;
    std::cout << "  " << program_name << " weights/ samples/ 10" << std::endl;
    std::cout << "  " << program_name << " --save-outputs results.json" << std::endl;
    std::cout << "  " << program_name << " weights/ samples/ --save-outputs results.json" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "LeNet-5 Inference Example - Mini-Infer" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::endl;
    
    // Auto-detect project root directory
    auto find_project_root = []() -> std::string {
        fs::path current = fs::current_path();
        
        // Try current directory and parent directories
        for (int i = 0; i < 5; ++i) {
            if (fs::exists(current / "models" / "python" / "lenet5")) {
                return current.string();
            }
            if (current.has_parent_path()) {
                current = current.parent_path();
            } else {
                break;
            }
        }
        
        // Fallback to current directory
        return fs::current_path().string();
    };
    
    std::string project_root = find_project_root();
    
    // Default parameters (for easy debugging)
    std::string weights_dir = (fs::path(project_root) / "models" / "python" / "lenet5" / "weights").string();
    std::string samples_dir = (fs::path(project_root) / "models" / "python" / "lenet5" / "test_samples" / "binary").string();
    int num_samples = -1;  // -1 means all samples
    std::string output_json = "";
    
    // Parse command line arguments (override defaults)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--save-outputs" && i + 1 < argc) {
            output_json = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (i == 1) {
            weights_dir = arg;
        } else if (i == 2) {
            samples_dir = arg;
        } else if (i == 3 && arg.find("--") != 0) {
            num_samples = std::atoi(arg.c_str());
        }
    }
    
    // Display configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Project root: " << project_root << std::endl;
    std::cout << "  Working directory: " << fs::current_path().string() << std::endl;
    std::cout << "  Weights directory: " << weights_dir << std::endl;
    std::cout << "  Samples directory: " << samples_dir << std::endl;
    if (num_samples > 0) {
        std::cout << "  Number of samples: " << num_samples << std::endl;
    } else {
        std::cout << "  Number of samples: all" << std::endl;
    }
    std::cout << std::endl;
    
    // Check directories exist
    if (!fs::exists(weights_dir)) {
        std::cerr << "Error: Weights directory not found: " << weights_dir << std::endl;
        std::cerr << "\nPlease export weights first:" << std::endl;
        std::cerr << "  cd models/python/lenet5" << std::endl;
        std::cerr << "  python export_lenet5.py --format weights" << std::endl;
        return 1;
    }
    
    if (!fs::exists(samples_dir)) {
        std::cerr << "Error: Samples directory not found: " << samples_dir << std::endl;
        std::cerr << "\nPlease export test samples first:" << std::endl;
        std::cerr << "  cd models/python/lenet5" << std::endl;
        std::cerr << "  python export_mnist_samples.py" << std::endl;
        return 1;
    }
    
    try {
        // Load weights
        std::cout << "Step 1: Loading Weights" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        auto weights = utils::LeNet5Weights::load(weights_dir);
        std::cout << std::endl;
        
        // Print weight statistics
        weights.print_stats();
        
        // Create model
        std::cout << "\nStep 2: Creating Model" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        LeNet5 model(weights);
        std::cout << std::endl;
        
        // Run inference
        std::cout << "Step 3: Running Inference" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        test_lenet5(model, samples_dir, num_samples, output_json);
        
        std::cout << "\n[SUCCESS] Inference completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
