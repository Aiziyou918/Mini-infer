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
        
        // Create operators
        // Conv1: 1 → 6 channels, 5x5 kernel
        conv1_param_ = operators::Conv2DParam(
            5, 5,  // kernel_h, kernel_w
            1, 1,  // stride_h, stride_w
            0, 0,  // padding_h, padding_w
            1,     // groups
            true   // use_bias
        );
        conv1_ = std::make_shared<operators::Conv2D>(conv1_param_);
        
        // Conv2: 6 → 16 channels, 5x5 kernel
        conv2_param_ = operators::Conv2DParam(
            5, 5,  // kernel_h, kernel_w
            1, 1,  // stride_h, stride_w
            0, 0,  // padding_h, padding_w
            1,     // groups
            true   // use_bias
        );
        conv2_ = std::make_shared<operators::Conv2D>(conv2_param_);
        
        // Pooling: 2x2 max pooling
        pool_param_ = operators::PoolingParam(
            operators::PoolingType::MAX,
            2, 2,  // kernel_h, kernel_w
            2, 2,  // stride_h, stride_w
            0, 0   // padding_h, padding_w
        );
        pool_ = std::make_shared<operators::Pooling>(pool_param_);
        
        // ReLU activation
        relu_ = std::make_shared<operators::ReLU>();
        
        // Linear (Fully Connected) layers
        fc1_ = std::make_shared<operators::Linear>(operators::LinearParam(256, 120, true));
        fc2_ = std::make_shared<operators::Linear>(operators::LinearParam(120, 84, true));
        fc3_ = std::make_shared<operators::Linear>(operators::LinearParam(84, 10, true));
        
        std::cout << "LeNet-5 model created successfully" << std::endl;
    }
    
    /**
     * @brief Forward pass
     * 
     * @param input Input tensor [batch_size, 1, 28, 28]
     * @return Output tensor [batch_size, 10]
     */
    std::shared_ptr<core::Tensor> forward(std::shared_ptr<core::Tensor> input) {
        auto x = input;
        
        // Conv1 + ReLU + Pool
        std::vector<std::shared_ptr<core::Tensor>> conv1_outputs;
        auto status = conv1_->forward({x, weights_.conv1_weight, weights_.conv1_bias}, conv1_outputs);
        if (status != core::Status::SUCCESS || conv1_outputs.empty()) {
            std::cerr << "Error: Conv1 forward failed (status=" << static_cast<int>(status) << ")" << std::endl;
            return nullptr;
        }
        x = conv1_outputs[0];  // [N, 6, 24, 24]
        
        std::vector<std::shared_ptr<core::Tensor>> relu1_outputs;
        status = relu_->forward({x}, relu1_outputs);
        if (status != core::Status::SUCCESS || relu1_outputs.empty()) {
            std::cerr << "Error: ReLU1 forward failed" << std::endl;
            return nullptr;
        }
        x = relu1_outputs[0];
        
        std::vector<std::shared_ptr<core::Tensor>> pool1_outputs;
        status = pool_->forward({x}, pool1_outputs);
        if (status != core::Status::SUCCESS || pool1_outputs.empty()) {
            std::cerr << "Error: Pool1 forward failed" << std::endl;
            return nullptr;
        }
        x = pool1_outputs[0];  // [N, 6, 12, 12]
        
        // Conv2 + ReLU + Pool
        std::vector<std::shared_ptr<core::Tensor>> conv2_outputs;
        status = conv2_->forward({x, weights_.conv2_weight, weights_.conv2_bias}, conv2_outputs);
        if (status != core::Status::SUCCESS || conv2_outputs.empty()) {
            std::cerr << "Error: Conv2 forward failed" << std::endl;
            return nullptr;
        }
        x = conv2_outputs[0];  // [N, 16, 8, 8]
        
        std::vector<std::shared_ptr<core::Tensor>> relu2_outputs;
        status = relu_->forward({x}, relu2_outputs);
        if (status != core::Status::SUCCESS || relu2_outputs.empty()) {
            std::cerr << "Error: ReLU2 forward failed" << std::endl;
            return nullptr;
        }
        x = relu2_outputs[0];
        
        std::vector<std::shared_ptr<core::Tensor>> pool2_outputs;
        status = pool_->forward({x}, pool2_outputs);
        if (status != core::Status::SUCCESS || pool2_outputs.empty()) {
            std::cerr << "Error: Pool2 forward failed" << std::endl;
            return nullptr;
        }
        x = pool2_outputs[0];  // [N, 16, 4, 4]
        
        // Flatten: [N, 16, 4, 4] → [N, 256]
        int batch_size = x->shape()[0];
        x = reshape(x, {batch_size, 256});
        
        // FC1 + ReLU
        std::vector<std::shared_ptr<core::Tensor>> fc1_outputs;
        status = fc1_->forward({x, weights_.fc1_weight, weights_.fc1_bias}, fc1_outputs);
        if (status != core::Status::SUCCESS || fc1_outputs.empty()) {
            std::cerr << "Error: FC1 forward failed" << std::endl;
            return nullptr;
        }
        x = fc1_outputs[0];  // [N, 120]
        
        std::vector<std::shared_ptr<core::Tensor>> relu3_outputs;
        status = relu_->forward({x}, relu3_outputs);
        if (status != core::Status::SUCCESS || relu3_outputs.empty()) {
            std::cerr << "Error: ReLU3 forward failed" << std::endl;
            return nullptr;
        }
        x = relu3_outputs[0];
        
        // FC2 + ReLU
        std::vector<std::shared_ptr<core::Tensor>> fc2_outputs;
        status = fc2_->forward({x, weights_.fc2_weight, weights_.fc2_bias}, fc2_outputs);
        if (status != core::Status::SUCCESS || fc2_outputs.empty()) {
            std::cerr << "Error: FC2 forward failed" << std::endl;
            return nullptr;
        }
        x = fc2_outputs[0];  // [N, 84]
        
        std::vector<std::shared_ptr<core::Tensor>> relu4_outputs;
        status = relu_->forward({x}, relu4_outputs);
        if (status != core::Status::SUCCESS || relu4_outputs.empty()) {
            std::cerr << "Error: ReLU4 forward failed" << std::endl;
            return nullptr;
        }
        x = relu4_outputs[0];
        
        // FC3 (no activation)
        std::vector<std::shared_ptr<core::Tensor>> fc3_outputs;
        status = fc3_->forward({x, weights_.fc3_weight, weights_.fc3_bias}, fc3_outputs);
        if (status != core::Status::SUCCESS || fc3_outputs.empty()) {
            std::cerr << "Error: FC3 forward failed" << std::endl;
            return nullptr;
        }
        x = fc3_outputs[0];  // [N, 10]
        
        return x;
    }
    
private:
    utils::LeNet5Weights weights_;
    
    // Operators
    std::shared_ptr<operators::Conv2D> conv1_;
    std::shared_ptr<operators::Conv2D> conv2_;
    std::shared_ptr<operators::Pooling> pool_;
    std::shared_ptr<operators::ReLU> relu_;
    std::shared_ptr<operators::Linear> fc1_;
    std::shared_ptr<operators::Linear> fc2_;
    std::shared_ptr<operators::Linear> fc3_;
    
    // Parameters
    operators::Conv2DParam conv1_param_;
    operators::Conv2DParam conv2_param_;
    operators::PoolingParam pool_param_;
    
    /**
     * @brief Reshape tensor
     */
    std::shared_ptr<core::Tensor> reshape(
        std::shared_ptr<core::Tensor> input,
        std::vector<int64_t> new_shape
    ) {
        auto output = core::Tensor::create(
            core::Shape(new_shape),
            input->dtype()
        );
        
        // Simple memcpy (assumes contiguous memory and same total size)
        std::memcpy(output->data(), input->data(), 
                    input->shape().numel() * sizeof(float));
        
        return output;
    }
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
