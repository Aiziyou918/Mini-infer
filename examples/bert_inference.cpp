/**
 * @file bert_inference.cpp
 * @brief BERT ONNX Inference Example for Mini-Infer
 *
 * Features:
 * 1. Import Tiny-BERT model from ONNX file
 * 2. Build inference engine using Runtime Engine
 * 3. Load pre-tokenized test samples (input_ids, attention_mask, token_type_ids)
 * 4. Execute inference and output classification results
 * 5. Compare with PyTorch reference outputs
 *
 * Usage:
 *   bert_inference <model.onnx> <samples_dir> [num_samples] [OPTIONS]
 *
 * Example:
 *   bert_inference bert_tiny.onnx ../models/python/bert/test_samples 10
 */

#ifdef MINI_INFER_ONNX_ENABLED

#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/runtime/execution_context.h"
#include "mini_infer/runtime/inference_plan.h"
#include "mini_infer/utils/logger.h"

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

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Load BERT input tensor from binary file (INT64)
 */
std::shared_ptr<core::Tensor> load_int64_tensor(
    const std::string& filepath,
    const std::vector<int64_t>& shape
) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + filepath);
    }

    // Calculate expected size
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    size_t expected_bytes = static_cast<size_t>(numel) * sizeof(int64_t);

    // Create tensor with INT64 type
    auto tensor = core::Tensor::create(
        core::Shape(shape),
        core::DataType::INT64
    );

    // Read data directly into tensor
    file.read(reinterpret_cast<char*>(tensor->data()), expected_bytes);

    if (!file) {
        throw std::runtime_error("Failed to read data from: " + filepath);
    }

    return tensor;
}

/**
 * @brief Calculate softmax probabilities
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
    std::vector<float> exp_values(static_cast<size_t>(numel));
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
 * @brief Get argmax of tensor
 */
int argmax(const std::shared_ptr<core::Tensor>& tensor) {
    const float* data = static_cast<const float*>(tensor->data());
    int64_t numel = tensor->shape().numel();

    int max_idx = 0;
    float max_val = data[0];

    for (int64_t i = 1; i < numel; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = static_cast<int>(i);
        }
    }

    return max_idx;
}

// =============================================================================
// BERT Sample Structure
// =============================================================================

/**
 * @brief BERT sample information
 */
struct BertSample {
    int index;
    std::string text;
    int label;
    std::string label_name;
    std::string input_ids_path;
    std::string attention_mask_path;
    std::string token_type_ids_path;
    std::vector<int64_t> shape;
};

/**
 * @brief Load sample metadata from directory
 *
 * Scans the binary directory for sample files and extracts metadata.
 */
std::vector<BertSample> load_samples(const std::string& samples_dir, int max_samples = -1) {
    std::vector<BertSample> samples;

    fs::path binary_dir = fs::path(samples_dir) / "binary";

    if (!fs::exists(binary_dir)) {
        throw std::runtime_error("Binary directory not found: " + binary_dir.string());
    }

    // Find all input_ids files
    std::vector<fs::path> input_files;
    for (const auto& entry : fs::directory_iterator(binary_dir)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("_input_ids.bin") != std::string::npos) {
            input_files.push_back(entry.path());
        }
    }

    std::sort(input_files.begin(), input_files.end());

    if (max_samples > 0 && static_cast<size_t>(max_samples) < input_files.size()) {
        input_files.resize(static_cast<size_t>(max_samples));
    }

    for (size_t i = 0; i < input_files.size(); ++i) {
        BertSample sample;
        sample.index = static_cast<int>(i);

        std::string stem = input_files[i].stem().string();
        std::string base = stem.substr(0, stem.find("_input_ids"));

        sample.input_ids_path = (binary_dir / (base + "_input_ids.bin")).string();
        sample.attention_mask_path = (binary_dir / (base + "_attention_mask.bin")).string();
        sample.token_type_ids_path = (binary_dir / (base + "_token_type_ids.bin")).string();

        // Default shape: batch=1, seq_len=128
        sample.shape = {1, 128};
        sample.label = -1;  // Unknown until we load metadata
        sample.label_name = "unknown";

        samples.push_back(sample);
    }

    // Try to load metadata JSON for labels
    fs::path metadata_path = fs::path(samples_dir) / "samples_metadata.json";
    if (fs::exists(metadata_path)) {
        std::ifstream meta_file(metadata_path);
        if (meta_file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(meta_file)),
                                std::istreambuf_iterator<char>());

            // Simple JSON parsing for labels (basic implementation)
            for (auto& sample : samples) {
                std::string search_key = "\"index\": " + std::to_string(sample.index);
                size_t pos = content.find(search_key);
                if (pos != std::string::npos) {
                    // Find label
                    size_t label_pos = content.find("\"label\":", pos);
                    if (label_pos != std::string::npos && label_pos < pos + 500) {
                        size_t num_start = content.find_first_of("0123456789", label_pos);
                        if (num_start != std::string::npos) {
                            sample.label = std::stoi(content.substr(num_start, 1));
                            sample.label_name = (sample.label == 1) ? "positive" : "negative";
                        }
                    }

                    // Find text
                    size_t text_pos = content.find("\"text\":", pos);
                    if (text_pos != std::string::npos && text_pos < pos + 500) {
                        size_t quote_start = content.find('"', text_pos + 7);
                        size_t quote_end = content.find('"', quote_start + 1);
                        if (quote_start != std::string::npos && quote_end != std::string::npos) {
                            sample.text = content.substr(quote_start + 1, quote_end - quote_start - 1);
                        }
                    }

                    // Find shape
                    size_t shape_pos = content.find("\"shape\":", pos);
                    if (shape_pos != std::string::npos && shape_pos < pos + 500) {
                        size_t bracket_start = content.find('[', shape_pos);
                        size_t bracket_end = content.find(']', bracket_start);
                        if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
                            std::string shape_str = content.substr(bracket_start + 1,
                                                                   bracket_end - bracket_start - 1);
                            // Parse [1, 128]
                            size_t comma = shape_str.find(',');
                            if (comma != std::string::npos) {
                                sample.shape[0] = std::stoll(shape_str.substr(0, comma));
                                sample.shape[1] = std::stoll(shape_str.substr(comma + 1));
                            }
                        }
                    }
                }
            }
        }
    }

    return samples;
}

// =============================================================================
// BERT Inference Test
// =============================================================================

/**
 * @brief Test BERT model on samples
 */
void test_bert_model(
    std::shared_ptr<runtime::InferencePlan> plan,
    runtime::ExecutionContext& ctx,
    const std::vector<std::string>& input_names,
    const std::string& output_name,
    const std::string& samples_dir,
    int num_samples = -1,
    const std::string& output_json = ""
) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Testing BERT Model on Text Classification Samples" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Load samples
    auto samples = load_samples(samples_dir, num_samples);

    std::cout << "\nLoaded " << samples.size() << " samples from: " << samples_dir << std::endl;
    std::cout << "Input shape: [" << samples[0].shape[0] << ", " << samples[0].shape[1] << "]" << std::endl;
    std::cout << std::endl;

    // Store results
    struct SampleResult {
        int index;
        std::string text;
        int label;
        int predicted;
        std::string predicted_name;
        std::vector<float> logits;
        std::vector<float> probabilities;
        float confidence;
        bool correct;
    };
    std::vector<SampleResult> results;

    int correct = 0;
    int total = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& sample : samples) {
        try {
            // Load inputs
            auto input_ids = load_int64_tensor(sample.input_ids_path, sample.shape);
            auto attention_mask = load_int64_tensor(sample.attention_mask_path, sample.shape);
            auto token_type_ids = load_int64_tensor(sample.token_type_ids_path, sample.shape);

            // Prepare inputs map
            std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;

            // Map input names (BERT typically has: input_ids, attention_mask, token_type_ids)
            for (const auto& name : input_names) {
                if (name.find("input_ids") != std::string::npos || name == "input_ids") {
                    inputs[name] = input_ids;
                } else if (name.find("attention_mask") != std::string::npos || name == "attention_mask") {
                    inputs[name] = attention_mask;
                } else if (name.find("token_type_ids") != std::string::npos || name == "token_type_ids") {
                    inputs[name] = token_type_ids;
                }
            }

            // Execute inference
            auto status = ctx.set_inputs(inputs);
            if (status == core::Status::SUCCESS) {
                status = plan->execute(&ctx);
            }

            if (status != core::Status::SUCCESS) {
                std::cerr << "Error: Inference failed for sample " << sample.index << std::endl;
                continue;
            }

            // Get output
            auto outputs = ctx.named_outputs();
            auto it = outputs.find(output_name);
            auto output_tensor = it != outputs.end() ? it->second : nullptr;

            if (!output_tensor) {
                // Try to find output by partial match
                for (const auto& [name, tensor] : outputs) {
                    if (name.find("logits") != std::string::npos) {
                        output_tensor = tensor;
                        break;
                    }
                }
            }

            if (!output_tensor) {
                std::cerr << "Error: Output tensor not found for sample " << sample.index << std::endl;
                continue;
            }

            // Get logits
            const float* logits_data = static_cast<const float*>(output_tensor->data());
            int64_t num_classes = output_tensor->shape().numel();
            std::vector<float> logits(logits_data, logits_data + num_classes);

            // Calculate probabilities
            auto probabilities = softmax(output_tensor);

            // Get prediction
            int predicted = argmax(output_tensor);
            float confidence = probabilities[predicted];
            std::string predicted_name = (predicted == 1) ? "positive" : "negative";

            total++;

            bool is_correct = (sample.label >= 0 && predicted == sample.label);
            if (is_correct) correct++;

            // Store result
            results.push_back({
                sample.index,
                sample.text,
                sample.label,
                predicted,
                predicted_name,
                logits,
                probabilities,
                confidence,
                is_correct
            });

            // Print result
            std::cout << "Sample " << std::setw(4) << sample.index << ": ";
            std::cout << "pred=" << predicted << " (" << std::setw(8) << predicted_name << ")";
            std::cout << ", conf=" << std::fixed << std::setprecision(4) << confidence;

            if (sample.label >= 0) {
                std::cout << ", label=" << sample.label << " (" << std::setw(8) << sample.label_name << ")";
                std::cout << (is_correct ? " [CORRECT]" : " [WRONG]");
            }

            std::cout << std::endl;

            // Print text preview
            if (!sample.text.empty()) {
                std::string preview = sample.text.substr(0, 50);
                if (sample.text.length() > 50) preview += "...";
                std::cout << "         \"" << preview << "\"" << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Error processing sample " << sample.index << ": " << e.what() << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Total samples: " << total << std::endl;

    if (correct > 0 || total > 0) {
        float accuracy = (total > 0) ? (100.0f * correct / total) : 0.0f;
        std::cout << "Correct: " << correct << " / " << total << std::endl;
        std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    }

    float avg_time = (total > 0) ? (duration.count() / static_cast<float>(total)) : 0.0f;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average time per sample: " << std::fixed << std::setprecision(2) << avg_time << " ms" << std::endl;
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
        json_file << "  \"model_type\": \"BERT-ONNX\",\n";
        json_file << "  \"samples_directory\": \"" << samples_dir << "\",\n";
        json_file << "  \"total_samples\": " << total << ",\n";
        json_file << "  \"correct\": " << correct << ",\n";
        json_file << "  \"accuracy\": " << std::fixed << std::setprecision(2)
                  << (total > 0 ? (correct * 100.0f / total) : 0.0f) << ",\n";
        json_file << "  \"total_time_ms\": " << duration.count() << ",\n";
        json_file << "  \"results\": [\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            json_file << "    {\n";
            json_file << "      \"index\": " << r.index << ",\n";
            json_file << "      \"text\": \"" << r.text << "\",\n";
            json_file << "      \"label\": " << r.label << ",\n";
            json_file << "      \"predicted\": " << r.predicted << ",\n";
            json_file << "      \"predicted_name\": \"" << r.predicted_name << "\",\n";

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

            json_file << "      \"confidence\": " << std::fixed << std::setprecision(6) << r.confidence << ",\n";
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

// =============================================================================
// Main Function
// =============================================================================

/**
 * @brief Print usage instructions
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name
              << " <model.onnx> <samples_dir> [num_samples] [OPTIONS]" << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  model.onnx   - BERT ONNX model file" << std::endl;
    std::cout << "  samples_dir  - Directory containing tokenized test samples" << std::endl;
    std::cout << "  num_samples  - Number of samples to test (optional, default: all)" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --save-outputs <file>  - Save inference results to JSON file" << std::endl;
    std::cout << "  --verbose              - Enable verbose logging" << std::endl;
    std::cout << "  -h, --help             - Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " bert_tiny.onnx ../models/python/bert/test_samples" << std::endl;
    std::cout << "  " << program_name << " bert_tiny.onnx samples/ 10" << std::endl;
    std::cout << "  " << program_name << " bert_tiny.onnx samples/ --save-outputs results.json" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "BERT ONNX Inference - Mini-Infer" << std::endl;
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
        // Step 1: Parse ONNX model
        std::cout << "Step 1: Parsing ONNX Model" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        importers::OnnxParser parser;
        parser.set_verbose(verbose);

        auto graph_uptr = parser.parse_from_file(model_path);

        if (!graph_uptr) {
            std::cerr << "Failed to parse ONNX model: " << parser.get_error() << std::endl;
            return 1;
        }
        std::shared_ptr<graph::Graph> graph = std::move(graph_uptr);

        std::cout << "[SUCCESS] Model parsed successfully!" << std::endl;
        std::cout << "Graph has " << graph->node_count() << " nodes" << std::endl;
        std::cout << std::endl;

        // Step 2: Build Inference Plan
        std::cout << "Step 2: Building Inference Plan" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        runtime::EngineConfig config;
        config.device_type = core::DeviceType::CPU;
        config.enable_profiling = true;
        config.enable_graph_optimization = true;
        config.enable_memory_planning = true;

        // Enable dynamic shapes with a fixed profile to avoid undefined shapes in BERT graphs.
        config.enable_dynamic_shapes = true;
        auto profile = std::make_shared<runtime::OptimizationProfile>();
        const core::Shape fixed_shape({1, 128});
        auto input_names_from_graph = graph->inputs();
        if (input_names_from_graph.empty()) {
            input_names_from_graph = {"input_ids", "attention_mask", "token_type_ids"};
        }
        for (const auto& name : input_names_from_graph) {
            profile->set_shape_range(name, fixed_shape, fixed_shape, fixed_shape);
            profile->set_input_dtype(name, core::DataType::INT64);
        }
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

        std::cout << "  Inputs (" << input_names.size() << "): ";
        for (const auto& name : input_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;

        std::cout << "  Outputs (" << output_names.size() << "): ";
        for (const auto& name : output_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl << std::endl;

        if (input_names.empty() || output_names.empty()) {
            std::cerr << "Error: Model has no inputs or outputs!" << std::endl;
            return 1;
        }

        // Step 3: Run tests
        std::cout << "Step 3: Running BERT Inference Tests" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        test_bert_model(plan, *ctx, input_names, output_names[0], samples_dir, num_samples, output_json);

        std::cout << "\n[SUCCESS] BERT ONNX inference test completed!" << std::endl;

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
