/**
 * @file lenet5_onnx_test.cpp
 * @brief LeNet-5 ONNX 完整测试程序（对标手动版本）
 * 
 * 功能：
 * 1. 从 ONNX 文件导入 LeNet-5 模型
 * 2. 使用 Runtime Engine 构建推理引擎
 * 3. 加载 MNIST 测试样本
 * 4. 批量推理并计算准确率
 * 5. 输出详细结果和统计信息
 * 
 * Usage:
 *   lenet5_onnx_test <model.onnx> <samples_dir> [num_samples] [OPTIONS]
 * 
 * Example:
 *   lenet5_onnx_test lenet5.onnx ../models/python/lenet5/test_samples/binary 100
 */

#ifdef MINI_INFER_ONNX_ENABLED

#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/runtime/engine.h"
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
 * @brief 计算 Softmax
 */
std::vector<float> softmax(const std::shared_ptr<core::Tensor>& logits) {
    const float* data = static_cast<const float*>(logits->data());
    int64_t numel = logits->shape().numel();
    
    // 找到最大值（数值稳定性）
    float max_val = data[0];
    for (int64_t i = 1; i < numel; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    // 计算 exp 和 sum
    std::vector<float> exp_values(numel);
    float sum = 0.0f;
    for (int64_t i = 0; i < numel; ++i) {
        exp_values[i] = std::exp(data[i] - max_val);
        sum += exp_values[i];
    }
    
    // 归一化
    for (int64_t i = 0; i < numel; ++i) {
        exp_values[i] /= sum;
    }
    
    return exp_values;
}

/**
 * @brief 测试 LeNet-5 ONNX 模型
 */
void test_lenet5_onnx(
    std::shared_ptr<runtime::Engine> engine,
    const std::string& input_name,
    const std::string& output_name,
    const std::string& samples_dir,
    int num_samples = -1,
    const std::string& output_json = ""
) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Testing LeNet-5 ONNX Model on MNIST Samples" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // 确定实际的样本目录
    std::string actual_samples_dir = samples_dir;
    fs::path samples_path(samples_dir);
    
    if (fs::exists(samples_path / "binary")) {
        actual_samples_dir = (samples_path / "binary").string();
        std::cout << "\nNote: Using binary subdirectory: " << actual_samples_dir << std::endl;
    }
    
    // 获取所有样本文件
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
    
    // 存储结果
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
            // 加载样本
            auto input_tensor = utils::load_mnist_sample(filepath.string());
            
            // 准备输入
            std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs;
            inputs[input_name] = input_tensor;
            
            // 执行推理
            std::unordered_map<std::string, std::shared_ptr<core::Tensor>> outputs;
            auto status = engine->forward(inputs, outputs);
            
            if (status != core::Status::SUCCESS) {
                std::cerr << "Error: Inference failed for " 
                          << filepath.filename() << std::endl;
                continue;
            }
            
            // 获取输出
            auto output_tensor = outputs[output_name];
            if (!output_tensor) {
                std::cerr << "Error: Output tensor not found for " 
                          << filepath.filename() << std::endl;
                continue;
            }
            
            // 获取 logits
            const float* logits_data = static_cast<const float*>(output_tensor->data());
            std::vector<float> logits(logits_data, logits_data + 10);
            
            // 计算概率
            auto probabilities = softmax(output_tensor);
            
            // 获取预测
            int predicted = utils::argmax(output_tensor);
            float confidence = probabilities[predicted];
            
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
            
            total++;
            bool is_correct = (predicted == label);
            if (is_correct && label != -1) correct++;
            
            // 存储结果
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
            
            // 打印结果
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
    
    // 打印摘要
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
    
    // 保存 JSON 输出（可选）
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
 * @brief 打印使用说明
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
    
    // 解析命令行参数
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
    
    // 显示配置
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
    
    // 检查文件存在
    if (!fs::exists(model_path)) {
        std::cerr << "Error: ONNX model file not found: " << model_path << std::endl;
        return 1;
    }
    
    if (!fs::exists(samples_dir)) {
        std::cerr << "Error: Samples directory not found: " << samples_dir << std::endl;
        return 1;
    }
    
    try {
        // Step 1: 解析 ONNX 模型
        std::cout << "Step 1: Parsing ONNX Model" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        importers::OnnxParser parser;
        parser.set_verbose(verbose);
        
        // 解析得到 unique_ptr，需要转换为 shared_ptr 以匹配 Engine::build 接口
        auto graph_uptr = parser.parse_from_file(model_path);
        
        if (!graph_uptr) {
            std::cerr << "Failed to parse ONNX model: " << parser.get_error() << std::endl;
            return 1;
        }
        std::shared_ptr<graph::Graph> graph = std::move(graph_uptr);
        
        std::cout << "[SUCCESS] Model parsed successfully!" << std::endl;
        std::cout << "Graph has " << graph->nodes().size() << " nodes" << std::endl;
        std::cout << std::endl;
        
        // Step 2: 构建 Runtime 引擎
        std::cout << "Step 2: Building Runtime Engine" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        runtime::EngineConfig config;
        config.device_type = core::DeviceType::CPU;
        config.enable_profiling = false;
        
        auto engine = std::make_shared<runtime::Engine>(config);
        auto status = engine->build(graph);
        
        if (status != core::Status::SUCCESS) {
            std::cerr << "Failed to build engine" << std::endl;
            return 1;
        }
        
        std::cout << "[SUCCESS] Engine built successfully!" << std::endl;
        
        // 获取输入输出名称
        auto input_names = engine->get_input_names();
        auto output_names = engine->get_output_names();
        
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
        
        // Step 3: 运行测试
        std::cout << "Step 3: Running Tests" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        test_lenet5_onnx(
            engine,
            input_names[0],
            output_names[0],
            samples_dir,
            num_samples,
            output_json
        );
        
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
