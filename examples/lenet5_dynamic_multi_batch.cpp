#include "mini_infer/core/tensor.h"
#include "mini_infer/graph/graph.h"
#include "mini_infer/importers/onnx_parser.h"
#include "mini_infer/runtime/execution_context.h"
#include "mini_infer/runtime/inference_plan.h"
#include "mini_infer/runtime/optimization_profile.h"
#include "mini_infer/utils/logger.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

using namespace mini_infer;
namespace fs = std::filesystem;

namespace {

struct Sample {
    std::string filename;
    int label{-1};
    std::vector<float> data;
};

struct BatchStats {
    size_t requested{0};
    size_t labeled{0};
    size_t correct{0};
    double time_ms{0.0};
};

std::vector<float> read_sample_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        MI_LOG_ERROR("Failed to open sample file: " + path.string());
        return {};
    }
    file.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

int extract_label(const std::string& filename) {
    const std::string token = "_label_";
    const auto pos = filename.find(token);
    if (pos == std::string::npos) {
        return -1;
    }
    const auto start = pos + token.size();
    const auto end = filename.find('.', start);
    if (end == std::string::npos) {
        return -1;
    }
    try {
        return std::stoi(filename.substr(start, end - start));
    } catch (...) {
        return -1;
    }
}

std::vector<Sample> load_samples(const std::string& directory, size_t min_samples) {
    std::vector<Sample> samples;
    if (!fs::exists(directory)) {
        MI_LOG_ERROR("Samples directory does not exist: " + directory);
        return samples;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& file : files) {
        Sample sample;
        sample.filename = file.filename().string();
        sample.label = extract_label(sample.filename);
        sample.data = read_sample_file(file);
        if (!sample.data.empty()) {
            samples.push_back(std::move(sample));
        }
    }

    if (samples.size() < min_samples) {
        MI_LOG_WARNING("Only " + std::to_string(samples.size()) +
                       " samples available, expected at least " + std::to_string(min_samples) +
                       ". Samples will be re-used for larger batches.");
    }

    return samples;
}

std::shared_ptr<core::Tensor> create_batch_tensor(const std::vector<Sample>& samples,
                                                  size_t batch_size, size_t offset) {
    constexpr size_t sample_elements = 1 * 28 * 28;
    auto tensor = std::make_shared<core::Tensor>(
        core::Shape({static_cast<int64_t>(batch_size), 1, 28, 28}),
        core::DataType::FLOAT32);

    auto* dst = static_cast<float*>(tensor->data());
    if (!dst) {
        MI_LOG_ERROR("Failed to allocate batch tensor memory");
        return nullptr;
    }

    for (size_t b = 0; b < batch_size; ++b) {
        const auto& sample = samples[(offset + b) % samples.size()];
        if (sample.data.size() != sample_elements) {
            MI_LOG_ERROR("Sample '" + sample.filename + "' has unexpected size " +
                         std::to_string(sample.data.size()));
            return nullptr;
        }
        std::memcpy(dst + b * sample_elements, sample.data.data(),
                    sample_elements * sizeof(float));
    }

    return tensor;
}

BatchStats run_dynamic_batch(runtime::InferencePlan& plan, runtime::ExecutionContext& ctx,
                             const std::string& input_name, const std::string& output_name,
                             const std::vector<Sample>& samples, size_t batch_size,
                             size_t offset) {
    BatchStats stats;
    stats.requested = batch_size;

    auto batch_tensor = create_batch_tensor(samples, batch_size, offset);
    if (!batch_tensor) {
        return stats;
    }

    std::unordered_map<std::string, std::shared_ptr<core::Tensor>> inputs{{input_name, batch_tensor}};
    auto start = std::chrono::high_resolution_clock::now();
    auto status = ctx.set_inputs(inputs);
    if (status == core::Status::SUCCESS) {
        status = plan.execute(&ctx);
    }
    auto end = std::chrono::high_resolution_clock::now();
    stats.time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("Forward pass failed for batch size " + std::to_string(batch_size));
        return stats;
    }

    auto outputs = ctx.named_outputs();
    auto output_it = outputs.find(output_name);
    if (output_it == outputs.end() || !output_it->second) {
        MI_LOG_ERROR("Output tensor '" + output_name + "' not found");
        return stats;
    }

    auto output_tensor = output_it->second;
    const auto total_elements = static_cast<size_t>(output_tensor->shape().numel());
    if (batch_size == 0 || total_elements == 0 || total_elements % batch_size != 0) {
        MI_LOG_ERROR("Unexpected output tensor shape: " + output_tensor->shape().to_string());
        return stats;
    }

    const size_t per_sample = total_elements / batch_size;
    const float* out_data = static_cast<const float*>(output_tensor->data());
    if (!out_data) {
        MI_LOG_ERROR("Output tensor has no data");
        return stats;
    }

    for (size_t b = 0; b < batch_size; ++b) {
        const auto& sample = samples[(offset + b) % samples.size()];
        if (sample.label < 0) {
            continue;
        }
        stats.labeled++;

        const float* logits = out_data + b * per_sample;
        const auto max_it = std::max_element(logits, logits + per_sample);
        const int predicted = static_cast<int>(std::distance(logits, max_it));
        if (predicted == sample.label) {
            stats.correct++;
        }
    }

    MI_LOG_INFO("Batch size " + std::to_string(batch_size) + ": latency=" +
                std::to_string(stats.time_ms) + " ms, labeled=" +
                std::to_string(stats.labeled) + ", correct=" + std::to_string(stats.correct));

    return stats;
}

std::shared_ptr<runtime::OptimizationProfile> build_profile() {
    auto profile = std::make_shared<runtime::OptimizationProfile>();
    profile->set_shape_range(
        "input",
        core::Shape({1, 1, 28, 28}),    // min
        core::Shape({8, 1, 28, 28}),    // optimal
        core::Shape({16, 1, 28, 28})    // max
    );
    return profile;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string model_path = "models/lenet5.onnx";
    std::string samples_dir = "models/python/lenet5/test_samples";
    float accuracy_threshold = 0.9f;
    size_t max_batch = 16;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--samples" && i + 1 < argc) {
            samples_dir = argv[++i];
        } else if (arg == "--accuracy-threshold" && i + 1 < argc) {
            accuracy_threshold = std::stof(argv[++i]);
        } else if (arg == "--max-batch" && i + 1 < argc) {
            max_batch = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " [--model path] [--samples dir] [--accuracy-threshold value]\n";
            return 0;
        }
    }

    MI_LOG_INFO("========================================");
    MI_LOG_INFO("LeNet-5 Dynamic Multi-Batch Demo");
    MI_LOG_INFO("========================================");

    mini_infer::importers::OnnxParser parser;
    auto graph_ptr = parser.parse_from_file(model_path);
    if (!graph_ptr) {
        MI_LOG_ERROR("Failed to load ONNX model: " + parser.get_error());
        return 1;
    }
    auto graph = std::shared_ptr<graph::Graph>(std::move(graph_ptr));

    runtime::EngineConfig config;
    config.enable_graph_optimization = true;
    config.enable_memory_planning = true;
    config.enable_dynamic_shapes = true;
    config.optimization_profile = build_profile();
    config.enable_profiling = true;

    auto plan = std::make_shared<runtime::InferencePlan>(config);
    auto status = plan->build(graph);
    if (status != core::Status::SUCCESS) {
        MI_LOG_ERROR("Failed to build plan");
        return 1;
    }

    auto ctx = plan->create_execution_context();
    if (!ctx) {
        MI_LOG_ERROR("Failed to create execution context");
        return 1;
    }

    const auto input_names = plan->get_input_names();
    const auto output_names = plan->get_output_names();
    if (input_names.empty() || output_names.empty()) {
        MI_LOG_ERROR("Engine does not expose input/output names");
        return 1;
    }

    const auto input_name = input_names[0];
    const auto output_name = output_names[0];

    auto samples = load_samples(samples_dir, max_batch);
    if (samples.empty()) {
        MI_LOG_ERROR("No valid samples available for inference");
        return 1;
    }

    std::vector<size_t> batch_sizes = {1, 4, 8, 12, std::min<size_t>(16, max_batch)};
    batch_sizes.erase(std::remove_if(batch_sizes.begin(), batch_sizes.end(),
                                     [max_batch](size_t b) { return b > max_batch; }),
                      batch_sizes.end());

    size_t offset = 0;
    size_t total_labeled = 0;
    size_t total_correct = 0;
    size_t total_requested = 0;
    double total_time_ms = 0.0;

    for (auto batch_size : batch_sizes) {
        auto stats =
            run_dynamic_batch(*plan, *ctx, input_name, output_name, samples, batch_size, offset);
        offset += batch_size;
        total_labeled += stats.labeled;
        total_correct += stats.correct;
        total_requested += stats.requested;
        total_time_ms += stats.time_ms;
    }

    const float accuracy =
        total_labeled > 0 ? static_cast<float>(total_correct) / static_cast<float>(total_labeled)
                          : 0.0f;

    MI_LOG_INFO("");
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("Summary");
    MI_LOG_INFO("========================================");
    MI_LOG_INFO("Batches processed: " + std::to_string(batch_sizes.size()));
    MI_LOG_INFO("Total samples requested: " + std::to_string(total_requested));
    MI_LOG_INFO("Total labeled: " + std::to_string(total_labeled));
    MI_LOG_INFO("Correct predictions: " + std::to_string(total_correct));
    MI_LOG_INFO("Overall accuracy: " + std::to_string(accuracy * 100.0f) + "%");
    MI_LOG_INFO("Total latency: " + std::to_string(total_time_ms) + " ms");

    const auto& memory_plan = plan->get_memory_plan();
    if (!memory_plan.pools.empty()) {
        MI_LOG_INFO("Memory planning pools: " + std::to_string(memory_plan.pools.size()) +
                    ", saving=" + std::to_string(memory_plan.memory_saving_ratio * 100.0f) + "%");
    }

    if (accuracy_threshold > 0.0f && accuracy < accuracy_threshold) {
        MI_LOG_ERROR("Accuracy " + std::to_string(accuracy * 100.0f) +
                     "% is below threshold " + std::to_string(accuracy_threshold * 100.0f) + "%");
        return 2;
    }

    MI_LOG_INFO("[SUCCESS] Dynamic multi-batch inference completed");
    return 0;
}
