/**
 * @file simple_loader.h
 * @brief Simple utilities for loading binary weights and MNIST samples
 * 
 * This file provides utilities to load:
 * 1. Binary weight files exported from Python (float32, raw format)
 * 2. MNIST test samples exported from Python
 */

#pragma once

#include "mini_infer/core/tensor.h"
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <cstring>

namespace mini_infer {
namespace utils {

/**
 * @brief Load a binary weight file (raw float32 format)
 * 
 * @param filepath Path to .bin file
 * @param shape Expected shape of the tensor
 * @return Tensor loaded from file
 */
inline std::shared_ptr<core::Tensor> load_binary_weight(
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
    size_t expected_bytes = numel * sizeof(float);
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size != expected_bytes) {
        throw std::runtime_error(
            "File size mismatch for " + filepath + ": " +
            "expected " + std::to_string(expected_bytes) + " bytes, " +
            "got " + std::to_string(file_size) + " bytes"
        );
    }
    
    // Create tensor
    auto tensor = core::Tensor::create(
        core::Shape(shape),
        core::DataType::FLOAT32
    );
    
    // Read data directly into tensor
    file.read(reinterpret_cast<char*>(tensor->data()), expected_bytes);
    
    if (!file) {
        throw std::runtime_error("Failed to read data from: " + filepath);
    }
    
    return tensor;
}

/**
 * @brief Load all LeNet-5 weights from a directory
 * 
 * Directory structure:
 *   weights_dir/
 *     conv1_weight.bin
 *     conv1_bias.bin
 *     conv2_weight.bin
 *     conv2_bias.bin
 *     fc1_weight.bin
 *     fc1_bias.bin
 *     fc2_weight.bin
 *     fc2_bias.bin
 *     fc3_weight.bin
 *     fc3_bias.bin
 */
struct LeNet5Weights {
    std::shared_ptr<core::Tensor> conv1_weight;  // [6, 1, 5, 5]
    std::shared_ptr<core::Tensor> conv1_bias;    // [6]
    std::shared_ptr<core::Tensor> conv2_weight;  // [16, 6, 5, 5]
    std::shared_ptr<core::Tensor> conv2_bias;    // [16]
    std::shared_ptr<core::Tensor> fc1_weight;    // [120, 256]
    std::shared_ptr<core::Tensor> fc1_bias;      // [120]
    std::shared_ptr<core::Tensor> fc2_weight;    // [84, 120]
    std::shared_ptr<core::Tensor> fc2_bias;      // [84]
    std::shared_ptr<core::Tensor> fc3_weight;    // [10, 84]
    std::shared_ptr<core::Tensor> fc3_bias;      // [10]
    
    /**
     * @brief Load all weights from directory
     */
    static LeNet5Weights load(const std::string& weights_dir) {
        LeNet5Weights weights;
        
        std::cout << "Loading LeNet-5 weights from: " << weights_dir << std::endl;
        
        try {
            // Conv1
            weights.conv1_weight = load_binary_weight(
                weights_dir + "/conv1_weight.bin", {6, 1, 5, 5});
            weights.conv1_bias = load_binary_weight(
                weights_dir + "/conv1_bias.bin", {6});
            std::cout << "  [SUCCESS] Conv1 loaded" << std::endl;
            
            // Conv2
            weights.conv2_weight = load_binary_weight(
                weights_dir + "/conv2_weight.bin", {16, 6, 5, 5});
            weights.conv2_bias = load_binary_weight(
                weights_dir + "/conv2_bias.bin", {16});
            std::cout << "  [SUCCESS] Conv2 loaded" << std::endl;
            
            // FC1
            weights.fc1_weight = load_binary_weight(
                weights_dir + "/fc1_weight.bin", {120, 256});
            weights.fc1_bias = load_binary_weight(
                weights_dir + "/fc1_bias.bin", {120});
            std::cout << "  [SUCCESS] FC1 loaded" << std::endl;
            
            // FC2
            weights.fc2_weight = load_binary_weight(
                weights_dir + "/fc2_weight.bin", {84, 120});
            weights.fc2_bias = load_binary_weight(
                weights_dir + "/fc2_bias.bin", {84});
            std::cout << "  [SUCCESS] FC2 loaded" << std::endl;
            
            // FC3
            weights.fc3_weight = load_binary_weight(
                weights_dir + "/fc3_weight.bin", {10, 84});
            weights.fc3_bias = load_binary_weight(
                weights_dir + "/fc3_bias.bin", {10});
            std::cout << "  [SUCCESS] FC3 loaded" << std::endl;
            
            std::cout << "All weights loaded successfully!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading weights: " << e.what() << std::endl;
            throw;
        }
        
        return weights;
    }
    
    /**
     * @brief Print weight statistics
     */
    void print_stats() const {
        auto print_tensor_stats = [](const std::string& name, 
                                      const std::shared_ptr<core::Tensor>& tensor) {
            const float* data = static_cast<const float*>(tensor->data());
            int64_t numel = tensor->shape().numel();
            
            float min_val = data[0], max_val = data[0];
            double sum = 0.0;
            
            for (int64_t i = 0; i < numel; ++i) {
                float val = data[i];
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
                sum += val;
            }
            
            double mean = sum / numel;
            
            std::cout << "  " << name << ": shape=";
            std::cout << "[";
            for (size_t i = 0; i < tensor->shape().ndim(); ++i) {
                std::cout << tensor->shape()[i];
                if (i < tensor->shape().ndim() - 1) std::cout << ", ";
            }
            std::cout << "], min=" << min_val 
                      << ", max=" << max_val 
                      << ", mean=" << mean << std::endl;
        };
        
        std::cout << "\nWeight Statistics:" << std::endl;
        print_tensor_stats("conv1_weight", conv1_weight);
        print_tensor_stats("conv1_bias", conv1_bias);
        print_tensor_stats("conv2_weight", conv2_weight);
        print_tensor_stats("conv2_bias", conv2_bias);
        print_tensor_stats("fc1_weight", fc1_weight);
        print_tensor_stats("fc1_bias", fc1_bias);
        print_tensor_stats("fc2_weight", fc2_weight);
        print_tensor_stats("fc2_bias", fc2_bias);
        print_tensor_stats("fc3_weight", fc3_weight);
        print_tensor_stats("fc3_bias", fc3_bias);
    }
};

/**
 * @brief MNIST sample information
 */
struct MNISTSample {
    int index;
    int label;
    std::string binary_path;
    std::string png_path;
};

/**
 * @brief Load a single MNIST sample
 * 
 * @param filepath Path to binary file (normalized float32, shape [1, 28, 28])
 * @return Tensor with shape [1, 1, 28, 28]
 */
inline std::shared_ptr<core::Tensor> load_mnist_sample(const std::string& filepath) {
    // MNIST samples are stored as [1, 28, 28], but we need [1, 1, 28, 28] for batch processing
    auto temp = load_binary_weight(filepath, {1, 28, 28});
    
    // Create tensor with correct shape [1, 1, 28, 28]
    auto tensor = core::Tensor::create(
        core::Shape({1, 1, 28, 28}),
        core::DataType::FLOAT32
    );
    
    // Copy data
    std::memcpy(tensor->data(), temp->data(), 1 * 28 * 28 * sizeof(float));
    
    return tensor;
}

/**
 * @brief Get argmax of a tensor (for classification)
 */
inline int argmax(const std::shared_ptr<core::Tensor>& tensor) {
    const float* data = static_cast<const float*>(tensor->data());
    int64_t numel = tensor->shape().numel();
    
    int max_idx = 0;
    float max_val = data[0];
    
    for (int64_t i = 1; i < numel; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/**
 * @brief Print tensor shape and first few values
 */
inline void print_tensor_debug(const std::string& name, 
                               const std::shared_ptr<core::Tensor>& tensor,
                               int max_values = 5) {
    std::cout << name << " shape: [";
    for (size_t i = 0; i < tensor->shape().ndim(); ++i) {
        std::cout << tensor->shape()[i];
        if (i < tensor->shape().ndim() - 1) std::cout << ", ";
    }
    std::cout << "]";
    
    const float* data = static_cast<const float*>(tensor->data());
    int64_t numel = tensor->shape().numel();
    
    std::cout << ", values: [";
    for (int i = 0; i < max_values && i < numel; ++i) {
        std::cout << data[i];
        if (i < max_values - 1 && i < numel - 1) std::cout << ", ";
    }
    if (numel > max_values) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

} // namespace utils
} // namespace mini_infer
