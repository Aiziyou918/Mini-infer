#pragma once

#include <cstdint>
#include <string>

namespace mini_infer {
namespace core {

/**
 * @brief Status codes
 * 
 */
enum class Status {
    SUCCESS = 0,
    ERROR_INVALID_ARGUMENT,
    ERROR_OUT_OF_MEMORY,
    ERROR_NOT_IMPLEMENTED,
    ERROR_RUNTIME,
    ERROR_BACKEND,
    ERROR_UNKNOWN
};

/**
 * @brief Convert a status code to a string
 * @param status The status code to convert
 * @return The string representation of the status code
 */
const char* status_to_string(Status status);

/**
 * @brief Device types
 */
enum class DeviceType {
    CPU,
    CUDA,
    UNKNOWN
};

/**
 * @brief Device information
 */
struct Device {
    DeviceType type{DeviceType::CPU};
    int32_t id{0};

    /**
     * @brief Convert the device to a string
     * @return The string representation of the device
     */
    std::string to_string() const;
};

} // namespace core
} // namespace mini_infer

