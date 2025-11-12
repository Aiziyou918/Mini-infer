#include "mini_infer/core/types.h"

namespace mini_infer {
namespace core {

const char* status_to_string(Status status) {
    switch (status) {
        case Status::SUCCESS:
            return "Success";
        case Status::ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case Status::ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case Status::ERROR_NOT_IMPLEMENTED:
            return "Not implemented";
        case Status::ERROR_RUNTIME:
            return "Runtime error";
        case Status::ERROR_BACKEND:
            return "Backend error";
        case Status::ERROR_UNKNOWN:
        default:
            return "Unknown error";
    }
}

std::string Device::to_string() const {
    std::string result;
    switch (type) {
        case DeviceType::CPU:
            result = "CPU";
            break;
        case DeviceType::CUDA:
            result = "CUDA:" + std::to_string(id);
            break;
        default:
            result = "Unknown";
            break;
    }
    return result;
}

} // namespace core
} // namespace mini_infer

