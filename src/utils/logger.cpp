#include "mini_infer/utils/logger.h"
#include <iostream>
#include <ctime>
#include <iomanip>

namespace mini_infer {
namespace utils {

Logger& Logger::get_instance() {
    static Logger instance;
    return instance;
}

void Logger::log(LogLevel level, const char* file, int line, const std::string& msg) {
    if (level < min_level_) {
        return;
    }
    
    // Get current time
    auto now = std::time(nullptr);
    auto* tm = std::localtime(&now);
    
    // Format output
    std::cout << "[" << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "] "
              << "[" << level_to_string(level) << "] "
              << "[" << file << ":" << line << "] "
              << msg << std::endl;
}

const char* Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
        case LogLevel::FATAL:   return "FATAL";
        default:                return "UNKNOWN";
    }
}

} // namespace utils
} // namespace mini_infer

