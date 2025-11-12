#pragma once

#include <string>
#include <sstream>

namespace mini_infer {
namespace utils {

/**
 * @brief Log level
 */
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};

/**
 * @brief Simple logger
 */
class Logger {
public:
    /**
     * @brief Get logger instance
     * 
     * @return Logger& 
     */
    static Logger& get_instance();
    
    /**
     * @brief Log message
     * 
     * @param level log level
     * @param file file name
     * @param line line number
     * @param msg message
     */
    void log(LogLevel level, const char* file, int line, const std::string& msg);
    
    /**
     * @brief Set log level
     * 
     * @param level log level
     */
    void set_level(LogLevel level) { min_level_ = level; }

    /**
     * @brief Get log level
     * 
     * @return LogLevel 
     */
    LogLevel get_level() const { return min_level_; }
    
private:
    Logger() = default;

    /**
     * @brief Minimum log level
     * @param level log level
     */
    LogLevel min_level_{LogLevel::INFO};
    
    /**
     * @brief Convert log level to string
     * 
     * @param level log level
     * @return const char* 
     */
    const char* level_to_string(LogLevel level);
};

} // namespace utils
} // namespace mini_infer

// Log macro definition
#ifdef MINI_INFER_ENABLE_LOGGING
#define MI_LOG_DEBUG(msg) \
    mini_infer::utils::Logger::get_instance().log( \
        mini_infer::utils::LogLevel::DEBUG, __FILE__, __LINE__, msg)
        
#define MI_LOG_INFO(msg) \
    mini_infer::utils::Logger::get_instance().log( \
        mini_infer::utils::LogLevel::INFO, __FILE__, __LINE__, msg)
        
#define MI_LOG_WARNING(msg) \
    mini_infer::utils::Logger::get_instance().log( \
        mini_infer::utils::LogLevel::WARNING, __FILE__, __LINE__, msg)
        
#define MI_LOG_ERROR(msg) \
    mini_infer::utils::Logger::get_instance().log( \
        mini_infer::utils::LogLevel::ERROR, __FILE__, __LINE__, msg)
        
#define MI_LOG_FATAL(msg) \
    mini_infer::utils::Logger::get_instance().log( \
        mini_infer::utils::LogLevel::FATAL, __FILE__, __LINE__, msg)
#else
#define MI_LOG_DEBUG(msg)
#define MI_LOG_INFO(msg)
#define MI_LOG_WARNING(msg)
#define MI_LOG_ERROR(msg)
#define MI_LOG_FATAL(msg)
#endif

