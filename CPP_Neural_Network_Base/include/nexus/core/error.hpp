#pragma once
#include "../config.hpp"
#include <string>
#include <source_location>

namespace nexus {

// ERROR HANDLING SYSTEM

enum class ErrorCode : i32 {
    Success = 0,
    NullPointer,
    InvalidDimension,
    DimensionMismatch,
    AlignmentViolation,
    OutOfMemory,
    ArenaOverflow,
    UnsupportedOperation,
    HardwareFeatureMissing,
    SerializationError,
    InvalidFormat,
    IndexOutOfBounds,
    DivisionByZero,
    NumericalOverflow
};

class NexusError {
public:
    ErrorCode code;
    std::string message;
    std::source_location location;
    
    NexusError(ErrorCode c, std::string msg, 
               std::source_location loc = std::source_location::current())
        : code(c), message(std::move(msg)), location(loc) {}
    
    std::string to_string() const {
        return "[Error " + std::to_string(static_cast<i32>(code)) + "] " + 
               message + " at " + location.file_name() + ":" + 
               std::to_string(location.line());
    }
};

template<typename T>
using Result = std::expected<T, NexusError>;

#define NEXUS_RETURN_IF_ERROR(expr) \
    do { \
        auto _res = (expr); \
        if (!_res) return std::unexpected(_res.error()); \
    } while(0)

}
