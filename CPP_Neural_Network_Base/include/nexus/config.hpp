#pragma once

// COMPILER FEATURE DETECTION

// Ensure strict C++20 compliance. 
// We check _MSVC_LANG specifically because Microsoft's compiler (MSVC) historically 
// under-reports its __cplusplus version for legacy compatibility reasons.
#if defined(_MSVC_LANG)
    #if _MSVC_LANG < 202002L
        #error "Nexus requires C++20 or later. Ensure /std:c++20 is set."
    #endif
#elif __cplusplus < 202002L
    #error "Nexus requires C++20 or later"
#endif

// Compiler-specific macros to force aggressive function inlining on the hot path
#if defined(__clang__)
    #define NEXUS_COMPILER_CLANG
    #define NEXUS_INLINE [[clang::always_inline]]
#elif defined(__GNUC__)
    #define NEXUS_COMPILER_GCC
    #define NEXUS_INLINE [[gnu::always_inline]]
#elif defined(_MSC_VER)
    #define NEXUS_COMPILER_MSVC
    #define NEXUS_INLINE __forceinline
#else
    #define NEXUS_INLINE inline
#endif

// PLATFORM DETECTION & ARCHITECTURE

#if defined(__linux__)
    #include <sys/mman.h> // For madvise (asynchronous memory pre-faulting)
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define NEXUS_ARCH_X86_64
    
    // SIMD Feature Detection
    // AVX-512 allows processing of 16 single-precision (32-bit) floats in one clock cycle.
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        #define NEXUS_HAS_AVX512
        #define NEXUS_SIMD_WIDTH 16
        #define NEXUS_SIMD_ALIGNMENT 64 // Strict requirement for AVX-512 memory loads
    #elif defined(__AVX2__)
        #define NEXUS_HAS_AVX2
        #define NEXUS_SIMD_WIDTH 8
        #define NEXUS_SIMD_ALIGNMENT 32
    #elif defined(__AVX__)
        #define NEXUS_HAS_AVX
        #define NEXUS_SIMD_WIDTH 8
        #define NEXUS_SIMD_ALIGNMENT 32
    #else
        #define NEXUS_SIMD_WIDTH 4
        #define NEXUS_SIMD_ALIGNMENT 16
    #endif
    
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define NEXUS_ARCH_ARM64
    #if defined(__ARM_FEATURE_SVE)
        #define NEXUS_HAS_SVE
        #define NEXUS_SIMD_WIDTH_DYNAMIC  // SVE has variable width
    #else
        #define NEXUS_HAS_NEON
        #define NEXUS_SIMD_WIDTH 4
        #define NEXUS_SIMD_ALIGNMENT 16
    #endif
#else
    #error "Nexus requires x86_64 or ARM64 architecture"
#endif

// Cache geometry (tunable via build flags)
// Used to size matrix tiles so they fit perfectly into CPU caches, avoiding RAM latency.
#ifndef NEXUS_CACHE_LINE_SIZE
    #define NEXUS_CACHE_LINE_SIZE 64
#endif

#ifndef NEXUS_L1_CACHE_SIZE
    #define NEXUS_L1_CACHE_SIZE 32768
#endif

#ifndef NEXUS_L2_CACHE_SIZE
    #define NEXUS_L2_CACHE_SIZE 1048576
#endif

#ifndef NEXUS_PAGE_SIZE
    #define NEXUS_PAGE_SIZE 4096
#endif

#ifdef _MSC_VER
    #include <intrin.h>
    #include <malloc.h>
#endif

// STANDARD LIBRARY HEADERS

#include <immintrin.h>      // SIMD intrinsics
#include <cstddef>          // std::size_t, std::byte, nullptr_t
#include <cstdint>          // Fixed-width integers
#include <cstring>          // std::memcpy, std::memset, std::memcmp
#include <cmath>            // std::exp, std::log, std::sqrt
#include <algorithm>        // std::min, std::max, std::swap
#include <type_traits>      // Type traits and SFINAE
#include <utility>          // std::move, std::forward, std::exchange
#include <array>            // std::array for fixed-size storage
#include <span>             // C++20 std::span
#include <optional>         // std::optional for error handling
#include <expected>         // C++23 std::expected (polyfill below if needed)
#include <string>           // std::string for error messages
#include <string_view>      // std::string_view for zero-copy strings
#include <source_location>  // C++20 source_location for debugging
#include <chrono>           // High-resolution timing
#include <numeric>          // std::accumulate, std::inner_product
#include <random>           // Weight initialization
#include <functional>       // std::function for callbacks
#include <memory>           // std::unique_ptr, std::make_unique
#include <vector>           // std::vector (used only for network topology)
#include <iostream>         // Debug output
#include <iomanip>          // Formatting
#include <fstream>          // Model serialization
#include <sstream>          // String streams
#include <atomic>           // std::atomic for thread-safe counters
#include <mutex>            // std::mutex for thread-safe arenas
#include <new>              // Placement new, std::align_val_t
#include <cstdlib>
#include <variant>  


// C++23 std::expected POLYFILL (Safe Variant Implementation)
// Implements zero-overhead error handling, avoiding the performance penalty 
// of C++ exceptions (try/catch blocks) during runtime execution.

#if __cplusplus < 202300L
namespace std {
    template<typename E>
    class unexpected {
        E error_;
    public:
        unexpected(E e) : error_(std::move(e)) {}
        const E& value() const { return error_; }
    };

    template<typename T, typename E>
    class expected {
        std::variant<T, E> data_;
        bool has_val_;
    public:
        expected(T val) : data_(std::move(val)), has_val_(true) {}
        expected(unexpected<E> err) : data_(std::move(err.value())), has_val_(false) {}
        
        bool has_value() const noexcept { return has_val_; }
        explicit operator bool() const noexcept { return has_val_; }
        
        T* operator->() { return &std::get<0>(data_); }
        T& operator*() { return std::get<0>(data_); }
        E& error() { return std::get<1>(data_); }
    };

    // Specialization for void returns (used by gemm)
    template<typename E>
    class expected<void, E> {
        std::optional<E> error_;
    public:
        expected() : error_(std::nullopt) {}
        expected(unexpected<E> err) : error_(std::move(err.value())) {}
        
        bool has_value() const noexcept { return !error_.has_value(); }
        explicit operator bool() const noexcept { return !error_.has_value(); }
        E& error() { return *error_; }
    };
}
#endif

// NAMESPACE NEXUS

namespace nexus {

// TYPE ALIASES & FORWARD DECLARATIONS

using f32 = float;
using f64 = double;
using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using usize = std::size_t;
using byte = std::byte;

// SIMD type selection
#ifdef NEXUS_HAS_AVX512
    using simd_f32 = __m512;
    using simd_f64 = __m512d;
    using simd_i32 = __m512i;
    #define NEXUS_SIMD_LOAD_F32 _mm512_loadu_ps
    #define NEXUS_SIMD_STORE_F32 _mm512_storeu_ps
    #define NEXUS_SIMD_SET1_F32 _mm512_set1_ps
    #define NEXUS_SIMD_ZERO_F32 _mm512_setzero_ps
    #define NEXUS_SIMD_FMA_F32 _mm512_fmadd_ps
    #define NEXUS_SIMD_ADD_F32 _mm512_add_ps
    #define NEXUS_SIMD_MUL_F32 _mm512_mul_ps
    #define NEXUS_SIMD_MAX_F32 _mm512_max_ps
    #define NEXUS_SIMD_MIN_F32 _mm512_min_ps
    #define NEXUS_SIMD_CMP_LT_F32 _mm512_cmp_ps_mask
    #define NEXUS_SIMD_MASK_MOV_F32 _mm512_mask_mov_ps
#elif defined(NEXUS_HAS_AVX2)
    using simd_f32 = __m256;
    using simd_f64 = __m256d;
    using simd_i32 = __m256i;
    #define NEXUS_SIMD_LOAD_F32 _mm256_loadu_ps
    #define NEXUS_SIMD_STORE_F32 _mm256_storeu_ps
    #define NEXUS_SIMD_SET1_F32 _mm256_set1_ps
    #define NEXUS_SIMD_ZERO_F32 _mm256_setzero_ps
    #define NEXUS_SIMD_FMA_F32 _mm256_fmadd_ps
    #define NEXUS_SIMD_ADD_F32 _mm256_add_ps
    #define NEXUS_SIMD_MUL_F32 _mm256_mul_ps
    #define NEXUS_SIMD_MAX_F32 _mm256_max_ps
    #define NEXUS_SIMD_MIN_F32 _mm256_min_ps
#endif

// Forward declarations
class MemoryArena;
class TensorView;
class Layer;
class NeuralNetwork;

}
