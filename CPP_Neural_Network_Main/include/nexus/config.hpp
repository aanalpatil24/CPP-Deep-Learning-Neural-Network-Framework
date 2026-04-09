#pragma once

// COMPILER FEATURE DETECTION
// HFT constraint: Ensure we are using C++20 for advanced templating and memory models.
// We check both standard __cplusplus and Microsoft's _MSVC_LANG macro.

#if __cplusplus < 202002L && (!defined(_MSVC_LANG) || _MSVC_LANG < 202002L)
#error "Nexus requires C++20 or later"
#endif

// Force aggressive inlining on hot-path functions based on the compiler.
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
    #include <sys/mman.h> // Required for madvise (asynchronous page faulting)
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define NEXUS_ARCH_X86_64
    
    // SIMD Feature Detection: We heavily rely on 512-bit registers to 
    // process 16 single-precision floats per clock cycle.
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        #define NEXUS_HAS_AVX512
        #define NEXUS_SIMD_WIDTH 16
        #define NEXUS_SIMD_ALIGNMENT 64 // AVX-512 strictly requires 64-byte aligned memory
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
        #define NEXUS_SIMD_WIDTH_DYNAMIC 
    #else
        #define NEXUS_HAS_NEON
        #define NEXUS_SIMD_WIDTH 4
        #define NEXUS_SIMD_ALIGNMENT 16
    #endif
#else
    #error "Nexus requires x86_64 or ARM64 architecture"
#endif

// Cache geometry definitions for Loop Tiling / Cache Blocking.
// By fitting matrices perfectly into L1/L2, we prevent CPU pipeline stalls.
#ifndef NEXUS_CACHE_LINE_SIZE
    #define NEXUS_CACHE_LINE_SIZE 64
#endif

#ifndef NEXUS_L1_CACHE_SIZE
    #define NEXUS_L1_CACHE_SIZE 32768 // 32KB standard L1 data cache
#endif

#ifndef NEXUS_PAGE_SIZE
    #define NEXUS_PAGE_SIZE 4096
#endif

#ifdef _MSC_VER
    #include <intrin.h>     // For MSVC _ReadWriteBarrier
    #include <malloc.h>     // For MSVC _aligned_malloc
#endif


// STANDARD LIBRARY HEADERS

#include <immintrin.h>      // Core hardware SIMD intrinsics
#include <cstddef>          
#include <cstdint>          
#include <cstring>          
#include <cmath>            
#include <algorithm>        
#include <type_traits>      
#include <utility>          
#include <array>            
#include <span>             
#include <optional>         
#include <expected>         
#include <string>           
#include <string_view>      
#include <source_location>  
#include <chrono>           
#include <numeric>          
#include <random>           
#include <functional>       
#include <memory>           
#include <vector>           
#include <iostream>         
#include <iomanip>          
#include <fstream>          
#include <sstream>          
#include <atomic>           
#include <mutex>            
#include <new>
#include <cstdlib>
#include <variant>  


// C++23 std::expected POLYFILL (Safe Variant Implementation)
// We use std::expected for zero-overhead error handling (no exceptions).
// Since it's a C++23 feature, we polyfill it using std::variant for C++20 compatibility.

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

    // Specialization for void returns (used by our GEMM functions)
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

// SIMD type selection: We map our framework types directly to hardware registers.
#ifdef NEXUS_HAS_AVX512
    using simd_f32 = __m512;
#elif defined(NEXUS_HAS_AVX2)
    using simd_f32 = __m256;
#endif

}
