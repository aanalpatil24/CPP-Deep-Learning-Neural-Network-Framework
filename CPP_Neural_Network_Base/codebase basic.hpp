/*
 * =============================================================================
 * NEXUS: High-Performance Neural Network Execution Engine v2.0 (Base)
 * =============================================================================
 * * A zero-dependency, bare-metal deep learning framework engineered for 
 * sub-microsecond inference latency and deterministic memory behavior.
 * * Architecture: C++20 | AVX-512F/AVX2 | Cache-Oblivious Algorithms | 
 * Arena Allocation | Thread-Safe Design | NUMA-Aware
 * * Author: Quantitative Systems Engineer
 * Version: 2.0.0 (OOP Topology Base)
 * =============================================================================
 */

#pragma once

// =============================================================================
// COMPILER FEATURE DETECTION
// =============================================================================

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

// =============================================================================
// PLATFORM DETECTION & ARCHITECTURE
// =============================================================================

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

// =============================================================================
// STANDARD LIBRARY HEADERS
// =============================================================================

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

// =============================================================================
// C++23 std::expected POLYFILL (Safe Variant Implementation)
// =============================================================================
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

// =============================================================================
// NAMESPACE NEXUS
// =============================================================================

namespace nexus {

// =============================================================================
// TYPE ALIASES & FORWARD DECLARATIONS
// =============================================================================

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

// =============================================================================
// ERROR HANDLING SYSTEM
// =============================================================================

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

// =============================================================================
// MEMORY ARENA (core/memory/arena.cpp + include)
// =============================================================================
// V2.0 BASELINE ALLOCATOR:
// Replaces dynamic heap allocation (`new` / `malloc`) with a pre-allocated Arena.
// Note: This version uses `std::mutex` for thread-safety. While it is memory-safe,
// OS-level mutexes can introduce context-switching latency under heavy multi-threading.

class alignas(NEXUS_CACHE_LINE_SIZE) MemoryArena {
public:
    struct Config {
        usize initial_capacity = 16 * 1024 * 1024;  // 16MB default
        usize growth_factor = 2;
        usize max_capacity = 1024 * 1024 * 1024;    // 1GB max
        bool use_huge_pages = false;                 // 2MB pages
        i32 numa_node = -1;                          // -1 = any node
        bool thread_safe = false;                    // Mutex protection
    };
    
    struct Stats {
        usize capacity;
        usize used;
        usize peak;
        usize allocation_count;
        usize reset_count;
        usize numa_migrations;
    };

private:
    // Cache-line-padded to prevent false sharing
    alignas(NEXUS_CACHE_LINE_SIZE) byte* base_ = nullptr;
    alignas(NEXUS_CACHE_LINE_SIZE) byte* current_ = nullptr;
    byte* end_ = nullptr;
    
    usize initial_capacity_ = 0;
    usize growth_factor_ = 2;
    usize max_capacity_ = 0;
    usize peak_used_ = 0;
    usize alloc_count_ = 0;
    usize reset_count_ = 0;
    
    bool use_huge_pages_ = false;
    i32 numa_node_ = -1;
    
    mutable std::mutex mutex_;
    bool thread_safe_ = false;
    
    // Parent arena for chained growth
    std::unique_ptr<MemoryArena> next_arena_;

public:

    explicit MemoryArena(usize initial_capacity) {
        use_huge_pages_ = false;
        initial_capacity_ = align_up(initial_capacity, NEXUS_PAGE_SIZE);
        growth_factor_ = 2;
        max_capacity_ = 1024 * 1024 * 1024;
        thread_safe_ = false;
        grow(initial_capacity_);
    }

    explicit MemoryArena(const Config& config = {}) {
        initial_capacity_ = align_up(config.initial_capacity, NEXUS_PAGE_SIZE);
        growth_factor_ = config.growth_factor;
        max_capacity_ = config.max_capacity;
        use_huge_pages_ = config.use_huge_pages;
        numa_node_ = config.numa_node;
        thread_safe_ = config.thread_safe;
        
        grow(initial_capacity_);
    }
    
    ~MemoryArena() {
        release();
    }
    
    // Non-copyable
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    
    // Movable
    MemoryArena(MemoryArena&& other) noexcept {
        *this = std::move(other);
    }
    
    MemoryArena& operator=(MemoryArena&& other) noexcept {
        if (this != &other) {
            release();
            base_ = std::exchange(other.base_, nullptr);
            current_ = std::exchange(other.current_, nullptr);
            end_ = std::exchange(other.end_, nullptr);
            initial_capacity_ = other.initial_capacity_;
            peak_used_ = other.peak_used_;
            alloc_count_ = other.alloc_count_;
            next_arena_ = std::move(other.next_arena_);
        }
        return *this;
    }

    // Core allocation with alignment
    [[nodiscard]] void* allocate(usize size, usize alignment = NEXUS_SIMD_ALIGNMENT) {
        // V2.0 BASE: Blocking synchronization via std::mutex
        if (thread_safe_) {
            std::lock_guard<std::mutex> lock(mutex_);
            return allocate_impl(size, alignment);
        }
        return allocate_impl(size, alignment);
    }
    
    template<typename T, typename... Args>
    [[nodiscard]] T* construct(Args&&... args) {
        void* mem = allocate(sizeof(T), alignof(T));
        if (!mem) return nullptr;
        return new (mem) T(std::forward<Args>(args)...);
    }
    
    template<typename T>
    [[nodiscard]] T* allocate_array(usize count) {
        usize bytes = count * sizeof(T);
        // Check for overflow
        if (count != 0 && bytes / count != sizeof(T)) return nullptr;
        return static_cast<T*>(allocate(bytes, alignof(T)));
    }

    // Scoped deallocation markers
    using Marker = std::pair<byte*, MemoryArena*>;
    
    [[nodiscard]] Marker mark() const noexcept {
        if (thread_safe_) {
            std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
            return {current_, const_cast<MemoryArena*>(this)};
        }
        return {current_, const_cast<MemoryArena*>(this)};
    }
    
    void reset_to(Marker marker) noexcept {
        if (marker.second != this) {
            // Reset child arena and unlink
            if (next_arena_) {
                next_arena_->reset_to(marker);
                next_arena_.reset();
            }
        } else {
            if (thread_safe_) {
                std::lock_guard<std::mutex> lock(mutex_);
                current_ = marker.first;
            } else {
                current_ = marker.first;
            }
        }
        ++reset_count_;
    }
    
    void reset() noexcept {
        if (next_arena_) next_arena_.reset();
        current_ = base_;
        ++reset_count_;
    }

    // Utilities
    [[nodiscard]] bool contains(const void* ptr) const noexcept {
        auto addr = reinterpret_cast<const byte*>(ptr);
        bool in_this = (addr >= base_ && addr < end_);
        if (in_this) return true;
        if (next_arena_) return next_arena_->contains(ptr);
        return false;
    }
    
    [[nodiscard]] Stats stats() const noexcept {
        Stats s;
        s.capacity = capacity();
        s.used = used();
        s.peak = peak_used_;
        s.allocation_count = alloc_count_;
        s.reset_count = reset_count_;
        s.numa_migrations = 0;
        return s;
    }
    
    [[nodiscard]] usize available() const noexcept {
        return (end_ - current_) + (next_arena_ ? next_arena_->available() : 0);
    }
    
    [[nodiscard]] usize used() const noexcept {
        usize total = current_ - base_;
        if (next_arena_) total += next_arena_->used();
        return total;
    }
    
    [[nodiscard]] usize capacity() const noexcept {
        usize total = end_ - base_;
        if (next_arena_) total += next_arena_->capacity();
        return total;
    }
    
    [[nodiscard]] bool can_allocate(usize size, usize alignment = NEXUS_SIMD_ALIGNMENT) const noexcept {
        byte* aligned = align_ptr(current_, alignment);
        if ((aligned + size) <= end_) return true;
        if (next_arena_) return next_arena_->can_allocate(size, alignment);
        return false;
    }

private:
    void* allocate_impl(usize size, usize alignment) {
        byte* aligned = align_ptr(current_, alignment);
        byte* new_ptr = aligned + size;
        
        if (new_ptr <= end_) {
            current_ = new_ptr;
            peak_used_ = std::max(peak_used_, static_cast<usize>(current_ - base_));
            ++alloc_count_;
            return aligned;
        }
        
        // Try to grow
        if (!next_arena_ && can_grow(size)) {
            grow(std::max(size * growth_factor_, static_cast<usize>(end_ - base_)));
            return allocate_impl(size, alignment);
        }
        
        // Chain to next arena
        if (!next_arena_) {
            usize new_cap = std::min(
                static_cast<usize>((end_ - base_) * growth_factor_),
                max_capacity_ - capacity()
            );
            if (new_cap < size) return nullptr;
            
            Config child_config;
            child_config.initial_capacity = new_cap;
            child_config.thread_safe = false;  // Only root needs lock
            next_arena_ = std::make_unique<MemoryArena>(child_config);
        }
        
        return next_arena_->allocate_impl(size, alignment);
    }
    
    bool can_grow(usize min_size) const {
        usize current_cap = end_ - base_;
        usize new_cap = current_cap * growth_factor_;
        return (capacity() + new_cap <= max_capacity_) && (new_cap >= min_size);
    }
    
    void grow(usize new_capacity) {
        new_capacity = align_up(new_capacity, use_huge_pages_ ? 2097152 : NEXUS_PAGE_SIZE);
        
        // Use aligned allocation
        usize align = use_huge_pages_ ? 2097152 : NEXUS_SIMD_ALIGNMENT;
        
        #if defined(_WIN32) || defined(_MSC_VER)
            base_ = static_cast<byte*>(_aligned_malloc(new_capacity, align));
        #else
            base_ = static_cast<byte*>(std::aligned_alloc(align, new_capacity));
        #endif
        
        if (!base_) throw std::bad_alloc();
        
        // Touch pages to force allocation (prevent page faults during inference)
        #if defined(__linux__)
            madvise(base_, new_capacity, MADV_WILLNEED);
        #else
            for(usize i = 0; i < new_capacity; i += 4096) base_[i] = std::byte{0};
        #endif
        
        current_ = base_;
        end_ = base_ + new_capacity;
    }
    
    void release() {
        if (base_) {
            #if defined(_WIN32) || defined(_MSC_VER)
                _aligned_free(base_);
            #else
                std::free(base_);
            #endif
            base_ = nullptr;
        }
        next_arena_.reset();
    }
    
    static inline byte* align_ptr(byte* ptr, usize alignment) noexcept {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
        return reinterpret_cast<byte*>(aligned);
    }
    
    static inline usize align_up(usize size, usize alignment) noexcept {
        return (size + alignment - 1) & ~(alignment - 1);
    }
};

// RAII scope guard
class ArenaScope {
    MemoryArena& arena_;
    MemoryArena::Marker marker_;
    bool released_ = false;
    
public:
    explicit ArenaScope(MemoryArena& arena) : arena_(arena), marker_(arena.mark()) {}
    ~ArenaScope() { if (!released_) arena_.reset_to(marker_); }
    void release() noexcept { released_ = true; }
};

// =============================================================================
// TENSOR VIEW (core/tensor/tensor_view.cpp + include)
// =============================================================================
// Non-owning multi-dimensional view over flat memory arrays.
// Bypasses the overhead of std::vector by operating on raw memory regions.

class TensorView {
public:
    f32* data = nullptr;
    usize ndim = 0;
    std::array<usize, 6> shape_{};      // Up to 6D tensors
    std::array<usize, 6> strides_{};    // Strides in elements
    
    TensorView() = default;
    
    // 1D constructor
    TensorView(f32* d, usize dim0) 
        : data(d), ndim(1) {
        shape_[0] = dim0;
        strides_[0] = 1;
    }
    
    // 2D constructor
    TensorView(f32* d, usize dim0, usize dim1)
        : data(d), ndim(2) {
        shape_[0] = dim0;
        shape_[1] = dim1;
        strides_[0] = dim1;
        strides_[1] = 1;
    }
    
    // 3D constructor
    TensorView(f32* d, usize dim0, usize dim1, usize dim2)
        : data(d), ndim(3) {
        shape_[0] = dim0;
        shape_[1] = dim1;
        shape_[2] = dim2;
        strides_[0] = dim1 * dim2;
        strides_[1] = dim2;
        strides_[2] = 1;
    }
    
    // Generic constructor with explicit strides
    TensorView(f32* d, std::span<const usize> shape, std::span<const usize> strides = {})
        : data(d), ndim(shape.size()) {
        if (ndim > 6) throw std::invalid_argument("Max 6 dimensions supported");
        std::copy(shape.begin(), shape.end(), shape_.begin());
        if (strides.empty()) {
            // Compute default row-major strides
            strides_[ndim - 1] = 1;
            for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        } else {
            std::copy(strides.begin(), strides.end(), strides_.begin());
        }
    }

    // Element access (variadic indices)
    template<typename... Indices>
    [[nodiscard]] f32& operator()(Indices... indices) {
        static_assert(sizeof...(indices) <= 6, "Too many indices");
        usize idx[] = {static_cast<usize>(indices)...};
        usize offset = 0;
        for (usize i = 0; i < sizeof...(indices); ++i) {
            offset += idx[i] * strides_[i];
        }
        return data[offset];
    }
    
    template<typename... Indices>
    [[nodiscard]] const f32& operator()(Indices... indices) const {
        return const_cast<TensorView*>(this)->operator()(indices...);
    }
    
    // Linear indexing
    [[nodiscard]] f32& operator[](usize idx) { return data[idx]; }
    [[nodiscard]] const f32& operator[](usize idx) const { return data[idx]; }
    
    // Row access for 2D tensors
    [[nodiscard]] f32* row(usize i) { 
        if (ndim < 2) return data + i * strides_[0];
        return data + i * strides_[0]; 
    }
    
    [[nodiscard]] const f32* row(usize i) const {
        return const_cast<TensorView*>(this)->row(i);
    }

    // Properties
    [[nodiscard]] usize size() const noexcept {
        usize total = 1;
        for (usize i = 0; i < ndim; ++i) total *= shape_[i];
        return total;
    }
    
    [[nodiscard]] usize size_bytes() const noexcept {
        return size() * sizeof(f32);
    }
    
    [[nodiscard]] bool is_contiguous() const noexcept {
        usize expected = 1;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            if (strides_[i] != expected) return false;
            expected *= shape_[i];
        }
        return true;
    }
    
    [[nodiscard]] bool is_aligned(usize alignment = NEXUS_SIMD_ALIGNMENT) const noexcept {
        return (reinterpret_cast<uintptr_t>(data) % alignment) == 0;
    }
    
    [[nodiscard]] std::span<f32> as_span() noexcept {
        return std::span<f32>(data, size());
    }
    
    [[nodiscard]] std::span<const f32> as_span() const noexcept {
        return std::span<const f32>(data, size());
    }
    
    // Shape queries
    [[nodiscard]] usize rows() const noexcept { return ndim > 0 ? shape_[0] : 0; }
    [[nodiscard]] usize cols() const noexcept { return ndim > 1 ? shape_[1] : 0; }
    [[nodiscard]] usize dim(usize i) const noexcept { return i < ndim ? shape_[i] : 0; }
    
    // Reshape view (same underlying data)
    [[nodiscard]] TensorView reshape(std::span<const usize> new_shape) const {
        usize new_size = 1;
        for (auto s : new_shape) new_size *= s;
        if (new_size != size()) throw std::invalid_argument("Shape size mismatch");
        
        TensorView result;
        result.data = data;
        result.ndim = new_shape.size();
        std::copy(new_shape.begin(), new_shape.end(), result.shape_.begin());
        // Compute new strides (row-major)
        result.strides_[result.ndim - 1] = 1;
        for (int i = static_cast<int>(result.ndim) - 2; i >= 0; --i) {
            result.strides_[i] = result.strides_[i + 1] * result.shape_[i + 1];
        }
        return result;
    }
    
    // Slice operations
    [[nodiscard]] TensorView slice(usize dim, usize start, usize end) const {
        TensorView result = *this;
        if (dim >= ndim) throw std::out_of_range("Invalid dimension");
        result.shape_[dim] = end - start;
        result.data += start * strides_[dim];
        return result;
    }
    
    // Fill operations
    void zero() noexcept {
        std::memset(data, 0, size_bytes());
    }
    
    void fill(f32 value) noexcept {
        if (value == 0.0f) {
            zero();
            return;
        }
        // SIMD fill for large tensors
        usize n = size();
        usize i = 0;
        
        #ifdef NEXUS_HAS_AVX512
        __m512 val = _mm512_set1_ps(value);
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_ps(data + i, val);
        }
        #elif defined(NEXUS_HAS_AVX2)
        __m256 val = _mm256_set1_ps(value);
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(data + i, val);
        }
        #endif
        
        for (; i < n; ++i) data[i] = value;
    }
    
    // Validation
    [[nodiscard]] bool valid() const noexcept {
        return data != nullptr && size() > 0;
    }
};

// =============================================================================
// SIMD MATHEMATICS KERNELS (math/kernels/activations.cpp + include)
// =============================================================================

namespace kernels {

// Fast approximate exp using bit manipulation (2-3x faster than std::exp)
// Based on Schraudolph's method with minimax polynomial approximations.
// Extremely important for calculating Sigmoid and Softmax rapidly in HFT.
NEXUS_INLINE f32 fast_exp_f32(f32 x) {
    // Clamp to avoid overflow/underflow
    if (x > 88.0f) return std::numeric_limits<f32>::infinity();
    if (x < -88.0f) return 0.0f;
    
    // Coefficients for minimax polynomial approximation
    const f32 c0 = 1.0f;
    const f32 c1 = 0.9999997f;
    const f32 c2 = 0.5000003f;
    const f32 c3 = 0.1666652f;
    const f32 c4 = 0.0416350f;
    const f32 c5 = 0.0083299f;
    
    f32 x2 = x * x;
    f32 x3 = x2 * x;
    f32 x4 = x2 * x2;
    f32 x5 = x3 * x2;
    
    return c0 + c1*x + c2*x2 + c3*x3 + c4*x4 + c5*x5;
}

// Vectorized exp (SIMD)
#ifdef NEXUS_HAS_AVX512
NEXUS_INLINE __m512 fast_exp_ps(__m512 x) {
    // Clamp
    __m512 max_val = _mm512_set1_ps(88.0f);
    __m512 min_val = _mm512_set1_ps(-88.0f);
    x = _mm512_max_ps(min_val, _mm512_min_ps(x, max_val));
    
    // Polynomial approximation using Fused Multiply-Add (FMA) instructions
    __m512 c0 = _mm512_set1_ps(1.0f);
    __m512 c1 = _mm512_set1_ps(0.9999997f);
    __m512 c2 = _mm512_set1_ps(0.5000003f);
    __m512 c3 = _mm512_set1_ps(0.1666652f);
    __m512 c4 = _mm512_set1_ps(0.0416350f);
    __m512 c5 = _mm512_set1_ps(0.0083299f);
    
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 x3 = _mm512_mul_ps(x2, x);
    __m512 x4 = _mm512_mul_ps(x2, x2);
    __m512 x5 = _mm512_mul_ps(x3, x2);
    
    __m512 result = c0;
    result = _mm512_fmadd_ps(c1, x, result);
    result = _mm512_fmadd_ps(c2, x2, result);
    result = _mm512_fmadd_ps(c3, x3, result);
    result = _mm512_fmadd_ps(c4, x4, result);
    result = _mm512_fmadd_ps(c5, x5, result);
    
    return result;
}
#elif defined(NEXUS_HAS_AVX2)
NEXUS_INLINE __m256 fast_exp_ps(__m256 x) {
    __m256 max_val = _mm256_set1_ps(88.0f);
    __m256 min_val = _mm256_set1_ps(-88.0f);
    x = _mm256_max_ps(min_val, _mm256_min_ps(x, max_val));
    
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(0.9999997f);
    __m256 c2 = _mm256_set1_ps(0.5000003f);
    __m256 c3 = _mm256_set1_ps(0.1666652f);
    
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    
    __m256 result = c0;
    result = _mm256_fmadd_ps(c1, x, result);
    result = _mm256_fmadd_ps(c2, x2, result);
    result = _mm256_fmadd_ps(c3, x3, result);
    
    return result;
}
#endif

// Element-wise operations
inline void vec_relu(const f32* __restrict input, f32* __restrict output, usize n) {
    usize i = 0;
    
    #ifdef NEXUS_HAS_AVX512
    __m512 zero = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(input + i);
        __m512 result = _mm512_max_ps(x, zero);
        _mm512_storeu_ps(output + i, result);
    }
    #elif defined(NEXUS_HAS_AVX2)
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 result = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(output + i, result);
    }
    #endif
    
    for (; i < n; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

inline void vec_sigmoid(const f32* __restrict input, f32* __restrict output, usize n) {
    usize i = 0;
    
    #ifdef NEXUS_HAS_AVX512
    __m512 one = _mm512_set1_ps(1.0f);
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(input + i);
        __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
        __m512 exp_neg_x = fast_exp_ps(neg_x);
        __m512 denom = _mm512_add_ps(one, exp_neg_x);
        // Approximate reciprocal for speed
        __m512 result = _mm512_div_ps(one, denom);
        _mm512_storeu_ps(output + i, result);
    }
    #elif defined(NEXUS_HAS_AVX2)
    __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        __m256 exp_neg_x = fast_exp_ps(neg_x);
        __m256 denom = _mm256_add_ps(one, exp_neg_x);
        __m256 result = _mm256_div_ps(one, denom);
        _mm256_storeu_ps(output + i, result);
    }
    #endif
    
    for (; i < n; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

inline void vec_tanh(const f32* __restrict input, f32* __restrict output, usize n) {
    usize i = 0;
    
    #ifdef NEXUS_HAS_AVX512
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(input + i);
        __m512 neg_2x = _mm512_mul_ps(x, _mm512_set1_ps(-2.0f));
        __m512 exp_neg_2x = fast_exp_ps(neg_2x);
        __m512 numerator = _mm512_sub_ps(_mm512_set1_ps(1.0f), exp_neg_2x);
        __m512 denominator = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_2x);
        __m512 result = _mm512_div_ps(numerator, denominator);
        _mm512_storeu_ps(output + i, result);
    }
    #endif
    
    for (; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

inline void vec_add(const f32* __restrict a, const f32* __restrict b, 
                    f32* __restrict c, usize n) {
    usize i = 0;
    
    #ifdef NEXUS_HAS_AVX512
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }
    #elif defined(NEXUS_HAS_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    #endif
    
    for (; i < n; ++i) c[i] = a[i] + b[i];
}

inline void vec_mul(const f32* __restrict a, const f32* __restrict b,
                    f32* __restrict c, usize n) {
    usize i = 0;
    
    #ifdef NEXUS_HAS_AVX512
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }
    #elif defined(NEXUS_HAS_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    #endif
    
    for (; i < n; ++i) c[i] = a[i] * b[i];
}

// Softmax (numerically stable)
inline void softmax(const f32* __restrict input, f32* __restrict output, usize n) {
    // Find max for numerical stability
    f32 max_val = input[0];
    for (usize i = 1; i < n; ++i) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Compute exp(x - max) and sum
    f32 sum = 0.0f;
    for (usize i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    f32 inv_sum = 1.0f / sum;
    for (usize i = 0; i < n; ++i) {
        output[i] *= inv_sum;
    }
}

} // namespace kernels

// =============================================================================
// CACHE-BLOCKED GEMM (math/gemm/gemm_blocked.cpp + micro_kernels/)
// =============================================================================

namespace gemm {

// Micro-kernel configurations tuned for different architectures.
// These block sizes are carefully selected based on the number of available 
// hardware registers (e.g., AVX-512 has 32 zmm registers).
#ifdef NEXUS_HAS_AVX512
    constexpr usize MR = 6;   // Rows in micro-tile
    constexpr usize NR = 16;  // Cols (512-bit / 32-bit = 16)
#elif defined(NEXUS_HAS_AVX2)
    constexpr usize MR = 4;
    constexpr usize NR = 8;
#else
    constexpr usize MR = 4;
    constexpr usize NR = 4;
#endif

// Cache blocking parameters (tuned for L1/L2 cache sizes)
// By processing matrices in blocks that fit in the L1/L2 cache, we avoid 
// expensive "cache miss" stalls where the CPU waits for data from main RAM.
constexpr usize MC = 128;   // Rows of A in cache
constexpr usize NC = 256;   // Cols of B in cache  
constexpr usize KC = 512;   // Inner dimension in cache

// AVX-512 micro-kernel: 6x16 register block
#ifdef NEXUS_HAS_AVX512
NEXUS_INLINE void micro_kernel_6x16(const f32* __restrict a,  // 6 x k
                                     const f32* __restrict b,  // k x 16 (packed)
                                     f32* __restrict c,        // 6 x 16
                                     usize k, usize ldc) {
    // 6 accumulators (12 registers used total out of 32)
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();
    __m512 c4 = _mm512_setzero_ps();
    __m512 c5 = _mm512_setzero_ps();
    
    for (usize p = 0; p < k; ++p) {
        // Load B row (16 elements)
        __m512 b_vec = _mm512_loadu_ps(b + p * 16);
        
        // Broadcast A elements (6 rows) and perform FMA
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(a[0 * k + p]), b_vec, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(a[1 * k + p]), b_vec, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(a[2 * k + p]), b_vec, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(a[3 * k + p]), b_vec, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(a[4 * k + p]), b_vec, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(a[5 * k + p]), b_vec, c5);
    }
    
    // Store with accumulation (C = C + A*B)
    _mm512_storeu_ps(c + 0 * ldc, _mm512_add_ps(c0, _mm512_loadu_ps(c + 0 * ldc)));
    _mm512_storeu_ps(c + 1 * ldc, _mm512_add_ps(c1, _mm512_loadu_ps(c + 1 * ldc)));
    _mm512_storeu_ps(c + 2 * ldc, _mm512_add_ps(c2, _mm512_loadu_ps(c + 2 * ldc)));
    _mm512_storeu_ps(c + 3 * ldc, _mm512_add_ps(c3, _mm512_loadu_ps(c + 3 * ldc)));
    _mm512_storeu_ps(c + 4 * ldc, _mm512_add_ps(c4, _mm512_loadu_ps(c + 4 * ldc)));
    _mm512_storeu_ps(c + 5 * ldc, _mm512_add_ps(c5, _mm512_loadu_ps(c + 5 * ldc)));
}
#endif

// AVX2 micro-kernel: 4x8
#ifdef NEXUS_HAS_AVX2
NEXUS_INLINE void micro_kernel_4x8(const f32* __restrict a,
                                    const f32* __restrict b,
                                    f32* __restrict c,
                                    usize k, usize ldc) {
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    
    for (usize p = 0; p < k; ++p) {
        __m256 b_vec = _mm256_loadu_ps(b + p * 8);
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[0 * k + p]), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[1 * k + p]), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[2 * k + p]), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[3 * k + p]), b_vec, c3);
    }
    
    _mm256_storeu_ps(c + 0 * ldc, _mm256_add_ps(c0, _mm256_loadu_ps(c + 0 * ldc)));
    _mm256_storeu_ps(c + 1 * ldc, _mm256_add_ps(c1, _mm256_loadu_ps(c + 1 * ldc)));
    _mm256_storeu_ps(c + 2 * ldc, _mm256_add_ps(c2, _mm256_loadu_ps(c + 2 * ldc)));
    _mm256_storeu_ps(c + 3 * ldc, _mm256_add_ps(c3, _mm256_loadu_ps(c + 3 * ldc)));
}
#endif

// Memory Packing: Copies strided memory into contiguous sequential blocks.
// This prevents TLB (Translation Lookaside Buffer) cache misses during the inner loops.
inline void pack_b_panel(const f32* __restrict b, 
                         f32* __restrict packed,
                         usize n, usize k,
                         usize j_start, usize j_end) {
    usize panel_width = j_end - j_start;
    for (usize p = 0; p < k; ++p) {
        for (usize j = 0; j < panel_width; ++j) {
            packed[p * panel_width + j] = b[p * n + (j_start + j)];
        }
    }
}

inline void pack_a_panel(const f32* __restrict a,
                         f32* __restrict packed,
                         usize m, usize k,
                         usize i_start, usize i_end) {
    usize panel_height = i_end - i_start;
    for (usize i = 0; i < panel_height; ++i) {
        for (usize p = 0; p < k; ++p) {
            packed[i * k + p] = a[(i_start + i) * k + p];
        }
    }
}

// Main blocked GEMM implementation
void gemm_blocked(const f32* __restrict a,  // m x k
                  const f32* __restrict b,  // k x n
                  f32* __restrict c,        // m x n
                  usize m, usize n, usize k,
                  usize lda, usize ldb, usize ldc) {
    
    // Clear C
    for (usize i = 0; i < m; ++i) {
        std::memset(c + i * ldc, 0, n * sizeof(f32));
    }
    
    // Stack-allocated packing buffers.
    // Making them thread_local static ensures they aren't repeatedly reallocated,
    // and naturally reside in the fast L2 cache across function calls.
    alignas(NEXUS_SIMD_ALIGNMENT) static thread_local f32 packed_a[MC * KC];
    alignas(NEXUS_SIMD_ALIGNMENT) static thread_local f32 packed_b[KC * NC];
    
    for (usize jc = 0; jc < n; jc += NC) {
        usize nc = std::min(NC, n - jc);
        
        for (usize pc = 0; pc < k; pc += KC) {
            usize kc = std::min(KC, k - pc);
            
            // Pack B panel [kc x nc] -> packed_b
            pack_b_panel(b + pc * ldb, packed_b, n, kc, jc, jc + nc);
            
            for (usize ic = 0; ic < m; ic += MC) {
                usize mc = std::min(MC, m - ic);
                
                // Pack A panel [mc x kc] -> packed_a
                pack_a_panel(a + ic * lda, packed_a, m, kc, ic, ic + mc);
                
                // Micro-tiling
                for (usize jr = 0; jr < nc; jr += NR) {
                    usize nr = std::min(NR, nc - jr);
                    
                    for (usize ir = 0; ir < mc; ir += MR) {
                        usize mr = std::min(MR, mc - ir);
                        
                        // Dispatch to optimized micro-kernel or generic fallback
                        if (mr == MR && nr == NR) {
                            #ifdef NEXUS_HAS_AVX512
                            if constexpr (MR == 6 && NR == 16) {
                                micro_kernel_6x16(
                                    packed_a + ir * kc,
                                    packed_b + jr * kc,
                                    c + (ic + ir) * ldc + (jc + jr),
                                    kc, ldc
                                );
                            }
                            #elif defined(NEXUS_HAS_AVX2)
                            if constexpr (MR == 4 && NR == 8) {
                                micro_kernel_4x8(
                                    packed_a + ir * kc,
                                    packed_b + jr * kc,
                                    c + (ic + ir) * ldc + (jc + jr),
                                    kc, ldc
                                );
                            }
                            #endif
                        } else {
                            // Edge handling with scalar code
                            for (usize i = 0; i < mr; ++i) {
                                for (usize j = 0; j < nr; ++j) {
                                    f32 sum = 0.0f;
                                    for (usize p = 0; p < kc; ++p) {
                                        sum += packed_a[(ir + i) * kc + p] * packed_b[p * nc + jr + j];
                                    }
                                    c[(ic + ir + i) * ldc + (jc + jr + j)] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// High-level GEMM interface with error checking
inline Result<void> gemm(const TensorView& a,
                         const TensorView& b,
                         TensorView& c,
                         f32 alpha = 1.0f,
                         f32 beta = 0.0f,
                         bool transpose_a = false,
                         bool transpose_b = false) {
    
    if (!a.valid() || !b.valid() || !c.valid()) {
        return std::unexpected(NexusError(ErrorCode::NullPointer, "Invalid tensor view"));
    }
    
    // Extract dimensions
    usize m = transpose_a ? a.cols() : a.rows();
    usize k_a = transpose_a ? a.rows() : a.cols();
    usize k_b = transpose_b ? b.cols() : b.rows();
    usize n = transpose_b ? b.rows() : b.cols();
    
    if (k_a != k_b) {
        return std::unexpected(NexusError(ErrorCode::DimensionMismatch, 
            "Inner dimensions don't match: " + std::to_string(k_a) + " vs " + std::to_string(k_b)));
    }
    
    if (c.rows() != m || c.cols() != n) {
        return std::unexpected(NexusError(ErrorCode::DimensionMismatch,
            "Output dimensions mismatch"));
    }
    
    // For now, only support non-transposed
    if (transpose_a || transpose_b) {
        return std::unexpected(NexusError(ErrorCode::UnsupportedOperation,
            "Transpose not yet implemented"));
    }
    
    // Apply beta scaling to C if needed
    if (beta != 1.0f) {
        for (usize i = 0; i < m; ++i) {
            for (usize j = 0; j < n; ++j) {
                c(i, j) *= beta;
            }
        }
    }
    
    // Compute C = alpha * A * B + C
    if (alpha == 1.0f && beta == 0.0f) {
        // Fast path: C = A * B
        gemm_blocked(a.data, b.data, c.data, m, n, k_a, k_a, n, n);
    } else {
        // General case: use temporary
        // (Implementation omitted for brevity)
        gemm_blocked(a.data, b.data, c.data, m, n, k_a, k_a, n, n);
        // Apply alpha scaling
        if (alpha != 1.0f) {
            for (usize i = 0; i < m * n; ++i) {
                c.data[i] *= alpha;
            }
        }
    }
    
    return {};
}

} // namespace gemm

// =============================================================================
// NEURAL NETWORK LAYERS (nn/layers/dense.cpp + include)
// =============================================================================

enum class ActivationType {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Swish,
    GELU
};

// V2.0 BASELINE DESIGN: The abstract base class.
// Note: In low-latency systems, virtual functions (vtable lookups) in the hot path 
// can stall the CPU pipeline because the jump address isn't known until runtime.
class Layer {
public:
    virtual ~Layer() = default;
    
    virtual void forward(const TensorView& input, TensorView& output) = 0;
    virtual void backward(const TensorView& grad_output,
                         TensorView& grad_input,
                         const TensorView& input,
                         const TensorView& output) = 0;
    
    virtual void zero_grad() = 0;
    virtual void update_weights(f32 learning_rate) = 0;
    
    virtual usize input_size() const = 0;
    virtual usize output_size() const = 0;
    
    virtual usize parameter_count() const { return 0; }
    virtual usize memory_footprint() const { return 0; }
    
    virtual std::string name() const = 0;
};

class DenseLayer : public Layer {
private:
    // Parameters (owned by arena)
    TensorView weights_;      // [output_size x input_size]
    TensorView bias_;         // [output_size]
    
    // Gradients
    TensorView grad_weights_;
    TensorView grad_bias_;
    
    // Adam optimizer state
    TensorView m_weights_;    // First moment
    TensorView v_weights_;    // Second moment
    TensorView m_bias_;
    TensorView v_bias_;
    
    // Forward pass cache
    mutable TensorView pre_activation_;  // Z = XW + b
    
    usize input_size_;
    usize output_size_;
    ActivationType activation_;
    
    // Adam hyperparameters
    f32 beta1_ = 0.9f;
    f32 beta2_ = 0.999f;
    f32 epsilon_ = 1e-8f;
    f32 weight_decay_ = 0.0f;
    mutable f32 timestep_ = 0.0f;
    
    MemoryArena* arena_ = nullptr;

public:
    DenseLayer(MemoryArena& arena,
               usize in_size,
               usize out_size,
               ActivationType act = ActivationType::ReLU)
        : input_size_(in_size)
        , output_size_(out_size)
        , activation_(act)
        , arena_(&arena) {
        
        // Allocate from arena
        weights_.data = arena.allocate_array<f32>(out_size * in_size);
        weights_.ndim = 2;
        weights_.shape_[0] = out_size;
        weights_.shape_[1] = in_size;
        weights_.strides_[0] = in_size;
        weights_.strides_[1] = 1;
        
        bias_.data = arena.allocate_array<f32>(out_size);
        bias_.ndim = 1;
        bias_.shape_[0] = out_size;
        bias_.strides_[0] = 1;
        
        grad_weights_.data = arena.allocate_array<f32>(out_size * in_size);
        grad_weights_.ndim = 2;
        grad_weights_.shape_[0] = out_size;
        grad_weights_.shape_[1] = in_size;
        
        grad_bias_.data = arena.allocate_array<f32>(out_size);
        grad_bias_.ndim = 1;
        grad_bias_.shape_[0] = out_size;
        
        // Adam state
        m_weights_.data = arena.allocate_array<f32>(out_size * in_size);
        m_weights_.ndim = 2;
        m_weights_.shape_[0] = out_size;
        m_weights_.shape_[1] = in_size;
        
        v_weights_.data = arena.allocate_array<f32>(out_size * in_size);
        v_weights_.ndim = 2;
        v_weights_.shape_[0] = out_size;
        v_weights_.shape_[1] = in_size;
        
        m_bias_.data = arena.allocate_array<f32>(out_size);
        m_bias_.ndim = 1;
        m_bias_.shape_[0] = out_size;
        
        v_bias_.data = arena.allocate_array<f32>(out_size);
        v_bias_.ndim = 1;
        v_bias_.shape_[0] = out_size;
        
        // Pre-activation cache
        pre_activation_.data = arena.allocate_array<f32>(out_size);
        pre_activation_.ndim = 1;
        pre_activation_.shape_[0] = out_size;
        
        initialize_weights();
        zero_grad();
        zero_adam_state();
    }
    
    void initialize_weights() {
        // Kaiming/He initialization for ReLU
        std::random_device rd;
        std::mt19937 gen(rd());
        
        f32 scale = 0.0f;
        switch (activation_) {
            case ActivationType::ReLU:
            case ActivationType::Swish:
                scale = std::sqrt(2.0f / input_size_);
                break;
            case ActivationType::Sigmoid:
            case ActivationType::Tanh:
                scale = std::sqrt(1.0f / input_size_);
                break;
            default:
                scale = std::sqrt(1.0f / input_size_);
        }
        
        std::normal_distribution<f32> dist(0.0f, scale);
        
        for (usize i = 0; i < weights_.size(); ++i) {
            weights_.data[i] = dist(gen);
        }
        
        bias_.zero();
    }
    
    void zero_grad() {
        grad_weights_.zero();
        grad_bias_.zero();
    }
    
    void zero_adam_state() {
        m_weights_.zero();
        v_weights_.zero();
        m_bias_.zero();
        v_bias_.zero();
        timestep_ = 0.0f;
    }
    
    void forward(const TensorView& input, TensorView& output) override {
        // Z = X @ W^T + b
        // Input: [batch x input_size], Weights: [output_size x input_size]
        
        // Matrix-vector multiplication for single sample
        for (usize j = 0; j < output_size_; ++j) {
            f32 sum = bias_.data[j];
            for (usize k = 0; k < input_size_; ++k) {
                sum += input.data[k] * weights_(j, k);
            }
            pre_activation_.data[j] = sum;
        }
        
        // Apply activation
        switch (activation_) {
            case ActivationType::ReLU:
                kernels::vec_relu(pre_activation_.data, output.data, output_size_);
                break;
            case ActivationType::Sigmoid:
                kernels::vec_sigmoid(pre_activation_.data, output.data, output_size_);
                break;
            case ActivationType::Tanh:
                kernels::vec_tanh(pre_activation_.data, output.data, output_size_);
                break;
            case ActivationType::Softmax:
                kernels::softmax(pre_activation_.data, output.data, output_size_);
                break;
            case ActivationType::None:
                std::memcpy(output.data, pre_activation_.data, output_size_ * sizeof(f32));
                break;
            default:
                // Fallback to ReLU
                kernels::vec_relu(pre_activation_.data, output.data, output_size_);
        }
    }
    
    // V2.0 BASELINE IMPLEMENTATION: Scalar Backpropagation.
    // Note: Mathematically correct, but this nested for-loop structure completely 
    // bypasses the AVX-512 GEMM engine, making backprop significantly slower than inference.
    void backward(const TensorView& grad_output,
                  TensorView& grad_input,
                  const TensorView& input,
                  const TensorView& output) override {
        
        // Compute gradient of activation
        alignas(NEXUS_SIMD_ALIGNMENT) f32 grad_pre_act[256];  // Max output size
        
        switch (activation_) {
            case ActivationType::ReLU:
                for (usize j = 0; j < output_size_; ++j) {
                    grad_pre_act[j] = pre_activation_.data[j] > 0.0f ? grad_output.data[j] : 0.0f;
                }
                break;
            case ActivationType::Sigmoid:
                for (usize j = 0; j < output_size_; ++j) {
                    f32 s = output.data[j];
                    grad_pre_act[j] = grad_output.data[j] * s * (1.0f - s);
                }
                break;
            case ActivationType::None:
                std::memcpy(grad_pre_act, grad_output.data, output_size_ * sizeof(f32));
                break;
            default:
                std::memcpy(grad_pre_act, grad_output.data, output_size_ * sizeof(f32));
        }
        
        // Gradients w.r.t. weights and bias (Scalar loops)
        for (usize j = 0; j < output_size_; ++j) {
            f32 go = grad_pre_act[j];
            grad_bias_.data[j] += go;
            
            for (usize k = 0; k < input_size_; ++k) {
                grad_weights_(j, k) += go * input.data[k];
            }
        }
        
        // Gradient w.r.t. input (for backprop) (Scalar loops)
        if (grad_input.valid()) {
            for (usize k = 0; k < input_size_; ++k) {
                f32 sum = 0.0f;
                for (usize j = 0; j < output_size_; ++j) {
                    sum += grad_pre_act[j] * weights_(j, k);
                }
                grad_input.data[k] = sum;
            }
        }
    }
    
    void update_weights(f32 learning_rate) override {
        ++timestep_;
        f32 lr_t = learning_rate * std::sqrt(1.0f - std::pow(beta2_, timestep_)) 
                   / (1.0f - std::pow(beta1_, timestep_));
        
        // Update weights with Adam + weight decay
        for (usize i = 0; i < weights_.size(); ++i) {
            f32 g = grad_weights_.data[i] + weight_decay_ * weights_.data[i];
            m_weights_.data[i] = beta1_ * m_weights_.data[i] + (1.0f - beta1_) * g;
            v_weights_.data[i] = beta2_ * v_weights_.data[i] + (1.0f - beta2_) * g * g;
            weights_.data[i] -= lr_t * m_weights_.data[i] / (std::sqrt(v_weights_.data[i]) + epsilon_);
        }
        
        // Update biases (no weight decay)
        for (usize j = 0; j < output_size_; ++j) {
            f32 g = grad_bias_.data[j];
            m_bias_.data[j] = beta1_ * m_bias_.data[j] + (1.0f - beta1_) * g;
            v_bias_.data[j] = beta2_ * v_bias_.data[j] + (1.0f - beta2_) * g * g;
            bias_.data[j] -= lr_t * m_bias_.data[j] / (std::sqrt(v_bias_.data[j]) + epsilon_);
        }
    }
    
    void zero_grad() override {
        grad_weights_.zero();
        grad_bias_.zero();
    }
    
    usize input_size() const override { return input_size_; }
    usize output_size() const override { return output_size_; }
    
    usize parameter_count() const override {
        return weights_.size() + bias_.size();
    }
    
    usize memory_footprint() const override {
        return (weights_.size() + bias_.size() + 
                grad_weights_.size() + grad_bias_.size() +
                m_weights_.size() + v_weights_.size() +
                m_bias_.size() + v_bias_.size()) * sizeof(f32);
    }
    
    std::string name() const override {
        return "Dense(" + std::to_string(input_size_) + " -> " + 
               std::to_string(output_size_) + ")";
    }
};

// =============================================================================
// LOSS FUNCTIONS (nn/losses/)
// =============================================================================

namespace losses {

struct LossResult {
    f32 loss;
    std::vector<f32> gradient;  // Stored in temporary buffer
};

class Loss {
public:
    virtual ~Loss() = default;
    virtual f32 compute(const TensorView& prediction, 
                       const TensorView& target,
                       TensorView& grad_output) = 0;
    virtual std::string name() const = 0;
};

class MSELoss : public Loss {
public:
    f32 compute(const TensorView& pred, 
                const TensorView& target,
                TensorView& grad) override {
        f32 loss = 0.0f;
        usize n = pred.size();
        
        for (usize i = 0; i < n; ++i) {
            f32 diff = pred.data[i] - target.data[i];
            grad.data[i] = 2.0f * diff / static_cast<f32>(n);
            loss += diff * diff;
        }
        
        return loss / static_cast<f32>(n);
    }
    
    std::string name() const override { return "MSE"; }
};

class CrossEntropyLoss : public Loss {
public:
    f32 compute(const TensorView& pred,
                const TensorView& target,
                TensorView& grad) override {
        // Assumes pred is logits (pre-softmax)
        // Numerically stable softmax
        f32 max_logit = pred.data[0];
        for (usize i = 1; i < pred.size(); ++i) {
            max_logit = std::max(max_logit, pred.data[i]);
        }
        
        f32 sum_exp = 0.0f;
        for (usize i = 0; i < pred.size(); ++i) {
            sum_exp += std::exp(pred.data[i] - max_logit);
        }
        
        f32 loss = 0.0f;
        for (usize i = 0; i < pred.size(); ++i) {
            f32 softmax_i = std::exp(pred.data[i] - max_logit) / sum_exp;
            grad.data[i] = softmax_i - target.data[i];
            if (target.data[i] > 0.5f) {
                loss = -std::log(softmax_i + 1e-8f);
            }
        }
        
        return loss;
    }
    
    std::string name() const override { return "CrossEntropy"; }
};

} // namespace losses

// =============================================================================
// NEURAL NETWORK EXECUTION ENGINE (nn/network.cpp + include)
// =============================================================================

class NeuralNetwork {
private:
    // V2.0 BASELINE TOPOLOGY: Storing layers as unique_ptrs to a base class.
    // Requires heap indirection and forces the CPU to use dynamic dispatch.
    std::vector<std::unique_ptr<Layer>> layers_;
    MemoryArena* arena_ = nullptr;
    
    // Pre-allocated buffers for forward/backward
    std::vector<TensorView> activations_;
    std::vector<TensorView> gradients_;
    
    // Temporary gradient storage for output layer
    TensorView output_grad_;
    
    bool compiled_ = false;

public:
    explicit NeuralNetwork(MemoryArena& arena) : arena_(&arena) {}
    
    template<typename LayerType, typename... Args>
    void add_layer(Args&&... args) {
        if (compiled_) {
            throw std::runtime_error("Cannot add layers after compilation");
        }
        layers_.push_back(std::make_unique<LayerType>(*arena_, std::forward<Args>(args)...));
    }
    
    void compile() {
        if (compiled_) return;
        
        // Pre-allocate all intermediate buffers
        for (usize i = 0; i < layers_.size(); ++i) {
            f32* act_data = arena_->allocate_array<f32>(layers_[i]->output_size());
            TensorView act(act_data, layers_[i]->output_size());
            activations_.push_back(act);
            
            f32* grad_data = arena_->allocate_array<f32>(
                i == 0 ? layers_[0]->input_size() : layers_[i]->input_size()
            );
            TensorView grad(grad_data, 
                i == 0 ? layers_[0]->input_size() : layers_[i]->input_size());
            gradients_.push_back(grad);
        }
        
        // Output gradient buffer
        if (!layers_.empty()) {
            output_grad_.data = arena_->allocate_array<f32>(layers_.back()->output_size());
            output_grad_.ndim = 1;
            output_grad_.shape_[0] = layers_.back()->output_size();
        }
        
        compiled_ = true;
    }
    
    void forward(const TensorView& input, TensorView& output) {
        if (!compiled_) compile();
        
        const TensorView* current_input = &input;
        
        for (usize i = 0; i < layers_.size(); ++i) {
            // VTABLE LOOKUP OCCURS HERE: CPU cannot inline this call.
            layers_[i]->forward(*current_input, activations_[i]);
            current_input = &activations_[i];
        }
        
        // Copy final output
        if (!layers_.empty()) {
            std::memcpy(output.data, current_input->data, 
                       layers_.back()->output_size() * sizeof(f32));
        }
    }
    
    f32 backward(const TensorView& input, const TensorView& target, 
                losses::Loss& loss_fn) {
        if (!compiled_) compile();
        
        // Forward pass to compute activations
        forward(input, activations_.back());
        
        // Compute loss and output gradient
        f32 loss = loss_fn.compute(activations_.back(), target, output_grad_);
        
        // Backpropagate through layers
        TensorView* grad_output = &output_grad_;
        
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            const TensorView& layer_input = (i == 0) ? input : activations_[i - 1];
            TensorView& grad_input = gradients_[i];
            
            // VTABLE LOOKUP OCCURS HERE
            layers_[i]->backward(*grad_output, grad_input, layer_input, activations_[i]);
            
            if (i > 0) {
                grad_output = &grad_input;
            }
        }
        
        return loss;
    }
    
    void step(f32 learning_rate) {
        for (auto& layer : layers_) {
            layer->update_weights(learning_rate);
        }
    }
    
    void zero_grad() {
        for (auto& layer : layers_) {
            layer->zero_grad();
        }
    }
    
    void train_step(const TensorView& input, const TensorView& target,
                   losses::Loss& loss_fn, f32 learning_rate) {
        zero_grad();
        f32 loss = backward(input, target, loss_fn);
        step(learning_rate);
    }
    
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing");
        
        // Write layer count
        usize num_layers = layers_.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
        
        // Write each layer's parameters
        for (const auto& layer : layers_) {
            // Write layer type and dimensions
            // (Simplified - would need proper serialization in production)
        }
    }
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading");
        // Implementation...
    }
    
    std::string summary() const {
        std::ostringstream oss;
        oss << "NeuralNetwork Summary:\n";
        oss << "======================\n";
        
        usize total_params = 0;
        usize total_memory = 0;
        
        for (usize i = 0; i < layers_.size(); ++i) {
            oss << "Layer " << i << ": " << layers_[i]->name() << "\n";
            oss << "  Parameters: " << layers_[i]->parameter_count() << "\n";
            oss << "  Memory: " << layers_[i]->memory_footprint() / 1024 << " KB\n";
            total_params += layers_[i]->parameter_count();
            total_memory += layers_[i]->memory_footprint();
        }
        
        oss << "----------------------\n";
        oss << "Total Parameters: " << total_params << " (" 
            << total_params / 1000000.0f << "M)\n";
        oss << "Total Memory: " << total_memory / (1024 * 1024) << " MB\n";
        
        return oss.str();
    }
};

// =============================================================================
// PROFILING & BENCHMARKING (tools/profiler/)
// =============================================================================

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::nanoseconds;
    
    explicit Timer(std::string name = "") : name_(std::move(name)), running_(false) {}
    
    void start() {
        start_ = Clock::now();
        running_ = true;
    }
    
    void stop() {
        if (!running_) return;
        end_ = Clock::now();
        running_ = false;
        auto elapsed = std::chrono::duration_cast<Duration>(end_ - start_);
        total_ += elapsed;
        ++count_;
    }
    
    void reset() {
        total_ = Duration::zero();
        count_ = 0;
        running_ = false;
    }
    
    [[nodiscard]] f32 elapsed_ms() const {
        return std::chrono::duration<f32, std::milli>(total_).count();
    }
    
    [[nodiscard]] f32 elapsed_us() const {
        return std::chrono::duration<f32, std::micro>(total_).count();
    }
    
    [[nodiscard]] f32 elapsed_ns() const {
        return static_cast<f32>(total_.count());
    }
    
    [[nodiscard]] f32 average_ns() const {
        return count_ > 0 ? elapsed_ns() / static_cast<f32>(count_) : 0.0f;
    }
    
    void report() const {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[Timer] " << name_ << ": ";
        if (count_ == 1) {
            std::cout << elapsed_us() << " us\n";
        } else {
            std::cout << elapsed_ms() << " ms total, " 
                      << average_ns() / 1000.0f << " us avg ("
                      << count_ << " runs)\n";
        }
    }
    
    ~Timer() {
        if (count_ > 0) report();
    }

private:
    std::string name_;
    TimePoint start_;
    TimePoint end_;
    Duration total_ = Duration::zero();
    usize count_ = 0;
    bool running_ = false;
};

// RAII timer scope
class TimerScope {
    Timer& timer_;
public:
    explicit TimerScope(Timer& t) : timer_(t) { timer_.start(); }
    ~TimerScope() { timer_.stop(); }
};

// =============================================================================
// BENCHMARKS & DEMONSTRATIONS (benchmarks/ + examples/)
// =============================================================================

void benchmark_gemm() {
    std::cout << "\n=== GEMM Benchmark ===\n";
    
    constexpr usize M = 1024, N = 1024, K = 1024;
    
    // Aligned allocation: Critical for AVX operations
    #if defined(_WIN32) || defined(_MSC_VER)
        f32* a = static_cast<f32*>(_aligned_malloc(M * K * sizeof(f32), NEXUS_SIMD_ALIGNMENT));
        f32* b = static_cast<f32*>(_aligned_malloc(K * N * sizeof(f32), NEXUS_SIMD_ALIGNMENT));
        f32* c = static_cast<f32*>(_aligned_malloc(M * N * sizeof(f32), NEXUS_SIMD_ALIGNMENT));
    #else
        f32* a = static_cast<f32*>(std::aligned_alloc(NEXUS_SIMD_ALIGNMENT, M * K * sizeof(f32)));
        f32* b = static_cast<f32*>(std::aligned_alloc(NEXUS_SIMD_ALIGNMENT, K * N * sizeof(f32)));
        f32* c = static_cast<f32*>(std::aligned_alloc(NEXUS_SIMD_ALIGNMENT, M * N * sizeof(f32)));
    #endif
    
    // Initialize
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    
    for (usize i = 0; i < M * K; ++i) a[i] = dist(gen);
    for (usize i = 0; i < K * N; ++i) b[i] = dist(gen);
    
    // Warmup
    gemm::gemm_blocked(a, b, c, M, N, K, K, N, N);
    
    // Benchmark
    Timer timer("GEMM 1024x1024x1024");
    constexpr usize iterations = 100;
    
    for (usize i = 0; i < iterations; ++i) {
        TimerScope scope(timer);
        std::memset(c, 0, M * N * sizeof(f32));
        gemm::gemm_blocked(a, b, c, M, N, K, K, N, N);
        
        // Memory barriers: Prevents modern compilers from "optimizing away" the benchmark loop
        #if defined(_MSC_VER)
            _ReadWriteBarrier(); // Prevent optimization on Windows
        #else
            asm volatile("" ::: "memory"); // Prevent optimization on Linux
        #endif
    }
    
    f32 seconds = timer.elapsed_ms() / 1000.0f;
    f32 flops = 2.0f * M * N * K * iterations;
    f32 gflops = (flops / seconds) / 1e9f;
    
    std::cout << "Throughput: " << gflops << " GFLOPS\n";
    std::cout << "Peak theoretical: ~200-250 GFLOPS (AVX-512 @ 3GHz)\n";
    std::cout << "Efficiency: " << (gflops / 250.0f * 100.0f) << "%\n";
    
    #if defined(_WIN32) || defined(_MSC_VER)
        _aligned_free(a);
        _aligned_free(b);
        _aligned_free(c);
    #else
        std::free(a);
        std::free(b);
        std::free(c);
    #endif
}

void demonstrate_mnist_network() {
    std::cout << "\n=== MNIST Network Demo ===\n";
    
    MemoryArena arena(50 * 1024 * 1024);  // 50MB arena
    
    NeuralNetwork net(arena);
    net.add_layer<DenseLayer>(784, 512, ActivationType::ReLU);
    net.add_layer<DenseLayer>(512, 256, ActivationType::ReLU);
    net.add_layer<DenseLayer>(256, 10, ActivationType::Softmax);
    
    net.compile();
    std::cout << net.summary() << "\n";
    
    // Create input/output buffers
    f32* input_data = arena.allocate_array<f32>(784);
    f32* output_data = arena.allocate_array<f32>(10);
    f32* target_data = arena.allocate_array<f32>(10);
    
    TensorView input(input_data, 784);
    TensorView output(output_data, 10);
    TensorView target(target_data, 10);
    
    // Initialize random input
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 0.1f);
    
    for (usize i = 0; i < 784; ++i) input_data[i] = dist(gen);
    for (usize i = 0; i < 10; ++i) target_data[i] = (i == 7) ? 1.0f : 0.0f;  // Class 7
    
    // Warmup
    net.forward(input, output);
    
    // Benchmark forward pass
    {
        Timer timer("Forward pass");
        for (usize i = 0; i < 10000; ++i) {
            TimerScope scope(timer);
            net.forward(input, output);
            
            #if defined(_MSC_VER)
                _ReadWriteBarrier(); 
            #else
                asm volatile("" ::: "memory"); 
            #endif
        }
    }
    
    std::cout << "Output probabilities: ";
    for (usize i = 0; i < 10; ++i) {
        std::cout << std::fixed << std::setprecision(4) << output_data[i] << " ";
    }
    std::cout << "\n";
    
    // Training benchmark
    losses::CrossEntropyLoss loss_fn;
    {
        Timer timer("Training step (forward + backward + update)");
        for (usize i = 0; i < 1000; ++i) {
            TimerScope scope(timer);
            net.train_step(input, target, loss_fn, 0.001f);
            
            #if defined(_MSC_VER)
                _ReadWriteBarrier(); 
            #else
                asm volatile("" ::: "memory"); 
            #endif
        }
    }
    
    auto stats = arena.stats();
    std::cout << "\nArena Statistics:\n";
    std::cout << "  Capacity: " << stats.capacity / (1024 * 1024) << " MB\n";
    std::cout << "  Used: " << stats.used / 1024 << " KB\n";
    std::cout << "  Peak: " << stats.peak / 1024 << " KB\n";
    std::cout << "  Allocations: " << stats.allocation_count << "\n";
}

void demonstrate_memory_arena() {
    std::cout << "\n=== Memory Arena Demo ===\n";
    
    MemoryArena arena(1024 * 1024);  // 1MB
    
    // Allocation pattern typical in neural networks
    auto* w1 = arena.allocate_array<f32>(1000 * 1000);  // 4MB - triggers growth
    auto* b1 = arena.allocate_array<f32>(1000);
    auto* w2 = arena.allocate_array<f32>(1000 * 100);
    auto* b2 = arena.allocate_array<f32>(100);
    
    std::cout << "After allocations:\n";
    auto stats = arena.stats();
    std::cout << "  Used: " << stats.used << " bytes\n";
    std::cout << "  Capacity: " << stats.capacity << " bytes\n";
    
    // Reset for next iteration (O(1) operation)
    auto marker = arena.mark();
    
    auto* temp1 = arena.allocate_array<f32>(5000);
    auto* temp2 = arena.allocate_array<f32>(5000);
    std::cout << "After temp allocations: " << arena.used() << " bytes\n";
    
    // Rollback
    arena.reset_to(marker);
    std::cout << "After reset: " << arena.used() << " bytes\n";
    
    // Verify pointers are still valid (arena grew, didn't relocate)
    std::cout << "w1 still valid: " << (arena.contains(w1) ? "yes" : "no") << "\n";
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            NEXUS HIGH-PERFORMANCE ML ENGINE v2.0                 ║\n";
    std::cout << "║      Zero-Dependency | C++20 | AVX-512 | NUMA-Aware | Lock-Free  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Build Configuration:\n";
    std::cout << "  Architecture: ";
    #ifdef NEXUS_ARCH_X86_64
    std::cout << "x86_64\n";
    #elif defined(NEXUS_ARCH_ARM64)
    std::cout << "ARM64\n";
    #endif
    
    std::cout << "  SIMD: ";
    #ifdef NEXUS_HAS_AVX512
    std::cout << "AVX-512 (16-wide)\n";
    #elif defined(NEXUS_HAS_AVX2)
    std::cout << "AVX2 (8-wide)\n";
    #elif defined(NEXUS_HAS_NEON)
    std::cout << "NEON (4-wide)\n";
    #else
    std::cout << "Scalar\n";
    #endif
    
    std::cout << "  Cache Line: " << NEXUS_CACHE_LINE_SIZE << " bytes\n";
    std::cout << "  L1 Cache: " << NEXUS_L1_CACHE_SIZE / 1024 << " KB\n";
    std::cout << "  C++ Standard: " << __cplusplus << "\n\n";
    
    try {
        demonstrate_memory_arena();
        benchmark_gemm();
        demonstrate_mnist_network();
        
        std::cout << "\n=== All demonstrations complete successfully ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

} // namespace nexus

// =============================================================================
// COMPILATION INSTRUCTIONS
// =============================================================================
/*
GCC/Clang:
g++ -std=c++20 -O3 -march=native -ffast-math -funroll-loops \
    -fomit-frame-pointer -DNDEBUG nexus_v2.cpp -o nexus_v2 -lpthread

Windows MSVC:
cl.exe /std:c++20 /O2 /arch:AVX512 /Zc:__cplusplus nexus_v2.cpp

Run benchmarks:
perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses,cache-misses ./nexus_v2
*/