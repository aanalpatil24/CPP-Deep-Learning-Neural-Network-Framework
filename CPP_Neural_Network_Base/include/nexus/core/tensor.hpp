#pragma once
#include "../config.hpp"
#include <array>
#include <span>
#include <stdexcept>

namespace nexus {

// TENSOR VIEW (core/tensor.hpp+ include)
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

}
