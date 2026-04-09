#pragma once
#include "../config.hpp"
#include <array>
#include <span>
#include <stdexcept>

namespace nexus {

// TENSOR VIEW
// A lightweight, non-owning view over contiguous memory.
// Similar to std::span, but tracks multidimensional shapes and strides.
// This prevents expensive deep copies of matrices during the hot path.

class TensorView {
public:
    f32* data = nullptr;
    usize ndim = 0;
    std::array<usize, 6> shape_{};      
    std::array<usize, 6> strides_{};    
    
    TensorView() = default;
    
    TensorView(f32* d, usize dim0) 
        : data(d), ndim(1) {
        shape_[0] = dim0;
        strides_[0] = 1;
    }
    
    TensorView(f32* d, usize dim0, usize dim1)
        : data(d), ndim(2) {
        shape_[0] = dim0;
        shape_[1] = dim1;
        strides_[0] = dim1;
        strides_[1] = 1;
    }
    
    TensorView(f32* d, std::span<const usize> shape, std::span<const usize> strides = {})
        : data(d), ndim(shape.size()) {
        if (ndim > 6) throw std::invalid_argument("Max 6 dimensions supported");
        std::copy(shape.begin(), shape.end(), shape_.begin());
        
        if (strides.empty()) {
            // Automatically calculate row-major (C-style) strides if none provided
            strides_[ndim - 1] = 1;
            for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        } else {
            std::copy(strides.begin(), strides.end(), strides_.begin());
        }
    }

    // Variadic template element access
    template<typename... Indices>
    [[nodiscard]] f32& operator()(Indices... indices) {
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
    
    [[nodiscard]] f32& operator[](usize idx) { return data[idx]; }
    [[nodiscard]] const f32& operator[](usize idx) const { return data[idx]; }
    
    [[nodiscard]] usize size() const noexcept {
        usize total = 1;
        for (usize i = 0; i < ndim; ++i) total *= shape_[i];
        return total;
    }
    
    [[nodiscard]] usize rows() const noexcept { return ndim > 0 ? shape_[0] : 0; }
    [[nodiscard]] usize cols() const noexcept { return ndim > 1 ? shape_[1] : 0; }
    
    void zero() noexcept {
        std::memset(data, 0, size() * sizeof(f32));
    }
    
    [[nodiscard]] bool valid() const noexcept {
        return data != nullptr && size() > 0;
    }
};

}
