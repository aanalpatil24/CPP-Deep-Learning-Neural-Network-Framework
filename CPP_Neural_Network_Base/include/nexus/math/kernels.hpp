#pragma once
#include "../config.hpp"
#include <cmath>
#include <algorithm>

namespace nexus {

// SIMD MATHEMATICS KERNELS (math/kernels.hpp + include)

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

} // namespace nexus
