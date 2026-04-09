#pragma once
#include "../config.hpp"
#include "../core/tensor.hpp"
#include "../core/error.hpp"

namespace nexus {

// CACHE-BLOCKED GEMM & TRANSPOSITION

namespace gemm {

// Cache configuration: Matrix math is broken into "Tiles" that fit perfectly 
// into L1 and L2 caches, avoiding expensive reads from main RAM.
constexpr usize MC = 128;
constexpr usize NC = 256;  
constexpr usize KC = 512;  

// Cache-blocked Matrix Transposition
// Standard matrix transposition thrashes the TLB (Translation Lookaside Buffer).
// Iterating in 32x32 chunks maintains memory locality.
inline void transpose_blocked(const f32* __restrict src, f32* __restrict dst, usize rows, usize cols) {
    constexpr usize BLOCK = 32; 
    for (usize i = 0; i < rows; i += BLOCK) {
        for (usize j = 0; j < cols; j += BLOCK) {
            for (usize ii = i; ii < std::min(i + BLOCK, rows); ++ii) {
                for (usize jj = j; jj < std::min(j + BLOCK, cols); ++jj) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

#ifdef NEXUS_HAS_AVX512
constexpr usize MR = 6;
constexpr usize NR = 16;

// The heart of the compute engine. 
// Processes a 6x16 sub-matrix entirely within the CPU registers without touching RAM.
NEXUS_INLINE void micro_kernel_6x16(const f32* __restrict a, const f32* __restrict b, 
                                     f32* __restrict c, usize k, usize ldc) {
    // Initialize 6 accumulators. We use 12 out of the 32 available AVX-512 zmm registers.
    __m512 c0 = _mm512_setzero_ps(); __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps(); __m512 c3 = _mm512_setzero_ps();
    __m512 c4 = _mm512_setzero_ps(); __m512 c5 = _mm512_setzero_ps();
    
    for (usize p = 0; p < k; ++p) {
        // Load a 16-float wide row of matrix B
        __m512 b_vec = _mm512_loadu_ps(b + p * 16);
        
        // Broadcast single scalars from matrix A to 512-bit registers, and perform
        // Fused Multiply-Add (A * B + C) in a single hardware instruction cycle.
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(a[0 * k + p]), b_vec, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(a[1 * k + p]), b_vec, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(a[2 * k + p]), b_vec, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(a[3 * k + p]), b_vec, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(a[4 * k + p]), b_vec, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(a[5 * k + p]), b_vec, c5);
    }
    
    // Store accumulated results back to main memory
    _mm512_storeu_ps(c + 0 * ldc, _mm512_add_ps(c0, _mm512_loadu_ps(c + 0 * ldc)));
    _mm512_storeu_ps(c + 1 * ldc, _mm512_add_ps(c1, _mm512_loadu_ps(c + 1 * ldc)));
    _mm512_storeu_ps(c + 2 * ldc, _mm512_add_ps(c2, _mm512_loadu_ps(c + 2 * ldc)));
    _mm512_storeu_ps(c + 3 * ldc, _mm512_add_ps(c3, _mm512_loadu_ps(c + 3 * ldc)));
    _mm512_storeu_ps(c + 4 * ldc, _mm512_add_ps(c4, _mm512_loadu_ps(c + 4 * ldc)));
    _mm512_storeu_ps(c + 5 * ldc, _mm512_add_ps(c5, _mm512_loadu_ps(c + 5 * ldc)));
}
#elif defined(NEXUS_HAS_AVX2)
constexpr usize MR = 4;
constexpr usize NR = 8;
NEXUS_INLINE void micro_kernel_4x8(const f32* __restrict a, const f32* __restrict b, 
                                    f32* __restrict c, usize k, usize ldc) {
    __m256 c0 = _mm256_setzero_ps(); __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(); __m256 c3 = _mm256_setzero_ps();
    
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
#else
constexpr usize MR = 4;
constexpr usize NR = 4;
#endif

// Panel packing routines convert standard 2D arrays into continuous blocks.
// This prevents TLB (Translation Lookaside Buffer) thrashing during GEMM execution.
inline void pack_b_panel(const f32* __restrict b, f32* __restrict packed, 
                         usize n, usize k, usize j_start, usize j_end) {
    usize panel_width = j_end - j_start;
    for (usize p = 0; p < k; ++p) {
        for (usize j = 0; j < panel_width; ++j) {
            packed[p * panel_width + j] = b[p * n + (j_start + j)];
        }
    }
}

inline void pack_a_panel(const f32* __restrict a, f32* __restrict packed, 
                         usize m, usize k, usize i_start, usize i_end) {
    usize panel_height = i_end - i_start;
    for (usize i = 0; i < panel_height; ++i) {
        for (usize p = 0; p < k; ++p) {
            packed[i * k + p] = a[(i_start + i) * k + p];
        }
    }
}

// Master Matrix Multiplication Routine (C = A * B)
void gemm_blocked(const f32* __restrict a, const f32* __restrict b, f32* __restrict c, 
                  usize m, usize n, usize k, usize lda, usize ldb, usize ldc) {
    
    for (usize i = 0; i < m; ++i) {
        std::memset(c + i * ldc, 0, n * sizeof(f32));
    }
    
    // Thread-local packing buffers guarantee they reside in L2 cache across loops.
    alignas(NEXUS_SIMD_ALIGNMENT) static thread_local f32 packed_a[MC * KC];
    alignas(NEXUS_SIMD_ALIGNMENT) static thread_local f32 packed_b[KC * NC];
    
    // Outer loops manage L2 Cache Blocking
    for (usize jc = 0; jc < n; jc += NC) {
        usize nc = std::min(NC, n - jc);
        for (usize pc = 0; pc < k; pc += KC) {
            usize kc = std::min(KC, k - pc);
            pack_b_panel(b + pc * ldb, packed_b, n, kc, jc, jc + nc);
            
            // Inner loops manage L1 Cache Blocking
            for (usize ic = 0; ic < m; ic += MC) {
                usize mc = std::min(MC, m - ic);
                pack_a_panel(a + ic * lda, packed_a, m, kc, ic, ic + mc);
                
                // Micro-tiling: Dispatch to hand-tuned AVX-512 kernels
                for (usize jr = 0; jr < nc; jr += NR) {
                    usize nr = std::min(NR, nc - jr);
                    for (usize ir = 0; ir < mc; ir += MR) {
                        usize mr = std::min(MR, mc - ir);
                        
                        if (mr == MR && nr == NR) {
                            #ifdef NEXUS_HAS_AVX512
                            if constexpr (MR == 6 && NR == 16) {
                                micro_kernel_6x16(packed_a + ir * kc, packed_b + jr * kc,
                                                  c + (ic + ir) * ldc + (jc + jr), kc, ldc);
                            }
                            #elif defined(NEXUS_HAS_AVX2)
                            if constexpr (MR == 4 && NR == 8) {
                                micro_kernel_4x8(packed_a + ir * kc, packed_b + jr * kc,
                                                 c + (ic + ir) * ldc + (jc + jr), kc, ldc);
                            }
                            #endif
                        } else {
                            // Scalar edge handling for dimensions that are not multiples of SIMD width.
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

inline Result<void> gemm(const TensorView& a, const TensorView& b, TensorView& c,
                         f32 alpha = 1.0f, f32 beta = 0.0f,
                         bool transpose_a = false, bool transpose_b = false) {
    if (!a.valid() || !b.valid() || !c.valid()) 
        return std::unexpected(NexusError(ErrorCode::NullPointer, "Invalid tensor view"));
    
    usize m = transpose_a ? a.cols() : a.rows();
    usize k_a = transpose_a ? a.rows() : a.cols();
    usize k_b = transpose_b ? b.cols() : b.rows();
    usize n = transpose_b ? b.rows() : b.cols();
    
    if (k_a != k_b) return std::unexpected(NexusError(ErrorCode::DimensionMismatch, "Inner dimensions don't match"));
    if (c.rows() != m || c.cols() != n) return std::unexpected(NexusError(ErrorCode::DimensionMismatch, "Output dimensions mismatch"));
    if (transpose_a || transpose_b) return std::unexpected(NexusError(ErrorCode::UnsupportedOperation, "Transpose handled externally via transpose_blocked"));
    
    if (beta != 1.0f && beta != 0.0f) {
        for (usize i = 0; i < m; ++i) {
            for (usize j = 0; j < n; ++j) c(i, j) *= beta;
        }
    }
    
    gemm_blocked(a.data, b.data, c.data, m, n, k_a, k_a, n, n);
    
    if (alpha != 1.0f) {
        for (usize i = 0; i < m * n; ++i) c.data[i] *= alpha;
    }
    
    return {};
}

}

}
