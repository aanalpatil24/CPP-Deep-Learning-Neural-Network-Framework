#include "../include/nexus/math/gemm.hpp"
#include "../include/nexus/core/tensor.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace nexus;

// Textbook O(N^3) matrix multiplication for Ground Truth
void naive_gemm(const f32* A, const f32* B, f32* C, usize M, usize N, usize K) {
    for (usize i = 0; i < M; ++i) {
        for (usize j = 0; j < N; ++j) {
            f32 sum = 0.0f;
            for (usize p = 0; p < K; ++p) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    std::cout << "[TEST] Validating AVX-512 GEMM against Scalar Baseline...\n";
    
    constexpr usize M = 64, N = 64, K = 64;
    alignas(64) f32 A[M * K];
    alignas(64) f32 B[K * N];
    alignas(64) f32 C_expected[M * N] = {0};
    alignas(64) f32 C_actual[M * N] = {0};

    // Populate with deterministic pseudo-random data
    for (usize i = 0; i < M * K; ++i) A[i] = static_cast<f32>(i % 17) * 0.1f;
    for (usize i = 0; i < K * N; ++i) B[i] = static_cast<f32>(i % 13) * 0.1f;

    naive_gemm(A, B, C_expected, M, N, K);
    gemm::gemm_blocked(A, B, C_actual, M, N, K, K, N, N);

    f32 max_error = 0.0f;
    for (usize i = 0; i < M * N; ++i) {
        f32 diff = std::abs(C_expected[i] - C_actual[i]);
        max_error = std::max(max_error, diff);
        if (diff > 1e-4f) {
            std::cerr << "FAIL: Mismatch at index " << i << "\n";
            return 1;
        }
    }

    std::cout << "[PASS] GEMM output mathematically verified. Max error: " << max_error << "\n";
    return 0;
}