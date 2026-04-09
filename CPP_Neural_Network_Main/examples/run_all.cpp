#include "../include/nexus/config.hpp"
#include "../include/nexus/core/arena.hpp"
#include "../include/nexus/core/tensor.hpp"
#include "../include/nexus/math/gemm.hpp"
#include "../include/nexus/nn/network.hpp"
#include "../include/nexus/utils/timer.hpp"
#include <random>

using namespace nexus;

// BENCHMARKS & DEMONSTRATIONS

void benchmark_gemm() {
    std::cout << "\n=== GEMM Benchmark ===\n";
    constexpr usize M = 1024, N = 1024, K = 1024;
    
    // Explicit 64-byte aligned allocations are required for _mm512_load_ps instructions.
    #if defined(_WIN32) || defined(_MSC_VER)
        f32* a = static_cast<f32*>(_aligned_malloc(M * K * sizeof(f32), NEXUS_SIMD_ALIGNMENT));
        f32* b = static_cast<f32*>(_aligned_malloc(K * N * sizeof(f32), NEXUS_SIMD_ALIGNMENT));
        f32* c = static_cast<f32*>(_aligned_malloc(M * N * sizeof(f32), NEXUS_SIMD_ALIGNMENT));
    #else
        f32* a = static_cast<f32*>(std::aligned_alloc(NEXUS_SIMD_ALIGNMENT, M * K * sizeof(f32)));
        f32* b = static_cast<f32*>(std::aligned_alloc(NEXUS_SIMD_ALIGNMENT, K * N * sizeof(f32)));
        f32* c = static_cast<f32*>(std::aligned_alloc(NEXUS_SIMD_ALIGNMENT, M * N * sizeof(f32)));
    #endif

    std::mt19937 gen(42);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    for (usize i = 0; i < M * K; ++i) a[i] = dist(gen);
    for (usize i = 0; i < K * N; ++i) b[i] = dist(gen);
    
    gemm::gemm_blocked(a, b, c, M, N, K, K, N, N);
    
    Timer timer("GEMM 1024x1024x1024");
    constexpr usize iterations = 100;
    for (usize i = 0; i < iterations; ++i) {
        TimerScope scope(timer);
        std::memset(c, 0, M * N * sizeof(f32));
        gemm::gemm_blocked(a, b, c, M, N, K, K, N, N);
        
        // Compiler barriers prevent compilers from dead-code eliminating the loop.
        #if defined(_MSC_VER)
            _ReadWriteBarrier();
        #else
            asm volatile("" ::: "memory");
        #endif
    }
    
    f32 seconds = timer.elapsed_ms() / 1000.0f;
    f32 flops = 2.0f * M * N * K * iterations;
    f32 gflops = (flops / seconds) / 1e9f;
    std::cout << "Throughput: " << gflops << " GFLOPS\n";
    
    #if defined(_WIN32) || defined(_MSC_VER)
        _aligned_free(a); _aligned_free(b); _aligned_free(c);
    #else
        std::free(a); std::free(b); std::free(c);
    #endif
}

void demonstrate_mnist_network() {
    std::cout << "\n=== MNIST Network Demo ===\n";
    MemoryArena arena(50 * 1024 * 1024);
    
// AVX-512 padding constraint: To maintain maximum register saturation and avoid edge-handling
// overhead, the MNIST dimension (784) is padded to 800, and output (10) is padded to 16.
    constexpr usize BATCH = 128;
    StaticNeuralNetwork net(
        arena, BATCH,
        DenseLayer(arena, 800, 512, ActivationType::ReLU),
        DenseLayer(arena, 512, 256, ActivationType::ReLU),
        DenseLayer(arena, 256, 16, ActivationType::Softmax)
    );
    
    std::cout << net.summary() << "\n";
    
    f32* input_data = arena.allocate_array<f32>(BATCH * 800);
    f32* target_data = arena.allocate_array<f32>(BATCH * 16);
    TensorView input(input_data, BATCH, 800);
    TensorView target(target_data, BATCH, 16);
    
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 0.1f);
    for (usize i = 0; i < BATCH * 800; ++i) input_data[i] = dist(gen);
    for (usize i = 0; i < BATCH * 16; ++i) target_data[i] = (i % 16 == 7) ? 1.0f : 0.0f;
    
    net.forward(input);
    
    {
        Timer timer("Forward pass");
        for (usize i = 0; i < 1000; ++i) {
            TimerScope scope(timer);
        // Save arena snapshot to instantly free memory allocated during the inference loop
            auto marker = arena.mark(); 
            net.forward(input);
            arena.reset_to(marker);
            
            #if defined(_MSC_VER)
                _ReadWriteBarrier();
            #else
                asm volatile("" ::: "memory");
            #endif
        }
    }
    
    losses::CrossEntropyLoss loss_fn;
    {
        Timer timer("Training step (forward + backward + update)");
        for (usize i = 0; i < 1000; ++i) {
            TimerScope scope(timer);
            auto marker = arena.mark();
            net.train_step(input, target, loss_fn, 0.001f);
            arena.reset_to(marker);
            
            #if defined(_MSC_VER)
                _ReadWriteBarrier();
            #else
                asm volatile("" ::: "memory");
            #endif
        }
    }
    
    auto stats = arena.stats();
    std::cout << "\nArena Statistics:\n  Capacity: " << stats.capacity / (1024 * 1024) << " MB\n";
    std::cout << "  Used: " << stats.used / 1024 << " KB\n";
}

void demonstrate_memory_arena() {
    std::cout << "\n=== Memory Arena Demo ===\n";
    MemoryArena arena(1024 * 1024);
    
    auto* w1 = arena.allocate_array<f32>(1000 * 1000); 
    auto* b1 = arena.allocate_array<f32>(1000);
    auto* w2 = arena.allocate_array<f32>(1000 * 100);
    auto* b2 = arena.allocate_array<f32>(100);
    
    auto stats = arena.stats();
    std::cout << "After allocations: Used " << stats.used << " bytes\n";
    
    // Mark snapshot. Anything allocated after this point can be freed in O(1) time.
    auto marker = arena.mark();
    auto* temp1 = arena.allocate_array<f32>(5000);
    std::cout << "After temp allocations: " << arena.used() << " bytes\n";
    
    // Free the temp block instantly by shifting the atomic pointer back to the marker.
    arena.reset_to(marker);
    std::cout << "After reset (O(1)): " << arena.used() << " bytes\n";
    std::cout << "w1 still valid: " << (arena.contains(w1) ? "yes" : "no") << "\n";
}

// MAIN ENTRY POINT

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            NEXUS HIGH-PERFORMANCE ML ENGINE v2.1                 ║\n";
    std::cout << "║      Zero-Dependency | C++20 | AVX-512 | Lock-Free CAS           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Build Configuration:\n  Architecture: ";
    #ifdef NEXUS_ARCH_X86_64
    std::cout << "x86_64\n";
    #endif
    
    std::cout << "  SIMD: ";
    #ifdef NEXUS_HAS_AVX512
    std::cout << "AVX-512 (16-wide)\n";
    #elif defined(NEXUS_HAS_AVX2)
    std::cout << "AVX2 (8-wide)\n";
    #else
    std::cout << "Scalar\n";
    #endif
    
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
