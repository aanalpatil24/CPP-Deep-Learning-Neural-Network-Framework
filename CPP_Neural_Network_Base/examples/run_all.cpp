#include "../include/nexus/config.hpp"
#include "../include/nexus/core/arena.hpp"
#include "../include/nexus/core/tensor.hpp"
#include "../include/nexus/math/gemm.hpp"
#include "../include/nexus/nn/network.hpp"
#include "../include/nexus/utils/timer.hpp"
#include <random>

using namespace nexus;

// BENCHMARKS & DEMONSTRATIONS (benchmarks/ + examples/)

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


// MAIN ENTRY POINT

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
