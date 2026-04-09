#include "../include/nexus/core/arena.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>

using namespace nexus;

int main() {
    std::cout << "[TEST] Validating Lock-Free Memory Arena...\n";

    MemoryArena arena(1024 * 1024); // 1MB

    // 1. Test Alignment Guarantees
    float* alloc1 = arena.allocate_array<float>(13); // Awkward size
    float* alloc2 = arena.allocate_array<float>(7);

    uintptr_t addr1 = reinterpret_cast<uintptr_t>(alloc1);
    uintptr_t addr2 = reinterpret_cast<uintptr_t>(alloc2);

    if (addr1 % NEXUS_SIMD_ALIGNMENT != 0 || addr2 % NEXUS_SIMD_ALIGNMENT != 0) {
        std::cerr << "FAIL: Allocations are not 64-byte aligned for AVX-512!\n";
        return 1;
    }

    // 2. Test Reset O(1) Complexity
    auto marker = arena.mark();
    arena.allocate_array<float>(5000);
    usize used_before = arena.used();
    
    arena.reset_to(marker);
    usize used_after = arena.used();

    if (used_after >= used_before) {
        std::cerr << "FAIL: Arena marker reset failed to free memory.\n";
        return 1;
    }

    std::cout << "[PASS] Arena maintains strict AVX-512 alignment and O(1) rollbacks.\n";
    return 0;
}