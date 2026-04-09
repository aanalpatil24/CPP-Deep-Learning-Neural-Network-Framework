#pragma once
#include "../config.hpp"
#include <mutex>
#include <atomic>
#include <memory>

namespace nexus {

// LOCK-FREE MEMORY ARENA
// In low-latency systems, dynamic heap allocation (malloc/new) is forbidden during
// the hot path due to OS context switching. This Arena pre-allocates a massive 
// contiguous block and hands out pointers lock-free.

class alignas(NEXUS_CACHE_LINE_SIZE) MemoryArena {
public:
    struct Config {
        usize initial_capacity = 16 * 1024 * 1024;  // 16MB default
        bool use_huge_pages = false;                 
    };
    
    struct Stats {
        usize capacity;
        usize used;
        usize peak;
        usize allocation_count;
    };

private:
    // Pointers are aligned to cache lines to prevent false-sharing across threads
    alignas(NEXUS_CACHE_LINE_SIZE) byte* base_ = nullptr;
    alignas(NEXUS_CACHE_LINE_SIZE) std::atomic<byte*> current_{nullptr};
    byte* end_ = nullptr;
    
    usize capacity_ = 0;
    std::atomic<usize> peak_used_{0};
    std::atomic<usize> alloc_count_{0};
    bool use_huge_pages_ = false;

public:
     
    // Convenience constructor for simple size allocations
    explicit MemoryArena(usize initial_capacity) {
        use_huge_pages_ = false;
        grow(initial_capacity);
    }

    // Advanced constructor for explicitly requesting OS huge pages
    explicit MemoryArena(const Config& config = {}) {
        use_huge_pages_ = config.use_huge_pages;
        grow(config.initial_capacity);
    }
    
    ~MemoryArena() {
        release();
    }
    
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    // The heart of the Lock-Free Allocator
    [[nodiscard]] void* allocate(usize size, usize alignment = NEXUS_SIMD_ALIGNMENT) {
        byte* old_curr = current_.load(std::memory_order_acquire);
        byte* aligned_ptr;
        byte* new_curr;

        // True Lock-Free Compare-And-Swap (CAS) Loop
        // This allows multiple threads to allocate memory simultaneously without kernel mutexes.
        do {
            // 1. Calculate the required alignment for the current pointer
            // AVX-512 requires strict 64-byte alignment, otherwise it segfaults.
            uintptr_t addr = reinterpret_cast<uintptr_t>(old_curr);
            uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
            aligned_ptr = reinterpret_cast<byte*>(aligned_addr);
            
            // 2. Determine where the new pointer ends
            new_curr = aligned_ptr + size;

            if (new_curr > end_) {
                throw std::bad_alloc(); // Arena exhausted
            }
            
            // 3. Atomically swap. If another thread allocated memory while we calculated, 
            // old_curr is automatically updated, and the loop retries instantly.
        } while (!current_.compare_exchange_weak(old_curr, new_curr, 
                                                 std::memory_order_acq_rel, 
                                                 std::memory_order_acquire));

        alloc_count_.fetch_add(1, std::memory_order_relaxed);
        usize current_used = new_curr - base_;
        
        // Lock-free peak tracking update
        usize peak = peak_used_.load(std::memory_order_relaxed);
        while (current_used > peak && 
               !peak_used_.compare_exchange_weak(peak, current_used, std::memory_order_relaxed)) {}

        return aligned_ptr;
    }
    
    // Helper to allocate and run object constructors in place
    template<typename T, typename... Args>
    [[nodiscard]] T* construct(Args&&... args) {
        void* mem = allocate(sizeof(T), std::max(alignof(T), NEXUS_SIMD_ALIGNMENT));
        if (!mem) return nullptr;
        return new (mem) T(std::forward<Args>(args)...);
    }
    
    // Helper to allocate raw C-style arrays
    template<typename T>
    [[nodiscard]] T* allocate_array(usize count) {
        return static_cast<T*>(allocate(count * sizeof(T), std::max(alignof(T), NEXUS_SIMD_ALIGNMENT)));
    }

    using Marker = byte*;
    
    // Grabs a snapshot of the current memory head
    [[nodiscard]] Marker mark() const noexcept {
        return current_.load(std::memory_order_acquire);
    }
    
    // Instantly frees all memory allocated after the marker. O(1) complexity.
    void reset_to(Marker marker) noexcept {
        current_.store(marker, std::memory_order_release);
    }
    
    void reset() noexcept {
        current_.store(base_, std::memory_order_release);
    }

    [[nodiscard]] bool contains(const void* ptr) const noexcept {
        auto addr = reinterpret_cast<const byte*>(ptr);
        return (addr >= base_ && addr < end_);
    }
    
    [[nodiscard]] Stats stats() const noexcept {
        Stats s;
        s.capacity = capacity_;
        s.used = used();
        s.peak = peak_used_.load(std::memory_order_relaxed);
        s.allocation_count = alloc_count_.load(std::memory_order_relaxed);
        return s;
    }
    
    [[nodiscard]] usize used() const noexcept {
        return current_.load(std::memory_order_acquire) - base_;
    }

private:
    void grow(usize new_capacity) {
        capacity_ = (new_capacity + 2097151) & ~2097151; // 2MB align to encourage Transparent Huge Pages (THP)
        usize align = use_huge_pages_ ? 2097152 : NEXUS_SIMD_ALIGNMENT;
        
        // Cross-platform aligned allocation
        #if defined(_WIN32) || defined(_MSC_VER)
            base_ = static_cast<byte*>(_aligned_malloc(capacity_, align));
        #else
            base_ = static_cast<byte*>(std::aligned_alloc(align, capacity_));
        #endif
        
        if (!base_) throw std::bad_alloc();
        
        // Asynchronous Memory Pre-faulting
        // We instruct the kernel to map physical pages to virtual pages immediately.
        // If we skip this, the first inference pass will incur massive page-fault latency spikes.
        #if defined(__linux__)
            madvise(base_, capacity_, MADV_WILLNEED);
        #else
            // Fallback: Manually touch one byte per page to force OS allocation
            for(usize i = 0; i < capacity_; i += 4096) base_[i] = std::byte{0};
        #endif
        
        current_.store(base_, std::memory_order_release);
        end_ = base_ + capacity_;
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
    }
};

// RAII scope guard to automatically release scratch memory when a function exits
class ArenaScope {
    MemoryArena& arena_;
    MemoryArena::Marker marker_;
    bool released_ = false;
    
public:
    explicit ArenaScope(MemoryArena& arena) : arena_(arena), marker_(arena.mark()) {}
    ~ArenaScope() { if (!released_) arena_.reset_to(marker_); }
    void release() noexcept { released_ = true; }
};

}
