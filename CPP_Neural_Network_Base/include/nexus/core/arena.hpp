#pragma once
#include "../config.hpp"
#include <mutex>
#include <atomic>
#include <memory>

namespace nexus {

// MEMORY ARENA (core/arena.cpp + include)
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

}
