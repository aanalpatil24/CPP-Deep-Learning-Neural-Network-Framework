#pragma once
#include "../config.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

namespace nexus {

// PROFILING & BENCHMARKING
// High-resolution hardware timing to validate our sub-microsecond latency claims.

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::nanoseconds;
    
    explicit Timer(std::string name = "") : name_(std::move(name)), running_(false) {}
    
    void start() { start_ = Clock::now(); running_ = true; }
    void stop() {
        if (!running_) return;
        end_ = Clock::now();
        running_ = false;
        total_ += std::chrono::duration_cast<Duration>(end_ - start_);
        ++count_;
    }
    void reset() { total_ = Duration::zero(); count_ = 0; running_ = false; }
    
    f32 elapsed_ms() const { return std::chrono::duration<f32, std::milli>(total_).count(); }
    f32 elapsed_us() const { return std::chrono::duration<f32, std::micro>(total_).count(); }
    f32 elapsed_ns() const { return static_cast<f32>(total_.count()); }
    f32 average_ns() const { return count_ > 0 ? elapsed_ns() / static_cast<f32>(count_) : 0.0f; }
    
    void report() const {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[Timer] " << name_ << ": ";
        if (count_ == 1) std::cout << elapsed_us() << " us\n";
        else std::cout << elapsed_ms() << " ms total, " << average_ns() / 1000.0f << " us avg (" << count_ << " runs)\n";
    }
    ~Timer() { if (count_ > 0) report(); }

private:
    std::string name_;
    TimePoint start_;
    TimePoint end_;
    Duration total_ = Duration::zero();
    usize count_ = 0;
    bool running_ = false;
};

class TimerScope {
    Timer& timer_;
public:
    explicit TimerScope(Timer& t) : timer_(t) { timer_.start(); }
    ~TimerScope() { timer_.stop(); }
};

}
