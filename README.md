# NEXUS - C++ Deep Learning Neural Network Framework

A zero-dependency, bare-metal deep learning execution framework engineered in **C++20** for sub-microsecond inference latency and deterministic memory behaviour. 
Designed from scratch for real-time quantitative risk analytics, this engine is built to bypass standard OS overhead and maximize silicon-level throughput.

## The Architectural Experiment
This repository contains two distinct implementations of the exact same mathematical engine to demonstrate the performance impact of hardware-aware C++ design.

### 1. `CPP_Neural_Network_Base/` (The Control)
A robust, modular implementation focusing on standard modern software engineering.

### Performance Features
* **OOP Architecture:** Uses a `Layer` base class with `virtual` functions for modularity.
* **Standard Concurrency:** Uses `std::mutex` for thread-safe memory access.
* **Heap Management:** Dynamic heap management using `std::vector` and standard heap allocations.
* **Scalar Math:** Standard nested-loop matrix multiplication (no vectorization).


### 2. `CPP_Neural_Network_Main/` (The Optimized Engine)
A performance-first implementation focusing on hardware-intrinsic engineering.

### Performance Features
* **Zero Virtual Dispatch:** Uses **C++20 Variadic Templates** to unroll the network at compile-time, eliminating vtable overhead.
* **Lock-Free Allocator:** Replaces mutexes with **Atomic Compare-And-Swap (CAS)** for wait-free memory allocation.
* **AVX-512 Vectorization:** Hand-coded SIMD kernels using `_mm512_fmadd_ps` for 16x parallel float processing.
* **Cache-Oblivious Tiling:** Matrix operations are blocked to fit **L1/L2 cache**, eliminating memory wait-states.
* **Zero-Allocation Hotpath:** Performs zero heap allocations during live inference to prevent latency spikes.


## Technical Stack

### **Core Language & Systems**
* **C++20 Standard:** Utilizes Concepts, Variadic Templates, and `std::span` for zero-overhead abstractions.
* **Lock-Free Concurrency:** Implemented via `std::atomic` to eliminate kernel-level context switching and thread contention.
* **Linear Arena Allocation:** Custom memory management providing $O(1)$ allocation/deallocation with strict **64-byte alignment** for SIMD requirements.

### **Hardware Intrinsics & Math**
* **Intel AVX-512:** Leverages 512-bit ZMM registers to process 16 single-precision floats per clock cycle.
* **Fused Multiply-Add (FMA):** Uses `_mm512_fmadd_ps` to compute dot products in a single hardware cycle ($A \times B + C$).
* **Cache-Oblivious Design:** Matrix operations are tile-blocked perfectly to L1/L2 cache geometries, minimizing TLB thrashing and memory latency.

### **DevOps & Tooling**
* **Build System:** **CMake 3.20+** configured for `-O3` and `-march=native`.
* **CI/CD:** **GitHub Actions** for automated performance regression and build verification.
* **Quality Control:** Strict adherence to **Clang-Tidy** and **Clang-Format** standards for enterprise-grade readability.


## Quick Start

Nexus is header-only and relies exclusively on free, open-source compilers. No heavy CMake files, no Docker containers, no cloud dependencies.
Simply include the headers and compile with GCC/Clang using C++20.

To compare the performance of the two implementations on your local hardware:

```bash
# Clone the repository
git clone <https://github.com/aanalpatil24/CPP-Deep-Learning-Neural-Network-Framework.git>
cd CPP-Deep-Learning-Neural-Network-Framework
cd CPP_Neural_Network

# Build and Run the Baseline (OOP) version
cd Cpp_Neural_Network_Base
chmod +x scripts/build.sh
./scripts/build.sh

# Build and Run the Optimized (SIMD) version
cd ../Cpp_Neural_Network_Main
chmod +x scripts/build.sh
./scripts/build.sh

```
