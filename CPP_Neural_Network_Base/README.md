# NEXUS - C++ Neural Network Engine Base

A robust, modular deep learning framework implemented using standard Modern C++ paradigms. This version serves as the foundational "Control Group" for the Nexus project, establishing a baseline for performance, accuracy, and architectural comparison.

* **Architecture:** C++20 | Standard OOP | Polymorphic Layers
* **Memory:** Standard Heap Allocation | OS-Managed Thread Safety
* **Performance Profile:**
    * **Logic:** Mathematically identical to the Optimized version.
    * **Execution:** Synchronous, blocking execution via standard OS primitives.
    * **Safety:** Utilizes `std::mutex` for guaranteed data integrity across threads.
    * **Accuracy:** Serves as the high-precision ground truth for SIMD validation.

---

## 🏗️ Architectural Highlights

* **Standard OOP Design:** Utilizes a `Layer` base class with `virtual` functions for `forward()` and `backward()` passes. This provides maximum code flexibility and ease of extending the framework with new layer types.
* **OS-Level Thread Safety:** Implements thread synchronization using `std::mutex` and `std::lock_guard`. While introducing context-switching overhead, it ensures a safe, race-free environment for concurrent training.
* **Dynamic Memory Management:** Leverages standard heap-allocated containers (`std::vector`) and smart pointers. This prioritizes memory safety and ease of use over raw allocation speed.
* **Scalar Mathematics:** Matrix operations are implemented using standard nested-loop algorithms. This avoids hardware-specific intrinsic complexity, ensuring the code remains highly portable across any CPU architecture.

## 🚀 Quick Start (Base Engine)

The Base Engine is designed for portability and ease of compilation. It does not require specific hardware extensions like AVX-512, making it ideal for standard development environments.

```bash
# Clone the repository
git clone <https://github.com/aanalpatil24/CPP-Deep-Learning-Neural-Network-Framework.git>
cd CPP-Deep-Learning-Neural-Network-Framework/Cpp_Neural_Network_Base

# Run the build script to compile the baseline benchmarks and tests
chmod +x scripts/build.sh
./scripts/build.sh
