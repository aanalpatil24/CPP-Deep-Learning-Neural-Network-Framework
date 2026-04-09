import os

def split_nexus_architecture(input_filename="nexus.hpp", out_dir="nexus_project"):
    if not os.path.exists(input_filename):
        print(f"[ERROR] '{input_filename}' not found. Please save your C++ code with this name.")
        return

    print(f"[*] Parsing monolithic engine: {input_filename}")
    
    # Pre-configure the headers with their required #includes to ensure cross-compilation works
    file_buffers = {
        "include/nexus/config.hpp": ["#pragma once\n\n"],
        "include/nexus/core/error.hpp": ["#pragma once\n#include \"../config.hpp\"\n#include <string>\n#include <source_location>\n\nnamespace nexus {\n\n"],
        "include/nexus/core/arena.hpp": ["#pragma once\n#include \"../config.hpp\"\n#include <mutex>\n#include <atomic>\n#include <memory>\n\nnamespace nexus {\n\n"],
        "include/nexus/core/tensor.hpp": ["#pragma once\n#include \"../config.hpp\"\n#include <array>\n#include <span>\n#include <stdexcept>\n\nnamespace nexus {\n\n"],
        "include/nexus/math/kernels.hpp": ["#pragma once\n#include \"../config.hpp\"\n#include <cmath>\n#include <algorithm>\n\nnamespace nexus {\n\n"],
        "include/nexus/math/gemm.hpp": ["#pragma once\n#include \"../config.hpp\"\n#include \"../core/tensor.hpp\"\n#include \"../core/error.hpp\"\n\nnamespace nexus {\n\n"],
        "include/nexus/nn/layers.hpp": ["#pragma once\n#include \"../config.hpp\"\n#include \"../core/arena.hpp\"\n#include \"../core/tensor.hpp\"\n#include \"../math/kernels.hpp\"\n#include \"../math/gemm.hpp\"\n\nnamespace nexus {\n\n"],
        "include/nexus/nn/losses.hpp": ["#pragma once\n#include \"../config.hpp\"\n#include \"../core/tensor.hpp\"\n#include <vector>\n\nnamespace nexus {\n\n"],
        "include/nexus/nn/network.hpp": ["#pragma once\n#include \"../config.hpp\"\n#include \"../core/arena.hpp\"\n#include \"../core/tensor.hpp\"\n#include \"layers.hpp\"\n#include \"losses.hpp\"\n#include <vector>\n#include <memory>\n#include <fstream>\n#include <sstream>\n\nnamespace nexus {\n\n"],
        "include/nexus/utils/timer.hpp": ["#pragma once\n#include \"../config.hpp\"\n#include <chrono>\n#include <iostream>\n#include <iomanip>\n#include <string>\n\nnamespace nexus {\n\n"],
        "examples/run_all.cpp": [
            "#include \"../include/nexus/config.hpp\"\n",
            "#include \"../include/nexus/core/arena.hpp\"\n",
            "#include \"../include/nexus/core/tensor.hpp\"\n",
            "#include \"../include/nexus/math/gemm.hpp\"\n",
            "#include \"../include/nexus/nn/network.hpp\"\n",
            "#include \"../include/nexus/utils/timer.hpp\"\n",
            "#include <random>\n\n",
            "using namespace nexus;\n\n"
        ]
    }

    # Map the C++ comments from the monolithic file to the new files
    route_map = {
        "COMPILER FEATURE DETECTION": "include/nexus/config.hpp",
        "ERROR HANDLING SYSTEM": "include/nexus/core/error.hpp",
        "MEMORY ARENA": "include/nexus/core/arena.hpp",
        "TENSOR VIEW": "include/nexus/core/tensor.hpp",
        "SIMD MATHEMATICS KERNELS": "include/nexus/math/kernels.hpp",
        "CACHE-BLOCKED GEMM": "include/nexus/math/gemm.hpp",
        "NEURAL NETWORK LAYERS": "include/nexus/nn/layers.hpp",
        "LOSS FUNCTIONS": "include/nexus/nn/losses.hpp",
        "NEURAL NETWORK EXECUTION ENGINE": "include/nexus/nn/network.hpp",
        "PROFILING & BENCHMARKING": "include/nexus/utils/timer.hpp",
        "BENCHMARKS & DEMONSTRATIONS": "examples/run_all.cpp"
    }

    with open(input_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    active_file = "include/nexus/config.hpp"
    skip_next_namespace = False

    for line in lines:
        stripped = line.strip()

        # Clean up global wrappers from the monolithic file
        if stripped == "namespace nexus {" and active_file == "include/nexus/config.hpp":
            file_buffers["include/nexus/config.hpp"].append("namespace nexus {\n")
            continue
        if stripped == "} // namespace nexus":
            continue
        if stripped == "#pragma once":
            continue
        if "COMPILATION INSTRUCTIONS" in stripped:
            break

        # Detect section boundaries and switch files
        for marker, target_file in route_map.items():
            if marker in stripped and stripped.startswith("//"):
                active_file = target_file
                print(f"    -> Routing [{marker}] to {target_file}")
                break

        file_buffers[active_file].append(line)

    # Close the namespaces securely
    for fname, buffer in file_buffers.items():
        if fname.endswith(".hpp"):
            buffer.append("\n} // namespace nexus\n")

    print(f"\n[*] Generating directory structure in ./{out_dir}/...")
    
    # Write C++ files to disk
    for fname, buffer in file_buffers.items():
        full_path = os.path.join(out_dir, fname)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.writelines(buffer)

    # Generate CMakeLists.txt
    cmake_content = """cmake_minimum_required(VERSION 3.20)
project(NexusEngine VERSION 2.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(MSVC)
    add_compile_options(/O2 /arch:AVX512 /Zc:__cplusplus)
else()
    add_compile_options(-O3 -march=native -ffast-math -mavx512f -mavx512dq)
endif()

add_executable(run_all examples/run_all.cpp)
target_include_directories(run_all PRIVATE include)
"""
    with open(os.path.join(out_dir, "CMakeLists.txt"), "w") as f:
        f.write(cmake_content)

    # Generate the Build/Test Bash Script
    os.makedirs(os.path.join(out_dir, "scripts"), exist_ok=True)
    build_sh_content = """#!/bin/bash
set -e
mkdir -p build/bin
echo "Building Nexus Engine..."
g++ -std=c++20 -O3 -march=native -ffast-math -mavx512f -mavx512dq -I./include examples/run_all.cpp -o build/bin/run_all
echo "Build complete. Executing benchmarks..."
./build/bin/run_all
"""
    with open(os.path.join(out_dir, "scripts", "build.sh"), "w", newline='\n') as f:
        f.write(build_sh_content)
    
    # Make bash script executable (Unix)
    if os.name == 'posix':
        os.chmod(os.path.join(out_dir, "scripts", "build.sh"), 0o755)

    print("[SUCCESS] Project successfully modularized! Open the 'nexus_project' folder.")

if __name__ == "__main__":
    split_nexus_architecture()