#!/bin/bash
set -e
mkdir -p build/bin
echo "Building Nexus Engine..."
g++ -std=c++20 -O3 -march=native -ffast-math -mavx512f -mavx512dq -I./include examples/run_all.cpp -o build/bin/run_all
echo "Build complete. Executing benchmarks..."
./build/bin/run_all
