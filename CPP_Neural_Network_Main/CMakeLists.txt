cmake_minimum_required(VERSION 3.20)
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
