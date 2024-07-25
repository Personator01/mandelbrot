cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

project(mandelbrot LANGUAGES CXX CUDA)

add_subdirectory(src)

