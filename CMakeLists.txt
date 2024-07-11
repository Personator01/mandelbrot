cmake_minimum_required(VERSION 3.10)

project(mandelbrot)

add_subdirectory(src)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
