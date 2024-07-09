add_subdirectory(lib)

add_executable(MandelbrotSingle ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(MandelbrotSingle PRIVATE nanogui)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
