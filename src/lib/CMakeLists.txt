#add_subdirectory(nanogui)
set(LODEPNG_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/lodepng} PARENT_SCOPE)
add_library(lodepng lodepng/lodepng.cpp)
