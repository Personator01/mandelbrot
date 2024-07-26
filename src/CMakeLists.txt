set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_subdirectory(lib)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
include_directories(${LODEPNG_INCLUDE})

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

find_package(SDL2 CONFIG COMPONENTS SDL2)
find_package(SDL2 CONFIG COMPONENTS SDL2main)
if (NOT ${SDL2_FOUND})
	if (EXISTS lib/sdl)
		add_subdirectory(lib/sdl)
	else()
		message(FATAL_ERROR "library sdl not found globally and not in src/lib. Download sdl and place it in lib/sdl.")
	endif()
endif()

macro(linklibs targetname)
	if (TARGET SDL2::SDL2main)
		target_link_libraries(${targetname} PRIVATE SDL2::SDL2main)
	endif()
	target_link_libraries(${targetname} PRIVATE SDL2::SDL2)
	#target_link_libraries(${targetname} PRIVATE nanogui)
	target_link_libraries(${targetname} PRIVATE lodepng)
endmacro()

add_executable(MandelbrotSingle 
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/mb_single.cpp)

add_executable(MandelbrotCuda
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/mb_cuda.cu)

add_executable(MandelbrotViewerCuda
	${CMAKE_CURRENT_SOURCE_DIR}/mainglcu.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/render.cu
	)

set_target_properties(MandelbrotViewerCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

linklibs(MandelbrotSingle)
linklibs(MandelbrotCuda)

target_link_libraries(MandelbrotViewerCuda PRIVATE OpenGL::OpenGL OpenGL::GLU ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
