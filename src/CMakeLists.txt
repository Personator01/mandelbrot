set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_subdirectory(lib)


find_package(SDL2 CONFIG COMPONENTS SDL2)
find_package(SDL2 CONFIG COMPONENTS SDL2main)
if (NOT ${SDL2_FOUND})
	if (EXISTS lib/sdl)
		add_subdirectory(lib/sdl)
	else()
		message(FATAL_ERROR "library sdl not found globally and not in src/lib. Download sdl and place it in lib/sdl.")
	endif()
endif()

macro(linksdl targetname)
	if (TARGET SDL2::SDL2main)
		target_link_libraries(${targetname} PRIVATE SDL2::SDL2main)
	endif()
	target_link_libraries(${targetname} PRIVATE SDL2::SDL2)
endmacro()

add_executable(MandelbrotSingle 
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/mb_single.cpp)

linksdl(MandelbrotSingle)
target_link_libraries(MandelbrotSingle PRIVATE nanogui)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
