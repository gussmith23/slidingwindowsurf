cmake_minimum_required(VERSION 2.8)
project(slidingwindowsurf)
find_package( OpenCV REQUIRED )
add_library( slidingwindowsurf src/slidingwindowsurf.cpp )
target_link_libraries( slidingwindowsurf ${OpenCV_LIBS} )
include_directories("include/")

# catch
include_directories("dependency/catch")

# test
add_executable(slidingwindowsurftest "test/slidingwindowsurftest.cpp")
target_link_libraries(slidingwindowsurftest slidingwindowsurf)
target_link_libraries( slidingwindowsurf ${OpenCV_LIBS} )
add_test(slidingwindowsurftest slidingwindowsurftest)
add_custom_command(
	TARGET slidingwindowsurftest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/test/resources/
                ${CMAKE_CURRENT_BINARY_DIR}/resources/)