
cmake_minimum_required(VERSION 3.5)

project(tensile_lib LANGUAGES CXX)

list( APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip )

find_package(Tensile REQUIRED)
find_package(hip)

tensile_add_library(tensile ${CMAKE_CURRENT_SOURCE_DIR}/hip)
target_link_libraries(tensile PRIVATE hip::device --amdgpu-target=gfx900)

add_library(tensile_lib main.cpp)
target_link_libraries(tensile_lib tensile)

