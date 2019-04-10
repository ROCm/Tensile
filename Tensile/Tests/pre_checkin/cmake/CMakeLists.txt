
cmake_minimum_required(VERSION 3.5)

project(tensile_lib LANGUAGES CXX)

find_package(Tensile REQUIRED)

# TensileCreateLibrary(
#   ${CMAKE_CURRENT_SOURCE_DIR}/hip_lite
#   HIP
#   ${Tensile_MERGE_FILES}
#   ${Tensile_SHORT_FILENAMES}
#   ${Tensile_PRINT_DEBUG}
#   Tensile_ROOT ${Tensile_ROOT}
# )

add_library(tensile_lib main.cpp)

