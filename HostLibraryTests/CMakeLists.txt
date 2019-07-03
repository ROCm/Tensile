################################################################################
# Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

cmake_minimum_required(VERSION 3.5)

project(TensileHostLibraryTest)

option(TENSILE_USE_HIP       "Use the Hip runtime." ON)
option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." ON)
option(TENSILE_USE_OPENMP    "Use LLVM for parsing config files." ON)
option(TENSILE_STATIC_ONLY   "Disable exposing Tensile symbols in a shared library." ON)
option(TENSILE_DISABLE_CTEST "Disable CTest integration." OFF)

if(TENSILE_USE_HIP)
    set(TENSILE_COMPONENTS Hip)
    if(TENSILE_USE_LLVM)
        set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} Client)
endif()
if(TENSILE_USE_LLVM)
    set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} LLVM)
endif()
if(TENSILE_USE_OPENMP)
    set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} OpenMP)
endif()
if(TENSILE_STATIC_ONLY)
    set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} STATIC_ONLY)
endif()

find_package(Tensile 4.9.9 EXACT REQUIRED ${TENSILE_COMPONENTS})

if(NOT TENSILE_DISABLE_CTEST)
    enable_testing()

    if(CMAKE_VERSION VERSION_GREATER 3.9)
        include(GoogleTest)
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." ON)
    else()
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." OFF)
    endif()
endif()

set(INSTALL_GTEST OFF CACHE BOOL "Install GTest")
add_subdirectory(googletest)

find_package(Boost COMPONENTS filesystem)

add_subdirectory(configs)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")
file(COPY ${SOLUTION_LIBRARY_FILES} DESTINATION "${TEST_DATA_DIR}")

include_directories("." ${GTEST_SOURCE_DIR}/include)
add_subdirectory(testlib)

set(test_sources ${test_sources}
    test.cpp

    ContractionProblem_test.cpp
    ContractionSelectionLibrary_test.cpp
    EmbeddedData_test.cpp
    KernelArguments_test.cpp
    ProjectedPerformance_test.cpp
    TensorDescriptor_test.cpp
    TestData_test.cpp
    Utils_test.cpp
)

if(TENSILE_USE_LLVM)
    set(test_sources ${test_sources}
        ContractionLibraryLoading_test.cpp
        llvm/LLVMYAMLContraction_test.cpp
        llvm/LibraryPerformance_test.cpp
        )
endif()

if(TENSILE_USE_HIP)
    add_subdirectory(hip)
endif()

message(STATUS "Test sources: ${test_sources}")
add_executable(TensileTests ${test_sources})

if(GTEST_INTEGRATION)
    gtest_discover_tests(TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
else()
    add_test(NAME TensileHostLibraryTests COMMAND TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

target_link_libraries(TensileTests gtest Tensile TensileTestLib Boost::filesystem)

if(HIP_FOUND)
    target_link_libraries(TensileTests ${HIP_TEST_LIBRARIES})
    target_link_libraries(TensileTests "--amdgpu-target=gfx900")
endif()

