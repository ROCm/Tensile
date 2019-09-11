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

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_EXTENSIONS OFF )

option(TENSILE_USE_HIP       "Use the Hip runtime." ON)
option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." ON)
option(TENSILE_USE_OPENMP    "Enable OpenMP multithreading." ON)
option(TENSILE_STATIC_ONLY   "Disable exposing Tensile symbols in a shared library." ON)
option(TENSILE_DISABLE_CTEST "Disable CTest integration." OFF)

if(TENSILE_USE_HIP)
    set(TENSILE_COMPONENTS HIP)
    if(TENSILE_USE_LLVM)
        set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} Client)
    endif()
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

if(NOT Tensile_FOUND)
    find_package(Tensile 4.12.0 EXACT REQUIRED ${TENSILE_COMPONENTS} PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../Tensile")
endif()

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

#if(TENSILE_USE_HIP AND NOT HIP_DIR)
if(TENSILE_USE_HIP)
    find_package(HIP REQUIRED CONFIG PATHS /opt/rocm)
endif()

if(TENSILE_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

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
    find_package(LLVM 6.0 REQUIRED CONFIG)

    set(test_sources ${test_sources}
        ContractionLibraryLoading_test.cpp
        llvm/LLVMYAMLContraction_test.cpp
        llvm/LibraryPerformance_test.cpp
        )
endif()

if(TENSILE_USE_HIP)
    add_subdirectory(hip)
endif()

add_executable(TensileTests ${test_sources})

if(NOT TENSILE_DISABLE_CTEST)
    if(GTEST_INTEGRATION)
        gtest_discover_tests(TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    else()
        add_test(NAME TensileHostLibraryTests COMMAND TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
endif()

target_link_libraries(TensileTests PUBLIC gtest TensileHost TensileTestLib Boost::filesystem)

if(TENSILE_USE_LLVM)
    target_link_libraries(TensileTests PUBLIC LLVMObjectYAML)
    target_include_directories(TensileTests PUBLIC ${LLVM_INCLUDE_DIRS})
endif()

if(TENSILE_USE_HIP)
    target_link_libraries(TensileTests PUBLIC ${HIP_TEST_LIBRARIES})
    #target_link_libraries(TensileTests PRIVATE "--amdgpu-target=gfx900")
endif()

if(TENSILE_USE_OPENMP)
    target_link_libraries(TensileTests PRIVATE "${OpenMP_EXE_LINKER_FLAGS}")
endif()

