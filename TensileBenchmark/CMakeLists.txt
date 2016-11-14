################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 2.8.12)

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )

set( TensileBenchmark_NAME TensileBenchmark )

get_filename_component(Tensile_DIR ${CMAKE_SOURCE_DIR} PATH)

set( CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${Tensile_DIR}/cmake )

############################################################
# require C++11
if(MSVC)
  # object-level build parallelism for VS, not just target-level
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
  set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
endif()
add_definitions( "-std=c++11" )

include(EnableWarnings)

# OpenCL 1.2
if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  find_package(OpenCL "1.2" REQUIRED)
  message(STATUS "${Tensile_BACKEND} backend selected." )
# HIP
elseif( Tensile_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  set (CMAKE_CXX_COMPILER ${HIPCC})
  message(STATUS "${Tensile_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
endif( )


# source files
# Sets TensileBenchmark_KernelFiles_GENERATED_DYNAMIC
include(${TensileBenchmark_DIR_GENERATED}/Kernels/TensileKernels.cmake)
# Sets TensileBenchmark_SolutionFiles_GENERATED_DYNAMIC
include(${TensileBenchmark_DIR_GENERATED}/Solutions/TensileSolutions.cmake)
# Sets TensileBenchmark_SRC_GENERATED_DYNAMIC
include(${TensileBenchmark_DIR_GENERATED}/Other/TensileBenchmark.cmake)

# Glob TensileLib source files
file(GLOB TensileLib_SRC
    ${Tensile_DIR}/TensileLib/src/*.cpp
)

function( convert_to_linux_path return_list_of_files )
  set(new_list)
  foreach( file ${ARGN} )
    file(TO_CMAKE_PATH ${file} result)
    list(APPEND new_list ${result})
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

convert_to_linux_path(TensileBenchmark_SRC
  ${TensileBenchmark_KernelFiles_GENERATED_DYNAMIC}
  ${TensileBenchmark_SolutionFiles_GENERATED_DYNAMIC}
  ${TensileBenchmark_SRC_GENERATED_DYNAMIC}
  ${TensileBenchmark_DIR_GENERATED}/Other/TensileSolutionCandidates.cpp
)

# benchmark executable
add_executable( ${TensileBenchmark_NAME}
    TensileBenchmark.cpp
    ${TensileLib_SRC}
    ${TensileBenchmark_SRC}
)
set_target_properties( ${TensileBenchmark_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" 
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_BINARY_DIR}/bin" )

##################################################
if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( ${TensileBenchmark_NAME} PRIVATE opencl )
  target_compile_definitions( ${TensileBenchmark_NAME} PUBLIC -DTensile_BACKEND_OPENCL12=1 -DTensile_BACKEND_HIP=0 )
elseif( Tensile_BACKEND MATCHES "HIP")
  target_include_directories( ${TensileBenchmark_NAME} SYSTEM
    PUBLIC  ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS}
  )
#  target_link_libraries( ${TensileBenchmark_NAME} PRIVATE ${HSA_LIBRARIES} )
  target_link_libraries( ${TensileBenchmark_NAME} PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( ${TensileBenchmark_NAME} PUBLIC -DTensile_BACKEND_OPENCL12=0 -DTensile_BACKEND_HIP=1 )
endif()

# include Tensile headers
target_include_directories( ${TensileBenchmark_NAME}
  PUBLIC  $<BUILD_INTERFACE:${Tensile_DIR}/TensileLib/include>
          $<BUILD_INTERFACE:${Tensile_DIR}/TensileLib/src>
	  $<BUILD_INTERFACE:${TensileBenchmark_DIR_GENERATED}>
	  $<BUILD_INTERFACE:${TensileBenchmark_DIR_GENERATED}/Kernels>
	  $<BUILD_INTERFACE:${TensileBenchmark_DIR_GENERATED}/Solutions>
	  $<BUILD_INTERFACE:${TensileBenchmark_DIR_GENERATED}/Other>
)

############################################################
# TensileBenchmark - generates ProblemSolutionMap.xml

target_compile_definitions( ${TensileBenchmark_NAME} PRIVATE -DTensile_SOLVER_ENABLED=0 -DTensile_LOGGER_ENABLED=1 )
if (CMAKE_CXX_COMPILER MATCHES ".*hipcc")
    # hipcc is a pearl script, so it requires a lot of extra escaping
    #target_compile_definitions(${TensileBenchmark_NAME} PUBLIC -DTensile_DIR_PROBLEMS=\\\"${TensileBenchmark_DIR_GENERATED}\\\")
    # This does not need to be defined but will fail to compile when not
    target_compile_definitions(${TensileBenchmark_NAME} PUBLIC -DTensile_DIR_SOLUTIONS=\\\"${TensileBenchmark_DIR_SOLUTIONS}\\\")
else()
    file( TO_CMAKE_PATH ${TensileBenchmark_DIR_SOLUTIONS} CBDS_PATH)
    message(STATUS "TensileBenchmark output dir: ${CBDS_PATH}")
    #target_compile_definitions(${TensileBenchmark_NAME} PUBLIC -DTensile_DIR_PROBLEMS="${CBDS_PATH}")
    # This does not need to be defined but will fail to compile when not
    target_compile_definitions(${TensileBenchmark_NAME} PUBLIC -DTensile_DIR_SOLUTIONS="${CBDS_PATH}")
endif()
