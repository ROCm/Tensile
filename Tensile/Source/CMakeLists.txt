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
project(TensileClient)
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR} )

###############################################################################
# Common Options
option( Tensile_ENABLE_HALF "Enable half precision data types" OFF)
option( Tensile_CLIENT_BENCMARK "ON=BenchmarkClient; OFF=LibraryClient" ON)
option( Tensile_MERGE_FILES "Merge kernels and solutions files" OFF)
set(Tensile_RUNTIME_LANGUAGE HIP CACHE STRING "Which runtime language to use")
set_property( CACHE Tensile_RUNTIME_LANGUAGE PROPERTY STRINGS HIP OCL )

###############################################################################
# Benchmark Client
set(ClientName "client")
if(Tensile_CLIENT_BENCHMARK)
  message(STATUS "Making BenchmarkClient")
  include(${CMAKE_SOURCE_DIR}/Generated.cmake)
  add_executable( ${ClientName}
    ${TensileClient_SOURCE}
    ${TensileClient_SOLUTIONS}
    ${TensileClient_KERNELS} )
  target_compile_definitions( ${ClientName} PUBLIC 
    -DTensile_CLIENT_BENCHMARK=1
    -DTensile_CLIENT_LIBRARY=0 )
  foreach( target ${Tensile_ISA} )
    target_link_libraries( ${ClientName} PRIVATE --amdgpu-target=${target} )
  endforeach()

else()
###############################################################################
# Library Client
  message(STATUS "Making LibraryClient")
  option( Tensile_SHORT_FILE_NAMES "Use short file names (for MSVC)" OFF)
  option( Tensile_LIBRARY_PRINT_DEBUG "TensileLib to print debug info" OFF)
  add_executable( ${ClientName}
    Client.cpp
    MathTemplates.cpp
    DeviceStats.h
    )
  target_compile_definitions( ${ClientName} PUBLIC 
    -DTensile_CLIENT_BENCHMARK=0
    -DTensile_CLIENT_LIBRARY=1 )
endif()

###############################################################################
# Visual Studio or HIPCC
if(MSVC)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" ) # parallel build
endif()

if (UNIX)
  # the Tensile library doesn't require c++11, but client/ReferenceCPU does
  # with addition of bfloat16 library requires c++14
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14"  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-deprecated-declarations" )
endif()

if(Tensile_MERGE_FILES)
  #include_directories(.)
else()
  include_directories( Kernels Solutions )
  include_directories(.)
endif()

###############################################################################
# find and use device libraries
if( Tensile_RUNTIME_LANGUAGE MATCHES "OCL")
  find_package(OpenCL "1.2" REQUIRED)
  target_link_libraries( ${ClientName} PUBLIC ${OPENCL_LIBRARIES} )
  target_compile_definitions( ${ClientName} PUBLIC 
    -DTensile_RUNTIME_LANGUAGE_OCL=1
    -DTensile_RUNTIME_LANGUAGE_HIP=0 )
  target_include_directories( ${ClientName} SYSTEM
    PUBLIC  ${OPENCL_INCLUDE_DIRS} ) 
elseif( Tensile_RUNTIME_LANGUAGE MATCHES "HIP")
  find_package( HIP REQUIRED )
  set (CMAKE_CXX_COMPILER ${HIPCC})
  target_include_directories( ${ClientName} SYSTEM
    PUBLIC  ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS} )
  target_link_libraries( ${ClientName} PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( ${ClientName} PUBLIC 
    -DTensile_RUNTIME_LANGUAGE_OCL=0
    -DTensile_RUNTIME_LANGUAGE_HIP=1
    )
  if( Tensile_ENABLE_HALF )
    target_compile_definitions( ${ClientName} PUBLIC  -DTensile_ENABLE_HALF )
  endif()
endif()


###############################################################################
# Create Tensile Library
if(NOT Tensile_CLIENT_BENCHMARK)
  include(${CMAKE_SOURCE_DIR}/TensileConfig.cmake)
  TensileCreateLibrary(
    ${Tensile_LOGIC_PATH}           # path
    ${Tensile_RUNTIME_LANGUAGE}     # OCL or HIP
    ${Tensile_MERGE_FILES}          # ON or OFF
    ${Tensile_SHORT_FILE_NAMES}     # ON or OFF
    ${Tensile_LIBRARY_PRINT_DEBUG}  # ON or OFF
    Tensile_ROOT ${Tensile_ROOT}
    )
  target_link_libraries( ${ClientName} PUBLIC Tensile )
endif()


