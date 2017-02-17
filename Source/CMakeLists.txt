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
option( Tensile_CLIENT_BENCMARK "ON=BenchmarkClient; OFF=LibraryClient" ON)
option( Tensile_MERGE_FILES "Merge kernels and solutions files" OFF)
set(Tensile_BACKEND HIP CACHE STRING "Which backend to use")
set_property( CACHE Tensile_BACKEND PROPERTY STRINGS HIP OCL )

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

else()
###############################################################################
# Library Client
  message(STATUS "Making LibraryClient")
  option( Tensile_SHORT_FILE_NAMES "Use short file names (for MSVC)" OFF)
  option( Tensile_LIBRARY_PRINT_DEBUG "TensileLib to print debug info" OFF)
  add_executable( ${ClientName}
    Client.cpp
    MathTemplates.cpp
    #Tools.cpp
    )
  target_compile_definitions( ${ClientName} PUBLIC 
    -DTensile_CLIENT_BENCHMARK=0
    -DTensile_CLIENT_LIBRARY=1 )
endif()

###############################################################################
# Visual Studio or not
if(MSVC)
  # object-level build parallelism for VS, not just target-level
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
  #set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
  #add_definitions( "-std=c++11" )
endif()

if(Tensile_MERGE_FILES)
  #include_directories(.)
else()
  include_directories( Kernels Solutions )
  include_directories(.)
endif()

###############################################################################
# find and use device libraries
if( Tensile_BACKEND MATCHES "OCL")
  find_package(OpenCL "1.2" REQUIRED)
  target_link_libraries( ${ClientName} PUBLIC ${OPENCL_LIBRARIES} )
  target_compile_definitions( ${ClientName} PUBLIC 
    -DTensile_BACKEND_OCL=1
    -DTensile_BACKEND_HIP=0 )
  target_include_directories( ${ClientName} SYSTEM
    PUBLIC  ${OPENCL_INCLUDE_DIRS} ) 
elseif( Tensile_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  set (CMAKE_CXX_COMPILER ${HIPCC})
  target_include_directories( ${ClientName} SYSTEM
    PUBLIC  ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS} )
  target_link_libraries( ${ClientName} PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( ${ClientName} PUBLIC 
    -DTensile_BACKEND_OCL=0
    -DTensile_BACKEND_HIP=1 )
endif()


###############################################################################
# Create Tensile Library
if(NOT Tensile_CLIENT_BENCHMARK)
  #include(${CMAKE_SOURCE_DIR}/TensileCreateLibrary.cmake)
  find_package(Tensile)
  TensileCreateLibrary(
    ${Tensile_LOGIC_PATH}           # path
    ${Tensile_BACKEND}              # OCL or HIP
    ${Tensile_MERGE_FILES}          # ON or OFF
    ${Tensile_SHORT_FILE_NAMES}     # ON or OFF
    ${Tensile_LIBRARY_PRINT_DEBUG}  # ON or OFF
    )
  target_link_libraries( ${ClientName} PUBLIC Tensile )
endif()


