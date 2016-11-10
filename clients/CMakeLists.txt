################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
cmake_minimum_required(VERSION 2.8.12)

include(AddTensileLib)
include(EnableWarnings)
set(Tensile_DIR_PROBLEMS ${PROJECT_BINARY_DIR}/problems CACHE PATH "path to problems used by clients")
file(MAKE_DIRECTORY ${Tensile_DIR_PROBLEMS})

find_package(PythonInterp)
find_package(OpenCL "1.2")
find_package(HIP)
if(NOT MSVC)
    add_definitions(-std=c++11)
endif()

set(Tensile_BACKENDS NONE)
if( HIP_FOUND )
  set( Tensile_BACKEND "HIP" CACHE STRING "Which of Tensile's backends to use?" )
  list(APPEND Tensile_BACKENDS HIP)
endif()

if( OPENCL_FOUND )
  set( Tensile_BACKEND "OpenCL_1.2" CACHE STRING "Which of Tensile's backends to use?" )
  list(APPEND Tensile_BACKENDS OpenCL_1.2)
endif()
message(STATUS "Tensile_BACKENDS: ${Tensile_BACKENDS}")

if(DEFINED Tensile_BACKEND)
    set_property( CACHE Tensile_BACKEND PROPERTY STRINGS ${Tensile_BACKENDS})
endif()

# OpenCL 1.2
if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  message(STATUS "${Tensile_BACKEND} backend selected." )
# HIP
elseif( Tensile_BACKEND MATCHES "HIP")
  set (CMAKE_CXX_COMPILER ${HIPCC})
  message(STATUS "${Tensile_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
endif( )
add_subdirectory( dnn )
add_subdirectory( gemm )
add_subdirectory( simple )
