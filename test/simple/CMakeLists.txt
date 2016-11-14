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

find_package(Tensile CONFIG)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake )

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
if(NOT MSVC)
    add_definitions(-std=c++11)
endif()

add_tensile_lib(SimpleTensileLib
    SOLUTIONS ${Tensile_SOLUTIONS}
    BACKEND ${Tensile_BACKEND}
    
)
if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( SimpleTensileLib PUBLIC opencl )
elseif( Tensile_BACKEND MATCHES "HIP")
  target_include_directories( SimpleTensileLib SYSTEM
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( SimpleTensileLib PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( SimpleTensileLib PUBLIC Tensile_BACKEND_HIP )
endif()

add_executable(main main.cpp)
target_link_libraries(main SimpleTensileLib)
