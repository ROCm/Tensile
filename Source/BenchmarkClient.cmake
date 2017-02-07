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


# require C++11
if(MSVC)
  # object-level build parallelism for VS, not just target-level
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
  set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
  add_definitions( "-std=c++11" )
endif()

# include generated parameters
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR} )
# benchmark includes
include(${CMAKE_SOURCE_DIR}/Generated.cmake)
include_directories( . Kernels Solutions )
# executable
add_executable( ${TensileClient}
  ${TensileBenchmark_Source}
  ${TensileBenchmark_Solutions}
  ${TensileBenchmark_Kernels}
)


###############################################################################
# Backend dependent parameters
if( TensileBenchmark_BACKEND MATCHES "OCL")
  find_package(OpenCL "1.2" REQUIRED)
  target_link_libraries( ${TensileClient} PRIVATE opencl )
  target_compile_definitions( ${TensileClient} PUBLIC 
    -DTensile_BACKEND_OCL=1
    -DTensile_BACKEND_HIP=0 )
  target_include_directories( ${TensileClient} SYSTEM
    PUBLIC  ${OPENCL_INCLUDE_DIRS} ) 
elseif( TensileBenchmark_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  set (CMAKE_CXX_COMPILER ${HIPCC})
  target_include_directories( ${TensileClient} SYSTEM
    PUBLIC  ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS} )
  target_link_libraries( ${TensileClient} PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( ${TensileClient} PUBLIC 
    -DTensile_BACKEND_OCL=0
    -DTensile_BACKEND_HIP=1 )
else()
  message(STATUS "No backend selected in Generated.cmake." )
endif()
