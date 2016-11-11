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

# The contains strings to big to compile on MSVC; exclude from default build
add_executable( simple EXCLUDE_FROM_ALL
  main.cpp
  kernel_opencl.h
  kernel_hip.h
  )

add_tensile_lib(simple_tensile EXCLUDE_FROM_ALL
    PROBLEMS ${Tensile_DIR_PROBLEMS}
    BACKEND ${Tensile_BACKEND}
    ENABLE_LOGGER
)

target_link_libraries( simple PUBLIC simple_tensile )

if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( simple_tensile PUBLIC opencl )
elseif( Tensile_BACKEND MATCHES "HIP")
  target_include_directories( simple_tensile SYSTEM
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( simple_tensile PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( simple_tensile PUBLIC Tensile_BACKEND_HIP )
endif()

set_property( TARGET simple PROPERTY FOLDER "clients" )
