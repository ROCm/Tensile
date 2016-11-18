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

set( SAMPLE_GEMM_NAME gemm )

# gemm sample source files
set( SAMPLE_GEMM_SRC gemm.cpp dnn_gemms.h )


# gemm executable
add_executable( ${SAMPLE_GEMM_NAME} EXCLUDE_FROM_ALL ${SAMPLE_GEMM_SRC} )

add_tensile_lib(TensileLib_GEMM EXCLUDE_FROM_ALL
  PROBLEMS ${Tensile_DIR_PROBLEMS}
  #SOLUTIONS ${Tensile_DIR_PROBLEMS}/../SolutionXMLs
  BACKEND ${Tensile_BACKEND}
  ENABLE_LOGGER
)

# gemm links Tensile
target_link_libraries( ${SAMPLE_GEMM_NAME} PUBLIC TensileLib_GEMM )

if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( TensileLib_GEMM PUBLIC opencl )
elseif( Tensile_BACKEND MATCHES "HIP")
  target_include_directories( TensileLib_GEMM SYSTEM
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( TensileLib_GEMM PUBLIC ${HSA_LIBRARIES} )
endif()

set_property( TARGET ${SAMPLE_GEMM_NAME} PROPERTY FOLDER "clients" )
