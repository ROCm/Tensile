################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
cmake_minimum_required(VERSION 2.8.12)

set( SAMPLE_GEMM_NAME gemm )

# gemm sample source files
set( SAMPLE_GEMM_SRC gemm.cpp dnn_gemms.h )


# gemm executable
add_executable( ${SAMPLE_GEMM_NAME} EXCLUDE_FROM_ALL ${SAMPLE_GEMM_SRC} )

add_tensile_lib(gemm_tensile EXCLUDE_FROM_ALL
    PROBLEMS ${Tensile_DIR_PROBLEMS}
    BACKEND ${Tensile_BACKEND}
    ENABLE_LOGGER
)

# gemm links Tensile
target_link_libraries( ${SAMPLE_GEMM_NAME} PUBLIC gemm_tensile )

if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( gemm_tensile PUBLIC opencl )
elseif( Tensile_BACKEND MATCHES "HIP")
  target_include_directories( gemm_tensile SYSTEM
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( gemm_tensile PUBLIC ${HSA_LIBRARIES} )
endif()

set_property( TARGET ${SAMPLE_GEMM_NAME} PROPERTY FOLDER "clients" )
