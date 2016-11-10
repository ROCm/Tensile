################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
cmake_minimum_required(VERSION 2.8.12)

set( SAMPLE_GEMM_NAME gemm )

# gemm sample source files
set( SAMPLE_GEMM_SRC gemm.cpp dnn_gemms.h )


# gemm executable
add_executable( ${SAMPLE_GEMM_NAME} EXCLUDE_FROM_ALL ${SAMPLE_GEMM_SRC} )

add_tensile_lib(TensileLib_GEMM EXCLUDE_FROM_ALL
    PROBLEMS ${Tensile_DIR_PROBLEMS}
    #SOLUTIONS ${Tensile_DIR_SOLUTIONS}
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
