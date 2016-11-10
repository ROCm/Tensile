################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
cmake_minimum_required(VERSION 2.8.12)

set( SAMPLE_DNN_NAME dnn)

# dnn sample source files
set( SAMPLE_DNN_SRC main.cpp )

# gemm executable
add_executable( ${SAMPLE_DNN_NAME} EXCLUDE_FROM_ALL ${SAMPLE_DNN_SRC} )

add_tensile_lib(dnn_tensile EXCLUDE_FROM_ALL
    PROBLEMS ${Tensile_DIR_PROBLEMS}
    BACKEND ${Tensile_BACKEND}
    ENABLE_LOGGER
)

# gemm links Tensile
target_link_libraries( ${SAMPLE_DNN_NAME} PUBLIC dnn_tensile )

if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( dnn_tensile PUBLIC opencl )
elseif( Tensile_BACKEND MATCHES "HIP")
  target_include_directories( dnn_tensile SYSTEM
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( dnn_tensile PUBLIC ${HSA_LIBRARIES} )
endif()

set_property( TARGET ${SAMPLE_DNN_NAME} PROPERTY FOLDER "clients" )
