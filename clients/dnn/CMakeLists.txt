################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################


set( SAMPLE_DNN_NAME dnn)

# dnn sample source files
set( SAMPLE_DNN_SRC main.cpp )

# gemm executable
add_executable( ${SAMPLE_DNN_NAME} EXCLUDE_FROM_ALL ${SAMPLE_DNN_SRC} )

add_cobalt_lib(dnn_cobalt EXCLUDE_FROM_ALL
    PROBLEMS ${Cobalt_DIR_PROBLEMS}
    BACKEND ${Cobalt_BACKEND}
    ENABLE_LOGGER
)

# gemm links Cobalt
target_link_libraries( ${SAMPLE_DNN_NAME} PRIVATE dnn_cobalt )

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( ${SAMPLE_DNN_NAME} PRIVATE opencl )
elseif( Cobalt_BACKEND MATCHES "HIP")
  target_include_directories( ${SAMPLE_DNN_NAME}
    PRIVATE $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( ${SAMPLE_DNN_NAME} PRIVATE ${HSA_LIBRARIES} )
endif()

set_property( TARGET ${SAMPLE_DNN_NAME} PROPERTY FOLDER "Clients" )
