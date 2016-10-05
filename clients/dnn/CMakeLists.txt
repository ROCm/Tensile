################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################


set( SAMPLE_DNN_NAME dnn)

# dnn sample source files
set( SAMPLE_DNN_SRC main.cpp )

# gemm executable
add_executable( ${SAMPLE_DNN_NAME} ${SAMPLE_DNN_SRC} )

# gemm links Cobalt
target_link_libraries( ${SAMPLE_DNN_NAME} PRIVATE ${CobaltLogger_NAME} )

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  add_definitions( -DCobalt_DIR_PROBLEMS="${Cobalt_DIR_PROBLEMS}" )
  target_link_libraries( ${SAMPLE_DNN_NAME} PRIVATE opencl )
elseif( Cobalt_BACKEND MATCHES "HIP")
  add_definitions( -DCobalt_DIR_PROBLEMS=\\\"${Cobalt_DIR_PROBLEMS}\\\" )
  target_include_directories( ${SAMPLE_DNN_NAME}
    PRIVATE $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( ${SAMPLE_DNN_NAME} PRIVATE ${HSA_LIBRARIES} )
endif()

set_property( TARGET ${SAMPLE_DNN_NAME} PROPERTY FOLDER "Clients" )
