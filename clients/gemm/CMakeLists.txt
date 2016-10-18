################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

set( SAMPLE_GEMM_NAME gemm )

# gemm sample source files
set( SAMPLE_GEMM_SRC gemm.cpp dnn_gemms.h )


# gemm executable
add_executable( ${SAMPLE_GEMM_NAME} EXCLUDE_FROM_ALL ${SAMPLE_GEMM_SRC} )

add_cobalt_lib(gemm_cobalt EXCLUDE_FROM_ALL
    PROBLEMS ${Cobalt_DIR_PROBLEMS}
    BACKEND ${Cobalt_BACKEND}
    ENABLE_LOGGER
)

# gemm links Cobalt
target_link_libraries( ${SAMPLE_GEMM_NAME} PUBLIC gemm_cobalt )

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( gemm_cobalt PUBLIC opencl )
elseif( Cobalt_BACKEND MATCHES "HIP")
  target_include_directories( gemm_cobalt
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( gemm_cobalt PUBLIC ${HSA_LIBRARIES} )
endif()

set_property( TARGET ${SAMPLE_GEMM_NAME} PROPERTY FOLDER "Clients" )
