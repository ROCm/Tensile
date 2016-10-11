################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

#if( NOT EXISTS ${CobaltBenchmark_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl)
#  MESSAGE( "Creating dummy SolutionTemplateInstantiations.inl for client")
#  file( WRITE ${CobaltBenchmark_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl
#      ""
#  )
#endif()

set( SAMPLE_GEMM_NAME gemm )

# gemm sample source files
set( SAMPLE_GEMM_SRC gemm.cpp dnn_gemms.h )


# gemm executable
add_executable( ${SAMPLE_GEMM_NAME} ${SAMPLE_GEMM_SRC} )
#set_property(TARGET ${SAMPLE_GEMM_NAME} PROPERTY CXX_STANDARD 11 )

# gemm links Cobalt
target_link_libraries( ${SAMPLE_GEMM_NAME} PRIVATE ${CobaltLogger_NAME} )

# include Cobalt headers
include_directories(
  ${CobaltLib_DIRS_INCLUDE}
  )

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  add_definitions( -DCobalt_DIR_PROBLEMS="${Cobalt_DIR_PROBLEMS}")
  target_link_libraries( ${SAMPLE_GEMM_NAME} PRIVATE opencl )
elseif( Cobalt_BACKEND MATCHES "HIP")
  add_definitions( -DCobalt_DIR_PROBLEMS=\\\"${Cobalt_DIR_PROBLEMS}\\\")
  target_include_directories( ${SAMPLE_GEMM_NAME}
    PRIVATE $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( ${SAMPLE_GEMM_NAME} PRIVATE ${HSA_LIBRARIES} )
endif()

set_property( TARGET ${SAMPLE_GEMM_NAME} PROPERTY FOLDER "Clients" )
