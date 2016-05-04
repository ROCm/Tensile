if( NOT EXISTS ${CobaltBenchmark_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl)
  MESSAGE( "Creating dummy SolutionTemplateInstantiations.inl for client")
  file( WRITE ${CobaltBenchmark_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl
      ""
  )
endif()

set( SAMPLE_GEMM_NAME gemm)

# gemm sample source files
set( SAMPLE_GEMM_SRC gemm.cpp )


# gemm executable
add_executable( ${SAMPLE_GEMM_NAME} ${SAMPLE_GEMM_SRC} )
#set_property(TARGET ${SAMPLE_GEMM_NAME} PROPERTY CXX_STANDARD 11 )

# gemm links Cobalt
target_link_libraries( ${SAMPLE_GEMM_NAME} ${CobaltLib_NAME} )

# include Cobalt headers
include_directories(
  ${CobaltLib_DIRS_INCLUDE}
  )

# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  target_link_libraries( ${SAMPLE_GEMM_NAME} ${OPENCL_LIBRARY} )
endif()


if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  add_definitions(
    -DCobaltBenchmark_DIR_PROBLEMS="${CobaltBenchmark_DIR_PROBLEMS}")
elseif( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  add_definitions(
    -DCobaltBenchmark_DIR_PROBLEMS=\\\"${CobaltBenchmark_DIR_PROBLEMS}\\\")
endif()
