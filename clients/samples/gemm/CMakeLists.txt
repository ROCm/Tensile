set( SAMPLE_GEMM_NAME gemm)

# gemm sample source files
set( SAMPLE_GEMM_SRC gemm.cpp )

# gemm executable
add_executable( ${SAMPLE_GEMM_NAME} ${SAMPLE_GEMM_SRC} )

# gemm links Cobalt
target_link_libraries( ${SAMPLE_GEMM_NAME} ${CobaltLib_NAME} )

# include Cobalt headers
include_directories( ${CobaltLib_DIRS_INCLUDE} )

# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  include_directories( ${OpenCL_INCLUDE_DIR} )
  target_link_libraries( ${SAMPLE_GEMM_NAME} ${OpenCL_LIBRARY} )
endif()

