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


add_definitions(-DCobalt_DIR_PROBLEMS="${Cobalt_DIR_PROBLEMS}")
add_definitions(-DCobalt_DIR_SOLUTIONS="${Cobalt_DIR_SOLUTIONS}")
add_definitions(-DCobalt_DIR_GENERATED="${Cobalt_DIR_GENERATED}")
