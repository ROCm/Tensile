
# gemm sample source files
set( SAMPLE_GEMM_SRC
  gemm.cpp
  )

# gemm executable
add_executable( gemm
  ${SAMPLE_GEMM_SRC}
  )

# gemm links Cobalt
target_link_libraries( gemm
  ${CobaltLib_NAME}
  )

# include Cobalt headers
include_directories(
  ${CobaltLib_INCLUDE_DIR}
  )

