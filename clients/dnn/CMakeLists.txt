
set( SAMPLE_DNN_NAME dnn)

# dnn sample source files
set( SAMPLE_DNN_SRC main.cpp )


# gemm executable
add_executable( ${SAMPLE_DNN_NAME} ${SAMPLE_DNN_SRC} )

# gemm links Cobalt
target_link_libraries( ${SAMPLE_DNN_NAME} ${CobaltLogger_NAME} )

# include Cobalt headers
include_directories(
  ${CobaltLib_DIRS_INCLUDE}
  )

# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  target_link_libraries( ${SAMPLE_DNN_NAME} ${OPENCL_LIBRARY} )
endif()


if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  add_definitions(
    -DCobalt_DIR_PROBLEMS="${Cobalt_DIR_PROBLEMS}")
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  target_link_libraries( ${SAMPLE_DNN_NAME} ${OPENCL_LIBRARIES} )
elseif( Cobalt_BACKEND MATCHES "HIP")
  add_definitions(
    -DCobalt_DIR_PROBLEMS=\\\"${Cobalt_DIR_PROBLEMS}\\\")
  include_directories( ${HCC_INCLUDE_DIRS} )
  target_link_libraries( ${SAMPLE_DNN_NAME} ${HSA_LIBRARIES} )
endif()

