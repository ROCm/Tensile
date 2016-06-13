add_executable( simple
  main.cpp
  kernel_opencl.h
  kernel_hip.h
  )
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  target_link_libraries( simple ${OPENCL_LIBRARIES} )
elseif( Cobalt_BACKEND MATCHES "HIP")
  include_directories( ${HIP_INCLUDE_DIRS} )
endif()

