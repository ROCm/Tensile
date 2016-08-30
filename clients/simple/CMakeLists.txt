# The contains strings to big to compile on MSVC; exclude from default build
add_executable( simple EXCLUDE_FROM_ALL
  main.cpp
  kernel_opencl.h
  kernel_hip.h
  )

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( simple PRIVATE opencl )
elseif( Cobalt_BACKEND MATCHES "HIP")
  include_directories( ${HIP_INCLUDE_DIRS} )
endif()

set_property( TARGET simple PROPERTY FOLDER "Clients" )
