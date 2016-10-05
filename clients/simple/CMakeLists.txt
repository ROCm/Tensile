################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

# The contains strings to big to compile on MSVC; exclude from default build
add_executable( simple EXCLUDE_FROM_ALL
  main.cpp
  kernel_opencl.h
  kernel_hip.h
  )

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( simple PRIVATE opencl )
elseif( Cobalt_BACKEND MATCHES "HIP")
  target_include_directories( simple
    PRIVATE $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( simple PRIVATE ${HSA_LIBRARIES} )
  target_compile_definitions( simple PRIVATE Cobalt_BACKEND_HIP )
endif()

set_property( TARGET simple PROPERTY FOLDER "Clients" )
