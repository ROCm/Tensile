################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

# The contains strings to big to compile on MSVC; exclude from default build
add_executable( simple EXCLUDE_FROM_ALL
  main.cpp
  kernel_opencl.h
  kernel_hip.h
  )

add_cobalt_lib(simple_cobalt EXCLUDE_FROM_ALL
    PROBLEMS ${Cobalt_DIR_PROBLEMS}
    BACKEND ${Cobalt_BACKEND}
    ENABLE_LOGGER
)

target_link_libraries( simple PUBLIC simple_cobalt )

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( simple_cobalt PUBLIC opencl )
elseif( Cobalt_BACKEND MATCHES "HIP")
  target_include_directories( simple_cobalt SYSTEM
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( simple_cobalt PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( simple_cobalt PUBLIC Cobalt_BACKEND_HIP )
endif()

set_property( TARGET simple PROPERTY FOLDER "clients" )
