################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
cmake_minimum_required(VERSION 2.8.12)

# The contains strings to big to compile on MSVC; exclude from default build
add_executable( simple EXCLUDE_FROM_ALL
  main.cpp
  kernel_opencl.h
  kernel_hip.h
  )

add_tensile_lib(simple_tensile EXCLUDE_FROM_ALL
    PROBLEMS ${Tensile_DIR_PROBLEMS}
    BACKEND ${Tensile_BACKEND}
    ENABLE_LOGGER
)

target_link_libraries( simple PUBLIC simple_tensile )

if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( simple_tensile PUBLIC opencl )
elseif( Tensile_BACKEND MATCHES "HIP")
  target_include_directories( simple_tensile SYSTEM
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( simple_tensile PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( simple_tensile PUBLIC Tensile_BACKEND_HIP )
endif()

set_property( TARGET simple PROPERTY FOLDER "clients" )
