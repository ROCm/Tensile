################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

include(AddCobaltLib)
include(EnableWarnings)
set(Cobalt_DIR_PROBLEMS ${PROJECT_BINARY_DIR}/problems CACHE PATH "path to problems used by clients")

find_package(OpenCL "1.2")
find_package(HIP)

set(Cobalt_BACKENDS NONE)

if( HIP_FOUND )
  set( Cobalt_BACKEND "HIP" CACHE STRING "Which of Cobalt's backends to use?" )
  list(APPEND Cobalt_BACKENDS HIP)
endif()

if( OPENCL_FOUND )
  set( Cobalt_BACKEND "OpenCL_1.2" CACHE STRING "Which of Cobalt's backends to use?" )
  list(APPEND Cobalt_BACKENDS OpenCL_1.2)
endif()
message(STATUS "Cobalt_BACKENDS: ${Cobalt_BACKENDS}")
set_property( CACHE Cobalt_BACKEND PROPERTY STRINGS ${Cobalt_BACKENDS})

# OpenCL 1.2
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  message(STATUS "${Cobalt_BACKEND} backend selected." )
# HIP
elseif( Cobalt_BACKEND MATCHES "HIP")
  set (CMAKE_CXX_COMPILER ${HIPCC})
  message(STATUS "${Cobalt_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
endif( )
add_subdirectory( dnn )
add_subdirectory( gemm )
add_subdirectory( simple )
