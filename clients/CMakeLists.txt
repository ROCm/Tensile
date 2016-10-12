################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

include(AddCobaltLib)
set(Cobalt_DIR_PROBLEMS ${PROJECT_BINARY_DIR}/problems CACHE PATH "path to problems used by clients")
# OpenCL 1.2
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  find_package(OpenCL "1.2" REQUIRED)
  message(STATUS "${Cobalt_BACKEND} backend selected." )
# HIP
elseif( Cobalt_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  message(STATUS "${Cobalt_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
endif( )
add_subdirectory( dnn )
add_subdirectory( gemm )
add_subdirectory( simple )
