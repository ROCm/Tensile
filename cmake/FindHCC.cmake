################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

# findHCC does not currently address versioning, i.e.
# a rich directory structure where version number is a subdirectory under root
# Also, supported only on UNIX 64 bit systems.

if( NOT DEFINED ENV{HSA_PATH} )
    set( ENV{HSA_PATH} /opt/rocm/hsa)
endif()

if( NOT DEFINED  ENV{HCC_PATH} )
    set( ENV{HCC_PATH} /opt/rocm/hcc)
endif()

find_library(HSA_LIBRARY
    NAMES  hsa-runtime64
    PATHS
      ENV HSA_PATH
      /opt/rocm
    PATH_SUFFIXES
      lib)

find_program(HCC
    NAMES  hcc
    PATHS
        ENV HCC_PATH
        /opt/rocm
    PATH_SUFFIXES
        /bin)

find_path(HCC_INCLUDE_DIR
    NAMES
        hc.hpp
    PATHS
        ENV HCC_PATH
        /opt/rocm
    PATH_SUFFIXES
        /include/hcc
        /include
    )

set(HSA_LIBRARIES ${HSA_LIBRARY})
set(HCC_INCLUDE_DIRS ${HCC_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  HCC
  FOUND_VAR HCC_FOUND
  REQUIRED_VARS HSA_LIBRARIES HCC_INCLUDE_DIRS HCC)

mark_as_advanced(
  HSA_LIBRARIES
  HCC_INCLUDE_DIRS
)
