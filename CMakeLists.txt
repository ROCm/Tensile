################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

################################################################################
# Global Options
################################################################################
cmake_minimum_required(VERSION 2.8.12)
# Use legacy variable reference and escape sequence evaluation. This might not
# be necessary, but is a warning in CMake 3.6
if(POLICY CMP0053)
  cmake_policy(SET CMP0053 OLD)
endif()

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )

project( Tensile )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

option( Tensile_BUILD_CLIENTS "Build Tensile clients" ON)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

############################################################
# Tensile Version
set( ${TensileProj_NAME}_VERSION_MAJOR 1 )
set( ${TensileProj_NAME}_VERSION_MINOR 1 )
set( ${TensileProj_NAME}_VERSION_PATCH 0 )
set( ${TensileProj_NAME}_VERSION_TWEAK 0 )
set( TensileProj_VERSION "${${TensileProj_NAME}_VERSION_MAJOR}.${${TensileProj_NAME}_VERSION_MINOR}.${${TensileProj_NAME}_VERSION_PATCH}.${${TensileProj_NAME}_VERSION_TWEAK}")


################################################################################
# Import standard install paths
################################################################################
include( GNUInstallDirs )

############################################################
# The following is cmake code to generate a config file package for Tensile
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( CMakePackageConfigHelpers )

set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( DATA_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/TensileLib )
set( config_package_location ${LIB_INSTALL_DIR}/cmake/TensileLib )

configure_package_config_file(
  TensileConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TensileConfig.cmake
  INSTALL_DESTINATION ${config_package_location}
  PATH_VARS LIB_INSTALL_DIR DATA_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/TensileConfig-version.cmake
  VERSION ${TensileProj_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Build entry point
if(WIN32)
set(Tensile_ENTRY_POINT ${CMAKE_CURRENT_BINARY_DIR}/bin/tensile.bat)
file(TO_NATIVE_PATH "${CMAKE_INSTALL_PREFIX}/${DATA_INSTALL_DIR}/TensileGen/Tensile.py" TENSILE_PY)
file(WRITE ${Tensile_ENTRY_POINT} "

python @TENSILE_PY@ %*

")
else()
set(Tensile_ENTRY_POINT ${CMAKE_CURRENT_BINARY_DIR}/bin/tensile)
file(WRITE ${Tensile_ENTRY_POINT} "#!/bin/bash
python @CMAKE_INSTALL_PREFIX@/@DATA_INSTALL_DIR@/TensileGen/Tensile.py \"$@\"

")
endif()

############################################################
# Library install logic

install(PROGRAMS
  ${Tensile_ENTRY_POINT}
  DESTINATION bin)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TensileConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/TensileConfig-version.cmake
  DESTINATION
    ${config_package_location} )

install( FILES
  TensileLib/include/Tensile.h
  DESTINATION include)

install(DIRECTORY
  TensileLib
  TensileGen
  TensileBenchmark
  cmake
  DESTINATION ${DATA_INSTALL_DIR}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cpp"
  PATTERN "*.py"
  PATTERN "*.txt"
  PATTERN "*.cmake"
)

############################################################
# Clients
set(Tensile_DIR ${CMAKE_CURRENT_SOURCE_DIR})
if (${Tensile_BUILD_CLIENTS})
  MESSAGE( STATUS "Building clients.")
  add_subdirectory(clients)
else()
  MESSAGE( STATUS "NOT building clients.")
endif()
