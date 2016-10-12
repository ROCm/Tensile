################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

################################################################################
# Global Options
################################################################################
cmake_minimum_required(VERSION 2.8.12)
# Use legacy variable reference and escape sequence evaluation. This might not
# be necessary, but is a warning in CMake 3.6
cmake_policy(SET CMP0053 OLD)

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )

project( Cobalt )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

############################################################
# Cobalt Version
set( ${CobaltProj_NAME}_VERSION_MAJOR 1 )
set( ${CobaltProj_NAME}_VERSION_MINOR 1 )
set( ${CobaltProj_NAME}_VERSION_PATCH 0 )
set( ${CobaltProj_NAME}_VERSION_TWEAK 0 )
set( CobaltProj_VERSION "${${CobaltProj_NAME}_VERSION_MAJOR}.${${CobaltProj_NAME}_VERSION_MINOR}.${${CobaltProj_NAME}_VERSION_PATCH}.${${CobaltProj_NAME}_VERSION_TWEAK}")


################################################################################
# Import standard install paths
################################################################################
include( GNUInstallDirs )

############################################################
# The following is cmake code to generate a config file package for Cobalt
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( CMakePackageConfigHelpers )

set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( DATA_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/cobalt )
set( config_package_location ${LIB_INSTALL_DIR}/cmake/cobalt )

configure_package_config_file(
  cobalt-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cobalt-config.cmake
  INSTALL_DESTINATION ${config_package_location}
  PATH_VARS LIB_INSTALL_DIR DATA_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/cobalt-config-version.cmake
  VERSION ${CobaltProj_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Build entry point
if(WIN32)
set(Cobalt_ENTRY_POINT ${CMAKE_CURRENT_BINARY_DIR}/bin/cobalt.bat)
file(WRITE ${Cobalt_ENTRY_POINT} "

python @CMAKE_INSTALL_PREFIX@\\@DATA_INSTALL_DIR@\\CobaltGen\\Cobalt.py %*

")
else()
set(Cobalt_ENTRY_POINT ${CMAKE_CURRENT_BINARY_DIR}/bin/cobalt)
file(WRITE ${Cobalt_ENTRY_POINT} "

#!/bin/bash
python @CMAKE_INSTALL_PREFIX@/@DATA_INSTALL_DIR@/CobaltGen/Cobalt.py \"$@\"

")
endif()

############################################################
# Library install logic

install(PROGRAMS
  ${Cobalt_ENTRY_POINT}
  DESTINATION bin)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cobalt-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cobalt-config-version.cmake
  DESTINATION
    ${config_package_location} )

install( FILES
  CobaltLib/include/Cobalt.h
  DESTINATION include)

install(DIRECTORY
  CobaltLib
  CobaltGen
  CobaltBenchmark
  cmake
  DESTINATION ${DATA_INSTALL_DIR}
)

############################################################
# Clients
set(Cobalt_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(clients)
