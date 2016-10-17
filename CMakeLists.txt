################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################


################################################################################
# Global Options
################################################################################
cmake_minimum_required(VERSION 2.8.12)

############################################################
# Cobalt project name
set( CobaltProj_NAME Cobalt )
set( CobaltLib_NAME CobaltLib )
set( CobaltLogger_NAME CobaltLogger )
set( CobaltBenchmark_NAME CobaltBenchmark )

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )

project( ${CobaltProj_NAME} C CXX )

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

############################################################
# FindOpenCL.cmake
set( CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${PROJECT_SOURCE_DIR} )
message( STATUS "CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}" )

############################################################
# require C++11
if(MSVC)
  # object-level build parallelism for VS, not just target-level
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
  set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
endif()
add_definitions( "-std=c++11" )

############################################################
# OPTION - CobaltLib Build Mode
# note: logger is always enabled in CobaltLogger
set( Cobalt_ENABLE_LOGGER Off CACHE BOOL "Enable logger in CobaltLib?" )

############################################################
# OPTION - Cobalt Backend
# - OpenCL 1.2
# - HIP
if( UNIX )
  set( Cobalt_BACKEND "HIP" CACHE STRING "Which of Cobalt's backends to use?" )
else()
  set( Cobalt_BACKEND "OpenCL_1.2" CACHE STRING "Which of Cobalt's backends to use?" )
endif()

set_property( CACHE Cobalt_BACKEND PROPERTY STRINGS OpenCL_1.2 HIP )

# OpenCL 1.2
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  find_package(OpenCL "1.2" REQUIRED)
  message(STATUS "${Cobalt_BACKEND} backend selected." )
  set( CobaltLib_NAME "${CobaltLib_NAME}_OCL" )
  set( CobaltLogger_NAME "${CobaltLogger_NAME}_OCL" )

# HIP
elseif( Cobalt_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  message(STATUS "${Cobalt_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
  if( ${HIP_PLATFORM} MATCHES "HIP" )
    set( CobaltLib_NAME "${CobaltLib_NAME}_HIP" )
    set( CobaltLogger_NAME "${CobaltLogger_NAME}_HIP" )
  else( )
    set( CobaltLib_NAME "${CobaltLib_NAME}_CUDA" )
    set( CobaltLogger_NAME "${CobaltLogger_NAME}_CUDA" )
  endif( )
endif( )

############################################################
# OPTION - Cobalt Optimizations
# - On - do write kernels specifically optimized for alpha=1, beta=0
# - Off - don't write kernels specifically optimized for alpha=1, beta=0
set( Cobalt_OPTIMIZE_ALPHA Off CACHE BOOL
  "Should Cobalt write optimized kernels for alpha=1?" )
set( Cobalt_OPTIMIZE_BETA Off CACHE BOOL
  "Should Cobalt write optimized kernels for beta=0?" )


############################################################
# Cobalt Headers needed for all subprojects
set( CobaltLib_DIRS_INCLUDE
  ${CMAKE_SOURCE_DIR}/CobaltLib/include
  ${CMAKE_SOURCE_DIR}/CobaltLib/src)
set( CobaltLib_API ${CMAKE_SOURCE_DIR}/CobaltLib/include/Cobalt.h )
source_group(include FILES ${CobaltLib_API} )

############################################################
# Directories
set( CobaltBenchmark_DIR_GENERATED ${CMAKE_BINARY_DIR}/CobaltGenerated/CobaltBenchmark )
set( CobaltLib_DIR_GENERATED   ${CMAKE_BINARY_DIR}/CobaltGenerated/CobaltLib )
set( Cobalt_DIR_PROBLEMS  ${CobaltBenchmark_DIR_GENERATED}/XML_ApplicationProblemTraces CACHE STRING "Directory from which to read application problem trace xml files")
set( Cobalt_DIR_SOLUTIONS ${CobaltBenchmark_DIR_GENERATED}/XML_BenchmarkSolutionTimes CACHE STRING "Directory from which to read benchmark solution times xml files")

file( MAKE_DIRECTORY ${CobaltBenchmark_DIR_GENERATED} )
file( MAKE_DIRECTORY ${CobaltLib_DIR_GENERATED} )
file( MAKE_DIRECTORY ${Cobalt_DIR_PROBLEMS} )
file( MAKE_DIRECTORY ${Cobalt_DIR_SOLUTIONS} )

############################################################
# CobaltGen Files
set( CobaltGen_SRC
  ${CMAKE_SOURCE_DIR}/CobaltGen/CobaltGenBackend.py
  ${CMAKE_SOURCE_DIR}/CobaltGen/CobaltGenBenchmark.py
  ${CMAKE_SOURCE_DIR}/CobaltGen/FileReader.py
  ${CMAKE_SOURCE_DIR}/CobaltGen/FileWriter.py
  ${CMAKE_SOURCE_DIR}/CobaltGen/KernelWriter.py
  ${CMAKE_SOURCE_DIR}/CobaltGen/SolutionCandidateGenerator.py
  ${CMAKE_SOURCE_DIR}/CobaltGen/SolutionWriter.py
  ${CMAKE_SOURCE_DIR}/CobaltGen/SolutionSelectionWriter.py
  ${CMAKE_SOURCE_DIR}/CobaltGen/Structs.py
)

############################################################
# Dynamic CMake files - will be over-written later
############################################################
# Benchmark Kernels
set( CobaltBenchmark_KernelFiles_CMAKE_DYNAMIC
    ${CobaltBenchmark_DIR_GENERATED}/Kernels/CobaltKernels.cmake )
if( NOT EXISTS ${CobaltBenchmark_KernelFiles_CMAKE_DYNAMIC})
file( WRITE ${CobaltBenchmark_KernelFiles_CMAKE_DYNAMIC}
    "set( CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC \"\" )\n"
    "source_group(Kernels FILES "
    "${CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC} )\n"
)
endif()

# Lib Kernels
set( CobaltLib_KernelFiles_CMAKE_DYNAMIC
  ${CobaltLib_DIR_GENERATED}/Kernels/CobaltKernels.cmake )
if( NOT EXISTS ${CobaltLib_KernelFiles_CMAKE_DYNAMIC})
file( WRITE ${CobaltLib_KernelFiles_CMAKE_DYNAMIC}
    "set( CobaltLib_KernelFiles_GENERATED_DYNAMIC \"\" )\n"
    "source_group(Kernels FILES ${CobaltLib_KernelFiles_GENERATED_DYNAMIC} )\n"
)
endif()

# Benchmark Solutions
set( CobaltBenchmark_SolutionFiles_CMAKE_DYNAMIC
  ${CobaltBenchmark_DIR_GENERATED}/Solutions/CobaltSolutions.cmake )
if( NOT EXISTS ${CobaltBenchmark_SolutionFiles_CMAKE_DYNAMIC})
  file( WRITE ${CobaltBenchmark_SolutionFiles_CMAKE_DYNAMIC}
    "set( CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC \"\" )\n"
    "source_group(Solutions FILES ${CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC} )\n"
  )
endif()

# Lib Solutions
set( CobaltLib_SolutionFiles_CMAKE_DYNAMIC
  ${CobaltLib_DIR_GENERATED}/Solutions/CobaltSolutions.cmake )
if( NOT EXISTS ${CobaltLib_SolutionFiles_CMAKE_DYNAMIC})
  file( WRITE ${CobaltLib_SolutionFiles_CMAKE_DYNAMIC}
    "set( CobaltLib_SolutionFiles_GENERATED_DYNAMIC \"\" )\n"
    "source_group(Solutions FILES ${CobaltLib_SolutionFiles_GENERATED_DYNAMIC} )\n"
  )
endif()

# Templates
if( NOT EXISTS
  ${CobaltLib_DIR_GENERATED}/Solutions/SolutionTemplateInstantiations.inl)
  #MESSAGE( "Creating dummy SolutionTemplateInstantiations.inl for CobaltLib")
  file( WRITE
    ${CobaltLib_DIR_GENERATED}/Solutions/SolutionTemplateInstantiations.inl
    "" )
endif()

# CobaltLib
set( CobaltLib_CMAKE_DYNAMIC
  ${CobaltLib_DIR_GENERATED}/Other/CobaltLib.cmake )
if( NOT EXISTS ${CobaltLib_CMAKE_DYNAMIC})
  file( WRITE ${CobaltLib_CMAKE_DYNAMIC}
  "include( \${CobaltLib_KernelFiles_CMAKE_DYNAMIC} )\n"
  "include( \${CobaltLib_SolutionFiles_CMAKE_DYNAMIC} )\n"
  "set( CobaltLib_SRC_GENERATED_DYNAMIC\n"
  "  \${CobaltLib_KernelFiles_GENERATED_DYNAMIC}\n"
  "  \${CobaltLib_SolutionFiles_GENERATED_DYNAMIC} )\n"
  )
endif()
include( ${CobaltLib_CMAKE_DYNAMIC} )


if( NOT EXISTS
    ${CobaltBenchmark_DIR_GENERATED}/Solutions/SolutionTemplateInstantiations.inl)
  #MESSAGE( "Creating dummy SolutionTemplateInstantiations.inl for benchmark")
  file( WRITE ${CobaltBenchmark_DIR_GENERATED}/Solutions/SolutionTemplateInstantiations.inl
    "" )
endif()


set( CobaltBenchmark_CMAKE_DYNAMIC
  ${CobaltBenchmark_DIR_GENERATED}/Other/CobaltBenchmark.cmake
)
if( NOT EXISTS ${CobaltBenchmark_CMAKE_DYNAMIC})
  file( WRITE ${CobaltBenchmark_CMAKE_DYNAMIC}
    "include( \${CobaltBenchmark_KernelFiles_CMAKE_DYNAMIC} )\n"
    "include( \${CobaltBenchmark_SolutionFiles_CMAKE_DYNAMIC} )\n"
    "set( CobaltBenchmark_SRC_GENERATED_DYNAMIC\n"
    "  \${CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC}\n"
    "  \${CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC} )\n" )
endif()
include( ${CobaltBenchmark_CMAKE_DYNAMIC} )



################################################################################
# XML Lists
################################################################################
set( Cobalt_XML_PROBLEMS_FILE
  ${Cobalt_DIR_PROBLEMS}/list_of_xmls.txt )
if( NOT EXISTS ${Cobalt_XML_PROBLEMS_FILE})
  file( WRITE ${Cobalt_XML_PROBLEMS_FILE}
    "The contents of this file isn't read; it only exists for CMake to check the timestamp for rebuilding target CobaltGenBenchmark.py. CobaltGenBackend.py will use all XMLs in this directory.\n" )
endif()
set( Cobalt_XML_SOLUTIONS_FILE
  ${Cobalt_DIR_SOLUTIONS}/list_of_xmls.txt )
if( NOT EXISTS ${Cobalt_XML_SOLUTIONS_FILE})
  file( WRITE ${Cobalt_XML_SOLUTIONS_FILE}
    "The contents of this file isn't read; it only exists for CMake to check the timestamp for rebuilding target CobaltGenBackend.py. CobaltGenBackend.py will use all XMLs in this directory.\n" )
endif()


################################################################################
# Import standard install paths
################################################################################
include( GNUInstallDirs )

################################################################################
# Cobalt Subprojects
################################################################################
add_subdirectory(CobaltLib)
add_subdirectory(CobaltBenchmark)
add_subdirectory(clients)
