
################################################################################
# Global Options
################################################################################
cmake_minimum_required(VERSION 2.8)

############################################################
# Cobalt project name
set( CobaltSuper_NAME CobaltSuper)
project( ${CobaltSuper_NAME} C CXX )

############################################################
# Cobalt Version
set( ${CobaltSuper_NAME}_VERSION_MAJOR 0 )
set( ${CobaltSuper_NAME}_VERSION_MINOR 1 )
set( ${CobaltSuper_NAME}_VERSION_PATCH 0 )

############################################################
# App Name
set( Cobalt_APP_NAME "App" CACHE STRING
  "Prefix for application-specific files" )
#set_property( CACHE Cobalt_APP_NAME )

############################################################
# Cobalt Build Mode
# - LOGGER - Cobalt.lib in log-only mode
# - BACKEND - Cobalt.lib in full-backend mode
set( Cobalt_BUILD_MODE "Cobalt_BUILD_MODE_LOGGER" CACHE STRING
  "Cobalt operation mode?" )
set_property( CACHE Cobalt_BUILD_MODE PROPERTY STRINGS
  Cobalt_BUILD_MODE_LOGGER
  Cobalt_BUILD_MODE_BACKEND )


############################################################
# Cobalt Backend
# - OpenCL 1.2
# - HCC
set( Cobalt_BACKEND "OpenCL_1.2" CACHE STRING
  "Which of Cobalt's backends to use?" )
set_property( CACHE Cobalt_BACKEND PROPERTY STRINGS
  OpenCL_1.2 HCC HSA )
# OpenCL 1.2
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  add_definitions( -DCobalt_BACKEND_OPENCL12=1 )
  add_definitions( -DCobalt_BACKEND_HCC=0 )
  add_definitions( -DCobalt_BACKEND_HSA=0 )
  find_package( OpenCL )
  message( STATUS "${Cobalt_BACKEND} backend selected." )
endif()
# HCC / Kalmar
if( Cobalt_BACKEND MATCHES "HCC")
  add_definitions( -DCobalt_BACKEND_OPENCL12=0 )
  add_definitions( -DCobalt_BACKEND_HCC=1 )
  add_definitions( -DCobalt_BACKEND_HSA=0 )
  message( SEND_ERROR "${Cobalt_BACKEND} backend not yet supported." )
endif()
# HSA
if( Cobalt_BACKEND MATCHES "HSA")
  add_definitions( -DCobalt_BACKEND_OPENCL12=0 )
  add_definitions( -DCobalt_BACKEND_HCC=0 )
  add_definitions( -DCobalt_BACKEND_HSA=1 )
  message( SEND_ERROR "${Cobalt_BACKEND} backend not yet supported." )
endif()

############################################################
# Cobalt Headers needed for all subprojects
set( CobaltLib_DIRS_INCLUDE ${CMAKE_SOURCE_DIR}/CobaltLib/include
  ${CMAKE_SOURCE_DIR}/CobaltLib/src)
set( CobaltLib_API ${CMAKE_SOURCE_DIR}/CobaltLib/include/Cobalt.h )
set( CobaltLib_NAME CobaltLib )
source_group(include FILES ${CobaltLib_API} )

############################################################
# Directories
set( Cobalt_DIR_GENERATED ${CMAKE_BINARY_DIR}/CobaltGenerated )
set( Cobalt_DIR_PROBLEMS ${Cobalt_DIR_GENERATED}/XML/AppProblemLogs )
set( Cobalt_FILE_PROBLEMS ${Cobalt_DIR_PROBLEMS}/${Cobalt_APP_NAME}_Log.xml )
set( Cobalt_DIR_PSM ${Cobalt_DIR_GENERATED}/XML/ProblemSolutionMaps )
set( Cobalt_FILE_PSM ${Cobalt_DIR_PSM}/${Cobalt_APP_NAME}_PSM.xml )

file( MAKE_DIRECTORY ${Cobalt_DIR_GENERATED} )
file( MAKE_DIRECTORY ${Cobalt_DIR_PROBLEMS} )
file( MAKE_DIRECTORY ${Cobalt_DIR_PSM} )

############################################################
# CobaltGen Files
set( CobaltGenBenchmark_SRC
  CobaltGen/CobaltGenBenchmark.py
  CobaltGen/FileReader.py
  CobaltGen/FileWriter.py
  CobaltGen/SolutionCandidateGenerator.py
  CobaltGen/SolutionSet.py
  CobaltGen/SolutionWriter.py
  CobaltGen/Structs.py
)
set( CobaltGenBackend_SRC
  CobaltGen/CobaltGenBackend.py
  CobaltGen/FileReader.py
  CobaltGen/FileWriter.py
  CobaltGen/SolutionSet.py
  CobaltGen/SolutionWriter.py
  CobaltGen/Structs.py
)

############################################################
# Dynamic CMake files - will be over-written later
############################################################
# Kernels
set( Cobalt_KernelFiles_CMAKE_DYNAMIC
    ${Cobalt_DIR_GENERATED}/Kernels/CobaltKernels.cmake )
if( NOT EXISTS ${Cobalt_KernelFiles_CMAKE_DYNAMIC})
  file( WRITE ${Cobalt_KernelFiles_CMAKE_DYNAMIC}
    "set( Cobalt_KernelFiles_GENERATED_DYNAMIC \"\" )\n"
  )
endif()
# Solutions
set( Cobalt_SolutionFiles_CMAKE_DYNAMIC
  ${Cobalt_DIR_GENERATED}/Solutions/CobaltSolutions.cmake )
if( NOT EXISTS ${Cobalt_SolutionFiles_CMAKE_DYNAMIC})
  file( WRITE ${Cobalt_SolutionFiles_CMAKE_DYNAMIC}
    "set( Cobalt_SolutionFiles_GENERATED_DYNAMIC \"\" )\n"
  )
endif()


################################################################################
# Cobalt Subprojects
################################################################################
add_subdirectory(CobaltLib)
add_subdirectory(CobaltBenchmark)
add_subdirectory(clients)
