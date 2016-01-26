################################################################################
# CobaltLib
# project to generate Cobat.lib (full backend) from ProbSolMap.xml
################################################################################
cmake_minimum_required(VERSION 2.8)
project( ${CobaltLib_NAME} C CXX)
set( ${CobaltLib_NAME}_VERSION_MAJOR 0 )
set( ${CobaltLib_NAME}_VERSION_MINOR 1 )
set( ${CobaltLib_NAME}_VERSION_PATCH 0 )


############################################################
# External Definitions
#set( Cobalt_BACKEND)
#set( Cobalt_DIR_PSM "" )
#set( Cobalt_DIR_GENERATED ${CMAKE_BINARY_DIR}/CobaltGenerated )

############################################################
# Cobalt Headers needed for all subprojects
#set( CobaltLib_DIR_INCLUDE ${CMAKE_SOURCE_DIR}/CobaltLib/include )
#set( CobaltLib_PUBLIC_HEADERS ${CobaltLib_DIR_INCLUDE}/Cobalt.h )
#source_group(API FILES ${CobaltLib_PUBLIC_HEADERS} )


############################################################
# CobaltLib source files
set( CobaltLib_SRC_STATIC
  src/Cobalt.cpp
  src/Logger.cpp
  src/Logger.h
  src/ReferenceTensorContraction.cpp
  src/ReferenceTensorContraction.h
  src/Solution.cpp
  src/Solution.h
  src/StructOperations.cpp
  src/StructOperations.h
)

############################################################
# CobaltLib log-only mode
# - don't generate any files
if ( Cobalt_BUILD_MODE MATCHES "Cobalt_BUILD_MODE_LOGGER")
  add_definitions( -DCobalt_LOGGER_ENABLED=1)
  add_definitions( -DCobalt_SOLVER_ENABLED=0)
  source_group( source FILES ${CobaltLib_SRC_STATIC} )

else()
################################################################################
# CobaltLib log-only mode
# - generate static and dynamic any files
  add_definitions( -DCobalt_LOGGER_ENABLED=1)
  add_definitions( -DCobalt_SOLVER_ENABLED=1)

  set( CobaltLib_SRC_GENERATED_STATIC
    ${Cobalt_DIR_GENERATED}/Kernels/CobaltKernels.h
    ${Cobalt_DIR_GENERATED}/Solutions/CobaltSolutions.h
    ${Cobalt_DIR_GENERATED}/Library/CobaltGetSolution.cpp
    ${Cobalt_DIR_GENERATED}/Library/CobaltGetSolution.h
  )
  #set( CobaltLib_SRC_GENERATED_DYNAMIC
    # needs to be put into another .cmake file
  #)
  set( CobaltLib_CMAKE_DYNAMIC
    ${Cobalt_DIR_GENERATED}/Library/CobaltLib.cmake
  )
  if( NOT EXISTS ${CobaltLib_CMAKE_DYNAMIC})
    file( WRITE ${CobaltLib_CMAKE_DYNAMIC}
      "include( \${Cobalt_KernelFiles_CMAKE_DYNAMIC} )\n"
      "include( \${Cobalt_SolutionFiles_CMAKE_DYNAMIC} )\n"
      "set( CobaltLib_SRC_GENERATED_DYNAMIC\n"
      "  \${Cobalt_KernelFiles_GENERATED_DYNAMIC}\n"
      "  \${Cobalt_KernelFiles_GENERATED_DYNAMIC} )\n"
    )
  endif()
  include( ${CobaltLib_CMAKE_DYNAMIC} )

  ############################################################
  # CobaltGenBackend.py - generates full library backend files
  add_custom_command(
    OUTPUT
      ${CobaltLib_SRC_GENERATED_STATIC}
      ${CobaltLib_SRC_GENERATED_DYNAMIC}
      ${CobaltLib_CMAKE_DYNAMIC}
    COMMAND
      python ${CMAKE_SOURCE_DIR}/CobaltGen/CobaltGenBackend.py
      --backend=${Cobalt_BACKEND}
      --input-path=${Cobalt_DIR_PSM}
      --output-path=${Cobalt_DIR_GENERATED}
    DEPENDS ${CobaltGenBackend_SRC}
  )
  source_group(
    source FILES
    ${CobaltLib_SRC_STATIC}
    ${CobaltLib_SRC_GENERATED_STATIC}
    ${CobaltLib_SRC_GENERATED_DYNAMIC}
  )
  source_group( CobaltGenBackend FILES ${CobaltGenBackend_SRC} )
endif()
add_definitions(-DLOG_FILE_PREFIX="${Cobalt_FILE_PROBLEMS}")



############################################################
# Cobalt include directories
include_directories(
  src
  ${CobaltLib_DIRS_INCLUDE}
  ${Cobalt_DIR_GENERATED}
  ${Cobalt_DIR_GENERATED}/Library
  ${Cobalt_DIR_GENERATED}/Kernels
  ${Cobalt_DIR_GENERATED}/Solutions
  )

############################################################
# build library
add_library( ${CobaltLib_NAME}
  ${CobaltLib_API}
  ${CobaltLib_SRC_STATIC}
  ${CobaltLib_SRC_GENERATED_STATIC}
  ${CobaltLib_SRC_GENERATED_DYNAMIC}
  )

############################################################
# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  include_directories( ${OpenCL_INCLUDE_DIR} )
  target_link_libraries( ${CobaltLib_NAME} ${OpenCL_LIBRARY} )
endif()


