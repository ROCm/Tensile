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
# CobaltLib source files
set( CobaltLib_SRC_STATIC
  src/Cobalt.cpp
  src/DeviceProfile.cpp
  src/DeviceProfile.h
  src/Logger.cpp
  src/Logger.h
  src/MathTemplates.cpp
  src/MathTemplates.h
  src/Problem.cpp
  src/Problem.h
  src/SolutionTensorContractionCPU.cpp
  src/SolutionTensorContractionCPU.h
  src/Solution.cpp
  src/Solution.h
  src/StructOperations.cpp
  src/StructOperations.h
  src/Tensor.cpp
  src/Tensor.h
  src/Tools.cpp
  src/Tools.h
)

############################################################
# CobaltLib log-only mode
# - don't generate any files
if ( Cobalt_BUILD_MODE MATCHES "LOG_ONLY")
  add_definitions( -DCobalt_LOGGER_ENABLED=1)
  add_definitions( -DCobalt_SOLVER_ENABLED=0)

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
      --input-path=${Cobalt_DIR_SOLUTIONS}
      --output-path=${Cobalt_DIR_GENERATED}
    DEPENDS ${CobaltGen_SRC}
  )
endif()

add_definitions(-DCobalt_DIR_PROBLEMS="${Cobalt_DIR_PROBLEMS}")
add_definitions(-DCobalt_DIR_SOLUTIONS="${Cobalt_DIR_SOLUTIONS}")
add_definitions(-DCobalt_DIR_GENERATED="${Cobalt_DIR_GENERATED}")



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
  ${CobaltGen_SRC}
  )

source_group(CobaltGen\\Scripts FILES ${CobaltGen_SRC})
source_group(CobaltGen\\Generated FILES
  ${CobaltLib_SRC_GENERATED_STATIC}
  ${CobaltLib_SRC_GENERATED_DYNAMIC}
)

############################################################
# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  include_directories( ${OpenCL_INCLUDE_DIR} )
  target_link_libraries( ${CobaltLib_NAME} ${OpenCL_LIBRARY} )
endif()


