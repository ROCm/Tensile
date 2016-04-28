################################################################################
# CobaltLib
# project to generate Cobat.lib (full backend) from ProbSolMap.xml
################################################################################
#cmake_minimum_required(VERSION 2.8)
#project( ${CobaltLib_NAME} C CXX)
#set( ${CobaltLib_NAME}_VERSION_MAJOR 0 )
#set( ${CobaltLib_NAME}_VERSION_MINOR 1 )
#set( ${CobaltLib_NAME}_VERSION_PATCH 0 )

if( NOT EXISTS ${CobaltLib_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl)
  MESSAGE( "Creating dummy SolutionTemplateInstantiations.inl for CobaltLib")
  file( WRITE ${CobaltLib_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl
      ""
  )
endif()


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
    ${CobaltLib_DIR_GENERATED}/Kernels/CobaltKernels.h
    ${CobaltLib_DIR_GENERATED}/Solutions/CobaltSolutions.h
    ${CobaltLib_DIR_GENERATED}/Other/CobaltGetSolution.cpp
    ${CobaltLib_DIR_GENERATED}/Other/CobaltGetSolution.h
  )
  #set( CobaltLib_SRC_GENERATED_DYNAMIC
    # needs to be put into another .cmake file
  #)
  set( CobaltLib_CMAKE_DYNAMIC
    ${CobaltLib_DIR_GENERATED}/Other/CobaltLib.cmake
  )
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

  ############################################################
  # CobaltGenBackend.py - generates full library backend files
  add_custom_command(
    OUTPUT
      ${CobaltLib_SRC_GENERATED_STATIC}
      ${CobaltLib_SRC_GENERATED_DYNAMIC}
      ${CobaltLib_CMAKE_DYNAMIC}
    COMMAND
      python -m cProfile ${CMAKE_SOURCE_DIR}/CobaltGen/CobaltGenBackend.py
      --backend=${Cobalt_BACKEND}
      --input-path=${CobaltBenchmark_DIR_SOLUTIONS}
      --output-path=${CobaltLib_DIR_GENERATED}
    DEPENDS ${CobaltGen_SRC}
  )
endif()

add_definitions(-DCobaltLib_DIR_PROBLEMS=\\\"${CobaltLib_DIR_PROBLEMS}\\\")
add_definitions(-DCobaltLib_DIR_SOLUTIONS=\\\"${CobaltLib_DIR_SOLUTIONS}\\\")
add_definitions(-DCobaltLib_DIR_GENERATED=\\\"${CobaltLib_DIR_GENERATED}\\\")



############################################################
# Cobalt include directories
include_directories(
  src
  ${CobaltLib_DIRS_INCLUDE}
  ${CobaltLib_DIR_GENERATED}
  ${CobaltLib_DIR_GENERATED}/Kernels
  ${CobaltLib_DIR_GENERATED}/Solutions
  ${CobaltLib_DIR_GENERATED}/Other
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
#set_property(TARGET ${CobaltLib_NAME} PROPERTY CXX_STANDARD 11 )

source_group(CobaltGen\\Scripts FILES ${CobaltGen_SRC})
source_group(CobaltGen\\Generated FILES
  ${CobaltLib_SRC_GENERATED_STATIC}
  ${CobaltLib_SRC_GENERATED_DYNAMIC}
)

############################################################
# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  MESSAGE( "CobaltLib linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  target_link_libraries( ${CobaltLib_NAME} ${OPENCL_LIBRARIES} )
elseif( Cobalt_BACKEND MATCHES "HIP")
  MESSAGE( "CobaltLib linking HIP: ${HCC_INCLUDE_DIRS}" )
  include_directories( ${HCC_INCLUDE_DIRS} )
  target_link_libraries( ${CobaltLib_NAME} ${HSA_LIBRARIES} )
endif()


