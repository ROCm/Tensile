################################################################################
# CobaltLogger & CobaltLib
################################################################################

set( CobaltLib_NAME Cobalt )
set( CobaltLogger_NAME CobaltLogger )

# OpenCL 1.2
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  find_package(OpenCL "1.2" REQUIRED)
  message(STATUS "${Cobalt_BACKEND} backend selected." )
  set( CobaltLib_NAME "${CobaltLib_NAME}-ocl" )
  set( CobaltLogger_NAME "${CobaltLogger_NAME}-ocl" )

# HIP
elseif( Cobalt_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  message(STATUS "${Cobalt_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
  if( ${HIP_PLATFORM} MATCHES "hcc" )
    set( CobaltLib_NAME "${CobaltLib_NAME}-hcc" )
    set( CobaltLogger_NAME "${CobaltLogger_NAME}-hcc" )
  else( )
    set( CobaltLib_NAME "${CobaltLib_NAME}-cuda" )
    set( CobaltLogger_NAME "${CobaltLogger_NAME}-cuda" )
  endif( )
endif( )

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

# this may need to check Windows vs Linux rather than opencl vs hip
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  add_definitions(-DCobaltLib_DIR_PROBLEMS="${CobaltLib_DIR_PROBLEMS}")
  add_definitions(-DCobaltLib_DIR_SOLUTIONS="${CobaltLib_DIR_SOLUTIONS}")
  add_definitions(-DCobaltLib_DIR_GENERATED="${CobaltLib_DIR_GENERATED}")
elseif( Cobalt_BACKEND MATCHES "HIP")
  add_definitions(-DCobaltLib_DIR_PROBLEMS=\\\"${CobaltLib_DIR_PROBLEMS}\\\")
  add_definitions(-DCobaltLib_DIR_SOLUTIONS=\\\"${CobaltLib_DIR_SOLUTIONS}\\\")
  add_definitions(-DCobaltLib_DIR_GENERATED=\\\"${CobaltLib_DIR_GENERATED}\\\")
endif()

################################################################################
# CobaltLogger
################################################################################


############################################################
# CobaltLogger
add_library( ${CobaltLogger_NAME}
  ${CobaltLib_API}
  ${CobaltLib_SRC_STATIC}
  )

target_include_directories( ${CobaltLogger_NAME}
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
)

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( ${CobaltLogger_NAME} PRIVATE opencl )
  target_compile_definitions( ${CobaltLogger_NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=1 -DCobalt_BACKEND_HIP=0 )
elseif( Cobalt_BACKEND MATCHES "HIP")
  target_include_directories( ${CobaltLogger_NAME}
    PUBLIC  $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
#  target_link_libraries( ${CobaltLogger_NAME} PRIVATE ${HSA_LIBRARIES} )
  target_link_libraries( ${CobaltLogger_NAME} PUBLIC $<BUILD_INTERFACE:${HSA_LIBRARIES}> )
  target_compile_definitions( ${CobaltLogger_NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=0 -DCobalt_BACKEND_HIP=1 )
endif()

target_compile_definitions( ${CobaltLogger_NAME} PRIVATE -DCobalt_SOLVER_ENABLED=0 -DCobalt_LOGGER_ENABLED=1 )

set_target_properties( ${CobaltLogger_NAME} PROPERTIES VERSION ${CobaltProj_VERSION} )
set_target_properties( ${CobaltLogger_NAME} PROPERTIES DEBUG_POSTFIX "-d" )
set_target_properties( ${CobaltLogger_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

################################################################################
# CobaltLib
################################################################################

# generated src files
set( CobaltLib_SRC_GENERATED_STATIC
  ${CobaltLib_DIR_GENERATED}/Kernels/CobaltKernels.h
  ${CobaltLib_DIR_GENERATED}/Solutions/CobaltSolutions.h
  ${CobaltLib_DIR_GENERATED}/Other/CobaltGetSolution.cpp
  ${CobaltLib_DIR_GENERATED}/Other/CobaltGetSolution.h
)

#include( ${CobaltLib_CMAKE_DYNAMIC} )


################################################################################
# CobaltLib log-only mode
# - generate static and dynamic any files


############################################################
# CobaltGenBackend.py - generates full library backend files
find_package( PythonInterp REQUIRED )

add_custom_command(
  OUTPUT
    ${CobaltLib_SRC_GENERATED_STATIC}
    ${CobaltLib_SRC_GENERATED_DYNAMIC}
    ${CobaltLib_CMAKE_DYNAMIC}
  COMMAND
    python ${CMAKE_SOURCE_DIR}/CobaltGen/CobaltGenBackend.py
    --backend=${Cobalt_BACKEND}
    --input-path=${Cobalt_DIR_SOLUTIONS}
    --output-path=${CobaltLib_DIR_GENERATED}
    --optimize-alpha=${Cobalt_OPTIMIZE_ALPHA}
    --optimize-beta=${Cobalt_OPTIMIZE_BETA}
  DEPENDS ${CobaltGen_SRC} ${Cobalt_XML_SOLUTIONS_FILE}
)

set( CobaltLib_SRC_GENERATED_DYNAMIC
    ${CobaltLib_SolutionFiles_GENERATED_DYNAMIC}
    ${CobaltLib_KernelFiles_GENERATED_DYNAMIC}
    ${CobaltLib_OtherFiles_GENERATED_DYNAMIC}
)

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
# CobaltLib
add_library( ${CobaltLib_NAME}
  ${CobaltLib_API}
  ${CobaltLib_SRC_STATIC}
  ${CobaltLib_SRC_GENERATED_STATIC}
  ${CobaltLib_SRC_GENERATED_DYNAMIC}
  ${CobaltGen_SRC}
  )

target_include_directories( ${CobaltLib_NAME}
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
)

set_target_properties( ${CobaltLib_NAME} PROPERTIES VERSION ${CobaltProj_VERSION} )
set_target_properties( ${CobaltLib_NAME} PROPERTIES DEBUG_POSTFIX "-d" )
set_target_properties( ${CobaltLib_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

#set_property(TARGET ${CobaltLib_NAME} PROPERTY CXX_STANDARD 11 )
if( ${Cobalt_ENABLE_LOGGER} )
  target_compile_definitions( ${CobaltLib_NAME} PRIVATE -DCobalt_SOLVER_ENABLED=1 -DCobalt_LOGGER_ENABLED=1 )
else()
  target_compile_definitions( ${CobaltLib_NAME} PRIVATE -DCobalt_SOLVER_ENABLED=1 -DCobalt_LOGGER_ENABLED=0 )
endif()

source_group(CobaltGen\\Scripts FILES ${CobaltGen_SRC})
source_group(CobaltGen\\Generated FILES
  ${CobaltLib_SRC_GENERATED_STATIC}
  ${CobaltLib_SRC_GENERATED_DYNAMIC}
)

############################################################
# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( ${CobaltLib_NAME} PRIVATE opencl )
  target_compile_definitions( ${CobaltLib_NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=1 -DCobalt_BACKEND_HIP=0 )
elseif( Cobalt_BACKEND MATCHES "HIP")
  message( STATUS "CobaltLib linking HIP: ${HCC_INCLUDE_DIRS}" )
  target_include_directories( ${CobaltLib_NAME}
    PUBLIC  $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( ${CobaltLib_NAME} PUBLIC $<BUILD_INTERFACE:${HSA_LIBRARIES}> )
  target_compile_definitions( ${CobaltLib_NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=0 -DCobalt_BACKEND_HIP=1 )
endif()

############################################################
# Library install logic

############################################################
# The following is cmake code to generate a config file package for Cobalt
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( CMakePackageConfigHelpers )

set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )
set( config_package_location ${LIB_INSTALL_DIR}/cmake/cobalt )

configure_package_config_file(
  cobalt-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cobalt-config.cmake
  INSTALL_DESTINATION ${config_package_location}
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/cobalt-config-version.cmake
  VERSION ${CobaltProj_VERSION}
  COMPATIBILITY SameMajorVersion
)

install( TARGETS ${CobaltLib_NAME} ${CobaltLogger_NAME}
  EXPORT cobalt-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install( EXPORT cobalt-targets
  DESTINATION
    ${config_package_location}
)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cobalt-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cobalt-config-version.cmake
  DESTINATION
    ${config_package_location} )

install( FILES
    include/Cobalt.h
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR} )
