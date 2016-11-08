################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )

set( CobaltBenchmark_NAME CobaltBenchmark )

get_filename_component(Cobalt_DIR ${CMAKE_SOURCE_DIR} PATH)

set( CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${Cobalt_DIR}/cmake )

############################################################
# require C++11
if(MSVC)
  # object-level build parallelism for VS, not just target-level
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
  set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
endif()
add_definitions( "-std=c++11" )

include(EnableWarnings)

# OpenCL 1.2
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  find_package(OpenCL "1.2" REQUIRED)
  message(STATUS "${Cobalt_BACKEND} backend selected." )
# HIP
elseif( Cobalt_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  set (CMAKE_CXX_COMPILER ${HIPCC})
  message(STATUS "${Cobalt_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
endif( )


# source files
# Sets CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC
include(${CobaltBenchmark_DIR_GENERATED}/Kernels/CobaltKernels.cmake)
# Sets CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC
include(${CobaltBenchmark_DIR_GENERATED}/Solutions/CobaltSolutions.cmake)
# Sets CobaltBenchmark_SRC_GENERATED_DYNAMIC
include(${CobaltBenchmark_DIR_GENERATED}/Other/CobaltBenchmark.cmake)

# Glob CobaltLib source files
file(GLOB CobaltLib_SRC
    ${Cobalt_DIR}/CobaltLib/src/*.cpp
)

function( convert_to_linux_path return_list_of_files )
  set(new_list)
  foreach( file ${ARGN} )
    file(TO_CMAKE_PATH ${file} result)
    list(APPEND new_list ${result})
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

convert_to_linux_path(CobaltBenchmark_SRC
  ${CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC}
  ${CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC}
  ${CobaltBenchmark_SRC_GENERATED_DYNAMIC}
  ${CobaltBenchmark_DIR_GENERATED}/Other/CobaltSolutionCandidates.cpp
)

# benchmark executable
add_executable( ${CobaltBenchmark_NAME}
    CobaltBenchmark.cpp
    ${CobaltLib_SRC}
    ${CobaltBenchmark_SRC}
)
set_target_properties( ${CobaltBenchmark_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" 
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_BINARY_DIR}/bin" )

##################################################
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( ${CobaltBenchmark_NAME} PRIVATE opencl )
  target_compile_definitions( ${CobaltBenchmark_NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=1 -DCobalt_BACKEND_HIP=0 )
elseif( Cobalt_BACKEND MATCHES "HIP")
  target_include_directories( ${CobaltBenchmark_NAME} SYSTEM
    PUBLIC  ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS}
  )
#  target_link_libraries( ${CobaltBenchmark_NAME} PRIVATE ${HSA_LIBRARIES} )
  target_link_libraries( ${CobaltBenchmark_NAME} PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( ${CobaltBenchmark_NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=0 -DCobalt_BACKEND_HIP=1 )
endif()

# include Cobalt headers
target_include_directories( ${CobaltBenchmark_NAME}
  PUBLIC  $<BUILD_INTERFACE:${Cobalt_DIR}/CobaltLib/include>
          $<BUILD_INTERFACE:${Cobalt_DIR}/CobaltLib/src>
          $<BUILD_INTERFACE:${CobaltBenchmark_DIR_GENERATED}>
          $<BUILD_INTERFACE:${CobaltBenchmark_DIR_GENERATED}/Kernels>
          $<BUILD_INTERFACE:${CobaltBenchmark_DIR_GENERATED}/Solutions>
          $<BUILD_INTERFACE:${CobaltBenchmark_DIR_GENERATED}/Other>
)

############################################################
# CobaltBenchmark - generates ProblemSolutionMap.xml

target_compile_definitions( ${CobaltBenchmark_NAME} PRIVATE -DCobalt_SOLVER_ENABLED=0 -DCobalt_LOGGER_ENABLED=1 )
if (CMAKE_CXX_COMPILER MATCHES ".*hipcc")
    # hipcc is a pearl script, so it requires a lot of extra escaping
    target_compile_definitions(${CobaltBenchmark_NAME} PUBLIC -DCobalt_DIR_PROBLEMS=\\\"${CobaltBenchmark_DIR_GENERATED}\\\")
    # This does not need to be defined but will fail to compile when not
    target_compile_definitions(${CobaltBenchmark_NAME} PUBLIC -DCobalt_DIR_SOLUTIONS=\\\"${CobaltBenchmark_DIR_GENERATED}\\\")
else()
    file( TO_CMAKE_PATH ${CobaltBenchmark_DIR_GENERATED} CBDG_PATH)
    message(STATUS "CobaltBenchmark output dir: ${CBDG_PATH}")
    target_compile_definitions(${CobaltBenchmark_NAME} PUBLIC -DCobalt_DIR_PROBLEMS="${CBDG_PATH}")
    # This does not need to be defined but will fail to compile when not
    target_compile_definitions(${CobaltBenchmark_NAME} PUBLIC -DCobalt_DIR_SOLUTIONS="${CBDG_PATH}")
endif()
