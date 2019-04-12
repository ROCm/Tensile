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

include(CMakeParseArguments)

# Compute the installation prefix relative to this file.
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
if(_IMPORT_PREFIX STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()

set(Tensile_ROOT ${_IMPORT_PREFIX})

function(tensile_add_library LIBRARY_NAME)
  set(options SHARED STATIC MERGE_FILES SHORT_FILE_NAMES LIBRARY_PRINT_DEBUG)
  set(oneValueArgs SOURCE_PATH RUNTIME)
  set(multiValueArgs)

  cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(Tensile_CREATE_COMMAND "${Tensile_ROOT}/bin/TensileCreateLibrary")
  set(Tensile_LOGIC_PATH ${PARSE_UNPARSED_ARGUMENTS})

  set(Tensile_RUNTIME_LANGUAGE HIP)
  if(PARSE_RUNTIME)
    set(Tensile_RUNTIME_LANGUAGE ${PARSE_RUNTIME})
  endif()

  if(PARSE_SOURCE_PATH)
    set(Tensile_SOURCE_PATH "${PARSE_SOURCE_PATH}")
  else()
    set(Tensile_SOURCE_PATH "${PROJECT_BINARY_DIR}/Tensile")
  endif()
  message(STATUS "Tensile_SOURCE_PATH=${Tensile_SOURCE_PATH}")

  # TensileLibraryWriter optional arguments
  if(PARSE_MERGE_FILES)
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--merge-files")
  else()
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--no-merge-files")
  endif()

  if(PARSE_SHORT_FILE_NAMES)
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--short-file-names")
  else()
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--no-short-file-names")
  endif()

  if(PARSE_LIBRARY_PRINT_DEBUG)
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--library-print-debug")
  else()
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--no-library-print-debug")
  endif()

  # TensileLibraryWriter positional arguments
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND}
    ${Tensile_LOGIC_PATH}
    ${Tensile_SOURCE_PATH}
    ${Tensile_RUNTIME_LANGUAGE}
    )

  #string( REPLACE ";" " " Tensile_CREATE_COMMAND "${Tensile_CREATE_COMMAND}")
  message(STATUS "Tensile_CREATE_COMMAND: ${Tensile_CREATE_COMMAND}")

  # execute python command
  execute_process(
    COMMAND ${Tensile_CREATE_COMMAND}
    RESULT_VARIABLE Tensile_CREATE_RESULT
  )
  if(Tensile_CREATE_RESULT)
    message(SEND_ERROR "Error generating kernels")
  endif()

  # glob generated source files
  if(PARSE_MERGE_FILES)
    file(GLOB Tensile_SOURCE_FILES
      ${Tensile_SOURCE_PATH}/*.cpp
      )
  else()
    file(GLOB Tensile_SOURCE_FILES
      ${Tensile_SOURCE_PATH}/*.cpp
      ${Tensile_SOURCE_PATH}/Kernels/*.cpp
      ${Tensile_SOURCE_PATH}/Solutions/*.cpp
      ${Tensile_SOURCE_PATH}/Logic/*.cpp
      )
  endif()

  # create Tensile Library
  set(options)
  if(PARSE_SHARED)
    set(options SHARED)
  endif()
  if(PARSE_STATIC)
    set(options STATIC)
  endif()
  add_library(${LIBRARY_NAME} ${options} ${Tensile_SOURCE_FILES})

  if(PARSE_MERGE_FILES)
    target_include_directories(${LIBRARY_NAME}
      PUBLIC $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}> )
  else()
    target_include_directories(${LIBRARY_NAME} PUBLIC
      $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}>
      $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}/Kernels>
      $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}/Solutions>
      $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}/Logic>
      $<INSTALL_INTERFACE:include> )
  endif()

  # define language for library source
  if( Tensile_RUNTIME_LANGUAGE MATCHES "OCL")
    target_compile_definitions( ${LIBRARY_NAME} PUBLIC
      -DTensile_RUNTIME_LANGUAGE_OCL=1 -DTensile_RUNTIME_LANGUAGE_HIP=0 )
  else()
    target_compile_definitions( ${LIBRARY_NAME} PUBLIC
      -DTensile_RUNTIME_LANGUAGE_OCL=0 -DTensile_RUNTIME_LANGUAGE_HIP=1 )
  endif()

endfunction()

################################################################################
# Create A Tensile Library from LibraryLogic.yaml files
################################################################################
function(TensileCreateLibrary
    Tensile_LOGIC_PATH
    Tensile_RUNTIME_LANGUAGE
    Tensile_MERGE_FILES
    Tensile_SHORT_FILE_NAMES
    Tensile_LIBRARY_PRINT_DEBUG )

  # Tensile_ROOT can be specified instead of installing
  set(oneValueArgs Tensile_ROOT)
  cmake_parse_arguments(PARSE "" "${oneValueArgs}" "" ${ARGN})

  if(PARSE_Tensile_ROOT)
    # python not pre-installed, use scripts downloaded to extern/Tensile
    include(FindPythonInterp)
    set(Tensile_CREATE_COMMAND ${PYTHON_EXECUTABLE} "${Tensile_ROOT}/Tensile/TensileCreateLibrary.py")
  else()
    set(Tensile_CREATE_COMMAND TensileCreateLibrary)
  endif()


  set(Tensile_SOURCE_PATH "${PROJECT_BINARY_DIR}/Tensile")
  message(STATUS "Tensile_SOURCE_PATH=${Tensile_SOURCE_PATH}")

  # TensileLibraryWriter optional arguments
  if(${Tensile_MERGE_FILES})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--merge-files")
  else()
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--no-merge-files")
  endif()

  if(${Tensile_SHORT_FILE_NAMES})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--short-file-names")
  else()
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--no-short-file-names")
  endif()

  if(${Tensile_LIBRARY_PRINT_DEBUG})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--library-print-debug")
  else()
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--no-library-print-debug")
  endif()

  # TensileLibraryWriter positional arguments
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND}
    ${Tensile_LOGIC_PATH}
    ${Tensile_SOURCE_PATH}
    ${Tensile_RUNTIME_LANGUAGE}
    )

  #string( REPLACE ";" " " Tensile_CREATE_COMMAND "${Tensile_CREATE_COMMAND}")
  message(STATUS "Tensile_CREATE_COMMAND: ${Tensile_CREATE_COMMAND}")

  # execute python command
  execute_process(
    COMMAND ${Tensile_CREATE_COMMAND}
    RESULT_VARIABLE Tensile_CREATE_RESULT
  )
  if(Tensile_CREATE_RESULT)
    message(SEND_ERROR "Error generating kernels")
  endif()

  # glob generated source files
  if( Tensile_MERGE_FILES )
    file(GLOB Tensile_SOURCE_FILES
      ${Tensile_SOURCE_PATH}/*.cpp
      )
  else()
    file(GLOB Tensile_SOURCE_FILES
      ${Tensile_SOURCE_PATH}/*.cpp
      ${Tensile_SOURCE_PATH}/Kernels/*.cpp
      ${Tensile_SOURCE_PATH}/Solutions/*.cpp
      ${Tensile_SOURCE_PATH}/Logic/*.cpp
      )
  endif()

  # create Tensile Library
  set(options)
  add_library(Tensile ${options} ${Tensile_SOURCE_FILES})
  # specify gpu targets
  set(Tensile_HIP_ISA "gfx803" "gfx900" "gfx906")
  foreach( target ${Tensile_HIP_ISA} )
    target_link_libraries( Tensile PRIVATE --amdgpu-target=${target} )
  endforeach()
  if( Tensile_MERGE_FILES )
    target_include_directories(Tensile
      PUBLIC $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}> )
  else()
    target_include_directories(Tensile PUBLIC
      $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}>
      $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}/Kernels>
      $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}/Solutions>
      $<BUILD_INTERFACE:${Tensile_SOURCE_PATH}/Logic>
      $<INSTALL_INTERFACE:include> )
  endif()

  # define language for library source
  if( Tensile_RUNTIME_LANGUAGE MATCHES "OCL")
    #find_package(OpenCL "1.2" REQUIRED)
    target_link_libraries( Tensile ${OPENCL_LIBRARIES} )
    target_compile_definitions( Tensile PUBLIC
      -DTensile_RUNTIME_LANGUAGE_OCL=1 -DTensile_RUNTIME_LANGUAGE_HIP=0 )
    target_include_directories( Tensile SYSTEM
      PUBLIC  ${OPENCL_INCLUDE_DIRS} ) 
  else()
    #find_package( HIP REQUIRED )
    set (CMAKE_CXX_COMPILER ${HIPCC})
    target_include_directories( Tensile SYSTEM
      PUBLIC  ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS} )
    target_link_libraries( Tensile PUBLIC ${HSA_LIBRARIES} )
    target_compile_definitions( Tensile PUBLIC
      -DTensile_RUNTIME_LANGUAGE_OCL=0 -DTensile_RUNTIME_LANGUAGE_HIP=1 )
  endif()

endfunction()


