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

if(NOT DEFINED Tensile_ROOT)
# Compute the installation prefix relative to this file.
get_filename_component(Tensile_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(Tensile_PREFIX "${Tensile_PREFIX}" PATH)

execute_process(COMMAND "${Tensile_PREFIX}/bin/TensileGetPath" OUTPUT_VARIABLE Tensile_ROOT)
endif()
list(APPEND CMAKE_MODULE_PATH "${Tensile_ROOT}/Source/cmake/")
list(APPEND CMAKE_MODULE_PATH "${Tensile_ROOT}/Source/")

if("HIP" IN_LIST Tensile_FIND_COMPONENTS)
    set(TENSILE_USE_HIP ON CACHE BOOL "Use HIP")
else()
    set(TENSILE_USE_HIP OFF CACHE BOOL "Use HIP")
endif()

if("LLVM" IN_LIST Tensile_FIND_COMPONENTS)
    set(TENSILE_USE_LLVM ON CACHE BOOL "Use LLVM")
else()
    set(TENSILE_USE_LLVM OFF CACHE BOOL "Use LLVM")
endif()

if("Client" IN_LIST Tensile_FIND_COMPONENTS)
    if(TENSILE_USE_HIP AND TENSILE_USE_LLVM)
        set(TENSILE_BUILD_CLIENT ON CACHE BOOL "Build Client")
    elseif(Tensile_FIND_REQUIRED_Client)
        message("Tensile client requires both Hip and LLVM.")
        set(Tensile_FOUND false)
    else()
        set(TENSILE_BUILD_CLIENT OFF CACHE BOOL "Build Client")
    endif()
else()
    set(TENSILE_BUILD_CLIENT OFF CACHE BOOL "Build Client")
endif()

if("STATIC_ONLY" IN_LIST Tensile_FIND_COMPONENTS)
    set(TENSILE_STATIC_ONLY ON CACHE BOOL "Disable exporting symbols from shared library.")
else()
    set(TENSILE_STATIC_ONLY OFF CACHE BOOL "Disable exporting symbols from shared library.")
endif()

add_subdirectory("${Tensile_ROOT}/Source" "Tensile")
include("${Tensile_ROOT}/Source/TensileCreateLibrary.cmake")

function(TensileCreateLibraryFiles
        Tensile_LOGIC_PATH Tensile_OUTPUT_PATH)

  # Tensile_ROOT can be specified instead of using the installed path.
  set(options NO_MERGE_FILES SHORT_FILE_NAMES)
  set(oneValueArgs TENSILE_ROOT EMBED_LIBRARY EMBED_KEY VAR_PREFIX Tensile_CODE_OBJECT_VERSION Tensile_COMPILER Tensile_ARCHITECTURE)
  set(multiValueArgs "")
  cmake_parse_arguments(Tensile "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(Script "${Tensile_ROOT}/bin/TensileCreateLibrary")
  message(STATUS "Tensile script: ${Script}")

  set(Options "")

  if(Tensile_NO_MERGE_FILES)
    set(Options ${Options} "--no-merge-files")
  else()
    set(Options ${Options} "--merge-files")
  endif()

  if(Tensile_SHORT_FILE_NAMES)
    set(Options ${Options} "--short-file-names")
  else()
    set(Options ${Options} "--no-short-file-names")
  endif()

  if(Tensile_EMBED_LIBRARY STREQUAL "")
  else()
    set(Options ${Options} "--embed-library=${Tensile_EMBED_LIBRARY}")
  endif()

  if(Tensile_EMBED_KEY STREQUAL "")
  else()
    set(Options ${Options} "--embed-library-key=${Tensile_EMBED_KEY}")
  endif()

  if(Tensile_CODE_OBJECT_VERSION STREQUAL "")
    set(Options ${Options} "--code-object-version=V2")
  else()
    set(Options ${Options} "--code-object-version=${Tensile_CODE_OBJECT_VERSION}")
  endif()
  
  if(Tensile_COMPILER STREQUAL "")
    set(Options ${Options} "--cxx-compiler=hcc")
  else()
    set(Options ${Options} "--cxx-compiler=${Tensile_COMPILER}")
  endif()
  
  if(Tensile_ARCHITECTURE STREQUAL "")
    set(Options ${Options} "--architecture=all")
  else()
    set(Options ${Options} "--architecture=${Tensile_ARCHITECTURE}")
  endif()
  
  set(CommandLine ${Script} ${Options} ${Tensile_LOGIC_PATH} ${Tensile_OUTPUT_PATH} HIP)
  message(STATUS "Tensile_CREATE_COMMAND: ${CommandLine}")

  if($ENV{TENSILE_SKIP_LIBRARY})
      message(STATUS "Skipping build of ${Tensile_OUTPUT_PATH}")
  else()
      execute_process(COMMAND ${CommandLine} RESULT_VARIABLE CommandResult)
      if(CommandResult)
        message(FATAL_ERROR "Error creating Tensile library: ${CommandResult}")
      endif()
  endif()

  if(Tensile_VAR_PREFIX STREQUAL "")
      set(Tensile_VAR_PREFIX TENSILE)
  endif()

  file(GLOB CodeObjects "${Tensile_OUTPUT_PATH}/library/*.co")
  file(GLOB HSACodeObjects "${Tensile_OUTPUT_PATH}/library/*.hsaco")
  set(LibraryFile "${Tensile_OUTPUT_PATH}/library/TensileLibrary.yaml")

  set("${Tensile_VAR_PREFIX}_CODE_OBJECTS" ${CodeObjects} ${HSACodeObjects} PARENT_SCOPE)
  set("${Tensile_VAR_PREFIX}_LIBRARY_FILE" "${LibraryFile}" PARENT_SCOPE)

  set("${Tensile_VAR_PREFIX}_ALL_FILES" ${CodeObjects} ${HSACodeObjects} ${LibraryFile} PARENT_SCOPE)

  if(Tensile_EMBED_LIBRARY STREQUAL "")
  else()

    add_library(${Tensile_EMBED_LIBRARY} "${Tensile_OUTPUT_PATH}/library/${Tensile_EMBED_LIBRARY}.cpp")
    target_link_libraries(${Tensile_EMBED_LIBRARY} PUBLIC TensileHost)

  endif()

endfunction()


