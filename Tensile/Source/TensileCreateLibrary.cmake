################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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

################################################################################
# Create A Tensile Library from LibraryLogic.yaml files
################################################################################
function(TensileCreateLibraryCmake
    Tensile_LOGIC_PATH
    Tensile_RUNTIME_LANGUAGE
    Tensile_COMPILER
    Tensile_CODE_OBJECT_VERSION
    Tensile_ARCHITECTURE
    Tensile_LIBRARY_FORMAT
    Tensile_MERGE_FILES
    Tensile_SHORT_FILE_NAMES
    Tensile_LIBRARY_PRINT_DEBUG )

# make Tensile_PACKAGE_LIBRARY and optional parameter
# to avoid breaking applications which us this
  if (ARGN)
    list (GET ARGN 0 Tensile_PACKAGE_LIBRARY)
    list (GET ARGN 1 Tensile_INCLUDE_LEGACY_CODE)
  else()
    set(Tensile_PACKAGE_LIBRARY OFF)
    set(Tensile_INCLUDE_LEGACY_CODE ON)
  endif()

  set(options PACKAGE_LIBRARY Tensile_INCLUDE_LEGACY_CODE)
  message(STATUS "Tensile_RUNTIME_LANGUAGE    from TensileCreateLibraryCmake : ${Tensile_RUNTIME_LANGUAGE}")
  message(STATUS "Tensile_CODE_OBJECT_VERSION from TensileCreateLibraryCmake : ${Tensile_CODE_OBJECT_VERSION}")
  message(STATUS "Tensile_COMPILER            from TensileCreateLibraryCmake : ${Tensile_COMPILER}")
  message(STATUS "Tensile_ARCHITECTURE        from TensileCreateLibraryCmake : ${Tensile_ARCHITECTURE}")
  message(STATUS "Tensile_LIBRARY_FORMAT      from TensileCreateLibraryCmake : ${Tensile_LIBRARY_FORMAT}")

  execute_process(COMMAND chmod 755 ${Tensile_ROOT}/bin/TensileCreateLibrary)
  execute_process(COMMAND chmod 755 ${Tensile_ROOT}/bin/Tensile)

  set(Tensile_CREATE_COMMAND "${Tensile_ROOT}/bin/TensileCreateLibrary")

  set(Tensile_SOURCE_PATH "${PROJECT_BINARY_DIR}/Tensile")
  message(STATUS "Tensile_SOURCE_PATH=${Tensile_SOURCE_PATH}")

  # TensileLibraryWriter optional arguments
  if(${Tensile_MERGE_FILES})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--merge-files")
  else()
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--no-merge-files")
  endif()

  if(${Tensile_PACKAGE_LIBRARY})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--package-library")
  endif()

  if( NOT ${Tensile_INCLUDE_LEGACY_CODE})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--no-legacy-components")
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

  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--architecture=${Tensile_ARCHITECTURE}")
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--code-object-version=${Tensile_CODE_OBJECT_VERSION}")
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--cxx-compiler=${Tensile_COMPILER}")
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--library-format=${Tensile_LIBRARY_FORMAT}")

  # TensileLibraryWriter positional arguments
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND}
    ${Tensile_LOGIC_PATH}
    ${Tensile_SOURCE_PATH}
    ${Tensile_RUNTIME_LANGUAGE}
    )

  #string( REPLACE ";" " " Tensile_CREATE_COMMAND "${Tensile_CREATE_COMMAND}")
  message(STATUS "Tensile_CREATE_COMMAND: ${Tensile_CREATE_COMMAND}")

  # execute python command
  if($ENV{TENSILE_SKIP_LIBRARY})
    message(STATUS "Skipping build of ${Tensile_OUTPUT_PATH}")
  else()
    execute_process(
      COMMAND ${Tensile_CREATE_COMMAND}
      RESULT_VARIABLE Tensile_CREATE_RESULT
    )
    if(Tensile_CREATE_RESULT)
      message(FATAL_ERROR "Error generating kernels")
    endif()
  endif()

  if ( ${Tensile_INCLUDE_LEGACY_CODE} )
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
  endif()

  if ( ${Tensile_INCLUDE_LEGACY_CODE} )
    # create Tensile Library
    set(options)
    add_library(Tensile ${options} ${Tensile_SOURCE_FILES})
    # specify gpu targets
    if( Tensile_ARCHITECTURE MATCHES "all" )
      set( Tensile_HIP_ISA "gfx803" "gfx900" "gfx906" "gfx908")
    else()
      set( Tensile_HIP_ISA ${Tensile_ARCHITECTURE})
    endif()
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
  endif()

  if ( ${Tensile_INCLUDE_LEGACY_CODE} )
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
  endif()

endfunction()
