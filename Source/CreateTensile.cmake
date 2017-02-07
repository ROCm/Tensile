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

function(CreateTensile
    Tensile_LOGIC_PATH
    Tensile_ROOT
    Tensile_BACKEND
    Tensile_MERGE_FILES
    Tensile_SHORT_FILE_NAMES
    Tensile_PRINT_DEBUG
    )
  message(STATUS "CreateTensile logic=${Tensile_LOGIC_PATH} root= ${Tensile_ROOT}")

  # create python command
  set(Tensile_SOURCE_PATH "${PROJECT_BINARY_DIR}/Tensile/${NAME}")
  set(Tensile_CREATE_COMMAND ${PYTHON_EXECUTABLE}
    ${Tensile_ROOT}/Scripts/LibraryWriter.py
    ${Tensile_LOGIC_PATH}
    ${Tensile_SOURCE_PATH}
    ${Tensile_BACKEND}
    ${Tensile_MERGE_FILES}
    ${Tensile_SHORT_FILE_NAMES}
    ${Tensile_PRINT_DEBUG}
    )
  string( REPLACE ";" " " Tensile_CREATE_COMMAND "${Tensile_CREATE_COMMAND}")
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

  # create Tensile
  set(options)
  add_library(Tensile ${options} ${Tensile_SOURCE_FILES})
  target_include_directories(${NAME}
      PUBLIC  $<BUILD_INTERFACE:${Tensile_DIR}/Tensile/include>
              $<BUILD_INTERFACE:${Tensile_DIR}/Tensile/src>
              $<BUILD_INTERFACE:${Tensile_DIR_GENERATED}>
              $<BUILD_INTERFACE:${Tensile_DIR_GENERATED}/Kernels>
              $<BUILD_INTERFACE:${Tensile_DIR_GENERATED}/Solutions>
              $<BUILD_INTERFACE:${Tensile_DIR_GENERATED}/Other>
              $<INSTALL_INTERFACE:include>
  )

  # define backend for library source
  if( Tensile_BACKEND MATCHES "OCL")
    target_compile_definitions( Tensile PUBLIC
      -DTensile_BACKEND_OPENCL12=1 -DTensile_BACKEND_HIP=0 )
  else()
    target_compile_definitions( Tensile PUBLIC
      -DTensile_BACKEND_OPENCL12=0 -DTensile_BACKEND_HIP=1 )
  endif()

endfunction()
