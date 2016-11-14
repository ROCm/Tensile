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

function(add_tensile_lib NAME)
    set(options OPTIMIZE_ALPHA OPTIMIZE_BETA ENABLE_LOGGER EXCLUDE_FROM_ALL)
    set(oneValueArgs SOLUTIONS PROBLEMS BACKEND)
    set(multiValueArgs)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TensileLib_DIR_GENERATED "${PROJECT_BINARY_DIR}/tensile/${NAME}")

    if(PARSE_OPTIMIZE_ALPHA)
        set(TensileGen_OPTIMIZE_ALPHA "ON")
    else()
        set(TensileGen_OPTIMIZE_ALPHA "OFF")
    endif()

    if(PARSE_OPTIMIZE_BETA)
        set(TensileGen_OPTIMIZE_BETA "ON")
    else()
        set(TensileGen_OPTIMIZE_BETA "OFF")
    endif()

    if(PARSE_SOLUTIONS)

        set(TensileLib_ENABLE_SOLVER 1)
        set(TensileGen_COMMAND ${PYTHON_EXECUTABLE} ${Tensile_DIR}/TensileGen/TensileGenBackend.py
            --backend=${PARSE_BACKEND}
            --input-path=${PARSE_SOLUTIONS}
            --output-path=${TensileLib_DIR_GENERATED}
            --optimize-alpha=${TensileGen_OPTIMIZE_ALPHA}
            --optimize-beta=${TensileGen_OPTIMIZE_BETA}
        )
        string (REPLACE ";" " " TensileGen_COMMAND_STR "${TensileGen_COMMAND}")
        message(STATUS "Generate kernels for ${NAME}: ${TensileGen_COMMAND_STR}")
        execute_process(
            COMMAND ${TensileGen_COMMAND}
            RESULT_VARIABLE TensileGen_RESULT
        )
        if(TensileGen_RESULT)
            message(SEND_ERROR "Error generating kernels")
        endif()
        # Glob TensileLib source files
        file(GLOB TensileLib_SRC
            ${Tensile_DIR}/TensileLib/src/*.cpp
            ${TensileLib_DIR_GENERATED}/Kernels/*.cpp
            ${TensileLib_DIR_GENERATED}/Solutions/*.cpp
            ${TensileLib_DIR_GENERATED}/Other/*.cpp
        )

    else()
        message(STATUS "No solutions for ${NAME}")
        file( WRITE ${TensileLib_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl "")
        # Glob TensileLib source files
        file(GLOB TensileLib_SRC
            ${Tensile_DIR}/TensileLib/src/*.cpp
        )
        set(TensileLib_ENABLE_SOLVER 0)

    endif()

    set(options)
    if(PARSE_EXCLUDE_FROM_ALL)
        list(APPEND options EXCLUDE_FROM_ALL)
    endif()
    add_library(${NAME} ${options} ${TensileLib_SRC})

    target_include_directories(${NAME}
        PUBLIC  $<BUILD_INTERFACE:${Tensile_DIR}/TensileLib/include>
                $<BUILD_INTERFACE:${Tensile_DIR}/TensileLib/src>
                $<BUILD_INTERFACE:${TensileLib_DIR_GENERATED}>
                $<BUILD_INTERFACE:${TensileLib_DIR_GENERATED}/Kernels>
                $<BUILD_INTERFACE:${TensileLib_DIR_GENERATED}/Solutions>
                $<BUILD_INTERFACE:${TensileLib_DIR_GENERATED}/Other>
                $<INSTALL_INTERFACE:include>
    )

    if( PARSE_BACKEND MATCHES "OpenCL_1.2")
        target_compile_definitions( ${NAME} PUBLIC -DTensile_BACKEND_OPENCL12=1 -DTensile_BACKEND_HIP=0 )
    elseif( PARSE_BACKEND MATCHES "HIP")
        target_compile_definitions( ${NAME} PUBLIC -DTensile_BACKEND_OPENCL12=0 -DTensile_BACKEND_HIP=1 )
    endif()

    if( ${PARSE_ENABLE_LOGGER} )
        set(TensileLib_ENABLE_LOGGER 1)
    else()
        set(TensileLib_ENABLE_LOGGER 0)
    endif()

    target_compile_definitions( ${NAME} PRIVATE 
        -DTensile_SOLVER_ENABLED=${TensileLib_ENABLE_SOLVER} 
        -DTensile_LOGGER_ENABLED=${TensileLib_ENABLE_LOGGER}
    )

    if(PARSE_PROBLEMS)
        if (CMAKE_CXX_COMPILER MATCHES ".*hipcc")
            # hipcc is a pearl script, so it requires a lot of extra escaping
            target_compile_definitions(${NAME} PUBLIC -DTensile_DIR_PROBLEMS=\\\"${PARSE_PROBLEMS}\\\")
        else()
            target_compile_definitions(${NAME} PUBLIC -DTensile_DIR_PROBLEMS="${PARSE_PROBLEMS}")
        endif()
    endif()

endfunction()
