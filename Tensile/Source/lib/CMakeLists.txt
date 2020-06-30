################################################################################
# Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

if(NOT DEFINED TENSILE_ROOT)
    get_filename_component(TENSILE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
endif()
if(NOT DEFINED TENSILE_SCRIPT_ROOT)
    set(TENSILE_SCRIPT_ROOT "${TENSILE_ROOT}/Tensile")
endif()

set(tensile_sources  ${tensile_sources}
    source/AMDGPU.cpp
    source/ArithmeticUnitTypes.cpp
    source/ContractionProblem.cpp
    source/ContractionSolution.cpp
    source/DataTypes.cpp
    source/Debug.cpp
    source/EmbeddedLibrary.cpp
    source/KernelArguments.cpp
    source/KernelLanguageTypes.cpp
    source/MessagePack.cpp
    source/TensorDescriptor.cpp
    source/TensorOps.cpp
    source/Tensile.cpp
    source/Utils.cpp
    )

if(TENSILE_USE_LLVM)
    find_package(LLVM 6.0 QUIET CONFIG)
    if(NOT LLVM_FOUND)
        find_package(LLVM 7.0 QUIET CONFIG)
        if(NOT LLVM_FOUND)
            find_package(LLVM 9.0 QUIET CONFIG)
            if(NOT LLVM_FOUND)
                find_package(LLVM REQUIRED CONFIG)
            endif()
        endif()
    endif()
    set(tensile_sources ${tensile_sources}
        source/llvm/YAML.cpp
        source/llvm/Loading.cpp
    )
endif()

if(TENSILE_USE_HIP)
    set(tensile_sources ${tensile_sources}
        source/hip/HipSolutionAdapter.cpp
        source/hip/HipHardware.cpp
        )
endif()

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library (TensileHost STATIC ${tensile_sources})

target_include_directories(TensileHost INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(Tensile_LIBRARY_FORMAT MATCHES "msgpack")
    find_package(msgpack)
    if(TARGET msgpackc-static)
        target_link_libraries(TensileHost PRIVATE msgpackc-static)
    else()
        target_link_libraries(TensileHost PRIVATE msgpackc)
    endif()
endif()

if(TENSILE_USE_LLVM)
    target_compile_definitions(TensileHost PUBLIC TENSILE_DEFAULT_SERIALIZATION)
    find_library(LLVMObjectYAML_LIBRARY
        NAMES LLVMObjectYAML
        PATHS ${LLVM_LIBRARY_DIR})
    message("LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")
    #Use LLVM shared libs if LLVMObjectYAML static lib cannot be found
    if(LLVMObjectYAML_LIBRARY)
        target_link_libraries(TensileHost PRIVATE LLVMObjectYAML )
    else()
        target_link_libraries(TensileHost PRIVATE LLVM )
    endif()
    target_include_directories(TensileHost PRIVATE ${LLVM_INCLUDE_DIRS})
endif()

if(TENSILE_STATIC_ONLY)
    target_compile_definitions(TensileHost PUBLIC TENSILE_STATIC_ONLY)
endif()

if(TENSILE_USE_HIP)
    target_compile_definitions(TensileHost PUBLIC TENSILE_USE_HIP)

    #add_subdirectory(client)
    target_link_libraries( TensileHost PUBLIC ${HSA_LIBRARIES} hip::device )

endif()

if(Tensile_LIBRARY_FORMAT MATCHES "yaml")
    target_compile_definitions(TensileHost PUBLIC -DTENSILE_YAML=1)
endif()

if(false)
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
endif()

