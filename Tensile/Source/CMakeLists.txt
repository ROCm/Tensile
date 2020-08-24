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

cmake_minimum_required(VERSION 3.5)

# Override all paths arguments as they do not work properly
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

option(TENSILE_NEW_CLIENT "Build the new client" ON)

if(TENSILE_NEW_CLIENT)

    project(Tensile)

    set(TENSILE_USE_HIP      ON CACHE BOOL "Use the Hip runtime.")
    set(TENSILE_USE_LLVM     ON CACHE BOOL "Use LLVM for parsing config files.")
    set(TENSILE_USE_OPENMP   ON CACHE BOOL "Use LLVM for parsing config files.")
    set(TENSILE_STATIC_ONLY  ON CACHE BOOL "Disable exposing Tensile symbols in a shared library.")

    if(CMAKE_CXX_COMPILER STREQUAL "hipcc")
      set(TENSILE_GPU_ARCHS gfx803 gfx900 gfx906 gfx908 gfx1010 CACHE STRING "GPU architectures")
    else()
      set(TENSILE_GPU_ARCHS gfx803 gfx900 gfx906 gfx908 CACHE STRING "GPU architectures")
    endif()

    include(CMakeDependentOption)
    CMAKE_DEPENDENT_OPTION(TENSILE_BUILD_CLIENT "Build the benchmarking client" ON
                           "TENSILE_USE_HIP;TENSILE_USE_LLVM" OFF)

    if(TENSILE_USE_HIP)
        find_package(HIP REQUIRED CONFIG PATHS /opt/rocm)
    endif()

    if(TENSILE_USE_OPENMP)
        #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_OPENMP")
        find_package(OpenMP QUIET)
        if (OPENMP_FOUND)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        else ()
            if(EXISTS /etc/redhat-release)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libgomp")
            else()
	        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
                set(OPENMP_LIBRARY /usr/lib/x86_64-linux-gnu/libomp.so)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OPENMP_LIBRARY}")
            endif()
        endif()
    endif()

    #set(TENSILE_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
    #if(TENSILE_USE_HIP)
    #    string(FIND "${CMAKE_CXX_COMPILER}" "hcc"   cxx_hcc)
    #    string(FIND "${CMAKE_CXX_COMPILER}" "hipcc" cxx_hipcc)

    #    if(cxx_hcc EQUAL "-1" AND cxx_hipcc EQUAL "-1")
    #        set(TENSILE_CXX_COMPILER "${HIP_BIN_INSTALL_DIR}/hcc")
    #    endif()
    #endif()

    add_subdirectory(lib)

    if(TENSILE_BUILD_CLIENT)
        add_subdirectory(client)
    endif()

    export(TARGETS TensileHost NAMESPACE Tensile:: FILE TensileExports.cmake)

else()

    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
    project(TensileClient)
    set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR} )

    ###############################################################################
    # Common Options
    option( Tensile_ENABLE_HALF "Enable half precision data types" OFF)
    option( Tensile_CLIENT_BENCHMARK "ON=BenchmarkClient; OFF=LibraryClient" ON)
    option( Tensile_RESUME_BENCHMARK "When benchmark interrupted ON=Resume the benchmark process; OFF=Restart the benchmark process" OFF)
    option( Tensile_MERGE_FILES "Merge kernels and solutions files" OFF)
    set(Tensile_RUNTIME_LANGUAGE HIP CACHE STRING "Which runtime language to use")
    set_property( CACHE Tensile_RUNTIME_LANGUAGE PROPERTY STRINGS HIP OCL )
    set(Tensile_CODE_OBJECT_VERSION V2 CACHE STRING "Which code object version to use")
    set_property( CACHE Tensile_CODE_OBJECT_VERSION PROPERTY STRINGS V2 V3)
    set_property( CACHE Tensile_COMPILER PROPERTY STRINGS hcc hipcc)
    set(Tensile_ARCHITECTURE all CACHE STRING "Which GPU architecture to use")
    set_property( CACHE Tensile_ARCHITECTURE PROPERTY STRINGS all gfx000 gfx900 gfx906 gfx908)
    set_property( CACHE Tensile_LIBRARY_FORMAT PROPERTY STRINGS yaml msgpack)

    message( STATUS "Tensile_CODE_OBJECT_VERSION from Tensile/Source/CMakeLists.txt: ${Tensile_CODE_OBJECT_VERSION}")
    message( STATUS "Tensile_COMPILER            from Tensile/Source/CMakeLists.txt: ${Tensile_COMPILER}")
    message( STATUS "Tensile_ARCHITECTURE        from Tensile/Source/CMakeLists.txt: ${Tensile_ARCHITECTURE}")
    message( STATUS "Tensile_LIBRARY_FORMAT      from Tensile/Source/CMakeLists.txt: ${Tensile_LIBRARY_FORMAT}")

    ###############################################################################
    # Benchmark Client
    set(ClientName "client")
    if(Tensile_CLIENT_BENCHMARK)
      message(STATUS "Making BenchmarkClient")
      include(${CMAKE_SOURCE_DIR}/Generated.cmake)
      add_executable( ${ClientName}
        ${TensileClient_SOURCE}
        ${TensileClient_SOLUTIONS}
        ${TensileClient_KERNELS} )
      target_compile_definitions( ${ClientName} PUBLIC
        -DTensile_CLIENT_BENCHMARK=1
        -DTensile_CLIENT_LIBRARY=0 )
      foreach( target ${Tensile_ISA} )
        target_link_libraries( ${ClientName} PRIVATE --amdgpu-target=${target} )
      endforeach()

    else()
    ###############################################################################
    # Library Client
      message(STATUS "Making LibraryClient")
      option( Tensile_SHORT_FILE_NAMES "Use short file names (for MSVC)" OFF)
      option( Tensile_LIBRARY_PRINT_DEBUG "TensileLib to print debug info" OFF)
      add_executable( ${ClientName}
        Client.cpp
        MathTemplates.cpp
        DeviceStats.h
        )
      target_compile_definitions( ${ClientName} PUBLIC
        -DTensile_CLIENT_BENCHMARK=0
        -DTensile_CLIENT_LIBRARY=1 )
    endif()

    if(Tensile_RESUME_BENCHMARK)
        target_compile_definitions( ${ClientName} PUBLIC
          -DTensile_RESUME_BENCHMARK )
    endif()
    ###############################################################################
    # Visual Studio or HIPCC
    if(MSVC)
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" ) # parallel build
    endif()

    if (UNIX)
      # the Tensile library doesn't require c++11, but client/ReferenceCPU does
      # with addition of bfloat16 library requires c++14
      #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14"  )
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-deprecated-declarations" )
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_HCC_COMPAT_MODE__=1")
    endif()

    if(Tensile_MERGE_FILES)
      #include_directories(.)
    else()
      include_directories( Kernels Solutions )
      include_directories(.)
    endif()

    ###############################################################################
    # find and use device libraries
    if( Tensile_RUNTIME_LANGUAGE MATCHES "OCL")
      find_package(OpenCL "1.2" REQUIRED)
      target_link_libraries( ${ClientName} PUBLIC ${OPENCL_LIBRARIES} )
      target_compile_definitions( ${ClientName} PUBLIC
        -DTensile_RUNTIME_LANGUAGE_OCL=1
        -DTensile_RUNTIME_LANGUAGE_HIP=0 )
      target_include_directories( ${ClientName} SYSTEM
        PUBLIC  ${OPENCL_INCLUDE_DIRS} )
    elseif( Tensile_RUNTIME_LANGUAGE MATCHES "HIP")
      find_package( HIP REQUIRED )
      set (CMAKE_CXX_COMPILER ${HIPCC})
      target_include_directories( ${ClientName} SYSTEM
        PUBLIC  ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS} )
      target_link_libraries( ${ClientName} PUBLIC ${HSA_LIBRARIES} )
      target_compile_definitions( ${ClientName} PUBLIC
        -DTensile_RUNTIME_LANGUAGE_OCL=0
        -DTensile_RUNTIME_LANGUAGE_HIP=1
        )
      if( Tensile_ENABLE_HALF )
        target_compile_definitions( ${ClientName} PUBLIC  -DTensile_ENABLE_HALF )
      endif()
    endif()

    ###############################################################################
    # Create Tensile Library
    if(NOT Tensile_CLIENT_BENCHMARK)
      set( PACKAGE_TENSILE_LIBRARY OFF )
      set( USE_LEGACY_CODE ON )
      include(${CMAKE_SOURCE_DIR}/TensileCreateLibrary.cmake)
      TensileCreateLibraryCmake(
        ${Tensile_LOGIC_PATH}           # path
        ${Tensile_RUNTIME_LANGUAGE}     # HIP or OCL
        ${Tensile_COMPILER}             # hcc or hipcc
        ${Tensile_CODE_OBJECT_VERSION}  # V2 or V3
        ${Tensile_ARCHITECTURE}         # all, gfx000, gfx803, gfx900, gfx906, gfx908
        ${Tensile_LIBRARY_FORMAT}       # yaml or msgpack
        ${Tensile_MERGE_FILES}          # ON or OFF
        ${Tensile_SHORT_FILE_NAMES}     # ON or OFF
        ${Tensile_LIBRARY_PRINT_DEBUG}  # ON or OFF
        ${PACKAGE_TENSILE_LIBRARY}
        ${USE_LEGACY_CODE}
        Tensile_ROOT ${Tensile_ROOT}
        )
      target_link_libraries( ${ClientName} PUBLIC Tensile )
    endif()

endif()

