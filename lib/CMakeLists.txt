
cmake_minimum_required(VERSION 3.5)

project(Tensile)
find_package(HIP CONFIG PATHS /opt/rocm/hip)
#message(${HIP_HIPCC_EXECUTABLE})
#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

option(USE_LLVM "Use LLVM for parsing config files." ON)
option(TENSILE_STATIC_ONLY "Disable exposing Tensile symbols in a shared library." ON)
option(TENSILE_DISABLE_TESTS "Disable ctest integration." OFF)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(USE_LLVM)
    find_package(LLVM REQUIRED CONFIG)
    add_definitions(-DTENSILE_DEFAULT_SERIALIZATION)
endif()

if(TENSILE_STATIC_ONLY)
    add_definitions(-DTENSILE_STATIC_ONLY)
endif()

if(NOT TENSILE_DISABLE_TESTS)
    enable_testing()

    if(CMAKE_VERSION VERSION_GREATER 3.9)
        include(GoogleTest)
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." ON)
    else()
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." OFF)
    endif()
endif()

if(NOT DEFINED TENSILE_ROOT)
    set(TENSILE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
if(NOT DEFINED TENSILE_SCRIPT_ROOT)
    set(TENSILE_SCRIPT_ROOT "${TENSILE_ROOT}/Tensile")
endif()

include("${TENSILE_SCRIPT_ROOT}/Source/TensileConfig.cmake")


#set(HIP_FOUND 1)

#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Werror -ftemplate-backtrace-limit=0" )

set(tensile_sources  ${tensile_sources}
    source/AMDGPU.cpp
    source/ContractionProblem.cpp
    source/ContractionSolution.cpp
    source/DataTypes.cpp
    source/Debug.cpp
    source/EmbeddedLibrary.cpp
    source/KernelArguments.cpp
    source/TensorDescriptor.cpp
    source/TensorOps.cpp
    source/Tensile.cpp
    source/Utils.cpp
    )

if(USE_LLVM)
    set(tensile_sources ${tensile_sources}
        source/llvm/YAML.cpp
        source/llvm/Loading.cpp
    )
include_directories (${LLVM_INCLUDE_DIRS})
endif()

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library (Tensile STATIC ${tensile_sources})

target_include_directories(Tensile INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(configs)

if(USE_LLVM)
    #target_link_libraries( Tensile PUBLIC LLVM )
    target_link_libraries( Tensile PUBLIC LLVMObjectYAML )
endif()

if(HIP_FOUND)
    add_subdirectory(client)
    add_subdirectory(source/hip)
    target_link_libraries( Tensile PUBLIC ${HSA_LIBRARIES} TensileHip )
endif()

if(NOT TENSILE_DISABLE_TESTS)
    add_subdirectory(test)
endif()

