
cmake_minimum_required(VERSION 3.5)

project(Tensile)
find_package(HIP CONFIG PATHS /opt/rocm/hip)
#message(${HIP_HIPCC_EXECUTABLE})
#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

option(USE_LLVM "Use LLVM for parsing config files." ON)

if(USE_LLVM)
find_package(LLVM REQUIRED CONFIG)
endif()

#set(HIP_FOUND 1)

#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Werror" )

set(tensile_sources  ${tensile_sources}
    source/AMDGPU.cpp
    source/ContractionProblem.cpp
    source/ContractionSolution.cpp
    source/DataTypes.cpp
    source/KernelArguments.cpp
    source/TensorDescriptor.cpp
    source/TensorOps.cpp
    source/Tensile.cpp
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
target_link_libraries( Tensile PUBLIC LLVM )
endif()

if(HIP_FOUND)
    add_subdirectory(source/hip)
    target_link_libraries( Tensile PUBLIC ${HSA_LIBRARIES} TensileHip )
endif()

add_subdirectory(test)

