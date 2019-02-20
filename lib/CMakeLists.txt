
cmake_minimum_required(VERSION 3.5)

project(Tensile)
find_package(HIP CONFIG PATHS /opt/rocm/hip)
#message(${HIP_HIPCC_EXECUTABLE})
#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

find_package(LLVM)

#set(HIP_FOUND 1)

#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Werror" )

set(tensile_sources  ${tensile_sources}
    source/AMDGPU.cpp
    source/GEMMProblem.cpp
    source/GEMMSolution.cpp
    source/KernelArguments.cpp
    source/TensorDescriptor.cpp
    source/TensorOps.cpp
    source/Tensile.cpp
    source/llvm/YAML.cpp
    source/llvm/Loading.cpp
    )

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/include" ${LLVM_INCLUDE_DIRS})

add_library (Tensile STATIC ${tensile_sources})

target_link_libraries( Tensile PUBLIC LLVMObjectYAML )

if(HIP_FOUND)
    add_subdirectory(source/hip)
    add_subdirectory(configs)
    target_link_libraries( Tensile PUBLIC ${HSA_LIBRARIES} TensileHip )
endif()

add_subdirectory(test)

