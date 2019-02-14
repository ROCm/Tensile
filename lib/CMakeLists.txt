
cmake_minimum_required(VERSION 3.5)

project(Tensile)

#find_package(HIP REQUIRED CONFIG PATHS /opt/rocm/hip)

set(HIP_FOUND 1)
if(HIP_FOUND)
    add_definitions(-D__HIP_PLATFORM_HCC__)
    set(HIP_INCLUDE_DIRS /opt/rocm/hip/include)
    set(HIP_LIBRARIES /opt/rocm/hip/lib/libhip_hcc.so)
    #set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
endif()

set(CMAKE_CXX_STANDARD 14)

if(HIP_FOUND)
    set(tensile_sources ${tensile_sources}
        source/hip/HipSolutionAdapter.cpp
        source/hip/HipHardware.cpp)
endif()

set(tensile_sources  ${tensile_sources}
    source/AMDGPU.cpp
    source/GEMMProblem.cpp
    source/GEMMSolution.cpp
    source/KernelArguments.cpp
    source/TensorDescriptor.cpp
    source/TensorOps.cpp
    )

add_library (Tensile STATIC ${tensile_sources})

target_include_directories (Tensile PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(HIP_FOUND)
    target_include_directories( Tensile SYSTEM PUBLIC  ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS} )
    target_link_libraries( Tensile PUBLIC ${HSA_LIBRARIES} )
endif()

add_subdirectory(test)

