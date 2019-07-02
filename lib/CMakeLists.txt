
#if(NOT DEFINED TENSILE_CXX_COMPILER)
#    set(TENSILE_CXX_COMPILER ${CMAKE_CXX_COMPILER})
#endif()
#
#set(CMAKE_CXX_COMPILER ${TENSILE_CXX_COMPILER})

cmake_minimum_required(VERSION 3.5)

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
    get_filename_component(TENSILE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
endif()
if(NOT DEFINED TENSILE_SCRIPT_ROOT)
    set(TENSILE_SCRIPT_ROOT "${TENSILE_ROOT}/Tensile")
endif()

include("${TENSILE_SCRIPT_ROOT}/Source/TensileConfig.cmake")


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

if(TENSILE_USE_LLVM)
    find_package(LLVM 6.0 REQUIRED CONFIG)
    set(tensile_sources ${tensile_sources}
        source/llvm/YAML.cpp
        source/llvm/Loading.cpp
    )
    include_directories (${LLVM_INCLUDE_DIRS})
endif()

if(TENSILE_USE_HIP)
    set(tensile_sources ${tensile_sources}
        source/hip/HipSolutionAdapter.cpp
        source/hip/HipHardware.cpp
        )
endif()

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library (Tensile STATIC ${tensile_sources})

target_include_directories(Tensile INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(configs)

if(TENSILE_USE_LLVM)
    target_compile_definitions(Tensile PUBLIC TENSILE_DEFAULT_SERIALIZATION)
    target_link_libraries( Tensile PUBLIC LLVMObjectYAML )
endif()

if(TENSILE_STATIC_ONLY)
    target_compile_definitions(Tensile PUBLIC TENSILE_STATIC_ONLY)
endif()

if(HIP_FOUND)
    target_compile_definitions(Tensile PUBLIC TENSILE_USE_HIP)

    add_subdirectory(client)
    target_link_libraries( Tensile PUBLIC ${HSA_LIBRARIES} hip::host )
endif()

if(NOT TENSILE_DISABLE_TESTS)
    add_subdirectory(test)
endif()

if(false)
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
endif()

