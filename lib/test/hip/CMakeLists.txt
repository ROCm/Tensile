
#find_package(ROCM REQUIRED CONFIG PATHS /opt/rocm)

#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

if(true)
TensileCreateLibraryFiles(
    "${TENSILE_ROOT}/lib/configs/lite_configs"
    "${CMAKE_CURRENT_BINARY_DIR}/test_kernels_lite"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY test_kernels_lite
    EMBED_KEY     kernels_lite
    VAR_PREFIX LITE
    )
endif()

TensileCreateLibraryFiles(
    "${TENSILE_ROOT}/lib/configs/lite_configs_mixed"
    "${CMAKE_CURRENT_BINARY_DIR}/test_kernels_lite_mixed"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY test_kernels_lite_mixed
    EMBED_KEY     kernels_lite_mixed
    VAR_PREFIX LITE
    )

if(true)
# Link a second copy of the lite_configs library to test that there are no issues with duplicate symbols.
TensileCreateLibraryFiles(
    "${TENSILE_ROOT}/lib/configs/lite_configs"
    "${CMAKE_CURRENT_BINARY_DIR}/test_kernels_lite_2"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY test_kernels_lite_2
    EMBED_KEY     kernels_lite_2
    )
endif()



set(test_yaml rocblas_sgemm_asm_single_kernel.yaml)
file(COPY ${test_yaml} DESTINATION .)
add_custom_target(test_code_object
                  COMMAND python ${tensile_script} ${test_yaml} code_object
                  DEPENDS ${test_yaml})

add_library(TensileHipTests STATIC
            HipSolutionAdapter_test.cpp
            RunGEMMKernel_test.cpp)

message(${gtest_SOURCE_DIR}/include)

target_link_libraries(TensileHipTests hip::host test_kernels_lite test_kernels_lite_2 test_kernels_lite_mixed)

target_include_directories(TensileHipTests
                           PRIVATE
                           ${gtest_SOURCE_DIR}/include)

#add_dependencies(test_code_object ${test_yaml})



if(false)
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
endif()
