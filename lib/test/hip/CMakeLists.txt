
#find_package(ROCM REQUIRED CONFIG PATHS /opt/rocm)

#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

if(true)
TensileCreateLibraryFiles(
    "${TENSILE_ROOT}/lib/configs/lite_configs"
    "${CMAKE_CURRENT_BINARY_DIR}/test_kernels_lite"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY test_kernels_lite
    EMBED_KEY     kernels_lite
    VAR_PREFIX LITE NO_MERGE_FILES
    )
endif()

file(MAKE_DIRECTORY "${TEST_DATA_DIR}/kernels_lite")
file(COPY ${LITE_ALL_FILES} DESTINATION "${TEST_DATA_DIR}/kernels_lite")

if(true)
TensileCreateLibraryFiles(
    "${TENSILE_ROOT}/lib/configs/lite_configs_mixed"
    "${CMAKE_CURRENT_BINARY_DIR}/test_kernels_lite_mixed"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY test_kernels_lite_mixed
    EMBED_KEY     kernels_lite_mixed
    VAR_PREFIX LITE_MIXED
    )
endif()

file(MAKE_DIRECTORY "${TEST_DATA_DIR}/kernels_lite_mixed")
file(COPY ${LITE_ALL_FILES} DESTINATION "${TEST_DATA_DIR}/kernels_lite_mixed")

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
    COMMAND python ${TENSILE_ROOT}/Tensile/Tensile.py ${test_yaml} code_object
    DEPENDS ${test_yaml})

add_library(TensileHipTests STATIC
            HipSolutionAdapter_test.cpp
            RunGEMMKernel_test.cpp)

target_link_libraries(TensileHipTests hip::host
                      test_kernels_lite
                      test_kernels_lite_2
                      test_kernels_lite_mixed
                      TensileClient)

target_include_directories(TensileHipTests
                           PRIVATE
                           ${gtest_SOURCE_DIR}/include)

#add_dependencies(test_code_object ${test_yaml})

