set(COMPILER "hipcc")
set(CODE_OBJECT_VERSION "V3")

TensileCreateLibraryFiles(
    "${CMAKE_CURRENT_SOURCE_DIR}/../configs/lite_configs_mixed"
    "${CMAKE_CURRENT_BINARY_DIR}/ocl_test_kernels_lite_mixed"
    TENSILE_ROOT "${TENSILE_SCRIPT_ROOT}"
    EMBED_LIBRARY ocl_test_kernels_lite_mixed
    EMBED_KEY     ocl_kernels_lite_mixed
    VAR_PREFIX OCL_LITE_MIXED
    NO_MERGE_FILES
    COMPILER ${COMPILER}
    CODE_OBJECT_VERSION ${CODE_OBJECT_VERSION}
    )

TensileCreateCopyTarget(
    copy_ocl_kernels_lite_mixed
    "${OCL_LITE_MIXED_ALL_FILES}"
    ${TEST_DATA_DIR}/ocl_kernels_lite_mixed
    )

add_dependencies(copy_ocl_kernels_lite_mixed OCL_LITE_MIXED_LIBRARY_TARGET)

set(test_sources ${test_sources}
    ocl/OclSolutionAdapter_test.cpp
    ocl/RunGEMMKernel_test.cpp
    PARENT_SCOPE)

set(OCL_TEST_LIBRARIES ocl_test_kernels_lite_mixed PARENT_SCOPE)

# Make sure that the TensileTests depends on targets below.
# Otherwise the TensileTests executable will segfault out.
set(OCL_TEST_COPY_TARGET_DEPS copy_ocl_kernels_lite_mixed PARENT_SCOPE)
