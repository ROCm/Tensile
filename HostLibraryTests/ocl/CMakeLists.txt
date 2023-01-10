################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

set(COMPILER "hipcc")
set(CODE_OBJECT_VERSION "default")

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
