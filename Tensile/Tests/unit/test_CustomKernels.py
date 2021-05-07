################################################################################
# Copyright 2021 Advanced Micro Devices, Inc. All rights reserved.
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

import pytest
import os
from Tensile.CustomKernels import getCustomKernelConfig, getCustomKernelContents
from Tensile.BenchmarkProblems import generateCustomKernelSolution
from Tensile.Common import globalParameters, assignGlobalParameters
import yaml

testKernelDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "customKernels")

@pytest.mark.parametrize("objs", [("TestKernel", testKernelDir)])
def test_FindCustomKernel(objs):
    try:
        name, directory = objs
        contents = getCustomKernelContents(name, directory)
        assert contents #If no exception
    except:
        assert False

configResult = yaml.safe_load(
"""  
ProblemType:
    OperationType: GEMM
    DataType: s
    TransposeA: False
    TransposeB: False
    UseBeta: True
    Batched: True
LoopDoWhile: False
WorkGroupMapping:  1
ThreadTile: [ 8, 8 ]
WorkGroup: [  8, 16,  1 ]
DepthU: 8
VectorWidth: 4
AssertSizeEqual: {3: 512}
AssertSizeMultiple: {0: 128, 1: 128}"""
)

# TODO when more custom kernels have been added - expand these lists
@pytest.mark.parametrize("objs", [("TestKernel", testKernelDir, configResult)])
def test_ReadCustomKernelConfig(objs):
    try:
        name, directory, result = objs
        config = getCustomKernelConfig(name, directory)
        config["custom.config"] = result  
    except:
        assert False

@pytest.mark.parametrize("objs", [("TestKernel", testKernelDir)])
def test_CreateSolutionFromCustomKernel(objs):
    try:
        assignGlobalParameters({})

        name, directory = objs
        solution = generateCustomKernelSolution(name, directory)
        assert solution["Valid"]
    except:
        assert False
