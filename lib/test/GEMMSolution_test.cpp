/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <Tensile/AMDGPU.hpp>
#include <Tensile/GEMMSolution.hpp>

#include <gtest/gtest.h>

using namespace Tensile;

TEST(GEMMSolution, Simple)
{
    GEMMSolution s;

    s.kernelName = "fooKernel";

    s.workGroupSize = dim3{128,1,1};
    s.macroTile = dim3{128,128,1};
    s.debugKernel = false;

    TensorDescriptor a(DataType::Float, {1534, 2147, 28});
    TensorDescriptor b(DataType::Float, {2147, 3481, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});
    TensorOps noOps;

    std::vector<float> array;

    GEMMProblem p(a, noOps, b, noOps, c, noOps, c, noOps, true);
    GEMMInputs i;
    i.a = array.data();
    i.b = array.data();
    i.c = array.data();
    i.d = array.data();

    i.alpha = 1.0f;
    i.beta  = 1.0f;

    AMDGPU h;

    std::vector<KernelInvocation> result = s.solve(p, i, h);

    ASSERT_EQ(result.size(), 1);

    EXPECT_EQ(result[0].workGroupSize, dim3({128,1,1}));
    EXPECT_EQ(result[0].numWorkGroups, dim3({12,28,28}));
    EXPECT_EQ(result[0].numWorkItems,  dim3({1536,28,28}));

    EXPECT_EQ(result[0].solution, &s);
}

