/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#include <gtest/gtest.h>

#include "OclBackend.hpp"
#include "RunGEMMKernelTest.hpp"

using namespace Tensile;

// Differentiate now on the OclBackend
using RunGEMMKernelTestOcl       = RunGEMMKernelTest<OclBackend>;
using RunGEMMKernelTestOclParams = RunGEMMKernelTestParams<OclBackend>;

// Define GEMMKernelTest for OpenCL
TEST_P(RunGEMMKernelTestOcl, TestBestSolution)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->TestBestSolution();
}

TEST_P(RunGEMMKernelTestOcl, TestAllSolutions)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->TestAllSolutions();
}

TEST_P(RunGEMMKernelTestOcl, TestAlphaZero)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->OverrideAlpha(0.0);
    typedTest->TestBestSolution();
}

TEST_P(RunGEMMKernelTestOcl, TestAlphaZeroSigned)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->OverrideAlpha(std::copysign(0.0, -1.0));
    typedTest->TestBestSolution();
}

TEST_P(RunGEMMKernelTestOcl, TestAlphaZeroABNull)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->OverrideAlpha(0.0);
    typedTest->NullifyAPtr();
    typedTest->NullifyBPtr();
    auto fail = false;
    try
    {
        ASSERT_NO_THROW();
    }
    catch(...)
    {
        fail = true;
    }
    ASSERT_EQ(fail, false);
}

INSTANTIATE_TEST_SUITE_P(
    OclSolutionAdapter,
    RunGEMMKernelTestOcl,
    ::testing::Combine(::testing::ValuesIn(RunGEMMKernelTestOclParams::TypedTests()),
                       ::testing::ValuesIn(RunGEMMKernelTestOclParams::TestProblems()),
                       ::testing::ValuesIn(RunGEMMKernelTestOclParams::TestLibraries()),
                       ::testing::ValuesIn(RunGEMMKernelTestOclParams::TestMemoryAlignments())));

INSTANTIATE_TEST_SUITE_P(
    OclSolutionAdapter_Extended,
    RunGEMMKernelTestOcl,
    ::testing::Combine(::testing::ValuesIn(RunGEMMKernelTestOclParams::TypedTests()),
                       ::testing::ValuesIn(RunGEMMKernelTestOclParams::TestProblemsExtended()),
                       ::testing::ValuesIn(RunGEMMKernelTestOclParams::TestLibraries()),
                       ::testing::ValuesIn(RunGEMMKernelTestOclParams::TestMemoryAlignments())));
