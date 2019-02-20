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

#include <gtest/gtest.h>

#include <Tensile/Tensile.hpp>
#include <Tensile/GEMMLibrary.hpp>
#include <Tensile/AMDGPU.hpp>

using namespace Tensile;

TEST(GEMMLibraryLoadingTest, Simple)
{
    auto library = LoadLibraryFile<GEMMProblem, GEMMSolution>("test/sample_library.yaml");

    ASSERT_NE(library, nullptr);
}

TEST(GEMMLibraryLoadingTest, MultipleKernels)
{
    auto library = LoadLibraryFile<GEMMProblem, GEMMSolution>("configs/TensileKernels.yaml");
    ASSERT_NE(library, nullptr);

    AMDGPU hardware;

    {
        GEMMProblem p = GEMMProblem::FromBLAS(false, false, 4, 4, 4, 4, 4, 4, true, false, 2);

        auto solution = library->findBestSolution(p, hardware);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(solution->name(), "Cijk_Ailk_Bljk_SB_MT128x128x08_K1");

    }

    {
        GEMMProblem p = GEMMProblem::FromBLAS(false,  true, 4, 4, 4, 4, 4, 4, true, false, 2);

        auto solution = library->findBestSolution(p, hardware);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(solution->name(), "Cijk_Ailk_Bjlk_SB_MT128x128x08_K1");

    }

    {
        GEMMProblem p = GEMMProblem::FromBLAS( true, false, 4, 4, 4, 4, 4, 4, true, false, 2);

        auto solution = library->findBestSolution(p, hardware);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(solution->name(), "Cijk_Alik_Bljk_SB_MT128x128x08_K1");

    }

    {
        GEMMProblem p = GEMMProblem::FromBLAS( true,  true, 4, 4, 4, 4, 4, 4, true, false, 2);

        auto solution = library->findBestSolution(p, hardware);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(solution->name(), "Cijk_Alik_Bjlk_SB_MT128x128x08_K1");

    }
}

