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

#include <Tensile/llvm/YAML.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <TestUtils.hpp>

#include "TestData.hpp"

using namespace Tensile;

struct LibraryPerformanceTest: public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(LibraryPerformanceTest, CreateProblem)
{
    for(int i = 0; i < 1000; i++)
        RandomGEMM();
}

TEST_P(LibraryPerformanceTest, LoadLibrary)
{
    auto filename = std::get<0>(GetParam());
    auto library = LoadLibraryFile<ContractionProblem>(TestData::Instance().file(filename).native());
}

TEST_P(LibraryPerformanceTest, FindSolution)
{
    std::string filename;
    bool hasNavi;
    std::tie(filename, hasNavi) = GetParam();

    auto library = LoadLibraryFile<ContractionProblem>(TestData::Instance().file(filename).native());

    {
        AMDGPU hardware(AMDGPU::Processor::gfx900, 64, "Vega 10");
        for(int i = 0; i < 10000; i++)
        {
            auto problem = RandomGEMM();
            auto solution = library->findBestSolution(problem, hardware);

            ASSERT_NE(solution, nullptr) << i << problem;
        }
    }

    if(hasNavi)
    {
        AMDGPU hardware(AMDGPU::Processor::gfx1010, 64, "Navi");
        for(int i = 0; i < 10000; i++)
        {
            auto problem = RandomGEMM();
            auto solution = library->findBestSolution(problem, hardware);

            ASSERT_NE(solution, nullptr) << i << problem;
        }
    }
}

TEST_P(LibraryPerformanceTest, Solve)
{
    std::string filename;
    bool hasNavi;
    std::tie(filename, hasNavi) = GetParam();

    auto library = LoadLibraryFile<ContractionProblem>(TestData::Instance().file(filename).native());

    AMDGPU hardware(AMDGPU::Processor::gfx900, 64, "Vega 10");

    auto problem = RandomGEMM();

    float a, b, c, d;
    TypedContractionInputs<float> inputs{&a, &b, &c, &d, 1.0, 2.0};

    auto solution = library->findBestSolution(problem, hardware);

    for(int i = 0; i < 10000; i++)
    {
        solution->solve(problem, inputs, hardware);
    }
}

TEST_P(LibraryPerformanceTest, FindAndSolve)
{
    std::string filename;
    bool hasNavi;
    std::tie(filename, hasNavi) = GetParam();

    auto library = LoadLibraryFile<ContractionProblem>(TestData::Instance().file(filename).native());

    {
        AMDGPU hardware(AMDGPU::Processor::gfx900, 64, "Vega 10");
        for(int i = 0; i < 10000; i++)
        {
            auto problem = RandomGEMM();
            auto solution = library->findBestSolution(problem, hardware);
            float a, b, c, d;
            TypedContractionInputs<float> inputs{&a, &b, &c, &d, 1.0, 2.0};

            ASSERT_NE(solution, nullptr) << i << problem;
            
            for(int j = 0; j < 1000; j++)
            {
                solution->solve(problem, inputs, hardware);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(LLVM, LibraryPerformanceTest,
        ::testing::Values(
            std::make_tuple("KernelsLite.yaml",      false),
            std::make_tuple("KernelsLiteMixed.yaml", false),
            std::make_tuple("KernelsLiteNavi.yaml",  true),
            std::make_tuple("KernelsTileLite.yaml",  false)
            ));

