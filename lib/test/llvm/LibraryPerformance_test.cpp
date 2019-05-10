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

#include <Tensile/Serialization.hpp>
#include <Tensile/llvm/YAML.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <TestUtils.hpp>

#include "TestData.hpp"

using namespace Tensile;

TEST(LibraryPerformanceTest, CreateProblem)
{
    for(int i = 0; i < 1000; i++)
        RandomGEMM();
}

TEST(LibraryPerformanceTest, LoadLibrary)
{
    auto library = LoadLibraryFile<ContractionProblem>(TestData::File("KernelsLiteMixed.yaml").native());
}

TEST(LibraryPerformanceTest, FindSolution)
{
    auto library = LoadLibraryFile<ContractionProblem>(TestData::File("KernelsLiteMixed.yaml").native());
    AMDGPU hardware(AMDGPU::Processor::gfx900, 64, "Vega 10");

    for(int i = 0; i < 10000; i++)
    {
        auto problem = RandomGEMM();

        auto solution = library->findBestSolution(problem, hardware);

        //ASSERT_NE(solution, nullptr);
    }
}
