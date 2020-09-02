/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
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

#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/Utils.hpp>

#include "TestData.hpp"
#include "TestUtils.hpp"

#include <Reference.hpp>

#include <cstddef>
#include <random>

using namespace Tensile;

#define ASSERT_RB(exp) ASSERT_EQ((exp), rocblas_status_success)

struct RunGEMMKernelSolutionSelectionTest : public ::testing::TestWithParam<ContractionProblem>
{
    std::vector<float> a_h;
    std::vector<float> b_h;
    std::vector<float> c_h;
    std::vector<float> d_h;
    std::vector<float> d_in_h;
    std::vector<float> d_ref_h;

    float* a_d     = nullptr;
    float* b_d     = nullptr;
    float* c_d     = nullptr;
    float* d_d     = nullptr;
    float* d_ref_d = nullptr;

    TypedContractionInputs<float> inputs;

    std::shared_ptr<Hardware> hardware;

    void SetUp() override
    {
        HIP_CHECK_EXC(hipSetDevice(0));
        ContractionProblem problem = GetParam();
        std::cout << problem << std::endl;

        a_h.resize(problem.a().totalAllocatedElements());
        b_h.resize(problem.b().totalAllocatedElements());
        c_h.resize(problem.c().totalAllocatedElements());
        d_h.resize(problem.d().totalAllocatedElements());
        d_in_h.resize(problem.d().totalAllocatedElements());

        std::mt19937 rng;

        InitTensor(a_h.data(), problem.a(), RandomInt<float>(), rng);
        InitTensor(b_h.data(), problem.b(), RandomAlternatingInt<float>(), rng);
        InitTensor(c_h.data(), problem.c(), RandomInt<float>(), rng);
        InitTensor(d_in_h.data(), problem.d(), RandomInt<float>(), rng);

        d_ref_h = d_h;

        CopyTensor(d_ref_h.data(), c_h.data(), problem.d(), problem.c());

        HIP_CHECK_EXC(hipMalloc(&a_d, problem.a().totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&b_d, problem.b().totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&c_d, problem.c().totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&d_d, problem.d().totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&d_ref_d, problem.d().totalAllocatedBytes()));

        HIP_CHECK_EXC(
            hipMemcpy(a_d, a_h.data(), problem.a().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(
            hipMemcpy(b_d, b_h.data(), problem.b().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(
            hipMemcpy(c_d, c_h.data(), problem.c().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(
            d_d, d_in_h.data(), problem.d().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(
            d_ref_d, d_ref_h.data(), problem.d().totalAllocatedBytes(), hipMemcpyHostToDevice));

        inputs.a = a_d;
        inputs.b = b_d;
        inputs.c = c_d;
        inputs.d = d_d;

        inputs.alpha = RandomInt<float>()(rng);
        if(problem.beta() == 1.0)
            inputs.beta = 1.0;
        else if(problem.beta() == 0.0)
            inputs.beta = 0.0;
        else
            inputs.beta = RandomInt<float>()(rng);

        hardware = hip::GetCurrentDevice();
        ASSERT_NE(hardware, nullptr);

        TypedContractionInputs<float> inputsRefHost;
        inputsRefHost.a     = a_h.data();
        inputsRefHost.b     = b_h.data();
        inputsRefHost.c     = c_h.data();
        inputsRefHost.d     = d_ref_h.data();
        inputsRefHost.alpha = inputs.alpha;
        inputsRefHost.beta  = inputs.beta;

        Client::SolveCPU(problem, inputsRefHost);
    }

    void TearDown() override
    {
        hipFree(a_d);
        hipFree(b_d);
        hipFree(c_d);
        hipFree(d_d);
        hipFree(d_ref_d);

        hipDeviceReset();
    }
};

TEST_P(RunGEMMKernelSolutionSelectionTest, KernelsTileSelection)
{
    ContractionProblem problem = GetParam();

    bool debug   = false;
    auto library = LoadLibraryFile<ContractionProblem>(
        TestData::Instance().file("tile_aware_selection/library/TensileLibrary").native());

    auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "tile_aware_selection");
    for(auto file : TestData::Instance().glob("tile_aware_selection/library/*.*co"))
        adapter->loadCodeObjectFile(file.native());

    for(auto file : TestData::Instance().glob("tile_aware_selection/library/*.*hsaco"))
        adapter->loadCodeObjectFile(file.native());

    ASSERT_NE(library, nullptr);

    auto solution = library->findBestSolution(problem, *hardware);

    ASSERT_NE(solution, nullptr);

    std::vector<KernelInvocation> result = solution->solve(problem, inputs, *hardware);

    adapter->launchKernels(result);

    HIP_CHECK_EXC(
        hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

    for(int i = 0; i < d_ref_h.size(); i++)
    {
        ASSERT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i;
    }
}

TEST_P(RunGEMMKernelSolutionSelectionTest, TileAwareMetricSelection)
{

    ContractionProblem problem = GetParam();

    bool debug   = false;
    auto library = LoadLibraryFile<ContractionProblem>(
        TestData::Instance().file("tile_aware_metric_selection/library/TensileLibrary").native());

    auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "fitness_selection");
    for(auto file : TestData::Instance().glob("tile_aware_metric_selection/library/*.*co"))
        adapter->loadCodeObjectFile(file.native());

    ASSERT_NE(library, nullptr);

    auto solution = library->findBestSolution(problem, *hardware);

    ASSERT_NE(solution, nullptr);

    std::vector<KernelInvocation> result = solution->solve(problem, inputs, *hardware);

    adapter->launchKernels(result);

    HIP_CHECK_EXC(
        hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

    for(int i = 0; i < d_ref_h.size(); i++)
    {
        ASSERT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    HipSolutionAdapter,
    RunGEMMKernelSolutionSelectionTest,
    ::testing::Values(
        ContractionProblem::GEMM(false, false, 4, 4, 6, 4, 6, 4, 1.5, false, 2),
        ContractionProblem::GEMM(false, true, 4, 4, 6, 4, 4, 4, 1.5, false, 2),
        ContractionProblem::GEMM(true, false, 4, 4, 6, 6, 6, 4, 1.5, false, 2),
        ContractionProblem::GEMM(true, true, 4, 4, 6, 6, 4, 4, 1.5, false, 2),
        ContractionProblem::GEMM(false, false, 234, 123, 634, 234, 634, 234, 1.5, false, 1),
        ContractionProblem::GEMM(false, false, 234, 123, 634, 245, 768, 249, 1.5, false, 12),
        ContractionProblem::GEMM(false, true, 234, 123, 634, 245, 768, 249, 1.5, false, 12),
        ContractionProblem::GEMM(true, false, 234, 123, 634, 768, 768, 249, 1.5, false, 12),
        ContractionProblem::GEMM(true, true, 234, 123, 634, 768, 768, 249, 1.5, false, 12)));
