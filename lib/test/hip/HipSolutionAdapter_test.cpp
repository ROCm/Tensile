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

#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/GEMMSolution.hpp>
#include <Tensile/Utils.hpp>

#include <TestUtils.hpp>

#include <random>
#include <rocblas.h>

using namespace Tensile;

#define ASSERT_RB(exp) ASSERT_EQ((exp), rocblas_status_success)

struct GEMMTest: public ::testing::TestWithParam<GEMMProblem>
{
    std::vector<float> a_h;
    std::vector<float> b_h;
    std::vector<float> c_h;
    std::vector<float> d_h;
    std::vector<float> d_ref_h;

    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;
    float *d_d = nullptr;
    float *d_ref_d = nullptr;

    GEMMInputs inputs;

    std::shared_ptr<Hardware> hardware;

	void SetUp() override
	{
        GEMMProblem problem = GetParam();

        a_h.resize(problem.a.totalAllocatedElements());
        b_h.resize(problem.b.totalAllocatedElements());
        c_h.resize(problem.c.totalAllocatedElements());
        d_h.resize(problem.d.totalAllocatedElements());

        InitTensor(a_h.data(), problem.a, RandomInt<float>());
        InitTensor(b_h.data(), problem.b, RandomAlternatingInt<float>());
        InitTensor(c_h.data(), problem.c, RandomInt<float>());
        InitTensor(d_h.data(), problem.d, RandomInt<float>());

        d_ref_h = c_h;

        HIP_CHECK_EXC(hipMalloc(&a_d,     problem.a.totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&b_d,     problem.b.totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&c_d,     problem.c.totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&d_d,     problem.d.totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&d_ref_d, problem.d.totalAllocatedBytes()));

        HIP_CHECK_EXC(hipMemcpy(a_d,     a_h.data(),     problem.a.totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(b_d,     b_h.data(),     problem.b.totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(c_d,     c_h.data(),     problem.c.totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(d_d,     d_h.data(),     problem.d.totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(d_ref_d, d_ref_h.data(), problem.d.totalAllocatedBytes(), hipMemcpyHostToDevice));

        inputs.a = a_d;
        inputs.b = b_d;
        inputs.c = c_d;
        inputs.d = d_d;

        inputs.alpha = RandomInt<float>()();
        if(problem.useBeta)
            inputs.beta = RandomInt<float>()();
        else
            inputs.beta = 0;

        hardware = std::make_shared<AMDGPU>();

        rocblas_handle roc = nullptr;
        ASSERT_RB(rocblas_create_handle(&roc));

        for(int i = 0; i < problem.blas_batchCount(); i++)
        {
            size_t a_offset = problem.a.index(0,0,i);
            size_t b_offset = problem.b.index(0,0,i);
            size_t d_offset = problem.d.index(0,0,i);

            ASSERT_RB(rocblas_sgemm(roc, rocblas_operation_none, rocblas_operation_none,
                                    problem.blas_m(), problem.blas_n(), problem.blas_k(),
                                    &inputs.alpha, a_d + a_offset, problem.a.storedStride(1),
                                    b_d + b_offset, problem.b.storedStride(1),
                                    &inputs.beta, d_ref_d + d_offset, problem.d.storedStride(1)));
        }

        HIP_CHECK_EXC(hipMemcpy(d_ref_h.data(), d_ref_d, problem.d.totalAllocatedBytes(), hipMemcpyDeviceToHost));

        ASSERT_RB(rocblas_destroy_handle(roc));

	}
	
    void TearDown() override
    {
        hipFree(a_d);
        hipFree(b_d);
        hipFree(c_d);
        hipFree(d_d);
        hipFree(d_ref_d);
    }
};

TEST_P(GEMMTest, Simple)
{
    GEMMProblem problem = GetParam();
    GEMMSolution solution;

    solution.kernelName = "Cijk_Ailk_Bljk_SB_MT128x128x08_K1";

    solution.workGroupSize = Tensile::dim3{256,1,1};
    solution.macroTile = Tensile::dim3{128,128,1};
    solution.debugKernel = false;

    std::vector<KernelInvocation> result = solution.solve(problem, inputs, *hardware);

    hip::SolutionAdapter adapter;
    adapter.loadCodeObjectFile(
            "test/code_object/1_BenchmarkProblems/Cijk_Ailk_Bljk_SB_00/00_Final/source/assembly/Cijk_Ailk_Bljk_SB_MT128x128x08_K1.co");

    adapter.launchKernels(result);

    HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, problem.d.totalAllocatedBytes(), hipMemcpyDeviceToHost));

    //std::cout << "A:";
    //WriteTensor(std::cout, a_h.data(), problem.a);

    //std::cout << "B:";
    //WriteTensor(std::cout, b_h.data(), problem.b);

    //std::cout << "C Input:";
    //WriteTensor(std::cout, c_h.data(), problem.c);

    //std::cout << "C Reference:";
    //WriteTensor(std::cout, d_ref_h.data(), problem.d);

    //std::cout << "C Result:";
    //WriteTensor(std::cout, d_h.data(), problem.c);

    for(int i = 0; i < d_ref_h.size(); i++)
    {
        ASSERT_FLOAT_EQ(d_h[i], d_ref_h[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(HipSolutionAdapter, GEMMTest,
        ::testing::Values(GEMMProblem::FromBLAS(false, false,  234,  123,  634,  245,  768,  249, true, false, 12),
                          //GEMMProblem::FromBLAS(false, false, 5760, 5760, 5760, 5760, 5760, 5760, true, false,  4),
                          GEMMProblem::FromBLAS(false, false,    4,    4,    6,    4,    6,    4, true, false,  2)));


