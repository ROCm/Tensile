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

#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/Utils.hpp>

#include <TestUtils.hpp>

#include <random>
#include <rocblas.h>

using namespace Tensile;

#define ASSERT_RB(exp) ASSERT_EQ((exp), rocblas_status_success)


struct RunGEMMKernelTest: public ::testing::TestWithParam<ContractionProblem>
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

    TypedContractionInputs<float> inputs;

    std::shared_ptr<Hardware> hardware;

	void SetUp() override
	{
        HIP_CHECK_EXC(hipSetDevice(0));
        ContractionProblem problem = GetParam();

        a_h.resize(problem.a().totalAllocatedElements());
        b_h.resize(problem.b().totalAllocatedElements());
        c_h.resize(problem.c().totalAllocatedElements());
        d_h.resize(problem.d().totalAllocatedElements());

        InitTensor(a_h.data(), problem.a(), RandomInt<float>());
        InitTensor(b_h.data(), problem.b(), RandomAlternatingInt<float>());
        InitTensor(c_h.data(), problem.c(), RandomInt<float>());
        InitTensor(d_h.data(), problem.d(), RandomInt<float>());

        //InitTensor(a_h.data(), problem.a, Iota<float>());
        //InitTensor(b_h.data(), problem.b, Iota<float>());
        //InitTensor(c_h.data(), problem.c, RandomInt<float>());
        //InitTensor(d_h.data(), problem.d, RandomInt<float>());

        d_ref_h = d_h;

        CopyTensor(d_ref_h.data(), c_h.data(), problem.d(), problem.c());

        HIP_CHECK_EXC(hipMalloc(&a_d,     problem.a().totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&b_d,     problem.b().totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&c_d,     problem.c().totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&d_d,     problem.d().totalAllocatedBytes()));
        HIP_CHECK_EXC(hipMalloc(&d_ref_d, problem.d().totalAllocatedBytes()));

        HIP_CHECK_EXC(hipMemcpy(a_d,     a_h.data(),     problem.a().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(b_d,     b_h.data(),     problem.b().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(c_d,     c_h.data(),     problem.c().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(d_d,     d_h.data(),     problem.d().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(d_ref_d, d_ref_h.data(), problem.d().totalAllocatedBytes(), hipMemcpyHostToDevice));

        inputs.a = a_d;
        inputs.b = b_d;
        inputs.c = c_d;
        inputs.d = d_d;

        inputs.alpha = RandomInt<float>()();
        if(problem.beta() == 1.0)
            inputs.beta = 1.0;
        else if(problem.beta() == 0.0)
            inputs.beta = 0.0;
        else
            inputs.beta = RandomInt<float>()();

        hardware = hip::GetCurrentDevice();
        ASSERT_NE(hardware, nullptr);

        rocblas_handle roc = nullptr;
        ASSERT_RB(rocblas_create_handle(&roc));

        for(int i = 0; i < problem.batchSize(0); i++)
        {
            size_t a_offset = problem.a().index(0,0,i);
            size_t b_offset = problem.b().index(0,0,i);
            size_t d_offset = problem.d().index(0,0,i);

            auto transA = problem.transA() ? rocblas_operation_transpose : rocblas_operation_none;
            auto transB = problem.transB() ? rocblas_operation_transpose : rocblas_operation_none;

            ASSERT_RB(rocblas_sgemm(roc, transA, transB,
                                    problem.freeSizeA(0), problem.freeSizeB(0), problem.boundSize(0),
                                    &inputs.alpha, a_d + a_offset, problem.a().strides()[1],
                                    b_d + b_offset, problem.b().strides()[1],
                                    &inputs.beta, d_ref_d + d_offset, problem.d().strides()[1]));
        }

        HIP_CHECK_EXC(hipMemcpy(d_ref_h.data(), d_ref_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

        ASSERT_RB(rocblas_destroy_handle(roc));

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

TEST_P(RunGEMMKernelTest, Simple)
{
    ContractionProblem problem = GetParam();
    if(problem.transA() || problem.transB()) return;

    ContractionSolution solution;

    solution.kernelName = "Cijk_Ailk_Bljk_SB_MT128x128x08_K1";

    solution.sizeMapping.workGroupSize = Tensile::dim3{256,1,1};
    solution.sizeMapping.macroTile = Tensile::dim3{128,128,1};
    solution.sizeMapping.depthU = 8;
    solution.sizeMapping.globalSplitU = 1;
    solution.sizeMapping.staggerStrideShift = 3;
    solution.sizeMapping.staggerU = 32;
    solution.sizeMapping.workGroupMapping = 8;

    solution.debugKernel = false;

    std::cout << "A: " << problem.a() << std::endl;
    std::cout << "B: " << problem.b() << std::endl;
    std::cout << "C: " << problem.c() << std::endl;
    std::cout << "D: " << problem.d() << std::endl;

    std::vector<KernelInvocation> result = solution.solve(problem, inputs, *hardware);

    hip::SolutionAdapter adapter(false);
    adapter.loadCodeObjectFile(
            "test/hip/code_object/1_BenchmarkProblems/Cijk_Ailk_Bljk_SB_00/00_Final/source/assembly/Cijk_Ailk_Bljk_SB_MT128x128x08_K1.co");
           //test/hip/code_object/1_BenchmarkProblems/Cijk_Ailk_Bjlk_SB_00/00_Final/source/assembly/Cijk_Ailk_Bjlk_SB_MT128x128x08_K1.co

    adapter.launchKernels(result);

    HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

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

TEST_P(RunGEMMKernelTest, Library)
{
    ContractionProblem problem = GetParam();

    auto library = LoadLibraryFile<ContractionProblem, ContractionSolution>("configs/TensileKernels.yaml");

    ASSERT_NE(library, nullptr);

    auto solution = library->findBestSolution(problem, *hardware);

    ASSERT_NE(solution, nullptr);

    std::vector<KernelInvocation> result = solution->solve(problem, inputs, *hardware);

    hip::SolutionAdapter adapter(true);
    adapter.loadCodeObjectFile("configs/TensileKernels.co");

    adapter.launchKernels(result);

    HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

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
        ASSERT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i;
    }
}

TEST_P(RunGEMMKernelTest, EmbeddedLibraryLite)
{
    ContractionProblem problem = GetParam();

    auto library = EmbeddedLibrary<ContractionProblem>::Get("kernels_lite");

    ASSERT_NE(library, nullptr);

    auto solution = library->findBestSolution(problem, *hardware);

    ASSERT_NE(solution, nullptr);

    std::vector<KernelInvocation> result = solution->solve(problem, inputs, *hardware);

    hip::SolutionAdapter adapter(true);
    adapter.loadEmbeddedCodeObjects("kernels_lite");

    adapter.launchKernels(result);

    HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

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
        ASSERT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i;
    }
}

TEST_P(RunGEMMKernelTest, KernelsLite)
{
    ContractionProblem problem = GetParam();

    //auto library = LoadLibraryFile<ContractionProblem, ContractionSolution>("configs/KernelsLite.yaml");
    auto library = EmbeddedLibrary<ContractionProblem>::Get("kernels_lite");

    ASSERT_NE(library, nullptr);

    auto solution = library->findBestSolution(problem, *hardware);

    ASSERT_NE(solution, nullptr);

    std::cout << solution->name() << std::endl;

    std::vector<KernelInvocation> result = solution->solve(problem, inputs, *hardware);

    hip::SolutionAdapter adapter(false);
    //adapter.loadCodeObjectFile("configs/KernelsLite.co");
    adapter.loadEmbeddedCodeObjects("kernels_lite");

    adapter.launchKernels(result);

    HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

    /*
    std::cout << "A:";
    WriteTensor(std::cout, a_h.data(), problem.a());

    std::cout << "B:";
    WriteTensor(std::cout, b_h.data(), problem.b());

    std::cout << "C Input:";
    WriteTensor(std::cout, c_h.data(), problem.c());

    std::cout << "C Reference:";
    WriteTensor(std::cout, d_ref_h.data(), problem.d());

    std::cout << "C Result:";
    WriteTensor(std::cout, d_h.data(), problem.c());
    */

    for(int i = 0; i < d_ref_h.size(); i++)
    {
        ASSERT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i;
    }
}

TEST_P(RunGEMMKernelTest, KernelsLiteExhaustive)
{
    ContractionProblem problem = GetParam();

    auto library = LoadLibraryFile<ContractionProblem>("configs/lite_library/Tensile/TensileLibrary.yaml");

    hip::SolutionAdapter adapter(false);
    adapter.loadCodeObjectFile("configs/lite_library/Tensile/TensileLibrary.co");

    ASSERT_NE(library, nullptr);

    auto solutions = library->findAllSolutions(problem, *hardware);

    ASSERT_GT(solutions.size(), 0);

    for(auto const& solution : solutions)
    {
        ASSERT_NE(solution, nullptr);

        std::cout << solution->name() << std::endl;

        std::vector<KernelInvocation> result = solution->solve(problem, inputs, *hardware);

        //OldTensile::CallOldTensile(problem.transA(), problem.transB(),
        //                           inputs.d, inputs.c, inputs.a, inputs.b,
        //                           inputs.alpha, inputs.beta,
        //                           problem.d().strides()[1], problem.d().strides()[2],
        //                           problem.a().strides()[1], problem.a().strides()[2],
        //                           problem.b().strides()[1], problem.b().strides()[2],
        //                           problem.d().sizes()[0],
        //                           problem.d().sizes()[1],
        //                           problem.d().sizes()[2],
        //                           problem.boundSize(0),
        //                           0, 0, nullptr, nullptr);

        adapter.launchKernels(result);

        HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

        /*
        std::cout << "A:";
        WriteTensor(std::cout, a_h.data(), problem.a());

        std::cout << "B:";
        WriteTensor(std::cout, b_h.data(), problem.b());

        std::cout << "C Input:";
        WriteTensor(std::cout, c_h.data(), problem.c());

        std::cout << "C Reference:";
        WriteTensor(std::cout, d_ref_h.data(), problem.d());

        std::cout << "C Result:";
        WriteTensor(std::cout, d_h.data(), problem.c());
        */

        for(int i = 0; i < d_ref_h.size(); i++)
        {
            ASSERT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(HipSolutionAdapter, RunGEMMKernelTest,
        ::testing::Values(
                        //ContractionProblem::GEMM(false, false, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, false,  4),
                        //ContractionProblem::GEMM(false,  true, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, false,  4),
                        //ContractionProblem::GEMM( true, false, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, false,  4),
                        //ContractionProblem::GEMM( true,  true, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, false,  4),
                          ContractionProblem::GEMM(false, false,    4,    4,    6,    4,    6,    4, 1.5, false,  2),
                          ContractionProblem::GEMM(false,  true,    4,    4,    6,    4,    4,    4, 1.5, false,  2),
                          ContractionProblem::GEMM( true, false,    4,    4,    6,    6,    6,    4, 1.5, false,  2),
                          ContractionProblem::GEMM( true,  true,    4,    4,    6,    6,    4,    4, 1.5, false,  2),
                          ContractionProblem::GEMM(false, false,  234,  123,  634,  245,  768,  249, 1.5, false, 12),
                          ContractionProblem::GEMM(false,  true,  234,  123,  634,  245,  768,  249, 1.5, false, 12),
                          ContractionProblem::GEMM( true, false,  234,  123,  634,  768,  768,  249, 1.5, false, 12),
                          ContractionProblem::GEMM( true,  true,  234,  123,  634,  768,  768,  249, 1.5, false, 12),
                          RandomGEMM(),
                          RandomGEMM(),
                          RandomGEMM(),
                          RandomGEMM(),
                          RandomGEMM(),
                          RandomGEMM(),
                          RandomGEMM(),
                          RandomGEMM(),
                          RandomGEMM()
                          ));


