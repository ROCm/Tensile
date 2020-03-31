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

#include <Tensile/AMDGPU_Detail.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

#include "TestData.hpp"
#include "TestUtils.hpp"

#include <Reference.hpp>

#include <cstddef>
#include <random>
#include <unordered_map>

using namespace Tensile;

#define ASSERT_RB(exp) ASSERT_EQ((exp), rocblas_status_success)

namespace Tensile {
    namespace hip {
        inline std::ostream & operator<<(std::ostream & stream, std::shared_ptr<SolutionAdapter> const& ptr)
        {
            if(ptr)
                return stream << "*" << *ptr;
            else
                return stream << "(nullptr)";
        }
    }
}

struct RunGEMMKernelTest: public ::testing::TestWithParam<
                          std::tuple<ContractionProblem,
                                     std::tuple<std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                                                std::shared_ptr<hip::SolutionAdapter>,
                                                bool  // is a solution required?
                                               >,
                                     bool  // align allocations to start or end of pages?
                                    >
                          >
{
    std::vector<float> a_h;
    std::vector<float> b_h;
    std::vector<float> c_h;
    std::vector<float> d_h;
    std::vector<float> d_in_h;
    std::vector<float> d_ref_h;

    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;
    float *d_d = nullptr;
    float *d_ref_d = nullptr;

    float *a_d_alloc = nullptr;
    float *b_d_alloc = nullptr;
    float *c_d_alloc = nullptr;
    float *d_d_alloc = nullptr;
    float *d_ref_d_alloc = nullptr;

    TypedContractionInputs<float> inputs;

    static std::unordered_map<ContractionProblem, std::vector<float>> referenceCache;

    std::shared_ptr<Hardware> hardware;

	void SetUp() override
	{
        HIP_CHECK_EXC(hipSetDevice(0));
        ContractionProblem problem = std::get<0>(GetParam());
        bool startOfPage = std::get<2>(GetParam());
        //std::cout << problem << std::endl;

        a_h.resize(problem.a().totalAllocatedElements());
        b_h.resize(problem.b().totalAllocatedElements());
        c_h.resize(problem.c().totalAllocatedElements());
        d_h.resize(problem.d().totalAllocatedElements());
        d_in_h.resize(problem.d().totalAllocatedElements());

        std::mt19937 rng;

        InitTensor(a_h.data(),    problem.a(), RandomInt<float>(), rng);
        InitTensor(b_h.data(),    problem.b(), RandomAlternatingInt<float>(), rng);
        InitTensor(c_h.data(),    problem.c(), RandomInt<float>(), rng);
        InitTensor(d_in_h.data(), problem.d(), RandomInt<float>(), rng);

        //InitTensor(a_h.data(), problem.a, Iota<float>());
        //InitTensor(b_h.data(), problem.b, Iota<float>());
        //InitTensor(c_h.data(), problem.c, RandomInt<float>());
        //InitTensor(d_h.data(), problem.d, RandomInt<float>());

        d_ref_h = d_h;

        CopyTensor(d_ref_h.data(), c_h.data(), problem.d(), problem.c());

        constexpr size_t pageSize = 4 * 1024 * 1024;

        size_t aSize = RoundUpToMultiple(problem.a().totalAllocatedBytes(), pageSize);
        size_t bSize = RoundUpToMultiple(problem.b().totalAllocatedBytes(), pageSize);
        size_t cSize = RoundUpToMultiple(problem.c().totalAllocatedBytes(), pageSize);
        size_t dSize = RoundUpToMultiple(problem.d().totalAllocatedBytes(), pageSize);

        HIP_CHECK_EXC(hipMalloc(&a_d_alloc,     aSize));
        HIP_CHECK_EXC(hipMalloc(&b_d_alloc,     bSize));
        HIP_CHECK_EXC(hipMalloc(&c_d_alloc,     cSize));
        HIP_CHECK_EXC(hipMalloc(&d_d_alloc,     dSize));
        HIP_CHECK_EXC(hipMalloc(&d_ref_d_alloc, dSize));

        if(startOfPage)
        {
            a_d = a_d_alloc;
            b_d = b_d_alloc;
            c_d = c_d_alloc;
            d_d = d_d_alloc;
            d_ref_d = d_ref_d_alloc;
        }
        else
        {
            a_d     = a_d_alloc     + ((aSize - problem.a().totalAllocatedBytes()) / sizeof(float));
            b_d     = b_d_alloc     + ((bSize - problem.b().totalAllocatedBytes()) / sizeof(float));
            c_d     = c_d_alloc     + ((cSize - problem.c().totalAllocatedBytes()) / sizeof(float));
            d_d     = d_d_alloc     + ((dSize - problem.d().totalAllocatedBytes()) / sizeof(float));
            d_ref_d = d_ref_d_alloc + ((dSize - problem.d().totalAllocatedBytes()) / sizeof(float));
        }

        HIP_CHECK_EXC(hipMemcpy(a_d,     a_h.data(),     problem.a().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(b_d,     b_h.data(),     problem.b().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(c_d,     c_h.data(),     problem.c().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(d_d,     d_in_h.data(),  problem.d().totalAllocatedBytes(), hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(d_ref_d, d_ref_h.data(), problem.d().totalAllocatedBytes(), hipMemcpyHostToDevice));

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

        auto iter = referenceCache.find(problem);
        if(iter == referenceCache.end())
        {
            TypedContractionInputs<float> inputsRefHost;
            inputsRefHost.a = a_h.data();
            inputsRefHost.b = b_h.data();
            inputsRefHost.c = c_h.data();
            inputsRefHost.d = d_ref_h.data();
            inputsRefHost.alpha = inputs.alpha;
            inputsRefHost.beta = inputs.beta;

            Client::SolveCPU(problem, inputsRefHost);

            referenceCache[problem] = d_ref_h;
        }
        else
        {
            d_ref_h = iter->second;
        }
        
	}
	
    void TearDown() override
    {
        hipFree(a_d_alloc);
        hipFree(b_d_alloc);
        hipFree(c_d_alloc);
        hipFree(d_d_alloc);
        hipFree(d_ref_d_alloc);

        hipDeviceReset();
    }
};

std::unordered_map<ContractionProblem, std::vector<float>> RunGEMMKernelTest::referenceCache;

TEST_P(RunGEMMKernelTest, BestSolution)
{
    bool debug = Debug::Instance().printPredicateEvaluation();

    auto params = GetParam();

    ContractionProblem problem = std::get<0>(params);


    std::shared_ptr<SolutionLibrary<ContractionProblem>> library;
    std::shared_ptr<hip::SolutionAdapter> adapter;
    bool requiredMatch;

    std::tie(library, adapter, requiredMatch) = std::get<1>(params);
    if(debug)
        std::cout << problem << std::endl << adapter << std::endl;

    ASSERT_NE(library, nullptr);
    ASSERT_NE(adapter, nullptr);

    auto solution = library->findBestSolution(problem, *hardware);

    if(requiredMatch)
    {
        ASSERT_NE(solution, nullptr);
    }
    else
    {
        if(!solution)
            return;
    }

    {
        ASSERT_NE(solution->problemPredicate, nullptr);
        EXPECT_EQ((*solution->problemPredicate)(problem), true);

        std::ostringstream msg;
        ASSERT_EQ(solution->problemPredicate->debugEval(problem, msg), true) << msg.str();
    }

    {
        ASSERT_NE(solution->hardwarePredicate, nullptr);
        EXPECT_EQ((*solution->hardwarePredicate)(*hardware), true);

        std::ostringstream msg;
        ASSERT_EQ(solution->hardwarePredicate->debugEval(*hardware, msg), true) << msg.str();
    }

    if(debug)
    {
        std::cout << "a: " << std::hex << inputs.a << ".." << inputs.a + problem.a().totalAllocatedElements() << std::endl;
        std::cout << "b: " << std::hex << inputs.b << ".." << inputs.b + problem.b().totalAllocatedElements() << std::endl;
        std::cout << "c: " << std::hex << inputs.c << ".." << inputs.c + problem.c().totalAllocatedElements() << std::endl;
        std::cout << "d: " << std::hex << inputs.d << ".." << inputs.d + problem.d().totalAllocatedElements() << std::endl;
    }

    std::vector<KernelInvocation> result = solution->solve(problem, inputs, *hardware);

    adapter->launchKernels(result);

    HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

    /*
    std::cout << "alpha: " << inputs.alpha << ", beta: " << inputs.beta
              << ", transA: " << problem.transA() << ", transB: " << problem.transB() << std::endl;
    std::cout << "A:";
    WriteTensor(std::cout, a_h.data(), problem.a());

    std::cout << "B:";
    WriteTensor(std::cout, b_h.data(), problem.b());

    std::cout << "C Input:";
    WriteTensor(std::cout, c_h.data(), problem.c());

    std::cout << "D Input:";
    WriteTensor(std::cout, d_in_h.data(), problem.c());

    std::cout << "D Reference:";
    WriteTensor(std::cout, d_ref_h.data(), problem.d());

    std::cout << "D Result:";
    WriteTensor(std::cout, d_h.data(), problem.c());
    */

    bool fail = false;

#pragma omp parallel for
    for(int i = 0; i < d_ref_h.size(); i++)
    {
#pragma omp flush(fail)
        if(!fail)
            EXPECT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i << ": " << (fail=true);
    }
    ASSERT_EQ(fail, false);
}

TEST_P(RunGEMMKernelTest, AllSolutions)
{
    bool debug = Debug::Instance().printPredicateEvaluation();

    auto params = GetParam();

    ContractionProblem problem = std::get<0>(params);

    std::shared_ptr<SolutionLibrary<ContractionProblem>> library;
    std::shared_ptr<hip::SolutionAdapter> adapter;
    bool requiredMatch;

    std::tie(library, adapter, requiredMatch) = std::get<1>(params);

    ASSERT_NE(library, nullptr);
    ASSERT_NE(adapter, nullptr);

    auto solutions = library->findAllSolutions(problem, *hardware);

    if(requiredMatch)
    {
        ASSERT_GT(solutions.size(), 0);
    }

    for(auto const& solution : solutions)
    {
        ASSERT_NE(solution, nullptr);

        {
            ASSERT_NE(solution->problemPredicate, nullptr);
            EXPECT_EQ((*solution->problemPredicate)(problem), true);

            std::ostringstream msg;
            ASSERT_EQ(solution->problemPredicate->debugEval(problem, msg), true) << msg.str();
            if(debug)
                std::cout << msg.str() << std::endl;
        }

        {
            ASSERT_NE(solution->hardwarePredicate, nullptr);
            EXPECT_EQ((*solution->hardwarePredicate)(*hardware), true);

            std::ostringstream msg;
            ASSERT_EQ(solution->hardwarePredicate->debugEval(*hardware, msg), true) << msg.str();
                if(debug)
                    std::cout << msg.str() << std::endl;
        }

        //std::cout << solution->name() << std::endl;

        std::vector<KernelInvocation> result = solution->solve(problem, inputs, *hardware);

        adapter->launchKernels(result);

        HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));

        /*
        std::cout << "alpha: " << inputs.alpha << ", beta: " << inputs.beta
                  << ", transA: " << problem.transA() << ", transB: " << problem.transB() << std::endl;
        std::cout << "A:";
        WriteTensor(std::cout, a_h.data(), problem.a());

        std::cout << "B:";
        WriteTensor(std::cout, b_h.data(), problem.b());

        std::cout << "C Input:";
        WriteTensor(std::cout, c_h.data(), problem.c());

        std::cout << "D Input:";
        WriteTensor(std::cout, d_in_h.data(), problem.c());

        std::cout << "D Reference:";
        WriteTensor(std::cout, d_ref_h.data(), problem.d());

        std::cout << "D Result:";
        WriteTensor(std::cout, d_h.data(), problem.c());
        */

        bool fail = false;

#pragma omp parallel for
        for(int i = 0; i < d_ref_h.size(); i++)
        {
#pragma omp flush(fail)
            if(!fail)
                EXPECT_FLOAT_EQ(d_h[i], d_ref_h[i]) << i << ": " << (fail=true);
        }
        ASSERT_EQ(fail, false);

    }

}

std::vector<ContractionProblem> TestProblems()
{
    return std::vector<ContractionProblem>
    {
        //ContractionProblem::GEMM(false, false, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, false,  4),
        //ContractionProblem::GEMM(false,  true, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, false,  4),
        //ContractionProblem::GEMM( true, false, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, false,  4),
        //ContractionProblem::GEMM( true,  true, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, false,  4),
          ContractionProblem::GEMM(false, false,    4,    4,    6,    4,    6,    4, 1.5, false,  2),
          ContractionProblem::GEMM(false,  true,    4,    4,    6,    4,    4,    4, 1.5, false,  2),
          ContractionProblem::GEMM( true, false,    4,    4,    6,    6,    6,    4, 1.5, false,  2),
          ContractionProblem::GEMM( true,  true,    4,    4,    6,    6,    4,    4, 1.5, false,  2),

          ContractionProblem::GEMM(false, false,   15,   15,   15,   15,   15,   15, 1.5, false, 1),
          ContractionProblem::GEMM(false,  true,   15,   15,   15,   15,   15,   15, 1.5, false, 1),
          ContractionProblem::GEMM( true, false,   15,   15,   15,   15,   15,   15, 1.5, false, 1),
          ContractionProblem::GEMM( true,  true,   15,   15,   15,   15,   15,   15, 1.5, false, 1),

          ContractionProblem::GEMM(false, false,   16,   16,   16,   16,   16,   16, 1.5, false, 1),
          ContractionProblem::GEMM(false,  true,   16,   16,   16,   16,   16,   16, 1.5, false, 1),
          ContractionProblem::GEMM( true, false,   16,   16,   16,   16,   16,   16, 1.5, false, 1),
          ContractionProblem::GEMM( true,  true,   16,   16,   16,   16,   16,   16, 1.5, false, 1),

          ContractionProblem::GEMM(false, false,   17,   17,   17,   17,   17,   17, 1.5, false, 1),
          ContractionProblem::GEMM(false,  true,   17,   17,   17,   17,   17,   17, 1.5, false, 1),
          ContractionProblem::GEMM( true, false,   17,   17,   17,   17,   17,   17, 1.5, false, 1),
          ContractionProblem::GEMM( true,  true,   17,   17,   17,   17,   17,   17, 1.5, false, 1),

          ContractionProblem::GEMM(false, false,   31,   31,   31,   31,   31,   31, 1.5, false, 1),
          ContractionProblem::GEMM(false,  true,   31,   31,   31,   31,   31,   31, 1.5, false, 1),
          ContractionProblem::GEMM( true, false,   31,   31,   31,   31,   31,   31, 1.5, false, 1),
          ContractionProblem::GEMM( true,  true,   31,   31,   31,   31,   31,   31, 1.5, false, 1),

          ContractionProblem::GEMM(false, false,   32,   32,   32,   32,   32,   32, 1.5, false, 1),
          ContractionProblem::GEMM(false,  true,   32,   32,   32,   32,   32,   32, 1.5, false, 1),
          ContractionProblem::GEMM( true, false,   32,   32,   32,   32,   32,   32, 1.5, false, 1),
          ContractionProblem::GEMM( true,  true,   32,   32,   32,   32,   32,   32, 1.5, false, 1),

          ContractionProblem::GEMM(false, false,   33,   33,   33,   33,   33,   33, 1.5, false, 1),
          ContractionProblem::GEMM(false,  true,   33,   33,   33,   33,   33,   33, 1.5, false, 1),
          ContractionProblem::GEMM( true, false,   33,   33,   33,   33,   33,   33, 1.5, false, 1),
          ContractionProblem::GEMM( true,  true,   33,   33,   33,   33,   33,   33, 1.5, false, 1),

          ContractionProblem::GEMM(false, false,   34,   34,   34,   34,   34,   34, 1.5, false, 1),
          ContractionProblem::GEMM(false,  true,   34,   34,   34,   34,   34,   34, 1.5, false, 1),
          ContractionProblem::GEMM( true, false,   34,   34,   34,   34,   34,   34, 1.5, false, 1),
          ContractionProblem::GEMM( true,  true,   34,   34,   34,   34,   34,   34, 1.5, false, 1),

          ContractionProblem::GEMM(false, false,  234,  123,  634,  234,  634,  234, 1.5, false, 1),
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
          RandomGEMM(),

          ContractionProblem::GEMM(false, false,    1,    4,    6,    1,    6,    1, 1.5, false,  1),
          ContractionProblem::GEMM(false, false,    4,    1,    6,    4,    6,    4, 1.5, false,  1),
          ContractionProblem::GEMM(false, false,    4,    4,    1,    4,    1,    4, 1.5, false,  1),

          ContractionProblem::GEMM(false,  true,    1,    4,    6,    1,    4,    1, 1.5, false,  1),
          ContractionProblem::GEMM(false,  true,    4,    1,    6,    4,    1,    4, 1.5, false,  1),
          ContractionProblem::GEMM(false,  true,    4,    4,    1,    4,    4,    4, 1.5, false,  1),

          ContractionProblem::GEMM( true, false,    1,    4,    6,    6,    6,    1, 1.5, false,  1),
          ContractionProblem::GEMM( true, false,    4,    1,    6,    6,    6,    4, 1.5, false,  1),
          ContractionProblem::GEMM( true, false,    4,    4,    1,    1,    1,    4, 1.5, false,  1),

          ContractionProblem::GEMM( true,  true,    1,    4,    6,    6,    4,    1, 1.5, false,  1),
          ContractionProblem::GEMM( true,  true,    4,    1,    6,    6,    1,    4, 1.5, false,  1),
          ContractionProblem::GEMM( true,  true,    4,    4,    1,    1,    4,    4, 1.5, false,  1),

          ContractionProblem::GEMM(false,  true,    1,  128,  256,    1,  270, 49928, 1.5, false, 1),
          ContractionProblem::GEMM(false,  true,  384,    1,  384,  384,  270, 49928, 1.5, false, 1),
          ContractionProblem::GEMM( true,  true,    4,    4,    1,    1,    4,     4, 1.5, false, 1)
    };
}

std::vector<std::tuple<std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                       std::shared_ptr<hip::SolutionAdapter>,
                       bool>>
TestLibraries()
{
    bool debug = Debug::Instance().printKernelArguments();

    std::vector<std::tuple<std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                           std::shared_ptr<hip::SolutionAdapter>,
                           bool>
               > rv;

    {
        auto library = EmbeddedLibrary<ContractionProblem>::Get("kernels_lite");
        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "kernels_lite");
        adapter->loadEmbeddedCodeObjects("kernels_lite");
        rv.emplace_back(library, adapter, false);
    }

    {
        auto library = EmbeddedLibrary<ContractionProblem>::Get("kernels_lite_mixed");
        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "kernels_lite_mixed");
        adapter->loadEmbeddedCodeObjects("kernels_lite_mixed");
        rv.emplace_back(library, adapter, true);
    }

    {
        auto library = LoadLibraryFile<ContractionProblem>(TestData::Instance().file("kernels_lite/TensileLibrary.yaml").native());
        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "kernels_lite (file)");
        for(auto file: TestData::Instance().glob("kernels_lite/*.*co"))
            adapter->loadCodeObjectFile(file.native());

        rv.emplace_back(library, adapter, false);
    }

    {
        auto library = LoadLibraryFile<ContractionProblem>(TestData::Instance().file("kernels_lite_mixed/TensileLibrary.yaml").native());
        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "kernels_lite_mixed (file)");
        for(auto file: TestData::Instance().glob("kernels_lite_mixed/*.*co"))
            adapter->loadCodeObjectFile(file.native());

        rv.emplace_back(library, adapter, true);
    }

    {
        auto library = LoadLibraryFile<ContractionProblem>(TestData::Instance().file("tile_aware_selection/library/TensileLibrary.yaml").native());

        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "tile_aware_selection");
        for(auto file: TestData::Instance().glob("tile_aware_selection/library/*.*co"))
            adapter->loadCodeObjectFile(file.native());

        for(auto file: TestData::Instance().glob("tile_aware_selection/library/*.*hsaco"))
            adapter->loadCodeObjectFile(file.native());

        rv.emplace_back(library, adapter, false);
    }

    auto envDir = TestData::Env("TENSILE_TEST_LIBRARY");
    if(envDir)
    {
        auto library = LoadLibraryFile<ContractionProblem>(envDir.file("TensileLibrary.yaml").native());
        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "TENSILE_TEST_LIBRARY");

        for(auto file: envDir.glob("*.co"))
        {
            adapter->loadCodeObjectFile(file.native());
        }

        for(auto file: envDir.glob("*.hsaco"))
        {
            adapter->loadCodeObjectFile(file.native());
        }

        rv.emplace_back(library, adapter, false);
    }


    return rv;
}

INSTANTIATE_TEST_SUITE_P(HipSolutionAdapter, RunGEMMKernelTest,
        ::testing::Combine(::testing::ValuesIn(TestProblems()),
                           ::testing::ValuesIn(TestLibraries()),
                           ::testing::Values(false, true)));


