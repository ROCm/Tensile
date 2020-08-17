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

namespace Tensile
{
    namespace hip
    {
        inline std::ostream& operator<<(std::ostream&                           stream,
                                        std::shared_ptr<SolutionAdapter> const& ptr)
        {
            if(ptr)
                return stream << "*" << *ptr;
            else
                return stream << "(nullptr)";
        }

    } // namespace hip
} // namespace Tensile

using ProblemParams = std::tuple<bool, //   transA
                                 bool, //   transB
                                 size_t, // m
                                 size_t, // n
                                 size_t, // k
                                 size_t, // lda
                                 size_t, // ldb
                                 size_t, // ldc
                                 double, // beta
                                 size_t>; // batchCount

ProblemParams RandomGEMMParams{false, false, -1, -1, -1, -1, -1, -1, -1.0, -1};

using SolutionParams = std::tuple<std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                                  std::shared_ptr<hip::SolutionAdapter>,
                                  bool>; // is a solution required?

enum class MemoryPageAlignment : int
{
    BEGIN = 0,
    END   = 1
};

template <typename T>
inline void expectEqual(T const& l, T const& r, size_t i, bool& fail)
{
    if(!fail)
    {
        EXPECT_EQ(l, r) << i << ": " << (fail = true);
    }
}

template <>
inline void expectEqual(float const& l, float const& r, size_t i, bool& fail)
{
    if(!fail)
    {
        EXPECT_FLOAT_EQ(l, r) << i << ": " << (fail = true);
    }
}

template <>
inline void expectEqual(double const& l, double const& r, size_t i, bool& fail)
{
    if(!fail)
    {
        EXPECT_DOUBLE_EQ(l, r) << i << ": " << (fail = true);
    }
}

template <typename T>
void expectEqual(std::vector<T> const& l, std::vector<T> const& r)
{
    bool fail = false;

    auto size = std::min(l.size(), r.size());

#pragma omp parallel for
    for(size_t i = 0; i < size; i++)
    {
#pragma omp flush(fail)
        expectEqual(l[i], r[i], i, fail);
    }
    ASSERT_EQ(fail, false);
}

struct GEMMKernelTest
{
    virtual void SetUp(ProblemParams const&, SolutionParams const&, MemoryPageAlignment const&) = 0;
    virtual void TestBestSolution()                                                             = 0;
    virtual void TestAllSolutions()                                                             = 0;
    virtual void TearDown()                                                                     = 0;
    virtual void OverrideAlpha(double)                                                          = 0;
    virtual void NullifyAPtr()                                                                  = 0;
    virtual void NullifyBPtr()                                                                  = 0;
    virtual std::string ToString() const                                                        = 0;
};

inline std::ostream& operator<<(std::ostream& stream, std::shared_ptr<GEMMKernelTest> const& ptr)
{
    if(ptr)
        return stream << "*" << ptr->ToString();
    else
        return stream << "(nullptr)";
}

template <typename TypedInputs>
struct TypedGEMMKernelTest : public GEMMKernelTest
{
    // Extract type info
    using AType     = typename TypedInputs::AType;
    using BType     = typename TypedInputs::BType;
    using CType     = typename TypedInputs::CType;
    using DType     = typename TypedInputs::DType;
    using AlphaType = typename TypedInputs::AlphaType;
    using BetaType  = typename TypedInputs::BetaType;

    std::vector<AType> a_h;
    std::vector<BType> b_h;
    std::vector<CType> c_h;
    std::vector<DType> d_h;
    std::vector<DType> d_in_h;
    std::vector<DType> d_ref_h;

    AType* a_d     = nullptr;
    BType* b_d     = nullptr;
    CType* c_d     = nullptr;
    DType* d_d     = nullptr;
    DType* d_ref_d = nullptr;

    AType* a_d_alloc     = nullptr;
    BType* b_d_alloc     = nullptr;
    CType* c_d_alloc     = nullptr;
    DType* d_d_alloc     = nullptr;
    DType* d_ref_d_alloc = nullptr;

    TypedInputs inputs_h;
    TypedInputs inputs_d;

    std::shared_ptr<Hardware> hardware;

    // Test input components
    ContractionProblem                                   problem;
    std::shared_ptr<SolutionLibrary<ContractionProblem>> library;
    std::shared_ptr<hip::SolutionAdapter>                adapter;
    std::shared_ptr<ContractionSolution>                 solution;
    bool                                                 requiredMatch;
    MemoryPageAlignment                                  memoryAlignment;

    static std::unordered_map<size_t, std::vector<DType>> referenceCache;

    template <typename T>
    Tensile::DataType dataType() const
    {
        return TypeInfo<T>::Enum;
    }

    Tensile::ContractionProblem createProblem(ProblemParams const& props) const
    {
        if(props == RandomGEMMParams)
        {
            static std::mt19937 rng;

            std::uniform_int_distribution<int> random_bool(0, 1);
            // std::uniform_int_distribution<int> random_size(2,8192);
            std::uniform_int_distribution<int> random_padding(0, 32);
            std::uniform_int_distribution<int> random_batch(1, 10);
            std::uniform_int_distribution<int> random_beta(0, 2);

            std::uniform_real_distribution<double> random_size(1.0, std::log(8192.0));

            bool transA = random_bool(rng);
            bool transB = random_bool(rng);

            size_t m = std::exp(random_size(rng)) + 1;
            size_t n = std::exp(random_size(rng)) + 1;
            size_t k = std::exp(random_size(rng)) + 1;

            int    beta_category = random_beta(rng);
            double beta;
            if(beta_category == 0)
                beta = 0.0;
            else if(beta_category == 1)
                beta = 1.0;
            else
                beta = 1.2;

            auto random_pad = [&](size_t cols, size_t rows, size_t& ld, size_t& stride) {
                ld = cols;

                bool pad_ld = random_bool(rng);

                if(pad_ld)
                {
                    size_t padding = random_padding(rng);
                    if(padding == 0)
                        ld = RoundUpToMultiple<size_t>(ld, 128);
                    else
                        ld += padding;
                }

                stride = ld * rows;

                bool pad_stride = random_bool(rng);

                if(pad_stride)
                {
                    size_t padding = random_padding(rng);

                    if(padding == 0)
                        stride = RoundUpToMultiple<size_t>(stride, 256);
                    else
                        stride += padding;
                }
            };

            size_t lda, ldb, ldc, ldd;
            size_t strideA, strideB, strideC, strideD;

            if(transA)
                random_pad(k, m, lda, strideA);
            else
                random_pad(m, k, lda, strideA);

            if(transB)
                random_pad(n, k, ldb, strideB);
            else
                random_pad(k, n, ldb, strideB);

            random_pad(m, n, ldc, strideC);

            // ldd support not yet merged in.
            ldd     = ldc;
            strideD = strideC;
            // random_pad(m, n, ldd, strideD);

            size_t batchCount = random_batch(rng);

            return ContractionProblem::GEMM_Strides(transA,
                                                    transB,
                                                    dataType<AType>(),
                                                    dataType<BType>(),
                                                    dataType<CType>(),
                                                    dataType<DType>(),
                                                    m,
                                                    n,
                                                    k,
                                                    batchCount,
                                                    lda,
                                                    strideA,
                                                    ldb,
                                                    strideB,
                                                    ldc,
                                                    strideC,
                                                    ldd,
                                                    strideD,
                                                    beta);
        }

        bool   transA     = std::get<0>(props);
        bool   transB     = std::get<1>(props);
        size_t m          = std::get<2>(props);
        size_t n          = std::get<3>(props);
        size_t k          = std::get<4>(props);
        size_t lda        = std::get<5>(props);
        size_t ldb        = std::get<6>(props);
        size_t ldc        = std::get<7>(props);
        double beta       = std::get<8>(props);
        size_t batchCount = std::get<9>(props);

        ContractionProblem::FreeIndices free(2);
        ContractionProblem::BoundIndex  bound;

        free[0].isA = true;
        free[0].i = free[0].c = free[0].d = 0;
        free[1].isA                       = false;
        free[1].i = free[1].c = free[1].d = 1;

        TensorDescriptor a, b, c, d;
        if(transA)
        {
            a         = TensorDescriptor(dataType<AType>(), {k, m}, {1, lda});
            free[0].i = 1;
            bound.a   = 0;
        }
        else
        {
            a         = TensorDescriptor(dataType<AType>(), {m, k}, {1, lda});
            free[0].i = 0;
            bound.a   = 1;
        }

        if(transB)
        {
            b         = TensorDescriptor(dataType<BType>(), {n, k}, {1, ldb});
            free[1].i = 0;
            bound.b   = 1;
        }
        else
        {
            b         = TensorDescriptor(dataType<BType>(), {k, n}, {1, ldb});
            free[1].i = 1;
            bound.b   = 0;
        }

        ContractionProblem::FreeIndices  freeIndices{free};
        ContractionProblem::BatchIndices batchIndices;
        ContractionProblem::BoundIndices boundIndices{bound};

        d = TensorDescriptor(dataType<DType>(), {m, n}, {1, ldc});

        a.appendDim(batchCount);
        b.appendDim(batchCount);
        d.appendDim(batchCount);

        batchIndices.push_back({2, 2, 2, 2});

        c = d;

        TensorOps nop;

        return ContractionProblem(
            a, nop, b, nop, c, nop, d, nop, freeIndices, batchIndices, boundIndices, beta);
    }

    void SetUp(ProblemParams const&       probParams,
               SolutionParams const&      solParams,
               MemoryPageAlignment const& alignment) override
    {
        // Extract testing inputs
        problem                                   = createProblem(probParams);
        std::tie(library, adapter, requiredMatch) = solParams;
        memoryAlignment                           = alignment;

        HIP_CHECK_EXC(hipSetDevice(0));

        a_h.resize(problem.a().totalAllocatedElements());
        b_h.resize(problem.b().totalAllocatedElements());
        c_h.resize(problem.c().totalAllocatedElements());
        d_h.resize(problem.d().totalAllocatedElements());
        d_in_h.resize(problem.d().totalAllocatedElements());

        std::mt19937 rng;

        InitTensor(a_h.data(), problem.a(), RandomInt<AType>(), rng);
        InitTensor(b_h.data(), problem.b(), RandomAlternatingInt<BType>(), rng);
        InitTensor(c_h.data(), problem.c(), RandomInt<CType>(), rng);
        InitTensor(d_in_h.data(), problem.d(), RandomInt<DType>(), rng);

        d_ref_h = d_h;

        CopyTensor(d_ref_h.data(), c_h.data(), problem.d(), problem.c());

        constexpr size_t pageSize = 4 * 1024 * 1024;

        size_t aSize = RoundUpToMultiple(problem.a().totalAllocatedBytes(), pageSize);
        size_t bSize = RoundUpToMultiple(problem.b().totalAllocatedBytes(), pageSize);
        size_t cSize = RoundUpToMultiple(problem.c().totalAllocatedBytes(), pageSize);
        size_t dSize = RoundUpToMultiple(problem.d().totalAllocatedBytes(), pageSize);

        // Initialize device allocations and data
        HIP_CHECK_EXC(hipMalloc(&a_d_alloc, aSize));
        HIP_CHECK_EXC(hipMalloc(&b_d_alloc, bSize));
        HIP_CHECK_EXC(hipMalloc(&c_d_alloc, cSize));
        HIP_CHECK_EXC(hipMalloc(&d_d_alloc, dSize));
        HIP_CHECK_EXC(hipMalloc(&d_ref_d_alloc, dSize));

        if(alignment == MemoryPageAlignment::BEGIN)
        {
            a_d     = a_d_alloc;
            b_d     = b_d_alloc;
            c_d     = c_d_alloc;
            d_d     = d_d_alloc;
            d_ref_d = d_ref_d_alloc;
        }
        else
        {
            a_d     = a_d_alloc + ((aSize - problem.a().totalAllocatedBytes()) / sizeof(AType));
            b_d     = b_d_alloc + ((bSize - problem.b().totalAllocatedBytes()) / sizeof(BType));
            c_d     = c_d_alloc + ((cSize - problem.c().totalAllocatedBytes()) / sizeof(CType));
            d_d     = d_d_alloc + ((dSize - problem.d().totalAllocatedBytes()) / sizeof(DType));
            d_ref_d = d_ref_d_alloc + ((dSize - problem.d().totalAllocatedBytes()) / sizeof(DType));
        }

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

        // Initialize inputs for device calcs
        inputs_d.a = a_d;
        inputs_d.b = b_d;
        inputs_d.c = c_d;
        inputs_d.d = d_d;

        // Initialize alpha
        inputs_d.alpha = RandomInt<AlphaType>()(rng);

        // Initialize beta
        if(problem.beta() == 1.0)
        {
            inputs_d.beta = static_cast<BetaType>(1.0);
        }
        else if(problem.beta() == 0.0)
        {
            inputs_d.beta = static_cast<BetaType>(0.0);
        }
        else
        {
            inputs_d.beta = RandomInt<BetaType>()(rng);
        }

        // Initialize inputs for CPU reference calcs
        inputs_h.a     = a_h.data();
        inputs_h.b     = b_h.data();
        inputs_h.c     = c_h.data();
        inputs_h.d     = d_ref_h.data();
        inputs_h.alpha = inputs_d.alpha;
        inputs_h.beta  = inputs_d.beta;

        // Initialize hardware
        hardware = hip::GetCurrentDevice();
        ASSERT_NE(hardware, nullptr);
    }

    void OverrideAlpha(double val) override
    {
        inputs_h.alpha = inputs_d.alpha = static_cast<AlphaType>(val);
    }

    void NullifyAPtr() override
    {
        inputs_h.a = inputs_d.a = nullptr;
    }

    void NullifyBPtr() override
    {
        inputs_h.b = inputs_d.b = nullptr;
    }

    void calcCPU()
    {
        // Run reference CPU calc.
        // Hash keys use problem descr combined with alpha and beta
        // because they will affect the reference calcs.
        auto key  = Tensile::hash_combine(problem, inputs_h.alpha, inputs_h.beta);
        auto iter = referenceCache.find(key);
        if(iter == referenceCache.end())
        {
            Client::SolveCPU(problem, inputs_h);
            referenceCache[key] = d_ref_h;
        }
        else
        {
            d_ref_h = iter->second;
        }
    }

    void calcGPU()
    {
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

        if(Debug::Instance().printPredicateEvaluation())
        {
            std::cout << "a: " << std::hex << inputs_d.a << ".."
                      << inputs_d.a + problem.a().totalAllocatedElements() << std::endl;
            std::cout << "b: " << std::hex << inputs_d.b << ".."
                      << inputs_d.b + problem.b().totalAllocatedElements() << std::endl;
            std::cout << "c: " << std::hex << inputs_d.c << ".."
                      << inputs_d.c + problem.c().totalAllocatedElements() << std::endl;
            std::cout << "d: " << std::hex << inputs_d.d << ".."
                      << inputs_d.d + problem.d().totalAllocatedElements() << std::endl;
        }

        std::vector<KernelInvocation> result = solution->solve(problem, inputs_d, *hardware);

        adapter->launchKernels(result);

        HIP_CHECK_EXC(
            hipMemcpy(d_h.data(), d_d, problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));
    }

    void TestBestSolution() override
    {
        if(Debug::Instance().printPredicateEvaluation())
        {
            std::cout << problem << std::endl << adapter << std::endl;
        }

        ASSERT_NE(library, nullptr);
        ASSERT_NE(adapter, nullptr);

        solution = library->findBestSolution(problem, *hardware);

        if(requiredMatch)
        {
            ASSERT_NE(solution, nullptr);
        }
        else
        {
            if(!solution)
            {
                return;
            }
        }

        calcCPU();
        calcGPU();
        expectEqual(d_h, d_ref_h);
    }

    void TestAllSolutions() override
    {
        if(Debug::Instance().printPredicateEvaluation())
        {
            std::cout << problem << std::endl << adapter << std::endl;
        }

        ASSERT_NE(library, nullptr);
        ASSERT_NE(adapter, nullptr);

        auto solutions = library->findAllSolutions(problem, *hardware);

        if(requiredMatch)
        {
            ASSERT_GT(solutions.size(), 0);
        }

        calcCPU();

        for(auto const& testSolution : solutions)
        {
            solution = testSolution;
            calcGPU();
            expectEqual(d_h, d_ref_h);
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

        a_h.clear();
        b_h.clear();
        c_h.clear();
        d_h.clear();
        d_in_h.clear();
    }

    inline std::string ToString() const override
    {
        return std::string("TypedKernelShortCircuitTest<") + TypeInfo<AType>::Name() + ", "
               + TypeInfo<BType>::Name() + ", " + TypeInfo<CType>::Name() + ", "
               + TypeInfo<DType>::Name() + ", " + TypeInfo<AlphaType>::Name() + ", "
               + TypeInfo<BetaType>::Name() + ", " + std::string(">");
    }
};

template <typename TypedInputs>
using DType = typename TypedGEMMKernelTest<TypedInputs>::DType;

template <typename TypedInputs>
std::unordered_map<size_t, std::vector<DType<TypedInputs>>>
    TypedGEMMKernelTest<TypedInputs>::referenceCache;

namespace std
{
    template <>
    struct hash<std::complex<float>>
    {
        inline size_t operator()(std::complex<float> const& obj) const
        {
            return Tensile::hash_combine(obj.real(), obj.imag());
        }
    };
    template <>
    struct hash<std::complex<double>>
    {
        inline size_t operator()(std::complex<double> const& obj) const
        {
            return Tensile::hash_combine(obj.real(), obj.imag());
        }
    };

    template <>
    struct hash<Tensile::BFloat16>
    {
        inline size_t operator()(Tensile::BFloat16 const& obj) const
        {
            return hash<decltype(obj.data)>()(obj.data);
        }
    };

#ifdef TENSILE_USE_HALF

    template <>
    struct hash<Tensile::Half>
    {
        inline size_t operator()(Tensile::Half const& obj) const
        {
            return hash<float>()(static_cast<float>(obj));
        }
    };

#else

    template <>
    struct hash<Tensile::Half>
    {
        inline size_t operator()(Tensile::Half const& obj) const
        {
            return hash<decltype(obj.value)>()(obj.value);
        }
    };

#endif // TENSILE_USE_HALF

} // namespace std

struct RunGEMMKernelTest
    : public ::testing::TestWithParam<std::tuple<std::shared_ptr<GEMMKernelTest>,
                                                 ProblemParams,
                                                 SolutionParams,
                                                 MemoryPageAlignment>>
{
    void SetUp() override
    {
        auto param          = GetParam();
        auto typedTest      = std::get<0>(param);
        auto problemParams  = std::get<1>(param);
        auto solutionParams = std::get<2>(param);
        auto pageAlignment  = std::get<3>(param);
        typedTest->SetUp(problemParams, solutionParams, pageAlignment);
    }
    void TearDown() override
    {
        auto param     = GetParam();
        auto typedTest = std::get<0>(param);
        typedTest->TearDown();
    }
};

TEST_P(RunGEMMKernelTest, TestBestSolution)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->TestBestSolution();
}

TEST_P(RunGEMMKernelTest, TestAllSolutions)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->TestAllSolutions();
}

TEST_P(RunGEMMKernelTest, TestAlphaZero)
{
    auto param     = GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->OverrideAlpha(0.0);
    typedTest->TestBestSolution();
}

TEST_P(RunGEMMKernelTest, TestAlphaZeroABNull)
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

std::vector<std::shared_ptr<GEMMKernelTest>> TypedTests()
{
    static auto testFloat = std::make_shared<TypedGEMMKernelTest<TypedContractionInputs<float>>>();
    //     static auto testDouble = std::make_shared<TypedGEMMKernelTest<TypedContractionInputs<double>>>();
    //     static auto testCFloat = std::make_shared<TypedGEMMKernelTest<TypedContractionInputs<std::complex<float>>>>();
    //     static auto testCDouble = std::make_shared<TypedGEMMKernelTest<TypedContractionInputs<std::complex<double>>>>();
    //     static auto testInt8x4 = std::make_shared<TypedGEMMKernelTest<TypedContractionInputs<Int8x4, Int8x4, int32_t, int32_t>>>();
    //     static auto testInt32 = std::make_shared<TypedGEMMKernelTest<TypedContractionInputs<int32_t>>>();
    //     static auto testHalf = std::make_shared<TypedGEMMKernelTest<TypedContractionInputs<Tensile::Half>>>();
    // #ifdef TENSILE_USE_BF16
    //     static auto testBF16 = std::make_shared<TypedGEMMKernelTest<BFloat16ContractionInputs>>();
    // #endif
    return std::vector<std::shared_ptr<GEMMKernelTest>>{
        testFloat,
        //         testDouble,
        //         testCFloat,
        //         testCDouble,
        //         testInt8x4,
        //         testInt32,
        //         testHalf,
        // #ifdef TENSILE_USE_BF16
        //         testBF16,
        // #endif
    };
}

std::vector<ProblemParams> TestProblems()
{
    return std::vector<ProblemParams>{

        //{false, false, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, 4},
        //{false,  true, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, 4},
        //{ true, false, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, 4},
        //{ true,  true, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, 4},
        {false, false, 4, 4, 6, 4, 6, 4, 1.5, 2},
        {false, true, 4, 4, 6, 4, 4, 4, 1.5, 2},
        {true, false, 4, 4, 6, 6, 6, 4, 1.5, 2},
        {true, true, 4, 4, 6, 6, 4, 4, 1.5, 2},

        {false, false, 15, 15, 15, 15, 15, 15, 1.5, 1},
        {false, true, 15, 15, 15, 15, 15, 15, 1.5, 1},
        {true, false, 15, 15, 15, 15, 15, 15, 1.5, 1},
        {true, true, 15, 15, 15, 15, 15, 15, 1.5, 1},

        {false, false, 16, 16, 16, 16, 16, 16, 1.5, 1},
        {false, true, 16, 16, 16, 16, 16, 16, 1.5, 1},
        {true, false, 16, 16, 16, 16, 16, 16, 1.5, 1},
        {true, true, 16, 16, 16, 16, 16, 16, 1.5, 1},

        {false, false, 17, 17, 17, 17, 17, 17, 1.5, 1},
        {false, true, 17, 17, 17, 17, 17, 17, 1.5, 1},
        {true, false, 17, 17, 17, 17, 17, 17, 1.5, 1},
        {true, true, 17, 17, 17, 17, 17, 17, 1.5, 1},

        {false, false, 31, 31, 31, 31, 31, 31, 1.5, 1},
        {false, true, 31, 31, 31, 31, 31, 31, 1.5, 1},
        {true, false, 31, 31, 31, 31, 31, 31, 1.5, 1},
        {true, true, 31, 31, 31, 31, 31, 31, 1.5, 1},

        {false, false, 32, 32, 32, 32, 32, 32, 1.5, 1},
        {false, true, 32, 32, 32, 32, 32, 32, 1.5, 1},
        {true, false, 32, 32, 32, 32, 32, 32, 1.5, 1},
        {true, true, 32, 32, 32, 32, 32, 32, 1.5, 1},

        {false, false, 33, 33, 33, 33, 33, 33, 1.5, 1},
        {false, true, 33, 33, 33, 33, 33, 33, 1.5, 1},
        {true, false, 33, 33, 33, 33, 33, 33, 1.5, 1},
        {true, true, 33, 33, 33, 33, 33, 33, 1.5, 1},

        {false, false, 34, 34, 34, 34, 34, 34, 1.5, 1},
        {false, true, 34, 34, 34, 34, 34, 34, 1.5, 1},
        {true, false, 34, 34, 34, 34, 34, 34, 1.5, 1},
        {true, true, 34, 34, 34, 34, 34, 34, 1.5, 1},

        {false, false, 234, 123, 634, 234, 634, 234, 1.5, 1},
        {false, false, 234, 123, 634, 245, 768, 249, 1.5, 12},
        {false, true, 234, 123, 634, 245, 768, 249, 1.5, 12},
        {true, false, 234, 123, 634, 768, 768, 249, 1.5, 12},
        {true, true, 234, 123, 634, 768, 768, 249, 1.5, 12},
        RandomGEMMParams,
        RandomGEMMParams,
        RandomGEMMParams,
        RandomGEMMParams,
        RandomGEMMParams,
        RandomGEMMParams,
        RandomGEMMParams,
        RandomGEMMParams,
        RandomGEMMParams,
        {false, false, 1, 4, 6, 1, 6, 1, 1.5, 1},
        {false, false, 4, 1, 6, 4, 6, 4, 1.5, 1},
        {false, false, 4, 4, 1, 4, 1, 4, 1.5, 1},

        {false, true, 1, 4, 6, 1, 4, 1, 1.5, 1},
        {false, true, 4, 1, 6, 4, 1, 4, 1.5, 1},
        {false, true, 4, 4, 1, 4, 4, 4, 1.5, 1},

        {true, false, 1, 4, 6, 6, 6, 1, 1.5, 1},
        {true, false, 4, 1, 6, 6, 6, 4, 1.5, 1},
        {true, false, 4, 4, 1, 1, 1, 4, 1.5, 1},

        {true, true, 1, 4, 6, 6, 4, 1, 1.5, 1},
        {true, true, 4, 1, 6, 6, 1, 4, 1.5, 1},
        {true, true, 4, 4, 1, 1, 4, 4, 1.5, 1},

        //{false, true, 1, 128, 256, 1, 270, 49928, 1.5, 1},
        {false, true, 384, 1, 384, 384, 270, 49928, 1.5, 1},
        {true, true, 4, 4, 1, 1, 4, 4, 1.5, 1},

        {false, false, 16328, 384, 384, 16328, 384, 16328, 2.0, 1},
        {false, true, 16328, 384, 384, 16328, 16328, 16328, 2.0, 1},
        {true, false, 16328, 384, 384, 384, 384, 16328, 2.0, 1},
        {true, true, 16328, 384, 384, 384, 16328, 16328, 2.0, 1}};
}

std::vector<std::tuple<std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                       std::shared_ptr<hip::SolutionAdapter>,
                       bool>>
    TestLibraries()
{
    bool debug = Debug::Instance().printKernelArguments();

    std::vector<std::tuple<std::shared_ptr<SolutionLibrary<ContractionProblem>>,
                           std::shared_ptr<hip::SolutionAdapter>,
                           bool>>
        rv;

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
        rv.emplace_back(library, adapter, false);
    }

    {
        auto library = LoadLibraryFile<ContractionProblem>(
            TestData::Instance().file("kernels_lite/TensileLibrary").native());
        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "kernels_lite (file)");
        for(auto file : TestData::Instance().glob("kernels_lite/*.*co"))
            adapter->loadCodeObjectFile(file.native());

        rv.emplace_back(library, adapter, false);
    }

    {
        auto library = LoadLibraryFile<ContractionProblem>(
            TestData::Instance().file("kernels_lite_mixed/TensileLibrary").native());
        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "kernels_lite_mixed (file)");
        for(auto file : TestData::Instance().glob("kernels_lite_mixed/*.*co"))
            adapter->loadCodeObjectFile(file.native());

        rv.emplace_back(library, adapter, false);
    }

    {
        auto library = LoadLibraryFile<ContractionProblem>(
            TestData::Instance().file("tile_aware_selection/library/TensileLibrary").native());

        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "tile_aware_selection");
        for(auto file : TestData::Instance().glob("tile_aware_selection/library/*.*co"))
            adapter->loadCodeObjectFile(file.native());

        for(auto file : TestData::Instance().glob("tile_aware_selection/library/*.*hsaco"))
            adapter->loadCodeObjectFile(file.native());

        rv.emplace_back(library, adapter, false);
    }

    auto envDir = TestData::Env("TENSILE_TEST_LIBRARY");
    if(envDir)
    {
        auto library = LoadLibraryFile<ContractionProblem>(envDir.file("TensileLibrary").native());
        auto adapter = std::make_shared<hip::SolutionAdapter>(debug, "TENSILE_TEST_LIBRARY");

        for(auto file : envDir.glob("*.co"))
        {
            adapter->loadCodeObjectFile(file.native());
        }

        for(auto file : envDir.glob("*.hsaco"))
        {
            try
            {
                adapter->loadCodeObjectFile(file.native());
            }
            catch(std::logic_error& exc)
            {
            }
        }

        rv.emplace_back(library, adapter, false);
    }

    return rv;
}

std::vector<MemoryPageAlignment> TestMemoryAlignments()
{
    return std::vector<MemoryPageAlignment>{MemoryPageAlignment::BEGIN, MemoryPageAlignment::END};
}
INSTANTIATE_TEST_SUITE_P(HipSolutionAdapter,
                         RunGEMMKernelTest,
                         ::testing::Combine(::testing::ValuesIn(TypedTests()),
                                            ::testing::ValuesIn(TestProblems()),
                                            ::testing::ValuesIn(TestLibraries()),
                                            ::testing::ValuesIn(TestMemoryAlignments())));
