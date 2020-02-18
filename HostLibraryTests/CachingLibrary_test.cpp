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

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>

#include <memory>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

TEST(Cache, Simple)
{
    using namespace Tensile;

    CacheMap<int, int> cache(-1);

    EXPECT_EQ(-1, cache.find(5));

    cache.add(5, 7);
    EXPECT_EQ(7, cache.find(5));

    cache.add(5, 9);
    EXPECT_EQ(7, cache.find(5));
}

TEST(Cache, Threaded)
{
    using namespace Tensile;
    CacheMap<int, int> cache(-1);

    #pragma omp parallel num_threads(32)
    {
        int seed = 0;
#ifdef _OPENMP
        seed = omp_get_thread_num();
#endif
        std::uniform_int_distribution<int> dist(0,10);
        std::mt19937 rng(seed);

        for(int i = 0; i < 10000; i++)
        {
            int key = dist(rng);
            int value = key+1;

            int lookup = cache.find(key);
            if(lookup != -1)
                EXPECT_EQ(lookup, value);

            cache.add(key, value);
            EXPECT_EQ(value, cache.find(key));
        }
    }
}

TEST(Hashing, TensorDescriptor)
{
    using namespace Tensile;

    TensorDescriptor a(DataType::Float, {15, 8, 20});
    TensorDescriptor b(DataType::Int32, {15, 8, 20});
    TensorDescriptor c(DataType::Float, {15, 8, 20}, {1, 15, 15*8});
    TensorDescriptor d(DataType::Float, {15, 8, 20}, {1, 17, 19*8});

    EXPECT_NE(std::hash<TensorDescriptor>()(a), std::hash<TensorDescriptor>()(b));
    EXPECT_EQ(std::hash<TensorDescriptor>()(a), std::hash<TensorDescriptor>()(c));

    EXPECT_NE(std::hash<TensorDescriptor>()(c), std::hash<TensorDescriptor>()(d));
}

TEST(Hashing, ContractionProblem)
{
    using namespace Tensile;

    ContractionProblem a = ContractionProblem::GEMM(false, true, 5, 7, 9, 5, 5, 5, 3.0, false, 5);
    ContractionProblem b = ContractionProblem::GEMM(false, true, 5, 7, 9, 7, 5, 5, 3.0, false, 5);
    ContractionProblem c = ContractionProblem::GEMM(false, true, 5, 7, 9, 0, 5, 5, 3.0, false, 5);
    ContractionProblem d = ContractionProblem::GEMM(false, true, 5, 7, 9, 5, 5, 5, 3.0, false, 5);

    EXPECT_NE(std::hash<ContractionProblem>()(a), std::hash<ContractionProblem>()(b));
    EXPECT_NE(std::hash<ContractionProblem>()(a), std::hash<ContractionProblem>()(c));

    EXPECT_EQ(std::hash<ContractionProblem>()(a), std::hash<ContractionProblem>()(d));
}

TEST(Hashing, AMDGPU)
{
    using namespace Tensile;

    std::vector<AMDGPU::Processor> processors{
        AMDGPU::Processor::gfx803,
        AMDGPU::Processor::gfx900,
        AMDGPU::Processor::gfx906,
        AMDGPU::Processor::gfx908};

    std::vector<int> counts{16, 20, 40, 56, 60, 64};

    // There aren't that many possible combinations here so it's reasonable to
    // have no hash collisions.
    for(AMDGPU::Processor p1: processors)
    for(AMDGPU::Processor p2: processors)
    for(int c1: counts)
    for(int c2: counts)
    {
        AMDGPU g1(p1, c1, "g1");
        AMDGPU g2(p2, c2, "g2");

        if(p1 != p2 || c1 != c2)
        {
            EXPECT_NE(std::hash<AMDGPU>()(g1), std::hash<AMDGPU>()(g2)) << g1 << "/" << g2;
        }
        else
        {
            EXPECT_EQ(std::hash<AMDGPU>()(g1), std::hash<AMDGPU>()(g2)) << g1 << "/" << g2;
        }
    }
}

TEST(Hashing, Tuple)
{
    using namespace Tensile;

    using TwoInts = std::tuple<int, int>;

    TwoInts tup(4, 5);

    EXPECT_EQ(std::hash<TwoInts>()(tup), hash_combine(5, 4));
}

TEST(Hashing, Tuple2)
{
    using namespace Tensile;

    using TwoInts = std::tuple<ContractionProblem, AMDGPU>;

    TwoInts tup;

    size_t h = std::hash<TwoInts>()(tup);
}

TEST(CachingLibrary, Simple)
{
    using namespace Tensile;

    auto Solution0 = std::make_shared<ContractionSolution>();
    auto Solution1 = std::make_shared<ContractionSolution>();
    auto Solution2 = std::make_shared<ContractionSolution>();
    auto Solution3 = std::make_shared<ContractionSolution>();

    Solution0->index = 0;
    Solution1->index = 1;
    Solution2->index = 2;
    Solution3->index = 3;

    SolutionMap<ContractionSolution> map(
        {{0, Solution0}, {1, Solution1}, {2, Solution2}, {3, Solution3}});

    auto Library0 = std::make_shared<SingleContractionLibrary>(Solution0);
    auto Library1 = std::make_shared<SingleContractionLibrary>(Solution1);
    auto Library2 = std::make_shared<SingleContractionLibrary>(Solution2);
    auto Library3 = std::make_shared<SingleContractionLibrary>(Solution3);

    AMDGPU gpu;

    auto Problem0 = ContractionProblem::GEMM(false, false, 4,4,4,  4,4,4, 1.2, false, 1);
    auto Problem1 = ContractionProblem::GEMM(false, false, 6,6,6,  6,6,6, 1.2, false, 1);
    auto Problem2 = ContractionProblem::GEMM( true, false, 14,4,4, 4,4,4, 1.2, false, 1);
    auto Problem3 = ContractionProblem::GEMM( true,  true, 24,4,4, 4,4,4, 1.2, false, 1);

    using Key = std::array<size_t, 4>;
    using Table =
        Matching::DistanceMatchingTable<
            Key,
            ContractionProblem,
            std::shared_ptr<SolutionLibrary<ContractionProblem>>,
            std::shared_ptr<ContractionSolution>>;
    using Properties = std::vector<std::shared_ptr<Property<ContractionProblem>>>;

    Properties properties;

    {
        auto freeSizeA = std::make_shared<Contraction::FreeSizeA>(); freeSizeA->index = 0; properties.push_back(freeSizeA);
        auto freeSizeB = std::make_shared<Contraction::FreeSizeB>(); freeSizeB->index = 0; properties.push_back(freeSizeB);
        auto batchSize = std::make_shared<Contraction::BatchSize>(); batchSize->index = 0; properties.push_back(batchSize);
        auto boundSize = std::make_shared<Contraction::BoundSize>(); boundSize->index = 0; properties.push_back(boundSize);
    }

    std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

    using Entry = Matching::MatchingTableEntry< Key, std::shared_ptr<SolutionLibrary<ContractionProblem>>>;

    std::vector<Entry> table;

    {
        Entry map0{{ 4,4,1,4}, Library0, 1.0}; table.push_back(map0);
        Entry map1{{ 6,6,1,6}, Library1, 1.0}; table.push_back(map1);
        Entry map2{{14,4,1,4}, Library2, 1.0}; table.push_back(map2);
        Entry map3{{24,4,1,4}, Library3, 1.0}; table.push_back(map3);
    }

    matchingTable->table = table;

    using Distance = Matching::EuclideanDistance<std::array<size_t, 4>>;
    std::shared_ptr<Matching::EuclideanDistance<std::array<size_t, 4>>>
        pdistance = std::make_shared<Distance>();

    matchingTable->distance = pdistance;

    auto subLib = std::make_shared<
        ProblemMatchingLibrary<ContractionProblem>>();

    subLib->table = matchingTable;

    CachingLibrary<ContractionProblem> lib(subLib);

    EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem0, gpu));
    auto theSolution0 = lib.findBestSolution(Problem0, gpu);
    EXPECT_EQ(theSolution0, Solution0);
    auto theSolution0_cached = lib.findSolutionInCache(Problem0, gpu);
    EXPECT_EQ(theSolution0, theSolution0_cached);

    EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem1, gpu));
    auto theSolution1 = lib.findBestSolution(Problem1, gpu);
    EXPECT_EQ(theSolution1, Solution1);
    auto theSolution1_cached = lib.findSolutionInCache(Problem1, gpu);
    EXPECT_EQ(theSolution1, theSolution1_cached);

    EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem2, gpu));
    auto theSolution2 = lib.findBestSolution(Problem2, gpu);
    EXPECT_EQ(theSolution2, Solution2);
    auto theSolution2_cached = lib.findSolutionInCache(Problem2, gpu);
    EXPECT_EQ(theSolution2, theSolution2_cached);

    EXPECT_EQ(nullptr, lib.findSolutionInCache(Problem3, gpu));
    auto theSolution3 = lib.findBestSolution(Problem3, gpu);
    EXPECT_EQ(theSolution3, Solution3);
    auto theSolution3_cached = lib.findSolutionInCache(Problem3, gpu);
    EXPECT_EQ(theSolution3, theSolution3_cached);
}
