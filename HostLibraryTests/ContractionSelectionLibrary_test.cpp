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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/Distance.hpp>

using namespace Tensile;

TEST(ContractionSelectionLibraryTest, Single)
{
    std::shared_ptr<Hardware> hardware = std::make_shared<AMDGPU>(AMDGPU::Processor::gfx900, 64, "AMD Radeon Vega Frontier Edition");

    SingleContractionLibrary lib;

    lib.solution = std::make_shared<ContractionSolution>();

    auto problem = std::make_shared<ContractionProblem>();

    EXPECT_EQ(lib.findBestSolution(*problem, *hardware), lib.solution);
}

TEST(ContractionSelectionLibraryTest, GPUSelection)
{
    std::shared_ptr<Hardware> v10 = std::make_shared<AMDGPU>(AMDGPU::Processor::gfx900, 64, "AMD Radeon Vega Frontier Edition");
    std::shared_ptr<Hardware> v20 = std::make_shared<AMDGPU>(AMDGPU::Processor::gfx906, 60, "AMD Radeon Vega 7");

    auto v20Solution = std::make_shared<ContractionSolution>();
    auto genericSolution = std::make_shared<ContractionSolution>();

    std::shared_ptr<ContractionLibrary> v20Lib = std::make_shared<SingleContractionLibrary>(v20Solution);
    auto genericLib = std::make_shared<SingleContractionLibrary>(genericSolution);

    auto isV20 = std::make_shared<Predicates::GPU::ProcessorEqual>(AMDGPU::Processor::gfx906);
    std::shared_ptr<Predicates::Predicate<Hardware>> isAMDGPUV20 =
        std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(isV20);
    HardwarePredicate hardwareIsAMDGPUV20(isAMDGPUV20);

    ContractionHardwareSelectionLibrary::Row v20Row(hardwareIsAMDGPUV20, v20Lib);
    ContractionHardwareSelectionLibrary lib({v20Row});

    auto problem = std::make_shared<ContractionProblem>();

    EXPECT_EQ(lib.findBestSolution(*problem, *v20), v20Solution);
    EXPECT_EQ(lib.findBestSolution(*problem, *v10), std::shared_ptr<ContractionSolution>());

    HardwarePredicate allHardware(std::make_shared<Predicates::True<Hardware>>());
    lib.rows.push_back(std::make_pair(allHardware, genericLib));

    EXPECT_EQ(lib.findBestSolution(*problem, *v20), v20Solution);
    EXPECT_EQ(lib.findBestSolution(*problem, *v10), genericSolution);
}

TEST(ContractionSelectionLibraryTest, TransposeSelection)
{
    auto NNSolution = std::make_shared<ContractionSolution>();
    auto NTSolution = std::make_shared<ContractionSolution>();
    auto TNSolution = std::make_shared<ContractionSolution>();
    auto TTSolution = std::make_shared<ContractionSolution>();

    NNSolution->index = 0;
    NTSolution->index = 1;
    TNSolution->index = 2;
    TTSolution->index = 3;

    SolutionMap<ContractionSolution> map({{0, NNSolution}, {1, NTSolution}, {2, TNSolution}, {3, TTSolution}});

    std::shared_ptr<ContractionLibrary> NNLibrary = std::make_shared<SingleContractionLibrary>(NNSolution);
    std::shared_ptr<ContractionLibrary> NTLibrary = std::make_shared<SingleContractionLibrary>(NTSolution);
    std::shared_ptr<ContractionLibrary> TNLibrary = std::make_shared<SingleContractionLibrary>(TNSolution);
    std::shared_ptr<ContractionLibrary> TTLibrary = std::make_shared<SingleContractionLibrary>(TTSolution);

    auto lib = std::make_shared<ContractionProblemMapLibrary>();

    lib->property = std::make_shared<Contraction::OperationIdentifier>();
    lib->map["Contraction_l_Ailk_Bljk_Cijk_Dijk"] = NNLibrary;
    lib->map["Contraction_l_Ailk_Bjlk_Cijk_Dijk"] = NTLibrary;
    lib->map["Contraction_l_Alik_Bljk_Cijk_Dijk"] = TNLibrary;
    lib->map["Contraction_l_Alik_Bjlk_Cijk_Dijk"] = TTLibrary;

    AMDGPU gpu;

    auto NNProblem = ContractionProblem::GEMM(false, false, 4,4,4, 4,4,4, 1.2, false, 1);
    auto NTProblem = ContractionProblem::GEMM(false,  true, 4,4,4, 4,4,4, 1.2, false, 1);
    auto TNProblem = ContractionProblem::GEMM( true, false, 4,4,4, 4,4,4, 1.2, false, 1);
    auto TTProblem = ContractionProblem::GEMM( true,  true, 4,4,4, 4,4,4, 1.2, false, 1);

    //auto WeirdProblemC = ContractionProblem::FromBLAS( true,  true, 4,4,4, 4,4,4, false, false, 1);
    //WeirdProblemC.c.transpose(0,1);

    //auto WeirdProblemD = ContractionProblem::FromBLAS( true,  true, 4,4,4, 4,4,4, false, false, 1);
    //WeirdProblemD.d.transpose(0,1);

    EXPECT_EQ(lib->findBestSolution(NNProblem, gpu), NNSolution);
    EXPECT_EQ(lib->findBestSolution(NTProblem, gpu), NTSolution);
    EXPECT_EQ(lib->findBestSolution(TNProblem, gpu), TNSolution);
    EXPECT_EQ(lib->findBestSolution(TTProblem, gpu), TTSolution);

    //EXPECT_EQ(lib->findBestSolution(WeirdProblemC, gpu), nullptr);
    //EXPECT_EQ(lib->findBestSolution(WeirdProblemD, gpu), nullptr);

    MasterContractionLibrary mlib;
    mlib.solutions = map;
    mlib.library = lib;
}

#if 0
TEST(ContractionSelectionLibraryTest, Caching)
{

    auto Solution0 = std::make_shared<ContractionSolution>();
    auto Solution1 = std::make_shared<ContractionSolution>();
    auto Solution2 = std::make_shared<ContractionSolution>();
    auto Solution3 = std::make_shared<ContractionSolution>();

    Solution0->index = 0;
    Solution1->index = 1;
    Solution2->index = 2;
    Solution3->index = 3;

    SolutionMap<ContractionSolution> map({{0, Solution0}, {1, Solution1}, {2, Solution2}, {3, Solution3}});

    std::shared_ptr<ContractionLibrary> Library0 = std::make_shared<SingleContractionLibrary>(Solution0);
    std::shared_ptr<ContractionLibrary> Library1 = std::make_shared<SingleContractionLibrary>(Solution1);
    std::shared_ptr<ContractionLibrary> Library2 = std::make_shared<SingleContractionLibrary>(Solution2);
    std::shared_ptr<ContractionLibrary> Library3 = std::make_shared<SingleContractionLibrary>(Solution3);

    AMDGPU gpu;

    auto Problem0 = ContractionProblem::GEMM(false, false, 4,4,4, 4,4,4, 1.2, false, 1);
    auto Problem1 = ContractionProblem::GEMM(false,  false, 6,6,6, 6,6,6, 1.2, false, 1);
    auto Problem2 = ContractionProblem::GEMM( true, false, 14,4,4, 4,4,4, 1.2, false, 1);
    auto Problem3 = ContractionProblem::GEMM( true,  true, 24,4,4, 4,4,4, 1.2, false, 1);

    using Key = std::array<size_t, 4>;
    using Table = Matching::DistanceMatchingTable<Key, ContractionProblem, std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>, std::shared_ptr<ContractionSolution>>;
    using Properties = std::vector<std::shared_ptr<Property<ContractionProblem>>>;

    Properties properties;

    std::shared_ptr<Contraction::FreeSizeA> freeSizeA = std::make_shared<Contraction::FreeSizeA>(); freeSizeA->index = 0; properties.push_back(freeSizeA);
    std::shared_ptr<Contraction::FreeSizeB> freeSizeB = std::make_shared<Contraction::FreeSizeB>(); freeSizeB->index = 0; properties.push_back(freeSizeB);
    std::shared_ptr<Contraction::BatchSize> batchSize = std::make_shared<Contraction::BatchSize>(); batchSize->index = 0; properties.push_back(batchSize);
    std::shared_ptr<Contraction::BoundSize> boundSize = std::make_shared<Contraction::BoundSize>(); boundSize->index = 0; properties.push_back(boundSize);

    std::shared_ptr<Table> matchingTable = std::make_shared<Table>(properties);

    using Entry = Matching::MatchingTableEntry<Key, std::shared_ptr<SolutionLibrary<ContractionProblem, ContractionSolution>>>;

    std::vector<Entry> table; 

    Entry map0; map0.key = {4,4,1,4}; map0.value = Library0; map0.speed = 1.0; table.push_back(map0);
    Entry map1; map1.key = {6,6,1,6}; map1.value = Library1; map1.speed = 1.0; table.push_back(map1);
    Entry map2; map2.key = {14,4,1,4}; map2.value = Library2; map2.speed = 1.0; table.push_back(map2);
    Entry map3; map3.key = {24,4,1,4}; map3.value = Library3; map3.speed = 1.0; table.push_back(map3);

    matchingTable->table = table;

    Matching::EuclideanDistance<std::array<size_t, 4>>  d;
    std::shared_ptr<Matching::EuclideanDistance<std::array<size_t, 4>>> pdistance = std::make_shared<Matching::EuclideanDistance<std::array<size_t, 4>>>(d);

    matchingTable->distance = pdistance;

    ProblemMatchingLibrary<ContractionProblem, ContractionSolution> lib;	    
    lib.table = matchingTable;

    
    auto theSolution0 = lib.findBestSolution(Problem0, gpu); 
    EXPECT_EQ(theSolution0, Solution0);
    auto theSolution0_cached = lib.findSolutionInCache(Problem0, gpu); 
    EXPECT_EQ(theSolution0, theSolution0_cached);

    auto theSolution1 = lib.findBestSolution(Problem1, gpu); 
    EXPECT_EQ(theSolution1, Solution1);
    auto theSolution1_cached = lib.findSolutionInCache(Problem1, gpu);
    EXPECT_EQ(theSolution1, theSolution1_cached);

    auto theSolution2 = lib.findBestSolution(Problem2, gpu); 
    EXPECT_EQ(theSolution2, Solution2);
    auto theSolution2_cached = lib.findSolutionInCache(Problem2, gpu);
    EXPECT_EQ(theSolution2, theSolution2_cached);

    auto theSolution3 = lib.findBestSolution(Problem3, gpu); 
    EXPECT_EQ(theSolution3, Solution3);
    auto theSolution3_cached = lib.findSolutionInCache(Problem3, gpu);
    EXPECT_EQ(theSolution3, theSolution3_cached);

}
#endif 
