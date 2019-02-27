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
#include <Tensile/GEMMLibrary.hpp>
#include <Tensile/GEMMProblemPredicates.hpp>
#include <Tensile/llvm/YAML.hpp>

using namespace Tensile;

TEST(GEMMSelectionLibraryTest, Single)
{
    std::shared_ptr<Hardware> hardware = std::make_shared<AMDGPU>(AMDGPU::Processor::gfx900, 64, "AMD Radeon Vega Frontier Edition");

    SingleGEMMLibrary lib;

    lib.solution = std::make_shared<GEMMSolution>();

    auto problem = std::make_shared<GEMMProblem>();

    EXPECT_EQ(lib.findBestSolution(*problem, *hardware), lib.solution);
}

TEST(GEMMSelectionLibraryTest, GPUSelection)
{
    std::shared_ptr<Hardware> v10 = std::make_shared<AMDGPU>(AMDGPU::Processor::gfx900, 64, "AMD Radeon Vega Frontier Edition");
    std::shared_ptr<Hardware> v20 = std::make_shared<AMDGPU>(AMDGPU::Processor::gfx906, 60, "AMD Radeon Vega 7");

    auto v20Solution = std::make_shared<GEMMSolution>();
    auto genericSolution = std::make_shared<GEMMSolution>();

    std::shared_ptr<GEMMLibrary> v20Lib = std::make_shared<SingleGEMMLibrary>(v20Solution);
    auto genericLib = std::make_shared<SingleGEMMLibrary>(genericSolution);

    auto isV20 = std::make_shared<Predicates::GPU::ProcessorEqual>(AMDGPU::Processor::gfx906);
    std::shared_ptr<Predicates::Predicate<Hardware>> isAMDGPUV20 =
        std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(isV20);
    HardwarePredicate hardwareIsAMDGPUV20(isAMDGPUV20);

    GEMMHardwareSelectionLibrary::Row v20Row(hardwareIsAMDGPUV20, v20Lib);
    GEMMHardwareSelectionLibrary lib({v20Row});

    auto problem = std::make_shared<GEMMProblem>();

    EXPECT_EQ(lib.findBestSolution(*problem, *v20), v20Solution);
    EXPECT_EQ(lib.findBestSolution(*problem, *v10), std::shared_ptr<GEMMSolution>());

    HardwarePredicate allHardware(std::make_shared<Predicates::True<Hardware>>());
    lib.rows.push_back(std::make_pair(allHardware, genericLib));

    EXPECT_EQ(lib.findBestSolution(*problem, *v20), v20Solution);
    EXPECT_EQ(lib.findBestSolution(*problem, *v10), genericSolution);
}

TEST(GEMMSelectionLibraryTest, TransposeSelection)
{
    using namespace Predicates;
    using namespace Predicates::GEMM;

    auto NNSolution = std::make_shared<GEMMSolution>();
    auto NTSolution = std::make_shared<GEMMSolution>();
    auto TNSolution = std::make_shared<GEMMSolution>();
    auto TTSolution = std::make_shared<GEMMSolution>();

    NNSolution->index = 0;
    NTSolution->index = 1;
    TNSolution->index = 2;
    TTSolution->index = 3;

    SolutionMap<GEMMSolution> map({{0, NNSolution}, {1, NTSolution}, {2, TNSolution}, {3, TTSolution}});

    std::shared_ptr<GEMMLibrary> NNLibrary = std::make_shared<SingleGEMMLibrary>(NNSolution);
    std::shared_ptr<GEMMLibrary> NTLibrary = std::make_shared<SingleGEMMLibrary>(NTSolution);
    std::shared_ptr<GEMMLibrary> TNLibrary = std::make_shared<SingleGEMMLibrary>(TNSolution);
    std::shared_ptr<GEMMLibrary> TTLibrary = std::make_shared<SingleGEMMLibrary>(TTSolution);

    std::vector<size_t> N{0,1,2};
    std::vector<size_t> T{1,0,2};

    auto IsNN = GEMMProblemPredicate(std::make_shared<And<GEMMProblem>>
    (std::vector<std::shared_ptr<Predicate<GEMMProblem>>>{
        std::make_shared<ADimensionOrder>(N),
        std::make_shared<BDimensionOrder>(N),
        std::make_shared<CDimensionOrder>(N),
        std::make_shared<DDimensionOrder>(N)
    }));

    auto IsNT = GEMMProblemPredicate(std::make_shared<And<GEMMProblem>>
    (std::vector<std::shared_ptr<Predicate<GEMMProblem>>>{
        std::make_shared<ADimensionOrder>(N),
        std::make_shared<BDimensionOrder>(T),
        std::make_shared<CDimensionOrder>(N),
        std::make_shared<DDimensionOrder>(N)
    }));

    auto IsTN = GEMMProblemPredicate(std::make_shared<And<GEMMProblem>>
    (std::vector<std::shared_ptr<Predicate<GEMMProblem>>>{
        std::make_shared<ADimensionOrder>(T),
        std::make_shared<BDimensionOrder>(N),
        std::make_shared<CDimensionOrder>(N),
        std::make_shared<DDimensionOrder>(N)
    }));

    auto IsTT = GEMMProblemPredicate(std::make_shared<And<GEMMProblem>>
    (std::vector<std::shared_ptr<Predicate<GEMMProblem>>>{
        std::make_shared<ADimensionOrder>(T),
        std::make_shared<BDimensionOrder>(T),
        std::make_shared<CDimensionOrder>(N),
        std::make_shared<DDimensionOrder>(N)
    }));

    std::vector<GEMMProblemSelectionLibrary::Row> rows{
        std::make_pair(IsNN, NNLibrary),
        std::make_pair(IsNT, NTLibrary),
        std::make_pair(IsTN, TNLibrary),
        std::make_pair(IsTT, TTLibrary)
    };

    auto lib = std::make_shared<GEMMProblemSelectionLibrary>(rows);

    AMDGPU gpu;

    auto NNProblem = GEMMProblem::FromBLAS(false, false, 4,4,4, 4,4,4, false, false, 1);
    //auto NTProblem = GEMMProblem::FromBLAS(false,  true, 4,4,4, 4,4,4, false, false, 1);
    //auto TNProblem = GEMMProblem::FromBLAS( true, false, 4,4,4, 4,4,4, false, false, 1);
    //auto TTProblem = GEMMProblem::FromBLAS( true,  true, 4,4,4, 4,4,4, false, false, 1);

    //auto WeirdProblemC = GEMMProblem::FromBLAS( true,  true, 4,4,4, 4,4,4, false, false, 1);
    //WeirdProblemC.c.transpose(0,1);

    //auto WeirdProblemD = GEMMProblem::FromBLAS( true,  true, 4,4,4, 4,4,4, false, false, 1);
    //WeirdProblemD.d.transpose(0,1);

    EXPECT_EQ(lib->findBestSolution(NNProblem, gpu), NNSolution);
    //EXPECT_EQ(lib->findBestSolution(NTProblem, gpu), NTSolution);
    //EXPECT_EQ(lib->findBestSolution(TNProblem, gpu), TNSolution);
    //EXPECT_EQ(lib->findBestSolution(TTProblem, gpu), TTSolution);

    //EXPECT_EQ(lib->findBestSolution(WeirdProblemC, gpu), nullptr);
    //EXPECT_EQ(lib->findBestSolution(WeirdProblemD, gpu), nullptr);

    MasterGEMMLibrary mlib;
    mlib.solutions = map;
    mlib.library = lib;
    llvm::yaml::Output yout(llvm::outs());
    yout << mlib;
}

