
#include <gtest/gtest.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include "TestData.hpp"

#include <tuple>

using namespace Tensile;

TEST(ContractionFitnessTest, MatchingSize)
{
    auto library
        = LoadLibraryFile<ContractionProblem>(TestData::Instance().file("KernelsLite").native());
    ASSERT_NE(library, nullptr);

    AMDGPU hardware;

    {
        ContractionProblem p
            = ContractionProblem::GEMM(false, false, 64, 64, 256, 64, 64, 256, 1.0, false, 2);

        double fitness  = -1.0; //Initialize to fail test
        auto   solution = library->findBestSolution(p, hardware, &fitness);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(fitness, 0.0);
    }
}

TEST(ContractionFitnessTest, NonMatchingSize)
{
    auto library
        = LoadLibraryFile<ContractionProblem>(TestData::Instance().file("KernelsLite").native());
    ASSERT_NE(library, nullptr);

    AMDGPU hardware;

    {
        ContractionProblem p
            = ContractionProblem::GEMM(false, false, 65, 64, 256, 65, 64, 256, 1.0, false, 2);

        double fitness  = 0.0; //Initialize to fail test
        auto   solution = library->findBestSolution(p, hardware, &fitness);

        ASSERT_NE(solution, nullptr);
        EXPECT_NE(fitness, 0.0);
    }
}
