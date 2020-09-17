
#include <gtest/gtest.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include "TestData.hpp"

#include <tuple>

using namespace Tensile;

TEST(ContractionFitnessTest, MatchingSize)
{
    auto library = LoadLibraryFile<ContractionProblem>(
        TestData::Instance().file("KernelsLite").native());
    ASSERT_NE(library, nullptr);

    AMDGPU hardware;

    {
        ContractionProblem p
            = ContractionProblem::GEMM(false, false, 64, 64, 256, 64, 64, 256, 1.0, false, 2);

        auto solution_and_fitness = library->findBestSolutionWithFitness(p, hardware);
        auto solution = std::get<0>(solution_and_fitness);

        ASSERT_NE(solution, nullptr);
        EXPECT_EQ(std::get<1>(solution_and_fitness), 0.0);
        
    }
}

TEST(ContractionFitnessTest, NonMatchingSize)
{
    auto library = LoadLibraryFile<ContractionProblem>(
        TestData::Instance().file("KernelsLite").native());
    ASSERT_NE(library, nullptr);

    AMDGPU hardware;

    {
        ContractionProblem p
            = ContractionProblem::GEMM(false, false, 65, 64, 256, 65, 64, 256, 1.0, false, 2);

        auto solution_and_fitness = library->findBestSolutionWithFitness(p, hardware);
        auto solution = std::get<0>(solution_and_fitness);

        ASSERT_NE(solution, nullptr);
        EXPECT_NE(std::get<1>(solution_and_fitness), 0.0);
    }
}

   