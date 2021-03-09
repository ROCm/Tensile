/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
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

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Serialization.hpp>
#include <Tensile/llvm/YAML.hpp>

using namespace Tensile;

TEST(CUEfficiencyPredicate, CUEfficiency)
{
    std::string mydoc = "type: And\n"
                        "value: [{type: TruePred}, \n"
                        "        {type: CUEfficiency}]";

    llvm::yaml::Input yin(mydoc);

    std::shared_ptr<Predicates::Predicate<ContractionProblem>> p;

    yin >> p;
    ASSERT_FALSE(yin.error());
    EXPECT_NE(p, nullptr);

    ContractionProblem small
        = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1, false, 1);

    ContractionProblem large = ContractionProblem::GEMM(
        false, false, 10'000, 10'000, 10'000, 10'000, 10'000, 10'000, 1, false, 1);

    small.setPerformanceMetric(PerformanceMetric::CUEfficiency);
    large.setPerformanceMetric(PerformanceMetric::CUEfficiency);

    EXPECT_EQ((*p)(small), true);
    EXPECT_EQ((*p)(large), true);
}

TEST(CUEfficiencyPredicate, DeviceEfficiency)
{
    std::string mydoc = "type: And\n"
                        "value: [{type: TruePred}, \n"
                        "        {type: CUEfficiency}]";

    llvm::yaml::Input yin(mydoc);

    std::shared_ptr<Predicates::Predicate<ContractionProblem>> p;

    yin >> p;
    ASSERT_FALSE(yin.error());
    EXPECT_NE(p, nullptr);

    ContractionProblem small
        = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1, false, 1);

    ContractionProblem large = ContractionProblem::GEMM(
        false, false, 10'000, 10'000, 10'000, 10'000, 10'000, 10'000, 1, false, 1);

    small.setPerformanceMetric(PerformanceMetric::DeviceEfficiency);
    large.setPerformanceMetric(PerformanceMetric::DeviceEfficiency);

    EXPECT_EQ((*p)(small), false);
    EXPECT_EQ((*p)(large), false);
}

TEST(CUEfficiencyPredicate, Best)
{
    std::string mydoc = "type: And\n"
                        "value: [{type: TruePred}, \n"
                        "        {type: CUEfficiency}]";

    llvm::yaml::Input yin(mydoc);

    std::shared_ptr<Predicates::Predicate<ContractionProblem>> p;

    yin >> p;
    ASSERT_FALSE(yin.error());
    EXPECT_NE(p, nullptr);

    // Small enough that 'Best' should always return true for CUEfficiency,
    // even if exact logic changes
    ContractionProblem small
        = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1, false, 1);

    // Large enough that 'Best' should always return false for CUEfficiency,
    // even if exact logic changes
    ContractionProblem large = ContractionProblem::GEMM(
        false, false, 10'000, 10'000, 10'000, 10'000, 10'000, 10'000, 1, false, 1);

    small.setPerformanceMetric(PerformanceMetric::Auto);
    large.setPerformanceMetric(PerformanceMetric::Auto);

    EXPECT_EQ((*p)(small), true);
    EXPECT_EQ((*p)(large), false);
}
