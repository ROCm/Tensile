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

#include <Tensile/Serialization.hpp>
#include <Tensile/llvm/YAML.hpp>
#include <Tensile/GEMMLibrary.hpp>

using namespace Tensile;

TEST(LLVMYAMLTest, Simple)
{
    std::string mydoc = "name: foo\nworkGroup: [1,2,3]\nmacroTile: [2,4,6]\nthreadTile: [2,2,2]\nindex: 0";
    llvm::yaml::Input yin(mydoc);

    GEMMSolution s;

    yin >> s;
    ASSERT_FALSE(yin.error());

    EXPECT_EQ(s.name(), "foo");
    EXPECT_EQ(s.workGroupSize, dim3(1,2,3));
    EXPECT_EQ(s.macroTile, dim3(2,4,6));

}

TEST(LLVMYAMLTest, Predicate)
{
    std::string mydoc = "type: And\nvalue: [{type: True}, {type: False}, {type: ADimensionOrder, value: [0,1,2]}]";
    llvm::yaml::Input yin(mydoc);

    std::shared_ptr<Predicates::Predicate<GEMMProblem>> p;

    yin >> p;
    ASSERT_FALSE(yin.error());

    GEMMProblem prob;

    EXPECT_NE(p, nullptr);

    EXPECT_EQ((*p)(prob), false);

    llvm::yaml::Output yout(llvm::outs());
    yout << p;
    //EXPECT_EQ(s.name(), "foo");
    //EXPECT_EQ(s.workGroupSize, dim3(1,2,3));
    //EXPECT_EQ(s.macroTile, dim3(2,4,6));

}

TEST(LLVMYAMLTest, GEMMLibrary)
{
    std::string mydoc =
       "solutions:\n"
       "  - name: foo\n"
       "    workGroup: [1,2,3]\n"
       "    macroTile: [1,2,3]\n"
       "    threadTile: [1,2,3]\n"
       "    index: 0\n"
       "library:\n"
       "  type: Hardware\n"
       "  rows:\n"
       "      - predicate: { type: AMDGPU, value: { type: Processor, value: gfx900 } }\n"
       "        library:\n"
       "          type: Problem\n"
       "          rows:\n"
       "              - predicate: { type: ADimensionOrder, value: [0,1,2] }\n"
       "                library: { type: Single, index: 0 }\n"
       "";

    llvm::yaml::Input yin(mydoc);

    MasterGEMMLibrary l;

    yin >> l;

    ASSERT_FALSE(yin.error());

    llvm::yaml::Output yout(llvm::outs());
    yout << l;
}


