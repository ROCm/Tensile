/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/DecisionTree.hpp>
#include <Tensile/DecisionTreeLibrary.hpp>

using namespace Tensile;
using namespace DecisionTree;

using Key   = std::array<int64_t, 3>;
using DTree = Tree<Key, std::shared_ptr<ContractionLibrary>, std::shared_ptr<ContractionSolution>>;

/*
 * Tests for invalid tree structures
 * e.g. trees that try go out of bounds, trees with circular paths
 */
TEST(DecisionTree, ValidTree)
{
    DTree test_tree{{
        /* start */ {0, 700, 1, IDX_RETURN_FALSE},
        /* - 1 - */ {1, 300, IDX_RETURN_FALSE, IDX_RETURN_TRUE},
    }};

    /*
    Makes the following tree:
        start:(Key[0] <= 700?)
            |
            |
            |------------------------
            |                       |
           YES                      NO
            |                       |
            |                       |
        1:(Key[1] <= 300?)     2:(RETURN false)
            |
            |
            |------------------------
            |                       |
           YES                      NO
            |                       |
            |                       |
        3:(RETURN false)        4:(RETURN true)
    */

    EXPECT_TRUE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeEmpty)
{
    DTree test_tree{{}};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeIdxOOB)
{
    DTree test_tree{{
        /* start */ {0, 700, IDX_RETURN_TRUE, 7}, // right idx OOB
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeIdxOOBNeg)
{
    DTree test_tree{{
        /* start */ {0, 700, -3, IDX_RETURN_TRUE}, // left idx OOB
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeCircularShort)
{
    DTree test_tree{{
        /* start */ {0, 700, IDX_RETURN_TRUE, 0}, // right idx circular
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeCircularLong)
{
    DTree test_tree{{
        /* start */ {0, 700, 1, IDX_RETURN_TRUE},
        /* - 1 - */ {0, 700, 0, IDX_RETURN_TRUE}, // left idx circular
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidNoTrueNodes)
{
    DTree test_tree{{
        /* start */ {0, 700, 1, IDX_RETURN_FALSE},
        /* - 1 - */ {0, 700, IDX_RETURN_FALSE, IDX_RETURN_FALSE},
    }};

    EXPECT_FALSE(test_tree.valid());
}

/*
 * Tests for correct predictions.
 */
TEST(DecisionTree, SimplePrediction)
{
    DTree test_tree{{
        /* start */ {0, 700, IDX_RETURN_FALSE, IDX_RETURN_TRUE},
    }};

    Key test_input0{{800, 200, 200}};
    Key test_input1{{600, 200, 200}};

    EXPECT_EQ(test_tree.predict(test_input0), true); // Expected: start->true
    EXPECT_EQ(test_tree.predict(test_input1), false); // Expected: start->false
}

TEST(DecisionTree, MultiStepPrediction)
{
    DTree test_tree{{
        /* start */ {0, 700, 1, IDX_RETURN_FALSE},
        /* - 1 - */ {1, 300, IDX_RETURN_FALSE, IDX_RETURN_TRUE},
    }};

    Key test_input0{{800, 400, 200}};
    Key test_input1{{600, 400, 200}};
    Key test_input2{{600, 200, 200}};

    EXPECT_EQ(test_tree.predict(test_input0), false); // Expected: start->false
    EXPECT_EQ(test_tree.predict(test_input1), true); // Expected: start->1->true
    EXPECT_EQ(test_tree.predict(test_input2), false); // Expected: start->1->false
}

/*
 * Tests for libraries.
 */
TEST(DecisionTree, DecisionTreeLibrary)
{
    // Solutions
    auto Solution0 = std::make_shared<ContractionSolution>();
    auto Solution1 = std::make_shared<ContractionSolution>();
    auto Solution2 = std::make_shared<ContractionSolution>();
    auto Solution3 = std::make_shared<ContractionSolution>();

    Solution0->index = 0;
    Solution1->index = 1;
    Solution2->index = 2;
    Solution3->index = 3;

    auto Library0 = std::make_shared<SingleContractionLibrary>(Solution0);
    auto Library1 = std::make_shared<SingleContractionLibrary>(Solution1);
    auto Library2 = std::make_shared<SingleContractionLibrary>(Solution2);

    // Properties
    std::vector<std::shared_ptr<Property<ContractionProblem>>> properties;
    auto freeSizeA   = std::make_shared<Contraction::FreeSizeA>();
    freeSizeA->index = 0;
    properties.push_back(freeSizeA);
    auto freeSizeB   = std::make_shared<Contraction::FreeSizeB>();
    freeSizeB->index = 0;
    properties.push_back(freeSizeB);
    auto boundSize   = std::make_shared<Contraction::BoundSize>();
    boundSize->index = 0;
    properties.push_back(boundSize);

    // Make trees library
    std::vector<DTree> trees;

    DTree tree0{{
        {0, 700, IDX_RETURN_FALSE, IDX_RETURN_TRUE}, // YES for freeSizeA>7000
    }};
    tree0.value = Library0;
    trees.push_back(tree0);

    DTree tree1{{
        {1, 700, IDX_RETURN_FALSE, IDX_RETURN_TRUE}, // YES for freeSizeB>700
    }};
    tree1.value = Library1;
    trees.push_back(tree1);

    DTree tree2{{
        {0, 300, IDX_RETURN_TRUE, IDX_RETURN_FALSE}, // YES for freeSizeA<300
    }};
    tree2.value = Library2;
    trees.push_back(tree2);

    // Forest and full library - Note: Solution 3 as fallback
    using BForest = BasicForest<Key,
                                ContractionProblem,
                                std::shared_ptr<ContractionLibrary>,
                                std::shared_ptr<ContractionSolution>>;
    auto forest   = std::make_shared<BForest>(properties, Solution3);
    forest->trees = trees;

    auto dtreelib    = std::make_shared<DecisionTreeLibrary<ContractionProblem>>();
    dtreelib->forest = forest;

    // Problems
    auto Problem0
        = ContractionProblem::GEMM(false, false, 800, 800, 800, 800, 800, 800, 1, false, 1);
    auto Problem1
        = ContractionProblem::GEMM(false, false, 500, 800, 800, 500, 800, 500, 1, false, 1);
    auto Problem2
        = ContractionProblem::GEMM(false, false, 500, 500, 500, 500, 500, 500, 1, false, 1);

    // Tests
    AMDGPU gpu;
    EXPECT_EQ(dtreelib->findBestSolution(Problem0, gpu), Solution0);
    EXPECT_EQ(dtreelib->findBestSolution(Problem1, gpu), Solution1);
    EXPECT_EQ(dtreelib->findBestSolution(Problem2, gpu), Solution3); // No match, goes to fallback
}
