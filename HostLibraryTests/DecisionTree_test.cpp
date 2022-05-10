/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2020-2022 Advanced Micro Devices, Inc.
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

#include <Tensile/DecisionTree.hpp>

using namespace Tensile;
using namespace DecisionTree;
using Key = std::array<int64_t, 3>;

/*
 * Tests for invalid tree structures
 * e.g. trees that try go out of bounds, trees with circular paths
 */
TEST(DecisionTree, ValidTree)
{
    Tree<Key, int, int> test_tree{{
        /* start */ {0, 7000, 1, 2},
        /* 1 */ {1, 3000, 3, 4},
        /* 2 */ {-1, 0.0, 0, 0},
        /* 3 */ {-1, 0.0, 0, 0},
        /* 4 */ {-1, 1.0, 0, 0},
    }};

    /*
    Makes the following tree:
        start:(Key[0] <= 7000?)
            |
            |
            |------------------------
            |                       |
           YES                      NO
            |                       |
            |                       |
        1:(Key[1] <= 3000?)      2:(RETURN 0.0)
            |
            |
            |------------------------
            |                       |
           YES                      NO
            |                       |
            |                       |
        3:(RETURN 0.0)          4:(RETURN 1.0)
    */

    EXPECT_TRUE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeEmpty)
{
    Tree<Key, int, int> test_tree{{}};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeIdxOOB)
{
    Tree<Key, int, int> test_tree{{
        /* start */ {0, 7000, 1, 7}, // right idx OOB
        /* 1 */ {-1, 0.0, 0, 0},
        /* 2 */ {-1, 0.0, 0, 0},
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeIdxOOBNeg)
{
    Tree<Key, int, int> test_tree{{
        /* start */ {0, 7000, 1, -3}, // right idx OOB
        /* 1 */ {-1, 0.0, 0, 0},
        /* 2 */ {-1, 0.0, 0, 0},
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeCircularShort)
{
    Tree<Key, int, int> test_tree{{
        /* start */ {0, 7000, 1, 0}, // right idx circular
        /* 1 */ {-1, 0.0, 0, 0},
        /* 2 */ {-1, 0.0, 0, 0},
    }};

    EXPECT_FALSE(test_tree.valid());
}

TEST(DecisionTree, InvalidTreeCircularLong)
{
    Tree<Key, int, int> test_tree{{
        /* start */ {0, 7000, 1, 2},
        /* 1 */ {0, 7000, 0, 2}, // left idx circular
        /* 2 */ {-1, 0.0, 0, 0},
    }};

    EXPECT_FALSE(test_tree.valid());
}

/*
 * Tests for correct predictions.
 */
TEST(DecisionTree, SimplePrediction)
{
    float EXPECTED_RETURN = 1.0;

    Tree<Key, int, int> test_tree{{
        /* start */ {0, 7000, 1, 2},
        /* 1 */ {-1, 0.0, 0, 0},
        /* 2 */ {-1, EXPECTED_RETURN, 0, 0},
    }};

    Key test_input{{8000, 2000, 2000}};

    // Expected: start->2
    EXPECT_EQ(test_tree.predict(test_input), EXPECTED_RETURN);
}

TEST(DecisionTree, MultiStepPrediction)
{
    float EXPECTED_RETURN = 1.0;

    Tree<Key, int, int> test_tree{{
        /* start */ {0, 7000, 1, 2},
        /* 1 */ {1, 3000, 3, 4},
        /* 2 */ {-1, 0.0, 0, 0},
        /* 3 */ {-1, 0.0, 0, 0},
        /* 4 */ {-1, EXPECTED_RETURN, 0, 0},
    }};

    Key test_input{{6000, 4000, 2000}};

    // Expected: start->1->4
    EXPECT_EQ(test_tree.predict(test_input), EXPECTED_RETURN);
}
