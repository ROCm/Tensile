/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
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

#pragma once

#include <array>
#include <vector>


namespace SolutionSelection
{
/**
 * \ingroup SolutionSelection
 *
 * A Decision tree model for matching kernels to problem types.
 * See SolSelDecisionTree_test.cpp for usage and expected behaviour.
 */

    class DecisionTree
    {
        public:
        typedef enum
        {
            // NOTE: features first so they can index feature array
            DT_MSIZE,           // Decision node(s)
            DT_NSIZE,
            DT_KSIZE,
            DT_NUM_FEATURES,    // Note: must come after decision nodes

            DT_RETURN = DT_NUM_FEATURES,     // Return node
        } node_type_t;

        typedef std::array<float, DT_NUM_FEATURES> feature_arr_t;

        struct DecisionTreeNode
        {
            node_type_t type;       // Type indicates end node, or decision based on associated feature
            float value;            // Return value if end node, otherwise decision threshold
            int next_idx_left;
            int next_idx_right;
            /***
             *  TODO: node idx values could have reserved values for return if tree size is appropriately limited,
             *        in that case no return node would be required.
             ***/
        };


        DecisionTree(std::vector<DecisionTreeNode> tree) : _tree{ std::move(tree) } { }

        /****
         * default constructors/assignments/destructor
         ****/


        /* Return true if tree is valid */
        bool valid(bool verbose=false);

        /* Return tree prediction for given feature values */
        float predict(const feature_arr_t& feature_vals);


        private:
        std::vector<DecisionTreeNode> _tree;
    }; // class DecisionTree
} // namespace SolutionSelection
