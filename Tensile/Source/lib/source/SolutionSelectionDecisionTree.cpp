/**
 * MIT License
 *
 * Copyright 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <Tensile/SolutionSelectionDecisionTree.hpp>

#include <iostream>
#include <stdexcept>


namespace SolutionSelection
{
    bool DecisionTree::valid(bool verbose)
    {
        int tree_size = _tree.size();
        DecisionTreeNode current_node;
        bool valid = true;

        // Check for any invalid nodes
        for (int node_idx = 0; node_idx < tree_size; node_idx++)
        {
            current_node = _tree[node_idx];
            if (current_node.type != DT_RETURN)
            {
                // Avoid OOB on feature array
                if (current_node.type >= DT_NUM_FEATURES)
                {
                    if (verbose)
                    {
                        std::cout << "Node " << std::to_string(node_idx)
                                  << " invalid: Unrecognised type '" << std::to_string(current_node.type) << "'"
                                  << std::endl;
                    }
                    valid = false;
                }
                // Avoid OOB on tree
                if ((current_node.next_idx_left < 0)          || (current_node.next_idx_right < 0) ||
                    (tree_size <= current_node.next_idx_left) || (tree_size <= current_node.next_idx_right))
                {
                    if (verbose)
                    {
                        std::cout << "Node " << std::to_string(node_idx)
                                  << " invalid: child index OOB"
                                  << std::endl;
                    }
                    valid = false;
                }
                // Avoid circular trees
                if ((current_node.next_idx_left <= node_idx) || (current_node.next_idx_right <= node_idx))
                {
                    if (verbose)
                    {
                        std::cout << "Node " << std::to_string(node_idx)
                                  << " invalid: potentially circular tree"
                                  << std::endl;
                    }
                    valid = false;
                }
            }
        }

        return valid;
    }


    float DecisionTree::predict(const feature_arr_t& feature_vals)
    {
        int node_idx  = 0;
        int tree_size = _tree.size();
        DecisionTreeNode current_node;

        while (node_idx < tree_size)
        {
            current_node = _tree[node_idx];
            if (current_node.type == DT_RETURN)
            { /* End node */
                return current_node.value;     
            }

            // Note convention: branch left for less than, else right
            if (feature_vals[current_node.type] <= current_node.value)
            {
                node_idx = current_node.next_idx_left;
            }
            else
            {
                node_idx = current_node.next_idx_right;
            }
        }

        throw std::runtime_error("Decision Tree out of bounds error.");
        return -1;
    }
} // namespace Tensile
