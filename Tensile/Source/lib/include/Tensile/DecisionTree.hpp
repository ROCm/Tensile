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
#include <functional>
#include <vector>

#include <Tensile/ProblemKey.hpp>

namespace Tensile
{
    /**
     * \ingroup SolutionSelection
     *
     * A Decision tree model for matching kernels to problem types.
     * See SolSelDecisionTree_test.cpp for usage and expected behaviour.
     */
    namespace DecisionTree
    {
        struct Node
        {
            int   type;
            float value;
            int   nextIdxLeft;
            int   nextIdxRight;
        };

        template <typename Key, typename Value, typename ReturnValue>
        struct Tree
        {
            using Transform = std::function<ReturnValue(Value)>;

            Tree() = default;
            Tree(std::vector<Key> tree)
                : tree(std::move(tree))
            {
            }

            float predict(Key const& key) const
            {
                int  nodeIdx  = 0;
                int  treeSize = tree.size();
                Node currentNode;

                while(nodeIdx < treeSize)
                {
                    currentNode = tree[nodeIdx];
                    if(currentNode.type == -1)
                    { /* End node */
                        return currentNode.value;
                    }

                    // Note convention: branch left for less than, else right
                    if(key[currentNode.type] <= currentNode.value)
                    {
                        nodeIdx = currentNode.nextIdxLeft;
                    }
                    else
                    {
                        nodeIdx = currentNode.nextIdxRight;
                    }
                }

                throw std::runtime_error("Decision Tree out of bounds error.");
                return -1;
            }

            virtual ReturnValue getSolution(Transform transform) const
            {
                return transform(value);
            }

            bool valid(bool verbose = false) const {}

            std::vector<Node> tree;
            Value             value;
        };

        template <typename Object, typename Value, typename ReturnValue>
        struct Forest
        {
            using Properties = std::vector<std::shared_ptr<Property<Object>>>;
            using Transform  = std::function<ReturnValue(Value)>;

            Properties properties;

            virtual ReturnValue findBestMatch(Object const& problem, Transform transform) const = 0;

            virtual std::set<ReturnValue> matchesInOrder(Object const& problem,
                                                         Transform     transform) const = 0;
        };

        template <typename Key, typename Object, typename Value, typename ReturnValue>
        struct BasicForest : public Forest<Object, Value, ReturnValue>
        {
            using Base       = Forest<Object, Value, ReturnValue>;
            using Tree       = Tree<Key, Value, ReturnValue>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;

            std::vector<Tree> trees;

            virtual ReturnValue findBestMatch(Object const& problem,
                                              Transform     transform) const override
            {
                Key key = ProblemKey::keyForProblem<Key, Object>(problem, this->properties);
                for(Tree const& tree : trees)
                {
                    float result = tree.predict(key);
                    if(result > 0)
                        return tree.getSolution(transform);
                }
                std::cout << "no \"yes\" from tress" << std::endl;
                return ReturnValue();
            }

            virtual std::set<ReturnValue> matchesInOrder(Object const& problem,
                                                         Transform     transform) const override
            {
                bool debug = Debug::Instance().printPropertyEvaluation();

                std::set<ReturnValue> rv;
                for(Tree const& tree : trees)
                {
                    rv.insert(tree.getSolution(transform));
                }

                return rv;
            }
        };

    } // namespace DecisionTree
} // namespace Tensile
