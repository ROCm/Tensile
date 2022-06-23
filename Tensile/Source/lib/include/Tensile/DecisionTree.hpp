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

#pragma once

#include <array>
#include <functional>
#include <vector>

#include <Tensile/ProblemKey.hpp>
#include <Tensile/Properties.hpp>

namespace Tensile
{
    /**
     * \ingroup Tensile
     * \defgroup DecisionTree Decision Tree
     *
     * @brief Tree based decisions on a list of Property values
     *
     * Generic decision tress for deciding on an object based on a list of Properties
     * derived from the object. Used for DecisionTreeLibrary.
     */

    /**
     * \ingroup DecisionTree
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

        /**
         * @brief Generic decision tree
         *
         * @tparam Key type used for deciding
         * @tparam Value type managed by tree
         * @tparam ReturnValue type returned by tree
         */
        template <typename Key, typename Value, typename ReturnValue>
        struct Tree
        {
            using Transform = std::function<ReturnValue(Value)>;

            Tree() = default;
            Tree(std::vector<Node> tree)
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

            bool valid(bool verbose = false) const
            {
                int  treeSize = tree.size();
                Node currentNode;
                bool valid = true;

                if(treeSize == 0)
                {
                    if(verbose)
                    {
                        std::cout << "Tree invalid: no nodes " << std::endl;
                    }
                    return false;
                }

                // Check for any invalid nodes
                for(int nodeIdx = 0; nodeIdx < treeSize; nodeIdx++)
                {
                    currentNode = tree[nodeIdx];
                    if(currentNode.type != -1)
                    {
                        // Avoid OOB on feature array
                        if(currentNode.type >= std::tuple_size<Key>::value)
                        {
                            if(verbose)
                            {
                                std::cout << "Node " << std::to_string(nodeIdx)
                                          << " invalid: Unrecognised type '"
                                          << std::to_string(currentNode.type) << "'" << std::endl;
                            }
                            valid = false;
                        }
                        // Avoid OOB on tree
                        if((currentNode.nextIdxLeft < 0) || (currentNode.nextIdxRight < 0)
                           || (treeSize <= currentNode.nextIdxLeft)
                           || (treeSize <= currentNode.nextIdxRight))
                        {
                            if(verbose)
                            {
                                std::cout << "Node " << std::to_string(nodeIdx)
                                          << " invalid: child index OOB" << std::endl;
                            }
                            valid = false;
                        }
                        // Avoid circular trees
                        if((currentNode.nextIdxLeft <= nodeIdx)
                           || (currentNode.nextIdxRight <= nodeIdx))
                        {
                            if(verbose)
                            {
                                std::cout << "Node " << std::to_string(nodeIdx)
                                          << " invalid: potentially circular tree" << std::endl;
                            }
                            valid = false;
                        }
                    }
                }

                return valid;
            }

            std::vector<Node> tree;
            Value             value;
        };

        /**
         * @brief Abstract base class for a group of decision trees
         *
         * @tparam Object type used to query trees
         * @tparam Value type managed by trees
         * @tparam ReturnValue type returned by trees
         */
        template <typename Object, typename Value, typename ReturnValue>
        struct Forest
        {
            using Properties = std::vector<std::shared_ptr<Property<Object>>>;
            using Transform  = std::function<ReturnValue(Value)>;

            Forest() = default;
            Forest(Properties const& properties)
                : properties(properties)
            {
            }

            virtual ~Forest() = default;

            virtual ReturnValue findBestMatch(Object const& problem, Transform transform) const = 0;

            virtual std::set<ReturnValue> matchesInOrder(Object const& problem,
                                                         Transform     transform) const = 0;

            virtual std::string description() const = 0;

            Properties properties;
        };

        /**
         * @brief Forest that returns first successful prediction from managed trees
         *
         * @tparam Key type used by trees for deciding
         * @tparam Object type used to query trees
         * @tparam Value type managed by trees
         * @tparam ReturnValue type returned by trees
         */
        template <typename Key, typename Object, typename Value, typename ReturnValue>
        struct BasicForest : public Forest<Object, Value, ReturnValue>
        {
            using Base       = Forest<Object, Value, ReturnValue>;
            using Tree       = Tree<Key, Value, ReturnValue>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;

            BasicForest(ReturnValue nullValue = ReturnValue())
                : nullValue(nullValue)
            {
            }

            BasicForest(Properties const& properties, ReturnValue nullValue = ReturnValue())
                : Base(properties)
                , nullValue(nullValue)
            {
            }

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
                return nullValue;
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

            virtual std::string description() const override
            {
                return concatenate(
                    "Forest: Properties: ", this->properties, ", ", trees.size(), " tree(s)");
            }

            std::vector<Tree> trees;
            ReturnValue       nullValue;
        };
    } // namespace DecisionTree
} // namespace Tensile
