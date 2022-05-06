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

#include <Tensile/Debug.hpp>
#include <Tensile/DecisionTreeLibrary.hpp>

#include <cstddef>
#include <unordered_set>

namespace Tensile
{
    namespace Serialization
    {
        template <typename Key, typename Element, typename Return, typename IO>
        struct MappingTraits<DecisionTree::Tree<Key, Element, Return>, IO>
        {
            using Tree = DecisionTree::Tree<Key, Element, Return>;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, Tree& tree)
            {
                iot::mapRequired(io, "tree", tree.tree);
                iot::mapRequired(io, "value", tree.value);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<DecisionTreeLibrary<MyProblem, MySolution>, IO>
        {
            using Library    = DecisionTreeLibrary<MyProblem, MySolution>;
            using Properties = typename Library::Properties;
            using Element    = typename Library::Element;

            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                Properties properties;
                if(iot::outputting(io))
                {
                    properties = lib.properties;
                }
                iot::mapRequired(io, "properties", lib.properties);
                iot::mapRequired(io, "trees", lib.trees);
            }
        };

        template <typename IO>
        struct MappingTraits<DecisionTree::Node, IO>
        {
            using Node = typename DecisionTree::Node;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, Node& node)
            {
                iot::mapRequired(io, "type", node.type);
                iot::mapRequired(io, "value", node.value);
                iot::mapRequired(io, "nextIdxLeft", node.nextIdxLeft);
                iot::mapRequired(io, "nextIdxRight", node.nextIdxRight);
            }

            const static bool flow = true;
        };
    } // namespace Serialization
} // namespace Tensile
