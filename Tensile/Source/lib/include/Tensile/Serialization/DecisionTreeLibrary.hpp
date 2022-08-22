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

#include <Tensile/Debug.hpp>
#include <Tensile/DecisionTreeLibrary.hpp>

#include <cstddef>
#include <unordered_set>

namespace Tensile
{
    namespace Serialization
    {
        template <typename Key, typename Value, typename ReturnValue, typename IO>
        struct MappingTraits<DecisionTree::Tree<Key, Value, ReturnValue>, IO>
        {
            using Tree = DecisionTree::Tree<Key, Value, ReturnValue>;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, Tree& tree)
            {
                iot::mapRequired(io, "tree", tree.tree);
                iot::mapRequired(io, "value", tree.value);
            }
            const static bool flow = false;
        };

        template <typename Key, typename Object, typename Value, typename ReturnValue, typename IO>
        struct MappingTraits<DecisionTree::BasicForest<Key, Object, Value, ReturnValue>, IO>
        {
            using Forest = DecisionTree::BasicForest<Key, Object, Value, ReturnValue>;
            using iot    = IOTraits<IO>;

            static void mapping(IO& io, Forest& lib)
            {
                iot::mapRequired(io, "trees", lib.trees);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<DecisionTreeLibrary<MyProblem, MySolution>, IO>
        {
            using Library    = DecisionTreeLibrary<MyProblem, MySolution>;
            using Properties = typename Library::Forest::Properties;
            using Element    = typename Library::Element;

            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                Properties properties;
                if(iot::outputting(io))
                {
                    properties = lib.forest->properties;
                }
                iot::mapRequired(io, "properties", properties);

                bool success = false;
                if(properties.size() == 0)
                    iot::setError(io, "Matching table must have at least one property.");
                else if(properties.size() == 1)
                    success = mappingKey<std::array<int64_t, 1>>(io, lib, properties);
                else if(properties.size() == 2)
                    success = mappingKey<std::array<int64_t, 2>>(io, lib, properties);
                else if(properties.size() == 3)
                    success = mappingKey<std::array<int64_t, 3>>(io, lib, properties);
                else if(properties.size() == 4)
                    success = mappingKey<std::array<int64_t, 4>>(io, lib, properties);
                else if(properties.size() == 5)
                    success = mappingKey<std::array<int64_t, 5>>(io, lib, properties);
                else if(properties.size() == 6)
                    success = mappingKey<std::array<int64_t, 6>>(io, lib, properties);
                else if(properties.size() == 7)
                    success = mappingKey<std::array<int64_t, 7>>(io, lib, properties);
                else if(properties.size() == 8)
                    success = mappingKey<std::array<int64_t, 8>>(io, lib, properties);
                else if(properties.size() == 9)
                    success = mappingKey<std::array<int64_t, 9>>(io, lib, properties);
                else if(properties.size() == 10)
                    success = mappingKey<std::array<int64_t, 10>>(io, lib, properties);

                if(!success)
                    success = mappingKey<std::vector<int64_t>>(io, lib, properties);

                if(!success)
                    iot::setError(io, "Can't write out key: wrong type.");
            }

            template <typename Key>
            static bool mappingKey(IO& io, Library& lib, Properties const& properties)
            {
                using Forest = DecisionTree::
                    BasicForest<Key, MyProblem, Element, std::shared_ptr<MySolution>>;

                std::shared_ptr<Forest> forest;

                if(iot::outputting(io))
                {
                    forest = std::dynamic_pointer_cast<Forest>(lib.forest);
                    if(!forest)
                        return false;
                }
                else
                {
                    forest             = std::make_shared<Forest>();
                    forest->properties = properties;
                    lib.forest         = forest;
                }

                MappingTraits<Forest, IO>::mapping(io, *forest);

                return true;
            }

            const static bool flow = false;
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
