/**
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include <Tensile/Properties.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    namespace Matching
    {
        class Distance
        {
        public:
            virtual std::string type() const = 0;

            virtual double operator()(std::vector<size_t> const& a, std::vector<size_t> const& b) const = 0;

            virtual ~Distance() = default;
        };

        template <typename Value>
        struct MatchingTableEntry
        {
            std::vector<size_t> key;
            Value value;
            double speed;
        };

        template <typename Object, typename Value, typename ReturnValue>
        class MatchingTable
        {
        public:
            using Key = std::vector<size_t>;
            using Entry = MatchingTableEntry<Value>;
            using Transform = std::function<ReturnValue(Value)>;

            using Properties = std::vector<std::shared_ptr<Property<Object>>>;

            MatchingTable(ReturnValue nullValue = ReturnValue())
                : nullValue(nullValue)
            {
            }

            MatchingTable(Properties const& properties, ReturnValue nullValue = Value())
                : properties(properties),
                  nullValue(nullValue)
            {
            }

            virtual Key keyForProblem(Object const& object) const
            {
                bool debug = Debug::Get().printPropertyEvaluation();

                Key myKey;
                myKey.reserve(properties.size());

                for(auto const& prop: properties)
                    myKey.push_back((*prop)(object));

                if(debug)
                {
                    std::cout << "Object key: ";
                    streamJoin(std::cout, myKey, ", ");
                    std::cout << std::endl;
                }

                return myKey;
            }

            virtual ReturnValue findBestMatch(Object const& object, Transform transform) const
            {
                return findBestKeyMatch(keyForProblem(object), transform);
            }

            virtual std::vector<Value> matchesInOrder(Object const& object) const
            {
                return keyMatchesInOrder(keyForProblem(object));
            }

            virtual ReturnValue findBestKeyMatch(Key const& key, Transform transform) const = 0;
            virtual std::vector<Value> keyMatchesInOrder(Key const& key) const = 0;

            std::vector<std::shared_ptr<Property<Object>>> properties;
            std::vector<Entry> table;

        protected:
            ReturnValue nullValue;
        };

        template <typename Object, typename Value, typename ReturnValue>
        class DistanceMatchingTable: public MatchingTable<Object, Value, ReturnValue>
        {
        public:
            using Base = MatchingTable<Object, Value, ReturnValue>;
            using Key = typename Base::Key;
            using Entry = typename Base::Entry;
            using Properties = typename Base::Properties;
            using Transform = typename Base::Transform;

            DistanceMatchingTable(ReturnValue nullValue = ReturnValue())
                : Base(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                                  ReturnValue nullValue = ReturnValue())
                : Base(properties, nullValue)
            {
            }

            DistanceMatchingTable(std::shared_ptr<Distance> distance,
                                  Properties const& properties,
                                  ReturnValue nullValue = ReturnValue())
                : Base(properties, nullValue),
                  distance(distance)
            {
            }

            virtual ReturnValue findBestKeyMatch(Key const& key, Transform transform) const override
            {
                bool debug = Debug::Get().printPropertyEvaluation();

                double bestDistance = std::numeric_limits<double>::max();

                auto iter = this->table.begin();
                if(iter == this->table.end())
                    return this->nullValue;

                ReturnValue bestMatch = transform(iter->value);

                if(bestMatch)
                    bestDistance = (*distance)(key, iter->key);

                if(debug)
                {
                    std::cout << "Key: ";
                    streamJoin(std::cout, key, ", ");
                    std::cout << std::endl;

                    streamJoin(std::cout, iter->key, ", ");

                    std::cout << ": " << bestDistance << " <-- First" << std::endl;
                }

                iter++;

                while(iter != this->table.end())
                {
                    auto myMatch = transform(iter->value);

                    if(myMatch)
                    {
                        auto myDistance = (*distance)(key, iter->key);

                        if(debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": " << myDistance;

                            if(myDistance < bestDistance)
                                std::cout << " <-- Best so far";

                            std::cout << std::endl;
                        }

                        if(myDistance < bestDistance)
                        {
                            bestDistance = myDistance;
                            bestMatch = myMatch;
                        }
                    }

                    iter++;
                }

                return bestMatch;
            }

            virtual std::vector<Value> keyMatchesInOrder(Key const& key) const override
            {
                std::vector<std::pair<double, size_t>> indices(this->table.size());

                for(size_t i = 0; i < this->table.size(); i++)
                    indices[i] = std::make_pair((*distance)(key, this->table[i].key), i);

                std::sort(indices.begin(), indices.end());

                std::vector<Value> result;
                result.reserve(this->table.size());

                for(auto const& entry: indices)
                    result.push_back(this->table[entry.second].value);

                return result;
            }

            std::shared_ptr<Distance> distance;

        };
    }
}

