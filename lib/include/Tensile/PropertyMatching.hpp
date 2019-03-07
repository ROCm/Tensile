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

#include <string>
#include <vector>
#include <tuple>

namespace Tensile
{
    template<typename Object, typename Value = size_t>
    class Property
    {
    public:
        virtual std::string type() const = 0;

        virtual ~Property() = default;

        virtual Value operator()(Object const& object) const = 0;
    };

    namespace Matching
    {

        class Distance
        {
        public:
            virtual std::string type() const = 0;

            virtual double operator()(std::vector<size_t> const& a, std::vector<size_t> const& b) const = 0;
        };

        template <typename Value>
        struct MatchingTableEntry
        {
            std::vector<size_t> key;
            Value value;
            double speed;
        };

        template <typename Object, typename Value>
        class MatchingTable
        {
        public:
            using Key = std::vector<size_t>;
            using Entry = MatchingTableEntry<Value>;

            using Properties = std::vector<std::shared_ptr<Property<Object>>>;

            MatchingTable(Value nullValue = Value())
                : nullValue(nullValue)
            {
            }

            MatchingTable(Properties const& properties, Value nullValue = Value())
                : properties(properties),
                  nullValue(nullValue)
            {
            }

            virtual Key keyForProblem(Object const& object) const
            {
                Key myKey(properties.size());

                for(auto const& prop: properties)
                    myKey.push_back((*prop)(object));

                return myKey;
            }

            virtual Value findBestMatch(Object const& object) const
            {
                return findBestKeyMatch(keyForProblem(object));
            }

            virtual std::vector<Value> matchesInOrder(Object const& object) const
            {
                return std::move(keyMatchesInOrder(keyForProblem(object)));
            }

            virtual Value findBestKeyMatch(Key const& key) const = 0;
            virtual std::vector<Value> keyMatchesInOrder(Key const& key) const = 0;

            std::vector<std::shared_ptr<Property<Object>>> properties;
            std::vector<Entry> table;

        protected:
            Value nullValue;
        };

        template <typename Object, typename Value>
        class DistanceMatchingTable: public MatchingTable<Object, Value>
        {
        public:
            using Base = MatchingTable<Object, Value>;
            using Key = typename Base::Key;
            using Entry = typename Base::Entry;
            using Properties = typename Base::Properties;

            DistanceMatchingTable(Value nullValue = Value())
                : Base(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                                  Value nullValue = Value())
                : Base(properties, nullValue)
            {
            }

            DistanceMatchingTable(std::shared_ptr<Distance> distance,
                                  Properties const& properties,
                                  Value nullValue = Value())
                : Base(properties, nullValue),
                  distance(distance)
            {
            }

            virtual Value findBestKeyMatch(Key const& key) const override
            {
                auto iter = this->table.begin();
                if(iter == this->table.end())
                    return this->nullValue;

                Value bestMatch = iter->value;
                auto bestDistance = (*distance)(key, iter->key);

                iter++;

                while(iter != this->table.end())
                {
                    auto myDistance = (*distance)(key, iter->key);
                    if(myDistance < bestDistance)
                    {
                        bestDistance = myDistance;
                        bestMatch = iter->value;
                    }
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

