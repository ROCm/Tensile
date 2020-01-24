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

#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include <Tensile/Properties.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    /**
     * \ingroup Tensile
     * \defgroup PropertyMatching Property Matching
     * 
     * @brief Distance-based matching of Property values to a table.
     * 
     * Generic algorithm for comparing an object to a table of predefined
     * values based on Property objects and a Distance function. Used for
     * MatchingLibrary.
     */

    /**
     * \ingroup PropertyMatching
     */
    namespace Matching
    {
        /**
         * @brief Abstract Distance function base class
         */
        template <typename Key>
        class Distance
        {
        public:
            virtual std::string type() const = 0;
            virtual ~Distance() = default;

            virtual double operator()(Key const& a, Key const& b) const = 0;
        };

        template <typename Key, typename Value>
        struct MatchingTableEntry
        {
            Key    key;
            Value  value;
            double speed;
        };

        template <typename Object, typename Value, typename ReturnValue>
        struct MatchingTable
        {
            using Properties = std::vector<std::shared_ptr<Property<Object>>>;
            using Transform = std::function<ReturnValue(Value)>;

            MatchingTable() = default;
            MatchingTable(Properties const& properties)
                : properties(properties)
            {
            }

            virtual ~MatchingTable() = default;

            virtual ReturnValue findBestMatch(Object const& object, Transform transform) const = 0;

            virtual std::vector<Value> matchesInOrder(Object const& object) const = 0;

            virtual std::string description() const = 0;

            Properties properties;
        };

        /**
         * This exists to provide an abstraction around the different syntax of creating a vector of a size given 
         * at runtime vs. creating an array with a fixed size.
         */
        template <typename Key>
        struct KeyFactory
        {
        };

        template <typename T>
        struct KeyFactory<std::vector<T>>
        {
            static std::vector<T> MakeKey(size_t size)
            {
                return std::vector<T>(size);
            }
        };

        template <typename T, size_t N>
        struct KeyFactory<std::array<T, N>>
        {
            static std::array<T, N> MakeKey(size_t size)
            {
                return std::array<T, N>();
            }
        };

        template <typename Key, typename Object, typename Value, typename ReturnValue>
        class DistanceMatchingTable: public MatchingTable<Object, Value, ReturnValue>
        {
        public:
            using Base       = MatchingTable<Object, Value, ReturnValue>;
            using Entry      = MatchingTableEntry<Key, Value>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;

            DistanceMatchingTable(ReturnValue nullValue = ReturnValue())
                : nullValue(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                                  ReturnValue nullValue = ReturnValue())
                : Base(properties),
                  nullValue(nullValue)
            {
            }

            DistanceMatchingTable(std::shared_ptr<Distance<Key>> distance,
                                  Properties const& properties,
                                  ReturnValue nullValue = ReturnValue())
                : Base(properties),
                  nullValue(nullValue),
                  distance(distance)
            {
            }

            ReturnValue findBestKeyMatch(Key const& key, Transform transform) const
            {
                const bool debug = Debug::Instance().printPropertyEvaluation();

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
                    auto myDistance = (*distance)(key, iter->key);
                    bool thisMatch = false;

                    if(myDistance < bestDistance)
                    {
                        auto myMatch = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch = myMatch;
                            thisMatch = true;
                        }

                    }

                    if(debug)
                    {
                        streamJoin(std::cout, iter->key, ", ");
                        std::cout << ": " << myDistance;
                        if(myDistance < bestDistance)
                        {
                            std::cout << " <-- Best so far";

                        if(thisMatch)
                            std::cout << " (has a matching solution)";
                        else
                            std::cout << " (no match)";

                        }

                        std::cout << std::endl;
                    }

                    iter++;
                }

                return bestMatch;
            }

            std::vector<Value> keyMatchesInOrder(Key const& key) const
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

            Key keyForProblem(Object const& object) const
            {
                bool debug = Debug::Instance().printPropertyEvaluation();

                Key myKey = KeyFactory<Key>::MakeKey(this->properties.size());

                for(int i = 0; i < this->properties.size(); i++)
                    myKey[i] = (*this->properties[i])(object);

                if(debug)
                {
                    std::cout << "Object key: ";
                    streamJoin(std::cout, myKey, ", ");
                    std::cout << std::endl;
                }

                return myKey;
            }

            virtual ReturnValue findBestMatch(Object const& object, Transform transform) const override
            {
                return findBestKeyMatch(keyForProblem(object), transform);
            }

            virtual std::vector<Value> matchesInOrder(Object const& object) const override
            {
                return keyMatchesInOrder(keyForProblem(object));
            }

            virtual std::string description() const override
            {
                std::string rv = concatenate("Table: Properties: ", this->properties, ", ", table.size(), " rows, ");

                if(distance != nullptr)
                    rv += concatenate("Distance: ", distance->type());
                else
                    rv += "Distance: nullptr";

                return rv;
            }

            std::vector<Entry> table;
            std::shared_ptr<Distance<Key>> distance;

        protected:
            ReturnValue nullValue;
        };
    }
}

