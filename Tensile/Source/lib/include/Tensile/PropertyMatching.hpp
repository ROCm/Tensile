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

#include <Tensile/Debug.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/Properties.hpp>
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

            virtual std::string distanceType() const = 0;

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

        template <typename Key, typename Object, typename Value, typename ReturnValue, typename Distance>
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

            DistanceMatchingTable(Distance   const& distance,
                                  Properties const& properties,
                                  ReturnValue nullValue = ReturnValue())
                : Base(properties),
                  nullValue(nullValue),
                  distance(distance)
            {
            }

            virtual std::string distanceType() const
            {
                return Distance::Type();
            }

            ReturnValue findBestKeyMatch(Key const& key, Transform transform) const
            {
                const bool debug = Debug::Instance().printPropertyEvaluation();
                const bool naive = Debug::Instance().naivePropertySearch();

                if(naive)
                {
                    if(debug)
                        return findBestKeyMatch_NaiveSearch<true> (key, transform);
                    else
                        return findBestKeyMatch_NaiveSearch<false>(key, transform);
                }
                else
                {
                    if(debug)
                        return findBestKeyMatch_BinSearch<true> (key, transform);
                    else
                        return findBestKeyMatch_BinSearch<false>(key, transform);
                }

            }

            template <bool T_Debug>
            ReturnValue findBestKeyMatch_BinSearch(Key const& key, Transform transform) const
            {
                if(this->table.empty())
                    return this->nullValue;

                auto comp = [](Entry const& e, Key const& key)
                {
                    return e.key < key;
                };

                auto origIter = std::lower_bound(table.begin(), table.end(), key, comp);

                if(T_Debug)
                {
                    std::cout << "Key: ";
                    streamJoin(std::cout, key, ", ");
                    std::cout << std::endl;

                    std::cout << "Starting point: ";
                    streamJoin(std::cout, origIter->key, ", ");
                    std::cout << std::endl;

                    std::cout << "Rightward search..." << std::endl;
                }

                double bestDistance = std::numeric_limits<double>::max();
                auto bestMatch = this->nullValue;
                double bestSpeed = 0.0;

                ptrdiff_t count = 0;

                for(auto iter = origIter; iter != table.end(); iter++)
                {
                    if(bestMatch && !distance.improvementPossible(key, iter->key, 0, bestDistance))
                    {
                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": Stopping rightward search early." << std::endl;
                        }

                        break;
                    }

                    count++;

                    auto myDistance = distance(key, iter->key);
                    bool thisMatch = false;

                    if(myDistance < bestDistance || (myDistance == bestDistance && iter->speed > bestSpeed))
                    {
                        auto myMatch = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch = myMatch;
                            bestSpeed = iter->speed;
                            thisMatch = true;
                        }
                    }

                    if(T_Debug)
                    {
                        if(myDistance <= bestDistance)
                            std::cout << std::endl;

                        streamJoin(std::cout, iter->key, ", ");
                        std::cout << ": " << myDistance;

                        if(myDistance < bestDistance)
                            std::cout << " < ";
                        else if(myDistance > bestDistance)
                            std::cout << " > ";
                        else
                            std::cout << " == ";

                        std::cout << bestDistance;

                        if(myDistance < bestDistance)
                        {
                            if(thisMatch)
                                std::cout << " <-- Best so far";
                            else
                                std::cout << " <-- Best distance, but no matching solution";
                        }

                        std::cout << std::endl;
                    }
                }

                auto iter = table.rbegin();

                if(origIter != table.end())
                    iter = std::make_reverse_iterator(origIter);

                if(T_Debug)
                    std::cout << "Leftward search..." << std::endl;

                for(; iter != table.rend(); iter++)
                {
                    if(bestMatch && !distance.improvementPossible(key, iter->key, 0, bestDistance))
                    {
                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": Stopping leftward search early." << std::endl;
                        }

                        break;
                    }

                    count++;

                    auto myDistance = distance(key, iter->key);
                    bool thisMatch = false;

                    if(myDistance < bestDistance || (myDistance == bestDistance && iter->speed > bestSpeed))
                    {
                        auto myMatch = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch = myMatch;
                            bestSpeed = iter->speed;
                            thisMatch = true;
                        }
                    }

                    if(T_Debug)
                    {
                        if(myDistance <= bestDistance)
                            std::cout << std::endl;

                        streamJoin(std::cout, iter->key, ", ");
                        std::cout << ": " << myDistance;
                        
                        if(myDistance < bestDistance)
                            std::cout << " < ";
                        else if(myDistance > bestDistance)
                            std::cout << " > ";
                        else
                            std::cout << " == ";

                        std::cout << bestDistance;

                        if(myDistance < bestDistance)
                        {
                            if(thisMatch)
                                std::cout << " <-- Best so far";
                            else
                                std::cout << " <-- Best distance, but no matching solution";
                        }

                        std::cout << std::endl;
                    }
                }

                if((T_Debug || Debug::Instance().printLookupEfficiency()) && table.size() > 0)
                {
                    double considered = count;
                    considered /= table.size();
                    considered *= 100;
                    std::cout << "Considered " << considered << "% of entries." << std::endl;
                }

                return bestMatch;
            }

            template <bool T_Debug>
            ReturnValue findBestKeyMatch_NaiveSearch(Key const& key, Transform transform) const
            {
                double bestDistance = std::numeric_limits<double>::max();

                auto iter = this->table.begin();
                if(iter == this->table.end())
                    return this->nullValue;

                ReturnValue bestMatch = transform(iter->value);

                if(bestMatch)
                    bestDistance = distance(key, iter->key);

                if(T_Debug)
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
                    auto myDistance = distance(key, iter->key);
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

                    if(T_Debug)
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
                    indices[i] = std::make_pair(distance(key, this->table[i].key), i);

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

                rv += concatenate("Distance: ", Distance::Type());

                return rv;
            }

            std::vector<Entry> table;
            Distance distance;

        protected:
            ReturnValue nullValue;
        };
    }
}

