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

#include <Tensile/PropertyMatching.hpp>

namespace Tensile
{
    namespace Selection
    {

        template <typename Value>
        struct SelectionTableEntry
        {
            Value  value;
        };

        template <typename Object, typename Value, typename ReturnValue>
        struct SelectionTable
        {
            using Properties = std::vector<std::shared_ptr<Property<Object>>>;
            using Transform = std::function<ReturnValue(Value)>;

            SelectionTable() = default;
            SelectionTable(Properties const& properties)
                : properties(properties)
            {
            }

            virtual ~SelectionTable() = default;

            virtual ReturnValue findBestMatch(Object const& object, Transform transform) const = 0;

            virtual std::string description() const = 0;
            virtual std::vector<Value> getAllSolutions(Object const& object, Transform transform) const = 0; 

            Properties properties;
        };


        template <typename Key, typename Object, typename Value, typename ReturnValue>
        class GranularitySelectionTable: public SelectionTable<Object, Value, ReturnValue>
        {
        public:
            using Base       = SelectionTable<Object, Value, ReturnValue>;
            using Entry      = SelectionTableEntry<Value>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;

            GranularitySelectionTable(ReturnValue nullValue = ReturnValue())
                : nullValue(nullValue)
            {
            }

            GranularitySelectionTable(Properties const& properties,
                                  ReturnValue nullValue = ReturnValue())
                : Base(properties),
                  nullValue(nullValue)
            {
            }

            ReturnValue findBestMatch(Object const& object, Transform transform) const
            {
                const bool debug = Debug::Instance().printPropertyEvaluation();

                double bestPerformance = std::numeric_limits<double>::max();

                auto iter = this->table.begin();
                if(iter == this->table.end())
                    return this->nullValue;

                ReturnValue bestMatch = transform(iter->value);

                if(bestMatch)
                    bestPerformance = bestMatch->projectedPerformance(object);

                if(debug)
                {
                    std::cout << "best performance: " << bestPerformance << std::endl;
                }

                iter++;

                while(iter != this->table.end())
                {
                    ReturnValue myMatch = transform(iter->value);
                    bool thisMatch = false;

                    if(myMatch) 
                    {
                        auto myPerformance = myMatch->projectedPerformance(object);
                        if(myPerformance > bestPerformance)
                        {
                            bestPerformance = myPerformance;
                            bestMatch = myMatch;
                            thisMatch = true;
                        }

                        if(debug)
                        {
                            std::cout << ": " << myPerformance;
                            if(myPerformance < bestPerformance)
                            {
                                std::cout << " <-- Best so far";

                            if(thisMatch)
                                std::cout << " (has a matching solution)";
                            else
                                std::cout << " (no match)";

                            }

                            std::cout << std::endl;
                        }
                    }

                    iter++;
                }

                return bestMatch;
            }

            virtual std::vector<Value> getAllSolutions(Object const& object, Transform transform) const
            {
                std::vector<std::pair<double, size_t>> indices(this->table.size());

                auto iter = this->table.begin();
                if(iter == this->table.end())
                    return std::vector<Value>(); 

                for (size_t i = 0; i < this->table.size(); ++i)
                {
                    ReturnValue myValue = transform(this->table[i].value);
                    double perf = myValue->projectedPerformance(object);
                    indices[i] = std::make_pair(perf, i);
                }

                std::sort(indices.begin(), indices.end());

                std::vector<Value> result;
                result.reserve(this->table.size());

                for(auto const& entry: indices)
                    result.push_back(this->table[entry.second].value);

                return result;
            }

            virtual std::string description() const override
            {
                std::string rv = concatenate("Table: Properties: ", this->properties, ", ", table.size(), " rows, ");

                return rv;
            }

            std::vector<Entry> table;

        protected:
            ReturnValue nullValue;
        };
    }
}

