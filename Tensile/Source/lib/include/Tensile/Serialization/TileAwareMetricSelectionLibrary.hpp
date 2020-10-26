
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
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

#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/TileAwareMetricSelectionLibrary.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<TileAwareMetricSolutionTableEntry, IO>
        {
            using Entry = TileAwareMetricSolutionTableEntry; //<Key, Value>;
            using iot   = IOTraits<IO>;

            static void mapping(IO& io, Entry& entry)
            {
                iot::mapRequired(io, "key", entry.key);
                iot::mapRequired(io, "value", entry.value);
            }

            const static bool flow = true;
        };

        template <typename IO>
        struct MappingTraits<TileAwareMetricModelTableEntry, IO>
        {
            using Entry = TileAwareMetricModelTableEntry; //<Key, Value>;
            using iot   = IOTraits<IO>;

            static void mapping(IO& io, Entry& entry)
            {
                iot::mapRequired(io, "solution_id", entry.solution_id);
                iot::mapRequired(io, "problem", entry.problem);
            }

            const static bool flow = true;
        };


        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<TileAwareMetricSelectionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = TileAwareMetricSelectionLibrary<MyProblem, MySolution>;

            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                SolutionMap<MySolution>* ctx
                    = static_cast<SolutionMap<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "TileAwareMetricSelectionLibrary requires that context be "
                                  "set to a SolutionMap.");
                }

                //std::map<int, std::vector<std::vector<double>>> mappingIndices;
                //std::vector<TileAwareMetricSolutionTableEntry>  mapEntries;
                std::vector<int>                                     mappingIndices;
                std::vector<TileAwareMetricSolutionTableEntry>       mapEntries;
                //std::map<std::vector<size_t>, int>                   modelEntires;
                std::vector<TileAwareMetricModelTableEntry>          modelEntries;
                //std::vector<int>                      mappingIndices;
                //std::vector<ExactSelectionTableEntry> mapEntries;
                if(iot::outputting(io))
                {
                    mappingIndices.reserve(lib.solutions.size());

                    for(auto const& pair : lib.solutions)
                        mappingIndices.push_back(pair.first);

                    iot::mapRequired(io, "indices", mappingIndices);

                    for(auto it = lib.exactMap.begin(); it != lib.exactMap.end(); ++it)
                    {
                        TileAwareMetricSolutionTableEntry newEntry;
                        newEntry.key   = it->first;
                        newEntry.value = it->second;
                        mapEntries.push_back(newEntry);
                    }
                    iot::mapRequired(io, "exact", mapEntries);

                    for(auto it = lib.modelProblems.begin(); it != lib.modelProblems.end(); ++it)
                    {
                        //std::vector<size_t>         key;
                        //std::shared_ptr<MySolution> solution;
                        //std::vector<double>         problem;

                        TileAwareMetricModelTableEntry newEntry;
                        //std::map<std::vector<size_t>, int>  newEntry;
                        //newEntry.key   = it->key;
                        newEntry.solution_id  = it->solution_id;
                        newEntry.problem = it->problem;
                        modelEntries.push_back(newEntry);
                    }
                    iot::mapRequired(io, "model", modelEntries);
                }
                else
                {
                    iot::mapRequired(io, "indices", mappingIndices);
                    if(mappingIndices.empty())
                        iot::setError(io,
                                      "TileAwareMetricSelectionLibrary requires non empty "
                                      "mapping index set.");

                    for(int index : mappingIndices)
                    {
                        auto slnIter = ctx->find(index);
                        if(slnIter == ctx->end())
                        {
                            iot::setError(io, concatenate("Invalid solution index: ", index));
                        }
                        else
                        {
                            auto solution = slnIter->second;
                            lib.solutions.insert(std::make_pair(index, solution));
                        }
                    }

                    iot::mapRequired(io, "exact", mapEntries);

                    for(TileAwareMetricSolutionTableEntry entry : mapEntries)
                    {
                        std::vector<size_t> key   = entry.key;
                        int                 value = entry.value;
                        lib.exactMap.insert(std::pair<std::vector<size_t>, int>(key, value));
                    }

                    iot::mapRequired(io, "model", modelEntries);

                    for(TileAwareMetricModelTableEntry entry : modelEntries)
                    {
                        //std::vector<size_t> key   = entry.first;
                        //int                 value = entry.second;
                        //lib.exactMap.insert(std::pair<std::vector<size_t>, int>(key, value));
                        lib.modelProblems.push_back(entry);
                    }
                }
            }
        };
    } // namespace Serialization
} // namespace Tensile
