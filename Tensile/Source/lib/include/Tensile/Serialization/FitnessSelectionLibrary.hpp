
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

#include <Tensile/FitnessSelectionLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<FitnessSolutionTableEntry, IO>
        {
            using Entry = FitnessSolutionTableEntry; //<Key, Value>;
            //using ModelEntry = FitnessModelTableEntry;
            using iot   = IOTraits<IO>;

            static void mapping(IO& io, Entry& entry)
            {
                iot::mapRequired(io, "key", entry.key);
                iot::mapRequired(io, "value", entry.value);
            }

            const static bool flow = true;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<FitnessSelectionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = FitnessSelectionLibrary<MyProblem, MySolution>;

            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                SolutionMap<MySolution>* ctx
                    = static_cast<SolutionMap<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "FitnessSelectionLibrary requires that context be "
                                  "set to a SolutionMap.");
                }

                //std::vector<int>                      cd ;
                //std::vector<std::vector<double>>                      mappingIndices;
                std::map<int, std::vector<std::vector<double>>>         mappingIndices;
                std::vector<FitnessSolutionTableEntry> mapEntries;
                if(iot::outputting(io))
                {
                    //mappingIndices.reserve(lib.solutions.size());

                    //mappingIndices
                    /*for(auto const& pair : lib.solutions)
                    {
                        std::vector<double> none;
                        //mappingIndices.push_back(pair.first, none);
                        mappingIndices[pair.first] = none;
                    }*/

                    for (auto const& problem: lib.modelProblems)
                    {
                        int key = problem.key;
                        std::vector<std::vector<double>> problemList;
                        std::map<int, std::vector<std::vector<double>>>::iterator pIter = mappingIndices.find(key);
                        if (pIter == mappingIndices.end())
                        {
                            problemList = std::vector<std::vector<double>>();
                            mappingIndices[key] = problemList;
                        } else {
                            problemList = pIter->second;
                        }
                        problemList.push_back(problem.problem);
                    } 

                    iot::mapRequired(io, "indices", mappingIndices);

                    for(auto it = lib.exactMap.begin(); it != lib.exactMap.end(); ++it)
                    {
                        FitnessSolutionTableEntry newEntry;
                        newEntry.key   = it->first;
                        newEntry.value = it->second;
                        mapEntries.push_back(newEntry);
                    }
                    iot::mapRequired(io, "exact", mapEntries);
                }
                else
                {
                    iot::mapRequired(io, "indices", mappingIndices);
                    if(mappingIndices.empty())
                        iot::setError(io,
                                      "FitnessSelectionLibrary requires non empty "
                                      "mapping index set.");

                    //for(int index : mappingIndices)
                    for(auto const& pair : mappingIndices)
                    {
                        //auto slnIter = ctx->find(index);
                        auto slnIter = ctx->find(pair.first);
                        if(slnIter == ctx->end())
                        {
                            iot::setError(io, concatenate("Invalid solution index: ", index));
                        }
                        else
                        {
                            auto solution = slnIter->second;
                            //lib.solutions.insert(std::make_pair(index, solution));
                            lib.solutions.insert(std::make_pair(pair.first, solution));
                            for (auto const& model : pair.second)
                            {
                                //for (std::vector<double>::const_iterator it = model.begin() ; it != model.end(); ++it)
                                //{
                                    FitnessModelTableEntry<MySolution> modelProblem;
                                    modelProblem.key = pair.first;
                                    modelProblem.solution = solution;
                                    modelProblem.problem = model; //(*it);
                                    lib.modelProblems.push_back(modelProblem);
                                //}
                            }
                            //lib.modelProblems.push_back()
                        }
                    }

                    iot::mapRequired(io, "exact", mapEntries);

                    for(FitnessSolutionTableEntry entry : mapEntries)
                    {
                        std::vector<size_t> key   = entry.key;
                        int                 value = entry.value;
                        lib.exactMap.insert(std::pair<std::vector<size_t>, int>(key, value));
                    }
                }
            }
        };
    } // namespace Serialization
} // namespace Tensile
