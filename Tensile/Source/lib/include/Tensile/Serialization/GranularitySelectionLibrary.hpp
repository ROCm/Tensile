
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <Tensile/GranularitySelectionLibrary.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<GranularitySelectionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = GranularitySelectionLibrary<MyProblem, MySolution>;

            using iot = IOTraits<IO>;

            static void mapping(IO & io, Library & lib)
            {
                SolutionMap<MySolution> * ctx = static_cast<SolutionMap<MySolution> *>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io, "GranularitySelectionLibrary requires that context be set to a SolutionMap.");
                }

                std::vector<int> mappingIndexes;
                if(iot::outputting(io))
                {
                    auto iter = lib.solutions.begin();
                    while(iter != lib.solutions.end())
                    {
                        mappingIndexes.push_back(iter->first);
                        iter++;
                    }
                    iot::mapRequired(io, "idxs", mappingIndexes);
                }
                else
                {
                    iot::mapRequired(io, "idxs", mappingIndexes);
                    auto iter = mappingIndexes.begin();
                    if (iter == mappingIndexes.end())
                    {
                        iot::setError(io, "GranularitySelectionLibrary requires non empty mapping index set.");
                    }
                    while(iter != mappingIndexes.end())
                    {
                        int index = *iter;
                        auto slnIter = ctx->find(index);
                        if(slnIter == ctx->end())
                        {
                            std::ostringstream msg;
                            msg << "Invalid solution index: " << index;
                            iot::setError(io, msg.str());
                        }
                        auto solution = slnIter->second; 
                        lib.solutions.insert(std::make_pair(index, solution));
                        iter++;
                    }
                }
            }
        };  
    }
}
