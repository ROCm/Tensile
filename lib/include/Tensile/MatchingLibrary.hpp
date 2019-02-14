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

#include <vector>
#include <set>

#include <Tensile/PropertyMatching.hpp>

namespace Tensile
{
    template <typename MyProblem, typename MySolution>
    struct ProblemMatchingLibrary: public SolutionLibrary<MyProblem, MySolution>
    {
        using Element = std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>;
        using Table = Matching::MatchingTable<MyProblem, Element>;
        std::shared_ptr<Table> table;

        static std::string Key() { return "Matching"; }
        virtual std::string key() { return Key(); }

        
        virtual std::shared_ptr<MySolution>
            findBestSolution(std::shared_ptr<MyProblem> problem,
                             std::shared_ptr<Hardware> hardware) const
        {
            auto closestEntry = table.findBestMatch(problem);

            if(closestEntry)
                return closestEntry.findBestSolution(problem, hardware);

            return std::shared_ptr<MySolution>();
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(std::shared_ptr<MyProblem> problem,
                             std::shared_ptr<Hardware> hardware) const
        {
            SolutionSet<MySolution> rv;

            for(auto const& row: table.table)
            {
                auto rowLibrary = std::get<Table::EntryValue>(row);
                auto rowSolutions = rowLibrary.findAllSolutions(problem, hardware);
                rv.insert(rowSolutions.begin(), rowSolutions.end());
            }

            return rv;
        }
    };

}

