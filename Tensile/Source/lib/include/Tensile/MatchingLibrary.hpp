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
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct ProblemMatchingLibrary: public SolutionLibrary<MyProblem, MySolution>
    {
        using Element = std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>;
        using Table = Matching::MatchingTable<MyProblem, Element, std::shared_ptr<MySolution>>;
        std::shared_ptr<Table> table;
        mutable CachedProblemMap<ProblemKey<size_t>, Tensile::ProblemKeyHash<size_t> ,MySolution> problemMap;

        static std::string Type() { return "Matching"; }
        virtual std::string type() const override { return Type(); }
        virtual std::string description() const override
        {
            if(table == nullptr)
                return concatenate(type(), ", table: nullptr");
            else
                return concatenate(type(), ": ", table->description());
        }

        std::shared_ptr<MySolution>
            findSolutionInCache(MyProblem const& problem, Hardware const& hardware) const
        {
            ProblemKey<size_t> pkey = problem.getKey();
            pkey.addKeyAttribute(hardware.id());

            std::shared_ptr<MySolution> theSolution = problemMap.find(pkey);

            if (theSolution != nullptr)
                return theSolution;

            typename Table::Transform transform =
                [&](Element library) -> std::shared_ptr<MySolution>
                {
                    return library->findBestSolution(problem, hardware);
                };

            auto closestEntry = table->findBestMatch(problem, transform);

            if (closestEntry == nullptr)
                return closestEntry;

            problemMap.add(pkey, closestEntry);

            return closestEntry;
        }

        virtual std::shared_ptr<MySolution>
            findBestSolution(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            auto cachedSolution = findSolutionInCache(problem, hardware);
            return cachedSolution;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            bool debug = Debug::Instance().printPropertyEvaluation();

            SolutionSet<MySolution> rv;

            auto matches = table->matchesInOrder(problem);

            for(auto const& row: matches)
            {
                if(debug)
                    std::cout << row->description() << std::endl;

                auto rowSolutions = row->findAllSolutions(problem, hardware);
                rv.insert(rowSolutions.begin(), rowSolutions.end());

                if(debug)
                    std::cout << std::endl;
            }

            return rv;
        }
    };

}

