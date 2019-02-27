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

#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/PropertyMatching.hpp>

namespace Tensile
{
    template<typename MyProblem, typename MySolution, typename MyKey>
    using LibraryMap = std::unordered_map<MyKey, LibraryEntry<MyProblem, MySolution>>;

    /**
     * Represents a set of solutions where each problem can map to only one sub-library.
     * Examples are number of dimensions, transposes, etc.
     */
    template <typename MyProblem, typename MySolution, typename Key = std::string>
    struct ProblemMapLibrary: public SolutionLibrary<MyProblem, MySolution>
    {
        static std::string Type() { return "ProblemMap"; }
        virtual std::string type() const { return Type(); }

        std::shared_ptr<Property<MyProblem, Key>> property;
        LibraryMap<MyProblem, MySolution, Key> map;

        virtual std::shared_ptr<MySolution>
            findBestSolution(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            auto key = (*property)(problem);
            auto iter = map.find(key);

            if(iter == map.end())
                return std::shared_ptr<MySolution>();

            return iter->second->findBestSolution(problem, hardware);
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            auto key = (*property)(problem);
            auto iter = map.find(key);

            if(iter == map.end())
                return SolutionSet<MySolution>();

            return iter->second->findAllSolutions(problem, hardware);
        }
    };
}

