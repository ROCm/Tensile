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

namespace Tensile
{

    template <typename MySolution>
    using SolutionMap = std::map<int, std::shared_ptr<MySolution>>;

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct MasterSolutionLibrary: public SolutionLibrary<MyProblem, MySolution>
    {
        static std::string Type() { return "Master"; }
        std::string type() const override { return Type(); }
        std::string description() const override
        {
            if(library == nullptr)
                return concatenate(type(), " (", solutions.size(), " solutions, next level: nullptr)");
            else
                return concatenate(type(), " (", solutions.size(), " solutions, next level: ", library->type(), ")");
        }

        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> library;
        SolutionMap<MySolution> solutions;
        std::string version;

        MasterSolutionLibrary() = default;

        virtual std::shared_ptr<MySolution>
            findBestSolution(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            return library->findBestSolution(problem, hardware);
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            return library->findAllSolutions(problem, hardware);
        }
    };

}
