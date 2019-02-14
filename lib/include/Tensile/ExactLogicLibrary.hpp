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
#include <Tensile/Predicates.hpp>

namespace Tensile
{
    template <typename MyProblem, typename MySolution, typename MyPredicate>
    struct ExactLogicLibrary: public SolutionLibrary<MyProblem, MySolution>
    {
        using Element = std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>;
        using Row = std::pair<MyPredicate, Element>;
        std::vector<Row> rows;

        ExactLogicLibrary() = default;
        ExactLogicLibrary(std::initializer_list<Row> init)
            :rows(init)
        {
        }

        ExactLogicLibrary(std::vector<Row> const& init)
            :rows(init)
        {
        }

        virtual std::shared_ptr<MySolution>
            findBestSolution(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            std::shared_ptr<MySolution> rv;

            for(auto const& row: rows)
            {
                if(row.first(problem, hardware))
                {
                    rv = row.second->findBestSolution(problem, hardware);
                    if(rv)
                        return rv;
                }
            }

            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            SolutionSet<MySolution> rv;

            for(auto const& row: rows)
            {
                if(row.first(problem, hardware))
                {
                    auto rowSolutions = row.second->findAllSolutions(problem, hardware);
                    rv.insert(rowSolutions.begin(), rowSolutions.end());
                }
            }

            return rv;
        }
    };

    struct HardwarePredicate
    {
        std::shared_ptr<Predicates::Predicate<Hardware>> value;

        HardwarePredicate() = default;
        HardwarePredicate(std::shared_ptr<Predicates::Predicate<Hardware>> init)
            : value(init)
        {
        }

        template <typename Any>
        bool operator()(Any const& problem,
                        Hardware const& hardware) const
        {
            return (*value)(hardware);
        }
    };

    template <typename MyProblem,
              typename MySolution>
    struct HardwareSelectionLibrary:
        public ExactLogicLibrary<MyProblem, MySolution, HardwarePredicate>
    {
        using Base = ExactLogicLibrary<MyProblem, MySolution, HardwarePredicate>;

        HardwareSelectionLibrary() = default;
        HardwareSelectionLibrary(std::initializer_list<typename Base::Row> init)
            :Base(init)
        {
        }

        HardwareSelectionLibrary(std::vector<typename Base::Row> const& init)
            :Base(init)
        {
        }

        static std::string Key() { return "Hardware"; }
        virtual std::string key() const { return Key(); }
    };

    template <typename MyProblem>
    struct ProblemPredicate
    {
        std::shared_ptr<Predicates::Predicate<MyProblem>> value;

        ProblemPredicate() = default;
        ProblemPredicate(std::shared_ptr<Predicates::Predicate<MyProblem>> init)
            : value(init)
        {
        }

        bool operator()(MyProblem const& problem,
                        Hardware  const& hardware) const
        {
            return (*value)(problem);
        }
    };

    template <typename MyProblem,
              typename MySolution>
    struct ProblemSelectionLibrary:
        public ExactLogicLibrary<MyProblem,
                                 MySolution,
                                 ProblemPredicate<MyProblem>>
    {
        using Base = ExactLogicLibrary<MyProblem, MySolution, ProblemPredicate<MyProblem>>;

        ProblemSelectionLibrary() = default;
        ProblemSelectionLibrary(std::initializer_list<typename Base::Row> init)
            :Base(init)
        {
        }

        ProblemSelectionLibrary(std::vector<typename Base::Row> const& init)
            :Base(init)
        {
        }

        static std::string Key() { return "Problem"; }
        virtual std::string key() const { return Key(); }
    };

}

