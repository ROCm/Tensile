/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <map>
#include <memory>

#include <Tensile/Debug.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

namespace Tensile
{

    /**
 * \ingroup SolutionLibrary
 */
    template <typename MySolution>
    using SolutionMap = std::map<int, std::shared_ptr<MySolution>>;

    template <typename MySolution>
    struct LibraryIOContext
    {
        std::string                  filename;
        std::vector<LazyLoadingInit> preloaded;
        // If lazy loading is used, this may be updated in const functions
        SolutionMap<MySolution>* solutions;
        std::mutex*              solutionsGuard;
    };

    /**
 * \ingroup SolutionLibrary
 *
 * Root level library object. Contains all individual solutions in a map
 * for serialization purposes.
 */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct MasterSolutionLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        static std::string Type()
        {
            return "Master";
        }
        std::string type() const override
        {
            return Type();
        }
        std::string description() const override
        {
            if(library == nullptr)
                return concatenate(
                    type(), " (", solutions.size(), " solutions, next level: nullptr)");
            else
                return concatenate(type(),
                                   " (",
                                   solutions.size(),
                                   " solutions, next level: ",
                                   library->type(),
                                   ")");
        }

        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> library;
        SolutionMap<MySolution>                                 solutions;
        std::string                                             version;
        mutable std::mutex                                      solutionsGuard;

        MasterSolutionLibrary() = default;

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            if(Debug::Instance().printSolutionSelectionTime())
            {
                auto start  = std::chrono::steady_clock::now();
                auto result = findBestSolution_runner(problem, hardware, fitness);
                auto end    = std::chrono::steady_clock::now();

                double time = std::chrono::duration<double, std::micro>(end - start).count();
                std::cout << "Solution selection time: " << time << " us" << std::endl;

                return result;
            }
            else
            {
                return findBestSolution_runner(problem, hardware, fitness);
            }
        }

        std::shared_ptr<MySolution> findBestSolution_runner(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            double* fitness = nullptr) const
        {
            const int                   solution_index = Debug::Instance().getSolutionIndex();
            std::shared_ptr<MySolution> rv;

            if(solution_index >= 0)
            {
                std::cout << "Tensile will use solution index: " << solution_index << std::endl;
                std::cout
                    << "Warning: Tensile will only work for a particular transpose and data type."
                    << std::endl;
                std::cout << "Set TENSILE_SOLUTION_INDEX to a negative number to restore the "
                             "default behavior."
                          << std::endl;
                {
                    std::lock_guard<std::mutex> guard(solutionsGuard);
                    auto                        selected_solution = solutions.at(solution_index);

                    if((*selected_solution->problemPredicate)(problem)
                       && (*selected_solution->hardwarePredicate)(hardware))
                        rv = selected_solution;
                    else
                        return nullptr;
                }
            }
            else
                rv = library->findBestSolution(problem, hardware, fitness);

            if(Debug::Instance().printLibraryLogicIndex())
            {
                if(rv)
                    std::cout << "Library logic solution index of winning solution: "
                              << rv->info["SolutionIndex"] << std::endl;
                else
                    std::cout << "No solution found" << std::endl;
            }
            return rv;
        }
        virtual SolutionSet<MySolution> findAllSolutions(MyProblem const& problem,
                                                         Hardware const&  hardware) const override
        {
            return library->findAllSolutions(problem, hardware);
        }
    };

} // namespace Tensile
