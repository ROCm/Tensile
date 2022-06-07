/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022 Advanced Micro Devices, Inc.
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

#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include <algorithm>

namespace Tensile{

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct PlaceholderLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>       library;
        SolutionMap<MySolution>*                                      solutions;
        std::string                                                   filePrefix;
        std::string                                                   libraryDirectory;

        PlaceholderLibrary() = default;

        bool loadPlaceholderLibrary()
        {
            #ifdef TENSILE_MSGPACK
                std::string suffix = ".dat";
            #else
                std::string suffix = ".yaml";
            #endif

            auto newLibrary = LoadLibraryFile<MyProblem, MySolution>((libraryDirectory+"/"+filePrefix+suffix).c_str());
            auto mLibrary   = static_cast<MasterSolutionLibrary<MyProblem, MySolution>*>(newLibrary.get());
            library = mLibrary->library;
            solutions->insert(mLibrary->solutions.begin(), mLibrary->solutions.end());

            return mLibrary;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double* fitness = nullptr) override
        {
            if (!library){
                loadPlaceholderLibrary();
            }

            auto solution = library->findBestSolution(problem, hardware, fitness);

            std::string coFileDependency = filePrefix;

            if(solution){
                //Get xnack version of source kernel
                if(solution->isSourceKernel())
                {
                    std::string arch = hardware.archName();

                    if (coFileDependency.find("fallback") != std::string::npos)
                        coFileDependency += std::string("_")+arch+std::string(".hsaco");
                    else
                        coFileDependency += std::string(".hsaco");
                }
                else
                    coFileDependency += std::string(".co");
                solution->codeObjectFilename = coFileDependency;
            }

            return solution;
        }

        /**
         * Returns all `Solution` objects that are capable of correctly solving this
         * `problem` on this `hardware`.
         *
         * May return an empty set if no such object exists.
         */
        virtual SolutionSet<MySolution> findAllSolutions(MyProblem const& problem,
                                                         Hardware const&  hardware) override
        {
             return library->findAllSolutions(problem, hardware);
        }

        static std::string Type()
        {
            return "Placeholder";
        }

        virtual std::string type() const override
        {
            return Type();
        }

        virtual std::string description() const override
        {
            return this->type();
        }
    };

} // namespace Tensile
