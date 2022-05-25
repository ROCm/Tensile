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

#include <Tensile/SolutionLibrary>
#include <Tensile>/MasterSolutionLibrary>
#include <Tensile/Tensile.hpp>
 
namespace Tensile{

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct PlaceholderLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>       library;
        SolutionMap<MySolution>*                                      solutions;
        std::string                                                   filePrefix;
        std::string                                                   libraryDirectory;

        //Needs to:
        // load metadata *DONE
        // send new solutions back to MasterSolutionLibrary *DONE
        // tell adapter which code object files are needed
        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double* fitness = nullptr) const override
        {
            if (!library){
                auto newLibrary = LoadLibraryFile((libraryDirectory+"/"+filePrefix).c_str());
                library = newLibrary->library;
                solutions->insert(newLibrary->solutions.begin(), newLibrary->solutions.end());
            }
            
            return library->findBestSolution(problem, hardware, fitness);
        }

        /**
         * Returns all `Solution` objects that are capable of correctly solving this
         * `problem` on this `hardware`.
         *
         * May return an empty set if no such object exists.
         */
        virtual SolutionSet<MySolution> findAllSolutions(MyProblem const& problem,
                                                         Hardware const&  hardware) const override
        {
             
        }

        static std::string Type() const
        {
            return "Placeholder";
        } 

        virtual std::string type() const override
        {
            return Type();
        }

        virtual std::string description() const = 0;
    };

} // namespace Tensile
