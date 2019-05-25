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

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include "ResultReporter_fwd.hpp"

namespace Tensile
{
    namespace Client
    {
        class RunListener
        {
        public:
            virtual ~RunListener() = default;

            virtual void setReporter(std::shared_ptr<ResultReporter> reporter)
            {
                m_reporter = reporter;
            }

            /// Called at the beginning of each problem.
            virtual void setUpProblem(ContractionProblem const& problem)    {}

            /// Called at the beginning of each solution.
            virtual void setUpSolution(ContractionSolution const& solution) {}

            /// Loop condition.  Return true if we need another run for this solution.
            virtual bool needsMoreRunsInSolution() { return false; }
            /// Return true if we need another warmup run for this solution.
            virtual bool isWarmupRun() { return false; }

            /// Called before invoking a kernel.  If isWarmup is true, this is a 
            /// warmup run (even if this listener returned false).
            virtual void setUpRun(bool isWarmup) {}

            /// Called after invoking a kernel.
            virtual void tearDownRun() {}

            /// Called after tearDownRun(), if this is a warmup run.
            virtual void validate(std::shared_ptr<ContractionInputs> inputs) {}

            /// Called at end of each solution. Returns a summary of the results,
            /// although this is intended for internal use between the various listeners.
            virtual void tearDownSolution() = 0;

            /// Called at end of each problem.
            virtual void tearDownProblem() {}

            /// Called at end of program execution.  Print out a summary of the runs.
            virtual void report() const {}

            /// Called at end of program execution.  Return a non-zero value if a
            /// non-fatal error was previously recorded.
            virtual int error() const { return 0; }

        protected:
            std::shared_ptr<ResultReporter> m_reporter;
        };

    }
}

