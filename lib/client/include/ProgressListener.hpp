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

#include "RunListener.hpp"

#include "ResultReporter.hpp"

namespace Tensile
{
    namespace Client
    {
        class ProgressListener: public RunListener
        {
        public:
            ProgressListener() {}

            virtual bool needMoreBenchmarkRuns() const override { return false; }

            virtual void preBenchmarkRun() override
            {
                m_reporter->report(ResultKey::BenchmarkRunNumber, m_benchmarkRun);
            }

            virtual void postBenchmarkRun() override
            {
                m_benchmarkRun++;
                m_problemIndex = 0;
            }

            virtual void preProblem(ContractionProblem const& problem) override
            {
                m_reporter->report(ResultKey::ProblemIndex, m_problemIndex);

                m_reporter->report(ResultKey::OperationIdentifier, problem.operationIdentifier());

                m_reporter->report(ResultKey::TotalFlops, problem.flopCount());

                m_reporter->report(ResultKey::ASizes,   problem.a().sizes());
                m_reporter->report(ResultKey::BSizes,   problem.b().sizes());
                m_reporter->report(ResultKey::CSizes,   problem.c().sizes());
                m_reporter->report(ResultKey::DSizes,   problem.d().sizes());

                m_reporter->report(ResultKey::AStrides, problem.a().strides());
                m_reporter->report(ResultKey::BStrides, problem.b().strides());
                m_reporter->report(ResultKey::CStrides, problem.c().strides());
                m_reporter->report(ResultKey::DStrides, problem.d().strides());

                m_reporter->report(ResultKey::LDA, problem.a().strides()[1]);
                m_reporter->report(ResultKey::LDB, problem.b().strides()[1]);
                m_reporter->report(ResultKey::LDC, problem.c().strides()[1]);
                m_reporter->report(ResultKey::LDD, problem.d().strides()[1]);

                m_reporter->report(ResultKey::ProblemSizes,   problem.problemSizes());

            }

            virtual void postProblem() override {}

            virtual void preSolution(ContractionSolution const& solution) override
            {
                m_reporter->report(ResultKey::SolutionName, solution.name());
                m_reporter->report(ResultKey::SolutionIndex, solution.index);
            }

            virtual void postSolution() override {}

            virtual bool needMoreRunsInSolution() const override { return false; }

            virtual size_t numWarmupRuns() override { return 0; }
            virtual void   setNumWarmupRuns(size_t count) override {}
            virtual void   preWarmup() override {}

            virtual void   postWarmup() override {}
            virtual void   validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                           TimingEvents const& startEvents,
                                           TimingEvents const&  stopEvents) override {}

            virtual size_t numSyncs() override { return 0; }
            virtual void   setNumSyncs(size_t count) override {}
            virtual void   preSyncs() override {}
            virtual void   postSyncs() override {}

            virtual size_t numEnqueuesPerSync() override { return 0; }
            virtual void   setNumEnqueuesPerSync(size_t count) override {}
            virtual void   preEnqueues() override {}
            virtual void   postEnqueues(TimingEvents const& startEvents,
                                        TimingEvents const&  stopEvents) override {}
            virtual void   validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                            TimingEvents const& startEvents,
                                            TimingEvents const&  stopEvents) override {}

            virtual void finalizeReport() override {}

            virtual int error() const override { return 0; }

        private:
            size_t m_benchmarkRun = 0;
            size_t m_problemIndex = 0;
        };
    }
}
