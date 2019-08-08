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

#include <ProgressListener.hpp>

#include <iomanip>

#include <sys/time.h>

namespace Tensile
{
    namespace Client
    {
        ProgressListener::ProgressListener()
        {
        }

        bool ProgressListener::needMoreBenchmarkRuns() const override
        {
            return false;
        }

        void ProgressListener::preBenchmarkRun() override
        {
            m_reporter->report(ResultKey::BenchmarkRunNumber, m_benchmarkRun);
        }

        void ProgressListener::postBenchmarkRun() override
        {
            m_benchmarkRun++;
        }

        void ProgressListener::preProblem(ContractionProblem const& problem) override
        {

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

        void ProgressListener::postProblem() override
        {
        }

        void ProgressListener::preSolution(ContractionSolution const& solution) override
        {
            m_reporter->report(ResultKey::SolutionName, solution.name());
            m_reporter->report(ResultKey::SolutionIndex, solution.index);
        }

        void ProgressListener::postSolution() override
        {
        }

        bool ProgressListener::needMoreRunsInSolution() const override
        {
            return false;
        }

        size_t ProgressListener::numWarmupRuns() override
        {
            return 0;
        }

        void   ProgressListener::setNumWarmupRuns(size_t count) override
        {
        }

        void   ProgressListener::preWarmup() override
        {
        }

        void   ProgressListener::postWarmup() override
        {
        }

        void   ProgressListener::validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                       TimingEvents const& startEvents,
                                       TimingEvents const&  stopEvents) override
        {
        }

        size_t ProgressListener::numSyncs() override
        {
            return 0;
        }

        void   ProgressListener::setNumSyncs(size_t count) override
        {
        }

        void   ProgressListener::preSyncs() override
        {
        }

        void   ProgressListener::postSyncs() override
        {
        }

        size_t ProgressListener::numEnqueuesPerSync() override
        {
            return 0;
        }

        void   ProgressListener::setNumEnqueuesPerSync(size_t count) override
        {
        }

        void   ProgressListener::preEnqueues() override
        {
        }

        void   ProgressListener::postEnqueues(TimingEvents const& startEvents,
                                    TimingEvents const&  stopEvents) override
        {
        }

        void   ProgressListener::validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                        TimingEvents const& startEvents,
                                        TimingEvents const&  stopEvents) override
        {
            struct timeval tmnow;
            struct tm *tm;
            gettimeofday(&tmnow, NULL); // microsecond resolution
            tm = localtime(&tmnow.tv_sec);
            char prev_fill = std::cout.fill('0');

            std::ostringstream msg;
            msg.fill('0');
            msg << (tm->tm_year + 1900) << "-"
                << std::setw(2) << (tm->tm_mon + 1) << "-"
                << std::setw(2) << tm->tm_mday << " "
                << std::setw(2) << tm->tm_hour << ":"
                << std::setw(2) << tm->tm_min << ":"
                << std::setw(2) << tm->tm_sec << "."
                << std::setw(6) << static_cast<int>(tmnow.tv_usec);

            m_reporter->report(ResultKey::EnqueueTime, msg.str());
        }

        void ProgressListener::finalizeReport() override
        {
        }

        int ProgressListener::error() const override
        {
            return 0;
        }


    }
}
