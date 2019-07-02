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

#include <string>

namespace Tensile
{
    namespace Client
    {
        enum class LogLevel
        {
            Error = 0,
            Terse,
            Verbose,
            Debug,
            Count
        };

        namespace ResultKey
        {
            const std::string BenchmarkRunNumber = "run";

            // Problem definition
            const std::string ProblemIndex = "problem-index";
            const std::string OperationIdentifier = "operation";

            const std::string ASizes = "a-sizes";
            const std::string BSizes = "b-sizes";
            const std::string CSizes = "c-sizes";
            const std::string DSizes = "d-sizes";

            const std::string AStrides = "a-strides";
            const std::string BStrides = "b-strides";
            const std::string CStrides = "c-strides";
            const std::string DStrides = "d-strides";

            const std::string LDA = "lda";
            const std::string LDB = "ldb";
            const std::string LDC = "ldc";
            const std::string LDD = "ldd";

            const std::string TotalFlops   = "total-flops";
            const std::string ProblemSizes = "problem-sizes";

            // Solution information
            const std::string SolutionName = "solution";
            const std::string SolutionIndex = "solution-index";
            const std::string SolutionProgress = "solution-progress";

            // Performance-related
            const std::string Validation  = "validation";
            const std::string TimeNS      = "time-ns";
            const std::string SpeedGFlops = "gflops";
            const std::string EnqueueTime = "enqueue-time";

            // Hardware monitoring
            const std::string TempEdge            = "temp-edge";
            const std::string ClockRateSys        = "clock-sys";
            const std::string ClockRateSOC        = "clock-soc";
            const std::string ClockRateMem        = "clock-mem";
            const std::string FanSpeedRPMs        = "fan-rpm";
            const std::string HardwareSampleCount = "hardware-samples";
        };

        class ResultReporter: public RunListener
        {
        public:
            /**
             * Reports the value for a key, related to the current state of the run.
             */
            void report(std::string const& key, std::string const& value)
            {
                reportValue_string(key, value);
            }

            void report(std::string const& key, uint64_t value)
            {
                reportValue_uint(key, value);
            }

            void report(std::string const& key, int value)
            {
                reportValue_int(key, value);
            }

            void report(std::string const& key, int64_t value)
            {
                reportValue_int(key, value);
            }

            void report(std::string const& key, double value)
            {
                reportValue_double(key, value);
            }

            void report(std::string const& key, std::vector<size_t> const& value)
            {
                reportValue_sizes(key, value);
            }

            virtual void reportValue_string(std::string const& key, std::string const& value) = 0;
            virtual void reportValue_uint(  std::string const& key, uint64_t value) = 0;
            virtual void reportValue_int(   std::string const& key, int64_t value) = 0;
            virtual void reportValue_double(std::string const& key, double value) = 0;
            virtual void reportValue_sizes(std::string const& key, std::vector<size_t> const& value) = 0;

            virtual bool logAtLevel(LogLevel level) { return false; };

            /**
             * Records an informative message.  This may or may not actually get printed anywhere depending on settings.
             */
            template <typename T>
            void log(LogLevel level, T const& object)
            {
                if(logAtLevel(level))
                {
                    std::ostringstream msg;
                    msg << object;
                    logMessage(level, msg.str());
                }
            }

            virtual void logMessage(LogLevel level, std::string const& message) {}
            virtual void logTensor(LogLevel level, std::string const& name, void const* data, TensorDescriptor const& tensor) {}

            /// RunListener interface functions

            virtual void setReporter(std::shared_ptr<ResultReporter> reporter) override {}

            virtual bool needMoreBenchmarkRuns() const override { return false; }
            virtual void preBenchmarkRun() override {}
            virtual void postBenchmarkRun() override {}

            virtual void preProblem(ContractionProblem const& problem) override {}
            virtual void postProblem() override {}

            virtual void preSolution(ContractionSolution const& solution) override {}
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

            // finalizeReport() deliberately left out of here to force it to be implemented in subclasses.

            virtual int error() const override
            {
                return 0;
            }
        };
    }
}
