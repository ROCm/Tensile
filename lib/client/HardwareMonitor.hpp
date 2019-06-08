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

#include <chrono>

#include <boost/program_options.hpp>
#include <rocm_smi/rocm_smi.h>

namespace Tensile
{
    namespace Client
    {
        namespace po = boost::program_options;

        class HardwareMonitor
        {
        public:
            using rsmi_temperature_type_t = int;
            using clock = std::chrono::high_resolution_clock;

            // Monitor at the maximum possible rate.
            HardwareMonitor(int deviceIndex);
            // Limit collection to once per minPeriod.
            HardwareMonitor(int deviceIndex, clock::duration minPeriod);

            ~HardwareMonitor();

            void addTempMonitor(rsmi_temperature_type_t sensorType = 0,
                                rsmi_temperature_metric_t metric = RSMI_TEMP_CURRENT);
            void addClockMonitor(rsmi_clk_type_t clockType);
            void addFanSpeedMonitor(uint32_t sensorIndex = 0);

            double getAverageTemp(rsmi_temperature_type_t sensorIndex = 0, rsmi_temperature_metric_t metric = RSMI_TEMP_CURRENT);
            double getAverageClock(rsmi_clk_type_t clockType);
            double getAverageFanSpeed(uint32_t sensorIndex = 0);
            size_t getSamples() { return m_dataPoints; }

            void start();
            void stop();

            void runUntilEvent(hipEvent_t event);
            void runBetweenEvents(hipEvent_t startEvent, hipEvent_t stopEvent);
            void wait();

        private:
            static void InitROCmSMI();
            static uint32_t GetROCmSMIIndex(int hipDeviceIndex);

            void assertActive();
            void assertNotActive();

            void clearValues();
            void collectOnce();
            void sleepIfNecessary();

            void initThread();
            void collect();
            //void collectBetweenEvents(hipEvent_t startEvent, hipEvent_t stopEvent);

            clock::time_point m_lastCollection;
            clock::time_point m_nextCollection;
            clock::duration   m_minPeriod;

            std::atomic<bool> m_active;
            std::atomic<bool> m_isActive;
            std::atomic<bool> m_exit;
            hipEvent_t m_startEvent = nullptr;
            hipEvent_t m_stopEvent = nullptr;

            std::thread m_thread;

            std::mutex m_mutex;
            std::condition_variable m_cv;

            int      m_deviceIndex;
            uint32_t m_dv_ind;

            size_t m_dataPoints;

            std::vector<std::tuple<rsmi_temperature_type_t, rsmi_temperature_metric_t>> m_tempMetrics;
            std::vector<int64_t> m_tempValues;

            std::vector<rsmi_clk_type_t> m_clockMetrics;
            std::vector<uint64_t> m_clockValues;

            std::vector<uint32_t> m_fanMetrics;
            std::vector<int64_t> m_fanValues;
        };

        class HardwareMonitorListener: public RunListener
        {
        public:
            HardwareMonitorListener(po::variables_map const& args);

            virtual bool needMoreBenchmarkRuns() const override { return false; };
            virtual void preBenchmarkRun() override {};
            virtual void postBenchmarkRun() override {};
            virtual void preProblem(ContractionProblem const& problem) override {};
            virtual void postProblem() override {};
            virtual void preSolution(ContractionSolution const& solution) override {};
            virtual void postSolution() override {};
            virtual bool needMoreRunsInSolution() const override { return false; };

            virtual size_t numWarmupRuns() override { return 0; };
            virtual void   setNumWarmupRuns(size_t count) override {};
            virtual void   preWarmup() override {};
            virtual void   postWarmup() override {};
            virtual void   validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                           TimingEvents const& startEvents,
                                           TimingEvents const&  stopEvents) override {};

            virtual size_t numSyncs() override { return 0; };
            virtual void   setNumSyncs(size_t count) override {};
            virtual void   preSyncs() override {};
            virtual void   postSyncs() override {};

            virtual size_t numEnqueuesPerSync() override { return 0; };
            virtual void   setNumEnqueuesPerSync(size_t count) override {};
            virtual void   preEnqueues() override;
            virtual void   postEnqueues(TimingEvents const& startEvents,
                                        TimingEvents const&  stopEvents) override;
            virtual void   validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                            TimingEvents const& startEvents,
                                            TimingEvents const&  stopEvents) override;

            virtual void finalizeReport() const override {};

            virtual int error() const override { return 0; };

        private:

            bool m_useGPUTimer;
            HardwareMonitor m_monitor;
        };
    }
}

