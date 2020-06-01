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

#include "HardwareMonitor.hpp"

#include <unistd.h>

#include <hip/hip_runtime.h>

#include <Tensile/hip/HipUtils.hpp>

#include "HardwareMonitorListener.hpp"
#include "ResultReporter.hpp"

namespace Tensile
{
    namespace Client
    {
        HardwareMonitorListener::HardwareMonitorListener(po::variables_map const& args)
            : m_useGPUTimer(args["use-gpu-timer"].as<bool>())
            , m_active(args["hardware-monitor"].as<bool>())
        {
            if(!m_active)
                return;

            m_monitor = std::make_shared<HardwareMonitor>(args["device-idx"].as<int>());
            m_monitor->addTempMonitor(0);

            m_monitor->addClockMonitor(RSMI_CLK_TYPE_SYS);
            m_monitor->addClockMonitor(RSMI_CLK_TYPE_SOC);
            m_monitor->addClockMonitor(RSMI_CLK_TYPE_MEM);

            m_monitor->addFanSpeedMonitor();
        }

        void HardwareMonitorListener::preEnqueues()
        {
            if(m_active && !m_useGPUTimer)
                m_monitor->start();
        }

        void HardwareMonitorListener::postEnqueues(TimingEvents const& startEvents,
                                                   TimingEvents const& stopEvents)
        {
            if(!m_active)
                return;

            if(m_useGPUTimer)
            {
                m_monitor->runBetweenEvents(startEvents->front().front(),
                                            stopEvents->back().back());
            }
            else
            {
                m_monitor->stop();
            }
        }

        void HardwareMonitorListener::validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                                       TimingEvents const& startEvents,
                                                       TimingEvents const& stopEvents)
        {
            if(!m_active)
                return;

            m_monitor->wait();

            m_reporter->report(ResultKey::DeviceIndex, m_monitor->getDeviceIndex());
            m_reporter->report(ResultKey::TempEdge, m_monitor->getAverageTemp(0));

            m_reporter->report(ResultKey::ClockRateSys,
                               m_monitor->getAverageClock(RSMI_CLK_TYPE_SYS));
            m_reporter->report(ResultKey::ClockRateSOC,
                               m_monitor->getAverageClock(RSMI_CLK_TYPE_SOC));
            m_reporter->report(ResultKey::ClockRateMem,
                               m_monitor->getAverageClock(RSMI_CLK_TYPE_MEM));

            m_reporter->report(ResultKey::FanSpeedRPMs, m_monitor->getAverageFanSpeed());
            m_reporter->report(ResultKey::HardwareSampleCount, m_monitor->getSamples());
        }
    } // namespace Client
} // namespace Tensile
