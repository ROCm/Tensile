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

#include "HardwareMonitor.hpp"

#include <unistd.h>

#include <hip/hip_runtime.h>

#include <Tensile/hip/HipUtils.hpp>

#include "ResultReporter.hpp"

#define RSMI_CHECK_EXC(expr) \
    do \
    { \
        rsmi_status_t e = (expr); \
        if(e) \
        { \
            const char * errName = nullptr; \
            rsmi_status_string(e, &errName); \
            std::ostringstream msg; \
            msg << "Error " << e << "(" << errName << ") " \
                          << __FILE__ << ":" << __LINE__ << ": " << std::endl \
                      << #expr << std::endl; \
            throw std::runtime_error(msg.str()); \
        } \
    } while(0)


namespace Tensile
{
    namespace Client
    {
        uint32_t HardwareMonitor::GetROCmSMIIndex(int hipDeviceIndex)
        {
            InitROCmSMI();

            hipDeviceProp_t props;

            HIP_CHECK_EXC(hipGetDeviceProperties(&props, hipDeviceIndex));

            uint64_t hipPCIID = 0;
            hipPCIID |= props.pciDeviceID & 0xFF;
            hipPCIID |= ((props.pciBusID & 0xFF) << 8);
            hipPCIID |= (props.pciDomainID) << 16;

            uint32_t smiCount = 0;

            RSMI_CHECK_EXC(rsmi_num_monitor_devices(&smiCount));

            std::ostringstream msg;

            for(uint32_t smiIndex = 0; smiIndex < smiCount; smiIndex++)
            {
                uint64_t rsmiPCIID = 0;

                RSMI_CHECK_EXC(rsmi_dev_pci_id_get(smiIndex, &rsmiPCIID));

                msg << smiIndex << ": " << rsmiPCIID << std::endl;

                if(hipPCIID == rsmiPCIID)
                    return smiIndex;
            }

            throw std::runtime_error(concatenate("RSMI Can't find a device with PCI ID ", hipPCIID , "(",
                                                 props.pciDomainID, "-", props.pciBusID, "-", props.pciDeviceID, ")\n",
                                                 msg.str()));
        }

        void HardwareMonitor::InitROCmSMI()
        {
            static rsmi_status_t status = rsmi_init(0);
            RSMI_CHECK_EXC(status);
        }

        HardwareMonitor::HardwareMonitor(int deviceIndex, clock::duration minPeriod)
            : m_minPeriod(minPeriod),
              m_active(false),
              m_isActive(false),
              m_exit(false),
              m_deviceIndex(deviceIndex),
              m_dv_ind(GetROCmSMIIndex(deviceIndex)),
              m_dataPoints(0)
        {
            InitROCmSMI();

            initThread();
        }

        HardwareMonitor::HardwareMonitor(int deviceIndex)
            : m_minPeriod(clock::duration::zero()),
              m_active(false),
              m_deviceIndex(deviceIndex),
              m_dv_ind(GetROCmSMIIndex(deviceIndex)),
              m_dataPoints(0)
        {
            InitROCmSMI();

            initThread();
        }

        HardwareMonitor::~HardwareMonitor()
        {
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_active = false;
                m_exit = true;
            }
            m_cv.notify_all();
            m_thread.join();
        }

        void HardwareMonitor::initThread()
        {
            m_active = false;
            m_exit = false;
            m_thread = std::thread([=](){ this->collect(); });
        }

        void HardwareMonitor::addTempMonitor(rsmi_temperature_type_t sensorType, rsmi_temperature_metric_t metric)
        {
            assertNotActive();

            m_tempMetrics.emplace_back(sensorType, metric);
            m_tempValues.resize(m_tempMetrics.size());
        }

        void HardwareMonitor::addClockMonitor(rsmi_clk_type_t clockType)
        {
            assertNotActive();

            m_clockMetrics.push_back(clockType);
            m_clockValues.resize(m_clockMetrics.size());
        }

        void HardwareMonitor::addFanSpeedMonitor(uint32_t sensorIndex)
        {
            assertNotActive();

            m_fanMetrics.push_back(sensorIndex);
            m_fanValues.resize(m_fanMetrics.size());
        }

        double HardwareMonitor::getAverageTemp(rsmi_temperature_type_t sensorType, rsmi_temperature_metric_t metric)
        {
            assertNotActive();

            if(m_dataPoints == 0)
                throw std::runtime_error("No data points collected!");

            for(size_t i = 0; i < m_tempMetrics.size(); i++)
            {
                if(m_tempMetrics[i] == std::make_tuple(sensorType, metric))
                {
                    int64_t rawValue = m_tempValues[i];
                    if(rawValue == std::numeric_limits<int64_t>::max())
                        return std::numeric_limits<double>::quiet_NaN();

                    return static_cast<double>(rawValue) / (1000.0 * m_dataPoints);
                }
            }

            throw std::runtime_error(concatenate("Can't read temp value that wasn't requested: ", sensorType, " - ", metric));
        }

        double HardwareMonitor::getAverageClock(rsmi_clk_type_t clockType)
        {
            assertNotActive();

            if(m_dataPoints == 0)
                throw std::runtime_error("No data points collected!");

            for(size_t i = 0; i < m_clockMetrics.size(); i++)
            {
                if(m_clockMetrics[i] == clockType)
                {
                    uint64_t rawValue = m_clockValues[i];
                    if(rawValue == std::numeric_limits<uint64_t>::max())
                        return std::numeric_limits<double>::quiet_NaN();

                    return static_cast<double>(rawValue) / (1e6 * m_dataPoints);
                }
            }

            throw std::runtime_error(concatenate("Can't read clock value that wasn't requested: ", clockType));
        }

        double HardwareMonitor::getAverageFanSpeed(uint32_t sensorIndex)
        {
            assertNotActive();

            if(m_dataPoints == 0)
                throw std::runtime_error("No data points collected!");

            for(size_t i = 0; i < m_fanMetrics.size(); i++)
            {
                if(m_fanMetrics[i] == sensorIndex)
                {
                    int64_t rawValue = m_fanValues[i];
                    if(rawValue == std::numeric_limits<int64_t>::max())
                        return std::numeric_limits<double>::quiet_NaN();

                    return static_cast<double>(rawValue) / m_dataPoints;
                }
            }

            throw std::runtime_error(concatenate("Can't read fan value that wasn't requested: ", sensorIndex));
        }

        void HardwareMonitor::start()
        {
            runBetweenEvents(nullptr, nullptr);
        }

        void HardwareMonitor::stop()
        {
            assertActive();

            m_active = false;
        }

        void HardwareMonitor::runUntilEvent(hipEvent_t event)
        {
            runBetweenEvents(nullptr, event);
        }

        void HardwareMonitor::runBetweenEvents(hipEvent_t startEvent, hipEvent_t stopEvent)
        {
            assertNotActive();

            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_active = true;
                m_startEvent = startEvent;
                m_stopEvent = stopEvent;
            }
            m_cv.notify_all();
        }

        void HardwareMonitor::clearValues()
        {
            m_dataPoints = 0;

            for(auto & v: m_tempValues)  v = 0;
            for(auto & v: m_clockValues) v = 0;
            for(auto & v: m_fanValues)   v = 0;

            m_lastCollection = clock::time_point();
            m_nextCollection = clock::time_point();
        }

        void HardwareMonitor::collectOnce()
        {
            for(int i = 0; i < m_tempMetrics.size(); i++)
            {
                // if an error occurred previously, don't overwrite it.
                if(m_tempValues[i] == std::numeric_limits<int64_t>::max())
                    continue;

                rsmi_temperature_type_t sensorType;
                rsmi_temperature_metric_t metric;
                std::tie(sensorType, metric) = m_tempMetrics[i];

                int64_t newValue = 0;
                auto status = rsmi_dev_temp_metric_get(m_dv_ind, sensorType, metric, &newValue);
                if(status != RSMI_STATUS_SUCCESS)
                    m_tempValues[i] = std::numeric_limits<int64_t>::max();
                else
                    m_tempValues[i] += newValue;
            }

            for(int i = 0; i < m_clockMetrics.size(); i++)
            {
                // if an error occurred previously, don't overwrite it.
                if(m_clockValues[i] == std::numeric_limits<uint64_t>::max())
                    continue;

                rsmi_frequencies_t freq;

                uint64_t newValue = 0;
                auto status = rsmi_dev_gpu_clk_freq_get(m_dv_ind, m_clockMetrics[i], &freq);
                if(status != RSMI_STATUS_SUCCESS)
                {
                    m_clockValues[i] = std::numeric_limits<uint64_t>::max();
                }
                else
                {
                    m_clockValues[i] += freq.frequency[freq.current];
                }
            }

            for(int i = 0; i < m_fanMetrics.size(); i++)
            {
                // if an error occurred previously, don't overwrite it.
                if(m_fanValues[i] == std::numeric_limits<int64_t>::max())
                    continue;

                rsmi_frequencies_t freq;

                int64_t newValue = 0;
                auto status = rsmi_dev_fan_rpms_get(m_dv_ind, m_fanMetrics[i], &newValue);
                if(status != RSMI_STATUS_SUCCESS)
                    m_fanValues[i] = std::numeric_limits<int64_t>::max();
                else
                    m_fanValues[i] += newValue;
            }

            m_dataPoints++;
        }

        void HardwareMonitor::sleepIfNecessary()
        {
            //std::this_thread::sleep_until(m_nextCollection);

            m_lastCollection = clock::now();
            m_nextCollection = m_lastCollection + m_minPeriod;
        }

        void HardwareMonitor::collect()
        {
            std::unique_lock<std::mutex> lock(m_mutex);

            while(!m_exit)
            {
                while(!m_active && !m_exit) m_cv.wait(lock);

                if(m_exit)
                    return;

                m_isActive = true;

                clearValues();

                if(m_startEvent != nullptr)
                    HIP_CHECK_EXC(hipEventSynchronize(m_startEvent));

                do
                {

                    collectOnce();
                    sleepIfNecessary();

                    if(m_stopEvent != nullptr && hipEventQuery(m_stopEvent) == hipSuccess)
                        m_active = false;
                }
                while(m_active && !m_exit);

                m_isActive = false;
                lock.unlock();
                m_cv.notify_all();
                lock.lock();

            }
        }

        //void HardwareMonitor::collectBetweenEvents(hipEvent_t startEvent, hipEvent_t stopEvent)
        //{
        //    clearValues();

        //    if(startEvent != nullptr)
        //        HIP_CHECK_EXC(hipEventSynchronize(startEvent));

        //    do
        //    {
        //        collectOnce();
        //        sleepIfNecessary();
        //    }
        //    while(hipEventQuery(stopEvent) != hipSuccess);
        //    
        //}

        void HardwareMonitor::wait()
        {
            if(m_active && m_stopEvent == nullptr)
                throw std::runtime_error("Tried to wait without an end condition.");

            std::unique_lock<std::mutex> lock(m_mutex);
            while(m_isActive) m_cv.wait(lock);
        }

        void HardwareMonitor::assertActive()
        {
            if(!m_active)
                throw std::runtime_error("Monitor is not active.");
        }

        void HardwareMonitor::assertNotActive()
        {
            if(m_active)
                throw std::runtime_error("Monitor is active.");
        }

        HardwareMonitorListener::HardwareMonitorListener(po::variables_map const& args)
            : m_useGPUTimer(         args["use-gpu-timer"].as<bool>()),
              m_monitor(args["device-idx"].as<int>())
        {
            m_monitor.addTempMonitor(0);

            m_monitor.addClockMonitor(RSMI_CLK_TYPE_SYS);
            m_monitor.addClockMonitor(RSMI_CLK_TYPE_SOC);
            m_monitor.addClockMonitor(RSMI_CLK_TYPE_MEM);

            m_monitor.addFanSpeedMonitor();
        }

        void   HardwareMonitorListener::preEnqueues() override
        {
            if(!m_useGPUTimer)
                m_monitor.start();
        }

        void   HardwareMonitorListener::postEnqueues(TimingEvents const& startEvents,
                                                     TimingEvents const&  stopEvents) override
        {
            if(m_useGPUTimer)
            {
                m_monitor.runBetweenEvents(startEvents->front().front(), stopEvents->back().back());
            }
            else
            {
                m_monitor.stop();
            }
        }

        void   HardwareMonitorListener::validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                                         TimingEvents const& startEvents,
                                                         TimingEvents const&  stopEvents) override
        {
            m_monitor.wait();

            m_reporter->report(ResultKey::TempEdge,            m_monitor.getAverageTemp(0));

            m_reporter->report(ResultKey::ClockRateSys,        m_monitor.getAverageClock(RSMI_CLK_TYPE_SYS));
            m_reporter->report(ResultKey::ClockRateSOC,        m_monitor.getAverageClock(RSMI_CLK_TYPE_SOC));
            m_reporter->report(ResultKey::ClockRateMem,        m_monitor.getAverageClock(RSMI_CLK_TYPE_MEM));

            m_reporter->report(ResultKey::FanSpeedRPMs,        m_monitor.getAverageFanSpeed());
            m_reporter->report(ResultKey::HardwareSampleCount, m_monitor.getSamples());
        }
    }
}

