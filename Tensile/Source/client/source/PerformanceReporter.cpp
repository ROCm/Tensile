/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "PerformanceReporter.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>

namespace Tensile
{
    namespace Client
    {
        std::shared_ptr<PerformanceReporter> PerformanceReporter::Default(po::variables_map const& args)
        {
            int     deviceIndex = args["device-idx"].as<int>();
            double  l2ReadHits = args["perf-l2-read-hits"].as<double>();
            double  l2WriteHits = args["perf-l2-write-hits"].as<double>();
            double  readEff = args["perf-read-efficiency"].as<double>();
            bool    mfma = args["perf-mfma"].as<bool>();
 
            return std::make_shared<PerformanceReporter>(deviceIndex, l2ReadHits, l2WriteHits, readEff, mfma);
        }
        
        PerformanceReporter::PerformanceReporter(int deviceIndex, double l2ReadHits, double l2WriteHits, double readEff, bool mfma)
        {
            hipGetDeviceProperties(&props, deviceIndex);
            setNumCUs();
            setMemoryBusWidth();
            setPerfModel(l2ReadHits, l2WriteHits, readEff, mfma);
            setMagicNum();
            deviceProps = true;
            
            perf.l2ReadHitRate = getL2ReadHits();
            perf.l2WriteHitRate = getL2WriteHits();
            perf.readEff = getReadEff();
            perf.CUs = getNumCUs();
        }
        
        void PerformanceReporter::reportValue_uint(std::string const& key, uint64_t value) 
        {
            if(key == ResultKey::SpeedGFlops && deviceProps) 
            {
                reportValue_numeric(key, value);
            }
        }

        void PerformanceReporter::reportValue_double(std::string const& key, double value) 
        {
            if(key == ResultKey::ClockRateSys && deviceProps)
            {
                m_clockMhz = value;
                perf.clock = getClock();
            }
            if(!std::isnan(m_clockMhz) && deviceProps)
            {
                m_peakGFlops = getNumCUs()*getMagicNum()*getReadMultiplier()*m_clockMhz/1000;
                perf.peakGFlops = getPeakGFlops();
            }
            if(key == ResultKey::ClockRateMem && deviceProps)
            {
                m_memClockMhz = value;
                perf.memClock = getMemClock();
                m_memBandwidthMBps = m_memoryBusWidth*m_memClockMhz;
                perf.memBandwidthMBps = getMemBandwidthMBps();
            }
            if(key == ResultKey::SpeedGFlops && deviceProps) 
            {
                reportValue_numeric(key, value);
            }
        }

        template <typename T> 
        void PerformanceReporter::reportValue_numeric(std::string const& key, T value)
        {
            if(key == ResultKey::SpeedGFlops && deviceProps)
            {
                m_gFlops = (double)value;
                if(!std::isnan(m_peakGFlops) && deviceProps)
                {
                    m_eff = 100*m_gFlops/m_peakGFlops;
                    perf.efficiency = getEfficiency();
                }
            }
        }

        void PerformanceReporter::preProblem(ContractionProblem const& problem) 
        {
            int dataEnum = (int)problem.a().dataType();
            std::unordered_map<int,double> readMulMap = {{0,2},{1,1},{2,1},{3,0.5}, {4,4}, {5,8}, {6,2}, {7,4}};

            for(std::unordered_map<int,double>::iterator it=readMulMap.begin(); it != readMulMap.end(); it++)
            {
                if(it->first == dataEnum) m_readMul = it->second;
            }
            
            perf.readMul = getReadMultiplier();
        }

        void PerformanceReporter::preSolution(ContractionSolution const& solution)
        {
            report(ResultKey::L2BandwidthMBps, perf.memBandwidthMBps*perf.readMul);
        }
        
        void PerformanceReporter::postSolution()  
        {
            //report(ResultKey::PeakGFlops, perf.peakGFlops);
            //report(ResultKey::Efficiency, perf.efficiency);
            //report(ResultKey::SpeedGFlops, perf.gFlops);
            //report(ResultKey::L2BandwidthMBps, perf.memBandwidthMBps*perf.readMul);
            m_clockMhz = std::numeric_limits<double>::quiet_NaN();
            m_memClockMhz = std::numeric_limits<double>::quiet_NaN();
            m_gFlops = std::numeric_limits<double>::quiet_NaN();
            m_peakGFlops = std::numeric_limits<double>::quiet_NaN();
            m_memBandwidthMBps = std::numeric_limits<double>::quiet_NaN();
        }
        
        void PerformanceReporter::setPerfModel(double l2ReadHits, double l2WriteHits, double readEff, bool mfma)
        {
            m_l2ReadHits = l2ReadHits;
            m_l2WriteHits = l2WriteHits;
            m_readEff = readEff;
            m_mfma = mfma;
        }
        
        void PerformanceReporter::setNumCUs()
        {
            m_numCUs = props.multiProcessorCount;
        }

        void PerformanceReporter::setMemoryBusWidth()
        {
            m_memoryBusWidth = props.memoryBusWidth/1024;
        }

        void PerformanceReporter::setMagicNum()
        {
            if(getMfma()) m_magicNum = 128;
            else m_magicNum = 64;
        }

        int     PerformanceReporter::getNumCUs(){return m_numCUs;}
        int     PerformanceReporter::getMagicNum(){return m_magicNum;}
        double  PerformanceReporter::getMemClock(){return m_memClockMhz;}
        double  PerformanceReporter::getClock(){return m_clockMhz;}
        bool    PerformanceReporter::getMfma(){return m_mfma;}
        double  PerformanceReporter::getReadMultiplier(){return m_readMul;}
        double  PerformanceReporter::getPeakGFlops(){return m_peakGFlops;}
        double  PerformanceReporter::getL2ReadHits(){return m_l2ReadHits;}
        double  PerformanceReporter::getL2WriteHits(){return m_l2WriteHits;}
        double  PerformanceReporter::getReadEff(){return m_readEff;}
        double  PerformanceReporter::getEfficiency(){return m_eff;}
        double  PerformanceReporter::getMemBandwidthMBps(){return m_memBandwidthMBps;}

        void    PerformanceReporter::reportValue_int(std::string const& key, int64_t value) {}
        void    PerformanceReporter::reportValue_string(std::string const& key, std::string const& value) {}
        void    PerformanceReporter::reportValue_sizes(std::string const& key, std::vector<size_t> const& value) {}
        void    PerformanceReporter::finalizeReport() {}

    }
}
