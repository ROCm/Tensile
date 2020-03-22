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
                m_gFlops = value;
            }
        }

        void PerformanceReporter::reportValue_double(std::string const& key, double value) 
        {
            if(key == ResultKey::ClockRateSys && deviceProps)
            {
                m_clock = value;
                perf.clock = getClock();
            }
            if(!std::isnan(m_clock) && deviceProps)
            {
                pm.m_peakGFlops = getNumCUs()*getMagicNum()*getReadMultiplier()*m_clock/1000;
                perf.peakGFlops = getPeakGFlops();
            }
            if(key == ResultKey::ClockRateMem && deviceProps)
            {
                m_memClock = value;
                perf.memClock = getMemClock();
                pm.m_memBandwidthMBps = m_memoryBusWidth*m_memClock;
                perf.memBandwidthMBps = getMemBandwidthMBps();
            }
            if(key == ResultKey::SpeedGFlops && deviceProps) 
            {
                m_dgFlops = value;
            }
            if((!std::isnan(m_dgFlops) || !std::isnan(m_gFlops)) && !std::isnan(pm.m_peakGFlops) && deviceProps)
            {
                pm.gFlops = !std::isnan(m_gFlops) ? (double)m_gFlops : m_dgFlops;
                pm.m_eff = 100*pm.gFlops/pm.m_peakGFlops;
                perf.efficiency = getEfficiency();
            }
        }

        void PerformanceReporter::preSolution(ContractionSolution const& solution) 
        {
            report(ResultKey::PeakGFlops, pm.m_peakGFlops);
            report(ResultKey::Efficiency, pm.m_eff);
            report(ResultKey::SpeedGFlops, pm.gFlops);
            report(ResultKey::L2BandwidthMBps, pm.m_memBandwidthMBps*pm.m_readMul);

        }

        void PerformanceReporter::preProblem(ContractionProblem const& problem) 
        {
            int dataEnum = (int)problem.a().dataType();
            std::unordered_map<int,double> readMulMap = {{0,2},{1,1},{2,1},{3,0.5}, {4,4}, {5,8}, {6,2}, {7,4}};

            for(std::unordered_map<int,double>::iterator it=readMulMap.begin(); it != readMulMap.end(); it++)
            {
                if(it->first == dataEnum) pm.m_readMul = it->second;
            }
            
            perf.readMul = getReadMultiplier();
        }

        void PerformanceReporter::postSolution()  
        {
            m_clock = std::numeric_limits<double>::quiet_NaN();
            m_memClock = std::numeric_limits<double>::quiet_NaN();
            m_gFlops = std::numeric_limits<int64_t>::quiet_NaN();
            m_dgFlops = std::numeric_limits<double>::quiet_NaN();
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
        double  PerformanceReporter::getMemClock(){return m_memClock;}
        double  PerformanceReporter::getClock(){return m_clock;}
        bool    PerformanceReporter::getMfma(){return m_mfma;}
        double  PerformanceReporter::getReadMultiplier(){return pm.m_readMul;}
        double  PerformanceReporter::getPeakGFlops(){return pm.m_peakGFlops;}
        double  PerformanceReporter::getL2ReadHits(){return m_l2ReadHits;}
        double  PerformanceReporter::getL2WriteHits(){return m_l2WriteHits;}
        double  PerformanceReporter::getReadEff(){return m_readEff;}
        double  PerformanceReporter::getEfficiency(){return pm.m_eff;}
        double  PerformanceReporter::getMemBandwidthMBps(){return pm.m_memBandwidthMBps;}

        void    PerformanceReporter::reportValue_int(std::string const& key, int64_t value) {}
        void    PerformanceReporter::reportValue_string(std::string const& key, std::string const& value) {}
        void    PerformanceReporter::reportValue_sizes(std::string const& key, std::vector<size_t> const& value) {}
        void    PerformanceReporter::finalizeReport() {}

    }
}
