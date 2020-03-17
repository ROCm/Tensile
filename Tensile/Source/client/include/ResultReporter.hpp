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

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>

#include <boost/program_options.hpp>
#include <hip/hip_runtime.h>

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
            const std::string ProblemCount = "problem-count";
            const std::string ProblemProgress = "problem-progress";
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
            const std::string TimeUS      = "time-us";
            const std::string SpeedGFlops = "gflops";
            const std::string EnqueueTime = "enqueue-time";

            // Performance estimation and granularity
            const std::string Tile0Granularity = "tile0-granularity";
            const std::string Tile1Granularity = "tile1-granularity";
            const std::string CuGranularity = "cu-granularity";
            const std::string WaveGranularity = "wave-granularity";
            const std::string TotalGranularity = "total-granularity";
            const std::string TilesPerCu = "tiles-per-cu";

            const std::string MemReadBytes   = "mem-read-bytes";
            const std::string MemWriteBytes  = "mem-write-bytes";
            const std::string MemReadUs  = "mem-read-us";
            const std::string MemWriteUs = "mem-write-us";
            const std::string MemGlobalReads  = "mem-global-reads";
            const std::string MemGlobalWrites  = "mem-global-writes";
            const std::string AluUs      = "alu-us";
            const std::string Empty          = "empty";

            const std::string Efficiency        = "efficiency";
            const std::string L2ReadHits        = "l2-read-hits";
            const std::string L2WriteHits       = "l2-write-hits";
            const std::string ReadMultiplier    = "read-multiplier";
            const std::string L2BandwidthMBps   = "l2-bandwidth-mbps";
            const std::string PeakGFlops        = "peak-gflops";

            // Hardware monitoring
            const std::string TempEdge            = "temp-edge";
            const std::string ClockRateSys        = "clock-sys";
            const std::string ClockRateSOC        = "clock-soc";
            const std::string ClockRateMem        = "clock-mem";
            const std::string DeviceIndex         = "device-index";
            const std::string FanSpeedRPMs        = "fan-rpm";
            const std::string HardwareSampleCount = "hardware-samples";
        };
        
        struct PerformanceModel
        {
            double  gFlops = std::numeric_limits<double>::quiet_NaN();
            double  m_memBandwidthMBps = std::numeric_limits<double>::quiet_NaN();
            double  m_peakGFlops = std::numeric_limits<double>::quiet_NaN();
            double  m_eff = std::numeric_limits<double>::quiet_NaN();
            double  m_readMul = std::numeric_limits<double>::quiet_NaN();
        } pm;

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
            virtual void logTensor(LogLevel level, std::string const& name, void const* data, TensorDescriptor const& tensor, void const* ptrVal) {}

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

        class PerformanceReporter: public ResultReporter
        {
        public:
            static std::shared_ptr<PerformanceReporter> Default()
            {
                return std::make_shared<PerformanceReporter>();
            }

            virtual void reportValue_int(std::string const& key, int64_t value) override 
            {
                if(key == ResultKey::DeviceIndex && deviceProps == false) 
                {
                    m_deviceIndex = value;
                    hipGetDeviceProperties(&props, m_deviceIndex);
                    setNumCUs();
                    setMagicNum();
                    setMemoryBusWidth();
                    deviceProps = true;
                }
            }
            
            virtual void reportValue_uint(std::string const& key, uint64_t value) override
            {
                if(key == ResultKey::SpeedGFlops && deviceProps) 
                {
                    m_gFlops = value;
                    std::cout<<"m_gflops: " << m_gFlops << std::endl;
                }
                
            }

            virtual void reportValue_double(std::string const& key, double value) override
            {
                if(key == ResultKey::ClockRateSys && deviceProps)
                {
                    m_clock = value;
                    std::cout<<"m_clock: " << m_clock << std::endl;
                }
                if(!std::isnan(m_clock) && deviceProps)
                {
                    pm.m_peakGFlops = getNumCUs()*getMagicNum()*getReadMultiplier()*m_clock/1000;
                    std::cout<<"m_peakGFlops: " << pm.m_peakGFlops << std::endl;
                }
                if(key == ResultKey::ClockRateMem && deviceProps)
                {
                    m_memClock = value;
                    std::cout<<"m_memClock: " << m_memClock << std::endl;
                    pm.m_memBandwidthMBps = m_memoryBusWidth*m_memClock;
                    std::cout<<"m_memoryBusWidth: " << m_memoryBusWidth << std::endl;
                    std::cout<<"m_L2memBandwidthMBps: " << pm.m_memBandwidthMBps*pm.m_readMul << std::endl;
                    report(ResultKey::L2BandwidthMBps, pm.m_memBandwidthMBps*pm.m_readMul); 
                }
                if(key == ResultKey::SpeedGFlops && deviceProps) 
                {
                    m_dgFlops = value;
                    std::cout<<"m_dgflops: " << m_dgFlops << std::endl;
                }
                if((!std::isnan(m_dgFlops) || !std::isnan(m_gFlops)) && !std::isnan(pm.m_peakGFlops) && deviceProps)
                {
                    std::cout<<"m_gFlops: "<<m_gFlops<<" m_dgFlops: "<<m_dgFlops<<std::endl;
                    pm.gFlops = !std::isnan(m_gFlops) ? (double)m_gFlops : m_dgFlops;
                    pm.m_eff = 100*pm.gFlops/pm.m_peakGFlops;
                    std::cout<<"m_eff: " << pm.m_eff << std::endl;
                }
            }

            virtual void preSolution(ContractionSolution const& solution) override
            {
                report(ResultKey::PeakGFlops, pm.m_peakGFlops); 
                report(ResultKey::Efficiency, pm.m_eff); 
                report(ResultKey::SpeedGFlops, pm.gFlops); 
                report(ResultKey::L2BandwidthMBps, pm.m_memBandwidthMBps*pm.m_readMul); 
            }

            virtual void preProblem(ContractionProblem const& problem) override
            {
                int dataEnum = (int)problem.a().dataType();
                std::unordered_map<int,double> readMulMap = {{0,2},{1,1},{2,1},{3,0.5}, {4,4}, {5,8}, {6,2}, {7,4}};

                for(std::unordered_map<int,double>::iterator it=readMulMap.begin(); it != readMulMap.end(); it++)
                {
                    if(it->first == dataEnum) pm.m_readMul = it->second;
                }
    
                std::cout<<"m_readMul " << pm.m_readMul << std::endl;
            }

            virtual void postSolution() override 
            {
                m_clock = std::numeric_limits<double>::quiet_NaN();
                m_memClock = std::numeric_limits<double>::quiet_NaN();
                m_gFlops = std::numeric_limits<int64_t>::quiet_NaN();
                m_dgFlops = std::numeric_limits<double>::quiet_NaN();
            }
            
            void setNumCUs()
            {
                m_numCUs = props.multiProcessorCount;       
            }

            void setMemoryBusWidth()
            {
                m_memoryBusWidth = props.memoryBusWidth/1024;
            }

            void setMagicNum()
            {
                if(getNumCUs() == 120) m_magicNum = 128;
                else m_magicNum = 64;
            }

            int     getNumCUs(){return m_numCUs;}
            int     getMagicNum(){return m_magicNum;}
            double  getMemClock(){return m_memClock;}
            double  getClock(){return m_clock;}
            double  getReadMultiplier(){return pm.m_readMul;}
            double  getL2ReadHits(){return m_l2ReadHits;}
            double  getL2WriteHits(){return m_l2WriteHits;}
            double  getReadEff(){return m_readEff;}
            
            /*double  getMemBandwidthMBps(){return m_memoryBusWidth*m_memClock;}
            double  getL2BandwidthMBps(){return m_memBandwidthMBps*m_readMul;}
            double  getPeakGFlops(){return m_peakGFlops;}
            double  getEfficiency(){return m_eff;}
            */

            virtual void reportValue_string(std::string const& key, std::string const& value) override{}
            virtual void reportValue_sizes(std::string const& key, std::vector<size_t> const& value) override{}
            virtual void finalizeReport() override{}
        
        protected: 
            hipDeviceProp_t props;
            double  m_clock = std::numeric_limits<double>::quiet_NaN();
            double  m_memClock = std::numeric_limits<double>::quiet_NaN();
            double  m_dgFlops = std::numeric_limits<double>::quiet_NaN();
            int64_t m_gFlops = std::numeric_limits<int64_t>::quiet_NaN();
            int     m_magicNum;
            int     m_numCUs;
            int     m_memoryBusWidth;
            int64_t m_deviceIndex = -1;
            bool    deviceProps = false;
            double  m_l2ReadHits = 0.0; //figure out how to get from client...maybe use NaN
            double  m_l2WriteHits = 0.5; //figure how to get from client...maybe use std::numeric_limits<double>::quiet_NaN();
            double  m_readEff = 0.85; //figure how to get from client..maybe use NaN
        };
    }
}
