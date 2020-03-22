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

#pragma once

#include "ResultReporter.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include <boost/program_options.hpp>
#include <hip/hip_runtime.h>

namespace Tensile
{
    namespace Client
    {
        namespace po = boost::program_options;
        
        struct PerformanceModel
        {
            double  gFlops = std::numeric_limits<double>::quiet_NaN();
            double  m_memBandwidthMBps = std::numeric_limits<double>::quiet_NaN();
            double  m_peakGFlops = std::numeric_limits<double>::quiet_NaN();
            double  m_eff = std::numeric_limits<double>::quiet_NaN();
            double  m_readMul = std::numeric_limits<double>::quiet_NaN();
        } pm;

        class PerformanceReporter: public ResultReporter
        {
        public:
            static std::shared_ptr<PerformanceReporter> Default(po::variables_map const& args);
        
            PerformanceReporter(int deviceIndex);

            virtual void reportValue_int(std::string const& key, int64_t value) override;
            
            virtual void reportValue_uint(std::string const& key, uint64_t value) override;

            virtual void reportValue_double(std::string const& key, double value) override;

            virtual void preSolution(ContractionSolution const& solution) override;

            virtual void preProblem(ContractionProblem const& problem) override;

            virtual void postSolution() override;
            
            void setNumCUs();

            void setMemoryBusWidth();

            void setMagicNum();

            int     getNumCUs();
            int     getMagicNum();
            double  getMemClock();
            double  getClock();
            double  getReadMultiplier();
            double  getL2ReadHits();
            double  getL2WriteHits();
            double  getReadEff();
            double  getEfficiency();

            virtual void reportValue_string(std::string const& key, std::string const& value) override;
            virtual void reportValue_sizes(std::string const& key, std::vector<size_t> const& value) override;
            void finalizeReport() override;


        protected: 
            hipDeviceProp_t props;
            double  m_clock = std::numeric_limits<double>::quiet_NaN();
            double  m_memClock = std::numeric_limits<double>::quiet_NaN();
            double  m_dgFlops = std::numeric_limits<double>::quiet_NaN();
            int64_t m_gFlops = std::numeric_limits<int64_t>::quiet_NaN();
            int     m_magicNum;
            int     m_numCUs;
            int     m_memoryBusWidth;
            bool    deviceProps = false;
            double  m_l2ReadHits = 0.0; //figure out how to PerformanceReporter::get from client...maybe use NaN
            double  m_l2WriteHits = 0.5; //figure how to PerformanceReporter::get from client...maybe use std::numeric_limits<double>::quiet_NaN();
            double  m_readEff = 0.85; //figure how to PerformanceReporter::get from client..maybe use NaN
        };
    }
}
