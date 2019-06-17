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

#include "ResultReporter.hpp"

#include <string>
#include <unordered_set>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace Tensile
{
    namespace Client
    {
        class LogReporter: public ResultReporter
        {
        public:
            LogReporter(LogLevel level, std::initializer_list<const char *> keys, std::ostream & stream)
                : m_level(level),
                  m_stream(stream)
            {
                m_keys.reserve(keys.size());
                for(const char * key: keys)
                {
                    m_keys.push_back(key);
                    m_keySet.insert(key);
                }
            }

            LogReporter(LogLevel level, std::initializer_list<std::string> keys, std::ostream & stream)
                : m_level(level),
                  m_keys(keys),
                  m_keySet(keys),
                  m_stream(stream)
            {
            }

            LogReporter(LogLevel level, std::initializer_list<std::string> keys, std::shared_ptr<std::ostream> stream)
                : m_level(level),
                  m_keys(keys),
                  m_keySet(keys),
                  m_stream(*stream),
                  m_ownedStream(stream)
            {
            }

            static std::shared_ptr<LogReporter> Default(po::variables_map const& args)
            {
                using namespace ResultKey;
                return std::shared_ptr<LogReporter>(
                        new LogReporter(LogLevel::Debug,
                                        {OperationIdentifier, SolutionName,
                                         Validation, TimeNS, SpeedGFlops,
                                         TempEdge, ClockRateSys, ClockRateSOC, ClockRateMem,
                                         FanSpeedRPMs, HardwareSampleCount},
                                        std::cout));
            }

            virtual void reportValue_string(std::string const& key, std::string const& value) override
            {
                if(m_keySet.find(key) != m_keySet.end())
                {
                    if(m_inSolution)
                        m_currentSolutionRow[key] = value;
                    else
                        m_currentProblemRow[key] = value;
                }
            }

            virtual void reportValue_uint(std::string const& key, uint64_t value) override
            {
                reportValue_string(key, boost::lexical_cast<std::string>(value));
            }

            virtual void reportValue_int(std::string const& key, int64_t value) override
            {
                reportValue_string(key, boost::lexical_cast<std::string>(value));
            }

            virtual void reportValue_double(std::string const& key, double value) override
            {
                reportValue_string(key, boost::lexical_cast<std::string>(value));
            }

            virtual bool logAtLevel(LogLevel level) override
            {
                return level <= m_level;
            }

            virtual void logMessage(LogLevel level, std::string const& message) override
            {
                if(logAtLevel(level))
                {
                    m_stream << message;
                    m_stream.flush();
                }
            }

            template <typename T>
            void logTensorTyped(LogLevel level, std::string const& name, T const* data, TensorDescriptor const& tensor)
            {
                if(logAtLevel(level))
                {
                    m_stream << name << ": " << tensor << std::endl;
                    WriteTensor(m_stream, data, tensor);
                }
            }

            virtual void logTensor(LogLevel level, std::string const& name, void const* data, TensorDescriptor const& tensor) override
            {
                if(logAtLevel(level))
                {
                    if(tensor.dataType() == DataType::Float)
                        logTensorTyped(level, name, reinterpret_cast<float const*>(data), tensor);
                    else
                        throw std::runtime_error(concatenate("Can't log tensor of type ", tensor.dataType()));
                }
            }

            /// RunListener interface functions

            virtual void setReporter(std::shared_ptr<ResultReporter> reporter) override
            {}

            virtual void preProblem(ContractionProblem const& problem) override
            {
                m_inSolution = false;

                if(m_firstRun)
                {
                    streamJoin(m_stream, m_keys, ", ");
                    m_stream << std::endl;

                    m_firstRun = false;
                }

                report(ResultKey::OperationIdentifier, problem.operationIdentifier());
            }

            virtual void preSolution(ContractionSolution const& solution) override
            {
                m_inSolution = true;

                report(ResultKey::SolutionName, solution.name());
            }

            virtual void postSolution() override
            {
                bool first = true;

                for(auto key: m_keys)
                {
                    if(!first)
                        m_stream << ", ";
                    first = false;

                    auto fromSolution = m_currentSolutionRow.find(key);
                    auto fromProblem = m_currentProblemRow.find(key);

                    if(fromSolution != m_currentSolutionRow.end())
                    {
                        m_stream << boost::lexical_cast<std::string>(fromSolution->second);
                    }
                    else if(fromProblem != m_currentProblemRow.end())
                    {
                        m_stream << boost::lexical_cast<std::string>(fromProblem->second);
                    }
                    else
                    {
                        throw std::runtime_error(concatenate("Value not reported for key ", key));
                    }
                }

                m_stream << std::endl;

                m_inSolution = false;

                m_currentSolutionRow.clear();
            }

            virtual void postProblem() override
            {
                m_currentProblemRow.clear();
            }

            virtual void finalizeReport() const
            {
            }

        private:
            LogLevel m_level;
            std::vector<std::string> m_keys;
            std::unordered_set<std::string> m_keySet;

            std::unordered_map<std::string, std::string> m_currentProblemRow;
            std::unordered_map<std::string, std::string> m_currentSolutionRow;

            std::ostream & m_stream;
            std::shared_ptr<std::ostream> m_ownedStream;

            bool m_firstRun = true;
            bool m_inSolution = false;
        };
    }
}
