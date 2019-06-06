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

#include <unordered_set>
#include <string>

#include <boost/lexical_cast.hpp>

#include "RunListener.hpp"

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

        class ResultReporter: public RunListener
        {
        public:
            /**
             * Reports the value for a key, related to this run.
             */
            template <typename T>
            void report(std::string const& key, T const& value)
            {
                reportValue(key, boost::lexical_cast<std::string>(value));
            }

            virtual void reportValue(std::string const& key, std::string const& value) {}

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
            virtual void   postEnqueues() override {}
            virtual void   validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                            TimingEvents const& startEvents,
                                            TimingEvents const&  stopEvents) override {}

            // finalizeReport() deliberately left out of here to force it to be implemented in subclasses.

            virtual int error() const override
            {
                return 0;
            }
        };

        class MetaResultReporter: public ResultReporter
        {
        public:
            virtual void addReporter(std::shared_ptr<ResultReporter> reporter)
            {
                m_reporters.push_back(reporter);
            }

            virtual void reportValue(std::string const& key, std::string const& value)
            {
                for(auto r: m_reporters)
                    r->reportValue(key, value);
            }

            virtual bool logAtLevel(LogLevel level)
            {
                for(auto r: m_reporters)
                    if(r->logAtLevel(level))
                        return true;
                return false;
            }

            virtual void logMessage(LogLevel level, std::string const& message)
            {
                for(auto r: m_reporters)
                    r->logMessage(level, message);
            }

            virtual void logTensor(LogLevel level, std::string const& name, void const* data, TensorDescriptor const& tensor)
            {
                for(auto r: m_reporters)
                    r->logTensor(level, name, data, tensor);
            }

            /// RunListener interface functions

            virtual bool needMoreBenchmarkRuns() const override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    if((*iter)->needMoreBenchmarkRuns())
                        return true;

                return false;
            }

            virtual void preBenchmarkRun() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preBenchmarkRun();
            }

            virtual void postBenchmarkRun() override
            {
                for(auto iter = m_reporters.rbegin(); iter != m_reporters.rend(); iter++)
                    (*iter)->postBenchmarkRun();
            }

            virtual void preProblem(ContractionProblem const& problem) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preProblem(problem);
            }

            virtual void postProblem() override
            {
                for(auto iter = m_reporters.rbegin(); iter != m_reporters.rend(); iter++)
                    (*iter)->postProblem();
            }

            virtual void preSolution(ContractionSolution const& solution) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preSolution(solution);
            }

            virtual void postSolution() override
            {
                for(auto iter = m_reporters.rbegin(); iter != m_reporters.rend(); iter++)
                    (*iter)->postSolution();
            }

            virtual bool needMoreRunsInSolution() const override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    if((*iter)->needMoreRunsInSolution())
                        return true;

                return false;
            }

            virtual size_t numWarmupRuns() override { return 0; }
            virtual void   setNumWarmupRuns(size_t count) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->setNumWarmupRuns(count);
            }

            virtual void   preWarmup() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preWarmup();
            }

            virtual void   postWarmup() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->postWarmup();
            }

            virtual void   validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                           TimingEvents const& startEvents,
                                           TimingEvents const&  stopEvents) override 
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->validateWarmups(inputs, startEvents, stopEvents);
            }

            virtual size_t numSyncs() override
            {
                return 0;
            }

            virtual void   setNumSyncs(size_t count) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->setNumSyncs(count);
            }

            virtual void   preSyncs() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preSyncs();
            }

            virtual void   postSyncs() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->postSyncs();
            }

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            }

            virtual void   setNumEnqueuesPerSync(size_t count) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->setNumEnqueuesPerSync(count);
            }

            virtual void   preEnqueues() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preEnqueues();
            }
            virtual void   postEnqueues() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->postEnqueues();
            }

            virtual void   validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                            TimingEvents const& startEvents,
                                            TimingEvents const&  stopEvents) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->validateEnqueues(inputs, startEvents, stopEvents);
            }


            virtual void finalizeReport() const override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->finalizeReport();
            }

            virtual int error() const override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                {
                    int rv = (*iter)->error();
                    if(rv) return rv;
                }

                return 0;
            }

        private:

            std::vector<std::shared_ptr<ResultReporter>> m_reporters;
        };

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

            virtual void reportValue(std::string const& key, std::string const& value) override
            {
                if(m_keySet.find(key) != m_keySet.end())
                {
                    if(m_inSolution)
                        m_currentSolutionRow[key] = value;
                    else
                        m_currentProblemRow[key] = value;
                }
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

                report("operation", problem.operationIdentifier());
            }

            virtual void preSolution(ContractionSolution const& solution) override
            {
                m_inSolution = true;

                report("solution", solution.name());
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
