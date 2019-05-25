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

            virtual void setReporter(std::shared_ptr<ResultReporter> reporter) {}

            virtual void setUpProblem(ContractionProblem const& problem) {}
            virtual void setUpSolution(ContractionSolution const& solution) {}

            virtual bool needsMoreRunsInSolution() { return false; }
            virtual bool isWarmupRun() { return false; }

            virtual void setUpRun(bool isWarmup) {}
            virtual void tearDownRun() {}
            virtual void validate(std::shared_ptr<ContractionInputs> inputs) {}

            virtual void tearDownSolution() {}
            virtual void tearDownProblem() {}

            virtual void report() const {}

            virtual int error() const { return 0; }
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

            virtual void setUpProblem(ContractionProblem const& problem)
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->setUpProblem(problem);
            }

            virtual void setUpSolution(ContractionSolution const& solution)
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->setUpSolution(solution);
            }

            virtual bool needsMoreRunsInSolution()
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    if((*iter)->needsMoreRunsInSolution())
                        return true;

                return false;
            }

            virtual bool isWarmupRun()
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    if((*iter)->isWarmupRun())
                        return true;

                return false;
            }

            virtual void setUpRun(bool isWarmup)
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->setUpRun(isWarmup);
            }

            virtual void tearDownRun()
            {
                for(auto iter = m_reporters.rbegin(); iter != m_reporters.rend(); iter++)
                    (*iter)->tearDownRun();
            }

            virtual void tearDownProblem()
            {
                for(auto iter = m_reporters.rbegin(); iter != m_reporters.rend(); iter++)
                    (*iter)->tearDownProblem();
            }

            virtual void tearDownSolution()
            {
                for(auto iter = m_reporters.rbegin(); iter != m_reporters.rend(); iter++)
                    (*iter)->tearDownSolution();
            }

            virtual void validate(std::shared_ptr<ContractionInputs> inputs)
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->validate(inputs);
            }

            virtual void report() const
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->report();
            }

            virtual int error() const
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

            virtual void reportValue(std::string const& key, std::string const& value)
            {
                if(m_keySet.find(key) != m_keySet.end())
                {
                    if(m_inSolution)
                        m_currentSolutionRow[key] = value;
                    else
                        m_currentProblemRow[key] = value;
                }
            }

            virtual bool logAtLevel(LogLevel level) { return level <= m_level; };

            virtual void logMessage(LogLevel level, std::string const& message)
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

            virtual void logTensor(LogLevel level, std::string const& name, void const* data, TensorDescriptor const& tensor)
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

            virtual void setReporter(std::shared_ptr<ResultReporter> reporter) {}

            virtual void setUpProblem(ContractionProblem const& problem)
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

            virtual void setUpSolution(ContractionSolution const& solution)
            {
                m_inSolution = true;

                report("solution", solution.name());
            }

            virtual void tearDownSolution()
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

            virtual void tearDownProblem()
            {
                m_currentProblemRow.clear();
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
