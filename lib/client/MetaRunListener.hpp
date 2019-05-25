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

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include "RunListener.hpp"
#include "ResultReporter.hpp"

namespace Tensile
{
    namespace Client
    {
        class MetaRunListener: public RunListener
        {
        public:
            void addListener(std::shared_ptr<RunListener> listener)
            {
                if(!m_firstProblem)
                    throw std::runtime_error("Can't add listeners after benchmarking has begun.");

                listener->setReporter(m_reporter);

                m_listeners.push_back(listener);
            }

            void setReporter(std::shared_ptr<ResultReporter> reporter)
            {
                if(!m_firstProblem)
                    throw std::runtime_error("Can't set reporter after benchmarking has begun.");
                if(m_reporter)
                    throw std::runtime_error("Can't set reporter more than once.");

                m_reporter = reporter;
                for(auto const& listener: m_listeners)
                    listener->setReporter(reporter);

                m_listeners.insert(m_listeners.begin(), reporter);
            }

            virtual void setUpProblem(ContractionProblem const& problem)
            {
                for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                    (*iter)->setUpProblem(problem);
            }

            virtual void setUpSolution(ContractionSolution const& solution)
            {
                for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                    (*iter)->setUpSolution(solution);
            }

            virtual bool needsMoreRunsInSolution()
            {
                for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                    if((*iter)->needsMoreRunsInSolution())
                        return true;

                return false;
            }

            virtual bool isWarmupRun()
            {
                for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                    if((*iter)->isWarmupRun())
                        return true;

                return false;
            }

            virtual void setUpRun(bool isWarmup)
            {
                for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                    (*iter)->setUpRun(isWarmup);
            }

            virtual void tearDownRun()
            {
                for(auto iter = m_listeners.rbegin(); iter != m_listeners.rend(); iter++)
                    (*iter)->tearDownRun();
            }

            virtual void tearDownProblem()
            {
                for(auto iter = m_listeners.rbegin(); iter != m_listeners.rend(); iter++)
                    (*iter)->tearDownProblem();
            }

            virtual void tearDownSolution()
            {
                for(auto iter = m_listeners.rbegin(); iter != m_listeners.rend(); iter++)
                    (*iter)->tearDownSolution();
            }

            virtual void validate(std::shared_ptr<ContractionInputs> inputs)
            {
                for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                    (*iter)->validate(inputs);
            }

            virtual void report() const
            {
                for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                    (*iter)->report();
            }

            virtual int error() const
            {
                for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                {
                    int rv = (*iter)->error();
                    if(rv) return rv;
                }

                return 0;
            }

        private:

            bool m_firstProblem = true;

            std::vector<std::shared_ptr<RunListener>> m_listeners;
            std::vector<std::shared_ptr<ResultReporter>> m_reporters;
        };

    }
}

