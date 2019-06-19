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

#include <ResultFileReporter.hpp>

namespace Tensile
{
    namespace Client
    {
        std::shared_ptr<ResultFileReporter> ResultFileReporter::Default(po::variables_map const& args)
        {
            return std::make_shared<ResultFileReporter>(args["results-file"].as<std::string>());
        }

        ResultFileReporter::ResultFileReporter(std::string const& filename)
            : m_output(filename)
        {
            m_output.setHeaderForKey(ResultKey::ProblemIndex, "GFLOPS");
        }

        template <typename T>
        void ResultFileReporter::reportValue(std::string const& key, T const& value)
        {
            std::string valueStr = boost::lexical_cast<std::string>(value);
            if(key == ResultKey::Validation)
            {
                if(valueStr != "PASSED" && valueStr != "NO_CHECK")
                {
                    m_output.setValueForKey(m_solutionName, -1.0);
                    m_invalidSolution = true;
                }
            }
            else if(key == ResultKey::SolutionName)
            {
                m_solutionName = valueStr;
                m_output.setHeaderForKey(valueStr, valueStr);
            }
            else if(key == ResultKey::SpeedGFlops)
            {
                if(!m_invalidSolution)
                    m_output.setValueForKey(m_solutionName, value);
            }
            else
            {
                m_output.setValueForKey(key, value);
            }
        }

        void ResultFileReporter::reportValue_string(std::string const& key, std::string const& value) override
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_uint(  std::string const& key, uint64_t value) override
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_int(   std::string const& key, int64_t value) override
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_double(std::string const& key, double value) override
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_sizes(std::string const& key, std::vector<size_t> const& value) override
        {
            if(key == ResultKey::ProblemSizes)
            {
                for(size_t i = 0; i < value.size(); i++)
                {
                    std::string key = concatenate("Size", static_cast<char>('I'+i));
                    m_output.setHeaderForKey(key, key);
                    m_output.setValueForKey(key, value[i]);
                }

                // Values for these come separately.
                m_output.setHeaderForKey(ResultKey::LDD, "LDD");
                m_output.setHeaderForKey(ResultKey::LDC, "LDC");
                m_output.setHeaderForKey(ResultKey::LDA, "LDA");
                m_output.setHeaderForKey(ResultKey::LDB, "LDB");
                m_output.setHeaderForKey(ResultKey::TotalFlops, "TotalFlops");
            }
        }

        void ResultFileReporter::postProblem() override
        {
            m_output.writeCurrentRow();
        }

        void ResultFileReporter::postSolution() override
        {
            m_solutionName = "";
            m_invalidSolution = false;
        }

        void ResultFileReporter::finalizeReport() override
        {
        }
    }
}
