/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        std::shared_ptr<ResultFileReporter>
            ResultFileReporter::Default(po::variables_map const& args)
        {
            return std::make_shared<ResultFileReporter>(args["results-file"].as<std::string>(),
                                                        args["csv-export-extra-cols"].as<bool>(),
                                                        args["csv-merge-same-problems"].as<bool>());
        }

        ResultFileReporter::ResultFileReporter(std::string const& filename,
                                               bool               exportExtraCols,
                                               bool               mergeSameProblems)
            : m_output(filename)
            , m_extraCol(exportExtraCols)
            , m_mergeSameProblems(mergeSameProblems)
        {
            m_output.setHeaderForKey(ResultKey::ProblemIndex, "GFlops");
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
            else if(key == ResultKey::TimeUS)
            {
                // cascade from BenchmarkTimer, Time-US first
                ++m_currSolutionIdx;
                double timeUS = std::stod(valueStr);
                if(!m_invalidSolution)
                {
                    if(m_fasterTimeUS < 0 || m_fasterTimeUS > timeUS)
                    {
                        m_fasterTimeUS = timeUS;
                    }
                }
            }
            else if(key == ResultKey::SpeedGFlops)
            {
                // cascade from BenchmarkTimer, SpeedGFlops second
                if(!m_invalidSolution)
                {
                    m_output.setValueForKey(m_solutionName, value);

                    int64_t gflops = std::stoull(valueStr);
                    if(m_fastestGflops < gflops)
                    {
                        m_winnerSolution    = m_solutionName;
                        m_winnerSolutionIdx = m_currSolutionIdx;
                        m_fastestGflops     = gflops;
                    }
                }
            }
            else
            {
                m_output.setValueForKey(key, value);
            }
        }

        void ResultFileReporter::reportValue_string(std::string const& key,
                                                    std::string const& value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_uint(std::string const& key, uint64_t value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_int(std::string const& key, int64_t value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_double(std::string const& key, double value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_sizes(std::string const&         key,
                                                   std::vector<size_t> const& value)
        {
            if(key == ResultKey::ProblemSizes)
            {
                for(size_t i = 0; i < value.size(); i++)
                {
                    std::string key = concatenate("Size", static_cast<char>('I' + i));
                    m_output.setHeaderForKey(key, key);
                    m_output.setValueForKey(key, value[i]);
                }

                // Values for these come separately.
                m_output.setHeaderForKey(ResultKey::LDD, "LDD");
                m_output.setHeaderForKey(ResultKey::LDC, "LDC");
                m_output.setHeaderForKey(ResultKey::LDA, "LDA");
                m_output.setHeaderForKey(ResultKey::LDB, "LDB");
                m_output.setHeaderForKey(ResultKey::TotalFlops, "TotalFlops");
                if(m_extraCol)
                {
                    m_output.setHeaderForKey(ResultKey::FastestGFlops, "WinnerGFlops");
                    m_output.setHeaderForKey(ResultKey::TimeUS, "WinnerTimeUS");
                    m_output.setHeaderForKey(ResultKey::SolutionWinnerIdx, "WinnerIdx");
                    m_output.setHeaderForKey(ResultKey::SolutionWinner, "WinnerName");
                }
            }
        }

        void ResultFileReporter::mergeRow(std::unordered_map<std::string, std::string>& newRow)
        {
            m_currProbID = std::stoull(newRow[ResultKey::ProblemIndex]);
            if(m_probMap.count(m_currProbID) == 0)
            {
                m_probMap[m_currProbID] = newRow;
                return;
            }

            auto& oldRow = m_probMap[m_currProbID];
            for(auto& oldRowIter : oldRow)
            {
                const std::string& key = oldRowIter.first;
                if(key.compare(ResultKey::ProblemIndex) == 0
                   || key.find("Size") != std::string::npos || key.compare(ResultKey::LDD) == 0
                   || key.compare(ResultKey::LDC) == 0 || key.compare(ResultKey::LDA) == 0
                   || key.compare(ResultKey::LDB) == 0 || key.compare(ResultKey::TotalFlops) == 0)
                {
                    // these data should be the same for same problem
                    assert(oldRowIter.second == newRow[key]);
                }
                else if(key.compare(ResultKey::FastestGFlops) == 0)
                {
                    // if new row is better, update
                    uint64_t oldFastest = std::stoull(oldRowIter.second);
                    uint64_t newFastest = std::stoull(newRow[key]);
                    if(newFastest > oldFastest)
                    {
                        oldRow[ResultKey::FastestGFlops]     = newRow[ResultKey::FastestGFlops];
                        oldRow[ResultKey::TimeUS]            = newRow[ResultKey::TimeUS];
                        oldRow[ResultKey::SolutionWinnerIdx] = newRow[ResultKey::SolutionWinnerIdx];
                        oldRow[ResultKey::SolutionWinner]    = newRow[ResultKey::SolutionWinner];
                    }
                }
                else if(key.compare(ResultKey::TimeUS) == 0
                        || key.compare(ResultKey::SolutionWinnerIdx) == 0
                        || key.compare(ResultKey::SolutionWinner) == 0)
                {
                    // skip, we update these together with FastestGFlops
                    continue;
                }
                else
                {
                    // these are gflops for each solution
                    // if new row is better, update
                    uint64_t oldFastest = std::stoull(oldRowIter.second);
                    uint64_t newFastest = std::stoull(newRow[key]);
                    if(newFastest > oldFastest)
                    {
                        oldRow[key] = newRow[key];
                    }
                }
            }
        }

        void ResultFileReporter::postProblem()
        {
            if(m_extraCol)
            {
                // update winner
                m_output.setValueForKey(ResultKey::FastestGFlops, m_fastestGflops);
                m_output.setValueForKey(ResultKey::TimeUS, m_fasterTimeUS);
                m_output.setValueForKey(ResultKey::SolutionWinnerIdx, m_winnerSolutionIdx);
                m_output.setValueForKey(ResultKey::SolutionWinner, m_winnerSolution);
            }
            // reset
            m_winnerSolution    = "";
            m_currSolutionIdx   = -1;
            m_winnerSolutionIdx = -1;
            m_fastestGflops     = -1;
            m_fasterTimeUS      = -1;

            if(!m_mergeSameProblems)
            {
                m_output.writeCurrentRow();
            }
            else
            {
                std::unordered_map<std::string, std::string> curRow;
                m_output.readCurrentRow(curRow);
                m_output.clearCurrentRow();
                // for (auto & field : curRow )
                //     std::cout << "key:" << field.first << ", value:" << field.second << std::endl;
                this->mergeRow(curRow);
            }
        }

        void ResultFileReporter::postSolution()
        {
            m_solutionName    = "";
            m_invalidSolution = false;
        }

        void ResultFileReporter::finalizeReport()
        {
            if(m_mergeSameProblems)
            {
                for(auto& probIter : m_probMap)
                {
                    auto& single_row = probIter.second;
                    for(auto& field : single_row)
                    {
                        m_output.setValueForKey(field.first, field.second);
                    }
                    m_output.writeCurrentRow();
                }
            }
        }
    } // namespace Client
} // namespace Tensile
