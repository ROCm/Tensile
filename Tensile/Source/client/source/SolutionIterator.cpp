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

#include "SolutionIterator.hpp"

#include "ResultReporter.hpp"

namespace Tensile
{
    namespace Client
    {
        std::shared_ptr<SolutionIterator> SolutionIterator::Default(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
            std::shared_ptr<Hardware> hardware,
            po::variables_map const& args)
        {
            bool bestSolution = args["best-solution"].as<bool>();

            if(bestSolution)
            {
                return std::make_shared<BestSolutionIterator>(library, hardware);
            }
            else
            {
                int firstSolutionIdx = args["solution-start-idx"].as<int>();
                int numSolutions = args["num-solutions"].as<int>();

                return std::make_shared<AllSolutionsIterator>(library, hardware,
                                                              firstSolutionIdx, numSolutions);
            }
        }

        SolutionIterator::SolutionIterator(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
            std::shared_ptr<Hardware> hardware)
            : m_library(library),
              m_hardware(hardware)
        {
        }

        void SolutionIterator::preProblem(ContractionProblem const& problem)
        {
            m_problem = problem;
        }

        bool SolutionIterator::checkSolution(ContractionSolution const& solution)
        {
            if(!(*solution.hardwarePredicate)(*m_hardware))
            {
                m_reporter->report(ResultKey::Validation, "WRONG_HARDWARE");
                if(m_reporter->logAtLevel(LogLevel::Verbose))
                {
                    std::ostringstream msg;
                    solution.hardwarePredicate->debugEval(*m_hardware, msg);
                    m_reporter->log(LogLevel::Verbose, msg.str());
                }

                return false;
            }

            if(!(*solution.problemPredicate)(m_problem))
            {
                m_reporter->report(ResultKey::Validation, "DID_NOT_SATISFY_ASSERTS");
                if(m_reporter->logAtLevel(LogLevel::Verbose))
                {
                    std::ostringstream msg;
                    solution.problemPredicate->debugEval(m_problem, msg);
                    m_reporter->log(LogLevel::Verbose, msg.str());
                }

                return false;
            }

            return true;
        }

        bool SolutionIterator::runCurrentSolution()
        {
            auto solution = getSolution();
            return checkSolution(*solution);
        }

        AllSolutionsIterator::AllSolutionsIterator(
                std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
                std::shared_ptr<Hardware> hardware,
                int firstSolutionIdx, int numSolutions)
            : SolutionIterator(library, hardware)
        {
            m_firstSolutionIdx = firstSolutionIdx;

            if(m_firstSolutionIdx < 0)
                m_firstSolutionIdx = library->solutions.begin()->first;

            if(numSolutions < 0)
            {
                auto iter = library->solutions.rbegin();
                m_lastSolutionIdx = iter->first;
            }
            else
            {
                m_lastSolutionIdx = m_firstSolutionIdx + numSolutions-1;
            }

            m_currentSolutionIdx = m_firstSolutionIdx;
        }

        void AllSolutionsIterator::preProblem(ContractionProblem const& problem)
        {
            SolutionIterator::preProblem(problem);

            m_currentSolutionIdx = m_firstSolutionIdx;
        }

        void AllSolutionsIterator::postProblem()
        {
        }

        void AllSolutionsIterator::preSolution(ContractionSolution const& solution)
        {
            m_reporter->report(ResultKey::SolutionIndex,
                               m_currentSolutionIdx);
            m_reporter->report(ResultKey::SolutionProgress,
                               concatenate(m_currentSolutionIdx, "/", m_lastSolutionIdx));
        }

        void AllSolutionsIterator::postSolution()
        {
            m_currentSolutionIdx++;
        }

        bool AllSolutionsIterator::moreSolutionsInProblem() const
        {
            return m_currentSolutionIdx <= m_lastSolutionIdx;
        }

        std::shared_ptr<ContractionSolution> AllSolutionsIterator::getSolution()
        {
            auto iter = m_library->solutions.find(m_currentSolutionIdx);
            if(iter == m_library->solutions.end())
                return std::shared_ptr<ContractionSolution>();

            return iter->second;
        }

        BestSolutionIterator::BestSolutionIterator(
                std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library, 
                std::shared_ptr<Hardware> hardware)
            : SolutionIterator(library, hardware)
        {
        }

        void BestSolutionIterator::preProblem(ContractionProblem const& problem)
        {
            SolutionIterator::preProblem(problem);

            m_currentSolution = m_library->findBestSolution(m_problem, *m_hardware);
            m_usedCurrentSolution = false;
        }

        void BestSolutionIterator::postProblem()
        {
        }

        void BestSolutionIterator::preSolution(ContractionSolution const& solution)
        {
            m_reporter->report(ResultKey::SolutionIndex, 0);
            m_reporter->report(ResultKey::SolutionProgress, "1/1");
        }

        void BestSolutionIterator::postSolution()
        {
            m_usedCurrentSolution = true;
        }

        bool BestSolutionIterator::moreSolutionsInProblem() const
        {
            return !m_usedCurrentSolution;
        }

        std::shared_ptr<ContractionSolution> BestSolutionIterator::getSolution()
        {
            return m_currentSolution;
        }
    }
}