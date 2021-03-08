/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
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

#include "SolutionIterator.hpp"

#include "ResultReporter.hpp"

namespace Tensile
{
    namespace Client
    {
        std::shared_ptr<SolutionIterator> SolutionIterator::Default(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
            std::shared_ptr<Hardware>                                  hardware,
            po::variables_map const&                                   args)
        {
            bool bestSolution = args["best-solution"].as<bool>();

            if(bestSolution)
            {
                return std::make_shared<BestSolutionIterator>(library, hardware);
            }
            else
            {
                return std::make_shared<AllSolutionsIterator>(
                    library,
                    hardware,
                    args["solution-start-idx"].as<int>(),
                    args["num-solutions"].as<int>(),
                    AllSolutionsIterator::CreateCriteria(library, hardware, args),
                    args["run-criteria-verify"].as<bool>());
            }
        }

        SolutionIterator::SolutionIterator(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
            std::shared_ptr<Hardware>                                  hardware)
            : m_library(library)
            , m_hardware(hardware)
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
                    msg << std::endl;
                    m_reporter->log(LogLevel::Verbose, msg.str());
                }

                return false;
            }

            // Test if the persistent kernel is eligible for the current hw and solution
            m_problem.checkPersistentKernelEligibility(solution, *m_hardware);
            if(!(*solution.problemPredicate)(m_problem))
            {
                m_reporter->report(ResultKey::Validation, "DID_NOT_SATISFY_ASSERTS");
                if(m_reporter->logAtLevel(LogLevel::Verbose))
                {
                    std::ostringstream msg;
                    solution.problemPredicate->debugEval(m_problem, msg);
                    msg << std::endl;
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

        AllSolutionsIterator::RunCriteria AllSolutionsIterator::CreateCriteria(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
            std::shared_ptr<Hardware>                                  hardware,
            po::variables_map const&                                   args)
        {
            double granThresh = args["granularity-threshold"].as<double>();
            double memThresh  = args["mem-throughput-threshold"].as<double>();
            double minLDSUtil = args["min-lds-utilization"].as<double>();
            int    l2Speed    = args["l2-speed"].as<int>();
            int    aluRate    = args["alu-rate"].as<int>();

            PerformanceMetric perfMetric = args["performance-metric"].as<PerformanceMetric>();

            RunCriteria criteria;
            if(granThresh > 0.0)
            {
                criteria.push_back([=](ContractionProblem const&  problem,
                                       Hardware const&            hardware,
                                       ContractionSolution const& solution) {
                    auto   projPerf  = solution.projectedPerformance(problem, hardware);
                    double totalGran = projPerf.totalGranularity;

                    // For CUEfficiency benchmarking, low CU granularity is OK
                    if(perfMetric == PerformanceMetric::CUEfficiency)
                        totalGran /= projPerf.cuGranularity;

                    TestResult tr = (totalGran >= granThresh) ? TR::Run : TR::LowGranularity;
                    return FR{tr, totalGran, granThresh};
                });
            }
            if(memThresh > 0.0)
            {
                criteria.push_back([=](ContractionProblem const&  problem,
                                       Hardware const&            hardware,
                                       ContractionSolution const& solution) {
                    // TODO need: memory readBW and numChannels
                    // TODO get ALU from yaml file
                    size_t K   = problem.boundSize(0); // TODO - fix for multiple summations
                    size_t GSU = solution.sizeMapping.globalSplitU;
                    size_t LSU = solution.sizeMapping.workGroupSize.z;
                    size_t MT0 = solution.sizeMapping.macroTile.x;
                    size_t MT1 = solution.sizeMapping.macroTile.y;

                    size_t tileK = K / GSU / LSU;

                    size_t bpe = DataTypeInfo::Get(problem.a().dataType()).elementSize;

                    double bytesPerCU
                        = (MT0 * tileK * bpe) + (MT1 * tileK * bpe) + (MT0 * MT1 * bpe);
                    double cycles   = double(MT0 * MT1 * tileK * problem.flopsPerMac()) / aluRate;
                    double roofline = bytesPerCU / cycles; // bytes / CU / cycle

                    double l2SpeedPerCU   = l2Speed / perf.CUs; // bytes / CU / cycle
                    double memToCompRatio = l2SpeedPerCU / roofline;

                    TestResult tr
                        = (memToCompRatio >= memThresh) ? TR::Run : TR::LowMemoryThroughput;
                    return FR{tr, memToCompRatio, memThresh};
                });
            }
            // if(minLDSUtil > 0.0)
            // {
            //     criteria.push_back([minLDSUtil](ContractionProblem const&  problem,
            //                                     Hardware const&            hardware,
            //                                     ContractionSolution const& solution) {
            //         // TODO get actual LDS used by kernel
            //         double LDSUtil = 1.0;
            //         return LDSUtil >= minLDSUtil;
            //     });
            // }
            return criteria;
        }

        AllSolutionsIterator::AllSolutionsIterator(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
            std::shared_ptr<Hardware>                                  hardware,
            int                                                        firstSolutionIdx,
            int                                                        numSolutions,
            RunCriteria                                                runCriteria,
            bool                                                       criteriaVerify)
            : SolutionIterator(library, hardware)
            , m_numSolutionsSkipped(0)
            , m_runCriteria(runCriteria)
            , m_criteriaVerify(criteriaVerify)
        {
            m_firstSolutionIdx = firstSolutionIdx;

            if(m_firstSolutionIdx < 0)
                m_firstSolutionIdx = library->solutions.begin()->first;

            if(numSolutions < 0)
            {
                auto iter         = library->solutions.rbegin();
                m_lastSolutionIdx = iter->first;
            }
            else
            {
                m_lastSolutionIdx = m_firstSolutionIdx + numSolutions - 1;
            }

            m_currentSolutionIdx = m_firstSolutionIdx;
        }

        void AllSolutionsIterator::preProblem(ContractionProblem const& problem)
        {
            SolutionIterator::preProblem(problem);

            m_numSolutionsSkipped = 0;
            m_currentSolutionIdx  = m_firstSolutionIdx;
        }

        void AllSolutionsIterator::postProblem()
        {
            int numSolutions = m_lastSolutionIdx - m_firstSolutionIdx + 1;

            std::string val = std::to_string(numSolutions - m_numSolutionsSkipped) + "/"
                              + std::to_string(numSolutions);
            m_reporter->report(ResultKey::SolutionsRun, val);
        }

        void AllSolutionsIterator::preSolution(ContractionSolution const& solution)
        {
            {
                std::string idx  = "-1";
                auto        iter = solution.info.find("SolutionIndex");
                if(iter != solution.info.end())
                    idx = iter->second;
                m_reporter->report(ResultKey::SolutionLibraryIndex, idx);
            }
            m_reporter->report(ResultKey::SolutionIndex, m_currentSolutionIdx);
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

        bool AllSolutionsIterator::runCurrentSolution()
        {
            auto solution = getSolution();

            if(!checkSolution(*solution))
                return false;

            for(auto const& criterion : m_runCriteria)
            {
                FR fr = criterion(m_problem, *m_hardware, *solution);
                if(fr.value < fr.thresh)
                {
                    m_numSolutionsSkipped++;
                    if(m_criteriaVerify)
                    {
                        m_reporter->report(ResultKey::WouldSkip,
                                           TypeAbbrev(fr.reason) + ":" + std::to_string(fr.value));
                        return true;
                    }
                    else
                    {
                        m_reporter->report(ResultKey::Validation,
                                           "SKIPPED: " + ToString(fr.reason));
                        return false;
                    }
                }
            }
            return true;
        }

        std::string ToString(AllSolutionsIterator::TR tr)
        {
            switch(tr)
            {
            case AllSolutionsIterator::TR::Run:
                return "Run";
            case AllSolutionsIterator::TR::LowGranularity:
                return "LowGranularity";
            case AllSolutionsIterator::TR::LowMemoryThroughput:
                return "LowMemoryThroughput";
            default:;
            }
            return "Invalid";
        }

        std::string TypeAbbrev(AllSolutionsIterator::TR tr)
        {
            switch(tr)
            {
            case AllSolutionsIterator::TR::Run:
                return "run";
            case AllSolutionsIterator::TR::LowGranularity:
                return "grn";
            case AllSolutionsIterator::TR::LowMemoryThroughput:
                return "mem";
            default:;
            }
            return "Invalid";
        }

        std::ostream& operator<<(std::ostream& stream, const AllSolutionsIterator::TR& tr)
        {
            return stream << ToString(tr);
        }

        BestSolutionIterator::BestSolutionIterator(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
            std::shared_ptr<Hardware>                                  hardware)
            : SolutionIterator(library, hardware)
        {
        }

        void BestSolutionIterator::preProblem(ContractionProblem const& problem)
        {
            SolutionIterator::preProblem(problem);

            m_currentSolution     = m_library->findBestSolution(m_problem, *m_hardware);
            m_usedCurrentSolution = false;
        }

        void BestSolutionIterator::postProblem() {}

        void BestSolutionIterator::preSolution(ContractionSolution const& solution)
        {
            {
                std::string idx  = "-1";
                auto        iter = solution.info.find("SolutionIndex");
                if(iter != solution.info.end())
                    idx = iter->second;
                m_reporter->report(ResultKey::SolutionLibraryIndex, idx);
            }

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
    } // namespace Client
} // namespace Tensile
