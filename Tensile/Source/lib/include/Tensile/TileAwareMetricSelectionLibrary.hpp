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

#pragma once

#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/Properties.hpp>
#include <Tensile/Utils.hpp>

#include <Tensile/PropertyMatching.hpp>

namespace Tensile
{
    /**
 * \ingroup SolutionLibrary
 */
    struct TileAwareMetricSolutionTableEntry
    {
        std::vector<size_t> key;
        int                 value;
    };

    template <typename MySolution>
    struct TileAwareMetricModelTableEntry
    {
        int                 key;
        std::shared_ptr<MySolution>          solution;
        std::vector<double> problem;
    };

    //FitnessSelectionTableEntry
    /**
 * \ingroup SolutionLibrary
 *
 * Compares the tile sizes of each kernel, the dimensions of the problem,
 * and the number of compute units on the target GPU to select a kernel
 * that fits the best on the GPU with the lowest amount of waste
 * ("granularity loss").
 */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct TileAwareMetricSelectionLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        std::map<int, std::shared_ptr<MySolution>> solutions;
        std::map<std::vector<size_t>, int>         exactMap;
        std::vector<TileAwareMetricModelTableEntry<MySolution>>        modelProblems;

        static std::string Type()
        {
            return "TileAwareMetricSelection";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            std::string rv = this->type();

            return rv;
        }

        virtual std::shared_ptr<MySolution>
        findBestSolution(MyProblem const& problem, Hardware const& hardware) const override
        {
            const bool debug = Debug::Instance().printPropertyEvaluation();

            std::vector<size_t> key;
            size_t M = problem.freeSizeA(0);
            key.push_back(M);
            size_t N = problem.freeSizeB(0);
            key.push_back(N);
            size_t NumBatches = problem.batchSize(0);
            key.push_back(NumBatches);
            size_t K = problem.boundSize(0);
            key.push_back(K);

            auto exactMatch = exactMap.find(key);
            if(exactMatch != this->exactMap.end())
            {
                int index = exactMatch->second;

                auto rv = solutions.at(index);

                if(debug)
                {
                    std::cout << "Exact match: " << rv->description();
                    rv->problemPredicate->debugEval(problem, std::cout);
                    std::cout << std::endl;
                    rv->hardwarePredicate->debugEval(hardware, std::cout);
                    std::cout << std::endl;
                }

                if((*rv->problemPredicate)(problem) && (*rv->hardwarePredicate)(hardware))
                {
                    return rv;
                }
                else if(debug)
                {
                    std::cout << "Predicate failure" << std::endl;
                }
            }

            double                      bestDistance = std::numeric_limits<double>::max();
            std::shared_ptr<MySolution> bestSolution;

            /*
            for(auto const& row : solutions)
            {
                auto myPerformance
                    = row.second->projectedTAMetricPerformance(problem, hardware).speedGFlops;

                if(debug)
                {
                    std::cout << row.second->description() << ": " << myPerformance;
                }

                if(myPerformance > bestPerformance)
                {
                    if((*row.second->problemPredicate)(problem)
                       && (*row.second->hardwarePredicate)(hardware))
                    {
                        bestPerformance = myPerformance;
                        bestSolution    = row.second;

                        if(debug)
                            std::cout << " <-- Best so far";
                    }
                    else if(debug)
                    {
                        std::cout << " <-- Best, but predicate failure";
                    }

                    if(debug)
                    {
                        row.second->problemPredicate->debugEval(problem, std::cout);
                        std::cout << std::endl;
                        row.second->hardwarePredicate->debugEval(hardware, std::cout);
                        std::cout << std::endl;
                    }
                }
            }*/

            auto it = modelProblems.begin();

            //double M = 1.0, N = 1.0;
            //M = problem.freeSizeA(0);
            //N = problem.freeSizeB(0);
            //double NumBatches = 1;
            //double K = problem.boundSize(0); 


            while (it != modelProblems.end())
            {
                //int                 key;
                //std::shared_ptr<MySolution>          solution;
                //std::vector<double> problem;
                size_t model_M = (size_t)it->problem[0];
                size_t model_N = (size_t)it->problem[1];
                size_t model_batchSize = (size_t)it->problem[2];
                size_t model_K = (size_t)it->problem[3];

                ContractionSolution::TAMetricProblemScore ppReference = 
                  it->solution->computeProblemScore(
                    hardware, 
                    model_M, model_N, model_K, model_batchSize,
                    0, 0, 0, 0);

                ContractionSolution::TAMetricProblemScore pp = 
                  it->solution->computeProblemScore(
                    hardware, 
                    M, N, K, NumBatches,
                    0, 0, 0, 0);
                it++;

       
                double metric = std::numeric_limits<double>::max();
                
                if (ppReference.tile0Granularity > 0.0 && pp.tile0Granularity > 0.0)
                {
                  metric = abs(log(ppReference.tile0Granularity) - log(pp.tile0Granularity));
                }
                if (ppReference.tile0Granularity > 0.0 && pp.tile0Granularity > 0.0)
                {
                  if (metric < std::numeric_limits<double>::max())
                  {
                    metric += abs(log(ppReference.tile1Granularity) - log(pp.tile1Granularity));
                  }
                  else
                  {
                    metric = abs(log(ppReference.tile1Granularity) - log(pp.tile1Granularity));
                  }
                }  
                if (ppReference.suCuGranularity > 0.0 && pp.suCuGranularity > 0.0)
                {
                  if (metric < std::numeric_limits<double>::max())
                  {
                    metric += abs(log(ppReference.suCuGranularity) - log(pp.suCuGranularity));
                  }
                  else
                  {
                    metric = abs(log(ppReference.suCuGranularity) - log(pp.suCuGranularity));
                  }
                  //metric += abs(log(ppReference.cuGranularity) - log(pp.cuGranularity));
                }
                if (ppReference.suWaveGranularity > 0.0 && pp.suWaveGranularity > 0.0)
                {
                  if (metric < std::numeric_limits<double>::max())
                  {
                    metric += abs(log(ppReference.suWaveGranularity) - log(pp.suWaveGranularity));
                  }
                  else
                  {
                    metric = abs(log(ppReference.suWaveGranularity) - log(pp.suWaveGranularity));
                  }
                  //metric += abs(log(ppReference.waveGranularity) - log(pp.waveGranularity));
                }
                
                if (metric < bestDistance)
                {
                    bestDistance = metric;
                    bestSolution = it->solution;
                }

/*
                struct TAMetricProblemScore
        {
            double numTiles0  = 0.0; //! number of tiles in 0 dimension
            double numTiles1  = 0.0; //! number of tiles in 1 dimension
            double totalTiles = 0.0;
            double tilesPerCu = 0.0;

            //! Granularity is measured 0..1 with 1.0 meaning no granularity loss
            double tile0Granularity = 0.0; // loss due to tile0
            double tile1Granularity = 0.0;
            double cuGranularity    = 0.0;
            double waveGranularity  = 0.0;
            double totalGranularity = 0.0;
            double suTilesPerCu = 0.0;
            double suCuGranularity = 0.0;
            double waves = 0.0;
            double suWavesPerSimdx2 = 0.0;
            double suWaveGranularity = 0.0;

            double speedGFlops = 0.0; //! final gflops projection
            int    CUs         = 0;

            StaticTAMetricPerformanceModel staticModel;
        };*/

            }
            

            return bestSolution;
        }

        virtual SolutionSet<MySolution> findAllSolutions(MyProblem const& problem,
                                                         Hardware const&  hardware) const override
        {
            bool debug = Debug::Instance().printPropertyEvaluation();

            SolutionSet<MySolution> rv;

            for(auto const& row : solutions)
            {
                if(debug)
                {
                    std::cout << row.second->description() << ": ";
                }

                if((*row.second->problemPredicate)(problem)
                   && (*row.second->hardwarePredicate)(hardware))
                {
                    rv.insert(row.second);

                    if(debug)
                        std::cout << " Works";
                }
                else if(debug)
                {
                    if(debug)
                        std::cout << " Predicate failed";
                }

                if(debug)
                {
                    row.second->problemPredicate->debugEval(problem, std::cout);
                    std::cout << std::endl;
                    row.second->hardwarePredicate->debugEval(hardware, std::cout);
                    std::cout << std::endl;
                }
            }

            return rv;
        }
    };
} // namespace Tensile
