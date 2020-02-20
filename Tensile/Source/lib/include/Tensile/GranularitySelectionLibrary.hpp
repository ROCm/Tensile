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

#include <vector>
#include <set>

#include <Tensile/Properties.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/Utils.hpp>

#include <Tensile/PropertyMatching.hpp>

namespace Tensile
{
    /**
     * \ingroup SolutionLibrary
     */ 
    struct ExactSelectionTableEntry
    {
        std::vector<size_t> key;
        int value;
    };

    /**
     * \ingroup SolutionLibrary
     * 
     * Compares the tile sizes of each kernel, the dimensions of the problem,
     * and the number of compute units on the target GPU to select a kernel
     * that fits the best on the GPU with the lowest amount of waste
     * (“granularity loss”).
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct GranularitySelectionLibrary: public SolutionLibrary<MyProblem, MySolution>
    {
        std::map<int,std::shared_ptr<MySolution>> solutions;
        std::map<std::vector<size_t>,int> exactMap;

        static std::string Type() { return "GranularitySelection"; }
        virtual std::string type() const override { return Type(); }
        virtual std::string description() const override
        {
            std::string rv = this->type();

            return rv;
        }
        
        virtual std::shared_ptr<MySolution>
            findBestSolution(MyProblem const& problem,
                             Hardware  const& hardware) const override
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
            
            if (this->exactMap.find(key) != this->exactMap.end())
            {
                int index = this->exactMap.at(key);
                return solutions.at(index);
            }

            this->description();

            double bestPerformance = 0.0;
            auto iter = solutions.begin();
            if(iter == solutions.end())
                return std::shared_ptr<MySolution>();

            std::shared_ptr<MySolution> bestSolution = iter->second;
            if (bestSolution)
                bestPerformance = bestSolution->projectedPerformance(problem, hardware).speedGFlops;

            iter++;

            if(debug)
            {
                std::cout << "best performance: " << bestPerformance << std::endl;
            }

            while(iter != solutions.end())
            {
                auto mySolution = iter->second;
                double myPerformance = mySolution->projectedPerformance(problem, hardware).speedGFlops;

                if(mySolution)
                {
                    auto myPerformance = mySolution->projectedPerformance(problem, hardware).speedGFlops;
                    if(myPerformance > bestPerformance)
                    {
                        bestPerformance = myPerformance;
                        bestSolution = mySolution;
                    }

                    if(debug)
                    {
                        std::cout << ": " << myPerformance;
                        if(myPerformance < bestPerformance)
                        {
                            std::cout << " <-- Best so far";
                        } 
                    }
                }
                iter++;
            }
            return bestSolution;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const& problem,
                             Hardware  const& hardware) const override
        {
            bool debug = Debug::Instance().printPropertyEvaluation();

            SolutionSet<MySolution> rv;

            auto iter = solutions.begin();
           
            while(iter != solutions.end())
            {
                rv.insert(iter->second);
                if(debug)
                    std::cout << iter->second->description() << std::endl;
                iter++;
            }
            if(debug)
                std::cout << std::endl;

            return rv;
        }
    };
}

