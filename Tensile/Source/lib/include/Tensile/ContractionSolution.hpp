/**
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <memory>
#include <cstddef>
#include <string>
#include <vector>

#include <Tensile/Tensile.hpp>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/Predicates.hpp>

namespace Tensile
{

    /**
     * Represents a single kernel or set of kernels that can perform a single
     * tensor contraction.
     * 
     * Can generate `KernelInvocation` objects to solve a particular problem
     * given a set of `ContractionInputs`.
     */
    class ContractionSolution: public Solution
    {
    public:
        using Problem = ContractionProblem;
        using Inputs  = ContractionInputs;

        static std::string Type() { return "Contraction"; }
        virtual std::string type() const { return Type(); }

        virtual std::string KernelName() const { return kernelName; }
        virtual std::string name() const { return kernelName; }
        virtual std::string description() const { return kernelName; }

        bool isSourceKernel() const;

        /**
         * Calculate the projected performance based on granularity loss.
         */
        virtual double projectedPerformance(Problem const& problem, Hardware const& hardware) const;

        /**
         * Generate a set of kernel calls to solve a particular problem.
         */
        virtual std::vector<KernelInvocation> solve(Problem  const& problem,
                                                    Inputs   const& inputs,
                                                    Hardware const& hardware) const;

        template <typename TypedInputs>
        std::vector<KernelInvocation> solveTyped(Problem     const& problem,
                                                 TypedInputs const& inputs,
                                                 Hardware    const& hardware) const;

        template <typename TypedInputs>
        KernelInvocation generateSingleCall(Problem     const& problem,
                                            TypedInputs const& inputs,
                                            Hardware    const& hardware) const;

        template <typename TypedInputs>
        KernelInvocation generateBetaOnlyCall(Problem     const& problem,
                                              TypedInputs const& inputs,
                                              Hardware    const& hardware) const;

        template <typename TypedInputs>
        std::string betaOnlyKernelName(Problem     const& problem,
                                       TypedInputs const& inputs,
                                       Hardware    const& hardware) const;

        struct SizeMapping
        {
            dim3 workGroupSize;
            dim3 threadTile;
            dim3 macroTile;

            size_t staggerU;
            size_t depthU;
            size_t globalSplitU;
            size_t staggerStrideShift;
            int workGroupMapping;

            size_t packBatchDims;
            int    magicDivAlg=1;
            size_t persistentKernel;

            bool sourceKernel;
        };

        struct ProblemType
        {
            std::string operationIdentifier;
            DataType aType = DataType::Float;
            DataType bType = DataType::Float;
            DataType cType = DataType::Float;
            DataType dType = DataType::Float;
            bool highPrecisionAccumulate = false;
            bool useBeta = true;
            bool useInitialStridesAB = false;
            bool useInitialStridesCD = false;
        };

        int index;
        std::string kernelName;
        bool debugKernel = false;

        std::shared_ptr<Predicates::Predicate<Problem>>  problemPredicate =
            std::make_shared<Predicates::True<Problem>>();
        std::shared_ptr<Predicates::Predicate<Hardware>> hardwarePredicate =
            std::make_shared<Predicates::True<Hardware>>();

        SizeMapping sizeMapping;

        ProblemType problemType;

        /// Debugging purposes.  Shouldn't contain any vital information that isn't somewhere else.
        std::map<std::string, std::string> info;
        std::map<int, double> ideals;

        int32_t staggerUIter(Problem  const& problem,
                             Inputs   const& inputs,
                             Hardware const& hardware) const;

        uint32_t magicNumberAlg1(uint32_t x, uint32_t *magicShift) const;
        uint32_t magicNumberAlg2(uint32_t x, uint32_t *magicShift) const;
        uint32_t magicNumber(int magicDivAlg, uint32_t x, uint32_t *magicShift) const;
        uint32_t smallMagicNumber(uint32_t x) const;
    };

}

