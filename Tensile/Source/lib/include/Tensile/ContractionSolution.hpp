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

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Tensile/Tensile.hpp>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/Predicates.hpp>

namespace Tensile
{
    struct PerfModel
    {
        double clock = std::numeric_limits<double>::quiet_NaN(); 
        double memClock = std::numeric_limits<double>::quiet_NaN(); 
        double peakGFlops = std::numeric_limits<double>::quiet_NaN(); 
        double efficiency = std::numeric_limits<double>::quiet_NaN(); 
        double memBandwidthMBps= std::numeric_limits<double>::quiet_NaN(); 
        double readMul= std::numeric_limits<double>::quiet_NaN();
        double readEff=0.0;
        double l2ReadHitRate=0.0;
        double l2WriteHitRate=0.0;
        double CUs=0.0;
    } perf;
    
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

        //! Estimates based on problem size, solution tile, and  machine hardware charz:
        struct StaticPerformanceModel
        {
          size_t memReadBytesA=0.0; //! Estimated memory reads A
          size_t memReadBytesB=0.0; //! Estimated memory reads B
          size_t memReadBytesC=0.0; //! Estimated memory reads C
          size_t memWriteBytesD=0.0; //! Estimated memory writes D
          size_t memReadBytes=0.0;
          size_t memGlobalReads=0;
          size_t memGlobalWrites=0;

          //! Times in US
          double aluUs=0.0; //! Estimated alu cycles
          double memReadUs=0.0; //! Estimated memory read cycles
          double memWriteUs=0.0; //! Estimated memory write cycles
        
         };

        struct ProjectedPerformance
        {
          double numTiles0=0.0; //! number of tiles in 0 dimension
          double numTiles1=0.0; //! number of tiles in 1 dimension
          double tilesPerCu=0.0;

          //! Granularity is measured 0..1 with 1.0 meaning no granularity loss
          double tile0Granularity=0.0; // loss due to tile0
          double tile1Granularity=0.0;
          double cuGranularity=0.0;
          double waveGranularity=0.0;
          double totalGranularity=0.0;

          double speedGFlops=0.0; //! final gflops projection
          double efficiency=0.0;
          double CUs=0.0;
          
          StaticPerformanceModel staticModel;
        };
        
        StaticPerformanceModel staticPerformanceModel
          (double M, double N, double K, double NumBatches,  double MT0, double MT1, 
           double NumCUs, double totalGranularity, int globalSplitU) const;

        /**
         * Calculate the projected performance based on granularity loss.
         */
        ProjectedPerformance projectedPerformance(Problem const& problem, Hardware const& hardware) const;

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
            int    packSummationDims=0;
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

    std::ostream & operator<<(std::ostream & stream, ContractionSolution::StaticPerformanceModel const& spm);
    std::ostream & operator<<(std::ostream & stream, ContractionSolution::ProjectedPerformance const& spm);
}

