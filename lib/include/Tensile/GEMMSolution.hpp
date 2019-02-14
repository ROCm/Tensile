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
#include <string>
#include <vector>

#include "GEMMProblem.hpp"

namespace Tensile
{

    class GEMMSolution: public Solution
    {
    public:
        virtual std::string KernelName() const { return kernelName; }
        virtual std::string name() const { return kernelName; }
        virtual std::string description() const { return kernelName; }

        bool solves(GEMMProblem const& problem,
                    GEMMInputs  const& inputs,
                    Hardware    const& hardware) const;

        std::vector<KernelInvocation> solve(GEMMProblem const& problem,
                                            GEMMInputs  const& inputs,
                                            Hardware    const& hardware) const;

        KernelInvocation generateSingleCall(GEMMProblem const& problem,
                                            GEMMInputs  const& inputs,
                                            Hardware     const& hardware) const;

        struct ProblemCompatibility
        {
            std::vector<size_t> aDimensionOrder;
            std::vector<size_t> bDimensionOrder;
            std::vector<size_t> cDimensionOrder;
            std::vector<size_t> dDimensionOrder;

            bool useBeta;
            bool cdEqual;
            bool cdAllocationsEqual;
        };

        struct Options
        {
            enum class EdgeType: int
            {
                ShiftPtr,
                Branch,
                Count
            };

            enum class KernelLanguage: int
            {
                Source,
                Assembly,
                Count
            };

            enum class WorkGroupMappingType: int
            {
                B,
                Z,
                Count
            };

            int aggressivePerfMode;
            int assertFree0ElementMultiple;
            int assertFree1ElementMultiple;
            int assertMinApproxSize;
            int assertSummationElementMultiple;
            bool assignedDerivedParameters;
            bool assignedProblemIndependentDerivedParameters;
            bool bufferLoad;
            bool bufferStore;
            int checkDimOverflow;
            bool checkTensorDimAsserts;
            int depthU;
            bool directToLds;
            bool directToLdsA;
            bool directToLdsB;
            int disableKernelPieces;
            EdgeType edgeType;
            bool expandPointerSwap;
            int fractionalLoad;
            int globalLoadVectorWidthA;
            int globalLoadVectorWidthB;
            bool globalRead2A;
            bool globalRead2B;
            bool globalReadCoalesceGroupA;
            bool globalReadCoalesceGroupB;
            bool globalReadCoalesceVectorA;
            bool globalReadCoalesceVectorB;
            int globalReadVectorWidth;
            int globalSplitU;
            bool globalSplitUSummationAssignmentRoundRobin;
            bool globalSplitUWorkGroupMappingRoundRobin;
            int globalWriteVectorWidth;
            bool guaranteeNoPartialA;
            bool guaranteeNoPartialB;
            int innerUnroll;
            KernelLanguage kernelLanguage;
            int lsca;
            int lscb;
            int lspa;
            int lspb;
            int lvca;
            int lvcb;
            int lvpa;
            int lvpb;
            int ldsNumElements;
            int ldsNumElementsAlignedA;
            int ldsNumElementsAlignedB;
            int ldsOffsetA;
            int ldsOffsetA_Blk;
            int ldsOffsetB;
            int ldsOffsetB_Blk;
            int ldsPadA;
            int ldsPadB;
            int localDotLayout;
            bool localRead2A;
            bool localRead2B;
            int localSplitU;
            bool localWrite2A;
            bool localWrite2B;
            bool localWriteUseSgprA;
            bool localWriteUseSgprB;
            bool loopDoWhile;
            bool loopTail;
            int loopUnroll;
            int macroTile0;
            int macroTile1;
            int macroTileA;
            int macroTileB;
            int macroTileShapeMax;
            int macroTileShapeMin;
            int maxOccupancy;
            int minGlobalWriteVectorWidth;
            int nonTemporalA;
            int nonTemporalB;
            int nonTemporalC;
            int numElementsPerThread;
            int numGlobalWriteVectorsPerThread;
            int numLoadsA;
            int numLoadsB;
            int numLoadsCoalescedA;
            int numLoadsCoalescedB;
            int numLoadsPerpendicularA;
            int numLoadsPerpendicularB;
            int numThreads;
            int packBatchDims;
            int packFreeDims;
            int packGranularity;
            std::vector<char> PackedC0Indices;
            std::vector<char> PackedC1Indices;
            int performanceSyncLocation;
            int performanceWaitCount;
            int performanceWaitLocation;
            int persistentKernel;
            bool prefetchGlobalRead;
            bool prefetchLocalRead;
            int solutionIndex;
            std::string solutionNameMin;
            int subGroup0;
            int subGroup1;
            int subGroupA;
            int subGroupB;
            bool suppressNoLoadLoop;
            std::vector<int> threadTile;
            int threadTile0;
            int threadTile1;
            int threadTileA;
            int threadTileB;
            bool unrollMemFence;
            bool useSgprForGRO;
            bool valid;
            int vectorAtomicWidth;
            bool vectorStore;
            int vectorWidth;
            std::vector<int> workGroup;
            int workGroupMapping;
            WorkGroupMappingType workGroupMappingType;
        };

        Options options;

        std::string kernelName;

        dim3 workGroupSize;
        dim3 macroTile;
        bool debugKernel;

        int32_t staggerUIter(GEMMProblem const& problem,
                             GEMMInputs  const& inputs,
                             Hardware    const& hardware) const;

        uint32_t magicNumber(uint32_t x) const;
    };

}

