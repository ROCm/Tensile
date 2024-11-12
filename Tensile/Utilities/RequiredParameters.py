_requiredParametersMin = {
    "ProblemType": False,
    "LoopDoWhile": True,
    "LoopTail": False,
    "EdgeType": True,
    "InnerUnroll": True,
    "LocalDotLayout": True,
    "AggressivePerfMode": True,
    "PreloadKernelArguments": True,
    "KernelLanguage": True,
    "LdsPadA": True,
    "LdsPadB": True,
    "LdsBlockSizePerPadA": True,
    "LdsBlockSizePerPadB": True,
    "TransposeLDS": True,
    "UnrollMajorLDSA": True,
    "UnrollMajorLDSB": True,
    "ExtraMiLatencyLeft": True,
    "ExtraLatencyForLR": True,
    "VgprForLocalReadPacking": True,
    "ClusterLocalRead": True,
    "MaxOccupancy": True,
    "VectorWidth": True,
    "VectorWidthB": True,
    "VectorStore": True,
    "StoreVectorWidth": True,
    "GlobalLoadVectorWidthA": True,
    "GlobalLoadVectorWidthB": True,
    "GlobalReadVectorWidth": True,
    "LocalReadVectorWidth": True,
    "GlobalReadCoalesceVectorA": False,
    "GlobalReadCoalesceVectorB": False,
    "WaveSeparateGlobalReadA": True,
    "WaveSeparateGlobalReadB": True,
    "GlobalReadCoalesceGroupA": True,
    "GlobalReadCoalesceGroupB": True,
    "PrefetchGlobalRead": True,
    "PrefetchLocalRead": True,
    "UnrollMemFence": False,
    "GlobalRead2A": False,
    "GlobalRead2B": False,
    "LocalWrite2A": False,
    "LocalWrite2B": False,
    "LocalRead2A": False,
    "LocalRead2B": False,
    "SuppressNoLoadLoop": True,
    "ExpandPointerSwap": True,
    "ScheduleGlobalRead": False,
    "ScheduleLocalWrite": True,
    "ScheduleIterAlg": True,
    "OptPreLoopVmcnt": True,
    "LdcEqualsLdd": False,
    "GlobalReadPerMfma": True,
    "LocalWritePerMfma": True,
    "InterleaveAlpha": False,
    "OptNoLoadLoop": True,
    "PrefetchAcrossPersistent": True,
    "PrefetchAcrossPersistentMode": False,
    "BufferLoad": True,
    "BufferStore": True,
    "DirectToVgprA": True,
    "DirectToVgprB": True,
    "DirectToLdsA": True,
    "DirectToLdsB": True,
    "UseSgprForGRO": True,
    "UseInstOffsetForGRO": False,
    "AssertSummationElementMultiple": True,
    "AssertFree0ElementMultiple": True,
    "AssertFree1ElementMultiple": True,
    "AssertMinApproxSize": True,
    "AssertStrideAEqual": True,
    "AssertStrideBEqual": False,
    "AssertStrideCEqual": True,
    "AssertStrideDEqual": False,
    "AssertSizeEqual": True,
    "AssertSizeGreaterThan": True,
    "AssertSizeMultiple": True,
    "AssertSizeLessThan": True,
    "AssertAIGreaterThanEqual": False,
    "AssertAILessThanEqual": False,
    "AssertAlphaValue": False,
    "AssertBetaValue": True,
    "AssertCEqualsD": True,
    "CheckTensorDimAsserts": False,
    "CheckDimOverflow": False,
    "StaggerU": True,
    "StaggerUStride": True,
    "StaggerUMapping": True,
    "MagicDivAlg": True,
    "GlobalSplitU": True,
    "GlobalSplitUAlgorithm": True,
    "GlobalSplitUSummationAssignmentRoundRobin": False,
    "GlobalSplitUWorkGroupMappingRoundRobin": False,
    "GlobalSplitUAtomicAdd": False,
    "MacroTileShapeMin": False,
    "MacroTileShapeMax": False,
    "StreamK": False,
    "StreamKAtomic": False,
    "StreamKXCCMapping": False,
    "DebugStreamK": False,
    "PersistentKernel": True,
    "PersistentKernelAlongBatch": False,
    "PackBatchDims": False,
    "PackFreeDims": False,
    "PackSummationDims": False,
    "UnrollIncIsDepthU": False,
    "PackGranularity": False,
    "FractionalLoad": True,
    "Use64bShadowLimit": True,
    "VectorAtomicWidth": True,
    "NumLoadsCoalescedA": True,
    "NumLoadsCoalescedB": True,
    "WorkGroup": True,
    "WorkGroupMappingType": False,
    "WorkGroupMapping": True,
    "ThreadTile": True,
    "MACInstruction": True,
    "WavefrontSize": True,
    "MemoryModifierFormat": True,
    "MatrixInstruction": False,
    "DisableVgprOverlapping": True,
    "1LDSBuffer": True,
    "DisableAtomicFail": False,
    "DisableKernelPieces": False,
    "DepthU": False,
    "DepthULdsDivisor": False,
    "PerformanceSyncLocation": False,
    "PerformanceWaitLocation": False,
    "PerformanceWaitCount": False,
    "NonTemporalD": True,
    "NonTemporalC": True,
    "NonTemporalA": True,
    "NonTemporalB": True,
    "ForceStoreSC1": True,
    "ReplacementKernel": False,
    "CustomKernelName": False,
    "NoReject": False,
    "MinVgprNumber": False,
    "MaxVgprNumber": False,
    "StoreRemapVectorWidth": True,
    "SourceSwap": True,
    "AtomicAddC": True,
    "StorePriorityOpt": True,
    "NumElementsPerBatchStore": True,
    "StoreSyncOpt": True,
    "GroupLoadStore": True,
    "MIArchVgpr": True,
    "StoreCInUnroll": False,
    "StoreCInUnrollInterval": True,
    "StoreCInUnrollExact": False,
    "StoreCInUnrollPostLoop": False,
    "Fp16AltImpl": False,
    "Fp16AltImplRound": False,
    "ThreadSeparateGlobalReadA": True,
    "ThreadSeparateGlobalReadB": True,
    "MinKForGSU": True,
    "ISA": True,
    "CodeObjectVersion": False,
    "AssignedDerivedParameters": False,
    "AssignedProblemIndependentDerivedParameters": False,
    "DirectToLds": False,
    "EnableMatrixInstruction": False,
    "GlobalWriteVectorWidth": False,
    "GuaranteeNoPartialA": False,
    "GuaranteeNoPartialB": False,
    "LSCA": False,
    "LSCB": False,
    "LSPA": False,
    "LSPB": False,
    "LVCA": False,
    "LVCB": False,
    "LVPA": False,
    "LVPB": False,
    "LdsBlockSizePerPad": False,
    "LdsNumElements": False,
    "LdsNumElementsAlignedA": False,
    "LdsNumElementsAlignedB": False,
    "LdsOffsetA": False,
    "LdsOffsetA_Blk": False,
    "LdsOffsetB": False,
    "LdsOffsetB_Blk": False,
    "LocalSplitU": False,
    "LocalWriteUseSgprA": False,
    "LocalWriteUseSgprB": False,
    "LoopIters": False,
    "LoopUnroll": False,
    "MacroTile0": False,
    "MacroTile1": False,
    "MacroTileA": False,
    "MacroTileB": False,
    "MIBlock": False,
    "MIInputPerThread": False,
    "MIOutputVectorWidth": False,
    "MIRegPerOut": False,
    "MIWaveGroup": False,
    "MIWaveTile": False,
    "MIWaveTileA": False,
    "MIWaveTileB": False,
    "NumElementsPerThread": False,
    "NumGlobalWriteVectorsPerThread": False,
    "NumLoadsA": False,
    "NumLoadsB": False,
    "NumLoadsPerpendicularA": False,
    "NumLoadsPerpendicularB": False,
    "NumThreads": False,
    "PackedC0IdxChars": False,
    "PackedC0IndicesX": False,
    "PackedC1IdxChars": False,
    "PackedC1IndicesX": False,
    "SolutionIndex": False,
    "SolutionNameMin": False,
    "SubGroup0": False,
    "SubGroup1": False,
    "SubGroupA": False,
    "SubGroupB": False,
    "ThreadTile0": False,
    "ThreadTile1": False,
    "ThreadTileA": False,
    "ThreadTileB": False,
    "Valid": False,
    "_GlobalAccumulation": False,
    "_UseSgprForGRO": False,
    "_VectorStore": False,
    "_WorkspaceSizePerElemC": False,
    "_staggerStrideShift": False,
    "fractionalPerpOverhangA": False,
    "fractionalPerpOverhangB": False,
    "EnableF32XdlMathOp": False,
    "EnableMatrixInstructionStore": False,
    "DelayRemainingArguments": False,
    "VectorWidthA": False,
    "_DepthULds": False,
    "NoLdsWriteCode": False,
    "LdsInitCVgprs": False,
    "codeObjectFile": False,
    "allowLRVWBforTLUandMI": False,
    "Kernel": True,
    "MatrixInstM": False,
    "MatrixInstN": False,
    "MatrixInstK": False,
    "MatrixInstB": False,
    "MatrixInstBM": False,
    "MatrixInstBN": False,
    "StochasticRounding": False,
}


def getRequiredParametersMin():
    return _requiredParametersMin
