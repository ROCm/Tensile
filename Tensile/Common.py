################################################################################
# Copyright (C) 2016-2019 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from . import __version__
from collections import OrderedDict
from copy import deepcopy
from subprocess import Popen, PIPE

import itertools
import math
import os.path
import subprocess
import sys
import time


startTime = time.time()

# print level
# 0 - user wants no printing
# 1 - user wants limited prints
# 2 - user wants full prints

################################################################################
# Global Parameters
################################################################################
globalParameters = OrderedDict()

########################################
# common
########################################
globalParameters["MinimumRequiredVersion"] = "0.0.0"  # which version of tensile is required to handle all the features required by this configuration file
globalParameters["PrintLevel"] = 1                # how much info to print. 0=none, 1=standard, 2=verbose
# benchmarking
globalParameters["KernelTime"] = False            # T=use device timers, F=use host timers
globalParameters["PreciseKernelTime"] = True     # T=On hip, use the timestamps for kernel start and stop rather than separate events.  Can provide more accurate kernel timing.  For GlobalSplitU kernels, recommend disabling this to provide consistent
# timing between GSU / non-GSU kernels
globalParameters["CodeFromFiles"] = True          # if False byte arrays will be generated during Benchmarking phase as before
globalParameters["PinClocks"] = False             # T=pin gpu clocks and fan, F=don't
globalParameters["NumBenchmarks"] = 1             # how many benchmark data points to collect per problem/solution
globalParameters["SyncsPerBenchmark"] = 1         # how iterations of the stream synchronization for-loop to do per benchmark data point
globalParameters["EnqueuesPerSync"] = 1           # how many solution enqueues to perform per synchronization
globalParameters["SleepPercent"] = 300            # how long to sleep after every data point: 25 means 25% of solution time. Sleeping lets gpu cool down more.
# validation
globalParameters["NumElementsToValidate"] = 128   # number of elements to validate, 128 will be evenly spaced out (with prime number stride) across C tensor
globalParameters["ValidationMaxToPrint"] = 4      # maximum number of mismatches to print
globalParameters["ValidationPrintValids"] = False # print matches too
# steps
globalParameters["ForceRedoBenchmarkProblems"] = True # if False and benchmarking already complete, then benchmarking will be skipped when tensile is re-run
globalParameters["ForceRedoLibraryLogic"] = True      # if False and library logic already analyzed, then library logic will be skipped when tensile is re-run
globalParameters["ForceRedoLibraryClient"] = True     # if False and library client already built, then building library client will be skipped when tensile is re-run

# Compare CPU reference convolution model vs golden tensor contracton model
# Useful to test if conversion from tensor contraction is working as expected
# In this mode, the filter,stride,dilation are specified in the problem type.
# If the problem type uses constant Filter,Stride,Dilation,Pad* (ie these are not 'N'), then the
# specified constant MUST match the dimension in the problem or the tensile runtime will assert.
# The batch size, spatial dims, Cin, and Cout are always read from the problem description.
globalParameters["ConvolutionVsContraction"] = False
globalParameters["ShowProgressBar"] = True     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["SolutionSelectionAlg"] = 1          # algorithm to detetermine which solutions to keep. 0=removeLeastImportantSolutions, 1=keepWinnerSolutions (faster)
globalParameters["ExpandRanges"] = True          # expand ranges into exact configs before writing logic file.  False ignores ranges.
globalParameters["ExitAfterKernelGen"] = False     # Exit after generating kernels
globalParameters["ShowProgressBar"] = True     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["WavefrontWidth"] = 64     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["ExitOnFails"] = 1     # Exit if failures detected.
globalParameters["CpuThreads"] = -1  # How many CPU threads to use for kernel generation.  0=no threading, -1 == nproc, N=min(nproc,N).  TODO - 0 sometimes fails with a kernel name error?  0 does not check error codes correctly
# FROM MERGE
#globalParameters["CpuThreads"] = -4         # How many CPU threads to use for kernel generation.  0=no threading, <0 == nproc*abs(CpuThreads), N=min(nproc,N)

########################################
# optimization knob controls
########################################

globalParameters["UnrollLoopEfficiencyEnable"] = False   # if True split(S) MAC&LDS in each unroll iteration into n smaller groups..

########################################
# less common
########################################
globalParameters["CMakeBuildType"] = "Release"            # whether benchmark clients and library client should be release or debug
globalParameters["PrintSolutionRejectionReason"] = False  # when a solution is marked as invalid, print why

# how to initialize tensor data
# serial-in-u will use a sequence that increments in the K dimension
# This is a predictable patterns that can be checked as the kernel runs to detect
# when the wrong data is being used.
# trig_float initializes with the sin function to have non-zero values in the mantissa
# and exponent. It cannot be used for int8 or int32. Need to use tensileAlmostEqual
# not tensileEqual for checking the result.
globalParameters["DataInitTypeAB"] = 3            # 0=0, 1=1, 2=serial, 3=rand, 4=NaN, 5=serial-in-u, 6=trig_float.  Can be overridden by the DataInitTypeA or DataInitTypeB.  Eventually DataInitTypeAB will be retired.
globalParameters["DataInitTypeA"] = -1            # 0=0, 1=1, 2=serial, 3=rand, 4=NaN, 5=serial-in-u, 6=trig_float.  -1 uses value from DataInitTypeAB
globalParameters["DataInitTypeB"] = -1            # 0=0, 1=1, 2=serial, 3=rand, 4=NaN, 5=serial-in-u, 6=trig_float.  -1 uses value from DataInitTypeAB
globalParameters["DataInitTypeC"]  = 3            # 0=0, 1=1, 2=serial, 3=rand, 4=Na, 5=serial-in-uN, 6=trig_float.
globalParameters["DataInitTypeD"]  = 0            # 0=0, 1=1, 2=serial, 3=rand, 4=Na, 5=serial-in-uN, 6=trig_float.
globalParameters["DataInitTypeAlpha"] = 2         # 0=0, 1=1, 2=2, 3=rand, 4=NaN
globalParameters["DataInitTypeBeta"] = 2          # 0=0, 1=1, 2=2, 3=rand, 4=NaN
globalParameters["CEqualD"] = True               # Set to true if testing for the case where the pointer to C is the same as D.
# build parameters
globalParameters["CMakeCXXFlags"] = ""            # pass flags to cmake
globalParameters["CMakeCFlags"] = ""              # pass flags to cmake
globalParameters["DebugKernel"] = False           # assembly only, kernel gets buffer for debug "printing"; kernel writes data to memory, gets coppied to host and printed
globalParameters["LibraryPrintDebug"] = False     # solutions will print enqueue info when enqueueing a kernel

# Tensor printing controls:
globalParameters["PrintConvolutionUsage"] = 0      # Print Convolution usage info
globalParameters["PrintTensorA"] = 0          # Print TensorA after initialization
globalParameters["PrintTensorB"] = 0          # Print TensorB after initialization
globalParameters["PrintTensorC"] = 0          # Print TensorC.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintTensorD"] = 0          # Print TensorD.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintWinnersOnly"] = False      # Only print the solutions which become the fastest

# PrintMaxCols applies to dimensions where multiple cols are printed per line.
# PrintMaxRows applies to dimensions where one row is printed per line
# If PrintMax* is greater than the dimension, the middle elements will be repaced with "..."


# device selection
globalParameters["Platform"] = 0                  # select opencl platform
globalParameters["Device"] = 0                    # select hip device or opencl device within platform

# shouldn't need to change
globalParameters["DeviceLDS"] = 65536             # LDS bytes per CU, for computing occupancy
globalParameters["MaxLDS"] = 65536                # max LDS a kernel should attempt to use
globalParameters["MaxDepthU"] = 256               # max DepthU value to allow
globalParameters["ShortNames"] = False            # on windows kernel names can get too long; =True will convert solution/kernel names to serial ids
globalParameters["MergeFiles"] = True             # F=store every solution and kernel in separate file; T=store all solutions in single file
globalParameters["SupportedISA"] = [(8,0,3), (9,0,0), (9,0,6), (9,0,8)]             # assembly kernels writer supports these architectures
globalParameters["ClientBuildPath"] = "0_Build"                   # subdirectory for host code build directory.
globalParameters["NewClient"] = 1                                 # 1=Run old+new client, 2=run new client only (All In)
globalParameters["BenchmarkProblemsPath"] = "1_BenchmarkProblems" # subdirectory for benchmarking phases
globalParameters["BenchmarkDataPath"] = "2_BenchmarkData"         # subdirectory for storing final benchmarking data
globalParameters["LibraryLogicPath"] = "3_LibraryLogic"           # subdirectory for library logic produced by analysis
globalParameters["LibraryClientPath"] = "4_LibraryClient"         # subdirectory for building example library client
globalParameters["BenchmarkClientVersion"] = "Both"               # Old, New, Both

# internal, i.e., gets set during startup
globalParameters["CurrentISA"] = (0,0,0)
globalParameters["ROCmAgentEnumeratorPath"] = None      # /opt/rocm/bin/rocm_agent_enumerator
globalParameters["ROCmSMIPath"] = None                  # /opt/rocm/bin/rocm-smi
globalParameters["AssemblerPath"] = None                # /opt/rocm/bin/hcc
globalParameters["WorkingPath"] = os.getcwd()           # path where tensile called from
globalParameters["IndexChars"] =  "IJKLMNOPQRSTUVWXYZ"  # which characters to use for C[ij]=Sum[k] A[ik]*B[jk]
globalParameters["ScriptPath"] = os.path.dirname(os.path.realpath(__file__))            # path to Tensile/Tensile.py
globalParameters["SourcePath"] = os.path.join(globalParameters["ScriptPath"], "Source") # path to Tensile/Source/
globalParameters["HccVersion"] = "0,0,0"

# default runtime is selected based on operating system, user can override
if os.name == "nt":
  globalParameters["RuntimeLanguage"] = "OCL"
else:
  globalParameters["RuntimeLanguage"] = "HIP"

globalParameters["CodeObjectVersion"] = "V2"
globalParameters["CxxCompiler"] = "hcc"

# might be deprecated
globalParameters["EnableHalf"] = False
globalParameters["ClientArgs"] = ""

# Save a copy - since pytest doesn't re-run this initialization code and YAML files can override global settings - odd things can happen
defaultGlobalParameters = deepcopy(globalParameters)

################################################################################
# Enumerate Valid Solution Parameters
################################################################################
validWorkGroups = []
for numThreads in range(64, 1025, 64):
  for nsg in [ 1, 2, 4, 8, 16, 32, 64, 96, 128, 256 ]:
    for sg0 in range(1, numThreads//nsg+1):
      sg1 = numThreads//nsg//sg0
      if sg0*sg1*nsg == numThreads:
          workGroup = [sg0, sg1, nsg]
          validWorkGroups.append(workGroup)

validThreadTileSides = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
validThreadTiles = []
for i in validThreadTileSides:
  for j in validThreadTileSides:
    validThreadTiles.append([i, j])

validTensorAFormats = ('NCHW', 'NHWC', 'CNHW', 'NCDHW', 'NDHWC', 'CNDHW')
validTensorBFormats = ('NCHW', 'NHWC', 'CNHW', 'NCDHW', 'NDHWC', 'CNDHW', \
                        'KCYX', "CKYX", "CYXK",  'KCZYX', 'CKZYX', 'CZYXK')
validTensorDFormats = ('NCHW', 'NHWC', 'CNHW', 'NCDHW', 'NDHWC', 'CNDHW', \
                        'KCYX', "CKYX", "CYXK",  'KCZYX', 'CKZYX', 'CZYXK')
validMacroTileSides = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 6, 12, 24, 48, 96, 192, 384, 768 ]
validMacroTiles = []
validISA = [(0,0,0)]
validISA.extend(globalParameters["SupportedISA"])
depthUs = list(range(-16, 0))
depthUs.extend(list(range(2,512+1,1)))
for i in validMacroTileSides:
  for j in validMacroTileSides:
    validMacroTiles.append([i, j])

validMFMA = {}
validMFMA["H"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validMFMA["S"] = [[32,32,1,2], [32,32,2,1], [16,16,1,4], [16,16,4,1], [4,4,1,16]]
validMFMA["B"] = [[32,32,2,2], [32,32,4,1], [16,16,2,4], [16,16,8,1], [4,4,2,16]]
validMFMA["4xi8"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validMatrixInstructions = [[], [-1]] + validMFMA["H"] + validMFMA["S"] + validMFMA["B"] + validMFMA["4xi8"]

validParameters = {
    "LoopDoWhile":                [ False, True ], # Source. True=DoWhile, False=For loop
    "LoopTail":                   [ False, True ], # tail loop handles non multiples of unrolled summation loop

    # threads load elements from global into registers, then write from registers to LDS
    # these options affect those read/write patterns
    # coalesce-group=True  means adjacent threads will     read adjacent addresses; if the data needs to be transposed then adjacent threads will NOT write adjacent elements to LDS.
    # coalesce-group=False means adjacent threads will NOT read adjacent addresses; if the data needs to be transposed then adjacent threads will     write adjacent elements to LDS.
    # this parameter really only matters for transposing
    # =False means the L1 cache will do the transposing work and it is quite fast; then data is written coalesced (no bank conflicts) to LDS.
    # =True means the transpose will happen while writing to LDS, this usually has bank conflicts, but it appears the throughput is still fast enough to not slow the VALUs down.
    # it appears that the L1 cache can still achieve quite a bit of performance for GRCG=False, but overall it's usually faster to read coalesced
    "GlobalReadCoalesceGroupA":   [ False, True ],
    "GlobalReadCoalesceGroupB":   [ False, True ],

    # for transposes, this option governs how short-vectors should be read from global and written to lds
    # it is impossible to transpose data while operating on short-vectors for GlobalRead,LocalWrite and LocalRead; an odd number of those must be transposing and operating on vector components.
    # since data will be read from lds many more times than it will be written, data must always end up in lds such that short-vectors can be read from lds
    # =True means read short-vector from global and write its components to lds
    # =False means read vector components from global so that a full short-vector can be written to lds
    # both options were supported until a refactoring of the short-vector code (necessary to enable assembly) broke it. Since =True always seems to be faster, no time has been spend on fixing =False
    #  it may still work in source, but just not in assembly. The problem is the order in which elements are stored into vgprs, is different than the order in which they are written to lds. In source each
    #  loaded element gets a variable name which in independent of the order that they are written in the source code, but in assembly the values are just assigned vgprs in order and that order needs to be shuffles.
    "GlobalReadCoalesceVectorA":  [        True ], # FIXME =False worked before the vector refactor; fixing requires re-ordering load/store indices; but they aren't the faster option so not worth time right now
    "GlobalReadCoalesceVectorB":  [        True ],

    "PrefetchGlobalRead":         [ False, True ], # prefetch / double-buffer reads from global memory -> vgprs -> lds. Requires 2X LDS space, and VGPRs for buffering data on way into LDS
    "PrefetchLocalRead":          [ 0,1,2,3], # prefetch / double-buffer reads from lds (or 2 for triple-buffer, 3 for quad-buffer).  Increases size of ValuA/ValuB registers.

    # When splitting up the summation between workgroups, there are two options for organizing which workgroup will do what
    # If we begin with N workgroups and set GSU=4, there will now be 4N workgroups
    # GSUWGMRR=False means workgroup 0,1,2,3 will all work on the same tile; =True means workgroup 0, N-1, 2N-1, 3N-1 will all work on the same tile
    # GSUSARR=False means the 4 workgroups do whole chunks of the summation: k=0 -> K/4-1, k=K/4 -> 2K/4-1, k=2K/4 -> 3K/4-1, k=3K/4 -> 4K/4-1
    # GSUSARR=True means the 4 workgroups round robin split up the chunks of the summation: k=0 -> DU-1, 4DU -> 5DU-1, ...; k=1DU -> 2DU-1, 5DU -> 6DU-1...; ...
    "GlobalSplitU":               list(range(1, 1024+1)),
    "GlobalSplitUWorkGroupMappingRoundRobin":     [ False, True ],
    "GlobalSplitUSummationAssignmentRoundRobin":  [ False, True ],

    # in opencl for some compilers, performance improved by putting a memfence after each subiteration; it prevented the loads of one subiteration from being moved
    # into a prior iteration, which would help latency but it consumed more vgprs which was a net loss
    "UnrollMemFence":             [ False, True ],

    # not used yet; will refer to combining multiple reads into single instruction
    # such as ds_read_b32 -> ds_read2_b32
    # the pro is that it cuts in half the number of instructions
    # the con is that bits per offset is half, so arithmatic might be required to increment and reset offset vgprs
    "GlobalRead2A":               [ False, True ],
    "GlobalRead2B":               [ False, True ],
    "LocalWrite2A":               [ False, True ],
    "LocalWrite2B":               [ False, True ],
    "LocalRead2A":                [ False, True ],
    "LocalRead2B":                [ False, True ],

    # don't create a whole copy of the Unroll loop with loads removed - instead
    # use buffer limits to suppress global loads and ignore unnecessary ds_reads
    "SuppressNoLoadLoop":         [False, True],

    # For PrefetchGlobalRead=1, create a second copy of the unroll loop with
    # the LDS pointer swaps expanded into inline constants for LDS read and write instructions
    # This eliminates 4 vector XOR instructions used for pointer swap
    "ExpandPointerSwap":          [False, True],

    # Schedule global reads and global read incrementsinto LocalRead iterations
    # Can reduce pressure on local read instruction dispatch queue
    # 0=perform global reads at start of instruction loop
    # 1=schedule into the local read instruction iterations
    "ScheduleGlobalRead":         [0, 1],

    # Schedule local writes into LocalRead iterations.
    # Can reduce pressure on local read instruction dispatch queue
    "ScheduleLocalWrite":         [0, 1],

    # Scheduling algorithm to use for each iteration:
    # 0 = minimal/no scheduling.  Global Read and increments, followed by local reads,
    # followed by local writes, followed by MACs
    "ScheduleIterAlg":             [0, 1],

    # LDD Support
    # Allow LDD and StrideD to != LDC and StrideC for LDD <= LDC and LDD == M
    "LdcEqualsLdd":               [ False, True ],

    # Interleave alpha scale calculation with beta loads and address calcs - rather
    # than as a separate block of instructions
    "InterleaveAlpha":             [0, 1],

    # Create a copy of NoLoadLoop which interleaves the stores with the final mac
    # calculation and may perform other optimizations
    # 0 = no interleave
    # 1 = interleave one stores after required macs have completed execution
    # 2 = interleave two stores after required macs have completed execution
    "OptNoLoadLoop":               [0, 1, 2],

    # Prefetch across persistent kernel iterations - the no-load-loop computes the
    # tile assignment and next global read offset and launches the buffer loads for
    # the next tile in the sequence.
    "PrefetchAcrossPersistent":    [0, 1],

    "BufferLoad":                 [ False, True ],
    "BufferStore":                [ False, True ],

    # Attempt to load directly from global memory into LDS.
    # Assembly only
    # Requires BufferLoad, assembler support for lds modifier on buffer
    # loads (checked automatically), GlobalVectorWidth=1 (this is hw
    # requirement) and A/B must not require any transpose.
    # DirectToLds reduces load latency and eliminates the
    # G2L registers used to stage data.  Also replaces the
    # local write offset with an SGPR.
    # For an 8x8 TT with PrefetchGlobalRead=1 this can save 33 VGPRs.
    "DirectToLds":                [ False, True ],

    # Load options:
    # (GRO = Global Read Offset)
    # BufferLoad=0:
    #  = Use flat instructions with 64 bit GRO for each load
    #    + supports sizes up to 2^64
    #    - uses many VGPR for addressing
    #    - uses execmask+compares for edge detection
    #    - generates extra LDS traffic (could convert flat->global load)
    # BufferLoad=1:
    #  = Use buffer load instructions with 32-bit offset
    #    + Less VGPRS (32b offset vs 64-bit) needed for addressing
    #    + Uses hardware buffer limit for edge detection
    #    - Limited range - the bot-right corner of macro-tile (plus padding=GRVW
    #        for shift-pointer, if ShiftPtr is required) must be within 2^32.
    #      ShiftPtrPad = MayShift ? GRWV*BPE : 0
    #      For TLU=1: Unroll*StrideA1 + ShiftPtrPad <= 2^32
    #      For TLU=0: MT*StrideA1 + ShiftPtrPad <= 2^32
    #      These conditions should be checked using Assert - TODO
    #  = UseSgprForGRO=1:
    #    + Attempt to use SGPR for Global Read Offsets.
    #    + Use one VGPR base GRO + many SGPR GRO rather than many VGPR GRO.
    #    + Each SGPR stores an offset from base GlobalReadOffset+0.
    #    - Requirements for UseSgprForGRO=1:
    #      - BufferLoad=1
    #      - Use appropriate Assert*ElementMultiple or GRVW=1 to eliminate need for ShifPtr
    #        (UseSgprForGRO does not support ShiftPtr since ShiftPtr needs to potentially shift GRO)
    #  = KernelWriterAssembly also supports 64-bit 2D buffer size (see use64bPbcLimit)
    #    - Requires 4 instructions to move scalar limit and a couple SGPR
    #    - Enabled by default.  If the overhead matters we can add asserts/YAML parm to specialize


    # Converting VGPR GRO into SGPR GRO is usually a win
    # However, the mode may exhaust all available SGPR, in particular for large unroll
    # -1 attempt to use a hueristic to determine when the tile size will use too many SGPR and fall back to VGPR
    "UseSgprForGRO":              [ -1, 0, 1],

    # Some work-items in the group may not participate in the final buffer load.
    # Allows more flexibility in choosing DepthU.
    # 1= allocate extra addressing vpgr for edge cases
    # 2= use temp vgpr inside unroll loop, may save 1 VPR if both A and B have a fractional edge but costs v_alu
    "FractionalLoad":             [ 0, 1, 2] ,

    # Attempt to vectorize atomics
    # 1,2,4 : Number of elements to vectorize
    # -1 : Maximum supported value.  Half=2, Single=1, Double=1
    # Currently 32-bit CAS only, eventually might support more
    "VectorAtomicWidth":          [ -1, 1, 2 ] ,

    # Assertion properties
    # These provide information or assertions that the problem size meets certain requirements
    # for sizes or alignments.  The kernel generator can use this information to produce
    # a kernel which uses those assertions to produce a faster kernel.
    #
    # If modifying or adding Assertions also change ProblemProperties class in TensileTypes.h

    # Kernel generator will assume that the summation size is some multiple of the element size
    # and use this to optimize the kernel.
    # This can result in more efficient kernels, but requires runtime checking to ensure the specified
    # summation value meets the requirements.
    # (Recommended AF1EM value is 8 for half, 4 for single, 2 for double)
    #
    # Optimizations enabled by AssertSummationElementMultiple>1:
    #  - If >=2 for half:
    #     - Tail loop loads can be vectorized 2X to use dword
    #     - Enables asm kernels on V20
    #     - Can use DirectToLds for both unroll and tail loops
    #  - Tail loop can be unrolled up to InnerUnroll amount if AssertSummationElementMultiple%InnerUnroll==0
    #
    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertSummationElementMultiple": [1,2,4,8],

    # Kernel generator will assume that the FreeIndex[0] size is some multiple of the element size
    # and use this to optimize the kernel.
    # FreeIndex[0] is usually letter "I"
    # (Recommended AF0EM value is 8 for half, 4 for single, 2 for double)
    #
    # Optimizations enabled by AssertFree0ElementMultiple>1:
    # Load optimizations:
    #  - For TLU=1 matrix, if AF1WM>=GLVW then can enable UseSgprForGRO
    #      - Reduces registers used for address calculations
    #      - Enables FractionalLoad for more flexibility in address calcs
    #      - Removes address shift/unshift code
    #    - UseSgprForGRO will only be enabled if all matrices meet assertion requirements.
    #
    # Store Optimizations:
    #  - Can vectorize stores in edge tiles.  Vector width can be up to AF0EM.
    #   (since C matrix is always coalesced in Free0 index diretion and this assertion guarantees the index element multiple)
    #
    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertFree0ElementMultiple" : [1,2,4,8],

    # Kernel generator will assume that the FreeIndex[1] size is some multiple of the element size
    # and use this to optimize the kernel.
    # FreeIndex[1] is usually letter "J"
    # (Recommended AF1EM value is 8 for half, 4 for single, 2 for double)

    # Optimizations enabled by AssertFree1ElementMultiple>1:
    #  - See above AssertFree0ElementMultiple "Load optimizations"

    # 1 indicates no assertion (since all sizes are multiples of 1)
    "AssertFree1ElementMultiple" : [1,2,4,8],

    # Some kernels only work for certain sizes, see ProblemProperties in TensileTypes for exact defs
    "AssertMinApproxSize" : [0,1,2],

    # Generate code inside kernel to check Assertions on Tensor dimensions
    "CheckTensorDimAsserts":               [False, True],

    # Generate code inside kernel to check several dimension overflow cases, in particular around use of 32-bit calcs
    # 0 = no check, 1=checks for cases that should be avoided through assertions and kernel selection,
    # 2=checks for cases that should never happen
    "CheckDimOverflow":               [0,1,2],

    # Stagger the start summation position of the tiles.
    # Elements from the summation dimension are loaded at offsets rather than all starting at 0.
    # StaggerU is the max 'clicks' of StaggerUStride bytes where each wg starts ; see StaggerUMapping
    # for how the specific stagger for a given wg is determined.
    #
    # The tile assignment C are same as with StaggerOffset=0 ; the difference is the
    # order that the summation elements are added.
    # GRO will wrap back to the row start start when the edge is reached.
    #
    # This can be effective for TLU=0 style matrices where the K dimension is a large power-of-2.
    # In this case the start of each row of the tile is separated by an exact power-of-2
    # which causes poor dram, cache, and tlb behavior.  V20 has 16 channels each 256 bytes wide.

    # StaggerU adjusts the start position in the summation (aka 'U') dimension
    # to avoid these conflicts.  Both A and B matrix start at the adjusted position.
    # If >0 specifies the offset in multiples of the macro-tile "unroll" dim
    #  - Higher values will spread traffic to more channels but provide less L2 re-use.
    #  - StaggerU and WorkGroupMapping interact and should be tuned together -
    #    The WGM controls how tiles are assigned in C matrix, while StaggerU controls where those
    #    tiles start reading their summation dim parms.
    #  - StaggerU requires BufferLoad==1 and is silently ignored if BufferLoad==0
    "StaggerU":              [0,2,4,8,16,32,64],

    # Stride in bytes for each staggeru 'click'.
    # 256 is recommended since this is the width of memory channel (on gfx803,gfx900,gf906) - so
    # each click will start in a new memory channel and spread traffic among the 16 available channels.
    # For example StaggerUStride=256 and StaggerU=8 will use 8 unique starting points
    # in summation dimension, each offset by 256-bytes - provided the tensor dims are large
    # enough to support this.
    # StaggerUStride will be internally increased so it is an integer multiple of DepthU*BpeAB.
    # (the implementation requires this - the unroll iteration accesses data in steps of
    # DepthU*BPE
    "StaggerUStride":               [16,32,64,128,256,512,1024],

    # How the tile assignment (wg0, wg1, wg2) controls the initial StaggerU offset:
    # 0: Use wg0
    # 1: Use wg1
    # 2: Use wg2
    # 3: Use wgSerial, wgSerial = wg0 + (wg1 % WorkGroupMapping) * nwg0
    # 4: Debug mode, offset each tile max allowed StaggerU.  This just moves hotspot
    #    to a different bank since all workgroups still start at same point.
    "StaggerUMapping":       [0,1,2,3,4],

    # For Block Mapping type:
    # 0   : Use hardware-assigned wg number with no remapping.
    # N   : WG block width.  "Wrap" to a new wg1 "row" assignment after N WGs assigned in that row.
    # < 0 : Swaps the position of wg0 and wg1.  Does not change NumWorkGroups* or ProblemNumWorkGroups*. No longer supported.
    # Tensor C always mapped with first free coord as fastest moving
    # (Elements in this dimension are sequential in memory.
    #
    # For 2D nonbatched Matrix this means index order is I, then J
    # For 2D batched Matrix this means index order is I, then J, then K.
    #
    # Then for 2D case:
    #   - If drawn in row-major format, I is the width and J is the height.
    #   - WGM determines dimensions of the box used to assign tiles from C
    #   - WGM is the height of the box (in the J dimension)
    #   - Given WGM, the box width (in I dim) is determined by number of CUs
    #   - The box always moves across matrixC in the fastest-moving "I" dim, then
    #     wraps to next J.  TODO - might be useful to change this?
    #
    # Examples for 2D matrix:
    # WGM=8:  on CU64 machine this is a square box
    # WGM=1:  Short/Fat - this will cover maximum width in I dimension of C.  This matches hardware assigned mapping.
    # WGM=64: Tall/Skinny - this will cover maximum width in J dimention of C.
    #
    # Formula for wgSerial:
    # wgSerial = wg0 + (wg1 % WorkGroupMapping) * nwg0
    "WorkGroupMapping":           list(range(0,1024+1)),  # change a workgroup's id so that the all the workgroups on the gpu at a time are hitting L2 cache the best
    "WorkGroupMappingType":       ["B", "Z"],           # Blocking, Z-order (not any faster than blocking, especially for the arithmetic it requires)
    "MaxOccupancy":               list(range(1, 40+1)),       # wg / CU; if cache thrashing is hurting performance, this allocates extra lds to artificially limit occupancy
    "WorkGroup":                  validWorkGroups,      # ( wg0 x wg1 x LocalSplitU ) dimensions of the workgroup which will operate on a tile and share lds

    #ThreadTile: ( tt0 x tt1 ) dimensions of the C tile that each thread works on,
    # TT=4 and VW=4 means a thread will work on a tight 4x4 tile of C, where VW=1 means the tile will work on 16 spread out values
    # Generally, the VW determines the consecutive a WI will work on, then it will skip ahead SG0*VW elements to get to the next row of VGPR inputs
    "ThreadTile":                 validThreadTiles,
    "MacroTile":                  validMacroTiles,      # MT0 = wg0*tt0, MT1 = wg1*tt1

    # MatrixInstruction: (M x N x K x B)
    # XDLOPS tile definition, only valid for gfx908
    # If empty, do not use these instructions
    "MatrixInstruction":          validMatrixInstructions,

    # If positive, each switch includes switches <= the specified switch.
    # For example 3 will enable NoPostLoop+NoGlobalRead+NoLocalWrite
    # If negative, setting is precise and will disable only the specified code piece.
    # intended use is to evaluate which sections of the kernel are taking most of the execution time
    # 0=Baseline
    # 1= +NoPostLoop
    # 2= +NoGlobalRead
    # 3= +NoLocalWrite
    # 4= +NoLocalRead
    # 5= +NoWait +NoSync
    # 6= +NoMAC
    # 7= +NoPreLoop+ NoGlobalReadInc
    # 9= NullKernel
    # For example set DisableKernelPieces: [0,1,2,3,4,5,6,7,9]
    #   this will create a set of kernels with progessively more pieces of the kernel disabled
    "DisableKernelPieces":        list(range(-9,10)),         # disable pieces of the kernel, for performance isolation

    # assume atomics always work correctly.
    "DisableAtomicFail": [False, True],

    # 0  : standard launch
    # N>0 : launch persistent kernel with N workgroups per compute unit
    #       - Recommended min is enough WG to use all resources on the CU
    #       - Higher values result in shorter-running WG which are less 'persistent'
    #         this increases the switch time between work-groups but results in
    #         more opportunities to schedule other WG or recover if a wg runs long
    #         or all compute units were not available before the launch.
    #       - Host code will not launch more groups than tiles in the C space
    # Assertions/Requirements: NumWorkGroups0 * NumWorkGroups1 < 2^32
    "PersistentKernel":           range(0,512+1) ,       # Use persistent kernel.

    # Allow macro-tile to span batch dimensions and thus a single workgroup can work across batch dimensions.
    # This can improve utilization, in particular if macro-tile is larger than the lower dimensions.
    # The byte address of the last element in the packed array must fit in 2^32.
    # 0x0 = each workgroup works on a single batch dim.
    # 0x1 = pack Batch dimensions into wg0/A - works if all batch strides for B==0.
    #       Also must set AssertFree0ElementMultiple to >= GlobalReadVectorWidth
    # 0x2 = pack Batch dimensions into wg1/B - works if all batch strides for A==0
    #       Also must set AssertFree1ElementMultiple to >= GlobalReadVectorWidth
    # 0x3 = pack batch dims into both A and B. Could support any stride for A and B. (Not supported yet)
    "PackBatchDims":             [0,1,2],

    # Pack free dimensions
    # If True, allow macro-tile to span free dimensions.  Single workgroup can work across multiple free dimensions.
    # If False, macro-tile is always Free0*Free1.  Additional free dimensions are not supported.
    "PackFreeDims":              [False, True],

    # Granularity allowed when packing tensor dims.
    # Lower values are finer granularity which requires more dimension division operations on store path
    # but supports more flexible tensor dimes.
    # Higher values are coarser values - less dimension division operations but tensor dims must meet
    # more stringent element multiple requirements
    # 0x1 : Any dimension supported, compute dims after each element (not supported yet)
    # 0x2 : VectorWidth must not span tensor dim
    "PackGranularity": [2],

    # Controls desired width (#elements) for loads from global memory -> LDS.
    # and eliminates the pointer unshift logic
    # -1 : Set GlobalReadVectorWidth =  VectorWidth
    #  1 cannot be used for half type.
    "GlobalReadVectorWidth":      [ -1, 1, 2, 3, 4, 6, 8 ],

    # threads should read/write/operate on this many contiguous elements from the C matrix.
    # If VW=4 then thread0 will process 4 consec C elements, then thread1 next 4, etc.
    # If the ThreadTile is > VectorWidth then thread0 will next operate on the 4 elements in C at (4*NumThreads)
    # Typically the load vector width and store vector width are directly related to the VW.
    # The global load width is closely related to the width of local stores so
    # GlobalReadVectorWidth also ontrols local write width.
    # Local read width also matches since VectorWidth consec elements must be read
    # Typically matching 16 bytes is good choice since the stores will be optimally coalesced with 16 bytes/WI.
    # -1 means use the largest vector width up to 128 bits.
    # Using a VW too large which results in >16bytes/thread isn't supported
    "VectorWidth":                [ -1, 1, 2, 3, 4, 6, 8 ],

    # If False, store 1 element per instruction.
    # If True, store vector-width elements per instruction.
    "VectorStore":                    [False, True],

    # place upper and lower limits on the skinny-ness of macro tiles; shape=1 means square tile, like 64x64. shape=4 means 4x64 or 64x4 or 128x8...
    # these will just mark some kernels as invalid so that fewer kernels will be checked
    "MacroTileShapeMin":          list(range(1, 256+1)),
    "MacroTileShapeMax":          list(range(1, 256+1)),

    # when loading all the data from global into lds requires multiple load instructions, these parameters govern which
    # loads will pull which rectangle of data from global into lds
    # NLC=1 means one load along the coalesced dimension, which results in the most coalescing possible
    # NLC=-1 looks for the largest number of reads along the coalesced dimension which results in the least ammount of coalescing;
    # however in this case the stride between one load and another is a static value, therefore buffer loads only need one set of registers
    # whereas the =1 case has a stride which is a multiple of a kernel argument and therefore needs one address per load in the perpendicular dimension
    "NumLoadsCoalescedA":         list(range(-1, 64+1)),
    "NumLoadsCoalescedB":         list(range(-1, 64+1)),

    # DepthU, LocalSplitU (which is the 3rd number in WorkGroup), and LoopUnroll are closely related
    # LoopUnroll=4 means there are 4 subiterations within the loop, 4 actual iterations written in the code.
    # LocalSplit=2 means the workgroup is split up into 2 subgroups, and each subgroup is doing different parts of the summation.
    # subgroup0 does k=0-3, 8-11... and subgroup1 does k=4-7, 12-15...
    # So, each iteration through the summation loop, which has 4 actual subiterations, does 8 summation iterations, because each subgroup did 4;
    # and when data is read from global memory the threads read 8 elements along the summation dimension.
    # DepthU = LoopUnroll * LocalSplitU = 4*2 in this case
    # it made more sense for the user to directly control LocalSplitU and DepthU, then derrive afterwards LoopUnroll=DepthU/LocalSplitU
    # -1 : Only allow GLVW=1
    # -2 : Only allow max(GLVWA,GLVWB) < VW ?
    # -3 : Only allow min(GLVWA,GLVWB) < VW ?
    "DepthU":                     depthUs,

    # integer ammount of padding to put into LDS, in 2016 this didn't seem to help performance, profilers were showing that channel conflicts weren't really hurting
    # performance so this has been deprecated and probably doesn't work
    # -1 means use same padding as the VectorWidth if TLU=0 else 0.  (Padding only helps when transpose is required)
    "LdsPadA":                     [ -1, 0, 1, 2, 3, 4, 8],
    "LdsPadB":                     [ -1, 0, 1, 2, 3, 4, 8],

    # tinkered with adding extra syncs or waits in the assembly kernels to see if it would improve the sequencing between workgroups, "fully synchronous scheduling" is WAY more promising; this can be deprecated
    "PerformanceSyncLocation":    list(range(-1, 16*16+1)),
    "PerformanceWaitLocation":    list(range(-1, 16*16+1)),
    "PerformanceWaitCount":       list(range(-1, 16)),

    # add gls or slc after global memory read/writes to change cacheing, not cacheing the writes is promising and improved performance a tiny bit
    "NonTemporalC":               list(range(0,4)),
    "NonTemporalA":               list(range(0,4)),
    "NonTemporalB":               list(range(0,4)),

    # guard against out of bounds reads
    # None: don't guard
    # Branch: use if statements (source only, and doesn't support VW)
    # ShiftPtr: shift read pointers to be in bounds, then unshift registers (source & assembly),
    # ShiftPtr does not support very small problem dims < global load vector width since the shift
    # would move outside the array bounds.
    # If GLVW==1 or Assert*ElementMultiple for the coalesced dim is > GRVW, then shifting is not
    # necessary and the shift/unshift code will not be generated
    "EdgeType":                   [ "Branch", "ShiftPtr", "None" ], # None=don't guard against ou

    # Group together unroll iterations inside the unroll loop.
    # For example, InnerUnroll=2 will fetch LDS for two unroll iterations
    "InnerUnroll":                [1,2,4],

    # Arrange elements in LDS so N elements consec in U-dim are adjacent in LDS
    # 1 is default and results in no interleaving.
    # Implementation only supports LocalDotLayout that is a power-of-two
    "LocalDotLayout":             [1,2,4,8],

    # Aggressive performance mode
    # Some of these may cause instability, particularly s_setprio
    # 0=none, 1=add setprio, 2=add setprio and modify LDS to allow only 2 waves/simd
    "AggressivePerfMode":       [0,1,2],

    # Kernels should be written in assembly or source
    # if assembly, ISA will determine architecture
    # if source, Runtime will determine language
    # later on, we'll relax this to inner kernel languages and outer kernel languages, such as inline asm embedded in ocl or in llvm
    "KernelLanguage":             [ "Assembly", "Source" ],
    "ISA":                        validISA,       # arch for assembly kernels

    # Replaces assembly kernels if they are found in the directory Tensile/Tensile/ReplacementKernels
    "ReplacementKernel":          [False, True],

    "MinVgprNumber":                list(range(0,256)),

    "MaxVgprNumber":                list(range(0,257)),
    }


# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"LoopDoWhile":               [ False ] },
    {"LoopTail":                  [ True ] },
    {"EdgeType":                  [ "Branch" ] },
    {"InnerUnroll":               [ 1 ] },
    {"LocalDotLayout":            [ 1 ] },
    {"AggressivePerfMode":        [ 1 ] },
    {"KernelLanguage":            [ "Source" ] },
    {"LdsPadA":                   [ 0 ] },
    {"LdsPadB":                   [ 0 ] },
    {"MaxOccupancy":              [ 40 ] },
    {"VectorWidth":               [ -1 ] },
    {"VectorStore":               [ True ] },
    {"GlobalReadVectorWidth":     [ -1 ] },
    {"GlobalReadCoalesceVectorA": [ True ] },
    {"GlobalReadCoalesceVectorB": [ True ] },
    {"GlobalReadCoalesceGroupA":  [ True ] },
    {"GlobalReadCoalesceGroupB":  [ True ] },
    {"PrefetchGlobalRead":        [ True ] },
    {"PrefetchLocalRead":         [ 1 ] },
    {"UnrollMemFence":            [ False ] },
    {"GlobalRead2A":              [ True ] },
    {"GlobalRead2B":              [ True ] },
    {"LocalWrite2A":              [ True ] },
    {"LocalWrite2B":              [ True ] },
    {"LocalRead2A":               [ True ] },
    {"LocalRead2B":               [ True ] },
    {"SuppressNoLoadLoop":        [ False ]},
    {"ExpandPointerSwap":         [ True ]},

    {"ScheduleGlobalRead":        [ 1 ] },
    {"ScheduleLocalWrite":        [ 1 ] },
    {"ScheduleIterAlg":           [ 1 ] },

    {"LdcEqualsLdd":              [ True ] },
    {"InterleaveAlpha":           [ 0 ] },
    {"OptNoLoadLoop":             [ 1 ] },
    {"PrefetchAcrossPersistent":  [ 0 ] },

    {"BufferLoad":                [ True ] },
    {"BufferStore":               [ True ] },
    {"DirectToLds":               [ True ] },
    {"UseSgprForGRO":             [ -1 ] },
    {"AssertSummationElementMultiple": [ 1 ] },
    {"AssertFree0ElementMultiple": [ 1 ] },
    {"AssertFree1ElementMultiple": [ 1 ] },
    {"AssertMinApproxSize":        [ -1 ] },
    {"CheckTensorDimAsserts"      : [ False ] },
    {"CheckDimOverflow"           : [ 0 ] },

    {"StaggerU":                  [ 32 ] },   # recommend [0,32]
    {"StaggerUStride":            [ 256 ] },  # recommend 256 for V10,V20
    {"StaggerUMapping":           [ 0 ] },    # recommend [0,1]
    {"GlobalSplitU":              [ 1 ] },
    {"GlobalSplitUSummationAssignmentRoundRobin": [ True ] },
    {"GlobalSplitUWorkGroupMappingRoundRobin":    [ False ] },
    {"MacroTileShapeMin":         [ 1 ] },
    {"MacroTileShapeMax":         [ 64 ] },
    {"PersistentKernel":          [ 0 ] },
    {"PackBatchDims":             [ 0 ] },
    {"PackFreeDims":              [ 1 ] },
    {"PackGranularity":           [ 2 ] },
    {"FractionalLoad":            [ 0 ] },
    {"VectorAtomicWidth":         [ -1 ] },

    {"NumLoadsCoalescedA":        [ 1 ] },
    {"NumLoadsCoalescedB":        [ 1 ] },
    {"WorkGroup":                 [ [16,16,1]] },
    {"WorkGroupMappingType":      [ "B" ] },
    {"WorkGroupMapping":          [ 8 ] },
    {"ThreadTile":                [ [4,4] ] },
    {"MatrixInstruction":         [ [] ] },
    {"DisableAtomicFail":         [ 0 ] },
    {"DisableKernelPieces":       [ 0 ] },
    {"DepthU":                    [ -1 ] },
    {"PerformanceSyncLocation":   [ -1 ] },
    {"PerformanceWaitLocation":   [ -1 ] },
    {"PerformanceWaitCount":      [ -1 ] },
    {"NonTemporalC":              [ 0 ] },
    {"NonTemporalA":              [ 0 ] },
    {"NonTemporalB":              [ 0 ] },
    {"ReplacementKernel":         [ False ] },
    {"MinVgprNumber":             [0]},
    {"MaxVgprNumber":             [256]},
    ]
# benchmark these solution independently
defaultForkParameters = []
defaultBenchmarkForkParameters = []
defaultJoinParameters = []
defaultBenchmarkJoinParameters = []

# dictionary of defaults comprised for 1st option for each parameter
defaultSolution = {}
for paramList in [defaultBenchmarkCommonParameters, defaultForkParameters, \
    defaultBenchmarkForkParameters,defaultBenchmarkJoinParameters]:
  for paramDict in paramList:
    for key, value in paramDict.items():
      defaultSolution[key] = value[0]
# other non-benchmark options for solutions

# valid fields in ConvolutionConfig and explanations:
validConvolutionConfig= [
    # For OperationType == Convolution*
    # Examples: NCHW, NHWC, NCDHW, more
    # *HW* and *YX*   create solution with 2 spatial dimensions.
    # *DHW* and *ZYX* create solution with 3 spatial dimensions.
    "TensorAFormat",           # see validTensorAFormats
    "TensorBFormat",           # see validTensorBFormats
    "TensorDFormat",           # see validTensorDFormats

    # Each of the parms below specifies dimensions separated by 'x".
    # -  The notation follows 'convolution' convention so fastest-moving dimensions are last,
    #    and should mirror the order of the spatial dimension in the activation format.
    #    For example, in NCHW format Filter=3x1 is 3 in the H dimension and 1 in the W dimension.
    # -  2 or 3 dimensions are supported 'Filter:3x1' or 'Filter:3x3x1'.
    # - Use an integer to create a kernel with a compile-time constant
    #   Use "N" to create flexible kernel the value provided at runtime via appropriate
    #   size and stride values.
    # - 0 specifies the default.  Defaults below shown for 2 spatial dimensions; a 3-dimensional
    #   default will be created if the formats request 3 spacial dimensions.
    "Filter",                   # examples: 1x1,3x3,1x7,7x1,NxN,Nx5,3x3x3.  Default=1x1/1x1x1.
    "Stride",                   # examples 1x1,2x2,1xN, 2x2x2.  Default=1x1/1x1x1.
    "Dilation",                 # examples 1x1,2x2,1xN, 2x2x2.  Default=1x1/1x1x1.

    # Pad at start of each filter dimension. Recommend 0x0 when possible or NxN otherwise.
    # (performance difference from compile-time padding is not significant)
    "PadStart",                 # examples:1x1, 2x3, 2x2x2, NxN.  Default=0x0/0x0x0.
    # Pad at end of each filter dimension
    "PadEnd",                   # examples:1x1, 2x3, 2x2x2, NxN.  Default=0x0/0x0x0.

    # For grouped convolutions:
    "GroupCount",

    # pack spatial dims (d,h,w) into single tensor dim
    # This is preferred for cases where these dimensions are packed in memory
    # since it reduces addressing overhead and will produce a more efficient kernel
    # Default is 1, multiple dimensions will be created if needed for strides or otrher cases.
    "PackedSpatialDims",
    ]

################################################################################
# Default Problem Type
################################################################################
defaultProblemType = {
    # =GEMM uses TransposeA,B paramters and makes the problem type more readeable for users
    # =TensorContraction  requires specifying
    "OperationType":            "GEMM",           # GEMM, TensorContraction, ConvolutionForward, ConvolutionBackwardData, ConvolutionBackwardWeights

    "ConvolutionConfig":        [],               # See validConvolutionConfig

    "DataType":                 0,                # data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "DestDataType":             0,                # destination data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "ComputeDataType":             0,             # compute data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "UseBeta":                  True,             # =True use beta parameter (asm will check for B=0 and optimize the write for that), =False don't use beta parameter
    "HighPrecisionAccumulate":  False,            # f32 += f16*f16
    "SilentHighPrecisionAccumulate": False,       # Keep kernel names the same for HPA mode.  Useful for testing.

    "ComplexConjugateA":        False,            # complex data should be conjugated for "C" transpose case
    "ComplexConjugateB":        False,

    # for OperationType == GEMM
    "TransposeA":               False,            # =True means transA="T" or "C", =False means transA = "N"
    "TransposeB":               True,
    "Batched":                  False,            # add batching dimension

    # for OperationType == TensorContraction
    # - Indices < NumIndicesC are Free or Batch indices and appear in C and D
    # - Indices which appear in both A and B, and are < NumIndicesC are batch.  A and B must have same number of batch indices.
    # - Indices which appear in both A and B, and are >= NumIndicesC are summation. A and B must have same number of summation indices.
    # - Indices which appear in A or B (but not both), are Free.  A and B may have different numbers of free indices.
    # - Summation loops are nested from smallest index number to largest, with the largest summation index as the 'unroll' loop.
    # - Memory order of C and D matrices is always 0..NumIndicesC-1, with 0 as the fastest-moving.
    #   - By choosing index assignments the output can be 'transposed'.  For example if IA=[1,2] IB=[0,2] then 0 is the coalesced dim for C/D.
    #   - Likewise batch index may be assigned between two free indices to control the output order, ie to write in CNHW format.
    #   - For example : IA=[0,1,3] IB=[2,1,3].  0,2 are free indices;  1 is batch.
    "IndexAssignmentsA":        [0, 2],
    "IndexAssignmentsB":        [1, 2],
    "NumIndicesC":              2,
    "UseInitialStrides":        False,

    # List of pairs of [index, constValue].
    # EX: SetConstStrideA: [ [3, 1], [2, 4] ] sets
    #     strideA for index3 to constant '1' and stride for index2 to constant '4'.
    "SetConstStrideA":          [],
    "SetConstStrideB":          [],

    # ZeroPad:
    # Zero-pad will add leading and trailing "pad" elements to the specified free
    # dimension when accessed by specified summation dimension.
    #
    # Format is list of tuples of [freeDim, sumDim, padLeading, padTrailing].
    #  - freeDim is the anchor where the zero-pad starts.
    #  - sumDim is the summation dim to which the padding checking is added.
    #  - padLeading is the number of elements to pad before the Start element
    #  - padTrailing is the number of elements to pad before the last element.

    # - Terms:
    #   - Start is the first summation element
    #   - FreeSize is the size of the specified free dimension (freeDim)
    #   - SumSize is the size of the specified summation dimension (sumDim)
    # - Pad Ranges:
    #   - Ranges show below are inclusive on the start element and exclusive on the last element.
    #     For example, [0,3) is 0,1,2.
    #    - Elements in the region [Start-padLeading, Start) are in the leading pad region and will return 0.
    #    - Elements in the memory region [Start + freeSize + sumSize - padTrailing,  Start + freeSize + sumSize)
    #     are in the trailing pad region and will return 0.
    # - Strides:
    #   - SummationStride is applied to compute the element address before checking the regions.
    #   - FreeStride is applied to the computation of the Start element, padLeading, and padTrailing.
    #   -  No memory access is performed for elements in the Pad regions.
    #   - The Pad regions are handled by manipulating the tensor addressing and are not visible in actual memory.
    #     For example, a tensor with 2 rows, 16 elements/row, padLeading=padTrailing=2 occupies 32 elements in memory (not 40)
    #   - Typical use case is to set summationStride < freeSize, with padLeading+padTrailing+1 == summationStride.
    # - Caveats:
    #  - CPU reference model does not yet support zero-padding
    #  - Eventually leading and trailing YAML parm will be removed and instead be specified as runtime kernel parms
    #  - ZeroPad requires that the ElementEdge <= 2^32:
    #    This is SizeFree+SizeSum + Pad_Leading + PadTrailingPad + padding=GRWW for shift-pointer) bytes < 2^32
    #    Likely this is less than the standard buffer load limits (bottom-right corner of macro-tile)

    #  EX: ZeroPadA: [ [0,1,  2,3]] # TensorA free index 0 with sum index 1 has leading pad=2 and trailing pad=3
    # Note nesting of brackets ; the parm can contain multiple padding tuples.

    "ZeroPadA":                 [], # [ [0,1, 2,3]]
    "ZeroPadB":                 [], # Not fully supported/tested yet

    # for LD description
    "NumIndicesLD":            4,
    "IndexAssignmentsLD":       [3, 4, 5, 6],      # order is LDD, LDC, LDA, LDB

    # Tile aware solution selection
    "TileAwareSelection":       False
    }

defaultProblemSizes = [{"Range": [ [2880], 0, 0 ]}]
defaultBenchmarkFinalProblemSizes = [{"Range": [
    [64, 64, 64, 512], 0, 0 ]}]
defaultBatchedProblemSizes = [{"Range": [ [2880], 0, [1], 0 ]}]
defaultBatchedBenchmarkFinalProblemSizes = [{"Range": [
    [64, 64, 64, 512], 0, [1], 0 ]}]


defaultSolutionSummationSizes = [32,64,96,128,256,512,1024,2048,4096,8192,16192]


################################################################################
# Default Analysis Parameters
################################################################################
defaultAnalysisParameters = {
    "ScheduleName":       "Tensile",
    "DeviceNames":  "fallback",
    "ArchitectureName": "gfx000",
    "SolutionImportanceMin":      0.01, # = 0.01=1% total time saved by keeping this solution
    }


################################################################################
# Searching Nested Lists / Dictionaries
# to see if keys exist and what their values are
################################################################################
# param name in structures?
def inListOfDictionaries(param, dictionaries):
  for dictionary in dictionaries:
    if param in dictionary:
      return True
  return False
def inListOfListOfDictionaries(param, dictionaries):
  for dictionaryList in dictionaries:
    if inListOfDictionaries(param, dictionaryList):
      return True
  return False
def inListOfLists(param, lists):
  for l in lists:
    if param in l:
      return True
  return False

# get param values from structures.
def hasParam( name, structure ):
  if isinstance(structure, list):
    for l in structure:
      if hasParam(name, l):
        return True
    return False
  elif isinstance(structure, dict):
    return name in structure
  else:
    return name == structure
    #printExit("structure %s is not list or dict" % structure)

def getParamValues( name, structure ):
  if isinstance(structure, list):
    for l in structure:
      param = getParamValues(name, l)
      if param != None:
        return param
    return None
  elif isinstance(structure, dict):
    if name in structure:
      return structure[name]
    else:
      return None
  else:
    printExit("structure %s is not list or dict" % structure)

################################################################################
# Print Debug
################################################################################
def print1(message):
  if globalParameters["PrintLevel"] >= 1:
    print(message)
    sys.stdout.flush()
def print2(message):
  if globalParameters["PrintLevel"] >= 2:
    print(message)
    sys.stdout.flush()

def printWarning(message):
  print("Tensile::WARNING: %s" % message)
  sys.stdout.flush()
def printExit(message):
  print("Tensile::FATAL: %s" % message)
  sys.stdout.flush()
  sys.exit(-1)

################################################################################
# Locate Executables
# rocm-smi, hcc, rocm_agent_enumerator
################################################################################
def isExe( filePath ):
  return os.path.isfile(filePath) and os.access(filePath, os.X_OK)
def locateExe( defaultPath, exeName ): # /opt/rocm/bin, hcc
  # look in path first
  for path in os.environ["PATH"].split(os.pathsep):
    exePath = os.path.join(path, exeName)
    if isExe(exePath):
      return exePath
  # look in default path second
  exePath = os.path.join(defaultPath, exeName)
  if isExe(exePath):
    return exePath
  return None

# Try to assemble the asmString for the specified target processor
# Success is defined as assembler returning no error code or stderr/stdout
def tryAssembler(isaVersion, options, asmString):
  asmCmd = "%s -x assembler -target amdgcn-amdhsa -mcpu=%s %s -" \
             % (globalParameters["AssemblerPath"], isaVersion, options)

  sysCmd = "echo \"%s\" | %s" % (asmString, asmCmd)

  try:
    result = subprocess.check_output([sysCmd], shell=True,  stderr=subprocess.STDOUT).decode()
    if globalParameters["PrintLevel"] >=2:
        print("asm_cmd: ", asmCmd)
        print("output :", result)
    if result != "":
      return 0 # stdout and stderr must be empty
  except subprocess.CalledProcessError as e:
    if globalParameters["PrintLevel"] >=2:
        print("CalledProcessError", e)
    return 0 # error, not supported

  return 1 # syntax works


################################################################################
# Assign Global Parameters
# each global parameter has a default parameter, and the user
# can override them, those overridings happen here
################################################################################
def assignGlobalParameters( config ):

  global globalParameters

  print1("# Restoring default globalParameters")
  for key in defaultGlobalParameters:
    globalParameters[key] = defaultGlobalParameters[key]

  # Minimum Required Version
  if "MinimumRequiredVersion" in config:
    if not versionIsCompatible(config["MinimumRequiredVersion"]):
      printExit("Benchmark.yaml file requires version=%s is not compatible with current Tensile version=%s" \
          % (config["MinimumRequiredVersion"], __version__) )

  # User-specified global parameters
  print2("GlobalParameters:")
  for key in globalParameters:
    defaultValue = globalParameters[key]
    if key in config:
      configValue = config[key]
      if configValue == defaultValue:
        print2(" %24s: %8s (same)" % (key, configValue))
      else:
        print2(" %24s: %8s (overriden)" % (key, configValue))
    else:
      print2(" %24s: %8s (unspecified)" % (key, defaultValue))

  # ROCm Agent Enumerator Path
  globalParameters["ROCmAgentEnumeratorPath"] = locateExe("/opt/rocm/bin", "rocm_agent_enumerator")
  globalParameters["AssemblerPath"] = os.environ.get("TENSILE_ROCM_ASSEMBLER_PATH")
  if globalParameters["AssemblerPath"] is None:
    globalParameters["AssemblerPath"] = locateExe("/opt/rocm/bin", "hcc")
  globalParameters["ROCmSMIPath"] = locateExe("/opt/rocm/bin", "rocm-smi")
  globalParameters["ExtractKernelPath"] = locateExe("/opt/rocm/bin", "extractkernel")

  # read current gfx version
  if os.name != "nt" and globalParameters["CurrentISA"] == (0,0,0) and globalParameters["ROCmAgentEnumeratorPath"]:
    process = Popen([globalParameters["ROCmAgentEnumeratorPath"], "-t", "GPU"], stdout=PIPE)
    line = process.stdout.readline().decode()
    while line != "":
      gfxIdx = line.find("gfx")
      if gfxIdx >= 0:
        major = int(line[gfxIdx+3:gfxIdx+4])
        minor = int(line[gfxIdx+4:gfxIdx+5])
        step  = int(line[gfxIdx+5:gfxIdx+6])
        if (major,minor,step) in globalParameters["SupportedISA"]:
          print1("# Detected local GPU with ISA: gfx%u%u%u"%(major, minor, step))
          globalParameters["CurrentISA"] = (major, minor, step)
        line = process.stdout.readline().decode()
    if globalParameters["CurrentISA"] == (0,0,0):
      printWarning("Did not detect SupportedISA: %s; cannot benchmark assembly kernels." % globalParameters["SupportedISA"])
    if process.returncode:
      printWarning("%s exited with code %u" % (globalParameters["ROCmAgentEnumeratorPath"], process.returncode))

  # Determine assembler capabilities by testing short instructions sequences:
  globalParameters["AsmCaps"] = {}
  globalParameters["ArchCaps"] = {}
  for (v) in globalParameters["SupportedISA"] + [(0,0,0)]:
    globalParameters["AsmCaps"][v] = {}
    globalParameters["ArchCaps"][v] = {}
    isaVersion = "gfx" + "".join(map(str,v))
    globalParameters["AsmCaps"][v]["SupportedISA"] = tryAssembler(isaVersion, "", "")
    globalParameters["AsmCaps"][v]["HasExplicitCO"] = tryAssembler(isaVersion, "", "v_add_co_u32 v0,vcc,v0,1")
    globalParameters["AsmCaps"][v]["HasDirectToLds"] = tryAssembler(isaVersion, "", "buffer_load_dword v40, v36, s[24:27], s28 offen offset:0 lds")
    globalParameters["AsmCaps"][v]["HasAddLshl"] = tryAssembler(isaVersion, "", "v_add_lshl_u32 v47, v36, v34, 0x2")
    globalParameters["AsmCaps"][v]["HasSMulHi"] = tryAssembler(isaVersion, "", "s_mul_hi_u32 s47, s36, s34")
    globalParameters["AsmCaps"][v]["HasCodeObjectV3"] = tryAssembler(isaVersion, "-mno-code-object-v3", "")
    if tryAssembler(isaVersion, "", "s_waitcnt vmcnt(63)"):
      globalParameters["AsmCaps"][v]["MaxVmcnt"] = 63
    elif tryAssembler(isaVersion, "", "s_waitcnt vmcnt(15)"):
      globalParameters["AsmCaps"][v]["MaxVmcnt"] = 15
    else:
      globalParameters["AsmCaps"][v]["MaxVmcnt"] = 0

    caps = ""
    for k in globalParameters["AsmCaps"][v]:
      caps += " %s=%u" % (k, globalParameters["AsmCaps"][v][k])

    print1 ("# Asm caps for %s:%s" % (isaVersion, caps))
    globalParameters["ArchCaps"][v]["HasEccHalf"] = (v==(9,0,6) or v==(9,0,8))
    print1 ("# Arch caps for %s:%s" % (isaVersion, globalParameters["ArchCaps"][v]))

  # For ubuntu platforms, call dpkg to grep the version of hcc.  This check is platform specific, and in the future
  # additional support for yum, dnf zypper may need to be added.  On these other platforms, the default version of
  # '0.0.0' will persist

  # Due to platform.linux_distribution() being deprecated, just try to run dpkg regardless.
  # The alternative would be to install the `distro` package.
  # See https://docs.python.org/3.7/library/platform.html#platform.linux_distribution
  try:
    output = subprocess.run(["dpkg", "-l", "hcc"], check=True, stdout=subprocess.PIPE).stdout.decode()

    for line in output.split('\n'):
      if 'hcc' in line:
        globalParameters['HccVersion'] = line.split()[2]

  except (subprocess.CalledProcessError, OSError) as e:
      printWarning("Error: {} looking for package {}: {}".format('dpkg', 'hcc', e))

  for key in config:
    value = config[key]
    if key not in globalParameters:
      printWarning("Global parameter %s = %s unrecognised." % ( key, value ))
    globalParameters[key] = value



################################################################################
# Assign Parameters
# populate dst with src[key] else give it the default/backup value
################################################################################
def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, \
    defaultDictionary):
  if key in sourceDictionary:
    destinationDictionary[key] = sourceDictionary[key]
  else:
    destinationDictionary[key] = defaultDictionary[key]

# populate dst with src[key] else abort since it's required
def assignParameterRequired(destinationDictionary, key, sourceDictionary):
  if key in sourceDictionary:
    destinationDictionary[key] = sourceDictionary[key]
  else:
    printExit("Parameter \"%s\" must be defined in dictionary %s" % (key, sourceDictionary) )

def CPUThreadCount(enable=True):
  if not enable or globalParameters["CpuThreads"] == 0:
    return 0
  else:
    cpu_count = len(os.sched_getaffinity(0))
    cpuThreads = globalParameters["CpuThreads"]
    if cpuThreads < 0:
        return cpu_count*abs(cpuThreads)
    return min(cpu_count, cpuThreads)

def starmap_apply(item):
  func, item = item
  return func(*item)

def apply_print_exception(item, *args):
  #print(item, args)
  try:
    if len(args) > 0:
      func = item
      args = args[0]
      return func(*args)
    else:
      func, item = item
      return func(item)
  except Exception:
    import traceback
    traceback.print_exc()
    raise
  finally:
    sys.stdout.flush()
    sys.stderr.flush()

def ProcessingPool(enable=True):
  import multiprocessing
  import multiprocessing.dummy

  threadCount = CPUThreadCount()

  if (not enable) or threadCount <= 1:
    return multiprocessing.dummy.Pool(1)

  return multiprocessing.Pool(threadCount)

def ParallelMap(function, objects, message="", enable=True, method=None):
  """
  Generally equivalent to list(map(function, objects)), possibly executing in parallel.

    message: A message describing the operation to be performed.
    enable: May be set to false to disable parallelism.
    method: A function which can fetch the mapping function from a processing pool object.
        Leave blank to use .map(), other possiblities:
           - `lambda x: x.starmap` - useful if `function` takes multiple parameters.
           - `lambda x: x.imap` - lazy evaluation
           - `lambda x: x.imap_unordered` - lazy evaluation, does not preserve order of return value.
  """
  threadCount = CPUThreadCount(enable)
  pool = ProcessingPool(enable)

  if threadCount <= 1 and globalParameters["ShowProgressBar"]:
    # Provide a progress bar for single-threaded operation.
    # This works for method=None, and for starmap.
    mapFunc = map
    if method is not None:
      # itertools provides starmap which can fill in for pool.starmap.  It provides imap on Python 2.7.
      # If this works, we will use it, otherwise we will fallback to the "dummy" pool for single threaded
      # operation.
      try:
        mapFunc = method(itertools)
      except NameError:
        mapFunc = None

    if mapFunc is not None:
      from . import Utils
      return list(mapFunc(function, Utils.tqdm(objects, message)))

  mapFunc = pool.map
  if method: mapFunc = method(pool)

  objects = zip(itertools.repeat(function), objects)
  function = apply_print_exception

  countMessage = ""
  try:
    countMessage = " for {} tasks".format(len(objects))
  except TypeError: pass

  if message != "": message += ": "

  print("{0}Launching {1} threads{2}...".format(message, threadCount, countMessage))
  sys.stdout.flush()
  rv = mapFunc(function, objects)
  print("{0}Done.".format(message))
  sys.stdout.flush()
  pool.close()
  return rv

################################################################################
# Push / Pop Working Path
# store a WorkingPath where to write files (like benchmark files)
################################################################################
def pushWorkingPath( foldername ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  globalParameters["WorkingPath"] = \
      os.path.join(globalParameters["WorkingPath"], foldername )
  ensurePath( globalParameters["WorkingPath"] )
def popWorkingPath():
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  globalParameters["WorkingPath"] = \
      os.path.split(globalParameters["WorkingPath"])[0]
def ensurePath( path ):
  try:
    os.makedirs(path)
  except OSError:
    pass
  return path

def roundUp(f):
  return (int)(math.ceil(f))

################################################################################
# Is query version compatible with current version
# a yaml file is compatible with tensile if
# tensile.major == yaml.major and tensile.minor.step > yaml.minor.step
################################################################################
def versionIsCompatible(queryVersionString):
  (qMajor, qMinor, qStep) = queryVersionString.split(".")
  (tMajor, tMinor, tStep) = __version__.split(".")

  # major version must match exactly
  if qMajor != tMajor:
    return False

  # minor.patch version must be >=
  if int(qMinor) > int(tMinor):
    return False
  if qMinor == tMinor:
    if int(qStep) > int(tStep):
      return False
  return True

# convert python list to C++ initializer style syntax
def listToInitializer(l):
  return "{" + ','.join(map(str, l)) + "}"

################################################################################
# Progress Bar Printing
# prints "||||" up to width
################################################################################
class ProgressBar:
  def __init__(self, maxValue, width=80):
    self.char = '|'
    self.maxValue = maxValue
    self.width = width
    self.maxTicks = self.width - 7


    self.priorValue = 0
    self.fraction = 0
    self.numTicks = 0
    self.createTime = time.time()

  def increment(self, value=1):
    self.update(self.priorValue+value)

  def update(self, value):
    currentFraction = 1.0 * value / self.maxValue
    currentNumTicks = int(currentFraction * self.maxTicks)
    if currentNumTicks > self.numTicks:
      self.numTicks = currentNumTicks
      self.fraction = currentFraction
      self.printStatus()
    self.priorValue = value

  def printStatus(self):
    sys.stdout.write("\r")
    sys.stdout.write("[%-*s] %3d%%" \
        % (self.maxTicks, self.char*self.numTicks, self.fraction*100) )
    if self.numTicks == self.maxTicks:
      stopTime = time.time()
      sys.stdout.write(" (%-.1f secs elapsed)\n"%(stopTime-self.createTime))
    sys.stdout.flush()

  def finish(self): pass

# Append copyrights to all files generated by tensile since they belong to Tensile intellectual property
CMakeHeader = """################################################################################
# Copyright (C) 2016-2019 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

###################################################
# This file was generated by Tensile:             #
# https://github.com/ROCmSoftwarePlatform/Tensile #
###################################################


"""

CHeader = """/*******************************************************************************
* Copyright (C) 2016-2019 Advanced Micro Devices, Inc. All rights reserved.
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
*******************************************************************************/

/**************************************************
* This file was generated by Tensile:             *
* https://github.com/ROCmSoftwarePlatform/Tensile *
**************************************************/


"""

HR = "################################################################################"
