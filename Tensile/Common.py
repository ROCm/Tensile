################################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
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
from . import Parallel
from collections import OrderedDict
from copy import deepcopy


import math
import os.path
import subprocess
import sys
import time

startTime = time.time()

ParallelMap = Parallel.ParallelMap

# print level
# 0 - user wants no printing
# 1 - user wants limited prints
# 2 - user wants full prints

################################################################################
# Global Parameters
################################################################################
globalParameters = OrderedDict()
workingDirectoryStack = []

########################################
# common
########################################
globalParameters["MinimumRequiredVersion"] = "0.0.0" # which version of tensile is required to handle all the features required by this configuration file
globalParameters["PerformanceMetric"] = "DeviceEfficiency" # performance metric for benchmarking; one of {DeviceEfficiency, CUEfficiency}
globalParameters["PrintLevel"] = 1                # how much info to print in generator. 0=none, 1=standard, 2=verbose
globalParameters["ClientLogLevel"] = 3            # the log level of client. 0=Error, 1=Terse, 2=Verbose, 3=Debug (Aligned with ResultReporter.hpp)
# benchmarking
globalParameters["KernelTime"] = False            # T=use device timers, F=use host timers
globalParameters["PreciseKernelTime"] = True      # T=On hip, use the timestamps for kernel start and stop rather than separate events.  Can provide more accurate kernel timing.  For GlobalSplitU kernels, recommend disabling this to provide consistent
# timing between GSU / non-GSU kernels
globalParameters["CodeFromFiles"] = True          # if False byte arrays will be generated during Benchmarking phase as before
globalParameters["SortProblems"] = False          # sort problems by size; else use order in YAML file
globalParameters["PinClocks"] = False             # T=pin gpu clocks and fan, F=don't
globalParameters["HardwareMonitor"] = True        # False: disable benchmarking client monitoring clocks using rocm-smi.
globalParameters["NumBenchmarks"] = 1             # how many benchmark data points to collect per problem/solution
globalParameters["SyncsPerBenchmark"] = 1         # how iterations of the stream synchronization for-loop to do per benchmark data point
globalParameters["EnqueuesPerSync"] = 1           # how many solution enqueues to perform per synchronization
globalParameters["SleepPercent"] = 300            # how long to sleep after every data point: 25 means 25% of solution time. Sleeping lets gpu cool down more.
# validation
globalParameters["NumElementsToValidate"] = 128   # number of elements to validate, 128 will be evenly spaced out (with prime number stride) across C tensor
globalParameters["BoundsCheck"] = 0   # Bounds check
#1: Perform bounds check to find out of bounds reads/writes.  NumElementsToValidate must be -1.
#2: Perform bounds check by front side guard page
#3: Perform bounds check by back side guard page
#4: Perform bounds check by both back and front side guard page

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
globalParameters["GenerateSourcesAndExit"] = False # Exit after kernel source generation.
globalParameters["ShowProgressBar"] = True     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["WavefrontWidth"] = 64     # if False and library client already built, then building library client will be skipped when tensile is re-run
globalParameters["ExitOnFails"] = 1     # 1: Exit after benchmark run if failures detected.  2: Exit during benchmark run.
globalParameters["CpuThreads"] = -1  # How many CPU threads to use for kernel generation.  0=no threading, -1 == nproc, N=min(nproc,N).  TODO - 0 sometimes fails with a kernel name error?  0 does not check error codes correctly
# FROM MERGE
#globalParameters["CpuThreads"] = -4         # How many CPU threads to use for kernel generation.  0=no threading, <0 == nproc*abs(CpuThreads), N=min(nproc,N)

# even if error occurs in kernel generation (ie due to resource overflow),
# generate the kernel source anyway.  Tensile will also attempt to run
# the kernel.  Useful to examine and debug overflow errors.
globalParameters["ForceGenerateKernel"] = 0

########################################
# optimization knob controls
########################################

globalParameters["UnrollLoopEfficiencyEnable"] = False   # if True split(S) MAC&LDS in each unroll iteration into n smaller groups..

########################################
# less common
########################################
globalParameters["CMakeBuildType"] = "Release"            # whether benchmark clients and library client should be release or debug
globalParameters["PrintSolutionRejectionReason"] = False  # when a solution is marked as invalid, print why
globalParameters["LibraryFormat"] = "yaml"                # set library backend (either yaml or msgpack)

# True/False: CSV will/won't export WinnerGFlops, WinnerTimeUS, WinnerIdx, WinnerName.
# TODO - if no side-effect, we can set default to True. This can make analyzing "LibraryLogic" (AddFromCSV) faster
globalParameters["CSVExportWinner"] = False

# (When NumBenchmarks > 1). True: CSV will merge the rows of same Problem-ID. False: Each problem will write out "NumBenchmarks" rows
#   In old client - No effect, since in old client, CSV file only exports the last benchmark, somehow is not correct because the previous benchmarks are discarded
#   In new client - csv file exports "NumBenchmarks" rows for every problem. This also make the later analyzing slower
#                   Set this to "True" can merge the rows for same problem, hence can reduce the csv file size and speed up the later analyzing
# TODO - if side-effect, we can set default to True. This can make "getResults()" / "AddFromCSV()" faster
globalParameters["CSVMergeSameProblemID"] = False

# how to initialize tensor data
# serial-in-u will use a sequence that increments in the K dimension
# This is a predictable patterns that can be checked as the kernel runs to detect
# when the wrong data is being used.
# trig_float initializes with the sin function to have non-zero values in the mantissa
# and exponent. It cannot be used for int8 or int32. Need to use tensileAlmostEqual
# not tensileEqual for checking the result.
# NOTE- the following comments explaining the DataInitType are for OldClient and should be obsolete.
#       For NewClient, please read ClientWriter.py, the DataInitName(Enum)
#       - Problem-Independent: 0=0, 1=1, 2=2, 3=rand, 4=Nan, 5=Infinity, 6=BadInput(Nan), 7=BadOutput(Inf), 16=RandomNarrow
#       - Problem-dependent: 8=SerialID, 9=SerialDim0, 10=SerialDim1, 11=Identity, 12~15= Cos/Sin, Abs or Not
#       For A, B, C, D: All the InitMode (0~16) can be used
#       For Alpha/Beta: Only problem-independent init (0~7, 16) can be used,
#                       problem-dependent init (8~15) would cause a exception (Invalid InitMode) in New Client
globalParameters["DataInitTypeAB"] = 3            # 0=0, 1=1, 2=serial, 3=rand, 4=NaN, 5=serial-in-u, 6=trig_float.  Can be overridden by the DataInitTypeA or DataInitTypeB.  Eventually DataInitTypeAB will be retired.
globalParameters["DataInitTypeA"] = -1            # 0=0, 1=1, 2=serial, 3=rand, 4=NaN, 5=serial-in-u, 6=trig_float.  -1 uses value from DataInitTypeAB
globalParameters["DataInitTypeB"] = -1            # 0=0, 1=1, 2=serial, 3=rand, 4=NaN, 5=serial-in-u, 6=trig_float.  -1 uses value from DataInitTypeAB
globalParameters["DataInitTypeC"]  = 3            # 0=0, 1=1, 2=serial, 3=rand, 4=Na, 5=serial-in-uN, 6=trig_float.
globalParameters["DataInitTypeD"]  = 0            # 0=0, 1=1, 2=serial, 3=rand, 4=Na, 5=serial-in-uN, 6=trig_float.
globalParameters["DataInitTypeAlpha"] = 2         # 0=0, 1=1, 2=2, 3=rand, 4=NaN
globalParameters["DataInitTypeBeta"] = 2          # 0=0, 1=1, 2=2, 3=rand, 4=NaN
globalParameters["CEqualD"] = False               # Set to true if testing for the case where the pointer to C is the same as D.
globalParameters["BufferOffsetA"] = 0             # data offset of buffer A
globalParameters["BufferOffsetB"] = 0             # data offset of buffer B
globalParameters["BufferOffsetC"] = 0             # data offset of buffer C
globalParameters["BufferOffsetD"] = 0             # data offset of buffer D

# build parameters
globalParameters["CMakeCXXFlags"] = ""            # pass flags to cmake
globalParameters["CMakeCFlags"] = ""              # pass flags to cmake
globalParameters["DebugKernel"] = False           # assembly only, kernel gets buffer for debug "printing"; kernel writes data to memory, gets coppied to host and printed
globalParameters["LibraryPrintDebug"] = False     # solutions will print enqueue info when enqueueing a kernel

# debug for assembly
globalParameters["EnableAsserts"] = False         # Enable assembly debug assert
globalParameters["EnableDebugA"] = False          # Enable / Disable CheckValue1A
globalParameters["EnableDebugB"] = False          # Enable / Disable CheckValue1B
globalParameters["EnableDebugC"] = False          # Enable / Disable CheckValueC
globalParameters["ExpectedValueC"] = 16.0         # Expected C Value when CheckValueC, debug for Alpha*A*B
globalParameters["ForceCExpectedValue"] = False   # Force C to "DebugExpectedValueC", debug for global write

# Tensor printing controls:
globalParameters["PrintConvolutionUsage"] = 0      # Print Convolution usage info. 1=tensor fields,2=boilerplate info,4=print tensor mappings for specified ConvProblems
globalParameters["PrintTensorA"] = 0          # Print TensorA after initialization
globalParameters["PrintTensorB"] = 0          # Print TensorB after initialization
globalParameters["PrintTensorC"] = 0          # Print TensorC.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintTensorD"] = 0          # Print TensorD.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintTensorRef"] = 0          # Print reference tensor.  0x1=after init; 0x2=after copy-back; 0x3=both
globalParameters["PrintIndexAssignments"] = 0      # Print the tensor index assignment info
globalParameters["PrintWinnersOnly"] = False      # Only print the solutions which become the fastest
globalParameters["PrintCodeCommands"] = False  # print the commands used to generate the code objects (asm,link,hip-clang, etc)
globalParameters["DumpTensors"] = False        # If True, dump tensors to binary files instead of printing them.

# TODO - remove this when NewClient is mainstream
globalParameters["OldClientSourceTmp"] = True      # Use an intermediate sourceTmp dir to detect file changes and minimize rebuilds on old client

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

globalParameters["MaxFileName"] = 64              # If a file name would be longer than this, shorten it with a hash.
globalParameters["SupportedISA"] = [(8,0,3), (9,0,0), (9,0,6), (9,0,8), (9,0,10), (10,1,0), (10,1,1), (10,1,2), (10,3,0)] # assembly kernels writer supports these architectures

globalParameters["GenerateManifestAndExit"] = False               # Output manifest file with list of expected library objects and exit
globalParameters["ClientBuildPath"] = "0_Build"                   # subdirectory for host code build directory
globalParameters["NewClient"] = 2                                 # 1=Run old+new client, 2=run new client only (All In)
globalParameters["BenchmarkProblemsPath"] = "1_BenchmarkProblems" # subdirectory for benchmarking phases
globalParameters["BenchmarkDataPath"] = "2_BenchmarkData"         # subdirectory for storing final benchmarking data
globalParameters["LibraryLogicPath"] = "3_LibraryLogic"           # subdirectory for library logic produced by analysis
globalParameters["LibraryClientPath"] = "4_LibraryClient"         # subdirectory for building example library client
globalParameters["ClientExecutionLockPath"] = None                # Path for a file lock to ensure only one client is executed at once.  filelock module is required if this is enabled.
globalParameters["LibraryUpdateFile"] = ""                        # File name for writing indices and speeds suitable for updating an existing library logic file
globalParameters["LibraryUpdateComment"] = False                  # Include solution name as a comment in the library update file

# internal, i.e., gets set during startup
globalParameters["CurrentISA"] = (0,0,0)
globalParameters["ROCmAgentEnumeratorPath"] = None      # /opt/rocm/bin/rocm_agent_enumerator
globalParameters["ROCmSMIPath"] = None                  # /opt/rocm/bin/rocm-smi
globalParameters["AssemblerPath"] = None                # /opt/rocm/hip/bin/hipcc
globalParameters["WorkingPath"] = os.getcwd()           # path where tensile called from
globalParameters["IndexChars"] =  "IJKLMNOPQRSTUVWXYZ"  # which characters to use for C[ij]=Sum[k] A[ik]*B[jk]
globalParameters["ScriptPath"] = os.path.dirname(os.path.realpath(__file__))            # path to Tensile/Tensile.py
globalParameters["SourcePath"] = os.path.join(globalParameters["ScriptPath"], "Source") # path to Tensile/Source/
globalParameters["HipClangVersion"] = "0,0,0"

# default runtime is selected based on operating system, user can override
if os.name == "nt":
  globalParameters["RuntimeLanguage"] = "OCL"
else:
  globalParameters["RuntimeLanguage"] = "HIP"

globalParameters["CodeObjectVersion"] = "V3"
globalParameters["CxxCompiler"] = "hipcc"
globalParameters["Architecture"] = "all"

# might be deprecated
globalParameters["EnableHalf"] = False
globalParameters["ClientArgs"] = ""
globalParameters["NewClientArgs"] = ""
globalParameters["PackageLibrary"] = False
globalParameters["LegacyComponents"] = True

# perf model
globalParameters["PerfModelL2ReadHits"] = 0.0
globalParameters["PerfModelL2WriteHits"] = 0.15
globalParameters["PerfModelL2ReadBwMul"] = 2
globalParameters["PerfModelReadEfficiency"] = 0.85

# limitation for training
globalParameters["MaxWorkspaceSize"] = 32 * 1024 * 1024 # max workspace for training (32M)
globalParameters["MinKForGSU"] = 256 # min K size to use GlobalSplitU algorithm (only for HPA now)

# control if a solution is run for a given problem
globalParameters["GranularityThreshold"] = 0.0

# directory where custom kernels are located
globalParameters["CustomKernelDirectory"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CustomKernels")

globalParameters["PristineOnGPU"] = True # use Pristine memory on Tensile trainning verification or not

# directory where custom kernels are located
globalParameters["CustomKernelDirectory"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CustomKernels")

# Save a copy - since pytest doesn't re-run this initialization code and YAML files can override global settings - odd things can happen
defaultGlobalParameters = deepcopy(globalParameters)

# Translate GPU targets to filter filenames in Tensile_LOGIC directory
architectureMap = {
  'all':'_','gfx000':'none', 'gfx803':'r9nano', 'gfx900':'vega10',
  'gfx906':'vega20', 'gfx906:xnack+':'vega20', 'gfx906:xnack-':'vega20',
  'gfx908':'arcturus','gfx908:xnack+':'arcturus', 'gfx908:xnack-':'arcturus',
  'gfx90a':'aldebaran', 'gfx90a:xnack+':'aldebaran', 'gfx90a:xnack-':'aldebaran',
  'gfx1010':'navi10', 'gfx1011':'navi11', 'gfx1012':'navi12', 'gfx1030':'navi21'
}

def getArchitectureName(gfxName):
  if gfxName in architectureMap:
    return architectureMap[gfxName]
  else:
    for archKey in architectureMap:
      if gfxName in archKey:
        return architectureMap[archKey]
    return None

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

validThreadTileSides = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + list(range(20, 256, 4))
validThreadTiles = []
for i in validThreadTileSides:
  for j in validThreadTileSides:
    validThreadTiles.append([i, j])

validActivationFormats = ('NCHW', 'NHWC', 'CNHW', 'NCDHW', 'NDHWC', 'CNDHW')
validWeightFormats = ('KCYX', "KYXC", "CKYX", "CYXK",  'KCZYX', 'CKZYX', 'CZYXK')
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
validMFMA["D"] = [[16,16,4,1], [4,4,4,4]]
validMFMA["B1k"] = [[32,32,4,2], [32,32,8,1], [16,16,4,4], [16,16,16,1], [4,4,4,16]]
validMFMA["C"] = validMFMA["S"]
validMFMA["I8"] = validMFMA["4xi8"]
validTT = 16
validMFMA["_format9"] = []
for MFMA in [validMFMA["H"], validMFMA["S"], validMFMA["B"], validMFMA["4xi8"]]:
  for MI in MFMA:
    for bm in range(int(math.log(MI[3],2))+1):
      for tt0 in range(1,validTT+1):
        for tt1 in range(1,validTT+1):
          for wave_m in range (3):
            for wave_n in range(3):
              validMFMA["_format9"].append([MI[0],MI[1],MI[2],MI[3],2**bm,tt0,tt1,2**wave_m, 2**wave_n])
validMatrixInstructions = [[], [-1]] + validMFMA["H"] + validMFMA["S"] + validMFMA["B"] + validMFMA["4xi8"] + validMFMA["D"] + validMFMA["B1k"]
validMatrixInstructions = validMatrixInstructions + validMFMA["_format9"]

# The supported typed GEMM, each entry is (Ti, To, Tc).
# This is used in SolutionStruct.py::checkIfSupportedGEMMType()
validGEMMTypes = [ ('D','D','D'), ('S','S','S'), ('Z','Z','Z'), ('C','C','C'), \
                  ('H','H','H'), ('H','H','S'), ('H','S','S'), \
                  ('B','B','S'), ('B','S','S'), \
                  ('4xi8','I','I'), \
                  ('I8','I','I')]

# These type are newly supported and we would like to use a better file naming for them: _TiToTc_
# For the rest of the typed, we keep them with old existing naming.
typesUsingNewNaming = [ ('H','H','S'), ('H','S','S'), ('B','S','S'),('I8','I','I')]

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

    # original global read to lds is interlace, [w0,w1,w2,w3,w0,w1,w2,w3,w0,w1,w2,w3,w0,w1,w2,w3]
    # when WaveSeparateGlobalRead is enabled, LDS is divided to number of waves part.
    # each wave load a block memory to lds,     [w0,w0,w0,w0,w1,w1,w1,w1,w2,w2,w2,w2,w3,w3,w3,w3]
    # -1 is selected by logic, 0 disable, 1 enable.
    "WaveSeparateGlobalReadA":    [ 0, 1 ],
    "WaveSeparateGlobalReadB":    [ 0, 1 ],

    # PrefetchGlobalRead = 1:
    # Requires 2X LDS space, and VGPRs for buffering data on way into LDS
    #   prefetch / double-buffer reads from global memory -> vgprs -> lds.
    #
    # PrefetchGlobalRead = 2:
    # Do another prefetch while writing data from vgpr to lds.
    #   prefetch / double-buffer reads from global memory -> vgprs --> lds.
    #                                                              |-> prefetch reads
    "PrefetchGlobalRead":         [ 0, 1, 2 ],

    # number of iteration prefetch local reads from lds to VGPRs buffer = PLR % LoopIter
    # number of VGPRs buffer = min(PLR+1,LoopIters)
    # LoopIters = DepthU / LocalSplitU
    # (LoopIters /= MatrixInstruction_K)
    # ex. MT64x128x16_MI32x32x4x2_PLR1, we'll have 4 LoopIters, prefetch read 1 iteration, with 2 VGPRs buffer (2=min(1+1,4))
    #     befor loop:       plr[0]
    #           loop: iter0:plr[1] MAC_r[0], iter1:plr[0] MAC_r[1], iter2:plr[1] MAC_r[0], iter3:plr[0] MAC_r[1]
    #   no load loop: iter0:plr[1] MAC_r[0], iter1:plr[0] MAC_r[1], iter2:plr[1] MAC_r[0], iter3:       MAC_r[1]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR3, we'll have 4 LoopIters, prefetch read 3 iteration, with 4 VGPRs buffer (4=min(3+1,4))
    #     befor loop:       plr[0] plr[1] plr[2]
    #           loop: iter0:plr[3] MAC_r[0], iter1:plr[0] MAC_r[1], iter2:plr[1] MAC_r[2], iter3:plr[2] MAC_r[3]
    #   no load loop: iter0:plr[3] MAC_r[0], iter1:       MAC_r[1], iter2:       MAC_r[2], iter3:       MAC_r[3]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR5, we'll have 4 LoopIters, prefetch read 5%4=1 iteration, with 4 VGPRs buffer (4=min(5+1,4))
    #     befor loop:       plr[0]
    #           loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:plr[0] MAC_r[3]
    #   no load loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:       MAC_r[3]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR5_LRVW8, we'll have 4 LoopIters, prefetch read 5%4=1 iteration, with 4 VGPRs buffer (4=min(5+1,4)) , each read read 2 iterations
    #     befor loop:       plr[0:1]
    #           loop: iter0:plr[2:3] MAC_r[0], iter1: MAC_r[1], iter2: MAC_r[2], iter3:plr[0:1] MAC_r[3]
    #   no load loop: iter0:plr[2:3] MAC_r[0], iter1: MAC_r[1], iter2: MAC_r[2], iter3:         MAC_r[3]
    #
    # ex. MT64x128x16_MI32x32x4x2_PLR7, we'll have 4 LoopIters, prefetch read 7%4=3 iteration, with 4 VGPRs buffer (=min(7+1,4)) --> Exactly the same as PLR3
    #     befor loop:       plr[0]
    #           loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:plr[0] MAC_r[3]
    #   no load loop: iter0:plr[1] MAC_r[0], iter1:plr[2] MAC_r[1], iter2:plr[3] MAC_r[2], iter3:       MAC_r[3]
    "PrefetchLocalRead":          list(range(128+1)),

    # We use double LDS buffer when PrefetchGlobalRead.
    # While it reads data from LDS[0]/[1], it prefetch global data and writes to LDS[1]/[0]
    # If we can make sure all data are read from LDS to register before writing data to LDS, we can use 1 LDS buffer to save LDS memory.
    # this can help to generate Kernel that LDS usage originally exceed MaxLDS if using double LDS buffer,
    # or help to increase Occupancy.
    #     1 means: Force to use 1 LDS Buffer even with PrefetchGlobalRead
    #    -1 means: generator will use 1 LDS buffer only when LDS exceed MaxLDS
    # Use case:
    #    SIA2: 1LDSBuffer is set to 1 natively
    #    SIA3: 1LDSBuffer works only when PGR=True
    # TODO: optimize scheduling to support more cases.
    "1LDSBuffer": [-1 ,0, 1],

    # Split the unroll summation into multiple sections and combine the sections
    # GSU applies only to the unroll summation dimension
    "GlobalSplitU":               list(range(1, 1024+1)),

    # choose how to do GlobalSplitU
    # 1: use atomic operation to accumlate on one buffer
    # 2: each GSU group write to each own buffer and accumulate by another kernel
    "GlobalSplitUAlgorithm":      ["SingleBuffer", "MultipleBuffer"],

    # When splitting up the summation between workgroups, there are two options for organizing which workgroup will do what
    # If we begin with N workgroups and set GSU=4, there will now be 4N workgroups
    # GSUWGMRR=False means workgroup 0,1,2,3 will all work on the same tile; =True means workgroup 0, N-1, 2N-1, 3N-1 will all work on the same tile
    "GlobalSplitUWorkGroupMappingRoundRobin":     [ False, True ],
    # GSUSARR=False means the 4 workgroups do whole chunks of the summation: k=0 -> K/4-1, k=K/4 -> 2K/4-1, k=2K/4 -> 3K/4-1, k=3K/4 -> 4K/4-1
    # GSUSARR=True means the 4 workgroups round robin split up the chunks of the summation: k=0 -> DU-1, 4DU -> 5DU-1, ...; k=1DU -> 2DU-1, 5DU -> 6DU-1...; ...
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
    "ScheduleIterAlg":            [0, 1, 2, 3],

    # Optimizing Local Write Vmcnt in PreLoop when PGR is on, especially for PAP
    # 0: no optimization, force wait vmcnt 0
    # 1: do optimization, in PAP, this can avoid ds_write waiting for previous global store
    # Can always be True, set to False for debugging or comparison
    "OptPreLoopVmcnt":            [False, True],

    # LDD Support
    # Allow LDD and StrideD to != LDC and StrideC for LDD <= LDC and LDD == M
    # TODO: remove. legacy logic yaml in rocblas contains true and false for this parameter
    # remove this parameter will cause two kernels have same.
    # so we can't remove it until we clean logic yaml in rocblas
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
    #    - Requirements for DirectToLds=1:
    #      GlobalLoadVectorWidth? = 1
    #      TransposeLDS = 1 for TLU=0 case
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
    #  = UseInstOffsetForGRO=1:
    #    + Attempt to use Instruction offset for Global Read Offsets.
    #    + This feature avoid updating m0 for subsequent GRO(s) for directToLds feature
    #    - Requirements for UseInstOffsetForGRO=1:
    #      - BufferLoad=1
    #      - DirectToLds=1

    #  converting m0 update from LocalWriteAddrSGpr using  is usually win
    # -1 attempt to use a hueristic to determine when the tile size will use too many SGPR and fall back to VGPR
    "UseInstOffsetForGRO":              [ -1, 0, 1],


    # Converting VGPR GRO into SGPR GRO is usually a win
    # However, the mode may exhaust all available SGPR, in particular for large unroll
    # -1 attempt to use a hueristic to determine when the tile size will use too many SGPR and fall back to VGPR
    "UseSgprForGRO":              [ -1, 0, 1],

    # Some work-items in the group may not participate in the final buffer load.
    # Allows more flexibility in choosing DepthU.
    # 1= allocate extra addressing vpgr for edge cases
    # 2= use temp vgpr inside unroll loop, may save 1 VPR if both A and B have a fractional edge but costs v_alu
    "FractionalLoad":             [ 0, 1, 2] ,

    # Use a 64-bit shadow limit register to allow buffers larger than 2^32 bytes
    "Use64bShadowLimit":   [ True, False],

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


    # Assertions/Predicates that require stride to be specified value.
    # Dictionary of pairs of {position:constValue}
    # Unlike SetConstStride*, these use a position in the IndexAssignments* field:
    #   EX: "{2:0}"  means IndexAssignmentsB[2] must be 0 to run the solution.
    # Use this syntax to specify multiple Fork values in a YAML config file.

    #- AssertStrideAEqual:
    #  - {5: 2, 6: 2} # these are two AssertStrideAEqual predicates for the same solution.
    #  - {5: 2}       # this is a second solution generated with a single predicate.

    # Like other assertions, these are used when kernel is generated and checked before running kernel.
    "AssertStrideAEqual":  -1,

    "AssertStrideBEqual":  -1,

    "AssertStrideCEqual":  -1,
    "AssertStrideDEqual":  -1,

    # Assertions that require stride to be specified value.
    # Dictionary of pairs of {index, constValue}.
    # Index is a member of the global index assignments.
    "AssertSizeEqual":       -1,
    "AssertSizeGreaterThan": -1,
    "AssertSizeLessThan":    -1,
    "AssertSizeMultiple":    -1,

    #Assert values for alpha and beta
    "AssertBetaValue":       [False, 1, -1],
    "AssertAlphaValue":      [False, 1, -1],

    #Assert C==D
    "AssertCEqualsD": [False, True],

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
    # 3: Use wgSerial, wgSerial = wg0 + wg1 * nwg0 + wg2 * (nwg0 * nwg1)
    # 4: Debug mode, offset each tile max allowed StaggerU.  This just moves hotspot
    #    to a different bank since all workgroups still start at same point.
    "StaggerUMapping":       [0,1,2,3,4],


    # 0=don't use magic div (source only)
    # 1=magic div alg #1.  Slightly faster but limited range (if magic number is 2^32)
    # 2=magic div alg#2.  Slightly slower but handles all unsigned ints up to 2^32
    "MagicDivAlg":       [0,1,2],

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

    # Which instruction to use for MAC: MAD or FMA
    "MACInstruction":             ["MAD", "FMA"],
    "WavefrontSize":              [32, 64],

    # MatrixInstruction: (M x N x K x B)
    # XDLOPS tile definition, only valid for gfx908, gfx90a
    # MxNxKxB specifies matrix instruction variants
    #  MxNxB determines the shape of the C tile each instruction worked on
    #      K determines the unroll depth
    # If empty, do not use these instructions
    #
    # Alternative format: (M x N x K x B x MIBlockM x WaveTileM x WaveTileN x WaveM x WaveN)
    # (Note: MxN means M-by-N in the following comments)
    # MIBlockM determines how many blocks along M dimension for multi-block MI variants. Concrete examples:
    #  - MI 16x16x1x4 (4-block variant) with MIBlockM=4 -> (16x16)*(4x1)=64x16 tile per instruction executed
    #  - MI 32x32x1x2 (2-block variant) with MIBlockM=1 -> (32x32)*(1x2)=32x64 tile per instruction executed
    # WaveTileM/N are dimensions of the C tile each wave works on, and is close to the concept of ThreadTile in classic VALU kernels
    #  - WT 4x1 -> each wave executes 4x1 matrix instructions on the C tile of total area (4*MITileM)x(1*MITileN)
    # WaveM/N are dimensions of waves spawned for one workgroup where each wave consists of 64 threads
    #  - Wave2x2 -> a total of 4 waves in one workgroup of shape 2x2
    # Putting it all together:
    #  - [32, 32, 1, 2,  1,  4, 1,  2, 2]
    #     ^^^^^^^^^^^^   ^   ^^^^   ^^^^
    #      MatrixInst  BlkM   WT    Wave
    #  - means (32x64) per MI * (4x1) per wave * (2x2) per workgroup = (32*4*2)x(64*1*2) = 256x128 macro tile
    # Tensile will ignore the parameters ThreadTile and WorkGroup when the alternative format is used
    "MatrixInstruction":          validMatrixInstructions,

    # StoreRemap: Optimize MatrixInstruction store patterns to enhance performance.
    #             MI output data between each threads are along N dims.
    #             But global memory is along M dim continous.
    #             That mean global write between each threads are not continous.
    #             Therefore, store performance for MI instruction is poor.
    # How StoreRemap works in final store stage:
    #             1. Put all thread output data into LDS.
    #             2. All thread read data from LDS along M dims.
    #                (match global Memory continous direction)
    #             3. All thread write out data into global memory.
    # 0:   Disable StoreRemap (default)
    # 1~8: Enable StoreRemap and set the global write vector width
    # Suggest optimum value: fp32 = [2,4], fp16 or bf16 = [4,8] (dwordx2 and dowrdx4)
    # -1:  Use dwordx2 if support SRVW, or set SRVW to 0
    "StoreRemapVectorWidth":      [-1,0,1,2,4,8],

    # SourceSwap: Optimizes MatrixInstruction store pattern by swapping mfma input order.
    "SourceSwap":                 [False, True],

    # Disable overlapping AB-tile vgpr and read/write addr vgprs with C-tile vgprs
    # Valid only for MatrixInstruction enabled kernels, which by default overlaps
    # C-tile w/ AB-tile until it's due for v_accvgpr_read before the writeback. Illustrated below:
    # |<----------------------- valuC ----------------------->|
    # |<--- valuA/B --->|<-- R/W pointers -->|xxx|<- Spares ->|
    #                                          ^        ^
    #         (Reserved by persistent kernels) ^        ^
    #                       (Utilized by register pool) ^
    "DisableVgprOverlapping":     [False, True],

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
    # -1 : Automatically choose a "heuristic" value that can possibly get a better gain: (TilesPerWorkgroup = 1~2)
    #      Not based on any theory, but on some experiment observation, can be used to reduce the kernels
    #      Recommand [-1,0,1] for basic tuning
    # Assertions/Requirements: NumWorkGroups0 * NumWorkGroups1 < 2^32
    "PersistentKernel":           range(-1,512+1) ,       # Use persistent kernel.

    # True:  Batch dimension (WG.z) is also considered in persistent kernel
    # False: Not considered
    #        for problems with large batch-size, PKAB = True could help
    #        for problems with only one batch, PKAB = True/False should make no difference
    "PersistentKernelAlongBatch": [False,True],

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

    # Pack summation dims
    # If 0, a for loops are generated for each summation dimension.
    # If 1, summation dims are packed into a single loop and extracted as needed using mod/shift.  The innermost summation
    #  dimension must be an integer multiple of the unroll loop - in other words the load tile is contiguous in memory.
    #  In this mode, tensile can still prefetch data across the load tile dimension.
    # If 2, summations dims are packed into a single loop as above.  In addition, the load tile does not need to be
    #  contiguous in memory and can span summation dimensions. (not supported yet)
    "PackSummationDims":         [0,1],

    # debug mode, uses the PackSummationDims method to increment the unroll loop counter
    "UnrollIncIsDepthU":         [0,1],

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
    # NOTE: for input bpe=32, max GRVW is 4  (to fit dwordx4) (FP32), min GRVW is 1 (dword)
    #                 bpe=16, max GRVW is 8  (to fit dwordx4) (FP16), min GRVW is 2 (dword)
    #                 bpe=8,  max GRVW is 16 (to fit dwordx4) (INT8), min GRVW is 4 (dword)
    "GlobalReadVectorWidth":      [ -1, 1, 2, 3, 4, 6, 8, 16 ],

    # Controls desired width (#elements) for loads from LDS -> VGPR.
    # -1 : Set LocalReadVectorWidth =  VectorWidth
    #  1 cannot be used for half type.
    # used in combination with TransposeLDS=True
    # in TransposeLDS=1 case, use wider load to fetch elements in summation dimension from LDS
    # helps optimizing instruction scheduling between MFMA and nonMFMA instructions
    # NOTE: for input bpe=32, max LRVW is 4  (to fit ds_read_b128) (FP32)
    #                 bpe=16, max LRVW is 8  (to fit ds_read_b128) (FP16)
    #                 bpe=8,  max LRVW is 16 (to fit ds_read_b128) (INT8)

    "LocalReadVectorWidth":      [ -1, 1, 2, 4, 8, 16 ],

    # threads should read/write/operate on this many contiguous elements from the C matrix.
    # If VW=4 then thread0 will process 4 consec C elements, then thread1 next 4, etc.
    # If the ThreadTile is > VectorWidth then thread0 will next operate on the 4 elements in C at (4*NumThreads)
    # Typically the load vector width and store vector width are directly related to the VW.
    # The global load width is closely related to the width of local stores so
    # GlobalReadVectorWidth also controls local write width.
    # Local read width also matches since VectorWidth consec elements must be read
    # Typically matching 16 bytes is good choice since the stores will be optimally coalesced with 16 bytes/WI.
    # -1 means use the largest vector width up to 128 bits.
    # Using a VW too large which results in >16bytes/thread isn't supported
    "VectorWidth":                [ -1, 1, 2, 3, 4, 6, 8 ],

    # If 0, store 1 element per instruction.
    # If 1, store vector-width elements per instruction.
    # if -1, store vector-wide elements per instruction unless PBD would not generate a valid kernel
    "VectorStore":                    [-1, 0, 1],

    # Controls desired width (#elements) for stores from reg to global memory.
    # When MatrixInstruciton == None, derived parameter gwvw takes precedence.
    # -1 : Set StoreVectorWidth = VectorWidth
    "StoreVectorWidth":           [ -1, 1, 2, 3, 4, 6, 8 ],

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

    # DepthULdsDivisor determines how we pipeline the data from global memory to LDS
    # Instead of moving all in-flight data from the register buffer (G2L) to the LDS at once, we divide the G2L buffer into N portions and
    # write each portion of the G2L to LDS, read from LDS and do the actual matrix multiply-accumulate, before moving on to the portion and so on.
    # This helps cut down LDS usage by the value of the divisor. Helps increase CU occupancy or DepthU if kernel was previously LDS limited.
    #
    # The premise of this parameter is to be able to fetch all 256B (equivalent to 128 half's or 64 single's) in a TN laid-out problem size,
    # maximizing L2 channel efficiency, therefore this parameter only works for TN buffer layout
    #
    # Implementation-wise, for now it only supports ScheduleIterAlg=3 and TransposeLDS=1
    # In addition, it does not work with DirectToLds=1 because it needs the in-flight data to reside in registers
    "DepthULdsDivisor":           [1, 2], # [4, 8] not tested yet

    # integer ammount of padding to put into LDS, in 2016 this didn't seem to help performance, profilers were showing that channel conflicts weren't really hurting
    # performance so this has been deprecated and probably doesn't work
    # -1 means use same padding as the VectorWidth if TLU=0 else 0.  (Padding only helps when transpose is required)
    # With MatrixInstruciton: -1 means max(GRVW,MIInput) if TLU=0
    "LdsPadA":                     [ -1, 0, 1, 2, 3, 4, 8, 16],
    "LdsPadB":                     [ -1, 0, 1, 2, 3, 4, 8, 16],

    # Padding boundary for LDS. defines block-size for pad insertion. for every 'LdsBlockSizePerPad' bytes, LDS padding (pad value from LdsPad parameter)
    # is added (readOffset aware of the pad and adjusts offset value based on this parameter value).
    # Only support LdsBlockSizePerPad >= unrollDepth * BPE
    # 0 means disable LdsBlockSizePerPad,
    # -1 means round up to nearest power of 2 begin with 128
    "LdsBlockSizePerPad":          [-1, 0, 64, 128, 256, 512],

    # Transpose LDS format. Local store in Coalsced dimension , same as optimized global fetch dimension . applicable only in TLU=0 case for miSIMD(s)
    # TODO: No code for -1 ?
    "TransposeLDS":                [-1, 1, 0],

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
    "InnerUnroll":                [1,2,4,8,16,32,64],

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

    # Name of the custom kernel located in globalParameters["CustomKernelDirectory"].
    # a custom kernel is a user written assembly kernel with its associated configuration parameters included in a custom.config section
    # inside the yaml block between the --- and ... markers.  These parameters are only used for information purposes, not kernel generation.
    # Ex:
    # custom.config:
    #   ProblemType:
    #     OperationType: GEMM
    #     etc...
    #   ThreadTile: [8, 8]
    #   etc...
    #
    # Custom kernels can be included in a BenchmarkProblemSizeGroup by having their name (without file extension) listed under the "CustomKernels"
    # category alongside InitialSolutionParameters, BenchmarkCommonParameters, etc...
    "CustomKernelName":            -1,

    # Will allow a kernel to be accepted even when checks determine it's not viable.
    # Intended for use with custom kernels which have confirmed to be correct
    "NoReject":                    [False, True],

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
    {"LdsBlockSizePerPad":        [ 0 ] },
    {"TransposeLDS":              [ 0 ] },
    {"MaxOccupancy":              [ 40 ] },
    {"VectorWidth":               [ -1 ] },
    {"VectorStore":               [ -1 ] },
    {"StoreVectorWidth":         [ -1 ] },
    {"GlobalReadVectorWidth":     [ -1 ] },
    {"LocalReadVectorWidth":      [ -1 ] },
    {"GlobalReadCoalesceVectorA": [ True ] },
    {"GlobalReadCoalesceVectorB": [ True ] },
    {"WaveSeparateGlobalReadA":    [ 0 ] },
    {"WaveSeparateGlobalReadB":    [ 0 ] },
    {"GlobalReadCoalesceGroupA":  [ True ] },
    {"GlobalReadCoalesceGroupB":  [ True ] },
    {"PrefetchGlobalRead":        [ 1 ] },
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
    {"OptPreLoopVmcnt":           [ True ] },

    {"LdcEqualsLdd":              [ False ] },
    {"InterleaveAlpha":           [ 0 ] },
    {"OptNoLoadLoop":             [ 1 ] },
    {"PrefetchAcrossPersistent":  [ 0 ] },

    {"BufferLoad":                [ True ] },
    {"BufferStore":               [ True ] },
    {"DirectToLds":               [ False ] },
    {"UseSgprForGRO":             [ -1 ] },
    {"UseInstOffsetForGRO":       [ 0 ] },
    {"AssertSummationElementMultiple": [ 1 ] },
    {"AssertFree0ElementMultiple": [ 1 ] },
    {"AssertFree1ElementMultiple": [ 1 ] },
    {"AssertMinApproxSize":        [ -1 ] },
    {"AssertStrideAEqual":        [ {} ] },
    {"AssertStrideBEqual":        [ {} ] },
    {"AssertStrideCEqual":        [ {} ] },
    {"AssertStrideDEqual":        [ {} ] },
    {"AssertSizeEqual":           [ {} ] },
    {"AssertSizeGreaterThan":     [ {} ] },
    {"AssertSizeMultiple":        [ {} ] },
    {"AssertSizeLessThan":        [ {} ] },
    {"AssertAlphaValue":          [ False ]},
    {"AssertBetaValue":           [ False ]},
    {"AssertCEqualsD":            [ False ]},
    {"CheckTensorDimAsserts"      : [ False ] },
    {"CheckDimOverflow"           : [ 0 ] },

    {"StaggerU":                  [ 32 ] },   # recommend [0,32]
    {"StaggerUStride":            [ 256 ] },  # recommend 256 for V10,V20
    {"StaggerUMapping":           [ 0 ] },    # recommend [0,1]
    {"MagicDivAlg":               [ 2 ] },
    {"GlobalSplitU":              [ 1 ] },
    {"GlobalSplitUAlgorithm":     [ "MultipleBuffer" ] },
    {"GlobalSplitUSummationAssignmentRoundRobin": [ True ] },
    {"GlobalSplitUWorkGroupMappingRoundRobin":    [ False ] },
    {"MacroTileShapeMin":         [ 1 ] },
    {"MacroTileShapeMax":         [ 64 ] },
    {"PersistentKernel":          [ 0 ] },
    {"PersistentKernelAlongBatch":[ False ] },    # May be default True is better ?
    {"PackBatchDims":             [ 0 ] },
    {"PackFreeDims":              [ 1 ] },
    {"PackSummationDims":         [ 0 ] },
    {"UnrollIncIsDepthU":         [ 0 ] },
    {"PackGranularity":           [ 2 ] },
    {"FractionalLoad":            [ 0 ] },
    {"Use64bShadowLimit":         [ 1 ] },
    {"VectorAtomicWidth":         [ -1 ] },
    {"NumLoadsCoalescedA":        [ 1 ] },
    {"NumLoadsCoalescedB":        [ 1 ] },
    {"WorkGroup":                 [ [16,16,1]] },
    {"WorkGroupMappingType":      [ "B" ] },
    {"WorkGroupMapping":          [ 8 ] },
    {"ThreadTile":                [ [4,4] ] },
    {"MACInstruction":            [ '' ]},
    {"WavefrontSize":             [ 64 ]},
    {"MatrixInstruction":         [ [] ] },
    {"DisableVgprOverlapping":    [ False ] },
    {"1LDSBuffer":                [ 0 ] },
    {"DisableAtomicFail":         [ 0 ] },
    {"DisableKernelPieces":       [ 0 ] },
    {"DepthU":                    [ -1 ] },
    {"DepthULdsDivisor":          [ 1 ] },
    {"PerformanceSyncLocation":   [ -1 ] },
    {"PerformanceWaitLocation":   [ -1 ] },
    {"PerformanceWaitCount":      [ -1 ] },
    {"NonTemporalC":              [ 0 ] },
    {"NonTemporalA":              [ 0 ] },
    {"NonTemporalB":              [ 0 ] },
    {"ReplacementKernel":         [ False ] },
    {"CustomKernelName":          [ "" ] },
    {"NoReject":                  [ False ]},
    {"MinVgprNumber":             [0]},
    {"MaxVgprNumber":             [256]},
    {"StoreRemapVectorWidth":     [ 0 ] },
    {"SourceSwap":                [ False ] },
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

    # pack spatial dims (d,h,w) into single tensor dim when possible
    # This is preferred for cases where these dimensions are packed in memory
    # since it reduces addressing overhead and will produce a more efficient kernel
    # Default is 1, multiple dimensions will be created if needed for strides or other cases.
    "PackedSpatialDims",

    # pack filter dims (z,y,x) into single tensor dim when possible.
    # This is preferred for cases where these dimensions are packed in memory
    # since it reduces addressing overhead and will produce a more efficient kernel
    # Default is 1, multiple dimensions will be created if needed for dilations or other cases.
    "PackedFilterDims",

    # If 1:
    #  - Unroll index is the channel index
    #  - if PackSummationDims=0, this is likely highest perf since it provides a larger
    #    iteration count for the unroll loop.
    # If 0:
    #   - Unroll index is filter index (Forward,BackwardData) or spatial index (BackwardWeights)
    #   - provides better cache locality for most formats, but tigher looping.
    #   - Likely a good idea with PackSummationDims=1 since there is only one unroll loop.
    "UnrollOnChannel",

    # Input spatial dimensions (D,H,W)
    # Optional parameter for debug and testing.  This does not impact kernel generation.
    # If set,then each problem dimension size/stride will be checked to ensure they are
    # correctly specified. (TBD)
    # Also used by testbenches to compute consistent strides and sizes for auto-generated
    # problem sizes and strides.
    'Spatial',              # examples 56x56, 7x7.

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
    "ComputeDataType":          0,                # compute data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "UseBeta":                  True,             # =True use beta parameter (asm will check for B=0 and optimize the write for that), =False don't use beta parameter
    "HighPrecisionAccumulate":  False,            # f32 += f16*f16
    "SilentHighPrecisionAccumulate": False,       # Keep kernel names the same for HPA mode.  Useful for testing.

    "ComplexConjugateA":        False,            # complex data should be conjugated for "C" transpose case
    "ComplexConjugateB":        False,

    # for OperationType == GEMM
    "TransposeA":               False,            # =True means transA="T" or "C", =False means transA = "N"
    "TransposeB":               True,
    "Batched":                  False,            # add batching dimension
    "StridedBatched":           True,             # use to select general batch or strided batch

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

    # use initial strides for AB.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - GlobalReadWidth must be 1 (since elements are not guaranteed to be adjacent in memory)
    "UseInitialStridesAB":      False,

    # use initial strides for CD.
    # This has some performance impact for the increased flexibility:
    #   - Additional strides will be passed into the kernel and will occupy SGPR registers
    #   - Additional multiply on the store address path
    #   -VectorStore must be 0.  If VectorStore is -1, it will be silently set to 0 internally.
    "UseInitialStridesCD":      False,

    "AllowNoFreeDims":          False,  # allow A or B to specify no free dims
                                        # (if false, A and B must have at least one free dim)
                                        # (if true, A and B must have at least one free or batch dim)

    # SetConstStride* sets the specified stride in the problem.
    # These no longer generate predicates - see AssertStrideEqualA/B below
    # List of pairs of [index, constValue].
    # Index is a member of the global index assignments (not an offset into IndexAssignmentsA/B)
    # EX: SetConstStrideA: [ [3, 1], [2, 4] ] sets
    #     strideA for index3 to constant '1' and stride for index2 to constant '4'.
    "SetConstStrideA":          [],
    "SetConstStrideB":          [],

    # ZeroPad:
    # Zero-pad will add leading and trailing "pad" elements to the specified 'anchor'
    # dimension when accessed by specified summation dimension.
    #
    # Format is list of tuples of [freeDim, sumDim, padStart, padEnd].
    #  - freeDim is the anchor where the zero-pad starts.
    #  - sumDim is the summation dim to which the padding checking is added.
    #  - padStart is the number of elements to pad before the Start element
    #  - padEnd is the number of elements to pad before the last element.

    # - Terms:
    #   - Start is the first summation element
    #   - FreeSize is the size of the specified free dimension (freeDim)
    #   - SumSize is the size of the specified summation dimension (sumDim)
    # - Pad Ranges:
    #   - Ranges show below are inclusive on the start element and exclusive on the last element.
    #     For example, [0,3) is 0,1,2.
    #    - Elements in the region [Start-padStart, Start) are in the leading pad region and will return 0.
    #    - Elements in the memory region [Start + freeSize + sumSize - padEnd,  Start + freeSize + sumSize)
    #     are in the trailing pad region and will return 0.
    #    - Code actually checks for elementMem < padStart or elementMem>=elementEdge
    #      - elementMem is the memory offset of the element from the tensor base
    #      - elementEdge is FreeSize*FreeStride + (SumSize-1)*SumStride - padEnd
    #        - FreeStride is typically spatial*convolutionStride
    #        - SumStride is typically spatial*dilation
    #        - PadStart and PadStop should be scaled by spatial on the host before calling the kernel.
    #          (spatial is not available inside the kernel)
    #      - The GPU implementations shift the load tile by -padStart, then return 0s for any address <=0.
    #        The elementEdge is also shifted by -padStart.  This allows the global read offset to be used for the
    #        edge comparison.  Edge comparisons are performed with vector instructions so each work-item computes
    #        a different in/out value.
    #      - Multiple summations OR together their edge checks, so any OOB edge returns 0 for the load.
    #
    # - Strides:
    #   - padStart and padStop are passed as kernel arguments. These are scaled by the spatial dim on the host;
    #   - No memory access is performed for elements in the Pad regions.
    #   - The Pad regions are handled by manipulating the tensor addressing and are not visible in actual memory.
    #     For example, a tensor with 2 rows, 16 elements/row, padStart=padEnd=2 occupies 32 elements in memory (not 40)
    #   - Typical use case is to set summationStride < freeSize, with padStart+padEnd+1 == summationStride.
    # - Caveats:
    #  - ZeroPad requires that the ElementEdge <= 2^32:
    #    This is SizeFree+SizeSum + Pad_Leading + PadTrailingPad + padding=GRWW for shift-pointer) bytes < 2^32
    #    Likely this is less than the standard buffer load limits (bottom-right corner of macro-tile)

    #  EX: ZeroPadA: [ [0,1,  2,3]] # TensorA free index 0 with sum index 1 has leading pad=2 and trailing pad=3
    # Note nesting of brackets ; the parm can contain multiple padding tuples.
    #  EX: ZeroPadA: [ [0,1, -1,-1]]# Pads are dynamic and passed as part of the problem.

    "ZeroPadA":                 [], # [ [0,1, 2,3]]
    "ZeroPadB":                 [], # Not fully supported/tested yet

    # Summation dimension indices
    "MirrorDimsA":              [],
    "MirrorDimsB":              [],

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
# rocm-smi, hip-clang, rocm_agent_enumerator, clang-offload-bundler
################################################################################
def isExe( filePath ):
  return os.path.isfile(filePath) and os.access(filePath, os.X_OK)
def locateExe( defaultPath, exeName ): # /opt/rocm/bin, hip-clang
  # look in defaultPath first
  exePath = os.path.join(defaultPath, exeName)
  if isExe(exePath):
    return exePath
  # look in PATH second
  for path in os.environ["PATH"].split(os.pathsep):
    exePath = os.path.join(path, exeName)
    if isExe(exePath):
      return exePath
  return None

def GetAsmCaps(isaVersion):
  """ Determine assembler capabilities by testing short instructions sequences """
  rv = {}
  rv["SupportedISA"]    = tryAssembler(isaVersion, "")
  rv["HasExplicitCO"]   = tryAssembler(isaVersion, "v_add_co_u32 v0,vcc,v0,1")
  rv["HasExplicitNC"]   = tryAssembler(isaVersion, "v_add_nc_u32 v0,v0,1")

  rv["HasDirectToLds"]  = tryAssembler(isaVersion, "buffer_load_dword v40, v36, s[24:27], s28 offen offset:0 lds")
  rv["HasAddLshl"]      = tryAssembler(isaVersion, "v_add_lshl_u32 v47, v36, v34, 0x2")
  rv["HasLshlOr"]       = tryAssembler(isaVersion, "v_lshl_or_b32 v47, v36, 0x2, v34")
  rv["HasSMulHi"]       = tryAssembler(isaVersion, "s_mul_hi_u32 s47, s36, s34")
  rv["HasCodeObjectV3"] = tryAssembler(isaVersion, "", False, "-mcode-object-version=2")

  rv["HasMFMA"]         = tryAssembler(isaVersion, "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]")
  rv["HasMFMA_f64"]     = tryAssembler(isaVersion, "v_mfma_f64_16x16x4f64 v[0:7], v[32:33], v[36:37], v[0:7]")
  rv["HasMFMA_bf16_1k"] = tryAssembler(isaVersion, "v_mfma_f32_32x32x4bf16_1k a[0:31], v[32:33], v[36:37], a[0:31]")

  rv["v_mac_f16"]       = tryAssembler(isaVersion, "v_mac_f16 v47, v36, v34")

  rv["v_fma_f16"]       = tryAssembler(isaVersion, "v_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0,0]")
  rv["v_fmac_f16"]      = tryAssembler(isaVersion, "v_fma_f16 v47, v36, v34")

  rv["v_pk_fma_f16"]    = tryAssembler(isaVersion, "v_pk_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0]")
  rv["v_pk_fmac_f16"]   = tryAssembler(isaVersion, "v_pk_fma_f16 v47, v36, v34")

  rv["v_mad_mix_f32"]   = tryAssembler(isaVersion, "v_mad_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]")
  rv["v_fma_mix_f32"]   = tryAssembler(isaVersion, "v_fma_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]")

  rv["v_dot2_f32_f16"]  = tryAssembler(isaVersion, "v_dot2_f32_f16 v20, v36, v34, v20")
  rv["v_dot2c_f32_f16"] = tryAssembler(isaVersion, "v_dot2c_f32_f16 v47, v36, v34")

  rv["v_mac_f32"]       = tryAssembler(isaVersion, "v_mac_f32 v20, v21, v22")
  rv["v_fma_f32"]       = tryAssembler(isaVersion, "v_fma_f32 v20, v21, v22, v23")
  rv["v_fmac_f32"]      = tryAssembler(isaVersion, "v_fmac_f32 v20, v21, v22")

  rv["HasAtomicAdd"]    = tryAssembler(isaVersion, "buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen offset:0")


  if tryAssembler(isaVersion, "s_waitcnt vmcnt(63)"):
    rv["MaxVmcnt"] = 63
  elif tryAssembler(isaVersion, "s_waitcnt vmcnt(15)"):
    rv["MaxVmcnt"] = 15
  else:
    rv["MaxVmcnt"] = 0

  # TODO- Need to query the max cap, just like vmcnt as well?
  rv["MaxLgkmcnt"] = 15

  rv["SupportedSource"] = True

  return rv

def GetArchCaps(isaVersion):
  rv = {}
  rv["HasEccHalf"]       = (isaVersion==(9,0,6) or isaVersion==(9,0,8) or isaVersion==(9,0,10))
  rv["Waitcnt0Disabled"] = (isaVersion == (9,0,8) or isaVersion==(9,0,10))
  rv["SeparateVscnt"]    = isaVersion[0] == 10
  rv["CMPXWritesSGPR"]   = isaVersion[0] != 10
  rv["HasWave32"]        = isaVersion[0] == 10

  return rv

def tryAssembler(isaVersion, asmString, debug=False, *options):
  """
  Try to assemble the asmString for the specified target processor
  Success is defined as assembler returning no error code or stderr/stdout
  """
  options = list(options)
  if globalParameters["PrintLevel"] >= 2:
    debug = True

  if isaVersion[0] == 10:
    options += ['-mwavefrontsize64']

  args = [globalParameters["AssemblerPath"], '-x', 'assembler',
          '-target', 'amdgcn-amdhsa',
          '-mcpu='+gfxName(isaVersion),
          '-mcode-object-version=3',
          *options,
          '-']

  result = subprocess.run(args, input=asmString.encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  output = result.stdout.decode()

  if debug:
    print("isaVersion: ", isaVersion)
    print("asm_cmd:", ' '.join(args))
    print("asmString: ", asmString)
    print("output: ", output)
    print("return code: ", result.returncode)

  if output != "" or result.returncode != 0:
    return False
  return True

def gfxArch(name):
    import re
    match = re.search(r'gfx(\S{3,})', name)
    if not match: return None

    ipart = match.group(1)

    step = int(ipart[-1], 16)
    ipart = ipart[:-1]

    minor = int(ipart[-1])
    ipart = ipart[:-1]

    major = int(ipart)

    rv = (major, minor, step)

    return rv

def gfxName(arch):
    # convert last digit to hex because reasons
    name = str(arch[0]) + str(arch[1]) + ('%x' % arch[2])
    return 'gfx' + ''.join(map(str,name))

def restoreDefaultGlobalParameters():
  """
  Restores `globalParameters` back to defaults.
  """
  global globalParameters
  global defaultGlobalParameters
  # Can't just assign globalParameters = deepcopy(defaultGlobalParameters) because that would
  # result in dangling references, specifically in Tensile.Tensile().
  globalParameters.clear()
  for key, value in deepcopy(defaultGlobalParameters).items():
    globalParameters[key] = value

def printTable(rows):
  rows = list([[str(cell) for cell in row] for row in rows])
  colWidths = list([max([len(cell) for cell in col]) for col in zip(*rows)])

  for row in rows:
    for (width, cell) in zip(colWidths, row):
      pad = ' ' * (width - len(cell))
      print(pad, cell, sep='', end=' ')
    print()

def printCapTable(parameters):
  import itertools
  archs = [(0,0,0)] + parameters["SupportedISA"]
  gfxNames = list(map(gfxName, archs))

  headerRow = ['cap'] + gfxNames

  def capRow(caps, cap):
    return [cap] + [('1' if cap in caps[arch] and caps[arch][cap] else '0') for arch in archs]

  allAsmCaps = set(itertools.chain(*[caps.keys() for arch, caps in parameters["AsmCaps"].items()]))
  allAsmCaps = sorted(allAsmCaps, key=lambda k: (k.split("_")[-1], k))
  asmCapRows = [capRow(parameters["AsmCaps"], cap) for cap in allAsmCaps]

  allArchCaps = set(itertools.chain(*[caps.keys() for arch, caps in parameters["ArchCaps"].items()]))
  allArchCaps = sorted(allArchCaps)
  archCapRows = [capRow(parameters["ArchCaps"], cap) for cap in allArchCaps]

  printTable([headerRow] + asmCapRows + archCapRows)

################################################################################
################################################################################
def assignGlobalParameters( config ):
  """
  Assign Global Parameters
  Each global parameter has a default parameter, and the user
  can override them, those overridings happen here
  """

  global globalParameters

  # Minimum Required Version
  if "MinimumRequiredVersion" in config:
    if not versionIsCompatible(config["MinimumRequiredVersion"]):
      printExit("Config file requires version=%s is not compatible with current Tensile version=%s" \
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

  globalParameters["ROCmPath"] = "/opt/rocm"
  if "ROCM_PATH" in os.environ:
    globalParameters["ROCmPath"] = os.environ.get("ROCM_PATH")
  if "TENSILE_ROCM_PATH" in os.environ:
    globalParameters["ROCmPath"] = os.environ.get("TENSILE_ROCM_PATH")
  globalParameters["CmakeCxxCompiler"] = None
  if "CMAKE_CXX_COMPILER" in os.environ:
    globalParameters["CmakeCxxCompiler"] = os.environ.get("CMAKE_CXX_COMPILER")

  globalParameters["ROCmBinPath"] = os.path.join(globalParameters["ROCmPath"], "bin")

  # ROCm Agent Enumerator Path
  globalParameters["ROCmAgentEnumeratorPath"] = locateExe(globalParameters["ROCmBinPath"], "rocm_agent_enumerator")
  if "CxxCompiler" in config:
    globalParameters["CxxCompiler"] = config["CxxCompiler"]

  if "TENSILE_ROCM_ASSEMBLER_PATH" in os.environ:
    globalParameters["AssemblerPath"] = os.environ.get("TENSILE_ROCM_ASSEMBLER_PATH")
  elif globalParameters["AssemblerPath"] is None and globalParameters["CxxCompiler"] == "hipcc":
    globalParameters["AssemblerPath"] = locateExe(os.path.join(globalParameters["ROCmPath"], "llvm/bin"), "clang++")

  globalParameters["ROCmSMIPath"] = locateExe(globalParameters["ROCmBinPath"], "rocm-smi")
  globalParameters["ExtractKernelPath"] = locateExe(os.path.join(globalParameters["ROCmPath"], "hip/bin"), "extractkernel")
  globalParameters["ClangOffloadBundlerPath"] = locateExe(os.path.join(globalParameters["ROCmPath"], "llvm/bin"), "clang-offload-bundler")

  if "ROCmAgentEnumeratorPath" in config:
    globalParameters["ROCmAgentEnumeratorPath"] = config["ROCmAgentEnumeratorPath"]

  # read current gfx version
  if os.name != "nt" and globalParameters["CurrentISA"] == (0,0,0) and globalParameters["ROCmAgentEnumeratorPath"]:
    command = [globalParameters["ROCmAgentEnumeratorPath"]]#, "-t", "GPU"]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    for line in result.stdout.decode().split("\n"):
      arch = gfxArch(line.strip())
      if arch is not None:
        if arch in globalParameters["SupportedISA"]:
          print1("# Detected local GPU with ISA: " + gfxName(arch))
          globalParameters["CurrentISA"] = arch
    if globalParameters["CurrentISA"] == (0,0,0):
      printWarning("Did not detect SupportedISA: %s; cannot benchmark assembly kernels." % globalParameters["SupportedISA"])
    if result.returncode:
      printWarning("%s exited with code %u" % (globalParameters["ROCmAgentEnumeratorPath"], result.returncode))

  # TODO Remove this when rocm-smi supports gfx90a
  if globalParameters["CurrentISA"] == (9,0,10):
    printWarning("HardwareMonitor currently disabled for gfx90a")
    globalParameters["HardwareMonitor"] = False

  globalParameters["AsmCaps"] = {}
  globalParameters["ArchCaps"] = {}

  for v in globalParameters["SupportedISA"] + [(0,0,0)]:
    globalParameters["AsmCaps"][v] = GetAsmCaps(v)
    globalParameters["ArchCaps"][v] = GetArchCaps(v)

  if globalParameters["PrintLevel"] >= 1:
    printCapTable(globalParameters)

  globalParameters["SupportedISA"] = list([i for i in globalParameters["SupportedISA"] if globalParameters["AsmCaps"][i]["SupportedISA"]])

  validParameters["ISA"] = [(0,0,0), *globalParameters["SupportedISA"]]

  # For ubuntu platforms, call dpkg to grep the version of hip-clang.  This check is platform specific, and in the future
  # additional support for yum, dnf zypper may need to be added.  On these other platforms, the default version of
  # '0.0.0' will persist

  # Due to platform.linux_distribution() being deprecated, just try to run dpkg regardless.
  # The alternative would be to install the `distro` package.
  # See https://docs.python.org/3.7/library/platform.html#platform.linux_distribution
  try:
    output = subprocess.run(["hipcc", "--version"], check=True, stdout=subprocess.PIPE).stdout.decode()

    for line in output.split('\n'):
      if 'HIP version' in line:
        globalParameters['HipClangVersion'] = line.split()[2]
        print1("# Found  hipcc version " + globalParameters['HipClangVersion'])

  except (subprocess.CalledProcessError, OSError) as e:
      printWarning("Error: {} running {} {} ".format('hipcc', '--version',  e))

  for key in config:
    value = config[key]
    if key not in globalParameters:
      printWarning("Global parameter %s = %s unrecognised." % ( key, value ))
    globalParameters[key] = value

def setupRestoreClocks():
  import atexit
  def restoreClocks():
    if globalParameters["PinClocks"]:
      rsmi = globalParameters["ROCmSMIPath"]
      subprocess.call([rsmi, "-d", "0", "--resetclocks"])
      subprocess.call([rsmi, "-d", "0", "--setfan", "50"])
  atexit.register(restoreClocks)
setupRestoreClocks()

################################################################################
# Assign Parameters
# populate dst with src[key] else give it the default/backup value
################################################################################
def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, \
    defaultDictionary):
  if key in sourceDictionary:
    destinationDictionary[key] = deepcopy(sourceDictionary[key])
  else:
    destinationDictionary[key] = deepcopy(defaultDictionary[key])

# populate dst with src[key] else abort since it's required
def assignParameterRequired(destinationDictionary, key, sourceDictionary):
  if key in sourceDictionary:
    destinationDictionary[key] = deepcopy(sourceDictionary[key])
  else:
    printExit("Parameter \"%s\" must be defined in dictionary %s" % (key, sourceDictionary) )


################################################################################
# Push / Pop Working Path
# store a WorkingPath where to write files (like benchmark files)
################################################################################
def pushWorkingPath( foldername ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  globalParameters["WorkingPath"] = \
      os.path.join(globalParameters["WorkingPath"], foldername )
  return ensurePath( globalParameters["WorkingPath"] )
def popWorkingPath():
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  if len(workingDirectoryStack) == 0:
    globalParameters["WorkingPath"] = \
      os.path.split(globalParameters["WorkingPath"])[0]
  else:
    globalParameters["WorkingPath"] = workingDirectoryStack.pop()
def ensurePath( path ):
  try:
    os.makedirs(path)
  except OSError:
    pass
  return path
def setWorkingPath( fullPathName ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  workingDirectoryStack.append(globalParameters["WorkingPath"])
  globalParameters["WorkingPath"] = ensurePath(fullPathName)


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

def ClientExecutionLock():
  if not globalParameters["ClientExecutionLockPath"]:
    return open(os.devnull)

  import filelock
  return filelock.FileLock(globalParameters["ClientExecutionLockPath"])

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
# Copyright (C) 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
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
* Copyright (C) 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
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
