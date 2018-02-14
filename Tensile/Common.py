################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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
import os.path
import sys
from __init__ import __version__
from collections import OrderedDict
from subprocess import Popen, PIPE
import time
import platform

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

########################################
# less common
########################################
globalParameters["CMakeBuildType"] = "Release"            # whether benchmark clients and library client should be release or debug
globalParameters["PrintSolutionRejectionReason"] = False  # when a solution is marked as invalid, print why
# how to initialize tensor data
globalParameters["DataInitTypeAB"] = 3            # 0=0, 1=1, 2=serial, 3=rand, 4=NaN
globalParameters["DataInitTypeC"]  = 3            # 0=0, 1=1, 2=serial, 3=rand, 4=NaN
globalParameters["DataInitTypeAlpha"] = 2         # 0=0, 1=1, 2=2, 3=rand, 4=NaN
globalParameters["DataInitTypeBeta"] = 2          # 0=0, 1=1, 2=2, 3=rand, 4=NaN
# build parameters
globalParameters["CMakeCXXFlags"] = ""            # pass flags to cmake
globalParameters["CMakeCFlags"] = ""              # pass flags to cmake
globalParameters["DebugKernel"] = False           # assembly only, kernel gets buffer for debug "printing"; kernel writes data to memory, gets coppied to host and printed
globalParameters["LibraryPrintDebug"] = False     # solutions will print enqueue info when enqueueing a kernel
# device selection
globalParameters["Platform"] = 0                  # select opencl platform
globalParameters["Device"] = 0                    # select hip device or opencl device within platform

# shouldn't need to change
globalParameters["DeviceLDS"] = 65536             # LDS bytes per CU, for computing occupancy
globalParameters["MaxLDS"] = 65536                # max LDS a kernel should attempt to use
globalParameters["MaxDepthU"] = 256               # max DepthU value to allow
globalParameters["ShortNames"] = False            # on windows kernel names can get too long; =True will convert solution/kernel names to serial ids
globalParameters["MergeFiles"] = True             # F=store every solution and kernel in sepperate file; T=store all solutions in single file
globalParameters["SupportedISA"] = [(8,0,3), (9,0,0)]             # assembly kernels writer supports these architectures
globalParameters["BenchmarkProblemsPath"] = "1_BenchmarkProblems" # subdirectory for benchmarking phases
globalParameters["BenchmarkDataPath"] = "2_BenchmarkData"         # subdirectory for storing final benchmarking data
globalParameters["LibraryLogicPath"] = "3_LibraryLogic"           # subdirectory for library logic produced by analysis
globalParameters["LibraryClientPath"] = "4_LibraryClient"         # subdirectory for building example library client

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

# might be deprecated
globalParameters["SolutionMapHash"] = False
globalParameters["EnableHalf"] = False

################################################################################
# Enumerate Valid Solution Parameters
################################################################################
validWorkGroups = []
for numThreads in range(64, 1025, 64):
    for nsg in [ 1, 2, 4, 8, 16, 32, 64, 128, 256 ]:
        for sg0 in range(1, numThreads/nsg+1):
            sg1 = numThreads/nsg/sg0
            if sg0*sg1*nsg == numThreads:
                workGroup = [sg0, sg1, nsg]
                validWorkGroups.append(workGroup)


validThreadTileSides = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]
validThreadTiles = []
for i in validThreadTileSides:
    for j in validThreadTileSides:
        validThreadTiles.append([i, j])

validMacroTileSides = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 6, 12, 24, 48, 96, 192, 384, 768 ]
validMacroTiles = []
validISA = [(0,0,0)]
validISA.extend(globalParameters["SupportedISA"])
depthUs = range(-16, 0)
depthUs.extend(range(2,512+1,2))
for i in validMacroTileSides:
    for j in validMacroTileSides:
        validMacroTiles.append([i, j])
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
    "GlobalReadCoalesceGroupA":   [ False, True ], # True means
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

    "PrefetchGlobalRead":         [ False, True ], # prefetch / double-buffer reads from global memory -> vgprs -> lds
    "PrefetchLocalRead":          [ False, True ], # prefetch / double-buffer reads from lds

    # When splitting up the summation between workgroups, there are two options for organizing which workgroup will do what
    # If we begin with N workgroups and set GSU=4, there will now be 4N workgroups
    # GSUWGMRR=False means workgroup 0,1,2,3 will all work on the same tile; =True means workgroup 0, N-1, 2N-1, 3N-1 will all work on the same tile
    # GSUSARR=False means the 4 workgroups do whole chunks of the summation: k=0 -> K/4-1, k=K/4 -> 2K/4-1, k=2K/4 -> 3K/4-1, k=3K/4 -> 4K/4-1
    # GSUSARR=True means the 4 workgroups round robin split up the chunks of the summation: k=0 -> DU-1, 4DU -> 5DU-1, ...; k=1DU -> 2DU-1, 5DU -> 6DU-1...; ...
    "GlobalSplitU":               range(1, 1024+1),
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
    "BufferLoad":                 [ False, True] ,

    "WorkGroupMapping":           range(-1024,1024+1),  # change a workgroup's id so that the all the workgroups on the gpu at a time are hitting L2 cache the best
    "WorkGroupMappingType":       ["B", "Z"],           # Blocking, Z-order (not any faster than blocking, especially for the arithmetic it requires)
    "MaxOccupancy":               range(1, 40+1),       # wg / CU; if cache thrashing is hurting performance, this allocates extra lds to artificially limit occupancy
    "WorkGroup":                  validWorkGroups,      # ( wg0 x wg1 x LocalSplitU ) dimensions of the workgroup which will operate on a tile and share lds
    "ThreadTile":                 validThreadTiles,     # ( tt0 x tt1 ) dimensions of the C tile that each thread works on, TT=4 and VW=4 means a thread will work on a tight 4x4 tile of C, where VW=1 means the tile will work on 16 spread out values
    "MacroTile":                  validMacroTiles,      # MT0 = wg0*tt0, MT1 = wg1*tt1

    # threads should read/write/operate on this many contiguous elements. VW=4 on sgemm means read/write float4's.
    # -1 means use the largest vector width up to 128 bits. Using a VW too large which results in >128 bits isn't supported and should be faster
    "VectorWidth":                [ -1, 1, 2, 3, 4, 6, 8, 12, 16 ],

    # place upper and lower limits on the skinny-ness of macro tiles; shape=1 means square tile, like 64x64. shape=4 means 4x64 or 64x4 or 128x8...
    # these will just mark some kernels as invalid so that fewer kernels will be checked
    "MacroTileShapeMin":          range(1, 64+1),
    "MacroTileShapeMax":          range(1, 64+1),

    # when loading all the data from global into lds requires multiple load instructions, these parameters govern which
    # loads will pull which rectangle of data from global into lds
    # NLC=1 means one load along the coalesced dimension, which results in the most coalescing possible
    # NLC=-1 looks for the largest number of reads along the coalesced dimension which results in the least ammount of coalescing;
    # however in this case the stride between one load and another is a static value, therefore buffer loads only need one set of registers
    # whereas the =1 case has a stride which is a multiple of a kernel argument and therefore needs one address per load in the perpendicular dimension
    "NumLoadsCoalescedA":         range(-1, 64+1),
    "NumLoadsCoalescedB":         range(-1, 64+1),

    # DepthU, LocalSplitU (which is the 3rd number in WorkGroup), and LoopUnroll are closely related
    # LoopUnroll=4 means there are 4 subiterations within the loop, 4 actual iterations written in the code.
    # LocalSplit=2 means the workgroup is split up into 2 subgroups, and each subgroup is doing different parts of the summation.
    # subgroup0 does k=0-3, 8-11... and subgroup1 does k=4-7, 12-15...
    # So, each iteration through the summation loop, which has 4 actual subiterations, does 8 summation iterations, because each subgroup did 4; and when data is read from global memory the threads read 8 elements along the summation dimension.
    # DepthU = LoopUnroll * LocalSplitU = 4*2 in this case
    # it made more sense for the user to directly control LocalSplitU and DepthU, then derrive afterwards LoopUnroll=DepthU/LocalSplitU
    "DepthU":                     depthUs,

    # integer ammount of padding to put into LDS, in 2016 this didn't seem to help performance, profilers were showing that channel conflicts weren't really hurting
    # performance so this has been deprecated and probably doesn't work
    "LdsPad":                     [ 0, 1 ],

    # tinkered with adding extra syncs or waits in the assembly kernels to see if it would improve the sequencing between workgroups, "fully synchronous scheduling" is WAY more promising; this can be deprecated
    "PerformanceSyncLocation":    range(-1, 16*16+1),
    "PerformanceWaitLocation":    range(-1, 16*16+1),
    "PerformanceWaitCount":       range(-1, 16),

    # add gls or slc after global memory read/writes to change cacheing, not cacheing the writes is promising and improved performance a tiny bit
    "NonTemporalC":               range(0,4),
    "NonTemporalA":               range(0,4),
    "NonTemporalB":               range(0,4),

    # guard against out of bounds reads
    # None: don't guard
    # Branch: use if statements (source only, and doesn't support VW)
    # ShiftPtr: shift read pointers to be in bounds, then unshift registers (source & assembly), allows smallest supported problem size to be M or N >= global load vector width, i.e. 1
    # ShiftTile: todo. this is MIOpenGemm's strategy, probably eliminates unshift however smallest supported problem size would be tile size
    # BoundaryLoad: todo. use isa to set buffer/image load boundaries and out of bounds data automatically comes in as zero
    "EdgeType":                   [ "Branch", "ShiftPtr", "None" ], # None=don't guard against ou

    # Kernels should be written in assembly or source
    # if assembly, ISA will determine architecture
    # if source, Runtime will determine language
    # later on, we'll relax this to inner kernel languages and outer kernel languages, such as inline asm embedded in ocl or in llvm
    "KernelLanguage":             [ "Assembly", "Source" ],
    "ISA":                        validISA,       # arch for assembly kernels

    }
# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"LoopDoWhile":               [ False ] },
    {"LoopTail":                  [ True ] },
    {"EdgeType":                  [ "Branch" ] },
    {"KernelLanguage":            [ "Source" ] },
    {"LdsPad":                    [ 0 ] },
    {"MaxOccupancy":              [ 40 ] },
    {"VectorWidth":               [ -1 ] },
    {"GlobalReadCoalesceVectorA": [ True ] },
    {"GlobalReadCoalesceVectorB": [ True ] },
    {"GlobalReadCoalesceGroupA":  [ True ] },
    {"GlobalReadCoalesceGroupB":  [ True ] },
    {"PrefetchGlobalRead":        [ True ] },
    {"PrefetchLocalRead":         [ True ] },
    {"UnrollMemFence":            [ False ] },
    {"GlobalRead2A":              [ True ] },
    {"GlobalRead2B":              [ True ] },
    {"LocalWrite2A":              [ True ] },
    {"LocalWrite2B":              [ True ] },
    {"LocalRead2A":               [ True ] },
    {"LocalRead2B":               [ True ] },
    {"BufferLoad":                [ True ] },
    {"GlobalSplitU":              [ 1 ] },
    {"GlobalSplitUSummationAssignmentRoundRobin": [ True ] },
    {"GlobalSplitUWorkGroupMappingRoundRobin":    [ False ] },
    {"MacroTileShapeMin":         [ 1 ] },
    {"MacroTileShapeMax":         [ 64 ] },
    {"NumLoadsCoalescedA":        [ 1 ] },
    {"NumLoadsCoalescedB":        [ 1 ] },
    {"WorkGroup":                 [ [16,16,1]] },
    {"WorkGroupMappingType":      [ "B" ] },
    {"WorkGroupMapping":          [ 8 ] },
    {"ThreadTile":                [ [4,4] ] },
    {"DepthU":                    [ -1 ] },
    {"PerformanceSyncLocation":   [ -1 ] },
    {"PerformanceWaitLocation":   [ -1 ] },
    {"PerformanceWaitCount":      [ -1 ] },
    {"NonTemporalC":              [ 0 ] },
    {"NonTemporalA":              [ 0 ] },
    {"NonTemporalB":              [ 0 ] },
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
        for key, value in paramDict.iteritems():
            defaultSolution[key] = value[0]
# other non-benchmark options for solutions

################################################################################
# Default Problem Type
################################################################################
defaultProblemType = {
    # =GEMM uses TransposeA,B paramters and makes the problem type more readeable for users
    # =TensorContraction  requires specifying
    "OperationType":            "GEMM",

    "DataType":                 0,                # data types can specified by a variety of ways, such as "s", as listed in SolutionStructs.py::DataType
    "UseBeta":                  True,             # =True use beta parameter (asm will check for B=0 and optimize the write for that), =False don't use beta parameter
    "HighPrecisionAccumulate":  False,            # this was the original plan for specifying f32 += f16*f16, but its possible that Accumulation/Internal precision and output precision should be their own DataTypes altogether
    "ComplexConjugateA":        False,            # complex data should be conjugated for "C" transpose case
    "ComplexConjugateB":        False,

    # for gemm description
    "TransposeA":               False,            # =True means transA="T" or "C", =False means transA = "N"
    "TransposeB":               True,
    "Batched":                  False,            # add batching dimension

    # for tensor contraction description
    "IndexAssignmentsA":        [0, 2],
    "IndexAssignmentsB":        [1, 2],
    "NumIndicesC":              2,
    "UseInitialStrides":        False,


    }
defaultProblemSizes = [{"Range": [ [2880], 0, 0 ]}]
defaultBenchmarkFinalProblemSizes = [{"Range": [
    [64, 64, 64, 512], 0, 0 ]}]


################################################################################
# Default Analysis Parameters
################################################################################
defaultAnalysisParameters = {
    "ScheduleName":       "Tensile",
    "DeviceNames":  ["Unspecified"],
    #"BranchPenalty":              0, # microseconds / kernel
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
        print message
        sys.stdout.flush()
def print2(message):
    if globalParameters["PrintLevel"] >= 2:
        print message
        sys.stdout.flush()

def printWarning(message):
    print "Tensile::WARNING: %s" % message
    sys.stdout.flush()
def printExit(message):
    print "Tensile::FATAL: %s" % message
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

################################################################################
# Assign Global Parameters
# each global parameter has a default parameter, and the user
# can override them, those overridings happen here
################################################################################
def assignGlobalParameters( config ):
    global globalParameters

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
    globalParameters["AssemblerPath"] = locateExe("/opt/rocm/bin", "hcc")
    globalParameters["ROCmSMIPath"] = locateExe("/opt/rocm/bin", "rocm-smi")

    # read current gfx version
    if os.name != "nt" and globalParameters["CurrentISA"] == (0,0,0) and globalParameters["ROCmAgentEnumeratorPath"]:
        process = Popen([globalParameters["ROCmAgentEnumeratorPath"], "-t", "GPU"], stdout=PIPE)
        line = process.stdout.readline()
        while line != "":
            gfxIdx = line.find("gfx")
            if gfxIdx >= 0:
                major = int(line[gfxIdx+3:gfxIdx+4])
                minor = int(line[gfxIdx+4:gfxIdx+5])
                step  = int(line[gfxIdx+5:gfxIdx+6])
                if (major,minor,step) in globalParameters["SupportedISA"]:
                    print1("# Detected ISA: gfx%u%u%u"%(major, minor, step))
                    globalParameters["CurrentISA"] = (major, minor, step)
                line = process.stdout.readline()
        if globalParameters["CurrentISA"] == (0,0,0):
            printWarning("Did not detect SupportedISA: %s; cannot benchmark assembly kernels." % globalParameters["SupportedISA"])
        if process.returncode:
            printWarning("%s exited with code %u" % (globalParameters["ROCmAgentEnumeratorPath"], process.returncode))

    # For ubuntu platforms, call dpkg to grep the version of hcc.  This check is platform specific, and in the future
    # additional support for yum, dnf zypper may need to be added.  On these other platforms, the default version of
    # '0.0.0' will persist
    if platform.linux_distribution()[0] == "Ubuntu":
        process = Popen(["dpkg", "-l", "hcc"], stdout=PIPE)
        if process.returncode:
            printWarning("%s looking for package %s exited with code %u" % ('dpkg', 'hcc', process.returncode))

        line = process.stdout.readline()
        while line != "":
            packageIdx = line.find("hcc")
            if packageIdx >= 0:
                globalParameters["HccVersion"] = line.split()[2]
                break
            line = process.stdout.readline()

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


################################################################################
# Push / Pop Working Path
# store a WorkingPath where to write files (like benchmark files)
################################################################################
def pushWorkingPath( foldername ):
    globalParameters["WorkingPath"] = \
        os.path.join(globalParameters["WorkingPath"], foldername )
    ensurePath( globalParameters["WorkingPath"] )
def popWorkingPath():
    globalParameters["WorkingPath"] = \
        os.path.split(globalParameters["WorkingPath"])[0]
def ensurePath( path ):
    if not os.path.exists(path):
        os.makedirs(path)

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
    if qMinor > tMinor:
        return False
    if qMinor == tMinor:
        if qStep > tStep:
            return False
    return True

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

    def increment(self):
        self.update(self.priorValue + 1)

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
            sys.stdout.write("\n")
        sys.stdout.flush()

# Append copyrights to all files generated by tensile since they belong to Tensile intellectual property
CMakeHeader = """################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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
