################################################################################
# Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from copy import deepcopy
from .Common import print1, print2, hasParam, printExit, \
        defaultBenchmarkFinalProblemSizes, defaultBatchedBenchmarkFinalProblemSizes, \
        defaultBenchmarkCommonParameters, validParameters, globalParameters
from .CustomKernels import getAllCustomKernelNames
from .SolutionStructs import ProblemType, ProblemSizes, ActivationArgs


def getDefaultsForMissingParameters(paramList, defaultParams):
    """Returns all parameters (with values) in defaultParams not present in paramList"""
    benchmarkParams = []
    for paramDict in defaultParams:
        for paramName in paramDict:
            if not hasParam(paramName, paramList) \
                    or paramName == "ProblemSizes":
                benchmarkParams.append(paramDict)
    return benchmarkParams

def checkParametersAreValid(params, validParams):
    """Ensures paramaters in params exist and have valid values as specified by validParames"""
    for name, values in params.items():
        if name == "ProblemSizes":
            continue

        if name not in validParams:
            printExit("Invalid parameter name: {}\nValid parameters are {}." \
                    .format(name, sorted(validParameters.keys())))

        for value in values:
            if validParams[name] != -1 and value not in validParams[name]:
                msgBase = "Invalid parameter value: {} = {}\nValid values for {} are {}{}."
                msgExt = " (only first 32 combos printed)\nRefer to Common.py for more info" \
                        if len(validParams[name])>32 else ""
                printExit(msgBase.format(name, value, name, validParams[name][:32], msgExt))

def separateParameters(paramSetList):
    """Separates paramSetList into parameters with single and multiple values"""
    singleVaules = {}
    multiValues = {}
    for paramDict in paramSetList:
        for name, values in paramDict.items():
            if values == None:
                printExit("You must specify value(s) for parameter \"{}\"".format(name))
            if len(values) == 1 and name != "ProblemSizes":
                singleVaules[name] = values[0]
            elif len(values) > 1 and name != "ProblemSizes":
                multiValues[name] = values

    return singleVaules, multiValues

def checkCDBufferAndStrides(problemType, problemSizes, isCEqualD):
    """Ensures ldd == ldc when CEqualD"""
    if isCEqualD and problemType["OperationType"] == "GEMM":
        for problem in problemSizes.problems:
            ldd = problem.sizes[problemType["IndexAssignmentsLD"][0]]
            ldc = problem.sizes[problemType["IndexAssignmentsLD"][1]]
            if ldd != ldc:
                printExit("LDD({}) != LDC({}) causes unpredictable result when CEqualD(True)" \
                        .format(ldd, ldc))


class BenchmarkProcess:
    """Representation of benchmarking parameters and resulting steps"""

    def __init__(self, problemTypeConfig, problemSizeGroupConfig):
        """Create from the two sections of a config for a BenchmarkProblem"""
        self.problemType = ProblemType(problemTypeConfig)
        self.isBatched = "Batched" in problemTypeConfig and problemTypeConfig["Batched"]
        print2("# BenchmarkProcess beginning {}".format(self.problemType))

        # fill parameter values from config
        self.singleValueParams = {}
        self.multiValueParams  = {}
        self.customKernels     = []
        self.sizes             = None
        self.getConfigParameters(self.isBatched, problemSizeGroupConfig)

        # convert parameter lists to steps
        # previously, multiple benchmark steps were possible
        # currently only 1 benchmark step is possible; more may be added back later
        self.benchmarkSteps   = []
        self.benchmarkStepIdx = 0
        self.convertParametersToSteps()

    def getConfigParameters(self, isbatched, config):
        """Parse and validate benchmarking parameters in config"""
        print2("")
        print2("####################################################################")
        print1("# Filling in Parameters With Defaults")
        print2("####################################################################")
        print2("")

        # check for no longer supported legacy benchmark steps
        badParams = ["InitialSolutionParameters", "BenchmarkForkParameters", \
                     "JoinParameters", "BenchmarkJoinParameters"]
        badsInConfig = []

        for p in badParams:
            if config.get(p) is not None:
                badsInConfig.append(p)

        if len(badsInConfig) == 1:
            printExit("Benchmark step {} is no longer supported".format("'" + badsInConfig[0] + "'"))
        elif len(badsInConfig) > 1:
            printExit("Benchmark steps {} are no longer supported".format(badsInConfig))

        # get supported benchmark steps
        def getNonNoneFromConfig(key, default):
            if config.get(key) is not None:
                return config[key]
            else:
                return default

        benchmarkCommonParams = getNonNoneFromConfig("BenchmarkCommonParameters", [])
        forkParams            = getNonNoneFromConfig("ForkParameters", [])
        self.customKernels    = getNonNoneFromConfig("CustomKernels", [])

        activationConf = ""
        if "BenchmarkFinalParameters" in config:
            sizes          = config["BenchmarkFinalParameters"][0]["ProblemSizes"]
            if len(config["BenchmarkFinalParameters"]) == 2:
                activationConf = config["BenchmarkFinalParameters"][1]["ActivationArgs"]
        else:
            sizes = defaultBatchedBenchmarkFinalProblemSizes if isbatched \
                else defaultBenchmarkFinalProblemSizes
        self.problemSizes = ProblemSizes(self.problemType, sizes)
        checkCDBufferAndStrides(self.problemType, \
                self.problemSizes, globalParameters["CEqualD"])

        configParams = benchmarkCommonParams + forkParams

        self.activationArgs = ActivationArgs(self.problemType, activationConf)

        # validate and parse raw parameters into more usable forms
        for paramDict in configParams:
            checkParametersAreValid(paramDict, validParameters)

        missingParams = getDefaultsForMissingParameters( \
                configParams, deepcopy(defaultBenchmarkCommonParameters))

        self.singleValueParams, self.multiValueParams \
                = separateParameters(missingParams + configParams)

        # print summary of parameter values
        print2("Single Value Parameters:")
        for k, v in self.singleValueParams.items():
            print2("    {}: {}".format(k, v))

        print2("Multi-Value Parameters:")
        for k, v in self.multiValueParams.items():
            print2("    {}: {}".format(k, v))

    def convertParametersToSteps(self):
        """Create benchmark steps based on parsed parameters"""
        print2("")
        print2("####################################################################")
        print1("# Convert Parameters to Benchmark Step(s)")
        print2("####################################################################")
        print2("")

        # currently only a single step is supported
        print2("")
        print2("####################################################################")
        print1("# Benchmark Final")
        benchmarkStep = BenchmarkStep( \
                self.multiValueParams, \
                self.singleValueParams, \
                self.customKernels, \
                self.problemSizes, \
                self.activationArgs, \
                self.benchmarkStepIdx)
        self.benchmarkSteps.append(benchmarkStep)
        self.benchmarkStepIdx += 1

    def __len__(self):
        return len(self.benchmarkSteps)

    def __getitem__(self, key):
        return self.benchmarkSteps[key]

    def __str__(self):
        string = "BenchmarkProcess:\n"
        for step in self.benchmarkSteps:
            string += str(step)
        return string

    def __repr__(self):
        return self.__str__()


def constructForkPermutations(forkParams):
    """Constructs cartesian product of parameter values in forkParams"""
    totalPermutations = 1
    for _, values in forkParams.items():
        totalPermutations *= len(values)

    forkPermutations = []
    for i in range(0, totalPermutations):
        permutation = {}
        pIdx = i
        for name, v in forkParams.items():
            values = deepcopy(v)
            valueIdx = pIdx % len(v)
            permutation[name] = values[valueIdx]
            pIdx //= len(values)
        forkPermutations.append(permutation)
    return forkPermutations


class BenchmarkStep:
    """A single benchmark step which consists of constant and fork parameters and a set of sizes"""

    def __init__(self, forkParams, constantParams, customKernels, problemSizes, activationArgs, idx):
        """Basic constructor storing each argument"""
        self.forkParams     = forkParams
        self.constantParams = constantParams
        self.customKernels  = customKernels
        self.problemSizes   = problemSizes
        self.activationArgs = activationArgs
        self.stepIdx        = idx

        self.customKernelWildcard = False
        if self.customKernels == ["*"]:
            self.customKernels = getAllCustomKernelNames()
            self.customKernelWildcard = True

        print2("# Creating BenchmarkStep: {} fork params and {} sizes" \
                .format( len(forkParams), problemSizes.totalProblemSizes))

    def isFinal(self):
        """Legacy. Currently always returns true since only one benchmark step is possible"""
        return True

    def __str__(self):
        string = "{:02d}".format(self.stepIdx)
        if self.isFinal():
            string += "_Final"
        return string

    def __repr__(self):
        return self.__str__()
