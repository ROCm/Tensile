###############################################################################
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

from . import Common
from . import ClientWriter
from . import LibraryIO
from .Contractions import ProblemType as ContractionsProblemType
from .SolutionStructs import ProblemSizes, ProblemType
from .Common import print1, printExit, assignGlobalParameters, restoreDefaultGlobalParameters, HR
from .Tensile import addCommonArguments, argUpdatedGlobalParameters
from . import __version__

import argparse
import os
import sys


def getGlobalParams(config):
    globalParams = None
    try:
        globalParams = config["GlobalParameters"]
    except:
        pass

    return globalParams


def getProblemDict(config):
    problemDict = None
    try:
        problemDict = config["BenchmarkProblems"][0][0]
    except:
        pass
    else:
        return problemDict

    try:
        problemDict = config["ProblemType"]
    except:
        pass

    return problemDict


def getSizeList(config):
    sizeList = None
    try:
        sizeList = config["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
    except:
        pass
    else:
        return sizeList

    keep = True
    if type(config) is list:
        for i in config:
            if not (type(i) is dict and ("Exact" in i or "Range" in i)):
                keep = False
                break
    else:
        keep = False

    if keep:
        sizeList = config
    return sizeList


def parseConfig(config):

    globalParams = getGlobalParams(config)
    problemDict = getProblemDict(config)
    sizeList = getSizeList(config)

    return (globalParams, problemDict, sizeList)


def TensileClientConfig(userArgs):
    print1("")
    print1(HR)
    print1("#")
    print1("#  Tensile Client Config v{}".format(__version__))
    print1("#")
    print1(HR)
    print1("")

    # argument parsing
    argParser = argparse.ArgumentParser()
    argParser.add_argument("ConfigYaml", type=os.path.realpath, nargs="+",
                           help="Config yaml(s) containing parameters and problem and size information")
    argParser.add_argument("OutputConfig", type=os.path.realpath,
                           help="Path to output resulting client config file")

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)

    # parse config inputs
    configs = [LibraryIO.readYAML(x) for x in args.ConfigYaml]

    globalParams = {}
    problemDict = None
    sizeList = None

    for config in configs:
        # get all information we can get from each config
        (myGlobalParams, myProblemDict, mySizeList) = parseConfig(config)

        if myGlobalParams is not None:
            if globalParams == {}:
                globalParams = myGlobalParams
            else:
                printExit("duplicate global params")

        if myProblemDict is not None:
            if problemDict is None:
                problemDict = myProblemDict
            else:
                printExit("duplicate problem type")

        if mySizeList is not None:
            if sizeList is None:
                sizeList = mySizeList
            else:
                printExit("duplicate sizes")

    if problemDict is None:
        printExit("could not parse problem type from inputs")
    if sizeList is None:
        printExit("could not parse size list from inputs")

    ssProblemType = ProblemType(problemDict)
    conProblemType = ContractionsProblemType.FromOriginalState(ssProblemType)
    sizes = ProblemSizes(ssProblemType, sizeList)


    # update globals
    restoreDefaultGlobalParameters()
    assignGlobalParameters(globalParams)

    overrideParameters = argUpdatedGlobalParameters(args)
    for key, value in overrideParameters.items():
        print1("Overriding {0}={1}".format(key, value))
        Common.globalParameters[key] = value

    # write output
    ClientWriter.writeClientConfigIni(sizes, conProblemType, "", [], "", args.OutputConfig, None)


def main():
    TensileClientConfig(sys.argv[1:])
