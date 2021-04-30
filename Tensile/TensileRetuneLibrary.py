###############################################################################
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

from . import BenchmarkProblems
from . import ClientExecutable
from . import ClientWriter
from . import LibraryIO
from . import LibraryLogic
from . import Common
from .Common import globalParameters, print1, printWarning, ensurePath, assignGlobalParameters, \
                    pushWorkingPath, popWorkingPath, restoreDefaultGlobalParameters, HR
from .Tensile import addCommonArguments
from .SolutionStructs import ProblemSizes
from . import __version__

import argparse
import copy
import os
import shutil
import sys


def parseCurrentLibrary(libPath, skipRK):
    libYaml = LibraryIO.readYAML(libPath)
    # parseLibraryLogicData mutates the original data, so make a copy
    fields = LibraryIO.parseLibraryLogicData(copy.deepcopy(libYaml), libPath)
    (_, _, problemType, solutions, _, exactLogic, _, _, _) = fields

    # get performance metric
    if len(libYaml) > 10:
        Common.globalParameters["PerformanceMetric"] = libYaml[10]

    # process exactLogic into ProblemSizes
    sizes = []
    for (size, _) in exactLogic:
        sizes.append({"Exact": size})
    problemSizes = ProblemSizes(problemType, sizes)

    return (libYaml, solutions, problemSizes)


def runBenchmarking(solutions, problemSizes, outPath, update):
    # TODO some copy-pasting from BenchmarkProblems.benchmarkProblemType
    # could use a refactor to elimate duplicated code
    ClientExecutable.getClientExecutable()

    shortName = "benchmark"
    benchmarkDir = os.path.join(outPath, shortName)
    sourceDir = os.path.join(benchmarkDir, "source")
    ensurePath(sourceDir)

    pushWorkingPath(shortName)
    pushWorkingPath("source")

    filesToCopy = [
        "SolutionMapper.h",
        "Client.cpp",
        "Client.h",
        "CMakeLists.txt",
        "DeviceStats.h",
        "TensorUtils.h",
        "MathTemplates.cpp",
        "MathTemplates.h",
        "TensileTypes.h",
        "tensile_bfloat16.h",
        "KernelHeader.h",
        "ReferenceCPU.h",
        "SolutionHelper.cpp",
        "SolutionHelper.h",
        "Tools.cpp",
        "Tools.h",
        ]

    for f in filesToCopy:
        shutil.copy(
            os.path.join(globalParameters["SourcePath"], f),
            globalParameters["WorkingPath"] )
    if globalParameters["RuntimeLanguage"] == "OCL":
        shutil.copy(
            os.path.join(globalParameters["SourcePath"], "FindOpenCL.cmake"),
            globalParameters["WorkingPath"] )
    else:
        shutil.copy(
            os.path.join(globalParameters["SourcePath"], "FindHIP.cmake"),
            globalParameters["WorkingPath"] )

    # make directory for results and set update yaml file
    resultsDir = os.path.normpath(os.path.join(globalParameters["WorkingPath"], "../../Data"))
    ensurePath(resultsDir)
    libraryFile = os.path.join(resultsDir, "benchmark.yaml")

    updateFile = None
    if update:
        updateFile = os.path.join(resultsDir, "update.yaml")
        Common.globalParameters["LibraryUpdateFile"] = updateFile

    BenchmarkProblems.writeBenchmarkFiles(benchmarkDir, solutions, problemSizes, shortName, filesToCopy, [])

    popWorkingPath() # source

    libraryLogicPath = None
    forBenchmark = True
    # TODO make this work with TileAware selection
    returncode = ClientWriter.runClient(libraryLogicPath, forBenchmark, False)
    if returncode:
        printWarning("BenchmarkProblems: Benchmark Process exited with code {}".format(returncode))

    # write solutions yaml file
    for sol in solutions:
        sol["ISA"] = list(sol["ISA"])
    LibraryIO.writeSolutions(libraryFile, problemSizes, [solutions])

    popWorkingPath() # Data

    # copy results to expected directory
    out2 = os.path.join(globalParameters["WorkingPath"], "2_BenchmarkData")
    ensurePath(out2)
    shutil.copy(os.path.join(resultsDir, "benchmark.csv"), os.path.join(out2, "benchmark.csv"))
    shutil.copy(os.path.join(resultsDir, "benchmark.yaml"), os.path.join(out2, "benchmark.yaml"))

    return updateFile


def TensileRetuneLibrary(userArgs):
    print1("")
    print1(HR)
    print1("#")
    print1("#  Tensile Retune Library v{}".format(__version__))

    # argument parsing and related setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("LogicFile", type=os.path.realpath,
                           help="Library logic file to retune")
    argParser.add_argument("OutputPath",
                           help="Where to run benchmarks and output results")
    argParser.add_argument("--update-method", "-u", dest="updateMethod",
                           choices=["remake", "update", "both"], default="remake",
                           help="Method for making new library logic file")
    argParser.add_argument("--skip-replacement-kernels", "-s", dest="skipRK", action="store_true",
                           help="Exclude sizes and solutions related to replacement kernels. "
                                "Forces update-method to 'remake'")

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)

    libPath = args.LogicFile
    if args.skipRK:
        args.updateMethod = "remake"

    if args.updateMethod == "remake":
        update = False
        remake = True
    elif args.updateMethod == "update":
        update = True
        remake = False
    else: # args.updateMethod == "both"
        update = True
        remake = True

    ##############################################
    # Retuning
    ##############################################
    print1("#  Library Logic: {}".format(libPath))
    print1("#")
    print1(HR)
    print1("")

    # setup global parameters
    outPath = ensurePath(os.path.abspath(args.OutputPath))
    restoreDefaultGlobalParameters()
    assignGlobalParameters({"LibraryFormat": "msgpack",
                            "OutputPath": outPath,
                            "WorkingPath": outPath})

    # run main steps
    (rawYaml, solutions, problemSizes) = parseCurrentLibrary(libPath, args.skipRK)
    updateFile = runBenchmarking(solutions, problemSizes, outPath, update)

    if remake:
        # write library logic file
        LibraryLogic.main({"ScheduleName": rawYaml[1],
                       "ArchitectureName": rawYaml[2],
                       "DeviceNames": rawYaml[3] })

    if update:
        # read update yaml from benchmark client and update logic
        print1("")
        print1(HR)
        print1("# Reading update file from Benchmarking Client")
        updateLogic = LibraryIO.readYAML(updateFile)
        rawYaml[7] = updateLogic

        # write updated library logic (does not overwrite original)
        libName = os.path.basename(libPath)
        outFile = os.path.join(outPath, libName)

        print1("# Writing updated Library Logic: {}".format(outFile))
        LibraryIO.writeYAML(outFile, rawYaml, explicit_start=False, explicit_end=False)
        print1(HR)


def main():
    TensileRetuneLibrary(sys.argv[1:])
