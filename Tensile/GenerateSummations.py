
################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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

import os

import pandas as pd
import numpy as np
import yaml
import subprocess
import glob

from shutil import copyfile
from copy import deepcopy

from . import LibraryIO

from . import ClientWriter
from .Common import assignGlobalParameters, ensurePath, globalParameters, \
    gfxName, gfxArch, printExit, getArchitectureName
from .SolutionStructs import ProblemSizes


def getArchitecture(isaName):
    archid = gfxName(isaName)
    return getArchitectureName(archid)

def isValidArch(archName, currentArch):
    arch = gfxArch(archName)
    return currentArch == arch

# Including readSolutionRaw in this file. It is not clear that
# this functionality is needed outside the scope of this utility.
# For this utility we need the unaltered data. 
def readSolutionRaw(filename):
    try:
        stream = open(filename, "r")
    except IOError:
        print ("Cannot open file: %s" % filename )
        return None
    data = yaml.load(stream, yaml.SafeLoader)
    stream.close()
    
    versionString     = data[0]
    scheduleName      = data[1]
    architectureName  = data[2]
    deviceNames       = data[3]
    problemTypeState  = data[4]
    solutionStates    = data[5]
    indexOrder        = data[6]
    exactLogic        = data[7]
    rangeLogic        = data[8]
    otherFields       = []

    dataLength = len(data)
    if dataLength > 9:
        for idx in range(9, dataLength):
            otherFields.append(deepcopy(data[idx]))
    
    return (versionString, scheduleName, architectureName, deviceNames,\
        problemTypeState, solutionStates, indexOrder, exactLogic, rangeLogic, otherFields)

##############################################################################
# createLibraryForBenchmark
##############################################################################
def createLibraryForBenchmark(logicPath, libraryPath, currentPath):
    """
    takes the path of existing logic files as input and adds the summation
    model for each of the solutions. This is used in the Tile Aware Metirc
    Selection.
    """

    pythonExePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin", "TensileCreateLibrary")
    args = [pythonExePath, \
        "--merge-files", "--no-legacy-components", \
        "--new-client-only", "--no-short-file-names", "--no-library-print-debug", "--architecture=all", \
        "--code-object-version=V3", "--cxx-compiler=hipcc", "--library-format=yaml", \
        logicPath, libraryPath, "HIP"]

    try:
        subprocess.run(args, check=True, cwd=currentPath)
    except (subprocess.CalledProcessError, OSError) as e:
        printExit("ClientWriter Benchmark Process exited with error: {}".format(e))

def GenerateSummations(userArgs):

    inputLogicPath = userArgs[0]
    outputPath = userArgs[1]
    assignGlobalParameters({})

    currentISA = globalParameters["CurrentISA"]
    currentArchitecture = getArchitecture(currentISA)

    globPath = os.path.join(inputLogicPath, "{}*".format(currentArchitecture))
    logicFileNames = glob.glob(globPath)

    for logicFileName in logicFileNames:

        logicFileBaseName = os.path.basename(logicFileName)
        logicFileStem, ext = os.path.splitext(logicFileBaseName)
        if (ext != ".yaml"):
            continue

        currentPath = ensurePath(os.path.join(outputPath, logicFileStem))

        libPath = ensurePath(os.path.join(currentPath, "lib"))
        finalPath = ensurePath(os.path.join(currentPath, "final"))
        localLogicPath = ensurePath(os.path.join(currentPath, "logic"))
        localLogicFilePath = os.path.join(localLogicPath, logicFileBaseName)
        
        # Here we read in two version of the logic the first one fills the solutions with
        # defaults and modifies some of the parameters. The final logic file should be the 
        # same as the initial logic with the summation model added. To preseve the original
        # logic we also read in the raw unaltered version of the logic and stage the content
        # to write the final logic.
        logic = LibraryIO.readLibraryLogicForSchedule(logicFileName)
        rawLogic = readSolutionRaw(logicFileName)

        (versionStringR, scheduleNameR, architectureNameR, deviceNamesR, problemTypeStateR,\
            solutionStatesR, indexOrderR, exactLogicR, rangeLogicR, otherFieldsR) =\
            rawLogic

        (_, _, problemType, solutionsForSchedule, \
           _, _, _, _, _) = logic

        copyfile(logicFileName, localLogicFilePath)
        createLibraryForBenchmark(localLogicPath, libPath, currentPath)

        exactList = []

        solutionSummationSizes = [32,64,96,128,256,512,1024,2048,4096,8192,16384]

        for K in solutionSummationSizes:
            e = {"Exact" : [8192, 4096, 1, K]}
            exactList.append(e)

        libraryPath = libPath
        clientBuildDir = os.path.join(outputPath, "client") 
        problemTypeObj = problemType.state

        problemSizes = ProblemSizes(problemTypeObj, exactList)
        dataPath = ensurePath(os.path.join(outputPath, logicFileStem, "data"))
        configFilePath = ensurePath(os.path.join(outputPath, logicFileStem, "configs"))
        dataFilePath = os.path.join(dataPath, "benchmark.csv")
        configFile = os.path.join(configFilePath, "ClientParameters.ini")
        scriptPath = ensurePath(os.path.join(outputPath, logicFileStem, "script"))

        ClientWriter.CreateBenchmarkClientParametersForSizes(libraryPath, problemSizes, dataFilePath, configFile, problemTypeObj)
        ClientWriter.runNewClient(scriptPath, configFile, clientBuildDir)

        tensileLibraryFile = os.path.join(libPath, "library", "TensileLibrary.yaml")

        stream = open(tensileLibraryFile, "r")
        tensileLibrary = yaml.load(stream, yaml.SafeLoader)
        stream.close()

        libSolutions = tensileLibrary["solutions"]

        libSolutionNames = []
        for s in libSolutions:
            kenelName=s["name"]
            libSolutionNames.append(kenelName)

        working_data=pd.read_csv(dataFilePath).rename(str.strip,axis='columns')

        index_keys = working_data.SizeL.unique()
        solutionsDF = working_data.filter(like='Cij')

        perf_max = solutionsDF.max().max().item()

        solutionIndex = 0
        for s in solutionsForSchedule:
            s_stateR = solutionStatesR[solutionIndex]
            kenelName = libSolutionNames[solutionIndex]
            solutionIndex += 1
            perf_raw = working_data[kenelName] 
            perf = (1000*index_keys) / perf_raw
            model = np.polyfit(x=index_keys, y=perf, deg=1)
            slope = model[0].item()
            intercept = model[1].item()
            linearModel = {"slope": slope, "intercept": intercept, "max": perf_max}
            s_stateR["LinearModel"] = deepcopy(linearModel)

        rawLogicData = []
        rawLogicData.append(deepcopy(versionStringR))
        rawLogicData.append(deepcopy(scheduleNameR))
        rawLogicData.append(deepcopy(architectureNameR))
        rawLogicData.append(deepcopy(deviceNamesR))
        rawLogicData.append(deepcopy(problemTypeStateR))
        rawLogicData.append(deepcopy(solutionStatesR))
        rawLogicData.append(deepcopy(indexOrderR))
        rawLogicData.append(deepcopy(exactLogicR))
        rawLogicData.append(deepcopy(rangeLogicR))
        for idx in range(0, len(otherFieldsR)):
            rawLogicData.append(otherFieldsR[idx])
        
        localFinalLogic = os.path.join(finalPath, logicFileBaseName)

        yamlWriter = LibraryIO.YAMLWriter()
        yamlWriter.write(localFinalLogic, rawLogicData)

        outputFinal = ensurePath(os.path.join(outputPath, "final"))
        finalLogic = os.path.join(outputFinal, logicFileBaseName)

        copyfile(localFinalLogic, finalLogic)
