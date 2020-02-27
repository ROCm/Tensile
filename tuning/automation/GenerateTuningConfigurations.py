################################################################################
# Copyright (C) 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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
import sys
import argparse
import re
import shutil
from copy import deepcopy
from ExtractSizes import *
from TuningConfiguration import *


def CopyContent(inputFileList, outputFile):

    with open(outputFile,'wb') as wfd:
        for f in inputFileList:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)


def ProcesDefinitionFile(definitionFileName):

    definitionFile = open(definitionFileName, 'r')

    configurationMapper = {}

    for line in definitionFile:

        fileName, solutionName, problemType, transform, libraryType = tuple(line.strip().split(","))

        key = "%s_%s_%s" % (libraryType, problemType, transform)
        solutionFileName = "solutions/%s.yml" % solutionName

        configDefinitionList = None
        if key in configurationMapper:
            configDefinitionList = configurationMapper[key]
        else:
            configDefinitionList = []
            configurationMapper[key] = configDefinitionList 

        configDefinition = (solutionFileName, fileName)
        configDefinitionList.append(configDefinition)

    definitionFile.close()    

    return configurationMapper




def processFile(headerFileName, key, configDefinitionList, configurationPath, workingDirectoryName, outputFileName, outputPath):
    
    libraryName, solutionName, transformatnType = key.split('_') 

    typeFileName = "types/%s_%s.yml" % (solutionName, transformatnType)
    typeFilePath = os.path.join(configurationPath, typeFileName)

    contentFileNames = [headerFileName, typeFilePath]

    configurationFileName = "%s_%s" % (key,outputFileName)
    configurationFilePath = os.path.join(outputPath, configurationFileName)

    for configDefinition in configDefinitionList:
        problemFileName, sizeFileName = configDefinition
        
        problemFilePath = os.path.join(configurationPath, problemFileName)
        contentFileNames.append(problemFilePath)
        contentFileNames.append(sizeFileName)

    libraryFileName = "boiler/library_logic_%s_only.yml" % libraryName
    libraryFilePath = os.path.join(configurationPath, libraryFileName)

    contentFileNames.append(libraryFilePath)
    CopyContent(contentFileNames, configurationFilePath)

def GetSize(problemDefinition):
    m = int(problemDefinition["m"])
    n = int(problemDefinition["n"])
    k = int(problemDefinition["k"])
    b = 1

    if "batch" in problemDefinition:
        b = int(problemDefinition["batch"])

    return [m, n, b, k]

def ClassifySize(size):
    m = size[0]
    n = size[1]
    b = size[2]
    k = size[3]

    sizeKey = "none"

    scale = m * n

    tiny = 32 * 32
    small = 128 * 128
    medium = 512 * 512

    if b > 1:
        sizeKey = "batch"

    elif (scale <= tiny):
        sizeKey = "tiny"

    elif (scale <= small):
        sizeKey = "small"

    elif (scale <= medium):
        sizeKey = "medium"

    else:
        sizeKey = "large"

    return sizeKey

def GetProblemType(key,tileAware):
    _ , transposeA, transposeB, dType = key

    initialParams = {}

    if transposeA == "N":
        initialParams["TransposeA"] = False
    else:
        initialParams["TransposeA"] = True

    if transposeB == "N":
        initialParams["TransposeB"] = False
    else:
        initialParams["TransposeB"] = True
    initialParams["DataType"] = dType

    problemType = generateProblemType(initialParams,tileAware)

    return problemType

def generateBenchmarkGroupFromScheme(scheme, tileAware="false"):
    benchmarkGroup = generateEmptyBenchmarkGroup()

    commonParams = []
    forkParams = []
    finalParams = []

    if tileAware == "true":
        finalParams.append({"ProblemSizes":None}) 
        finalParams.append({"SolutionSummationSizes":[30,60,90,120,180,360,720,1440,2880,5000,10000,15000,20000,25000,30000]})

    for key in scheme:
        value = scheme[key]
        if len(value) > 1:
            d = {}
            v = deepcopy(value)
            d[key] = v
            forkParams.append(d)
        else:
            d = {}
            v = deepcopy(value)
            d[key] = v
            commonParams.append(d)

    benchmarkGroup["ForkParameters"] = forkParams
    benchmarkGroup["BenchmarkCommonParameters"] = commonParams
    benchmarkGroup["BenchmarkFinalParameters"] = finalParams

    return benchmarkGroup

def generateDefaultScheme():
    scheme={"EdgeType": ["ShiftPtr"],
            "KernelLanguage": ["Assembly"],
            "LoopTail": [True],
            "WorkGroupMapping": [1, 8, 16],
            "DepthU": [8,16,32],
            "VectorWidth": [-1],
            "GlobalSplitU": [1, 2, 4, 6, 8, 12, 16],
            "GlobalReadVectorWidth": [-1],
            #"LdsPadA": [0, -1 ],
            #"LdsPadB": [0, -1 ],
            #"UseSgprForGRO": [0, 1],
            "FractionalLoad": [1],
            "PrefetchGlobalRead": [False],
            "PrefetchLocalRead": [ False, True]}

    return scheme

def updateProblemGroupFromKey(problemKey, sizeKey,problemGroup,sizeList, tileAware="false"):

    _ , transposeA, transposeB, dType = problemKey
    
    transposeType = "%s%s" % (transposeA.lower(),transposeB.lower())
    benchmarkGroup = None

    scheme = generateDefaultScheme()

    if dType == "h":
        scheme["AssertSummationElementMultiple"] = [2]
        scheme["AssertFree0ElementMultiple"] = [2]
    
    if sizeKey == "batch":
        scheme["GlobalSplitU"] = [1]
        scheme["LdsPadA"] = [0, -1]
        scheme["LdsPadB"] = [0, -1]
        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware) 
        appendThreadTiles(benchmarkGroup, [[4,4],[4,2],[2,4],[4,8],[8,4],[8,8]])
        appendWorkGroups(benchmarkGroup, [[16,16,1],[16,8,1],[8,16,1]])
        appendSizes(benchmarkGroup,sizeList,tileAware)
    elif sizeKey == "tiny":
        scheme["GlobalSplitU"] = [1,2,3,4]
        scheme["LdsPadA"] = [0, -1]
        scheme["LdsPadB"] = [0, -1]
        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware) 
        appendThreadTiles(benchmarkGroup, [[2,2],[4,2],[2,4]])
        appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[4,16,4],[16,4,4],[32,8,4],[8,32,4]])
        #appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[8,16,4],[16,8,2],[16,8,4],[8,8,1],
        #    [8,8,2],[8,8,4],[4,16,4],[16,4,4],[4,8,8],[8,4,8],[4,4,4],[4,4,8]])
        appendSizes(benchmarkGroup,sizeList,tileAware)
    elif sizeKey == "small":
        scheme["GlobalSplitU"] = [1,2,4,6,8]
        scheme["LdsPadA"] = [0, -1]
        scheme["LdsPadB"] = [0, -1]
        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
        appendThreadTiles(benchmarkGroup, [[2,2],[4,2],[2,4],[4,4]])
        appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[4,16,4],[16,4,4]])
        #appendWorkGroups(benchmarkGroup, [[16,16,1],[8,8,4],[8,16,2],[8,16,4],[16,8,2],
        #    [16,8,4],[32,4,2],[4,32,2],[4,16,4],[16,4,4]])
        appendSizes(benchmarkGroup,sizeList,tileAware)
    elif sizeKey == "medium":
        if transposeType == "tn":
            scheme["GlobalSplitU"] = [1,2,4,6,8]
            scheme["DepthU"] = [8, 16]
            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware) 
            appendThreadTiles(benchmarkGroup, [[8,8],[4,6],[8,4],[6,4]])
            appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[4,16,4],[16,4,4]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
        else:
            scheme["GlobalSplitU"] = [1,2,4,6,8]
            scheme["LdsPadA"] = [0, -1]
            scheme["LdsPadB"] = [0, -1]
            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware) 
            appendThreadTiles(benchmarkGroup, [[8,8],[4,4],[6,4],[4,6]])
            appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[4,16,4],[16,4,4]])
            #appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[8,8,4],[8,8,8],[4,16,4],[16,4,4],[32,8,2],[8,32,2]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
    else: #sizeKey == "large"
        scheme["GlobalSplitU"] = [1,2,4,6,8,12]
        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware) 
        appendThreadTiles(benchmarkGroup, [[4,4],[6,4],[4,6],[4,8],[8,4],[8,8],[12,12]])
        appendWorkGroups(benchmarkGroup, [[16,16,1]])
        appendSizes(benchmarkGroup,sizeList,tileAware)

    problemGroup.append(benchmarkGroup)

def OutputConfigs(problemMapper, configPath, outputName, library, tileAware):

    keys = list(problemMapper.keys())

    configDefs = {}

    for key in keys:
        lineDefinitions = problemMapper[key]
        sizeMapper = {}
        for problemDefinition in lineDefinitions:
            size =  GetSize(problemDefinition)

            sizeKey = ClassifySize(size)
            if sizeKey not in sizeMapper:
                sizeMapper[sizeKey] = []
            sizeMapper[sizeKey].append(size)

        problemType = GetProblemType(key,tileAware)
        dataType = problemType["DataType"].lower()
        operationType = problemType["OperationType"].lower()

        problemTypeName = "%s%s" % (dataType, operationType)

        _, transposeA, transposeB, _ = key
        transpose = "%s%s" % (transposeA.lower(), transposeB.lower())
        problemKey = "%s_%s_%s" % (library, problemTypeName, transpose)
        configurationFileName = "%s_%s" % (problemKey,outputName)
        configurationFilePath = os.path.join(configPath, configurationFileName)


        newConfig = None 
        problemGroup = None

        if configurationFilePath in configDefs:
            newConfig = configDefs[configurationFilePath]
            problemGroup = newConfig.benchmarkProblems[0]
        else:
            newConfig = TuningConfiguration()
            newConfig.globalParameters = deepcopy(defaultHeader)
            newConfig.libraryLogic = deepcopy(libraryLogicMapper[library])
            newConfig.libraryClient = True
            problemGroup = [problemType]
            newConfig.benchmarkProblems = [problemGroup]
            configDefs[configurationFilePath] = newConfig

        for sizeKey in sizeMapper:
            sizeList = sizeMapper[sizeKey]
            updateProblemGroupFromKey(key,sizeKey,problemGroup,sizeList,tileAware)
        
    for key in configDefs:
        newConfig = configDefs[key]
        newConfig.writeLibraryLogic(key)

def GetOutputFileName(outputPath, namePart, key, ext):
    function, transposeA, transposeB, dType = key
    fileName = namePart

    if "strided" in function:
        fileName += "-strided-%s%s.%s" % (transposeA,transposeB,ext)
    else:
        fileName += "-%s%s.%s" % (transposeA,transposeB,ext)

    outputFileName = outputFileName = os.path.join(outputPath, fileName)
    return outputFileName


def generateRunScript(fileNames, outputPath,count='1'):
    
    scriptNames = ""

    for fileName in fileNames:
        fileBaseName = os.path.basename(fileName)
        namePart, _ = os.path.splitext(fileBaseName)
        scriptNames = "%s %s" % (scriptNames, namePart)

    runallTemplate = """#!/bin/bash

mkdir results%s

for NAME in%s
do
./${NAME}.sh > results%s/${NAME}.1 2>&1
done
""" 
    runallContent = runallTemplate % (count, scriptNames, count)
    doitFileName = os.path.join(outputPath, "doit_all"+count+".sh")
    doitFile = open(doitFileName,"w")
    doitFile.write(runallContent)
    doitFile.close() 

def OutputScript(problemMapper, scriptPath, namePart):

    keys = list(problemMapper.keys())

    scriptFileNames = []

    for key in keys:
        outputFileName = GetOutputFileName(scriptPath, namePart, key, "sh")
        scriptFileNames.append(outputFileName)
        lineDefinitions = problemMapper[key]
        lines = ["#!/bin/bash",""]
        for problemDefinition in lineDefinitions:
            rocblas_call = BuildRocBLASBenchmarkCall(problemDefinition)
            lines.append(rocblas_call)
        with open(outputFileName, 'w') as f:
            for line in lines:
                f.write("%s\n" % line)

    generateRunScript(scriptFileNames, scriptPath)
    
def OutputScript2(problemMapper, scriptPath, namePart):

    keys = list(problemMapper.keys())

    scriptFileNames = []
    
    for key in keys:
        outputFileName = GetOutputFileName(scriptPath, namePart, key, "sh")
        scriptFileNames.append(outputFileName)
        lineDefinitions = problemMapper[key]
        lines = ["#!/bin/bash",""]
        for problemDefinition in lineDefinitions:
            rocblas_call = BuildRocBLASBenchmarkCall(problemDefinition)
            lines.append(rocblas_call)
        with open(outputFileName, 'w') as f:
            for line in lines:
                if "rocblas-bench" in line:
                    f.write("ROCBLAS_TENSILE_LIBPATH=${TENSILE_LIBRARY} %s\n" % line)
                else:
                    f.write("%s\n" % line)
                    
    generateRunScript(scriptFileNames, scriptPath,'2')

def OutputProblemDefinitions(problemMapper, sizePath, namePart):

    keys = list(problemMapper.keys())

    for key in keys:
        lineDefinitions = problemMapper[key]
        outputFileName = GetOutputFileName(sizePath, namePart, key, "csv")
        output = open(outputFileName,"w+")
        writer = csv.DictWriter(output, fieldnames=rocblas_parameters, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(lineDefinitions)

def RunMain():

    userArgs = sys.argv[1:]

    argParser = argparse.ArgumentParser()

    if len(sys.argv) <= 6:
        argParser.add_argument("input_file_name", help="configuration file path")
    else:
        argParser.add_argument("input_logs", help="the input path for log files")
        argParser.add_argument("network_name", help="neural network name")

    argParser.add_argument("output_path", help="the output path")
    argParser.add_argument("output_file_name", help="the output file name")
    argParser.add_argument("library", help="the library Logic name")
    argParser.add_argument("tile_aware", help="true/false tile_aware_selection", default="false")    
 
    args = argParser.parse_args(userArgs)
    outputPath = args.output_path
    outputName = args.output_file_name
    library = args.library
    tileAware = args.tile_aware

    if len(sys.argv) <= 6:
        inputFileName = args.input_file_name
        inputFileBaseName = os.path.basename(inputFileName)
        namePart, _ = os.path.splitext(inputFileBaseName)
    else:
        inputPath = args.input_logs
        networkName = args.network_name
        allLogs = [inputPath+'/'+filename for filename in os.listdir(inputPath) if networkName in filename]

    if len(sys.argv) <= 6:
        problemMapper = ProcessFile(inputFileName)
    else:
        problemMapper = ProcessFiles(allLogs)

    configPath = os.path.join(outputPath, "configs")
    if not os.path.exists(configPath):
        os.makedirs(configPath)
    scriptPath = os.path.join(outputPath, "scripts")
    if not os.path.exists(scriptPath):
        os.makedirs(scriptPath)
    sizePath = os.path.join(outputPath, "sizes")
    if not os.path.exists(sizePath):
        os.makedirs(sizePath)

    OutputConfigs(problemMapper, configPath, outputName, library,tileAware)
    
    if len(sys.argv) <= 6:
        OutputScript(problemMapper, scriptPath, namePart)
        OutputScript2(problemMapper, scriptPath, namePart+'2')
        OutputProblemDefinitions(problemMapper, sizePath, namePart)
    else:
        OutputScript(problemMapper, scriptPath, networkName)
        OutputScript2(problemMapper, scriptPath, networkName+'2')
        OutputProblemDefinitions(problemMapper, sizePath, networkName)

if __name__ == "__main__":
    RunMain()
