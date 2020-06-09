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


def SetDefaultStrides(problemDefinition, m, n, k):
    #assuming we don't encounter TT sizes, else this should be k,n,m
    if problemDefinition["transposeB"] == "T":
        return [m, n, m]
    elif problemDefinition["transposeA"] == "N":
        return [m, k, m]
    return [k, k, m]

def GetSize(problemDefinition,disableStrides="false",mfma="false"):
    m = int(problemDefinition["m"])
    n = int(problemDefinition["n"])
    k = int(problemDefinition["k"])
    b = 1

    #workaround to deal with bug in xdlops generator
    if mfma == True:
        if m == 1:
            m = 4
        if n == 1:
            n = 4

    if "batch_count" or "batch" in problemDefinition:
        b = int(problemDefinition["batch_count"])

    if disableStrides == "true":
        return [m, n, b, k]
    elif disableStrides == "false":
        if problemDefinition["lda"] != 0 and problemDefinition["ldb"] != 0 and problemDefinition["ldc"] != 0:
            lda = int(problemDefinition["lda"])
            ldb = int(problemDefinition["ldb"])
            ldc = int(problemDefinition["ldc"])
            if int(problemDefinition["ldd"]) != 0:
                ldd = int(problemDefinition["ldd"])
            else:
                ldd = ldc
        else:
            ld = SetDefaultStrides(problemDefinition, m, n, k)
            lda = ld[0]
            ldb = ld[1]
            ldc = ld[2]
            ldd = ldc

    return [m, n, b, k, ldd, ldc, lda, ldb]

def ClassifySize(size,mfma="false"):
    m = size[0]
    n = size[1]
    b = size[2]
    k = size[3]

    sizeKey = "none"

    scale = m * n

    tiny = 32 * 32
    small = 128 * 128
    medium = 512 * 512

    if mfma == "true":
        sizeKey = "matrix"
    elif b > 1:
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

def generateBenchmarkGroupFromScheme(scheme,tileAware="false"):
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
            "WorkGroupMapping": [1,8],
            "DepthU": [4,8,16,32],
            "VectorWidth": [-1],
            "GlobalSplitU": [1],
            "GlobalReadVectorWidth": [-1],
            "FractionalLoad": [1],
            "PrefetchLocalRead": [False,True]}

    return scheme

def generateMfmaScheme():
    scheme={"EdgeType": ["ShiftPtr"],
            "KernelLanguage": ["Assembly"],
            "LoopTail": [True],
            "WorkGroupMapping": [1,8],
            "DepthU": [8,16,32],
            "SuppressNoLoadLoop": [True,False],
            "OptNoLoadLoop": [0,1],
            "VectorWidth": [-1],
            "GlobalSplitU": [1],
            "GlobalReadVectorWidth": [-1],
            "FractionalLoad": [1],
            "PrefetchLocalRead": [False,True]}

    return scheme

def updateProblemGroupFromKey(problemKey,sizeKey,problemGroup,sizeList,tileAware="false",mfma="false",rk="false"):
    _ , transposeA, transposeB, dType = problemKey

    transposeType = "%s%s" % (transposeA.lower(),transposeB.lower())
    benchmarkGroup = None

    if transposeType == "tn" and rk == "true" and mfma == "false":
        scheme = generateDefaultScheme()
        scheme["GlobalSplitU"] = [1,3]
        scheme["ReplacementKernel"] = [True]
        noExact = True
        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
        appendThreadTiles(benchmarkGroup, [[8,2],[8,4],[2,8],[4,8],[16,2],[16,4],[16,8],[2,16],[4,16],[8,16]])
        appendWorkGroups(benchmarkGroup, [[16,16,1],[8,8,1]])
        appendSizes(benchmarkGroup,sizeList,tileAware,noExact,rk)
        problemGroup.append(benchmarkGroup)

        benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
        appendThreadTiles(benchmarkGroup, [[4,4]])
        appendWorkGroups(benchmarkGroup, [[16,32,1]])
        appendSizes(benchmarkGroup,sizeList,tileAware,False,rk)
        problemGroup.append(benchmarkGroup)

    elif mfma == "true" and dType != "d":
        scheme = generateMfmaScheme()
        if dType == "h":
            scheme["AssertSummationElementMultiple"] = [2]
            scheme["AssertFree0ElementMultiple"] = [2]

        if transposeType == "tn" and rk == "true" and dType == "s":
            scheme["GlobalSplitU"] = [1,3]
            scheme["ReplacementKernel"] = [True]
            noExact = True
            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
            appendMatrixInstructions(benchmarkGroup, [[16, 16, 1, 4],[16, 16, 4, 1]])
            appendThreadTiles(benchmarkGroup, [[4,16],[8,16]])
            appendWorkGroups(benchmarkGroup, [[16,16,1]])
            appendSizes(benchmarkGroup,sizeList,tileAware,noExact,rk)
            problemGroup.append(benchmarkGroup)

            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
            appendMatrixInstructions(benchmarkGroup, [[16, 16, 1, 4],[16, 16, 4, 1]])
            appendThreadTiles(benchmarkGroup, [[4,16],[8,16]])
            appendWorkGroups(benchmarkGroup, [[16,16,1]])
            appendSizes(benchmarkGroup,sizeList,tileAware,False,rk)
            problemGroup.append(benchmarkGroup)
        else:
            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
            appendMatrixInstructions(benchmarkGroup, [[32, 32, 1, 2]])
            appendThreadTiles(benchmarkGroup, [[1,32],[2,32],[4,32],[1,64],[2,64]])
            appendWorkGroups(benchmarkGroup, [[16,16,1],[64,4,1]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
            problemGroup.append(benchmarkGroup)

            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
            appendMatrixInstructions(benchmarkGroup, [[32, 32, 2, 1]])
            appendThreadTiles(benchmarkGroup, [[1,32],[2,32],[4,32],[1,64],[2,64]])
            appendWorkGroups(benchmarkGroup, [[16,16,1]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
            problemGroup.append(benchmarkGroup)

            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
            appendMatrixInstructions(benchmarkGroup, [[16, 16, 1, 4]])
            appendThreadTiles(benchmarkGroup, [[4,16],[8,16],[2,32],[4,32],[2,64]])
            appendWorkGroups(benchmarkGroup, [[16,16,1],[64,4,1]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
            problemGroup.append(benchmarkGroup)

            benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
            appendMatrixInstructions(benchmarkGroup, [[16, 16, 4, 1]])
            appendThreadTiles(benchmarkGroup, [[4,16],[8,16],[2,32],[4,32],[2,64]])
            appendWorkGroups(benchmarkGroup, [[16,16,1]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
            problemGroup.append(benchmarkGroup)

    else:
        scheme = generateDefaultScheme()
        if dType == "h":
            scheme["AssertSummationElementMultiple"] = [2]
            scheme["AssertFree0ElementMultiple"] = [2]

        if sizeKey == "batch":
            if dType == "d":
                scheme["StaggerU"] = [0,32]
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[6,6],[6,4],[4,6],[8,4],[4,4],[4,8]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[16,8,1],[8,16,1],[16,32,1],[32,16,1]])
            else:
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[4,4],[4,6],[6,4],[4,8],[8,4],[8,8]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[16,8,2],[8,16,2],[4,16,4],[16,4,4],[8,8,4]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
        elif sizeKey == "tiny":
            scheme["GlobalSplitU"] = [1,4]
            if dType == "d":
                scheme["StaggerU"] = [0,32]
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[6,6],[6,4],[4,6],[8,4],[4,4],[4,8]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[16,8,1],[8,16,1],[16,32,1],[32,16,1]])
            else:
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[2,2],[4,2],[2,4],[4,4]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[32,8,4],[8,32,4],[8,8,4]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
        elif sizeKey == "small":
            scheme["GlobalSplitU"] = [1,4]
            if dType == "d":
                scheme["StaggerU"] = [0,32]
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[6,6],[6,4],[4,6],[8,4],[4,4],[4,8]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[16,8,1],[8,16,1],[16,32,1],[32,16,1]])
            else:
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[4,4],[4,6],[6,4],[4,8],[8,4],[8,8]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[4,16,4],[16,4,4],[8,8,4]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
        elif sizeKey == "medium":
            scheme["GlobalSplitU"] = [1,8]
            if transposeType == "tn" and dType != "d":
                scheme["DepthU"] = [8,16]
            if dType == "d":
                scheme["StaggerU"] = [0,32]
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[6,6],[6,4],[4,6],[8,4],[4,4],[4,8]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[16,8,1],[8,16,1],[16,32,1],[32,16,1]])
            else:
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[4,4],[4,6],[6,4],[4,8],[8,4],[8,8]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[8,16,2],[16,8,2],[8,8,4]])
            appendSizes(benchmarkGroup,sizeList,tileAware)
        else: #sizeKey == "large"
            if dType == "d":
                scheme["StaggerU"] = [0,32]
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[6,6],[6,4],[4,6],[8,4],[4,4],[4,8]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[16,8,1],[8,16,1],[32,16,1],[16,32,1]])
            else:
                benchmarkGroup = generateBenchmarkGroupFromScheme(scheme,tileAware)
                appendThreadTiles(benchmarkGroup, [[4,4],[6,4],[4,6],[4,8],[8,4],[8,8]])
                appendWorkGroups(benchmarkGroup, [[16,16,1],[16,8,2],[8,16,2],[8,8,4]])
            appendSizes(benchmarkGroup,sizeList,tileAware)

        problemGroup.append(benchmarkGroup)

def OutputConfigs(problemMapper, configPath, outputName, library, tileAware, mfma, rk, disableStrides):

    keys = list(problemMapper.keys())

    configDefs = {}

    for key in keys:
        lineDefinitions = problemMapper[key]
        sizeMapper = {}
        for problemDefinition in lineDefinitions:
            size =  GetSize(problemDefinition,disableStrides,mfma)
            if rk == "true":
                sizeKey = ClassifySize(size,rk)
            else:
                sizeKey = ClassifySize(size,mfma)
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
            if rk:
                newConfig.globalParameters["MergeFiles"] = True
            newConfig.libraryLogic = deepcopy(libraryLogicMapper[library])
            newConfig.libraryClient = True
            problemGroup = [problemType]
            newConfig.benchmarkProblems = [problemGroup]
            configDefs[configurationFilePath] = newConfig

        if mfma == "true" or rk == "true":
            updateProblemGroupFromKey(key,sizeKey,problemGroup,sizeMapper[sizeKey],tileAware,mfma,rk)
        else:
            for sizeKey in sizeMapper:
                sizeList = sizeMapper[sizeKey]
                updateProblemGroupFromKey(key,sizeKey,problemGroup,sizeList,tileAware,mfma,rk)

    for key in configDefs:
        newConfig = configDefs[key]
        newConfig.writeLibraryLogic(key)

def GetOutputFileName(outputPath, namePart, ext):
    fileName = namePart+".%s" % (ext)
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

def removeIter(lines):
    noiterlines = []
    separator = '-i'
    for line in lines:
        newline = line.split(separator, 1)[0]
        noiterlines.append(newline)
    return noiterlines

def OutputScript(problemMapper, scriptPath, namePart, disableStrides="false", probDef="both", initialization="rand_int"):
    keys = list(problemMapper.keys())

    scriptFileNames = []
    outputFileName = GetOutputFileName(scriptPath, namePart, "sh")
    outputFileName2 = GetOutputFileName(scriptPath, namePart+"-strided", "sh")
    outputFileName3 = GetOutputFileName(scriptPath, namePart+"-all", "sh")
    outputFileName4 = GetOutputFileName(scriptPath, namePart+"-verify", "sh")

    if probDef != "gemm":
        scriptFileNames.append(outputFileName2)
    if probDef != "batch":
        scriptFileNames.append(outputFileName)
    count = 0

    for key in keys:
        if disableStrides == "true":
            if  "ld" not in key or "stride" not in key:
                lineDefinitions = problemMapper[key]
        else:
            lineDefinitions = problemMapper[key]
        lines = ["#!/bin/bash",""]
        for problemDefinition in lineDefinitions:
            rocblas_call = BuildRocBLASBenchmarkCall(problemDefinition,disableStrides,initialization)
            lines.append(rocblas_call)
        noiterlines = removeIter(lines)
        with open(outputFileName, 'a') as f, open(outputFileName2, 'a') as g, open(outputFileName3, 'a') as h:
            for line in lines:
                if "strided" in line:
                    if "rocblas-bench" in line:
                        g.write("%s\n" % line)
                        h.write("%s\n" % line)
                    else:
                        g.write("%s\n" % line)
                        h.write("%s\n" % line)
                else:
                    if "rocblas-bench" in line:
                        f.write("%s\n" % line)
                        h.write("%s\n" % line)
                    else:
                        if "bash" in line:
                            if count == 0:
                                f.write("%s\n" % line)
                                g.write("%s\n" % line)
                                h.write("%s\n" % line)
                                count = 1
                        else:
                            f.write("%s\n" % line)
                            h.write("%s\n" % line)
        with open(outputFileName4, 'a') as i:
            for line in noiterlines:
                if "strided" in line:
                    if "rocblas-bench" in line:
                        i.write("%s -i 1 -v 1\n" % line)
                    else:
                        i.write("%s\n" % line)
                else:
                    if "rocblas-bench" in line:
                        i.write("%s -i 1 -v 1\n" % line)
                    else:
                        if "bash" in line:
                            if count == 1:
                                i.write("%s\n" % line)
                                count = 2
                        else:
                            i.write("%s\n" % line)
        noiterlines = []
        lines = []

    generateRunScript(scriptFileNames, scriptPath)

def OutputScript2(problemMapper, scriptPath, namePart, disableStrides="false", probDef="both", initialization="rand_int"):

    keys = list(problemMapper.keys())

    scriptFileNames = []
    outputFileName = GetOutputFileName(scriptPath, namePart, "sh")
    outputFileName2 = GetOutputFileName(scriptPath, namePart+"-strided", "sh")
    outputFileName3 = GetOutputFileName(scriptPath, namePart+"-all", "sh")
    outputFileName4 = GetOutputFileName(scriptPath, namePart+"-verify", "sh")

    if probDef != "gemm":
        scriptFileNames.append(outputFileName2)
    if probDef != "batch":
        scriptFileNames.append(outputFileName)
    count = 0

    for key in keys:
        if disableStrides == "true":
            if "ld" not in key or "stride" not in key:
                lineDefinitions = problemMapper[key]
        else:
            lineDefinitions = problemMapper[key]
        lines = ["#!/bin/bash",""]
        for problemDefinition in lineDefinitions:
            rocblas_call = BuildRocBLASBenchmarkCall(problemDefinition,disableStrides,initialization)
            lines.append(rocblas_call)
        noiterlines = removeIter(lines)
        with open(outputFileName, 'a') as f, open(outputFileName2, 'a') as g, open(outputFileName3, 'a') as h:
            for line in lines:
                if "strided" in line:
                    if "rocblas-bench" in line:
                        g.write("ROCBLAS_TENSILE_LIBPATH=${TENSILE_LIBRARY} %s\n" % line)
                        h.write("ROCBLAS_TENSILE_LIBPATH=${TENSILE_LIBRARY} %s\n" % line)
                    else:
                        g.write("%s\n" % line)
                        h.write("%s\n" % line)
                else:
                    if "rocblas-bench" in line:
                        f.write("ROCBLAS_TENSILE_LIBPATH=${TENSILE_LIBRARY} %s\n" % line)
                        h.write("ROCBLAS_TENSILE_LIBPATH=${TENSILE_LIBRARY} %s\n" % line)
                    else:
                        if "bash" in line:
                            if count == 0:
                                f.write("%s\n" % line)
                                g.write("%s\n" % line)
                                h.write("%s\n" % line)
                                count = 1
                        else:
                            f.write("%s\n" % line)
                            h.write("%s\n" % line)
        with open(outputFileName4, 'a') as i:
            for line in noiterlines:
                if "strided" in line:
                    if "rocblas-bench" in line:
                        i.write("ROCBLAS_TENSILE_LIBPATH=${TENSILE_LIBRARY} %s -i 1 -v 1\n" % line)
                    else:
                        i.write("%s\n" % line)
                else:
                    if "rocblas-bench" in line:
                        i.write("ROCBLAS_TENSILE_LIBPATH=${TENSILE_LIBRARY} %s -i 1 -v 1\n" % line)
                    else:
                        if "bash" in line:
                            if count == 1:
                                i.write("%s\n" % line)
                                count = 2
                        else:
                            i.write("%s\n" % line)
        noiterlines = []
        lines = []

    generateRunScript(scriptFileNames, scriptPath,'2')

def OutputProblemDefinitions(problemMapper, sizePath, namePart):

    keys = list(problemMapper.keys())
    outputFileName = GetOutputFileName(sizePath, namePart, "csv")

    for key in keys:
        lineDefinitions = problemMapper[key]
        output = open(outputFileName,"w+")
        writer = csv.DictWriter(output, fieldnames=rocblas_parameters, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(lineDefinitions)

def RunMain():

    userArgs = sys.argv[1:]

    argParser = argparse.ArgumentParser()

    if len(sys.argv) <= 11:
        argParser.add_argument("input_file_name", help="configuration file path")
    else:
        argParser.add_argument("input_logs", help="the input path for log files")
        argParser.add_argument("network_name", help="neural network name")

    argParser.add_argument("output_path", help="the output path")
    argParser.add_argument("output_file_name", help="the output file name")
    argParser.add_argument("library", help="the library Logic name")
    argParser.add_argument("tile_aware", help="true/false tile_aware_selection", default="false")
    argParser.add_argument("mfma", help="true/false mfma", default="false")
    argParser.add_argument("replacement_kernel", help="true/false replacement kernels", default="false")
    argParser.add_argument("disable_strides", help="true/false disable strides", default="false")
    argParser.add_argument("problem_definition", help="gemm, batch, or both", default="both")
    argParser.add_argument("initialization", help="rand_int or trig_float", default="rand_int")

    args = argParser.parse_args(userArgs)
    outputPath = args.output_path
    outputName = args.output_file_name
    library = args.library
    tileAware = args.tile_aware
    mfma = args.mfma
    rk = args.replacement_kernel
    disableStrides = args.disable_strides
    probDefinition = args.problem_definition
    initialization = args.initialization

    if len(sys.argv) <= 11:
        inputFileName = args.input_file_name
        inputFileBaseName = os.path.basename(inputFileName)
        namePart, _ = os.path.splitext(inputFileBaseName)
    else:
        inputPath = args.input_logs
        networkName = args.network_name
        allLogs = [inputPath+'/'+filename for filename in os.listdir(inputPath) if networkName in filename]

    if len(sys.argv) <= 11:
        problemMapper = ProcessFile(inputFileName)
    else:
        problemMapper = ProcessFiles(allLogs)

    configPath = os.path.join(outputPath, "configs")
    if not os.path.exists(configPath):
        os.makedirs(configPath)
    scriptPath = os.path.join(outputPath, "scripts")
    if not os.path.exists(scriptPath):
        os.makedirs(scriptPath)
    scriptPath2 = os.path.join(outputPath, "scripts2")
    if not os.path.exists(scriptPath2):
        os.makedirs(scriptPath2)
    sizePath = os.path.join(outputPath, "sizes")
    if not os.path.exists(sizePath):
        os.makedirs(sizePath)

    OutputConfigs(problemMapper,configPath,outputName,library,tileAware,mfma,rk,disableStrides)

    if len(sys.argv) <= 11:
        OutputScript(problemMapper, scriptPath, namePart, disableStrides, probDefinition, initialization)
        OutputScript2(problemMapper, scriptPath2, namePart+'2', disableStrides, probDefinition, initialization)
        OutputProblemDefinitions(problemMapper, sizePath, namePart)
    else:
        OutputScript(problemMapper, scriptPath, networkName, disableStrides, probDefinition, initialization)
        OutputScript2(problemMapper, scriptPath2, networkName+'2', disableStrides, probDefinition, initialization)
        OutputProblemDefinitions(problemMapper, sizePath, networkName)

if __name__ == "__main__":
    RunMain()
