################################################################################
# Copyright 2020 Advanced Micro Devices, Inc. All rights reserved.
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

import logging
import pytest
import os
import glob
import Tensile.TensileCreateLibrary as TensileCreateLibrary
import Tensile.LibraryIO as LibraryIO
import Tensile.Common as Common
import Tensile.ClientWriter as ClientWriter
import Tensile.SolutionStructs as SolutionStructs
import Tensile.BenchmarkProblems as BenchmarkProblems
import Tensile.BenchmarkStructs as BenchmarkStructs
import yaml

mylogger = logging.getLogger()

def test_assignParameters():
    problemTypeConfig = \
        {"Batched": True, "DataType": "s", "OperationType": "GEMM", "TransposeA": False, "TransposeB": False, "UseBeta": True}

    benchmarkCommonParameters = [{"LoopTail": [True]}, {"KernelLanguage": ["Assembly"]}, \
        {"EdgeType": ["ShiftPtr"]}, {"GlobalSplitU": [1]}, {"VectorWidth": [-1]}, {"FractionalLoad": [1]}, \
        {"PrefetchGlobalRead": [True]}]

    configForkParameters = \
        [{"WorkGroup": [[16, 16, 1]]}, {"ThreadTile": [[4, 4],[8, 8]]}]

    problemTypeObj, hardcodedParameters, initialSolutionParameters = \
        BenchmarkStructs.assignParameters(problemTypeConfig, benchmarkCommonParameters, configForkParameters)


    assert problemTypeObj != None
    assert hardcodedParameters != None
    assert initialSolutionParameters != None

def test_generateSolutions():

    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    problemTypeFilePath = os.path.join(dataDir, "library_data", "problemType.yaml")
    hardcodedParametersFilePath = os.path.join(dataDir, "library_data", "hardcodedParameters.yaml")
    initialSolutionParametersFilePath = os.path.join(dataDir, "library_data", "initialSolutionParameters.yaml")

    problemType = LibraryIO.readConfig(problemTypeFilePath)["ProblemType"]
    problemTypeObject = SolutionStructs.ProblemType(problemType)
    hardcodedParameters = LibraryIO.readConfig(hardcodedParametersFilePath)
    initialSolutionParameters = LibraryIO.readConfig(initialSolutionParametersFilePath)

    solutionList = BenchmarkProblems.generateForkedSolutions (problemTypeObject, hardcodedParameters, [initialSolutionParameters])

    assert len(solutionList) == 2

def test_loadSolutions(caplog):

    mylogger.debug("this is a test of debug log")
    mylogger.info("this is some info")
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    solutionsFilePath = os.path.join(dataDir, "solutions", "solutions_nn_3.yaml")

    fileSolutions = LibraryIO.readSolutions(solutionsFilePath)
    solutions = fileSolutions[1]
    kernels, _, _ = TensileCreateLibrary.generateKernelObjectsFromSolutions(solutions)
    assert len(solutions) == 3
    assert len(kernels) == 3


    solutionWriter, _, kernelWriterAssembly, \
        _, _ = TensileCreateLibrary.getSolutionAndKernelWriters(solutions, kernels)

    expectedSolutionName0 = "Cijk_Ailk_Bljk_SB_MT128x128x2_SE_TT8_8_WG16_16_1"
    expectedSolutionName1 = "Cijk_Ailk_Bljk_SB_MT64x64x2_SE_TT4_4_WG16_16_1"
    expectedSolutionName2 = "Cijk_Ailk_Bljk_SB_MT64x64x2_SE_TT4_8_WG16_8_1"

    actualSolutionName0 = solutionWriter.getSolutionName(solutions[0])
    actualSolutionName1 = solutionWriter.getSolutionName(solutions[1])
    actualSolutionName2 = solutionWriter.getSolutionName(solutions[2])

    assert expectedSolutionName0 == actualSolutionName0
    assert expectedSolutionName1 == actualSolutionName1
    assert expectedSolutionName2 == actualSolutionName2

    expectedKernelName0 = "Cijk_Ailk_Bljk_SB_MT128x128x2_SE_K1_TT8_8_WG16_16_1"
    expectedKernelName1 = "Cijk_Ailk_Bljk_SB_MT64x64x2_SE_K1_TT4_4_WG16_16_1"
    expectedKernelName2 = "Cijk_Ailk_Bljk_SB_MT64x64x2_SE_K1_TT4_8_WG16_8_1"

    actualKernelName0 = kernelWriterAssembly.getKernelName(kernels[0])
    actualKernelName1 = kernelWriterAssembly.getKernelName(kernels[1])
    actualKernelName2 = kernelWriterAssembly.getKernelName(kernels[2])

    assert expectedKernelName0 == actualKernelName0
    assert expectedKernelName1 == actualKernelName1
    assert expectedKernelName2 == actualKernelName2

@pytest.mark.skip(reason="System issue with find assempler called when assigning defaults")
def test_WriteClientLibraryFromSolutions(tmpdir):
    Common.globalParameters["MergeFiles"] = True
    Common.globalParameters["CodeObjectVersion"] = "V3"
    Common.globalParameters["YAML"] = True
    Common.globalParameters["CxxCompiler"] = "hipcc"
    Common.assignGlobalParameters({})

    libraryWorkingPath = tmpdir.mkdir("lib")
    buildWorkingPath = tmpdir.mkdir("build")


    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    solutionsFilePath = os.path.join(dataDir, "solutions", "solutions_nn_3.yaml")

    fileSolutions = LibraryIO.readSolutions(solutionsFilePath)
    solutions = fileSolutions[1]

    Common.setWorkingPath(buildWorkingPath)
    TensileCreateLibrary.WriteClientLibraryFromSolutions(solutions, libraryWorkingPath)
    Common.popWorkingPath()

    tensileLibraryPath = os.path.join(libraryWorkingPath, "library")

    hsacoFiles = glob.glob(tensileLibraryPath + "/*hsaco")
    assert (len(hsacoFiles) > 0)

    coFiles = glob.glob(tensileLibraryPath + "/*TensileLibrary*co")
    assert (len(coFiles) > 0)

    tensileYamlFilePath = os.path.join(tensileLibraryPath, "TensileLibrary.yaml")
    assert os.path.exists(tensileYamlFilePath) == 1

    config = None
    try:
        stream = open(tensileYamlFilePath, "r")
    except IOError:
        mylogger.error("Cannot open file: %s" % tensileYamlFilePath)
    config = yaml.load(stream, yaml.SafeLoader)
    stream.close()
    actualSolutions = config["solutions"]

    assert (len(actualSolutions) == 3)

    metadataYamlFilePath = os.path.join(tensileLibraryPath, "metadata.yaml")
    assert os.path.exists(metadataYamlFilePath) == 1

    metadata = None
    try:
        stream = open(metadataYamlFilePath, "r")
    except IOError:
        mylogger.error("Cannot open file: %s" % metadataYamlFilePath)
    metadata = yaml.load(stream, yaml.SafeLoader)
    stream.close()
    actualProblemType = metadata["ProblemType"]

    assert (len(actualProblemType) > 0)

def test_CreateBenchmarkClientParametersForSizes(tmpdir):

    Common.globalParameters["CurrentISA"] = (9,0,6)
    dataWorkingPath = tmpdir.mkdir("Data")
    configWorkingPath = tmpdir.mkdir("run_configs")
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    testDataPath = os.path.join(dataDir, "library_data")
    libraryPath = os.path.join(testDataPath, "library")
    metadataFilepath = os.path.join(libraryPath, "metadata.yaml")


    metadataFile = LibraryIO.readConfig(metadataFilepath)
    problemTypeDict = metadataFile["ProblemType"]
    sizes = [{"Exact": [196, 256, 64, 1024]}]
    problemSizes = SolutionStructs.ProblemSizes(problemTypeDict, sizes)

    dataFilePath = os.path.join(dataWorkingPath, "results.csv")
    configFile = os.path.join(configWorkingPath, "ClientParameters.ini")
    ClientWriter.CreateBenchmarkClientParametersForSizes(testDataPath, problemSizes, dataFilePath, configFile)

    assert os.path.exists(configFile) == 1




