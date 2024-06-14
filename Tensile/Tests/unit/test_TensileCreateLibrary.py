################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
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
import yaml
import contextlib
import uuid
import shutil

from pathlib import Path

mylogger = logging.getLogger()

def test_loadSolutions(caplog, useGlobalParameters):
    with useGlobalParameters():
        mylogger.debug("this is a test of debug log")
        mylogger.info("this is some info")
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
        solutionsFilePath = os.path.join(dataDir, "solutions", "solutions_nn_3.yaml")

        fileSolutions = LibraryIO.parseSolutionsFile(solutionsFilePath)
        solutions = fileSolutions[1]
        kernels, _, _ = TensileCreateLibrary.generateKernelObjectsFromSolutions(solutions)
        assert len(solutions) == 3
        assert len(kernels) == 3


        _, kernelWriterAssembly, \
            _, _ = TensileCreateLibrary.getKernelWriters(solutions, kernels)

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
    Common.globalParameters["CodeObjectVersion"] = "default"
    Common.globalParameters["YAML"] = True
    Common.globalParameters["CxxCompiler"] = "amdclang++"
    Common.assignGlobalParameters({})

    libraryWorkingPath = tmpdir.mkdir("lib")
    buildWorkingPath = tmpdir.mkdir("build")


    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    solutionsFilePath = os.path.join(dataDir, "solutions", "solutions_nn_3.yaml")

    fileSolutions = LibraryIO.parseSolutionsFile(solutionsFilePath)
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


    metadataFile = LibraryIO.readYAML(metadataFilepath)
    problemTypeDict = metadataFile["ProblemType"]
    sizes = [{"Exact": [196, 256, 64, 1024]}]
    problemSizes = SolutionStructs.ProblemSizes(problemTypeDict, sizes)

    dataFilePath = os.path.join(dataWorkingPath, "results.csv")
    configFile = os.path.join(configWorkingPath, "ClientParameters.ini")
    ClientWriter.CreateBenchmarkClientParametersForSizes(testDataPath, problemSizes, dataFilePath, configFile)

    assert os.path.exists(configFile) == 1

def test_verifyManifest():
  
    manifestFile = Path("test_manifest.txt")
    testFoo = Path("foo.asm")
    testBar = Path("bar.asm")

    # ensure clean state before running test
    with contextlib.suppress(FileNotFoundError):
        os.remove(manifestFile)
        os.remove(testFoo)    
        os.remove(testBar)

    # Verification should fail if the manifest can't be found
    with pytest.raises(FileNotFoundError, match=r"(.*) No such file or directory: 'test_manifest.txt'"):
        TensileCreateLibrary.verifyManifest(manifestFile)

    # Create an empty manifest 
    with open(manifestFile, mode="x") as manifest:
           
        assert TensileCreateLibrary.verifyManifest(manifestFile), "an empty manifest should always succeed"

        # add to file manifest that is not on disk
        manifest.write("foo.asm\n")
        manifest.flush()
        assert not TensileCreateLibrary.verifyManifest(manifestFile), "file in manifest are on disk, but shouldn't be"
    
        with open(testFoo, mode="x"):
            assert TensileCreateLibrary.verifyManifest(manifestFile), "file in manifest isn't on disk, but should be"
    
        manifest.write("bar.asm\n")
        manifest.flush()
        assert not TensileCreateLibrary.verifyManifest(manifestFile), "bar.asm in manifest should not be on disk"
  
    with open(testBar, mode="x"):
      assert TensileCreateLibrary.verifyManifest(manifestFile), "files in manifest isn't on disk, but should be"

    with open(manifestFile, "a") as generatedFile:  
        for filePath in range(5):
            generatedFile.write("%s\n" %(filePath) )

    assert not TensileCreateLibrary.verifyManifest(manifestFile), "files in manifest are on disk, but shouldn't be"

def test_findLogicFiles():

    def setup():
        baseDir = Path("no-commit-test-logic-files")

        # Start with clean state
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(baseDir)
        baseDir.mkdir()

        lazyLoading = False
        experimentalDir = "/experimental/"
        return baseDir, lazyLoading, experimentalDir

    def outputMatchesOldLogic1():
        baseDir, lazyLoading, experimentalDir = setup()
        logicArchs = set(Common.architectureMap.values())
        logicArchs.add("hip")
        logicArchs.remove("_")  # Remove the value `_` associated with key `all`

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / d, "foo", "yaml")

        result = TensileCreateLibrary.findLogicFiles(baseDir, logicArchs, lazyLoading, experimentalDir)
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return result == expected

    def outputMatchesOldLogic2():
        baseDir, lazyLoading, experimentalDir = setup()
        logicArchs = set(Common.architectureMap.values())
        logicArchs.add("hip")
        logicArchs.remove("_")  # Remove the value `_` associated with key `all`

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / d, d, "yaml")

        result = TensileCreateLibrary.findLogicFiles(baseDir, logicArchs, lazyLoading, experimentalDir)
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return result == expected
    
    def outputMatchesOldLogic3():
        baseDir, lazyLoading, experimentalDir = setup()
        logicArchs = set(Common.architectureMap["all"])

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / d, d, "yaml")

        result = TensileCreateLibrary.findLogicFiles(baseDir, logicArchs, lazyLoading, experimentalDir)
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return result == expected

    def verifyYamlAndYml():
        baseDir, lazyLoading, experimentalDir = setup()
        # Create directory structure with files that *don't* have arch in filename
        logicArchs = set(Common.architectureMap.values())
        logicArchs.add("hip")
        logicArchs.remove("_")

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / "yaml" / d, d, "yaml")
            createDirectoryWithYamls(baseDir / "yml" / d, d, "yml")

        result = TensileCreateLibrary.findLogicFiles(baseDir, logicArchs, lazyLoading, experimentalDir)
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return len(result) == len(expected)*2

    assert outputMatchesOldLogic1(), "Output differs from old logic, not backwards compatible."
    assert outputMatchesOldLogic2(), "Output differs from old logic, not backwards compatible."
    assert outputMatchesOldLogic3(), "Output differs from old logic, not backwards compatible."
    assert verifyYamlAndYml(), "Output should have twice as many files as old logic (which only parses .yaml)"


# ----------------
# Helper functions
# ----------------
def createDirectoryWithYamls(currentDir, prefix, ext, depth=3, nChildren=3):
    def recurse(currentDir, depth, nChildren):
        if depth == 0:
            return

        currentDir.mkdir(parents=True, exist_ok=True)
        file = f"{prefix}_{str(uuid.uuid4().hex)}.{ext}"
        with open(currentDir/file, mode="w"):
            pass

        for n in range(nChildren):
            recurse(currentDir / str(n), depth - 1, nChildren)
    recurse(currentDir, depth, nChildren)


def findLogicFiles_oldLogic(logicPath, logicArchs, lazyLoading, experimentalDir):
    # Recursive directory search
    logicFiles = []
    for root, dirs, files in os.walk(str(logicPath)):
        logicFiles += [os.path.join(root, f) for f in files
                        if os.path.splitext(f)[1]==".yaml" \
                        and (any(logicArch in os.path.splitext(f)[0] for logicArch in logicArchs) \
                        or "hip" in os.path.splitext(f)[0]) ]

    if not lazyLoading:
        logicFiles = [f for f in logicFiles if not experimentalDir in f]
    return logicFiles
