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

from copy import deepcopy
import logging
import pytest
import os
import glob
import Tensile.TensileCreateLibrary as TensileCreateLibrary
import Tensile.LibraryIO as LibraryIO
import Tensile.Common as Common
import Tensile.ClientWriter as ClientWriter
import Tensile.SolutionStructs as SolutionStructs
import Tensile.SolutionLibrary as SolutionLibrary
import yaml
import contextlib
import uuid
import shutil

from pathlib import Path
from typing import List

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

        _, kernelWriterAssembly, _, _ = TensileCreateLibrary.getKernelWriters(solutions, kernels)

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
    assert len(hsacoFiles) > 0

    coFiles = glob.glob(tensileLibraryPath + "/*TensileLibrary*co")
    assert len(coFiles) > 0

    tensileYamlFilePath = os.path.join(tensileLibraryPath, "TensileLibrary.yaml")
    assert os.path.exists(tensileYamlFilePath) == 1

    config = None
    try:
        stream = open(tensileYamlFilePath, "r")
    except IOError:
        mylogger.error("Cannot open file: %s" % tensileYamlFilePath)
    config = yaml.load(stream, yaml.CSafeLoader)
    stream.close()
    actualSolutions = config["solutions"]

    assert len(actualSolutions) == 3

    metadataYamlFilePath = os.path.join(tensileLibraryPath, "metadata.yaml")
    assert os.path.exists(metadataYamlFilePath) == 1

    metadata = None
    try:
        stream = open(metadataYamlFilePath, "r")
    except IOError:
        mylogger.error("Cannot open file: %s" % metadataYamlFilePath)
    metadata = yaml.load(stream, yaml.CSafeLoader)
    stream.close()
    actualProblemType = metadata["ProblemType"]

    assert len(actualProblemType) > 0


def test_CreateBenchmarkClientParametersForSizes(tmpdir):

    Common.globalParameters["CurrentISA"] = (9, 0, 6)
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
    ClientWriter.CreateBenchmarkClientParametersForSizes(
        testDataPath, problemSizes, dataFilePath, configFile
    )

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
    with pytest.raises(
        FileNotFoundError, match=r"(.*) No such file or directory: 'test_manifest.txt'"
    ):
        TensileCreateLibrary.verifyManifest(manifestFile)

    # Create an empty manifest
    with open(manifestFile, mode="x") as manifest:

        assert TensileCreateLibrary.verifyManifest(
            manifestFile
        ), "an empty manifest should always succeed"

        # add to file manifest that is not on disk
        manifest.write("foo.asm\n")
        manifest.flush()
        assert not TensileCreateLibrary.verifyManifest(
            manifestFile
        ), "file in manifest are on disk, but shouldn't be"

        with open(testFoo, mode="x"):
            assert TensileCreateLibrary.verifyManifest(
                manifestFile
            ), "file in manifest isn't on disk, but should be"

        manifest.write("bar.asm\n")
        manifest.flush()
        assert not TensileCreateLibrary.verifyManifest(
            manifestFile
        ), "bar.asm in manifest should not be on disk"

    with open(testBar, mode="x"):
        assert TensileCreateLibrary.verifyManifest(
            manifestFile
        ), "files in manifest isn't on disk, but should be"

    with open(manifestFile, "a") as generatedFile:
        for filePath in range(5):
            generatedFile.write("%s\n" % (filePath))

    assert not TensileCreateLibrary.verifyManifest(
        manifestFile
    ), "files in manifest are on disk, but shouldn't be"


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

        result = TensileCreateLibrary.findLogicFiles(
            baseDir, logicArchs, lazyLoading, experimentalDir
        )
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return result == expected

    def outputMatchesOldLogic2():
        baseDir, lazyLoading, experimentalDir = setup()
        logicArchs = set(Common.architectureMap.values())
        logicArchs.add("hip")
        logicArchs.remove("_")  # Remove the value `_` associated with key `all`

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / d, d, "yaml")

        result = TensileCreateLibrary.findLogicFiles(
            baseDir, logicArchs, lazyLoading, experimentalDir
        )
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return result == expected

    def outputMatchesOldLogic3():
        baseDir, lazyLoading, experimentalDir = setup()
        logicArchs = set(Common.architectureMap["all"])

        for d in logicArchs:
            createDirectoryWithYamls(baseDir / d, d, "yaml")

        result = TensileCreateLibrary.findLogicFiles(
            baseDir, logicArchs, lazyLoading, experimentalDir
        )
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

        result = TensileCreateLibrary.findLogicFiles(
            baseDir, logicArchs, lazyLoading, experimentalDir
        )
        expected = findLogicFiles_oldLogic(baseDir, logicArchs, lazyLoading, experimentalDir)
        return len(result) == len(expected) * 2

    assert outputMatchesOldLogic1(), "Output differs from old logic, not backwards compatible."
    assert outputMatchesOldLogic2(), "Output differs from old logic, not backwards compatible."
    assert outputMatchesOldLogic3(), "Output differs from old logic, not backwards compatible."
    assert (
        verifyYamlAndYml()
    ), "Output should have twice as many files as old logic (which only parses .yaml)"


def test_sanityCheck():
    # setup some dummy lists with files
    srcPaths = ["foo.hsaco", "bar.hsaco"]
    asmPaths = ["baz.co", "gru.co"]
    coPathsMatch = ["foo.hsaco", "bar.hsaco", "baz.co", "gru.co"]
    coPathsExtra = ["foo.hsaco", "bar.hsaco", "baz.co", "gru.co", "tux.hsaco"]
    coPathsMissing = ["foo.hsaco", "bar.hsaco", "baz.co"]

    TensileCreateLibrary.sanityCheck(srcPaths, asmPaths, coPathsMatch, False)
    # Ensure that old logic also succeeds
    sanityCheck_oldLogic(srcPaths, asmPaths, coPathsMatch, False)

    with pytest.raises(ValueError, match=r"(.*) unexpected code object files: \['tux.hsaco'\]"):
        TensileCreateLibrary.sanityCheck(srcPaths, asmPaths, coPathsExtra, False)
    # Ensure that old logic also fails
    with pytest.raises(Exception):
        try:
            sanityCheck_oldLogic(srcPaths, asmPaths, coPathsExtra, False)
        except:
            raise Exception

    with pytest.raises(ValueError, match=r"(.*) missing expected code object files: \['gru.co'\]"):
        TensileCreateLibrary.sanityCheck(srcPaths, asmPaths, coPathsMissing, False)
    # Ensure that old logic also fails
    with pytest.raises(Exception):
        try:
            sanityCheck_oldLogic(srcPaths, asmPaths, coPathsMissing, False)
        except:
            raise Exception


def test_generateClientConfig():

    outputPath: Path = Path.cwd() / "my-output"
    masterLibrary: Path = outputPath / "masterlib.data"
    codeObjectFiles = ["library/foo.hsaco", "library/bar.co"]
    configFile = outputPath / "best-solution.ini"

    cleanClientConfigTest(outputPath, masterLibrary, codeObjectFiles, configFile)

    with pytest.raises(
        FileNotFoundError,
        match=r"(.*) No such file or directory: '(.*)/my-output/best-solution.ini'",
    ):
        TensileCreateLibrary.generateClientConfig(outputPath, masterLibrary, codeObjectFiles)

    setupClientConfigTest(outputPath, masterLibrary, codeObjectFiles)

    TensileCreateLibrary.generateClientConfig(outputPath, masterLibrary, codeObjectFiles)

    assert configFile.is_file(), "{configFile} was not generated"

    with open(configFile, "r") as f:
        result = f.readlines()
        assert "library-file" in result[0], "missing library-file entry"
        assert "code-object" in result[1], "missing code-object entry"
        assert "code-object" in result[2], "missing code-object entry"
        assert "best-solution" in result[3], "missing best-solution entry"

    cleanClientConfigTest(outputPath, masterLibrary, codeObjectFiles, configFile)


class MasterLibraryMock:
    def __init__(self, libraries, data):
        self.lazyLibraries = libraries
        self.data = data


def test_generateMasterFileList():

    archs = ["arch0", "arch1"]
    libraries = {
        "arch0": MasterLibraryMock({"mylib0": MasterLibraryMock(None, 2)}, 0),
        "arch1": MasterLibraryMock({"mylib1": MasterLibraryMock(None, 3)}, 1),
    }

    result = TensileCreateLibrary.generateMasterFileList(libraries, archs, lazy=False)
    for idx, t in enumerate(result):
        assert t[0] == "TensileLibrary_arch" + str(idx), "Incorrect naming for key."
        assert isinstance(t[1], MasterLibraryMock), "Incorrect type for value."
        assert t[1].data == idx, "Incorrect data."

    result = TensileCreateLibrary.generateMasterFileList(libraries, archs, lazy=True)

    for idx, t in enumerate(result[0:2]):
        assert t[0] == "TensileLibrary_lazy_arch" + str(idx), "Incorrect naming for key."
        assert isinstance(t[1], MasterLibraryMock), "Incorrect type for value."
        assert t[1].data == idx, "Incorrect data."

    for idx, t in enumerate(result[2:4]):
        assert t[0] == "mylib" + str(idx), "Incorrect naming for key."
        assert isinstance(t[1], MasterLibraryMock), "Incorrect type for value."
        assert t[1].data == (idx + 2), "Incorrect data."


def test_logicDataAndSolutionsConstruction(initGlobalParametersForTCL):

    def testCase1(logicFiles: List[LibraryIO.LibraryLogic], separateArch: bool):
        # clear the set to prevent testing errors caused
        # by the fact that ArchitectureSet is shared across
        # instances of MasterSolutionLibrary.
        SolutionLibrary.MasterSolutionLibrary.ArchitectureSet.clear()

        masterLibraries = TensileCreateLibrary.makeMasterLibraries(
            logicFiles, separate=separateArch
        )
        arch = "gfx900" if separateArch else "full"

        if separateArch:
            assert (
                len(masterLibraries[arch].solutions.values()) == 17
            ), f"There should be 17 solutions prior to adding the fallback for {arch}."

            TensileCreateLibrary.addFallback(masterLibraries)

            assert (
                len(masterLibraries[arch].solutions.values()) == 19
            ), f"There should be 19 solutions after adding the fallback for {arch}."
        else:
            assert (
                len(masterLibraries[arch].solutions.values()) == 19
            ), f"There should be 19 solutions for {arch}."

        solutions = TensileCreateLibrary.generateSolutions(masterLibraries, separate=separateArch)
        assert isinstance(solutions, list), "generateSolutions should return a list."
        assert len(solutions) == 19, "There should be 19 solutions after adding the fallback."

    def testCase2(yamlFiles: List[str], separateArch: bool):
        # clear the set to prevent testing errors caused
        # by the fact that ArchitectureSet is shared across
        # instances of MasterSolutionLibrary.
        SolutionLibrary.MasterSolutionLibrary.ArchitectureSet.clear()

        masterLibraries = TensileCreateLibrary.generateLogicData(
            yamlFiles, version="foo", printLevel=0, separate=separateArch
        )
        arch = "gfx900" if separateArch else "full"

        assert (
            len(masterLibraries[arch].solutions.values()) == 19
        ), f"There should be 19 solutions for {arch}."

        solutions = TensileCreateLibrary.generateSolutions(masterLibraries, separate=separateArch)
        assert isinstance(solutions, list), "generateSolutions should return a list."
        assert len(solutions) == 19, "There should be 19 solutions after adding the fallback."

    def testCase3(logicFiles: List[LibraryIO.LibraryLogic]):
        # clear the set to prevent testing errors caused
        # by the fact that ArchitectureSet is shared across
        # instances of MasterSolutionLibrary.
        SolutionLibrary.MasterSolutionLibrary.ArchitectureSet.clear()

        masterLibraries = TensileCreateLibrary.makeMasterLibraries(logicFiles, separate=True)
        arch = "gfx900"

        assert (
            len(masterLibraries[arch].lazyLibraries.keys()) == 1
        ), f"There should be 1 key prior to adding the fallback for {arch}."
        assert (
            len(next(iter(masterLibraries[arch].lazyLibraries.values())).solutions.values()) == 17
        ), f"There should be 17 solutions prior to adding the fallback for {arch}."

        TensileCreateLibrary.addFallback(masterLibraries)

        assert (
            len(masterLibraries[arch].lazyLibraries.keys()) == 2
        ), f"There should be 2 keys after adding the fallback for {arch}."

        for name, lib in masterLibraries[arch].lazyLibraries.items():
            if "fallback" in name:
                assert len(lib.solutions.values()) == 2, "There should be 2 fallback solutions."
            else:
                assert len(lib.solutions.values()) == 17, "There should be 17 gfx900 solutions."

        solutions = TensileCreateLibrary.generateSolutions(masterLibraries, separate=True)
        assert isinstance(solutions, list), "generateSolutions should return a list."
        assert len(solutions) == 19, "There should be 19 solutions after adding the fallback."

    requiredArgs = ["--jobs=2", "/unused/logic/path", "/unused/output/path", "HIP"]
    rootPath = Path(__file__).parent.parent / "test_data" / "unit" / "solutions"
    yamlFiles = [
        rootPath / f for f in ["vega10_Cijk_Ailk_Bjlk_CB_GB.yaml", "hip_Cijk_Ailk_Bjlk_CB_GB.yaml"]
    ]

    with initGlobalParametersForTCL(["--architecture=gfx900"] + requiredArgs):
        logicFiles = TensileCreateLibrary.parseLibraryLogicFiles(yamlFiles)
        assert len(logicFiles) == 2, "The length of the logic files list is incorrect."

        for s in [True, False]:
            testCase1(logicFiles, separateArch=s)
            testCase2(yamlFiles, separateArch=s)

    with initGlobalParametersForTCL(
        ["--architecture=gfx900", "--lazy-library-loading"] + requiredArgs
    ):
        logicFiles = TensileCreateLibrary.parseLibraryLogicFiles(yamlFiles)
        assert len(logicFiles) == 2, "The length of the logic files list is incorrect."
        testCase3(logicFiles)


@pytest.fixture
def unittestPath(request):
    """Returns the path to the directory containing the current test file"""
    return request.path.parent


@pytest.fixture
def setupSolutionsAndKernels(unittestPath):
    """Reusable logic for setting up testable solutions and kernels"""
    Common.assignGlobalParameters({})
    _, _, _, _, _, lib = LibraryIO.parseLibraryLogicFile(
        unittestPath.parent / "test_data" / "unit" / "aldebaran_Cijk_AlikC_Bljk_ZB_GB.yaml"
    )
    solutions = [sol.originalSolution for sol in lib.solutions.values()]
    kernels, _, _ = TensileCreateLibrary.generateKernelObjectsFromSolutions(solutions)
    kernelWriterSource, kernelWriterAssembly, _, _ = TensileCreateLibrary.getKernelWriters(
        solutions, kernels
    )
    return solutions, kernels, kernelWriterAssembly, kernelWriterSource


def test_prepAsm(setupSolutionsAndKernels):
    solutions, kernels, kernelWriterAssembly, kernelWriterSource = setupSolutionsAndKernels
    buildPath = Path("no-commit-prep-asm")
    buildPath.mkdir(exist_ok=True)

    def testLinux():
        TensileCreateLibrary.prepAsm(
            kernelWriterAssembly, True, Path("no-commit-prep-asm"), (9, 0, 10), 1
        )

        expected = """#!/bin/sh 
# usage: asm-new.sh kernelName(no extension) [--wave32]
f=$1
shift
if [ ! -z "$1" ] && [ "$1" = "--wave32" ]; then
    wave=32
    shift
else
    wave=64
fi
h=gfx90a
if [ $wave -eq 32 ]; then
/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx90a -mno-wavefrontsize64 -c -o $f.o $f.s
else
/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx90a -mwavefrontsize64 -c -o $f.o $f.s
fi
/opt/rocm/bin/amdclang++ -target amdgcn-amd-amdhsa -Xlinker --build-id -o $f.co $f.o
cp $f.co ../../../library/${f}_$h.co
mkdir -p ../../../asm_backup && cp $f.s ../../../asm_backup/$f.s
"""

        with open(buildPath / "assembly" / "asm-new.sh", "r") as f:
            contents = f.read()
            assert contents == expected, "Assembler script doesn't match expectation"

    def testWindows():
        TensileCreateLibrary.prepAsm(
            kernelWriterAssembly, False, Path("no-commit-prep-asm"), (9, 0, 10), 1
        )

        expected = """@echo off
set f=%1

set arg2=--wave64
if [%2] NEQ [] set arg2=%2

set /A wave=64
if %arg2% EQU --wave32 set /A wave=32

set h=gfx90a
if %wave% == 32 (/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx90a -mno-wavefrontsize64 -c -o %f%.o %f%.s) else (/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx90a -mwavefrontsize64 -c -o %f%.o %f%.s)
/opt/rocm/bin/amdclang++ -target amdgcn-amd-amdhsa -Xlinker --build-id -o %f%.co %f%.o
copy %f%.co ..\..\..\library\%f%_%h%.co
"""

        with open(buildPath / "assembly" / "asm-new.bat", "r") as f:
            contents = f.read()
            assert contents == expected, "Assembler script doesn't match expectation"

    testLinux()
    testWindows()


def test_markDuplicateKernels(setupSolutionsAndKernels):
    _, kernels, kernelWriterAssembly, _ = setupSolutionsAndKernels

    shortname_idx = 0
    custom_idx1 = 1
    custom_idx2 = 2

    # Use deepcopy here, otherwise when the entry is updated later, both entries will be
    # marked as duplicates.
    kernels.append(deepcopy(kernels[shortname_idx]))
    kernels[custom_idx1]["CustomKernelName"] = "DUPLICATE"
    kernels[custom_idx2]["CustomKernelName"] = "DUPLICATE"

    kernelsOut = TensileCreateLibrary.markDuplicateKernels(kernels, kernelWriterAssembly)

    assert len(kernelsOut) == len(kernels), "Lengths of input and output should match"
    for i, k in enumerate(kernelsOut):
        if i == custom_idx2:
            assert (
                k.duplicate == True
            ), f"Kernel with custom name {kernels[i]['CustomKernelName']} should be a duplicate, but isn't, found {kernelWriterAssembly.getKernelFileBase(k)} instead"
        elif i == len(kernels) - 1:
            assert (
                k.duplicate == True
            ), f"Shortened name {kernelWriterAssembly.getKernelFileBase(k)} wasn't located"
        elif k["KernelLanguage"] == "Assembly":
            assert (
                k.duplicate == False
            ), f"Kernel with name {kernels[i]['CustomKernelName']} should not be marked as a duplicate, but is"


# ----------------
# Helper functions
# ----------------
def setupClientConfigTest(outputPath, masterLibrary, codeObjectFiles):
    outputPath.mkdir()
    with open(masterLibrary, "w") as testFile:
        testFile.write("foo")
    (outputPath / "library").mkdir()
    for f in codeObjectFiles:
        with open(outputPath / f, "w") as testFile:
            testFile.write("foo")


def cleanClientConfigTest(outputPath: Path, masterLibrary, codeObjectFiles, configFile):
    with contextlib.suppress(FileNotFoundError):
        os.remove(configFile)
        os.remove(masterLibrary)
        for f in codeObjectFiles:
            os.remove(outputPath / f)
        next(outputPath.iterdir()).rmdir()
        outputPath.rmdir()


def createDirectoryWithYamls(currentDir, prefix, ext, depth=3, nChildren=3):
    def recurse(currentDir, depth, nChildren):
        if depth == 0:
            return

        currentDir.mkdir(parents=True, exist_ok=True)
        file = f"{prefix}_{str(uuid.uuid4().hex)}.{ext}"
        with open(currentDir / file, mode="w"):
            pass

        for n in range(nChildren):
            recurse(currentDir / str(n), depth - 1, nChildren)

    recurse(currentDir, depth, nChildren)


def findLogicFiles_oldLogic(logicPath, logicArchs, lazyLoading, experimentalDir):
    # Recursive directory search
    logicFiles = []
    for root, dirs, files in os.walk(str(logicPath)):
        logicFiles += [
            os.path.join(root, f)
            for f in files
            if os.path.splitext(f)[1] == ".yaml"
            and (
                any(logicArch in os.path.splitext(f)[0] for logicArch in logicArchs)
                or "hip" in os.path.splitext(f)[0]
            )
        ]

    if not lazyLoading:
        logicFiles = [f for f in logicFiles if not experimentalDir in f]
    return logicFiles


def sanityCheck_oldLogic(sourceLibPaths, asmLibPaths, codeObjectFiles, genSourcesAndExit):
    bothLibSet = set(sourceLibPaths + asmLibPaths)
    setA = set(map(os.path.normcase, set(codeObjectFiles)))
    setB = set(map(os.path.normcase, bothLibSet))

    sanityCheck0 = setA - setB
    sanityCheck1 = setB - setA

    assert len(sanityCheck0) == 0, "Unexpected code object files: {}".format(sanityCheck0)
    if not genSourcesAndExit:
        assert len(sanityCheck1) == 0, "Missing expected code object files: {}".format(sanityCheck1)
