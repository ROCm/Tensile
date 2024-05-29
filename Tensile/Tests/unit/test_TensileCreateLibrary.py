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
    Common.globalParameters["CxxCompiler"] = "hipcc"
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
  
  manifestFile = "test_manifest.txt"
  # ensure clean state before running test
  with contextlib.suppress(FileNotFoundError):
    os.remove(manifestFile)
    os.remove("foo.asm")    
    os.remove("bar.asm")        
  # create an empty manifest 
  with open(manifestFile, "x") as manifest:
   
    assert TensileCreateLibrary.verifyManifest(manifestFile), "an empty manifest should always succeed"
        
    # add to file manifest that is not on disk
    manifest.write("foo.asm\n")
    manifest.flush()
    assert not TensileCreateLibrary.verifyManifest(manifestFile), "file in manifest should not be on disk"
  
    with open("foo.asm", "x"):
      assert TensileCreateLibrary.verifyManifest(manifestFile), "file in manifest should be on disk"
  
    manifest.write("bar.asm\n")
    manifest.flush()
    assert not TensileCreateLibrary.verifyManifest(manifestFile), "bar.asm in manifest should not be on disk"
  
    with open("bar.asm", "x"):
      assert TensileCreateLibrary.verifyManifest(manifestFile), "files in manifest should be on disk"

def test_parseArchitectures():
    # Test `all` architectures
    archs = 'all'
    result = TensileCreateLibrary.parseArchitectures(archs)
    expected = set([archs])
    assert result == expected, f"arch `{archs}` should parse to {expected} but instead maps to {result}"

    expected = set([item for item in filter(lambda x: x != "all", Common.architectureMap.keys())])
    # Test all architectures, but supplied separately with semicolon delimieter (cli usage)
    archs = "gfx000;gfx803;gfx900;gfx900:xnack-;gfx906;gfx906:xnack+;gfx906:xnack-;gfx908;gfx908:xnack+;"+\
            "gfx908:xnack-;gfx90a;gfx90a:xnack+;gfx90a:xnack-;gfx940;gfx940:xnack+;gfx940:xnack-;gfx941;"+\
            "gfx941:xnack+;gfx941:xnack-;gfx942;gfx942:xnack+;gfx942:xnack-;gfx1010;gfx1011;gfx1012;"+\
            "gfx1030;gfx1031;gfx1032;gfx1034;gfx1035;gfx1100;gfx1101;gfx1102"
    result = TensileCreateLibrary.parseArchitectures(archs)
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    # Test all architectures, but supplied separately with underscore delimieter (cmake usage)
    archs = "gfx000_gfx803_gfx900_gfx900:xnack-_gfx906_gfx906:xnack+_gfx906:xnack-_gfx908_gfx908:xnack+_"+\
             "gfx908:xnack-_gfx90a_gfx90a:xnack+_gfx90a:xnack-_gfx940_gfx940:xnack+_gfx940:xnack-_gfx941_"+\
             "gfx941:xnack+_gfx941:xnack-_gfx942_gfx942:xnack+_gfx942:xnack-_gfx1010_gfx1011_gfx1012_"+\
             "gfx1030_gfx1031_gfx1032_gfx1034_gfx1035_gfx1100_gfx1101_gfx1102"
    result = TensileCreateLibrary.parseArchitectures(archs)
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    # Test with wrong input type
    archs = ["gfx942:xnack-", "gfx1010", "gfx1011"]
    with pytest.raises(AttributeError):
        TensileCreateLibrary.parseArchitectures(archs)

    # Test with comma delimiter instead of semicolon delimiter
    archs = "gfx803,gfx906,gfx941,gfx1102"
    with pytest.raises(ValueError, match=r"(.*) is not a known Gfx architecture."):
        TensileCreateLibrary.parseArchitectures(archs)

    # Test with space delimiter instead of semicolon delimiter
    archs = "gfx900 gfx90a:xnack+ gfx1010"
    with pytest.raises(ValueError, match=r"(.*) is not a known Gfx architecture."):
        TensileCreateLibrary.parseArchitectures(archs)

    # Test with colon instead of semicolon delimiter
    archs = "gfx900:gfx90a:xnack+:gfx1010"
    with pytest.raises(ValueError, match=r"(.*) is not a known Gfx architecture."):
        TensileCreateLibrary.parseArchitectures(archs)

    # Test with hyphen instead of semicolon delimiter
    archs = "gfx900-gfx90a:xnack+-gfx1010"
    with pytest.raises(ValueError, match=r"(.*) is not a known Gfx architecture."):
        TensileCreateLibrary.parseArchitectures(archs)

    # Test with leading underscore delimiter
    archs = ";gfx803;gfx906;"
    expected = set(["gfx803", "gfx906"])
    result = TensileCreateLibrary.parseArchitectures(archs)
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    # Test with leading underscore delimiter
    archs = "_gfx803_gfx906_"
    expected = set(["gfx803", "gfx906"])
    result = TensileCreateLibrary.parseArchitectures(archs)
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    # Test an unsupported architecture - an unsupported architecture isn't
    archs = "gfx90Z"
    with pytest.raises(ValueError, match="`gfx90z` is not a known Gfx architecture."):
        result = TensileCreateLibrary.parseArchitectures(archs)

    # Test an unsupported architecture and a supported one
    archs = "gfx90a;gfx90Z;gfx90a"
    with pytest.raises(ValueError, match="`gfx90z` is not a known Gfx architecture."):
        result = TensileCreateLibrary.parseArchitectures(archs)

    # Test an unsupported architecture and `all` - If `all` comes second it is rejected
    archs = "gfx90Z;all"
    with pytest.raises(ValueError):
        result = TensileCreateLibrary.parseArchitectures(archs)

    # Test an unsupported architecture and `all` - if `all` comes first it is accepted
    archs = "all;gfx90Z"
    expected = set(["all"])
    result = TensileCreateLibrary.parseArchitectures(archs)
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

def test_mapGfxArchitectures():
    # Test `all` architectures
    archs = set(["all"])
    expected = set([Common.architectureMap["all"]])
    result = TensileCreateLibrary.mapGfxArchitectures(archs)
    assert result == expected, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    gfx_archs = set([x[0] for x in filter(lambda x: x[0] != "all", Common.architectureMap.items())])
    named_archs = set([x[1] for x in filter(lambda x: x[0] != "all", Common.architectureMap.items())])

    # Test all architectures, but supplied separately
    result = TensileCreateLibrary.mapGfxArchitectures(gfx_archs)
    assert result == named_archs, f"arch `{archs}` should map to {expected} but instead maps to {result}"

    # Test named architectures as inputs instead of gfx architectures
    with pytest.raises(ValueError, match=r"(.*) is not a supported architecture"):
        TensileCreateLibrary.mapGfxArchitectures(named_archs)
