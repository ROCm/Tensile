################################################################################
# Copyright 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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
from copy import deepcopy

import yaml

from Tensile.__init__ import __version__
import Tensile.LibraryIO as LibraryIO

version = "- {MinimumRequiredVersion: " + __version__ + "}\n"

vega20Prefix = version + r"""
- vega20
- gfx906
- [Device 66a0, Device 66a1, Device 66a7, Device 66af, Vega 20]
"""

sizes = r"""
- DummyIndexAssignment
- - - [128, 128, 1, 128]
    - [0, 80.0]
  - - [512, 512, 1, 512]
    - [1, 85.0]
"""

dvEffLogicSuffix = r"""
- null
- null
- DeviceEfficiency
"""

cuEffLogicSuffix = r"""
- null
- null
- CUEfficiency
"""

legacyLogicSuffix = r"""
- null
"""

def createLibraryLogic(suffix):
    # paths to test data
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
    problemTypePath = os.path.realpath( \
            os.path.join(dataDir, "library_data", "problemType.yaml"))
    solutionParametersPath = os.path.realpath( \
            os.path.join(dataDir, "library_data", "initialSolutionParameters.yaml"))

    # read test data
    problemType = LibraryIO.readYAML(problemTypePath)["ProblemType"]
    solutionParameters = LibraryIO.readYAML(solutionParametersPath)

    # solutions
    sol0 = deepcopy(solutionParameters)
    sol0["SolutionIndex"] = 0
    sol0["SolutionNameMin"] = "foo"
    sol0["ProblemType"] = problemType

    sol1 = deepcopy(solutionParameters)
    sol1["SolutionIndex"] = 1
    sol1["SolutionNameMin"] = "bar"
    sol0["ProblemType"] = problemType

    # other components
    prefixData = yaml.load(vega20Prefix, yaml.SafeLoader)
    sizeData = yaml.load(sizes, yaml.SafeLoader)
    suffixData = yaml.load(suffix, yaml.SafeLoader)

    # combine all components
    return prefixData + [problemType] + [[sol0, sol1]] + sizeData + suffixData

def test_parseSolutionsData(useGlobalParameters):
    with useGlobalParameters():
        # paths to test data
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        dataDir = os.path.realpath(os.path.join(scriptDir, "..", "test_data", "unit"))
        solutionsPath = os.path.realpath(os.path.join( \
                dataDir, "solutions", "solutions_nn_3.yaml"))

        solutions = LibraryIO.readYAML(solutionsPath)

        LibraryIO.parseSolutionsData(solutions, "test_parseSolutionsData")
        assert True

def test_parseLibraryLogicData_legacy(useGlobalParameters):
    with useGlobalParameters():
        LibraryIO.parseLibraryLogicData(createLibraryLogic(legacyLogicSuffix), \
                "test_parseLibraryLogicData_legacy")
        assert True

def test_parseLibraryLogicData_dvEff(useGlobalParameters):
    with useGlobalParameters():
        LibraryIO.parseLibraryLogicData(createLibraryLogic(dvEffLogicSuffix), \
                "test_parseLibraryLogicData_dvEff")
        assert True

def test_parseLibraryLogicData_cuEff(useGlobalParameters):
    with useGlobalParameters():
        LibraryIO.parseLibraryLogicData(createLibraryLogic(cuEffLogicSuffix), \
                "test_parseLibraryLogicData_cuEff")
        assert True
