################################################################################
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

from .Common import printExit, printWarning, versionIsCompatible
from .SolutionStructs import Solution, ProblemSizes, ProblemType
from . import __version__
from . import Common
from . import SolutionLibrary

try:
    import yaml
except ImportError:
    printExit("You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions.")

try:
    import msgpack
except ImportError:
    print("Message pack python library not detected. Must use YAML backend instead.")


###################
# Writing functions
###################
def write(filename_noExt, data, format="yaml"):
    """Writes data to file with specified format; extension is appended based on format."""
    if format == "yaml":
        writeYAML(filename_noExt + ".yaml", data)
    elif format == "msgpack":
        writeMsgPack(filename_noExt + ".dat", data)
    else:
        printExit("Unrecognized format {}".format(format))

def writeYAML(filename, data, **kwargs):
    """Writes data to file in YAML format."""
    # set default kwags for yaml dump
    if "explicit_start" not in kwargs:
        kwargs["explicit_start"] = True
    if "explicit_end" not in kwargs:
        kwargs["explicit_end"] = True
    if "default_flow_style" not in kwargs:
        kwargs["default_flow_style"] = None

    with open(filename, "w") as f:
        yaml.dump(data, f, **kwargs)

def writeMsgPack(filename, data):
    """Writes data to file in Message Pack format."""
    with open(filename, "wb") as f:
        msgpack.pack(data, f)

def writeSolutions(filename, problemSizes, solutions):
    """Writes solution YAML file."""

    # convert objects to nested dictionaries
    solutionStates = []
    for hardcoded in solutions:
        for solution in hardcoded:
            solutionState = solution.getAttributes()
            solutionState["ProblemType"] = solutionState["ProblemType"].state
            solutionState["ProblemType"]["DataType"] = \
                    solutionState["ProblemType"]["DataType"].value
            solutionState["ProblemType"]["DestDataType"] = \
                    solutionState["ProblemType"]["DestDataType"].value
            solutionState["ProblemType"]["ComputeDataType"] = \
                    solutionState["ProblemType"]["ComputeDataType"].value
            solutionStates.append(solutionState)
    # write dictionaries
    with open(filename, "w") as f:
        f.write("- MinimumRequiredVersion: %s\n" % __version__ )
        f.write("- ProblemSizes:\n")
        if problemSizes:
            for sizeRange in problemSizes.ranges:
                f.write("  - Range: %s\n" % sizeRange)
            for problemExact in problemSizes.exacts:
                #FIXME-problem, this ignores strides:
                f.write("  - Exact: %s\n" % str(problemExact))

        yaml.dump(solutionStates, f, default_flow_style=None)


###############################
# Reading and parsing functions
###############################
def readYAML(filename):
    """Reads and returns YAML data from file."""
    with open(filename, "r") as f:
        data = yaml.load(f, yaml.SafeLoader)
    return data

def parseSolutionsFile(filename):
    """Wrapper function to read and parse a solutions file."""
    return parseSolutionsData(readYAML(filename), filename)

def parseSolutionsData(data, srcFile="?"):
    """Parses problem sizes and solutions from the data of a solutions file."""

    if len(data) < 3:
        printExit("Solution file {} is missing required fields (len = {} < 3".format(srcFile, len(data)))

    versionString = data[0]["MinimumRequiredVersion"]
    if not versionIsCompatible(versionString):
        printWarning("Version = {} in solution file {} does not match Tensile version = {}" \
                .format(srcFile, versionString, __version__) )

    if "ProblemSizes" not in data[1]:
        printExit("Solution file {} doesn't begin with ProblemSizes".format(srcFile))

    problemSizesConfig = data[1]["ProblemSizes"]

    solutions = []
    for i in range(2, len(data)):
        solutionState = data[i]
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        solutionObject = Solution(solutionState)
        solutions.append(solutionObject)
    problemType = solutions[0]["ProblemType"]
    problemSizes = ProblemSizes(problemType, problemSizesConfig)
    return (problemSizes, solutions)

def parseLibraryLogicFile(filename):
    """Wrapper function to read and parse a library logic file."""
    return parseLibraryLogicData(readYAML(filename), filename)

def parseLibraryLogicData(data, srcFile="?"):
    """Parses the data of a library logic file."""

    if len(data) < 9:
        printExit("Library logic file {} is missing required fields (len = {} < 9)".format(srcFile, len(data)))

    versionString     = data[0]["MinimumRequiredVersion"]
    scheduleName      = data[1]
    architectureName  = data[2] if isinstance(data[2], str) else data[2]["Architecture"]
    deviceNames       = data[3]
    problemTypeState  = data[4]
    solutionStates    = data[5]
    indexOrder        = data[6]
    exactLogic        = data[7]
    rangeLogic        = data[8]

    if not versionIsCompatible(versionString):
        printWarning("Version = {} in library logic file {} does not match Tensile version = {}" \
                .format(srcFile, versionString, __version__) )

    # unpack problemType
    problemType = ProblemType(problemTypeState)
    # unpack solutions
    solutions = []
    for i in range(0, len(solutionStates)):
        solutionState = solutionStates[i]
        if solutionState["KernelLanguage"] == "Assembly":
            solutionState["ISA"] = Common.gfxArch(architectureName)
        else:
            solutionState["ISA"] = (0, 0, 0)
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        solutionObject = Solution(solutionState)

        if solutionObject["ProblemType"] != problemType:
            printExit("ProblemType in library logic file {} doesn't match solution: {} != {}" \
                    .format(srcFile, problemType, solutionObject["ProblemType"]))
        solutions.append(solutionObject)

    newLibrary = SolutionLibrary.MasterSolutionLibrary.FromOriginalState(data, solutions)

    return (scheduleName, deviceNames, problemType, solutions, indexOrder, \
            exactLogic, rangeLogic, newLibrary, architectureName)

def rawLibraryLogic(data):
    """Returns a tuple of the data in a library logic file."""
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
            otherFields.append(data[idx])

    return (versionString, scheduleName, architectureName, deviceNames,\
            problemTypeState, solutionStates, indexOrder, exactLogic, rangeLogic, otherFields)


#################
# Other functions
#################
def createLibraryLogic(schedulePrefix, architectureName, deviceNames, logicTuple):
    """Creates the data for a library logic file suitable for writing to YAML."""
    problemType   = logicTuple[0]
    solutions     = logicTuple[1]
    indexOrder    = logicTuple[2]
    exactLogic    = logicTuple[3]
    rangeLogic    = logicTuple[4]

    tileSelection = False
    if len(logicTuple) > 5 and logicTuple[5]:
        tileSelection = True

    data = []
    # Tensile version
    data.append({"MinimumRequiredVersion":__version__})
    # schedule name
    data.append(schedulePrefix)     # change from Tensile to vega10
    data.append(architectureName)
    # schedule device names
    data.append(deviceNames)
    # problem type
    problemTypeState = problemType.state
    problemTypeState["DataType"] = \
            problemTypeState["DataType"].value
    problemTypeState["DestDataType"] = \
            problemTypeState["DestDataType"].value
    problemTypeState["ComputeDataType"] = \
            problemTypeState["ComputeDataType"].value
    data.append(problemTypeState)
    # solutions
    solutionList = []
    for solution in solutions:
        solutionState = solution.getAttributes()
        solutionState["ProblemType"] = solutionState["ProblemType"].state
        solutionState["ProblemType"]["DataType"] = \
                solutionState["ProblemType"]["DataType"].value
        solutionState["ProblemType"]["DestDataType"] = \
                solutionState["ProblemType"]["DestDataType"].value
        solutionState["ProblemType"]["ComputeDataType"] = \
                solutionState["ProblemType"]["ComputeDataType"].value
        solutionList.append(solutionState)

    if tileSelection:
        tileSolutions = logicTuple[5]
        for solution in tileSolutions:
            solutionState = solution.getAttributes()
            solutionState["ProblemType"] = solutionState["ProblemType"].state
            solutionState["ProblemType"]["DataType"] = \
                    solutionState["ProblemType"]["DataType"].value
            solutionState["ProblemType"]["DestDataType"] = \
                    solutionState["ProblemType"]["DestDataType"].value
            solutionState["ProblemType"]["ComputeDataType"] = \
                    solutionState["ProblemType"]["ComputeDataType"].value
            solutionList.append(solutionState)

    data.append(solutionList)
    # index order
    data.append(indexOrder)

    # exactLogic
    exactLogicList = []
    for key in exactLogic:
        exactLogicList.append([list(key), exactLogic[key]])
    data.append(exactLogicList)

    # rangeLogic
    data.append(rangeLogic)

    if tileSelection:
        tileSelectionLogic = {}
        tileSelectionIndices = logicTuple[6]
        tileSelectionLogic["TileSelectionIndices"] = tileSelectionIndices
        data.append(tileSelectionLogic)
    else:
        data.append(None)

    data.append(logicTuple[7])
    return data
