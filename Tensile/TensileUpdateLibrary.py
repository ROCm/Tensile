###############################################################################
# Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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


from . import LibraryIO

from . import Common
from .Common import assignGlobalParameters, globalParameters, print1, ensurePath, \
        restoreDefaultGlobalParameters, HR
from .Tensile import addCommonArguments, argUpdatedGlobalParameters
from . import __version__

import argparse
import copy
import os
import sys


def TensileRetuneLibrary(userArgs):
    print1("")
    print1(HR)
    print1("#")
    print1("#  Tensile Update Library v{}".format(__version__))

    # argument parsing and related setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("LogicFile", type=os.path.realpath,
                           help="Library logic file to update")
    argParser.add_argument("OutputFile",
                           help="Output file for update logic file")

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)

    libPath = args.LogicFile
    print1("#  Library Logic: {}".format(libPath))
    print1("#")
    print1(HR)
    print1("")


    ##############################################
    # Retuning
    ##############################################
    restoreDefaultGlobalParameters()
    assignGlobalParameters({})
    overrideParameters = argUpdatedGlobalParameters(args)
    for key, value in overrideParameters.items():
        print1("Overriding {0}={1}".format(key, value))
        Common.globalParameters[key] = value


    #outPath = ensurePath(os.path.abspath(args.OutputPath))

    libYaml = LibraryIO.readYAML(libPath)
    # parseLibraryLogicData mutates the original data, so make a copy
    fields = LibraryIO.parseLibraryLogicData(copy.deepcopy(libYaml), libPath)
    (scheduleName, deviceNames, problemType, solutions, indexOrder, \
            exactLogic, rangeLogic, newLibrary, architectureName) = fields

    logicTuple = (problemType, solutions, indexOrder, exactLogic, rangeLogic, None, None, None)
    #print(exactLogic)
    updated = LibraryIO.createLibraryLogic(scheduleName, architectureName, deviceNames, logicTuple)

    LibraryIO.writeYAML(args.OutputFile, updated, explicit_start=False, explicit_end=False)


def main():
    TensileRetuneLibrary(sys.argv[1:])
