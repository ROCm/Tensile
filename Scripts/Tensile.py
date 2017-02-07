################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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

from Common import *
import YAMLIO
import BenchmarkProblems
import LibraryLogic
import LibraryWriter
import LibraryClient
import BenchmarkClient

def executeStepsInConfig( config ):
  # ensure working directory exists
  ensurePath(globalParameters["WorkingPath"])


  if "Parameters" in config:
    assignGlobalParameters( config["Parameters"] )
  else:
    assignGlobalParameters({})
  print ""

  ##############################################################################
  # Benchmark Problems
  ##############################################################################
  benchmarkDataPath = os.path.join(globalParameters["WorkingPath"], \
      globalParameters["BenchmarkDataPath"])
  if "BenchmarkProblems" in config:
    if os.path.exists(benchmarkDataPath):
      resultFiles = os.listdir(benchmarkDataPath)
    else:
      resultFiles = []

    if len(resultFiles) < 2* len(config["BenchmarkProblems"]) \
            or globalParameters["ForceRedoBenchmarkProblems"]:
      BenchmarkProblems.main( config["BenchmarkProblems"] )
      print ""
    else:
      print "# Benchmarking already done."

  ##############################################################################
  # Library Logic
  ##############################################################################
  libraryLogicDataPath = os.path.join(globalParameters["WorkingPath"], \
      globalParameters["LibraryLogicPath"])
  if "LibraryLogic" in config:
    if os.path.exists(libraryLogicDataPath):
      libraryLogicFiles = os.listdir(libraryLogicDataPath)
    else:
      libraryLogicFiles = []
    if len(libraryLogicFiles) < 1 or globalParameters["ForceRedoLibraryLogic"]:
      LibraryLogic.main( config["LibraryLogic"] )
      print ""
    else:
      print "# LibraryLogic already done."
    print ""

  ##############################################################################
  # Write Library
  ##############################################################################
  #if "WriteLibrary" in config:
  #  LibraryWriter.main( config["WriteLibrary"] )
  #  print ""

  ##############################################################################
  # Write Client
  ##############################################################################
  if "LibraryClient" in config:
    LibraryClient.main( config["LibraryClient"] )
    print ""

  #if "BenchmarkClient" in config:
  #  BenchmarkClient.main( config["BenchmarkClient"] )
  #  print ""




################################################################################
# Tensile - Main
################################################################################
if len(sys.argv) < 2:
  print("Usage: python Tensile.py config.yaml output_path")
  sys.exit(1)
else:
  if len(sys.argv) == 3:
    globalParameters["WorkingPath"] = os.path.abspath(sys.argv[2])
  else:
    globalParameters["WorkingPath"] = os.getcwd()

  configPath = os.path.realpath( sys.argv[1] )
  print("Tensile::Main ConfigFile: %s" % (configPath) )
  config = YAMLIO.readConfig( configPath )
  executeStepsInConfig( config )
  sys.exit(0)

