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
import argparse

from Common import *
import YAMLIO
import BenchmarkProblems
import LibraryLogic
import ClientWriter

def executeStepsInConfig( config ):
  # ensure working directory exists
  ensurePath(globalParameters["WorkingPath"])


  if "Parameters" in config:
    assignGlobalParameters( config["Parameters"] )
  else:
    assignGlobalParameters({})
  print1("")

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
      print1("")
    else:
      print1("# Benchmarking already done.")

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
      print1("")
    else:
      print1("# LibraryLogic already done.")
    print1("")


  ##############################################################################
  # Write Client
  ##############################################################################
  if "LibraryClient" in config:
    ClientWriter.main( config["LibraryClient"] )
    print1("")


################################################################################
# Tensile - below entry points call here
################################################################################
def Tensile(userArgs):
  argParser = argparse.ArgumentParser()
  argParser.add_argument("ConfigFilePath", \
      help="Path to top-level config.yaml file")
  argParser.add_argument("OutputPath", \
      help="Where to build and run benchmark steps.")
  args = argParser.parse_args(userArgs)
  globalParameters["WorkingPath"] = os.path.abspath(args.OutputPath)

  configPath = os.path.realpath( args.ConfigFilePath )
  print("Tensile::Main ConfigFile: %s" % (configPath) )
  config = YAMLIO.readConfig( configPath )
  executeStepsInConfig( config )
  sys.exit(0)


################################################################################
# Entry points
################################################################################


# installed "tensile_rocblas_sgemm" command
def TensileROCBLASSGEMM():
  configPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
      "..", "Configs", "rocblas_sgemm.yaml")
  Tensile([configPath, "."])


# installed "tensile_rocblas_dgemm" command
def TensileROCBLASDGEMM():
  configPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
      "..", "Configs", "rocblas_dgemm.yaml")
  Tensile([configPath, "."])


# installed "tensile_rocblas_cgemm" command
def TensileROCBLASCGEMM():
  configPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
      "..", "Configs", "rocblas_cgemm.yaml")
  Tensile([configPath, "."])


# installed "tensile_rocblas_zgemm" command
def TensileROCBLASZGEMM():
  configPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
      "..", "Configs", "rocblas_zgemm.yaml")
  Tensile([configPath, "."])


# installed "tensile_sgemm" command
def TensileSGEMM5760():
  configPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
      "..", "Configs", "sgemm_5760.yaml")
  Tensile([configPath, "."])


# installed "tensile" command
def main():
    Tensile(sys.argv[1:])


# script run from commandline
if __name__ == "__main__":
  main()
