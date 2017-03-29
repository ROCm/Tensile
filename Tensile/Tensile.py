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

from Common import globalParameters, print1, ensurePath, assignGlobalParameters
import YAMLIO
import BenchmarkProblems
import LibraryLogic
import ClientWriter
from __init__ import __version__

def executeStepsInConfig( config ):

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

  # setup argument parser
  argParser = argparse.ArgumentParser()
  argParser.add_argument("config_file", \
      help="benchmark config.yaml file")
  argParser.add_argument("output_path", \
      help="path where to conduct benchmark")
  argParser.add_argument("--version", action="version", \
      version="%(prog)s {version}".format(version=__version__))
  argParser.add_argument("-d", "--device", dest="device", type=int, \
      help="override which device to benchmark")
  argParser.add_argument("-p", "--platform", dest="platform", type=int, \
      help="override which OpenCL platform to benchmark")
  argParser.add_argument("--runtime-language", dest="RuntimeLanguage", \
      choices=["HIP", "OCL"], help="override which runtime language to use")
  argParser.add_argument("--kernel-language", dest="KernelLanguage", \
      choices=["HIP", "OCL"], help="override which kernel language to use")
  argParser.add_argument("-v", "--verbose", action="store_true", \
      help="set PrintLevel=2 and LibraryPrintDebug=True")
  argParser.add_argument("--debug", action="store_true", \
      help="set PrintLevel=2, LibraryPrintDebug=True and CMakeBuildType=Debug")

  # parse arguments
  args = argParser.parse_args(userArgs)
  globalParameters["WorkingPath"] = os.path.abspath(args.output_path)
  configPath = os.path.realpath( args.config_file)
  print1("# ConfigFile: %s" % (configPath) )
  config = YAMLIO.readConfig( configPath )
  ensurePath(globalParameters["WorkingPath"])

  # assign global parameters
  if "GlobalParameters" in config:
    assignGlobalParameters( config["GlobalParameters"] )
  else:
    assignGlobalParameters({})

  # override config with command-line options
  if args.device:
    print1("# Command-line override: Device")
    globalParameters["Device"] = args.device
  if args.platform:
    print1("# Command-line override: Platform")
    globalParameters["Platform"] = args.platform
  if args.RuntimeLanguage:
    print1("# Command-line override: RuntimeLanguage")
    globalParameters["RuntimeLanguage"] = args.RuntimeLanguage
  if args.KernelLanguage:
    print1("# Command-line override: KernelLanguage")
    globalParameters["KernelLanguage"] = args.KernelLanguage
  if args.verbose:
    print1("# Command-line override: PrintLevel")
    globalParameters["PrintLevel"] = 2
    globalParameters["LibraryPrintDebug"] = True
  if args.debug:
    print1("# Command-line override: Debug")
    globalParameters["PrintLevel"] = 2
    globalParameters["LibraryPrintDebug"] = True
    globalParameters["CMakeBuildType"] = "Debug"

  print1("")
  executeStepsInConfig( config )
  sys.exit(0)


def TensileConfigPath(*args):
  return os.path.join(os.path.dirname(os.path.realpath(__file__)), "Configs", *args)

################################################################################
# Entry points
################################################################################


# installed "tensile_rocblas_sgemm" command
def TensileROCBLASSGEMM():
  Tensile([TensileConfigPath("rocblas_sgemm.yaml"), "."])


# installed "tensile_rocblas_dgemm" command
def TensileROCBLASDGEMM():
  Tensile([TensileConfigPath("rocblas_dgemm.yaml"), "."])


# installed "tensile_rocblas_cgemm" command
def TensileROCBLASCGEMM():
  Tensile([TensileConfigPath("rocblas_cgemm.yaml"), "."])


# installed "tensile_rocblas_zgemm" command
def TensileROCBLASZGEMM():
  Tensile([TensileConfigPath("rocblas_zgemm.yaml"), "."])


# installed "tensile_sgemm" command
def TensileSGEMM5760():
  Tensile([TensileConfigPath("sgemm_5760.yaml"), "."])


# installed "tensile" command
def main():
    Tensile(sys.argv[1:])


# script run from commandline
if __name__ == "__main__":
  main()
