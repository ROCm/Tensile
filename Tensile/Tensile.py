################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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

if __name__ == "__main__":
    print("This file can no longer be run as a script.  Run 'Tensile/bin/Tensile' instead.")
    exit(1)

import os
import sys
import argparse
from .Common import globalParameters, print1, ensurePath, \
    assignGlobalParameters, restoreDefaultGlobalParameters, HR
from . import BenchmarkProblems
from . import ClientWriter
from . import LibraryLogic
from . import LibraryIO
from . import __version__

###############################################################################
# Execute Steps in Config
# called from Tensile() below
# calls
#   BenchmarkProblems.main() to run benchmark steps
#   LibraryLogic.main() to analyse final benchmark data and produce logic/yaml
#   ClientWriter.main() to create client which calls library based on above yaml
################################################################################
def executeStepsInConfig( config ):

  ##############################################################################
  # Benchmark Problems
  ##############################################################################
  if "BenchmarkProblems" in config:
    BenchmarkProblems.main( config["BenchmarkProblems"] )
    print1("")


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
      if config["LibraryLogic"] != None:
        libraryLogicConfig = config["LibraryLogic"]
      else:
        libraryLogicConfig = {}
      LibraryLogic.main( libraryLogicConfig )
      print1("")
    else:
      print1("# LibraryLogic already done.")
    print1("")


  ##############################################################################
  # Write Client
  ##############################################################################
  if "LibraryClient" in config:
    if config["LibraryClient"] != None:
      libraryClientConfig = config["LibraryClient"]
    else:
      libraryClientConfig = {}
    ClientWriter.main( libraryClientConfig )
    print1("")

def addCommonArguments(argParser):
  """
  Add a common set of arguments to `argParser`.

  Currently used by the main Tensile script and the unit tests but could also be used for TensileCreateLibrary.
  """
  def splitExtraParameters(par):
    """
    Allows the --global-parameters option to specify any parameters from the command line.
    """

    (key, value) = par.split("=")
    value = eval(value)
    return (key, value)

  argParser.add_argument("-d", "--device", dest="device", type=int, \
      help="override which device to benchmark")
  argParser.add_argument("-p", "--platform", dest="platform", type=int, \
      help="override which OpenCL platform to benchmark")
  argParser.add_argument("--runtime-language", dest="RuntimeLanguage", \
      choices=["HIP", "OCL"], help="override which runtime language to use")
  argParser.add_argument("--code-object-version", dest="CodeObjectVersion", \
      choices=["V2", "V3"], default="V3", help="HSA code-object version")
  argParser.add_argument("-v", "--verbose", action="store_true", \
      help="set PrintLevel=2")
  argParser.add_argument("--debug", dest="debug", action="store_true", \
      help="set PrintLevel=2 and CMakeBuildType=Debug")
  argParser.add_argument("--short-names", dest="shortNames", action="store_true", \
      help="use serial kernel and solution names")
  argParser.add_argument("--no-merge-files", dest="noMergeFiles", action="store_true", \
      help="kernels and solutions written to individual files")
  argParser.add_argument("--cxx-compiler", dest="CxxCompiler", choices=["hcc", "hipcc"], \
      action="store", default="hipcc", help="select which compiler to use")
  argParser.add_argument("--library-format", dest="LibraryFormat", choices=["yaml", "msgpack"], \
      action="store", default="yaml", help="select which library format to use")
  argParser.add_argument("--client-build-path", default=None)
  argParser.add_argument("--client-lock", default=None)

  argParser.add_argument("--global-parameters", nargs="+", type=splitExtraParameters, default=[])

def argUpdatedGlobalParameters(args):
  """
  Returns a dictionary with `globalParameters` keys that should be updated based on `args`.
  """
  rv = {}
  # override config with command-line options
  if args.device:
    print1("# Command-line override: Device")
    rv["Device"] = args.device
  if args.platform:
    print1("# Command-line override: Platform")
    rv["Platform"] = args.platform
  if args.RuntimeLanguage:
    print1("# Command-line override: RuntimeLanguage")
    rv["RuntimeLanguage"] = args.RuntimeLanguage
  if args.CodeObjectVersion:
    print1("# Command-line override: CodeObjectVersion")
    rv["CodeObjectVersion"] = args.CodeObjectVersion
  if args.verbose:
    print1("# Command-line override: PrintLevel")
    rv["PrintLevel"] = 2
  if args.debug:
    print1("# Command-line override: Debug")
    rv["PrintLevel"] = 2
    rv["CMakeBuildType"] = "Debug"
  if args.shortNames:
    rv["ShortNames"] = True
  if args.noMergeFiles:
    rv["MergeFiles"] = False
  if args.CxxCompiler:
    rv['CxxCompiler'] = args.CxxCompiler
    if rv['CxxCompiler'] == "hipcc" and not args.CodeObjectVersion:
      rv["CodeObjectVersion"] = "V3"
  print1("")
  if args.client_build_path:
    rv["ClientBuildPath"] = args.client_build_path
  if args.client_lock:
    rv["ClientExecutionLockPath"] = args.client_lock

  for key, value in args.global_parameters:
    rv[key] = value

  return rv

################################################################################
# Tensile
# - below entry points call here
################################################################################
def Tensile(userArgs):
  global globalParameters

  # 1st half of splash
  print1("")
  print1(HR)
  print1("#")
  print1("#  Tensile v%s" % (__version__) )

  # setup argument parser
  argParser = argparse.ArgumentParser()
  argParser.add_argument("config_file", type=os.path.realpath, help="benchmark config.yaml file")
  argParser.add_argument("output_path", \
      help="path where to conduct benchmark")
  argParser.add_argument("--version", action="version", \
      version="%(prog)s {version}".format(version=__version__))
  # argParser.add_argument("--hcc-version", dest="HccVersion", \
  #     help="This can affect what opcodes are emitted by the assembler")
  addCommonArguments(argParser)

  # parse arguments
  args = argParser.parse_args(userArgs)

  configPath = args.config_file

  # 2nd half of splash
  print1("#  Config: %s" % (configPath) )
  print1("#")
  print1(HR)
  print1("")

  print1("# Restoring default globalParameters")
  restoreDefaultGlobalParameters()

  # CxxCompiler and LibraryFormat needs to be updated before assignGlobalParameters.
  if args.CxxCompiler:
    globalParameters['CxxCompiler'] = args.CxxCompiler
  if args.LibraryFormat:
      globalParameters['LibraryFormat'] = args.LibraryFormat

  # read config
  config = LibraryIO.readConfig( configPath )
  globalParameters["ConfigPath"] = configPath

  # assign global parameters
  if "GlobalParameters" in config:
    assignGlobalParameters( config["GlobalParameters"] )
  else:
    assignGlobalParameters({})

  globalParameters["OutputPath"] = ensurePath(os.path.abspath(args.output_path))
  globalParameters["WorkingPath"] = globalParameters["OutputPath"]

  overrideParameters = argUpdatedGlobalParameters(args)

  for key, value in overrideParameters.items():
    print("Overriding {0}={1}".format(key, value))
    globalParameters[key] = value

  #globalParameters["NewClient"] = 2
  #globalParameters["PrintCodeCommands"] = True

  # Execute Steps in the config script
  executeStepsInConfig(config)


def TensileConfigPath(*args):
  return os.path.join(os.path.dirname(os.path.realpath(__file__)), "Configs", *args)

def TensileTestPath(*args):
  return os.path.join(os.path.dirname(os.path.realpath(__file__)), "Tests", *args)


################################################################################
# Entry points
# the first several of these can be deprecated, only main() is used
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

