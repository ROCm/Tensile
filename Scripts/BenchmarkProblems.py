import Common
import Structs

import sys

from Common import *

################################################################################
# Benchmark Step
# - check if this step needs to be performed
# - which problem sizes will be benchmarked
# - which solutions will be benchmarked
#   - know prior step to read results from
#   - simple algorithm to choose winner, and record it
# - Write
#   - solution files
#   - problem iteration file
#   - CMakeLists.txt
# - Copy static files to build directory
# - compile and run executable
# - read in csv data and write winner.yaml
################################################################################
class BenchmarkStep:

  def __init__(self, config):
    pass
    # what parameters were determined already
    # what parameters do I need to read from previous step
    # what parameters will I benchmark
    # what winners will I parse from my data


################################################################################
# Benchmark Process
# steps in config need to be expanded and
# missing elements need to be assigned a default
################################################################################
class BenchmarkProcess:

  def __init__(self, config, initialSolutionParameters):

    """
- into common we put in all default-common parameters that don't show up in config's common/forked/joined followed by config's common
- into forked we put in all default-forked parameters that don't show up above or in config's forked/joined followed by config's forked parameters
- into joined we put in all default-joined parameters that don't show up above or in config's joined followed by config's joined parameters
    """
    # benchmark common steps
    # benchmark fork steps
    # benchmark join steps
    # benchmark final
    # if "BenchmarkFinal" not in
    pass



################################################################################
def benchmarkProblemType( config ):

  # read problem type
  if "ProblemType" in config:
    problemTypeConfig = config["ProblemType"]
  else:
    problemTypeConfig = {}
    printWarning("No ProblemType in config: %s; using defaults." % str(config) )
  problemType = Structs.ProblemType(problemTypeConfig)
  printStatus("Beginning %s" % str(problemType))

  # read initial solution parameters
  solutionConfig = { "ProblemType": problemTypeConfig }
  if "InitialSolutionParameters" not in config:
    printWarning("No InitialSolutionParameters; using defaults.")
  else:
    solutionConfig.update(config["InitialSolutionParameters"])
  initialSolutionParameters = Structs.Solution(solutionConfig)
  printExtra("InitialSolutionParameters: %s" % str(initialSolutionParameters))

  if "BenchmarkCommonSolutionParameters" not in config:
    pass
  # object for default benchmarking
  # default order of parameters
  # default of which values to benchmark
  # default fork
  # default join
  # parameters not mentioned anywhere are moved to before first fork


    # if Fork
      # expand winners
    # if Join
      # contract winners
    # if Problems
      # update problems
    # if Parameter
    # check if parameter already benchmarked and redo=False
    # create solution candidates
      # InitialSolutionParameters
      # DeterminedSolutionParameters
        # analyse prior step and clearly write out winners or prior step here
      # this step's list of parameters to benchmark
    # create benchmark executable
      # kernel and solution files
      # problem iteration file
      # cmake for generated files
      # copy static files
    # compile benchmark executable
    # run benchmark executable
  # Final BenchmarkSolutions

# benchmarking directory structure
# build/
#   BenchmarkProblemTypes
#     Cij_Aik_Bjk_SBOI
#       Step1
#       Step2
#       Final
#       Data
#         step1.csv
#         step1.yaml
#         step2.csv
#         step2.yaml
#         final.csv
#     Cij_Aik_Bkj_SBOI
#       ...
#   Analysis
#     Cij_Aik_Bjk_SBOI.yaml
#     Cij_Aik_Bjk_SBOI.yaml
#     LibName.yaml
  printStatus("%s DONE." % str(problemType))

def main(  config ):
  printStatus("Beginning")
  for problemType in config:
    if problemType is None:
      benchmarkProblemType({})
    else:
      benchmarkProblemType(problemType)
    if globalParameters["DebugPrintLevel"] >= 1:
      print ""

  printStatus("DONE.")
