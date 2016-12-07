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

  def __init__(self, config):
    printStatus("Beginning")

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

    # get benchmark steps from config
    configBenchmarkCommonParameters = config["BenchmarkCommonParameters"] if "BenchmarkCommonParameters" in config else []
    configForkParameters = config["ForkParameters"] if "ForkParameters" in config else []
    configBenchmarkForkParameters = config["BenchmarkForkParameters"] if "BenchmarkForkParameters" in config else []
    configJoinParameters = config["JoinParameters"] if "JoinParameters" in config else []
    configBenchmarkJoinParameters = config["BenchmarkJoinParameters"] if "BenchmarkJoinParameters" in config else []

# TODO - how to insert and override problem sizes?

    # (1) into common we put in all Dcommon that
    # don't show up in Ccommon/Cfork/CBfork/Cjoin/CBjoin
    # followed by Ccommon
    benchmarkCommonParameters = []
    for param in defaultBenchmarkCommonParameters:
      paramName = param[0]
      if not keyInListOfListOfDictionaries(paramName, [configBenchmarkCommonParameters, configForkParameters, configBenchmarkForkParameters, configJoinParameters, configBenchmarkJoinParameters]):
        benchmarkCommonParameters.append(param)
    for param in configBenchmarkCommonParameters:
      benchmarkCommonParameters.append(param)

    # (2) into fork we put in all Dfork that
    # don't show up in Bcommon/Cfork/CBfork/Cjoin/CBjoin
    # followed by Cfork
    forkParameters = []
    for param in defaultForkParameters:
      paramName = param[0]
      if not keyInListOfListOfDictionaries(paramName, [benchmarkCommonParameters, configForkParameters, configBenchmarkForkParameters, configJoinParameters, configBenchmarkJoinParameters]):
        forkParameters.append(param)
    for param in configForkParameters:
      forkParameters.append(param)

    # (3) into Bfork we put in all DBfork that
    # don't show up in Bcommon/Bfork/CBfork/Cjoin/CBjoin
    # followed by CBforked
    benchmarkForkParameters = []
    for param in defaultBenchmarkForkParameters:
      paramName = param[0]
      if not keyInListOfListOfDictionaries(paramName, [benchmarkCommonParameters, forkParameters, configBenchmarkForkParameters, configJoinParameters, configBenchmarkJoinParameters]):
        benchmarkForkParameters.append(param)
    for param in configBenchmarkForkParameters:
      benchmarkForkParameters.append(param)

    # (4) into join we put in all Djoin that
    # don't show up in Bcommon/Bfork/CBfork/Cjoin/CBjoin
    # followed by CBforked
    joinParameters = []
    for param in defaultJoinParameters:
      paramName = param[0]
      if not keyInListOfListOfDictionaries(paramName, [benchmarkCommonParameters, forkParameters, benchmarkForkParameters, configJoinParameters, configBenchmarkJoinParameters]):
        joinParameters.append(param)
    for param in configJoinParameters:
      joinParameters.append(param)

    # (5) into Bjoin we put in all DBjoin that
    # don't show up in Bcommon/Bfork/BBfork/Bjoin/CBjoin
    # followed by CBjoin
    benchmarkJoinParameters = []
    for param in defaultBenchmarkJoinParameters:
      paramName = param[0]
      if not keyInListOfListOfDictionaries(paramName, [benchmarkCommonParameters, forkParameters, benchmarkForkParameters, joinParameters, configBenchmarkJoinParameters]):
        benchmarkJoinParameters.append(param)
    for param in configBenchmarkJoinParameters:
      benchmarkJoinParameters.append(param)


    # benchmarkCommonParameters
    printExtra("benchmarkCommonParameters")
    for step in benchmarkCommonParameters:
      print step
    # forkParameters
    printExtra("forkParameters")
    for param in forkParameters:
      print param
    # benchmarkForkParameters
    printExtra("benchmarkForkParameters")
    for step in benchmarkForkParameters:
      print step
    # joinParameters
    printExtra("joinParameters")
    for param in joinParameters:
      print param
    # benchmarkJoinParameters
    printExtra("benchmarkJoinParameters")
    for step in benchmarkJoinParameters:
      print step

    # Final Benchmark
    # if "BenchmarkFinal" in config:
    printStatus("DONE.")








################################################################################
def benchmarkProblemType( config ):

  """
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
  """


  # object for default benchmarking
  benchmarkProcess = BenchmarkProcess(config)
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
