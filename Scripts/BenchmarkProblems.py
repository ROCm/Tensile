import sys
from copy import *

from BenchmarkProcess import *
from Common import *
from Structs import *


################################################################################
def benchmarkProblemType( config ):


  # convert cnofig to full benchmark process (resolves defaults)
  benchmarkProcess = BenchmarkProcess(config)
  problemTypeName = str(benchmarkProcess.problemType)
  pushWorkingPath(problemTypeName)
  ensurePath(os.path.join(globalParameters["WorkingPath"],"Data"))

  totalBenchmarkSteps = len(benchmarkProcess)
  for benchmarkStepIdx in range(0, totalBenchmarkSteps):
    benchmarkStep = benchmarkProcess[benchmarkStepIdx]
    print benchmarkStepIdx
    stepName = str(benchmarkStep)
    pushWorkingPath(stepName)

    popWorkingPath()

  popWorkingPath()

    #TODO - resume here creating benchmark

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
#       1_ParamName
#       2_ParamName
#       3_Fork
#       4_Join
#       5_Final
#       0_Data
#         step1.csv
#         step2.csv
#         final.csv
#     Cij_Aik_Bkj_SBOI
#       ...
#   Analysis
#     Cij_Aik_Bjk_SBOI.yaml
#     Cij_Aik_Bjk_SBOI.yaml
#     LibName.yaml

def main(  config ):
  printStatus("Beginning")
  pushWorkingPath("1_BenchmarkProblemTypes")
  for problemType in config:
    if problemType is None:
      benchmarkProblemType({})
    else:
      benchmarkProblemType(problemType)
    if globalParameters["DebugPrintLevel"] >= 1:
      print ""

  printStatus("DONE.")
  popWorkingPath()
