import sys
from copy import *

from BenchmarkProcess import *
from Common import *
from Structs import *


################################################################################
def benchmarkProblemType( config ):



  # convert cnofig to full benchmark process (resolves defaults)
  benchmarkProcess = BenchmarkProcess(config)
  totalBenchmarkSteps = len(benchmarkProcess)
  for benchmarkStepIdx in range(0, totalBenchmarkSteps):
    benchmarkStep = benchmarkProcess[benchmarkStepIdx]
    print benchmarkStepIdx
  # for step in benchmarkProcess:


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
