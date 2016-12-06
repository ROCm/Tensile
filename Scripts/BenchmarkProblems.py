import Common
import Structs

from Common import printDebug

def benchmarkProblemType( config ):

  # read problem type
  if "ProblemType" in config:
    problemTypeConfig = config["ProblemType"]
    problemType = Structs.ProblemType(problemTypeConfig)
  else:
    sys.exit("Tensile::%s::%s: ERROR - No ProblemType in config: %s" % ( __file__, __line__, str(config) ))
  printDebug(1,"Tensile::BenchmarkProblemType: %s" % str(problemType) )

  # read initial solution parameters
  solutionConfig = { "ProblemType": problemTypeConfig }
  if "InitialSolutionParameters" not in config:
    printDebug(1,"Tensile::BenchmarkProblemType: WARNING - No InitialSolutionParameters; using defaults.")
  else:
    solutionConfig.update(config["InitialSolutionParameters"])
  initialSolutionParameters = Structs.Solution(solutionConfig)
  printDebug(1,"Tensile::BenchmarkProblemType: InitialSolutionParameters: %s" % str(initialSolutionParameters))

  # for each step in BenchmarkParameters
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
  printDebug(1,"Tensile::BenchmarkProblems::main")
  for problemType in config:
    benchmarkProblemType(problemType)
    printDebug(1,"")

  pass
