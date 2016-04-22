################################################################################
# SolutionSet
################################################################################
class SolutionSet:
  def __init__( self, problem):
    self.solutionCorrectnessParameters = SolutionCorrectnessParameters(problem)
    self.solutionPerformanceParameters = getSolutionSet(problem)


################################################################################
# ProblemPerformanceCharacterization
# - Parameters of the problem which determine which SolutionPerformance
#   options should be tried
# - Problem->ProblemPerformanceType; MAP[ProblemPerformanceType]->SolutionPerformanceOptions
################################################################################
class ProblemPerformanceCharacterization:
  def __init__(self, problem):
    GEMM squarish
    GEMM small K
    GEMM skinny
    M,N to try exactly matching the tile size
    Convolution stride



################################################################################
# Map Problem Type to Solution Options
# - input: problem
# - output: list of SolutionPerformanceParameters
# - this method is hand-coded
################################################################################
def getSolutionSet( problem ):
  sppList = []

  #GEMM squarish
  #GEMM small K
  #GEMM skinny
  #M,N to try exactly matching the tile size
  #Convolution stride

  return sppList
