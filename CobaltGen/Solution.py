################################################################################
# SolutionCorrectnessParameters
################################################################################
class SolutionCorrectnessParameters:
  def __init__( \
      self, \
      tensA, \
      tensB, \
      tensC, \
      operation ):
# This different from ProblemDescriptor, PD cares about M,N but SolutionCorrectness does not about the exact values of M,N
# SC cares about numDim of A, B, C.


################################################################################
# SolutionPerformanceParameters
################################################################################
class SolutionPerformanceParameters:
  def __init__( \
      self, \
      alphaIsIdentity, \
      betaIsIdentity, \
      orderOfDimensions, \
      tileSizes, \
      loadGlobalToLocalEnum, \
      branchLocationEnum, \
      multipleKernels ): # for K%4096


# A Solution (SolutionCorrectnessParameters + SolutionPerformanceParameters) contain all info necessary to write the kernel

################################################################################
# SolutionImplementation
################################################################################
class SolutionImplementation:

  def writeOpenCLSolutions(self):
