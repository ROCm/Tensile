import Structs
import KernelWriter

################################################################################
# SolutionCorrectnessParameters
################################################################################
#class SolutionCorrectnessParameters:
#  def __init__( \
#      self, \
#      tensA, \
#      tensB, \
#      tensC, \
#      operation ):
#    pass
# This different from ProblemDescriptor, PD cares about M,N but SolutionCorrectness does not about the exact values of M,N
# SC cares about numDim of A, B, C.


################################################################################
# SolutionPerformanceParameters
################################################################################
#class SolutionPerformanceParameters:
#  def __init__( \
#      self, \
#      alphaIsIdentity, \
#      betaIsIdentity, \
#      orderOfDimensions, \
#      tileSizes, \
#      loadGlobalToLocalEnum, \
#      branchLocationEnum, \
#      multipleKernels ): # for K%4096
#    pass


# A Solution (SolutionCorrectnessParameters + SolutionPerformanceParameters) contain all info necessary to write the kernel

################################################################################
# SolutionImplementation
################################################################################
#class SolutionImplementation:

#  def writeOpenCLSolutions(self):
#    pass

################################################################################
# SolutionWriter
################################################################################
class SolutionWriter:

  ##############################################################################
  # SolutionWriter
  ##############################################################################
  def __init__(self, backend):
    self.backend = backend
    self.kernelWriter = KernelWriter.KernelWriter(self.backend)

  ##############################################################################
  # getName
  ##############################################################################
  def getName(self, solution):
    solutionName = ""
    for kernel in solution.kernels:
      solutionName += self.kernelWriter.getName(kernel) + "_"
    return solutionName


  ##############################################################################
  # getSourceString
  ##############################################################################
  def getSourceString(self, solution):
    return "solution source string"


  ##############################################################################
  # getHeaderString
  ##############################################################################
  def getHeaderString(self, solution):
    return "solution source string"

