import Structs
import KernelWriter

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

