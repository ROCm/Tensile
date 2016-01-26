import Structs
import KernelWriter

################################################################################
# SolutionWriter
################################################################################
class SolutionWriter:

  indexChars = [ "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", \
      "T", "U", "V", "W", "X", "Y", "Z" ]

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
    solutionName = self.kernelWriter.getName( \
        solution.kernels[0])
    solutionName += "_G"
    solutionName += str(solution.kernelGrid[0])
    solutionName += solution.branch[0].getChar()
    solutionName += str(solution.kernelGrid[1])
    return solutionName


  ##############################################################################
  # getSourceString
  ##############################################################################
  def getSourceString(self, solution):
    solutionName = self.getName(solution)
    s = ""
    # includes
    s += "#include \""
    s += solutionName
    s += ".h\"\n"
    s += "\n"

    # contructor signature
    s += "/* solution constructor */\n"
    s += solutionName + "::" + solutionName
    s += "( CobaltProblem inputProblem )\n"
    s += "    : CobaltSolutionOpenCL( inputProblem ) {\n"
    s += "\n"

    # solution properties (common to all kernels)
    s += "  /* solution properties */\n"
    for i in range(0, len(solution.kernels[0].indexOrderC)):
      s += "  indexOrderC[" + str(i) + "] = " \
          + str(solution.kernels[0].indexOrderC[i]) + ";\n"
    for i in range(0, len(solution.kernels[0].indexOrderSummation)):
      s += "  indexOrderSummation[" + str(i) + "] = " \
          + str(solution.kernels[0].indexOrderSummation[i]) + ";\n"
    s += "  indexAssignmentDim0 = " \
        + str(solution.kernels[0].indexAssignmentDim0) + ";\n"
    s += "  indexAssignmentDim1 = " \
        + str(solution.kernels[0].indexAssignmentDim1) + ";\n"
    s += "\n"

    # tile properties (common to all kernels)
    s += "  /* tile properties */\n"
    s += "  workGroup = { " + str(solution.kernels[0].tile.workGroup[0]) \
        + ", " + str(solution.kernels[0].tile.workGroup[1]) + ", 1 };\n"
    s += "  microTile = { " + str(solution.kernels[0].tile.microTile[0]) \
        + ", " + str(solution.kernels[0].tile.microTile[1]) + ", 1 };\n"
    s += "  numUnrolls = " + str(len(solution.kernels[0].unrolls)) + ";\n"
    for i in range(0,len(solution.kernels[0].unrolls)):
      s += "  unrolls[" + str(i) + "] = " \
          + str(solution.kernels[0].unrolls[i]) + ";\n"
    s += "\n"

    # kernels
    s += "  /* kernels */\n"
    s += "  kernelGrid = { " \
        + str(solution.kernelGrid[0]) + ", " \
        + str(solution.kernelGrid[1]) + ", " \
        + str(solution.kernelGrid[2]) + " };\n"
    s += "  numKernels = " + str(len(solution.kernels)) + ";\n"
    for i in range(0, len(solution.kernels)):
      s += "  kernels[" + str(i) + "] = " \
          + self.kernelWriter.getName(solution.kernels[i]) + "__kernel;\n"
    s += "  edge = { %s, %s };\n" \
        % ("true" if solution.branch[0].isMultiple() else "false", \
        "true" if solution.branch[1].isMultiple() else "false" )
    s += "\n"

    # compile kernels if needed

    # kernel arguments
    s += "  /* kernel arguments */\n"
    s += "  numKernelArgs = 6; // pointers and enqueues\n"
    s += "\n"

    s += "  /* C strides */\n"
    for i in range(0,len(solution.kernels[0].indexOrderC)):
      s += "  kernelArgs[numKernelArgs] = &problem.tensorC.dimensions[" \
          + str(i) + "].stride; // strideC" + self.indexChars[i] + "\n"
      s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorC" \
          + ".dimensions[" + str(i) + "].stride);\n"
      s += "  numKernelArgs++;\n"
    s += "\n"

    s += "  /* A strides */\n"
    for i in range(0,len(solution.kernels[0].operation.indexAssignmentsA)):
      s += "  kernelArgs[numKernelArgs] = &problem.tensorA.dimensions[" \
          + str(i) + "].stride; // strideA" + self.indexChars[ \
          solution.kernels[0].operation.indexAssignmentsA[i]] + "\n"
      s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorA" \
          + ".dimensions[" + str(i) + "].stride);\n"
      s += "  numKernelArgs++;\n"
    s += "\n"

    s += "  /* B strides */\n"
    for i in range(0,len(solution.kernels[0].operation.indexAssignmentsB)):
      s += "  kernelArgs[numKernelArgs] = &problem.tensorB.dimensions[" \
          + str(i) + "].stride; // strideB" + self.indexChars[ \
          solution.kernels[0].operation.indexAssignmentsB[i]] + "\n"
      s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorB" \
          + ".dimensions[" + str(i) + "].stride);\n"
      s += "  numKernelArgs++;\n"
    s += "\n"

    s += "  /* free index sizes */\n"
    for i in range(0,solution.kernels[0].operation.numIndicesFree \
        + solution.kernels[0].operation.numIndicesBatch ):
      s += "  kernelArgs[numKernelArgs] = &problem.tensorB.dimensions[" \
          + str(i) + "].stride; // size" + self.indexChars[i] + "\n"
      s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorB" \
          + ".dimensions[" + str(i) + "].size);\n"
      s += "  numKernelArgs++;\n"
    s += "\n"

    s += "  /* summation index sizes */\n"
    for i in range(solution.kernels[0].operation.numIndicesFree \
          + solution.kernels[0].operation.numIndicesBatch, \
            solution.kernels[0].operation.numIndicesFree \
          + solution.kernels[0].operation.numIndicesBatch \
          + solution.kernels[0].operation.numIndicesSummation ):
      # which index of A sums this
      idx = -1
      for j in range(0,len(solution.kernels[0].operation.indexAssignmentsA)):
        if solution.kernels[0].operation.indexAssignmentsA[j] == i:
          idx = j
          break
      s += "  kernelArgs[numKernelArgs] = &problem.tensorA.dimensions[" \
          + str(idx) + "].size; // size" + self.indexChars[i] + "\n"
      s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorA" \
          + ".dimensions[" + str(idx) + "].size);\n"
      s += "  numKernelArgs++;\n"
    s += "\n"

    if solution.kernels[0].operation.alpha:
      s += "  /* alpha */\n"
      s += "  kernelArgs[numKernelArgs] = problem.operation.alpha;\n"
      s += "  kernelArgSizes[numKernelArgs] = getCobaltDataTypeSize( " \
          + "problem.operation.alphaType );\n"
      s += "  numKernelArgs++;\n"
    else:
      s += "  /* alpha unused */\n"
    s += "\n"

    if solution.kernels[0].operation.beta:
      s += "  /* beta */\n"
      s += "  kernelArgs[numKernelArgs] = problem.operation.beta;\n"
      s += "  kernelArgSizes[numKernelArgs] = getCobaltDataTypeSize( " \
          + "problem.operation.betaType );\n"
      s += "  numKernelArgs++;\n"
    else:
      s += "  /* beta unused */\n"
    s += "\n"

# close constructor
    s += "} // end constructor\n"

# open enqueue
# close enqueue
    return s


  ##############################################################################
  # getHeaderString
  ##############################################################################
  def getHeaderString(self, solution):
    solutionName = self.getName(solution)
    s = ""
    # includes
    s += "#include \"Solution.h\"\n"
    s += "\n"

    # class declaration
    s += "/* solution class */\n"
    s += "class " + solutionName + " : public CobaltSolutionOpenCL {\n"
    s += "public:\n"
    s += "  /* constructor */\n"
    s += "  " + solutionName + "( CobaltProblem inputProblem );\n"
    s += "\n"
    s += "}; // end class\n"
    return s


