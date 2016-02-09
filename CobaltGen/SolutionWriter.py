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
    s += "  size_t indexOrderC[" + str(len(solution.kernels[0].indexOrderC)) \
        + "] = { " + str(solution.kernels[0].indexOrderC[0])
    for i in range(1, len(solution.kernels[0].indexOrderC)):
      s += ", " + str(solution.kernels[0].indexOrderC[i])
    s += " };\n"

    s += "  size_t indexOrderSummation[" \
        + str(len(solution.kernels[0].indexOrderSummation)) + "] = { " \
        + str(solution.kernels[0].indexOrderSummation[0])
    for i in range(1, len(solution.kernels[0].indexOrderSummation)):
      s += ", " + str(solution.kernels[0].indexOrderSummation[i])
    s += " };\n"
    s += "  size_t indexAssignmentDim[3] = { " \
        + str(solution.kernels[0].indexAssignmentDim0) + ", " \
        + str(solution.kernels[0].indexAssignmentDim1) + ", " \
        + str(len(solution.kernels[0].indexOrderC) \
        +solution.kernels[0].indexOrderSummation[ \
        len(solution.kernels[0].indexOrderSummation)-1]) + " };\n"
    s += "\n"

    # tensorC index assignments
    s += "  indexAssignmentCd0 = " \
        + str(solution.kernels[0].indexAssignmentDim0) + ";\n"
    s += "  indexAssignmentCd1 = " \
        + str(solution.kernels[0].indexAssignmentDim1) + ";\n"

    # tensorA,B index assignments
    d0InTensorA = False
    indexAssignmentAd0or1 = -1
    indexAssignmentAdU = -1
    for i in range(0,len(solution.kernels[0].operation.indexAssignmentsA)):
      index = solution.kernels[0].operation.indexAssignmentsA[i]
      if index == solution.kernels[0].indexAssignmentDim0:
        d0InTensorA = True
      if index == solution.kernels[0].indexAssignmentDim0 \
          or index == solution.kernels[0].indexAssignmentDim1:
        indexAssignmentAd0or1 = i
      if index == len(solution.kernels[0].indexOrderC) \
          + solution.kernels[0].indexOrderSummation[ \
          len(solution.kernels[0].indexOrderSummation)-1]:
        indexAssignmentAdU = i
    indexAssignmentBd0or1 = -1
    indexAssignmentBdU = -1
    for i in range(0,len(solution.kernels[0].operation.indexAssignmentsB)):
      index = solution.kernels[0].operation.indexAssignmentsB[i]
      if index == solution.kernels[0].indexAssignmentDim0 \
          or index == solution.kernels[0].indexAssignmentDim1:
        indexAssignmentBd0or1 = i
      if index == len(solution.kernels[0].indexOrderC)\
          + solution.kernels[0].indexOrderSummation[ \
          len(solution.kernels[0].indexOrderSummation)-1]:
        indexAssignmentBdU = i
    s += "  d0InTensorA = " + ("true" if d0InTensorA else "false") + ";\n"
    s += "  indexAssignmentAd0or1 = " \
        + str(indexAssignmentAd0or1) + ";\n"
    s += "  indexAssignmentAdU = " \
        + str(indexAssignmentAdU) + ";\n"
    s += "  indexAssignmentBd0or1 = " \
        + str(indexAssignmentBd0or1) + ";\n"
    s += "  indexAssignmentBdU = " \
        + str(indexAssignmentBdU) + ";\n"
    s += "\n"

    # tile properties (common to all kernels)
    s += "  /* tile properties */\n"
    s += "  size_t workGroup[workDim] = { " \
        + str(solution.kernels[0].tile.workGroup[0]) \
        + ", " + str(solution.kernels[0].tile.workGroup[1]) + ", 1 };\n"
    s += "  size_t microTile[workDim] = { " \
        + str(solution.kernels[0].tile.microTile[0]) \
        + ", " + str(solution.kernels[0].tile.microTile[1]) + ", 1 };\n"
    s += "  size_t numUnrolls[" + str(len(solution.kernels[0].unrolls)) \
        + "] = { " + str(solution.kernels[0].unrolls[0])
    for i in range(1,len(solution.kernels[0].unrolls)):
      s += ", " + str(solution.kernels[0].unrolls[i])
    s += " };\n"
    s += "\n"

    # kernels
    s += "  /* kernels */\n"
    s += "  kernelGrid[0] = " + str(solution.kernelGrid[0]) + ";\n"
    s += "  kernelGrid[1] = " + str(solution.kernelGrid[1]) + ";\n"
    s += "  kernelGrid[2] = " + str(solution.kernelGrid[2]) + ";\n"
    s += "  numKernels = " + str(len(solution.kernels)) + ";\n"
    for i in range(0, len(solution.kernels)):
      if solution.kernels[i] == None:
        s += "  kernelSources[" + str(i) + "] = nullptr;\n"
        s += "  kernels[" + str(i) + "] = nullptr;\n"
      else:
        name = self.kernelWriter.getName(solution.kernels[i])
        srcName = name + "_src"
        kernelName = name + "_kernel;\n"
        s += "  kernelSources[" + str(i) + "] = " + srcName + ";\n"
        s += "  kernels[" + str(i) + "] = " + kernelName + ";\n"
    # edges
    s += "  edge[0] = %s;\n" % ("true" if solution.branch[0].isMultiple() else "false")
    s += "  edge[1] = %s;\n" % ("true" if solution.branch[1].isMultiple() else "false")
    s += "  edge[2] = false;\n"
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
      if i == solution.kernels[0].indexAssignmentDim0:
        s += "  kernelArgIdxDim0 = numKernelArgs;\n"
      if i == solution.kernels[0].indexAssignmentDim1:
        s += "  kernelArgIdxDim1 = numKernelArgs;\n"
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
      if i == \
            solution.kernels[0].operation.numIndicesFree \
          + solution.kernels[0].operation.numIndicesBatch \
          + solution.kernels[0].operation.numIndicesSummation - 1:
        s += "  kernelArgIdxSummation = numKernelArgs;\n"
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

    # include kernels
    for kernel in solution.kernels:
      if kernel != None:
        s += "#include \"" + self.kernelWriter.getName(kernel) + ".h\"\n"
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


