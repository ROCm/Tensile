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
  # getTemplateArgList
  ##############################################################################
  def getTemplateArgList(self, solution):
    templateArgList = "<"
    templateArgList += solution.kernels[0].dataTypeC.toCpp() + ","
    templateArgList += solution.kernels[0].dataTypeA.toCpp() + ","
    templateArgList += solution.kernels[0].dataTypeB.toCpp() + ","
    if solution.kernels[0].operation.useAlpha:
      templateArgList += solution.kernels[0].operation.alphaType.toCpp() + ","
    else:
      templateArgList += "void,"
    if solution.kernels[0].operation.useBeta:
      templateArgList += solution.kernels[0].operation.betaType.toCpp() + ">"
    else:
      templateArgList += "void>"
    return templateArgList


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
    s += "namespace Cobalt {\n"
    s += "\n"
    s += "/* solution constructor */\n"
    s += "template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >\n"
    s += solutionName + "<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::" + solutionName
    s += "( const Problem & inputProblem )\n"
    s += "    : SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>( inputProblem ) {\n"
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
    s += "  workGroup[0] = " \
        + str(solution.kernels[0].tile.workGroup[0]) + ";\n"
    s += "  workGroup[1] = " \
        + str(solution.kernels[0].tile.workGroup[1]) + ";\n"
    s += "  workGroup[2] = 1;\n"
    s += "  microTile[0] = " \
        + str(solution.kernels[0].tile.microTile[0]) + ";\n"
    s += "  microTile[1] = " \
        + str(solution.kernels[0].tile.microTile[1]) + ";\n"
    s += "  microTile[2] = 1;\n"
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
    numKernels = 0
    for i in range(0, len(solution.kernels)):
      if solution.kernels[i] == None:
        s += "  kernelSources[" + str(i) + "] = nullptr;\n"
        s += "  kernels[" + str(i) + "] = nullptr;\n"
      else:
        name = self.kernelWriter.getName(solution.kernels[i])
        srcName = name + "_src"
        kernelName = name + "_kernel"
        s += "  kernelSources[" + str(i) + "] = " + srcName + ";\n"
        s += "  kernels[" + str(i) + "] = " + kernelName + ";\n"
        numKernels += 1
    s += "  numKernels = " + str(numKernels) + ";\n"
    # edges
    s += "  edge[0] = %s;\n" % ("true" if solution.branch[0].isMultiple() else "false")
    s += "  edge[1] = %s;\n" % ("true" if solution.branch[1].isMultiple() else "false")
    s += "  edge[2] = false;\n"
    s += "\n"


    # kernel arguments
    s += "  /* kernel arguments */\n"
    s += "  numKernelArgs = 0; // pointers and offsets\n"
    s += "\n"

    s += "  /* preprocessor optimizations */\n"
    s += "  argOffsets = %s;\n" % ("true" if not solution.kernels[0].ppdOffsets else "false")
    s += "  argSizes = %s;\n" % ("true" if not solution.kernels[0].ppdAll else "false")

    if not solution.ppdAll:

      # strides
      firstStride = 0
      if solution.ppdLeadingStride:
        firstStride = 1
      lastStrideC = len(solution.kernels[0].indexOrderC)
      lastStrideA = len(solution.kernels[0].operation.indexAssignmentsA)
      lastStrideB = len(solution.kernels[0].operation.indexAssignmentsB)
      if solution.ppdAll:
        lastStrideC = firstStride
        lastStrideA = firstStride
        lastStrideB = firstStride
      s += "  /* C strides */\n"
      for i in range(firstStride,lastStrideC):
        s += "  kernelArgs[numKernelArgs] = &problem.tensorC[" \
            + str(i) + "].stride; // strideC" + self.indexChars[i] + "\n"
        s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorC" \
            + "[" + str(i) + "].stride);\n"
        s += "  numKernelArgs++;\n"
      s += "\n"
      
      s += "  /* A strides */\n"
      for i in range(firstStride,lastStrideA):
        s += "  kernelArgs[numKernelArgs] = &problem.tensorA[" \
            + str(i) + "].stride; // strideA" + self.indexChars[ \
            solution.kernels[0].operation.indexAssignmentsA[i]] + "\n"
        s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorA" \
            + "[" + str(i) + "].stride);\n"
        s += "  numKernelArgs++;\n"
      s += "\n"
      
      s += "  /* B strides */\n"
      for i in range(firstStride,lastStrideB):
        s += "  kernelArgs[numKernelArgs] = &problem.tensorB[" \
            + str(i) + "].stride; // strideB" + self.indexChars[ \
            solution.kernels[0].operation.indexAssignmentsB[i]] + "\n"
        s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorB" \
            + "[" + str(i) + "].stride);\n"
        s += "  numKernelArgs++;\n"
      s += "\n"
      
      
      
      s += "  /* free index sizes */\n"
      for i in range(0,solution.kernels[0].operation.numIndicesFree \
          + solution.kernels[0].operation.numIndicesBatch ):
        if i == solution.kernels[0].indexAssignmentDim0:
          s += "  kernelArgIdxDim0 = numKernelArgs;\n"
        if i == solution.kernels[0].indexAssignmentDim1:
          s += "  kernelArgIdxDim1 = numKernelArgs;\n"
        s += "  kernelArgs[numKernelArgs] = &problem.tensorC[" \
            + str(i) + "].size; // size" + self.indexChars[i] + "\n"
        s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorC" \
            + "[" + str(i) + "].size);\n"
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
        s += "  kernelArgs[numKernelArgs] = &problem.tensorA[" \
            + str(idx) + "].size; // size" + self.indexChars[i] + "\n"
        s += "  kernelArgSizes[numKernelArgs] = sizeof(problem.tensorA" \
            + "[" + str(idx) + "].size);\n"
        s += "  numKernelArgs++;\n"
      s += "\n"

    # alpha & beta
    s += "  /* alpha & beta */\n"
    s += "  requireAlpha = " + ("true" if solution.kernels[0].operation.useAlpha else "false")
    s += ";\n"
    s += "  requireBeta = " + ("true" if solution.kernels[0].operation.useBeta else "false")
    s += ";\n"
    s += "\n"

    # close constructor
    s += "  /* determine globalWorkSize */\n"
    s += "  assignWorkSizes();\n"
    s += "\n"

    # close constructor
    s += "} // constructor\n"
    s += "\n\n"

    # toString
    s += "/* toString */\n"
    s += "template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >\n"
    s += "std::string " + solutionName \
        + "<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toString( size_t indentLevel) const {\n"
    s += "  return \"" + solutionName + "\";\n"
    s += "} // toString\n"
    s += "\n"

    # toString XML
    s += "template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >\n"
    s += "std::string " + solutionName \
        + "<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toStringDetailXML( size_t indentLevel) const {\n"
    s += "  std::string indent = Cobalt::indent(indentLevel);\n"
    s += "  std::string detail = \"\";\n"
    s += "  detail += indent + \"<ImplementationDetails\";\n"
    s += "  detail += \" kernelGrid0=\\\"" + str(solution.kernelGrid[0]) + "\\\"\";\n"
    s += "  detail += \" kernelGrid1=\\\"" + str(solution.kernelGrid[1]) + "\\\"\";\n"
    s += "  detail += \" kernelGrid2=\\\"" + str(solution.kernelGrid[2]) + "\\\"\";\n"
    s += "  detail += \" branch0=\\\"" + str(solution.branch[0].value) + "\\\"\";\n"
    s += "  detail += \" branch1=\\\"" + str(solution.branch[1].value) + "\\\"\";\n"
    s += "  detail += \" ppdOffsets=\\\"" + str(solution.ppdOffsets) + "\\\"\";\n"
    s += "  detail += \" ppdLeadingStride=\\\"" + str(solution.ppdLeadingStride) + "\\\"\";\n"
    s += "  detail += \" ppdAll=\\\"" + str(solution.ppdAll) + "\\\"\";\n"
    s += "  detail += \">\\n\";\n"
    for k in range(0, len(solution.kernels)):
      kernel = solution.kernels[k]
      s += "  detail += indent + \"  <Kernel\";\n"
      if kernel != None:
        s += "  detail += \" index=\\\"" + str(k) + "\\\"\";\n"
        s += "  detail += \" name=\\\"" + self.kernelWriter.getName(kernel) + "\\\"\";\n"
        s += "  detail += \" workGroup0=\\\"" + str(kernel.tile.workGroup[0]) + "\\\"\";\n"
        s += "  detail += \" workGroup1=\\\"" + str(kernel.tile.workGroup[1]) + "\\\"\";\n"
        s += "  detail += \" microTile0=\\\"" + str(kernel.tile.microTile[0]) + "\\\"\";\n"
        s += "  detail += \" microTile1=\\\"" + str(kernel.tile.microTile[1]) + "\\\"\";\n"
        s += "  detail += \" branch0=\\\"" + str(kernel.tile.branch[0].value) + "\\\"\";\n"
        s += "  detail += \" branch1=\\\"" + str(kernel.tile.branch[1].value) + "\\\"\";\n"
        s += "  detail += \" unroll0=\\\"" + str(kernel.unrolls[0]) + "\\\"\";\n"
        if len(kernel.unrolls) > 1:
          s += "  detail += \" unroll1=\\\"" + str(kernel.unrolls[1]) + "\\\"\";\n"
        else:
          s += "  detail += \" unroll1=\\\"" + str(0) + "\\\"\";\n"
      else:
        s += "  detail += \" name=\\\"" + "None" + "\\\"\";\n"
      s += "  detail += \" />\\n\";\n"

    s += "  detail += indent + \"</ImplementationDetails>\\n\";\n"
    s += "  return detail;\n"
    s += "} // toStringDetailXML\n"
    s += "\n"

    # explicit template instantiation
    s += "/* explicit template instantiation */\n"
    #s += "template class SolutionOpenCL" \
    #    + self.getTemplateArgList(solution) + ";\n"
    s += "template class " + solutionName \
        + self.getTemplateArgList(solution) + ";\n"

    s += "\n"
    s += "} // namespace\n"
    s += "\n"

    return s


  ##############################################################################
  # getHeaderString
  ##############################################################################
  def getHeaderString(self, solution):
    solutionName = self.getName(solution)
    s = ""
    s += "#ifndef " + solutionName.upper() + "_H\n"
    s += "#define " + solutionName.upper() + "_H\n\n"
    # includes
    s += "#include \"Solution.h\"\n"
    s += "#include \"Tools.h\"\n"
    s += "\n"

    # include kernels
    for kernel in solution.kernels:
      if kernel != None:
        s += "#include \"" + self.kernelWriter.getName(kernel) + ".h\"\n"
    s += "\n"

    # class declaration
    s += "\n"
    s += "namespace Cobalt {\n"
    s += "\n"
    s += "/* solution class */\n"
    s += "template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >\n"
    s += "class " + solutionName + " : public SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {\n"
    s += "public:\n"
    s += "  /* constructor */\n"
    s += "  " + solutionName + "( const Problem & inputProblem );\n"
    #s += "  ~" + solutionName + "() {printf(\"~"+solutionName+"\\n\");}\n"
    s += "\n"
    s += "  std::string toString( size_t indentLevel) const;\n"
    s += "  std::string toStringDetailXML( size_t indentLevel) const;\n"
    s += "\n"
    s += "}; // class\n"
    s += "\n"
    s += "} // namespace\n"
    s += "\n"
    s += "#endif\n"
    s += "\n"
    return s


