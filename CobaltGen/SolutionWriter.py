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
    if solution.kernels[0].problem.operation.useAlpha:
      templateArgList += solution.kernels[0].problem.operation.alphaType.toCpp() + ","
    else:
      templateArgList += "void,"
    if solution.kernels[0].problem.operation.useBeta:
      templateArgList += solution.kernels[0].problem.operation.betaType.toCpp() + ">"
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
    if self.backend.isOpenCL():
      s += "    : SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>( inputProblem ) {\n"
    else:
      s += "    : SolutionHIP<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>( inputProblem ) {\n"
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
    s += "  this->indexAssignmentCd0 = " \
        + str(solution.kernels[0].indexAssignmentDim0) + ";\n"
    s += "  this->indexAssignmentCd1 = " \
        + str(solution.kernels[0].indexAssignmentDim1) + ";\n"

    # tensorA,B index assignments
    d0InTensorA = False
    indexAssignmentAd0or1 = -1
    indexAssignmentAdU = -1
    for i in range(0,len(solution.kernels[0].problem.operation.indexAssignmentsA)):
      index = solution.kernels[0].problem.operation.indexAssignmentsA[i]
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
    for i in range(0,len(solution.kernels[0].problem.operation.indexAssignmentsB)):
      index = solution.kernels[0].problem.operation.indexAssignmentsB[i]
      if index == solution.kernels[0].indexAssignmentDim0 \
          or index == solution.kernels[0].indexAssignmentDim1:
        indexAssignmentBd0or1 = i
      if index == len(solution.kernels[0].indexOrderC)\
          + solution.kernels[0].indexOrderSummation[ \
          len(solution.kernels[0].indexOrderSummation)-1]:
        indexAssignmentBdU = i
    s += "  this->d0InTensorA = " + ("true" if d0InTensorA else "false") + ";\n"
    s += "  this->indexAssignmentAd0or1 = " \
        + str(indexAssignmentAd0or1) + ";\n"
    s += "  this->indexAssignmentAdU = " \
        + str(indexAssignmentAdU) + ";\n"
    s += "  this->indexAssignmentBd0or1 = " \
        + str(indexAssignmentBd0or1) + ";\n"
    s += "  this->indexAssignmentBdU = " \
        + str(indexAssignmentBdU) + ";\n"
    s += "\n"

    # tile properties (common to all kernels)
    s += "  /* tile properties */\n"
    s += "  this->workGroup[0] = " \
        + str(solution.kernels[0].tile.workGroup[0]) + ";\n"
    s += "  this->workGroup[1] = " \
        + str(solution.kernels[0].tile.workGroup[1]) + ";\n"
    s += "  this->workGroup[2] = 1;\n"
    s += "  this->microTile[0] = " \
        + str(solution.kernels[0].tile.microTile[0]) + ";\n"
    s += "  this->microTile[1] = " \
        + str(solution.kernels[0].tile.microTile[1]) + ";\n"
    s += "  this->microTile[2] = 1;\n"
    s += "  size_t numUnrolls[" + str(len(solution.kernels[0].unrolls)) \
        + "] = { " + str(solution.kernels[0].unrolls[0])
    for i in range(1,len(solution.kernels[0].unrolls)):
      s += ", " + str(solution.kernels[0].unrolls[i])
    s += " };\n"
    s += "\n"

    # kernels
    s += "  /* kernels */\n"
    s += "  this->kernelGrid[0] = " + str(solution.kernelGrid[0]) + ";\n"
    s += "  this->kernelGrid[1] = " + str(solution.kernelGrid[1]) + ";\n"
    s += "  this->kernelGrid[2] = " + str(solution.kernelGrid[2]) + ";\n"
    numKernels = 0
    if self.backend.isOpenCL():
      for i in range(0, len(solution.kernels)):
        if solution.kernels[i] == None:
          s += "  this->kernelSources[" + str(i) + "] = nullptr;\n"
          s += "  this->kernels[" + str(i) + "] = nullptr;\n"
        else:
          name = self.kernelWriter.getName(solution.kernels[i])
          srcName = name + "_src"
          kernelName = name + "_kernel"
          s += "  this->kernelSources[" + str(i) + "] = " + srcName + ";\n"
          s += "  this->kernels[" + str(i) + "] = " + kernelName + ";\n"
          numKernels += 1
    s += "  this->numKernels = " + str(numKernels) + ";\n"
    # edges
    s += "  this->edge[0] = %s;\n" % ("true" if solution.branch[0].isMultiple() else "false")
    s += "  this->edge[1] = %s;\n" % ("true" if solution.branch[1].isMultiple() else "false")
    s += "  this->edge[2] = false;\n"
    s += "\n"


    # kernel arguments
    s += "  /* kernel arguments */\n"
    if self.backend.isOpenCL():
      s += "  this->numKernelArgs = 0; // pointers and offsets\n"
    else:
      s += "  this->numKernelArgs = 3; // pointers and offsets\n"
    s += "\n"

    s += "  /* preprocessor optimizations */\n"
    s += "  this->argOffsets = %s;\n" % ("true" if not solution.kernels[0].ppdOffsets else "false")
    s += "  this->argSizes = %s;\n" % ("true" if not solution.kernels[0].ppdAll else "false")
    s += "  this->argLeadingStrides = %s;\n" % ("true" if not solution.kernels[0].ppdLeadingStride else "false")
    s += "  if ( !this->argOffsets && inputProblem.useOffsets) {\n"
    s += "    throw cobaltStatusSolutionDoesNotSupportOffsets;\n"
    s += "  }\n"
    s += "  if ( (!this->argLeadingStrides && inputProblem.tensorC[0].stride != 1) || inputProblem.tensorA[0].stride != 1 ||  inputProblem.tensorB[0].stride != 1 ) {\n"
    s += "    throw cobaltStatusSolutionDoesNotSupportLeadingStrides;\n"
    s += "  }\n"
    s += "\n"
    if not solution.ppdAll:

      # strides
      firstStride = 0
      if solution.ppdLeadingStride:
        firstStride = 1
      lastStrideC = len(solution.kernels[0].indexOrderC)
      lastStrideA = len(solution.kernels[0].problem.operation.indexAssignmentsA)
      lastStrideB = len(solution.kernels[0].problem.operation.indexAssignmentsB)
      if solution.ppdAll:
        lastStrideC = firstStride
        lastStrideA = firstStride
        lastStrideB = firstStride
      s += "  /* C strides */\n"
      for i in range(firstStride,lastStrideC):
        s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorC[" \
            + str(i) + "].stride; // strideC" + self.indexChars[i] + "\n"
        if self.backend.isOpenCL():
          s += "  this->kernelArgSizes[this->numKernelArgs] = sizeof(inputProblem.tensorC" \
              + "[" + str(i) + "].stride);\n"
        s += "  this->numKernelArgs++;\n"
      s += "\n"

      s += "  /* A strides */\n"
      for i in range(firstStride,lastStrideA):
        s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorA[" \
            + str(i) + "].stride; // strideA" + self.indexChars[ \
            solution.kernels[0].problem.operation.indexAssignmentsA[i]] + "\n"
        if self.backend.isOpenCL():
          s += "  this->kernelArgSizes[this->numKernelArgs] = sizeof(inputProblem.tensorA" \
              + "[" + str(i) + "].stride);\n"
        s += "  this->numKernelArgs++;\n"
      s += "\n"

      s += "  /* B strides */\n"
      for i in range(firstStride,lastStrideB):
        s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorB[" \
            + str(i) + "].stride; // strideB" + self.indexChars[ \
            solution.kernels[0].problem.operation.indexAssignmentsB[i]] + "\n"
        if self.backend.isOpenCL():
          s += "  this->kernelArgSizes[this->numKernelArgs] = sizeof(inputProblem.tensorB" \
              + "[" + str(i) + "].stride);\n"
        s += "  this->numKernelArgs++;\n"
      s += "\n"



      s += "  /* free index sizes */\n"
      for i in range(0,solution.kernels[0].problem.operation.numIndicesFree \
          + solution.kernels[0].problem.operation.numIndicesBatch ):
        if i == solution.kernels[0].indexAssignmentDim0:
          s += "  this->kernelArgIdxDim0 = this->numKernelArgs;\n"
        if i == solution.kernels[0].indexAssignmentDim1:
          s += "  this->kernelArgIdxDim1 = this->numKernelArgs;\n"
        s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorC[" \
            + str(i) + "].size; // size" + self.indexChars[i] + "\n"
        if self.backend.isOpenCL():
          s += "  this->kernelArgSizes[this->numKernelArgs] = sizeof(inputProblem.tensorC" \
              + "[" + str(i) + "].size);\n"
        s += "  this->numKernelArgs++;\n"
      s += "\n"

      s += "  /* summation index sizes */\n"
      for i in range(solution.kernels[0].problem.operation.numIndicesFree \
            + solution.kernels[0].problem.operation.numIndicesBatch, \
              solution.kernels[0].problem.operation.numIndicesFree \
            + solution.kernels[0].problem.operation.numIndicesBatch \
            + solution.kernels[0].problem.operation.numIndicesSummation ):
        # which index of A sums this
        idx = -1
        for j in range(0,len(solution.kernels[0].problem.operation.indexAssignmentsA)):
          if solution.kernels[0].problem.operation.indexAssignmentsA[j] == i:
            idx = j
            break
        if i == \
              solution.kernels[0].problem.operation.numIndicesFree \
            + solution.kernels[0].problem.operation.numIndicesBatch \
            + solution.kernels[0].problem.operation.numIndicesSummation - 1:
          s += "  this->kernelArgIdxSummation = this->numKernelArgs;\n"
        s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorA[" \
            + str(idx) + "].size; // size" + self.indexChars[i] + "\n"
        if self.backend.isOpenCL():
          s += "  this->kernelArgSizes[this->numKernelArgs] = sizeof(inputProblem.tensorA" \
              + "[" + str(idx) + "].size);\n"
        s += "  this->numKernelArgs++;\n"
      s += "\n"

    # alpha & beta
    s += "  /* alpha & beta */\n"
    s += "  this->requireAlpha = " + ("true" if solution.kernels[0].problem.operation.useAlpha else "false")
    s += ";\n"
    s += "  this->requireBeta = " + ("true" if solution.kernels[0].problem.operation.useBeta else "false")
    s += ";\n"
    s += "\n"

    # close constructor
    s += "  /* determine globalWorkSize */\n"
    s += "  this->assignKernelArgs();\n"
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
    s += "  detail += indent + \"<ID\";\n"
    s += "  detail += \" kG0=\\\"" + str(solution.kernelGrid[0]) + "\\\"\";\n"
    s += "  detail += \" kG1=\\\"" + str(solution.kernelGrid[1]) + "\\\"\";\n"
    s += "  detail += \" kG2=\\\"" + str(solution.kernelGrid[2]) + "\\\"\";\n"
    s += "  detail += \" b0=\\\"" + str(solution.branch[0].value) + "\\\"\";\n"
    s += "  detail += \" b1=\\\"" + str(solution.branch[1].value) + "\\\"\";\n"
    s += "  detail += \" ppdO=\\\"" + str(solution.ppdOffsets) + "\\\"\";\n"
    s += "  detail += \" ppdLS=\\\"" + str(solution.ppdLeadingStride) + "\\\"\";\n"
    s += "  detail += \" ppdAll=\\\"" + str(solution.ppdAll) + "\\\"\";\n"
    s += "  detail += \">\\n\";\n"
    for k in range(0, len(solution.kernels)):
      kernel = solution.kernels[k]
      if kernel != None:
        s += "  detail += indent + \"  <K\";\n"
        s += "  detail += \" i=\\\"" + str(k) + "\\\"\";\n"
        s += "  detail += \" wG0=\\\"" + str(kernel.tile.workGroup[0]) + "\\\"\";\n"
        s += "  detail += \" wG1=\\\"" + str(kernel.tile.workGroup[1]) + "\\\"\";\n"
        s += "  detail += \" mT0=\\\"" + str(kernel.tile.microTile[0]) + "\\\"\";\n"
        s += "  detail += \" mT1=\\\"" + str(kernel.tile.microTile[1]) + "\\\"\";\n"
        s += "  detail += \" b0=\\\"" + str(kernel.tile.branch[0].value) + "\\\"\";\n"
        s += "  detail += \" b1=\\\"" + str(kernel.tile.branch[1].value) + "\\\"\";\n"
        s += "  detail += \" u0=\\\"" + str(kernel.unrolls[0]) + "\\\"\";\n"
        if len(kernel.unrolls) > 1:
          s += "  detail += \" u1=\\\"" + str(kernel.unrolls[1]) + "\\\"\";\n"
        else:
          s += "  detail += \" u1=\\\"" + str(0) + "\\\"\";\n"
        s += "  detail += \" />\\n\";\n"

    s += "  detail += indent + \"</ID>\\n\";\n"
    s += "  return detail;\n"
    s += "} // toStringDetailXML\n"
    s += "\n"

    # enqueue
    s += "template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >\n"
    s += "CobaltStatus " + solutionName \
        + "<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::enqueue(\n"
    s += "      CobaltTensorData tensorDataC,\n"
    s += "      CobaltTensorData tensorDataA,\n"
    s += "      CobaltTensorData tensorDataB,\n"
    s += "      CobaltScalarData alpha,\n"
    s += "      CobaltScalarData beta,\n"
    s += "      CobaltControl & ctrl ) {\n"
    s += "\n"
    s += "  unsigned int kernelIdx = 0;\n"
    s += "\n"
    for k in range(0, len(solution.kernels)):
      kernel = solution.kernels[k]
      if kernel != None:
        s += "  for (unsigned int i = 0; i < this->numEnqueues[kernelIdx]; i++) {\n"
        s += "    hipLaunchKernel(\n"
        s += "        HIP_KERNEL_NAME(%s),\n" \
            % self.kernelWriter.getName(kernel)
        s += "        dim3(\n"
        s += "            this->globalWorkSize[kernelIdx][0],\n"
        s += "            this->globalWorkSize[kernelIdx][1],\n"
        s += "            this->globalWorkSize[kernelIdx][2]),\n"
        s += "        dim3(\n"
        s += "            this->localWorkSize[0],\n"
        s += "            this->localWorkSize[1],\n"
        s += "            this->localWorkSize[2]),\n"
        s += "        0, // groupMemBytes\n"
        s += "        ctrl.queues[i%ctrl.numQueues],\n"
        s += "        static_cast<TypeC*>(tensorDataC.data),\n"
        s += "        static_cast<TypeA*>(tensorDataA.data),\n"
        s += "        static_cast<TypeB*>(tensorDataB.data),\n"
        s += "        *static_cast<TypeAlpha*>(alpha.data),\n"
        s += "        *static_cast<TypeBeta*>(beta.data),\n"
        s += "        this->enqueueArgs[kernelIdx][i][0]+tensorDataC.offset,\n"
        s += "        this->enqueueArgs[kernelIdx][i][1]+tensorDataA.offset,\n"
        s += "        this->enqueueArgs[kernelIdx][i][2]+tensorDataB.offset"
        numStrides = len(solution.kernels[0].problem.tensorC.dimensions) \
            + len(solution.kernels[0].problem.tensorA.dimensions) \
            + len(solution.kernels[0].problem.tensorB.dimensions)
        if solution.kernels[0].ppdLeadingStride:
          numStrides -= 3
        numSizes = len(solution.kernels[0].indexOrderC) + len(solution.kernels[0].indexOrderSummation)
        numKernelArgs = numStrides + numSizes
        for i in range(0, numKernelArgs):
          s += ",\n        this->enqueueArgs[kernelIdx][i][%u]" % (i+3) 
        s += " );\n"
        s += "  }\n"
        s += "\n"
        s += "kernelIdx++;\n"

    s += "}\n"
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
    s += "class " + solutionName
    if self.backend.isOpenCL():
      s += " : public SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {\n"
    else:
      s += " : public SolutionHIP<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {\n"
    s += "public:\n"
    s += "  /* constructor */\n"
    s += "  " + solutionName + "( const Problem & inputProblem );\n"
    #s += "  ~" + solutionName + "() {printf(\"~"+solutionName+"\\n\");}\n"
    s += "\n"
    s += "  std::string toString( size_t indentLevel) const;\n"
    s += "  std::string toStringDetailXML( size_t indentLevel) const;\n"
    s += "  CobaltStatus enqueue(\n"
    s += "      CobaltTensorData tensorDataC,\n"
    s += "      CobaltTensorData tensorDataA,\n"
    s += "      CobaltTensorData tensorDataB,\n"
    s += "      CobaltScalarData alpha,\n"
    s += "      CobaltScalarData beta,\n"
    s += "      CobaltControl & ctrl );\n"
    s += "\n"
    s += "}; // class\n"
    s += "\n"
    s += "} // namespace\n"
    s += "\n"
    s += "#endif\n"
    s += "\n"
    return s


