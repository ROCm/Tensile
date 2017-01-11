################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from Structs import *
from KernelWriter import *

################################################################################
# SolutionWriter
################################################################################
class SolutionWriter:

  indexChars = [ "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", \
      "T", "U", "V", "W", "X", "Y", "Z" ]

  ##############################################################################
  # SolutionWriter
  ##############################################################################
  def __init__(self, backend, solutionMinNaming, kernelMinNaming):
    self.backend = backend
    self.kernelWriter = KernelWriter(self.backend, solutionMinNaming)
    self.solutionMinNaming = solutionMinNaming
    self.kernelMinNaming = kernelMinNaming

    self.streamName = "hipStream_t" if self.backend == "HIP" \
        else "cl_command_queue"
    self.eventName = "hipEvent_t" if self.backend == "HIP" \
        else "cl_event"
    self.statusName = "hipStatus_t" if self.backend == "HIP" \
        else "cl_int"
    self.strideList = []
    self.sizeList = []

  ##############################################################################
  # getSourceString
  ##############################################################################
  def getSourceString(self, solution):
    solutionName = Solution.getNameMin(solution.state, self.solutionMinNaming)
    kernels = solution.getKernels()
    s = ""
    t = ""
    # includes
    s += "#include \"%s.h\"\n" % solutionName
    s += "\n"

    # solution function signature
    s += self.getSolutionSignature(solution)
    s += " {\n"
    t += "  "

    # index assignments
    s += "\n%s/* index assignments */\n" % (t)
    s += "%sconst unsigned int indexD0 = %u;\n" \
        % (t, solution["ProblemType"]["Index0"])
    s += "%sconst unsigned int indexD1 = %u\n" \
        % (t, solution["ProblemType"]["Index1"])
    s += "%sconst unsigned int indexDU = %u\n" \
        % (t, solution["ProblemType"]["IndexUnroll"])

    # num kernels
    s += "\n%s/* num kernels */\n" % (t)
    s += "%sconst unsigned int kernelGrid = { %u, %u, %u };\n" \
        % (t, solution["KernelGrid"][0], solution["KernelGrid"][1], \
        solution["KernelGrid"][0] )
    s += "%sconst unsigned int numKernels = %u;\n" % (t, len(kernels))
    s += "%sunsigned int numEnqueues[numKernels] = { 1" % (t)
    for i in range(1, len(kernels)):
      s += ", 1"
    s += " };\n"

    # grid size
    s += "\n%s/* grid sizes */\n" % (t)
    s += "%sconst unsigned int workDim = 3;\n" % (t)
    s += "%sconst unsigned int threadTile = { %u, %u };\n" \
        % (t, solution["ThreadTile0"], solution["ThreadTile1"])
    if self.backend == "HIP":
      s += "%sdim3 localWorkSize = { %u, %u, 1 };\n" \
          % (t, solution["WorkGroup0"], solution["WorkGroup1"])
      s += "%sdim3 globalWorkSize[numKernels];\n" % (t)
    # grid size [2]
    s += "%sglobalWorkSize[0][2] = 1;\n" % (t)
    for i in range(0, solution["ProblemType"]["NumIndicesC"]):
      if i != solution["ProblemType"]["Index0"] and i != solution["ProblemType"]["Index1"]:
        s += "%sglobalWorkSize[0][2] *= tensorC[%u].size;\n" % (t, i)

    # TODO only handling kernelGrid = 1,1,1 and single kernel

    # grid size [0,1]
    s += "%sunsigned int sizeOfC0 = tensorC[indexD0].size;\n" % (t)
    s += "%sunsigned int sizeOfC1 = tensorC[indexD1].size;\n" % (t)
    s += "%sunsigned int macroTile0 = localWorkSize[0] * threadTile[0];\n" % (t)
    s += "%sunsigned int macroTile1 = localWorkSize[1] * threadTile[1];\n" % (t)
    s += "%sunsigned int totalWorkGroups0 = sizeOfC0 / macroTile0;\n" % (t)
    s += "%sunsigned int totalWorkGroups1 = sizeOfC1 / macroTile1;\n" % (t)

    if solution["EdgeMultiKernel"]:
      s += "%s// b/c multi kernel, don't add extra work-group here\n" % (t)
    else:
      s += "%s// b/c single kernel, add extra work-group here if edge needed\n" % (t)
      s += "%sif (totalWorkGroups0*macroTile0 < sizeOfC0) { totalWorkGroups0++; }\n" % (t)
      s += "%sif (totalWorkGroups1*macroTile1 < sizeOfC1) { totalWorkGroups1++; }\n" % (t)
    s += "%sglobalWorkSize[0][0] = totalWorkGroups0 / kernelGrid[0];\n" % (t)
    s += "%sglobalWorkSize[0][1] = totalWorkGroups1 / kernelGrid[1];\n" % (t)


    # offsets
    s += "\n%s/* offsets */\n" % (t)
    s += "%sunsigned int offsets[numKernels][maxEnqueues][3];\n" % (t)
    for kernelIdx in range(0, len(kernels)):
      s += "%soffsets[%u][0][0] = 1; // tensorC\n" % (t, kernelIdx)
      s += "%soffsets[%u][0][1] = 1; // tensorA\n" % (t, kernelIdx)
      s += "%soffsets[%u][0][2] = 1; // tensorB\n" % (t, kernelIdx)

    # index sizes
    s += "\n%s/* index sizes */\n" % (t)
    s += "%sunsigned int sizes[numKernels][maxEnqueues][%u];\n" \
        % (t, solution["ProblemType"]["TotalIndices"])
    for kernelIdx in range(0, len(kernels)):
      kernel = kernels[kernelIdx]
      kernelName = Solution.getNameMin(kernel, self.kernelMinNaming)
      # free index sizes
      for i in range(0,solution["ProblemType"]["NumIndicesFree"] \
          + solution["ProblemType"]["NumIndicesBatch"] ):
        s += "%ssizes[%u][0][%u] = tensorC[%u].size; // size%s\n" \
            % (t, kernelIdx, i, i, self.indexChars[i])
      # summation index sizes
      for i in range(solution["ProblemType"]["NumIndicesC"], \
              solution["ProblemType"]["TotalIndices"] ):
        # which index of A sums this
        idx = -1
        for j in range(0,len(solution["ProblemType"]["IndexAssignmentsA"])):
          if solution["ProblemType"]["IndexAssignmentsA"][j] == i:
            idx = j
            break
        lastParam = i == solution["ProblemType"]["TotalIndices"]-1
        s += "%ssizes[%u][1][%u] = tensorA[%u].size; // size%s\n" \
            % (t, kernelIdx, i, idx, self.indexChars[i])

    #enqueue the kernels
    for kernelIdx in range(0, len(kernels)):
      kernel = kernels[kernelIdx]
      kernelName = Solution.getNameMin(kernel, self.kernelMinNaming)
      s += "\n%s/* kernel %u: %s */\n" % (t, kernelIdx, kernelName)
      s += "%sunsigned int kernelIdx = %u;\n" % (t, kernelIdx)
      typeName = solution["ProblemType"]["DataType"].toCpp()
      s += "%sfor (unsigned int enqueueIdx = 0; enqueueIdx < numEnqueues[%u]; enqueueIdx++) {\n" % (t, kernelIdx)
      t += "  "
      if self.backend == "HIP":
        s += "%shipLaunchKernel(\n" % (t)
        t += "  "
        s += "%sHIP_KERNEL_NAME(%s),\n" % (t, kernelName)
        s += "%sglobalWorkSize,\n" % (t)
        s += "%slocalWorkSize,\n" % (t)
        s += "%s0, // groupMemBytes\n" % (t)
        s += "%sstream,\n" % (t)
        s += "%sc,\n" % (t)
        s += "%sa,\n" % (t)
        s += "%sb,\n" % (t)
        s += "%salpha,\n" % (t)
        s += "%s%sbeta,\n" % (t, \
            "" if solution["ProblemType"]["UseBeta"] else "//")
        s += "%soffsets[kernelIdx][enqueueIdx][0] + offsetC,\n" % (t)
        s += "%soffsets[kernelIdx][enqueueIdx][1] + offsetA,\n" % (t)
        s += "%soffsets[kernelIdx][enqueueIdx][2] + offsetB,\n" % (t)
        # strides
        for stride in self.strideList:
          s += "%s%s,\n" % (t, stride)
        # sizes
        for i in range(0, solution["ProblemType"]["TotalIndices"]):
          lastParam = i == solution["ProblemType"]["TotalIndices"]-1
          s += "%ssizes[kernelIdx][enqueueIdx][%u]%s\n" \
              % (t, i, "" if lastParam else "," )
        s += "    );\n"
        s += "  }\n"
    s += "\n"
    s += "  return 0;\n"
    s += "}\n"
    s += "\n"











    """
      numStrides = kernel["ProblemType"]["NumIndicesC"] \
          + len(kernel["ProblemType"]["IndexAssignmentsA"]) \
          + len(kernel["ProblemType"]["IndexAssignmentsB"])
      if not kernel["ProblemType"]["UseInitialStrides"]:
        numStrides -= 3
      numSizes = solution["ProblemType"]["NumIndicesC"] \
          + solution["ProblemType"]["NumIndicesSummation"]
      numKernelArgs = numStrides + numSizes
      for i in range(0, numKernelArgs):
        s += ",\n        this->enqueueArgs[kernelIdx][i][%u]" % (i+3)
      s += ");\n"
      s += "//hipStreamSynchronize( ctrl.queues[enqueueIdx%ctrl.numQueues] );\n"














    # contructor signature
    s += "\n"
    s += "/* solution constructor */\n"
    s += "template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >\n"
    s += solutionName + "<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::" + solutionName
    s += "( const Problem & inputProblem )\n"
    if self.backend == "OCL":
      s += "    : SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>( inputProblem ) {\n"
    else:
      s += "    : SolutionHIP<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>( inputProblem ) {\n"
    s += "\n"

    # solution properties (common to all kernels)
    s += "  /* solution properties */\n"
    #print solution.state
    s += "  // size_t indexAssignmentDim[3] = { " \
        + str(solution["ProblemType"]["Index0"]) + ", " \
        + str(solution["ProblemType"]["Index1"]) + ", " \
        + str(solution["ProblemType"]["IndexUnroll"]) + " };\n"
    s += "\n"

    # tensorC index assignments
    s += "  this->indexAssignmentCd0 = " \
        + str(solution["ProblemType"]["Index0"]) + ";\n"
    s += "  this->indexAssignmentCd1 = " \
        + str(solution["ProblemType"]["Index1"]) + ";\n"

    # tensorA,B index assignments
    s += "  this->d0InTensorA = " + \
        ("true" if solution["ProblemType"]["TileA"] == 0 else "false") + ";\n"
    s += "  this->indexAssignmentA0or1 = " \
        + str(solution["ProblemType"]["TileA"]) + ";\n"
    s += "  this->indexAssignmentAdU = " \
        + str(solution["ProblemType"]["IndexUnrollA"]) + ";\n"
    s += "  this->indexAssignmentB0or1 = " \
        + str(solution["ProblemType"]["TileB"]) + ";\n"
    s += "  this->indexAssignmentBdU = " \
        + str(solution["ProblemType"]["IndexUnrollB"]) + ";\n"
    s += "\n"

    # tile properties (common to all kernels)
    s += "  /* tile properties */\n"
    s += "  this->workGroup[0] = " \
        + str(solution["WorkGroup0"]) + ";\n"
    s += "  this->workGroup[1] = " \
        + str(solution["WorkGroup1"]) + ";\n"
    s += "  this->workGroup[2] = 1;\n"
    s += "  this->microTile[0] = " \
        + str(solution["ThreadTile0"]) + ";\n"
    s += "  this->microTile[1] = " \
        + str(solution["ThreadTile1"]) + ";\n"
    s += "  this->microTile[2] = 1;\n"
    s += "\n"

    # kernels
    s += "  /* kernels */\n"
    s += "  this->kernelGrid[0] = " + str(solution["KernelGrid"][0]) + ";\n"
    s += "  this->kernelGrid[1] = " + str(solution["KernelGrid"][1]) + ";\n"
    s += "  this->kernelGrid[2] = " + str(solution["KernelGrid"][2]) + ";\n"
    numKernels = 0
    if self.backend == "OCL":
      for i in range(0, len(solution.kernels)):
        if solution.kernels[i] == None:
          s += "  this->kernelSources[" + str(i) + "] = nullptr;\n"
          #s += "  this->kernels[" + str(i) + "] = nullptr;\n"
        else:
          name = Solution.getName(solution.kernels[i], self.solutionMinNaming)
          srcName = name + "_src"
          kernelName = name + "_kernel"
          s += "  this->kernelSources[" + str(i) + "] = " + srcName + ";\n"
          #s += "  this->kernels[" + str(i) + "] = " + kernelName + ";\n"
          numKernels += 1
    s += "  this->numKernels = " + str(numKernels) + ";\n"
    # edges
    s += "  this->edge[0] = %s;\n" % ("true" \
        if solution["EdgeMultiKernel"] else "false")
    s += "  this->edge[1] = %s;\n" % ("true" \
        if solution["EdgeMultiKernel"] else "false")
    s += "  this->edge[2] = false;\n"
    s += "\n"


    # kernel arguments
    s += "  /* kernel arguments */\n"
    s += "  this->numKernelArgs = 3; // pointers and offsets\n"
    s += "\n"

    s += "  /* preprocessor optimizations */\n"
    s += "  // this->argOffsets = true;\n"
    s += "  // this->argSizes = true;\n"
    s += "  this->argLeadingStrides = %s;\n" % ("true" if not solution["ProblemType"]["UseInitialStrides"] else "false")
    s += "  if ( !this->argLeadingStrides && (inputProblem.tensorC[0].stride != 1 || inputProblem.tensorA[0].stride != 1 ||  inputProblem.tensorB[0].stride != 1) ) {\n"
    s += "    // problem uses leading strides but solution doesn't support offsets\n"
    s += "    // tensileGetSolution shouldn't have returned me\n"
    s += "    throw tensileStatusInvalidParameter;\n"
    s += "  }\n"
    s += "\n"

    # strides
    firstStride = 0
    if solution["ProblemType"]["UseInitialStrides"]:
      firstStride = 1
    lastStrideC = solution["ProblemType"]["NumIndicesC"]
    lastStrideA = len(solution["ProblemType"]["IndexAssignmentsA"])
    lastStrideB = len(solution["ProblemType"]["IndexAssignmentsB"])
    s += "  /* C strides */\n"
    for i in range(firstStride,lastStrideC):
      s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorC[" \
          + str(i) + "].stride; // strideC" + self.indexChars[i] + "\n"
      s += "  this->numKernelArgs++;\n"
    s += "\n"

    s += "  /* A strides */\n"
    for i in range(firstStride,lastStrideA):
      s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorA[" \
          + str(i) + "].stride; // strideA" + self.indexChars[ \
          solution["ProblemType"]["IndexAssignmentsA"][i]] + "\n"
      s += "  this->numKernelArgs++;\n"
    s += "\n"

    s += "  /* B strides */\n"
    for i in range(firstStride,lastStrideB):
      s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorB[" \
          + str(i) + "].stride; // strideB" + self.indexChars[ \
          solution["ProblemType"]["IndexAssignmentsB"][i]] + "\n"
      #if self.backend == "OCL":
      #  s += "  this->kernelArgSizes[this->numKernelArgs] = sizeof(inputProblem.tensorB" \
      #      + "[" + str(i) + "].stride);\n"
      s += "  this->numKernelArgs++;\n"
      s += "\n"



      s += "  /* free index sizes */\n"
      for i in range(0,solution["ProblemType"]["NumIndicesFree"] \
          + solution["ProblemType"]["NumIndicesBatch"] ):
        if i == solution["ProblemType"]["Index0"]:
          s += "  this->kernelArgIdxDim0 = this->numKernelArgs;\n"
        if i == solution["ProblemType"]["Index1"]:
          s += "  this->kernelArgIdxDim1 = this->numKernelArgs;\n"
        s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorC[" \
            + str(i) + "].size; // size" + self.indexChars[i] + "\n"
        #if self.backend == "OCL":
        #  s += "  this->kernelArgSizes[this->numKernelArgs] = sizeof(inputProblem.tensorC" \
        #      + "[" + str(i) + "].size);\n"
        s += "  this->numKernelArgs++;\n"
      s += "\n"

      s += "  /* summation index sizes */\n"
      for i in range(solution["ProblemType"]["NumIndicesFree"] \
            + solution["ProblemType"]["NumIndicesBatch"], \
              solution["ProblemType"]["NumIndicesFree"] \
            + solution["ProblemType"]["NumIndicesBatch"] \
            + solution["ProblemType"]["NumIndicesSummation"] ):
        # which index of A sums this
        idx = -1
        for j in range(0,len(solution["ProblemType"]["IndexAssignmentsA"])):
          if solution["ProblemType"]["IndexAssignmentsA"][j] == i:
            idx = j
            break
        if i == \
              solution["ProblemType"]["NumIndicesFree"] \
            + solution["ProblemType"]["NumIndicesBatch"] \
            + solution["ProblemType"]["NumIndicesSummation"] - 1:
          s += "  this->kernelArgIdxSummation = this->numKernelArgs;\n"
        s += "  this->kernelArgs[this->numKernelArgs] = &inputProblem.tensorA[" \
            + str(idx) + "].size; // size" + self.indexChars[i] + "\n"
        #if self.backend == "OCL":
        #  s += "  this->kernelArgSizes[this->numKernelArgs] = sizeof(inputProblem.tensorA" \
        #      + "[" + str(idx) + "].size);\n"
        s += "  this->numKernelArgs++;\n"
      s += "\n"

    # alpha & beta
    s += "  /* alpha & beta */\n"
    s += "  // this->requireAlpha = true;\n"
    s += ";\n"
    s += "  this->requireBeta = " + ("true" if solution["ProblemType"]["UseBeta"] else "false")
    s += ";\n"
    s += "\n"

    # assign kernel args
    s += "  /* determine globalWorkSize */\n"
    s += "  this->assignKernelArgs();\n"
    s += "\n"

    # compile kernels
    #if self.backend == "OCL":
      #s += "  // compile kernels\n"
      #s += "  const char *buildOptions = \"-cl-std=CL2.0\";\n"
      #s += "  for (size_t i = 0; i < this->numKernels; i++) {\n"
      #s += "    kernels[i] = nullptr;\n"
      #s += "    if (kernelSources[i]) {\n"
      #s += "      makeKernel( &kernels[i], ctrl.queues[0], kernelSources[i], buildOptions );\n"
      #s += "    }\n"
      #s += "  }\n"
      #s += "\n"


    # opencl global size *= local size
    if self.backend == "OCL":
      s += "\n"
      s += "  for (unsigned int kernelIdx = 0; kernelIdx < this->maxNumKernels; kernelIdx++) {\n"
      s += "    for (unsigned int i = 0; i < this->workDim; i++) {\n"
      s += "      this->globalWorkSize[kernelIdx][i] *= this->localWorkSize[i];\n"
      s += "    }\n"
      s += "  }\n"
      s += "\n"

    # close constructor
    s += "} // constructor\n"
    s += "\n\n"

    # toString
    s += "/* toString */\n"
    s += "template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >\n"
    s += "std::string " + solutionName \
        + "<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toString( size_t ) const {\n"
    s += "  return \"" + solutionName + "\";\n"
    s += "} // toString\n"
    s += "\n"

    # enqueue
    if self.backend == "HIP":
      s += "template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >\n"
      s += "TensileStatus " + solutionName \
          + "<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::enqueue(\n"
      s += "      TensileTensorData tensorDataC,\n"
      s += "      TensileTensorDataConst tensorDataA,\n"
      s += "      TensileTensorDataConst tensorDataB,\n"
      s += "      TensileScalarData alpha,\n"
      s += "      TensileScalarData beta,\n"
      s += "      TensileControl & ctrl ) {\n"
      s += "\n"
      s += "  unsigned int kernelIdx = 0;\n"
      s += "  unsigned int enqueueIdx = 0;\n"
      s += "\n"
      kernels = solution.getKernels()
      for k in range(0, len(kernels)):
        kernel = kernels[k]
        if kernel != None:
          s += "  for (unsigned int i = 0; i < this->numEnqueues[kernelIdx]; i++) {\n"
          s += "\n"
          if False:
            s += "printf(\"hipKernelLaunch(%s):\\n    g{%u,%u,%u};\\n    l{%u,%u,%u};\\n    p{%p,%p,%p};\\n    ab{%f,%f};\\n    o{%u,%u,%u};\\n    s{%u,%u,%u,%u,%u,%u}\\n\""
            s += ",\n        \"" + Solution.getNameMin(kernel, self.solutionMinNaming) + "\""
            s += ",\n        (unsigned int)this->globalWorkSize[kernelIdx][0], (unsigned int)this->globalWorkSize[kernelIdx][1], (unsigned int)this->globalWorkSize[kernelIdx][2]"
            s += ",\n        (unsigned int)this->localWorkSize[0], (unsigned int)this->localWorkSize[1], (unsigned int)this->localWorkSize[2]"
            s += ",\n        static_cast<TypeC*>(tensorDataC.data), static_cast<const TypeA*>(tensorDataA.data), static_cast<const TypeB*>(tensorDataB.data)"
            if kernel.dataTypeC.isReal():
              s += ",\n        *static_cast<const TypeAlpha*>(alpha.data), *static_cast<const TypeBeta*>(beta.data)"
            else:
              s += ",\n        static_cast<const TypeAlpha*>(alpha.data)->x, static_cast<const TypeBeta*>(beta.data)->y"
            s += ",\n        (unsigned int)this->enqueueArgs[kernelIdx][i][0]"
            s += ",\n        (unsigned int)this->enqueueArgs[kernelIdx][i][1]"
            s += ",\n        (unsigned int)this->enqueueArgs[kernelIdx][i][2]"
            s += ",\n        (unsigned int)this->enqueueArgs[kernelIdx][i][3]"
            s += ",\n        (unsigned int)this->enqueueArgs[kernelIdx][i][4]"
            s += ",\n        (unsigned int)this->enqueueArgs[kernelIdx][i][5]"
            s += ",\n        (unsigned int)this->enqueueArgs[kernelIdx][i][6]"
            s += ",\n        (unsigned int)this->enqueueArgs[kernelIdx][i][7]"
            s += ",\n        (unsigned int)this->enqueueArgs[kernelIdx][i][8]);\n"
            s += "\n"
          s += "    hipLaunchKernel(\n"
          s += "        HIP_KERNEL_NAME(%s),\n" \
              % Solution.getNameMin(kernel, self.kernelMinNaming)
          s += "        dim3(\n"
          s += "            this->globalWorkSize[kernelIdx][0],\n"
          s += "            this->globalWorkSize[kernelIdx][1],\n"
          s += "            this->globalWorkSize[kernelIdx][2]),\n"
          s += "        dim3(\n"
          s += "            this->localWorkSize[0],\n"
          s += "            this->localWorkSize[1],\n"
          s += "            this->localWorkSize[2]),\n"
          s += "        0, // groupMemBytes\n"
          s += "        ctrl.queues[enqueueIdx%ctrl.numQueues],\n"
          s += "        static_cast<TypeC*>(tensorDataC.data),\n"
          s += "        static_cast<const TypeA*>(tensorDataA.data),\n"
          s += "        static_cast<const TypeB*>(tensorDataB.data),\n"
          s += "        *static_cast<const TypeAlpha*>(alpha.data),\n"
          s += "        *static_cast<const TypeBeta*>(beta.data),\n"
          s += "        this->enqueueArgs[kernelIdx][i][0]+tensorDataC.offset,\n"
          s += "        this->enqueueArgs[kernelIdx][i][1]+tensorDataA.offset,\n"
          s += "        this->enqueueArgs[kernelIdx][i][2]+tensorDataB.offset"
          numStrides = kernels[0]["ProblemType"]["NumIndicesC"] \
              + len(kernels[0]["ProblemType"]["IndexAssignmentsA"]) \
              + len(kernels[0]["ProblemType"]["IndexAssignmentsB"])
          if not kernels[0]["ProblemType"]["UseInitialStrides"]:
            numStrides -= 3
          numSizes = solution["ProblemType"]["NumIndicesC"] + solution["ProblemType"]["NumIndicesSummation"]
          numKernelArgs = numStrides + numSizes
          for i in range(0, numKernelArgs):
            s += ",\n        this->enqueueArgs[kernelIdx][i][%u]" % (i+3)
          s += ");\n"
          s += "hipStreamSynchronize( ctrl.queues[enqueueIdx%ctrl.numQueues] );\n"

    #      s += "    enqueueIdx++;\n"
    #      s += "  }\n"
    #      s += "  kernelIdx++;\n"
    s += "\n"
    s += "  if (enqueueIdx > ctrl.numQueues) {\n"
    s += "    ctrl.numQueuesUsed = ctrl.numQueues;\n"
    s += "  } else {\n"
    s += "    ctrl.numQueuesUsed = enqueueIdx;\n"
    s += "  }\n"
    s += "  return tensileStatusSuccess;\n"
    s += "} // end solution_enqueue()\n"
    s += "\n"
    """

    s += "\n"
    s += "\n"
    s += "\n"
    s += "/* Solution Parameters\n"
    s += Solution.getParametersIndented(solution.state, "  ")
    s += "*/\n"
    s += "\n"

    return s


  ##############################################################################
  # getHeaderString
  ##############################################################################
  def getHeaderString(self, solution):
    solutionName = Solution.getNameMin(solution.state, self.solutionMinNaming)
    s = ""
    s += "#ifndef " + solutionName.upper() + "_H\n"
    s += "#define " + solutionName.upper() + "_H\n\n"
    # includes
    s += "//#include \"Solution.h\"\n"
    s += "//#include \"Tools.h\"\n"
    s += "\n"

    # include kernels
    for kernel in solution.getKernels():
      if kernel != None:
        s += "#include \"" + \
            Solution.getNameMin(kernel, self.kernelMinNaming) + ".h\"\n"
    s += "\n"

    # function declaration
    s += self.getSolutionSignature(solution) + ";\n"
    s += "\n"
    s += "#endif\n"
    s += "\n"
    return s

  ########################################
  # get function signature
  def getSolutionSignature(self, solution):
    self.strideList = []
    self.sizeList = []
    t = "" # indent
    s = ""
    solutionName = Solution.getNameMin(solution.state, self.solutionMinNaming)
    s += "%s%s %s(\n" % (t, self.statusName, solutionName)
    t += "    "
    # data ptrs
    typeName = solution["ProblemType"]["DataType"].toCpp()
    if self.backend == "HIP":
      s += "%s%s *c,\n" % (t, typeName)
      s += "%s%s *a,\n" % (t, typeName)
      s += "%s%s *b,\n" % (t, typeName)
    else:
      s += "%scl_mem c,\n" % (t)
      s += "%scl_mem a,\n" % (t)
      s += "%scl_mem b,\n" % (t)
    s += "%s%s alpha,\n" % (t, typeName)
    if solution["ProblemType"]["UseBeta"]:
      s += "%s%s beta,\n" % (t, typeName)
    s += "%sunsigned int offsetC,\n" % (t)
    s += "%sunsigned int offsetA,\n" % (t)
    s += "%sunsigned int offsetB,\n" % (t)

    # initial strides ?
    firstStride = 1
    if solution["ProblemType"]["UseInitialStrides"]:
      firstStride = 0
    lastStrideC = solution["ProblemType"]["NumIndicesC"]
    lastStrideA = len(solution["ProblemType"]["IndexAssignmentsA"])
    lastStrideB = len(solution["ProblemType"]["IndexAssignmentsB"])
    # c strides
    for i in range(firstStride,lastStrideC):
      #s += "%sC[%u].stride, // strideC%s\n" \
      #    % (t, i, self.indexChars[i])
      self.strideList.append("strideC%u%s" % (i, self.indexChars[i]))
    # a strides
    for i in range(firstStride,lastStrideA):
      #s += "%stensorA[%u].stride, // strideA%s\n" \
      #    % (t, i, self.indexChars[ \
      #    solution["ProblemType"]["IndexAssignmentsA"][i]] )
      self.strideList.append("strideA%u%s" % (i, \
          self.indexChars[solution["ProblemType"]["IndexAssignmentsA"][i]]))
    # b strides
    for i in range(firstStride,lastStrideB):
      #s += "%stensorB[%u].stride, // strideB%s\n" \
      #    % (t, i, self.indexChars[ \
      #    solution["ProblemType"]["IndexAssignmentsB"][i]] )
      self.strideList.append("strideB%u%s" % (i, \
          self.indexChars[solution["ProblemType"]["IndexAssignmentsB"][i]]))
    # c sizes
    for i in range(0,solution["ProblemType"]["TotalIndices"]):
      #s += "%ssizes[%u][0][%u] = tensorC[%u].size; // size%s\n" \
      #    % (t, kernelIdx, i, i, self.indexChars[i])
      self.sizeList.append("size%s" % self.indexChars[i])
    for stride in self.strideList:
      s += "%sunsigned int %s,\n" % (t, stride)
    for size in self.sizeList:
      s += "%sunsigned int %s,\n" % (t, size)

    s += "%s%s stream,\n" % (t, self.streamName)
    s += "%sunsigned int numInputEvents,\n" % (t)
    s += "%s%s *inputEvents,\n" % (t, self.eventName)
    s += "%s%s outputEvent )" % (t, self.eventName)
    return s


  ########################################
  # get full source code
  # called from BenchmarkProblems
  def getSourceFileString(self, solution):
    # TODO append copyright
    return self.getSourceString(solution)

  ########################################
  # get full header code
  # called from BenchmarkProblems
  def getHeaderFileString(self, solution):
    # TODO append copyright
    return self.getHeaderString(solution)
