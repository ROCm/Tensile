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

from SolutionStructs import Solution
from KernelWriter import KernelWriter
from Common import globalParameters, print2

################################################################################
# SolutionWriter
################################################################################
class SolutionWriter:

  indexChars = globalParameters["IndexChars"]

  ##############################################################################
  # SolutionWriter
  ##############################################################################
  def __init__(self, solutionMinNaming, solutionSerialNaming, \
      kernelMinNaming, kernelSerialNaming):
    self.backend = globalParameters["Backend"]
    self.solutionMinNaming = solutionMinNaming
    self.solutionSerialNaming = solutionSerialNaming
    self.kernelMinNaming = kernelMinNaming
    self.kernelSerialNaming = kernelSerialNaming
    self.kernelWriter = KernelWriter( kernelMinNaming, kernelSerialNaming)

    self.streamName = "hipStream_t" if self.backend == "HIP" \
        else "cl_command_queue"
    self.eventName = "hipEvent_t" if self.backend == "HIP" \
        else "cl_event"
    self.statusName = "hipError_t" if self.backend == "HIP" \
        else "cl_int"
    self.strideList = []
    self.sizeList = []


  ##############################################################################
  # get solution name
  ##############################################################################
  def getSolutionName(self, solution):
    if globalParameters["ShortNames"]:
      solutionName = Solution.getNameSerial(solution, self.solutionSerialNaming)
    else:
      solutionName = Solution.getNameMin(solution, self.solutionMinNaming)
    return solutionName


  ##############################################################################
  # getSourceString
  ##############################################################################
  def getSourceString(self, solution):
    kernels = solution.getKernels()
    kernelNames = []
    for kernel in kernels:
      kernelName = self.kernelWriter.getKernelName(kernel)
      kernelNames.append( kernelName )

    s = ""
    t = ""
    # includes


    if not globalParameters["MergeFiles"]:
      solutionName = self.getSolutionName(solution)
      s += "#include \"%s.h\"\n" % solutionName
      s += "\n"

    # solution function signature
    s += self.getSolutionSignature(solution)
    s += " {\n"
    t += "  "

    # kernels
    s += "\n%s/* kernels */\n" % (t)
    s += "%sconst unsigned int numKernels = %u; // 1 or 4\n" % (t, len(kernels))
    if self.backend == "OCL":
      s += "%sconst char *kernelSources[numKernels] = {\n" % (t)
      t += "  "
      for kernelIdx in range(0, len(kernelNames)):
        kernelName = kernelNames[kernelIdx]
        s += "%s%s_src%s\n" % (t, kernelName, \
            "," if kernelIdx < len(kernels)-1 else "" )
      t = t[2:]
      s += "%s};\n" % (t)
      s += "%scl_kernel kernels[numKernels];\n" % (t)
      s += "%sconst char *buildOptions = \"-cl-std=cl2.0\";\n" % (t)
      s += "%sfor (unsigned int i = 0; i < numKernels; i++) {\n" % (t)
      s += "%s  tensileGetCompiledOpenCLKernel(\n" % (t)
      s += "%s      &kernels[i],\n" % (t)
      s += "%s      kernelSources[i],\n" % (t)
      s += "%s      stream,\n" % (t)
      s += "%s      buildOptions);\n" % (t)
      s += "%s}\n" % (t)

    else:
      pass

    # index assignments
    s += "\n%s/* index assignments */\n" % (t)
    s += "%sconst unsigned int indexD0 = %u;\n" \
        % (t, solution["ProblemType"]["Index0"])
    s += "%sconst unsigned int indexD1 = %u;\n" \
        % (t, solution["ProblemType"]["Index1"])
    s += "%sconst unsigned int indexDU = %u;\n" \
        % (t, solution["ProblemType"]["IndexUnroll"])

    # num enqueues
    s += "\n%s/* num kernels */\n" % (t)
    kernelHasMaxSizes = solution["KernelMaxSizes"][0] > 0
    s += "%sconst bool kernelHasMaxSizes = %s;\n" % (t, "true" if kernelHasMaxSizes else "false")
    s += "%s%sconst unsigned int kernelMaxSizes[3] = { %u, %u, %u }; // max for dim 0, 1, U\n" \
        % (t, "" if kernelHasMaxSizes else "//", solution["KernelMaxSizes"][0], solution["KernelMaxSizes"][1], \
        solution["KernelMaxSizes"][0] )
    s += "%sunsigned int numEnqueues[numKernels] = { 1" % (t)
    for i in range(1, len(kernels)):
      s += ", 1"
    s += " };\n"

    # grid size
    s += "\n%s/* grid sizes */\n" % (t)
    s += "%sconst unsigned int workDim = 3;\n" % (t)
    s += "%sconst unsigned int threadTile[2] = { %u, %u };\n" \
        % (t, solution["ThreadTile0"], solution["ThreadTile1"])
    s += "%ssize_t localWorkSize[3] = { %u, %u, 1 };\n" \
        % (t, solution["WorkGroup0"], solution["WorkGroup1"])
    s += "%ssize_t globalWorkSize[numKernels][3];\n" % (t)
    # grid size [2]
    s += "%sglobalWorkSize[0][2] = 1;\n" % (t)
    for i in range(0, solution["ProblemType"]["NumIndicesC"]):
      if i != solution["ProblemType"]["Index0"] and i != solution["ProblemType"]["Index1"]:
        s += "%sglobalWorkSize[0][2] *= size%s;\n" % (t, self.indexChars[i])

    # TODO only handling kernelMaxSizes = 1,1,1 and single kernel
    s += "%s%stensileCalculateSizesForEdgeMultiKernel();\n" % (t, "" if solution["EdgeMultiKernel"] else "//" )
    s += "%s%stensileCalculateSizesForKernelMaxSizes();\n" % (t, "" if kernelHasMaxSizes else "//" )


    # grid size [0,1]
    s += "%sunsigned int sizeOfC0 = size%s;\n" % (t, \
        self.indexChars[solution["ProblemType"]["Index0"]])
    s += "%sunsigned int sizeOfC1 = size%s;\n" % (t, \
        self.indexChars[solution["ProblemType"]["Index1"]])
    s += "%sunsigned int macroTile0 = static_cast<unsigned int>(localWorkSize[0] * threadTile[0]);\n" % (t)
    s += "%sunsigned int macroTile1 = static_cast<unsigned int>(localWorkSize[1] * threadTile[1]);\n" % (t)
    s += "%sunsigned int totalWorkGroups0 = sizeOfC0 / macroTile0;\n" % (t)
    s += "%sunsigned int totalWorkGroups1 = sizeOfC1 / macroTile1;\n" % (t)

    if solution["EdgeMultiKernel"]:
      s += "%s// b/c multi kernel, don't add extra work-group here\n" % (t)
    else:
      s += "%s// b/c single kernel, add extra work-group here if edge needed\n" % (t)
      s += "%sif (totalWorkGroups0*macroTile0 < sizeOfC0) { totalWorkGroups0++; }\n" % (t)
      s += "%sif (totalWorkGroups1*macroTile1 < sizeOfC1) { totalWorkGroups1++; }\n" % (t)

    s += "%sglobalWorkSize[0][0] = totalWorkGroups0%s;\n" % (t, "*localWorkSize[0]" if self.backend == "OCL" else "")
    s += "%sglobalWorkSize[0][1] = totalWorkGroups1%s;\n" % (t, "*localWorkSize[1]" if self.backend == "OCL" else "")


    # offsets
    s += "\n%s/* offsets */\n" % (t)
    s += "%sunsigned int offsets[numKernels][1][3];\n" % (t)
    for kernelIdx in range(0, len(kernels)):
      s += "%soffsets[%u][0][0] = offsetC; // tensorC\n" % (t, kernelIdx)
      s += "%soffsets[%u][0][1] = offsetA; // tensorA\n" % (t, kernelIdx)
      s += "%soffsets[%u][0][2] = offsetB; // tensorB\n" % (t, kernelIdx)

    # index sizes
    s += "\n%s/* index sizes */\n" % (t)
    s += "%sunsigned int sizes[numKernels][1][%u];\n" \
        % (t, solution["ProblemType"]["TotalIndices"])
    for kernelIdx in range(0, len(kernels)):
      kernel = kernels[kernelIdx]
      kernelName = self.kernelWriter.getKernelName(kernel)
      # free index sizes
      for i in range(0,solution["ProblemType"]["NumIndicesFree"] \
          + solution["ProblemType"]["NumIndicesBatch"] ):
        s += "%ssizes[%u][0][%u] = size%s;\n" \
            % (t, kernelIdx, i, self.indexChars[i])
      # summation index sizes
      for i in range(solution["ProblemType"]["NumIndicesC"], \
              solution["ProblemType"]["TotalIndices"] ):
        lastParam = i == solution["ProblemType"]["TotalIndices"]-1
        s += "%ssizes[%u][0][%u] = size%s;\n" \
            % (t, kernelIdx, i, self.indexChars[i])

    s += "\n"
    s += "%sTensileStatus status;\n" % (t)
    s += "\n"

    #enqueue the kernels
    for kernelIdx in range(0, len(kernels)):
      kernel = kernels[kernelIdx]
      kernelName = self.kernelWriter.getKernelName(kernel)
      s += "\n%s/* kernel %u: %s */\n" % (t, kernelIdx, kernelName)
      s += "%sunsigned int kernelIdx = %u;\n" % (t, kernelIdx)
      typeName = solution["ProblemType"]["DataType"].toCpp()
      if self.backend == "OCL":
        # set kernel args same for all enqueues
        s += "%s// kernel args same for all enqueues\n" % (t)
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(cl_mem), &dataC ); tensileStatusCheck(status);\n" % (t, 0)
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(cl_mem), &dataA ); tensileStatusCheck(status);\n" % (t, 1)
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(cl_mem), &dataB ); tensileStatusCheck(status);\n" % (t, 2)
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(%s), &alpha ); tensileStatusCheck(status);\n" % (t, 3, typeName)
        s += "%s%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(%s), &beta ); tensileStatusCheck(status);\n" % (t, \
            "" if solution["ProblemType"]["UseBeta"] else "//", 4, typeName)
        argIdx = 5 if solution["ProblemType"]["UseBeta"] else 4
        argIdx += 3 # skipping offsets here
        for stride in self.strideList:
          s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(unsigned int), &%s ); tensileStatusCheck(status);\n" % (t, argIdx, stride)
          argIdx += 1
        for sizeIdx in range(0, solution["ProblemType"]["TotalIndices"]):
          if sizeIdx not in [ solution["ProblemType"]["Index0"],  solution["ProblemType"]["Index1"], solution["ProblemType"]["IndexUnroll"] ]:
            s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(unsigned int), &size%s ); tensileStatusCheck(status);\n" % (t, argIdx, self.indexChars[sizeIdx])
          argIdx += 1



      s += "%sfor (unsigned int enqueueIdx = 0; enqueueIdx < numEnqueues[%u]; enqueueIdx++) {\n" % (t, kernelIdx)
      t += "  "
      # debug print kernel dimensions
      if globalParameters["LibraryPrintDebug"]:
        s += "%sprintf(\"%s: g{ %%u, %%u, %%u } l{ %%u, %%u, %%u}\\n\", static_cast<unsigned int>(globalWorkSize[kernelIdx][0]), static_cast<unsigned int>(globalWorkSize[kernelIdx][1]), static_cast<unsigned int>(globalWorkSize[kernelIdx][2]), static_cast<unsigned int>(localWorkSize[0]), static_cast<unsigned int>(localWorkSize[1]), static_cast<unsigned int>(localWorkSize[2]) );\n" % (t, kernelName)
      # debug print kernel arguments
      # offsets
        for i in range(0, 3):
          s += "%sprintf(\"  offset[%u] = %%u\\n\", offsets[kernelIdx][enqueueIdx][%u]);\n" % (t, i, i)
        # strides
        for stride in self.strideList:
          s += "%sprintf(\"  %s = %%u\\n\", %s);\n" % (t, stride, stride)
        # sizes
        for i in range(0, solution["ProblemType"]["TotalIndices"]):
          s += "%sprintf(\"  sizes[kernelIdx][enqueueIdx][%u] = %%u\\n\", sizes[kernelIdx][enqueueIdx][%u] );\n" % (t, i, i )

      if self.backend == "OCL":
        # set kernel args different for all enqueues
        argIdx = 5 if solution["ProblemType"]["UseBeta"] else 4
        # offsets
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(unsigned int), &offsets[kernelIdx][enqueueIdx][0]); tensileStatusCheck(status);\n" % (t, argIdx )
        argIdx += 1
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(unsigned int), &offsets[kernelIdx][enqueueIdx][1]); tensileStatusCheck(status);\n" % (t, argIdx )
        argIdx += 1
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(unsigned int), &offsets[kernelIdx][enqueueIdx][2]); tensileStatusCheck(status);\n" % (t, argIdx )
        argIdx += 1
        argIdx += len(self.strideList)
        # sizes
        for sizeIdx in range(0, solution["ProblemType"]["TotalIndices"]):
          if sizeIdx in [ solution["ProblemType"]["Index0"],  solution["ProblemType"]["Index1"], solution["ProblemType"]["IndexUnroll"] ]:
            s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(unsigned int), &size%s ); tensileStatusCheck(status);\n" % (t, argIdx, self.indexChars[sizeIdx])
          argIdx += 1

        # enqueue
        s += "%sstatus = clEnqueueNDRangeKernel(\n" % (t)
        t += "  "
        s += "%sstream,\n" % (t)
        s += "%skernels[kernelIdx],\n" % (t)
        s += "%sworkDim,\n" % (t)
        s += "%sNULL, // globalWorkOffset\n" % (t)
        s += "%sglobalWorkSize[kernelIdx],\n" % (t)
        s += "%slocalWorkSize,\n" % (t)
        s += "%snumInputEvents,\n" % (t)
        s += "%sinputEvents,\n" % (t)
        s += "%soutputEvent );\n" % (t)
        s += "%stensileStatusCheck(status);\n" % (t)

      else:
        s += "%shipLaunchKernel(\n" % (t)
        t += "  "
        s += "%sHIP_KERNEL_NAME(%s),\n" % (t, kernelName)
        s += "%sdim3(globalWorkSize[kernelIdx][0], globalWorkSize[kernelIdx][1], globalWorkSize[kernelIdx][2]),\n" % (t)
        s += "%sdim3(localWorkSize[0], localWorkSize[1], localWorkSize[2]),\n" % (t)
        s += "%s0, // groupMemBytes\n" % (t)
        s += "%sstream,\n" % (t)
        s += "%sdataC,\n" % (t)
        s += "%sdataA,\n" % (t)
        s += "%sdataB,\n" % (t)
        s += "%salpha,\n" % (t)
        s += "%s%sbeta,\n" % (t, \
            "" if solution["ProblemType"]["UseBeta"] else "//")
        s += "%soffsets[kernelIdx][enqueueIdx][0],\n" % (t)
        s += "%soffsets[kernelIdx][enqueueIdx][1],\n" % (t)
        s += "%soffsets[kernelIdx][enqueueIdx][2],\n" % (t)
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
    s += "  return tensileStatusSuccess;\n"
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
    s += "  this->kernelMaxSizes[0] = " + str(solution["KernelMaxSizes"][0]) + ";\n"
    s += "  this->kernelMaxSizes[1] = " + str(solution["KernelMaxSizes"][1]) + ";\n"
    s += "  this->kernelMaxSizes[2] = " + str(solution["KernelMaxSizes"][2]) + ";\n"
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
      #s += "  const char *buildOptions = \"-cl-std=CL1.2\";\n"
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
    solutionName = self.getSolutionName(solution)
    s = ""
    #s += "#ifndef " + solutionName.upper() + "_H\n"
    #s += "#define " + solutionName.upper() + "_H\n\n"
    if not globalParameters["MergeFiles"]:
      s += "#pragma once\n\n"
      s += "#include \"TensileTypes.h\"\n"
      s += "#include \"SolutionHelper.h\"\n"
      s += "#include \"Tools.h\"\n"
      s += "\n"

      # include kernels
      for kernel in solution.getKernels():
        if kernel != None:
          kernelName = self.kernelWriter.getKernelName(kernel)
          s += "#include \"" + kernelName + ".h\"\n"
      s += "\n"

    # function declaration
    s += self.getSolutionSignature(solution) + ";\n"
    s += "\n"
    #s += "#endif\n"
    s += "\n"
    return s

  ########################################
  # get solution arguments
  def getArgList(self, solution):
    self.strideList = []
    self.sizeList = []
    argList = []

    # data ptrs
    typeName = solution["ProblemType"]["DataType"].toCpp()
    if self.backend == "HIP":
      argList.append("%s *dataC" % (typeName))
      argList.append("const %s *dataA" % (typeName))
      argList.append("const %s *dataB" % (typeName))
    else:
      argList.append("cl_mem dataC")
      argList.append("cl_mem dataA")
      argList.append("cl_mem dataB")
    argList.append("%s alpha" % (typeName))
    if solution["ProblemType"]["UseBeta"]:
      argList.append("%s beta" % typeName)
    argList.append("unsigned int offsetC")
    argList.append("unsigned int offsetA")
    argList.append("unsigned int offsetB")

    # initial strides ?
    firstStride = 1
    if solution["ProblemType"]["UseInitialStrides"]:
      firstStride = 0
    lastStrideC = solution["ProblemType"]["NumIndicesC"]
    lastStrideA = len(solution["ProblemType"]["IndexAssignmentsA"])
    lastStrideB = len(solution["ProblemType"]["IndexAssignmentsB"])
    # c strides
    for i in range(firstStride,lastStrideC):
      self.strideList.append("strideC%u%s" % (i, self.indexChars[i]))
    # a strides
    for i in range(firstStride,lastStrideA):
      self.strideList.append("strideA%u%s" % (i, \
          self.indexChars[solution["ProblemType"]["IndexAssignmentsA"][i]]))
    # b strides
    for i in range(firstStride,lastStrideB):
      self.strideList.append("strideB%u%s" % (i, \
          self.indexChars[solution["ProblemType"]["IndexAssignmentsB"][i]]))
    # c sizes
    for i in range(0,solution["ProblemType"]["TotalIndices"]):
      self.sizeList.append("size%s" % self.indexChars[i])
    for stride in self.strideList:
      argList.append("unsigned int %s" % stride)
    for size in self.sizeList:
      argList.append("unsigned int %s" % size)

    argList.append("%s stream" % self.streamName)
    argList.append("unsigned int numInputEvents")
    argList.append("%s *inputEvents" % self.eventName)
    argList.append("%s *outputEvent" % self.eventName)
    return argList

  ########################################
  # get function signature
  def getSolutionSignature(self, solution):
    t = "" # indent
    s = ""
    solutionName = self.getSolutionName(solution)
    s += "%s%s %s(\n" % (t, self.statusName, solutionName)
    t += "    "
    argList = self.getArgList(solution)
    for i in range(0, len(argList)):
      argString = argList[i]
      s += "%s%s%s" % (t, argString, ",\n" if i < len(argList)-1 else ")" )
    return s


  ########################################
  # get full source code
  # called from BenchmarkProblems
  def getSourceFileString(self, solution):
    fileStr = "" # CHeader
    fileStr += self.getSourceString(solution)
    return fileStr


  ########################################
  # get full header code
  # called from BenchmarkProblems
  def getHeaderFileString(self, solution):
    fileStr = "" # CHeader
    fileStr += self.getHeaderString(solution)
    return fileStr


  ##############################################################################
  # are solution parameters (dict) self-consistent
  ##############################################################################
  @ staticmethod
  def solutionParametersConsistent(solution):
    printReason = False

    numThreads = solution["WorkGroup0"]*solution["WorkGroup1"]
    if numThreads > globalParameters["MaxThreads"]:
      if printReason: print2("rejecting %u threads" % numThreads)
      return False

    # how many elements to load
    if solution["ProblemType"]["TLUA"]:
      totalElementsParaA = solution["MacroTile0"]
      totalElementsPerpA = solution["LoopUnroll"]
    else:
      totalElementsParaA = solution["LoopUnroll"]
      totalElementsPerpA = solution["MacroTile0"]

    if solution["ProblemType"]["TLUB"]:
      totalElementsParaB = solution["MacroTile1"]
      totalElementsPerpB = solution["LoopUnroll"]
    else:
      totalElementsParaB = solution["LoopUnroll"]
      totalElementsPerpB = solution["MacroTile1"]
    totalElementsA = totalElementsParaA * totalElementsPerpA
    totalElementsB = totalElementsParaB * totalElementsPerpB

    # how many load instructions
    if totalElementsA % numThreads != 0:
      if printReason: print2("totalElementsA %u %% numThreads %u != 0" \
          % (totalElementsA, numThreads))
      return False
    else:
      solution["NumLoadsA"] = totalElementsA / numThreads
    if totalElementsB % numThreads != 0:
      if printReason: print2("totalElementsB %u %% numThreads %u != 0" \
          % (totalElementsB, numThreads))
      return False
    else:
      solution["NumLoadsB"] = totalElementsB / numThreads

      # how many loads para
      if solution["NumLoadsA"] % solution["NumLoadsCoalescedA"] != 0:
        if printReason: print2("numLoadsA %u %% numLoadsParaA %u != 0" \
            % (solution["NumLoadsA"], solution["NumLoadsCoalescedA"]))
        return False
      else:
        solution["NumLoadsPerpendicularA"] = solution["NumLoadsA"] \
            / solution["NumLoadsCoalescedA"]
      if solution["NumLoadsB"] % solution["NumLoadsCoalescedB"] != 0:
        if printReason: print2("numLoadsB %u %% numLoadsParaB %u != 0" \
            % (solution["NumLoadsB"], solution["NumLoadsCoalescedB"]))
        return False
      else:
        solution["NumLoadsPerpendicularB"] = solution["NumLoadsB"] \
            / solution["NumLoadsCoalescedB"]

    # load size para/perp A
    if totalElementsParaA % solution["NumLoadsCoalescedA"] != 0:
      if printReason: print2("totalElementsParaA %u %% numLoadsParaA %u != 0" \
          % (totalElementsParaA, solution["NumLoadsCoalescedA"]))
      return False
    else:
      loadSizeParaA = totalElementsParaA / solution["NumLoadsCoalescedA"]
    if totalElementsPerpA % solution["NumLoadsPerpendicularA"] != 0:
      if printReason: print2("totalElementsPerpA %u %% numLoadsPerpA %u != 0" \
          % (totalElementsPerpA, solution["NumLoadsPerpendicularA"]))
      return False
    else:
      loadSizePerpA = totalElementsPerpA / solution["NumLoadsPerpendicularA"]

    # load size para/perp B
    if totalElementsParaB % solution["NumLoadsCoalescedB"] != 0:
      if printReason: print2("totalElementsParaB %u %% numLoadsParaB %u != 0" \
          % (totalElementsParaB, solution["NumLoadsCoalescedB"]))
      return False
    else:
      loadSizeParaB = totalElementsParaB / solution["NumLoadsCoalescedB"]
    if totalElementsPerpB % solution["NumLoadsPerpendicularB"] != 0:
      if printReason: print2("totalElementsPerpB %u %% numLoadsPerpB %u != 0" \
          % (totalElementsPerpB, solution["NumLoadsPerpendicularB"]))
      return False
    else:
      loadSizePerpB = totalElementsPerpB / solution["NumLoadsPerpendicularB"]

    # too much LDS
    sizeLDS = solution["LoopUnroll"] \
        * (solution["PadLDS"] * 2 + solution["MacroTile0"] \
        + solution["MacroTile1"] ) \
        * solution["ProblemType"]["DataType"].numBytes()
    if sizeLDS > globalParameters["MaxLDS"]:
      if printReason: print2("Kernel Uses %u > %u bytes" % ( sizeLDS, globalParameters["MaxLDS"]))
      return False

    # Compiler may be causing incorrect reads on ROCm1.4 from DT on 2/21/17
    # TODO is this a bug in KernelWriter, check against later ROCm's
    if globalParameters["Backend"] == "HIP":
      if solution["ProblemType"]["DataType"].value == DataType.single:
        if solution["MacroTile0"] == 128 or solution["MacroTile1"] == 128:
          if solution["NumLoadsCoalescedA"] != 1 and solution["NumLoadsCoalescedB"] != 8:
            return False
      elif solution["ProblemType"]["DataType"].value == DataType.double:
        if globalParameters["Backend"] == "HIP":
          if solution["MacroTile0"] == 128 or solution["MacroTile1"] == 128:
            return False

    return True

