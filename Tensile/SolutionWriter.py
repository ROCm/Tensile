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

from SolutionStructs import Solution, DataType
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
    self.language = globalParameters["RuntimeLanguage"]
    self.solutionMinNaming = solutionMinNaming
    self.solutionSerialNaming = solutionSerialNaming
    self.kernelMinNaming = kernelMinNaming
    self.kernelSerialNaming = kernelSerialNaming
    self.kernelWriter = KernelWriter( kernelMinNaming, kernelSerialNaming)

    self.streamName = "hipStream_t" if self.language == "HIP" \
        else "cl_command_queue"
    self.eventName = "hipEvent_t" if self.language == "HIP" \
        else "cl_event"
    self.statusName = "hipError_t" if self.language == "HIP" \
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
    if globalParameters["KernelLanguage"] == "OCL":
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
    s += "%sunsigned int numEnqueues[numKernels] = { 1" % (t)
    for i in range(1, len(kernels)):
      s += ", 1"
    s += " };\n"

    # grid size
    s += "\n%s/* grid sizes */\n" % (t)
    s += "%sconst unsigned int workDim = 3;\n" % (t)
    s += "%sconst unsigned int threadTile[2] = { %u, %u };\n" \
        % (t, solution["ThreadTile0"], solution["ThreadTile1"])
    s += "%sconst unsigned int groupSize[2] = { %u, %u };\n" \
        % (t, solution["SubGroup0"], solution["SubGroup1"])
    s += "%ssize_t localWorkSize[3] = { %3u, 1, 1 };\n" \
        % (t, solution["NumThreads"])
    s += "%ssize_t globalWorkSize[numKernels][3];\n" % (t)
    # grid size [2]
    s += "%sglobalWorkSize[0][2] = 1;\n" % (t)
    for i in range(0, solution["ProblemType"]["NumIndicesC"]):
      if i != solution["ProblemType"]["Index0"] and i != solution["ProblemType"]["Index1"]:
        s += "%sglobalWorkSize[0][2] *= size%s;\n" % (t, self.indexChars[i])


    # grid size [0,1]
    s += "%sunsigned int sizeOfC0 = size%s;\n" % (t, \
        self.indexChars[solution["ProblemType"]["Index0"]])
    s += "%sunsigned int sizeOfC1 = size%s;\n" % (t, \
        self.indexChars[solution["ProblemType"]["Index1"]])
    s += "%sunsigned int macroTile0 = static_cast<unsigned int>(groupSize[0] * threadTile[0]);\n" % (t)
    s += "%sunsigned int macroTile1 = static_cast<unsigned int>(groupSize[1] * threadTile[1]);\n" % (t)
    s += "%sunsigned int totalWorkGroups0 = sizeOfC0 / macroTile0;\n" % (t)
    s += "%sunsigned int totalWorkGroups1 = sizeOfC1 / macroTile1;\n" % (t)

    s += "%s// b/c single kernel, add extra work-group here if edge needed\n" % (t)
    s += "%sif (totalWorkGroups0*macroTile0 < sizeOfC0) { totalWorkGroups0++; }\n" % (t)
    s += "%sif (totalWorkGroups1*macroTile1 < sizeOfC1) { totalWorkGroups1++; }\n" % (t)

    s += "%sglobalWorkSize[0][0] = totalWorkGroups0%s;\n" % (t, "*localWorkSize[0]" if self.language == "OCL" else "")
    s += "%sglobalWorkSize[0][1] = totalWorkGroups1%s;\n" % (t, "*localWorkSize[1]" if self.language == "OCL" else "")


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
      if self.language == "OCL":
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

      if self.language == "OCL":
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
        s += "%sif( inputEvents != nullptr )\n" % (t)
        s += "%s  hipEventRecord(inputEvents[enqueueIdx], stream );\n" % (t)
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
        s += "%sif( outputEvent != nullptr )\n" % (t)
        s += "%s  hipEventRecord(outputEvent[enqueueIdx], stream );\n" % (t)
      s += "  }\n"
    s += "\n"
    s += "  return tensileStatusSuccess;\n"
    s += "}\n"
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
    s = ""
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
    if self.language == "HIP":
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


