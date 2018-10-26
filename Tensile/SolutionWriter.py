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
from KernelWriterSource import KernelWriterSource
from Common import globalParameters

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
    # only using getKernelName from KernelWriter so child doesn't matter
    self.kernelWriter = KernelWriterSource( kernelMinNaming, kernelSerialNaming)

    self.streamName = "hipStream_t" if self.language == "HIP" \
        else "cl_command_queue"
    self.eventName = "hipEvent_t" if self.language == "HIP" \
        else "cl_event"
    # rocblas expects Tensile routines to return hip error codes
    self.statusName = "TensileStatus"
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
  def getSourceString(self, solution, kernelsWithBuildErrs):
    kernels = solution.getKernels()
    kernelNames = []
    kernelBuildErr = 0
    for kernel in kernels:
      kernelName = self.kernelWriter.getKernelName(kernel)
      if kernelName in kernelsWithBuildErrs:
        kernelBuildErr = 1
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
    if kernelBuildErr:
      s += "%s  return tensileStatusFailure; // One or more kernels had build failures (%s)\n" % (t, kernelNames)
      s += "%s}\n" % (t)
      return s

    t += "  "
    s += "%sTensileStatus status;\n" % (t)

    # hipFunction Struct
    if solution["KernelLanguage"] == "Assembly":
      s += "\n"
      s += "%s/* module function args */\n" % (t)
      s += "%sstruct {\n" % t
      t += "  "
      if globalParameters["DebugKernel"]:
        s += "%sunsigned int *debugBuffer;\n" % t
      # Tensor sizes in bytes, excluding batch dims and accounting for zero strides
      # Do these first since they are 64-bits and want to avoid any unneeded padding:
      s += "%s// Size of lowest Tensor's lowest 2 dims, in bytes.  Does not include bath dim or higher (>2) order dimensions\n" % t
      s += "%suint64_t tensor2dSizeC;\n" % t
      s += "%suint64_t tensor2dSizeA;\n" % t
      s += "%suint64_t tensor2dSizeB;\n" % t
      solutionArgs = self.getArgList(solution["ProblemType"], True, False, False)
      for arg in solutionArgs:
        if arg[0] == "TensileHalf":
          s += "%s%s %s[2];\n" % (t, arg[0], arg[1])
        else:
          s += "%s%s %s;\n" % (t, arg[0], arg[1])


      if solution["PersistentKernel"]:
        # pass in the number of groups since not available in WG
        s += "%sunsigned int numGroupTiles0;\n" % t
        s += "%sunsigned int numGroupTiles1;\n" % t

      s += "%sunsigned int pad;\n" % t # FIXME can this be removed?
      t = t[2:]
      s += "%s} hipFunctionArgs;\n" % t
      #s += "%sprintf(\"hipFunctionArgsSize: %%lu\\n\", sizeof(hipFunctionArgs));\n" % t
      s += "%ssize_t hipFunctionArgsSize = sizeof(hipFunctionArgs);\n" % t
      s += "%svoid *hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &hipFunctionArgs, HIP_LAUNCH_PARAM_BUFFER_SIZE, &hipFunctionArgsSize, HIP_LAUNCH_PARAM_END};\n" % t
      #s += "%sprintf(\"size: %%lu\\n\", sizeof(unsigned int));\n" % t
      #s += "%sprintf(\"hipFunctionArgsSize: %%lu\\n\", sizeof(hipFunctionArgs));\n" % t
      #for arg in solutionArgs:
      #  s += "%sprintf(\"%s: %%lu\\n\", static_cast<char*>(static_cast<void*>(&hipFunctionArgs.%s)) - static_cast<char*>(static_cast<void*>(&hipFunctionArgs.%s)));\n" % (t, arg[1], arg[1], solutionArgs[0][1])

    # NOTE: host compiler aligns size of structs to 64-bits (at least) and aligns the offset of pointers to 64-bits, therefore, having pointers which are not at the beginning of the struct may get padded/shifted by the host compiler and, therefore, not coppied correctly to gpu

    if globalParameters["RuntimeLanguage"] == "HIP":
      s += "%sint deviceId;\n" % (t)
      s += "%shipGetDevice(&deviceId);\n" % (t)
    if solution["ProblemType"]["DataType"].isInt8x4() and solution["ProblemType"]["HighPrecisionAccumulate"]:
      if globalParameters["RuntimeLanguage"] == "HIP":
        s += "%shipDeviceProp_t deviceProperties;\n" % (t)
        s += "%shipGetDeviceProperties(&deviceProperties, deviceId);\n" % (t)
        s += "%sint gcnArch = deviceProperties.gcnArch;\n" % (t)
        s += "%sif(gcnArch != 906)return tensileStatusFailure;\n" % (t)

    # kernels
    s += "\n%s/* kernels */\n" % (t)
    s += "%sconst unsigned int numKernels = %u; // 1 or 4\n" % (t, len(kernels))

    if solution["KernelLanguage"] == "Source" and globalParameters["RuntimeLanguage"] == "OCL":
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

      if solution["GlobalSplitU"] > 1:
        for beta in solution.getKernelsBetaOnly():
          kernelName = self.kernelWriter.getKernelNameBetaOnly(beta)
          s += "%scl_kernel kernel_%s;\n" % (t, kernelName)
          s += "%s  tensileGetCompiledOpenCLKernel(\n" % (t)
          s += "%s      &kernel_%s,\n" % (t, kernelName)
          s += "%s      %s_src,\n" % (t, kernelName)
          s += "%s      stream,\n" % (t)
          s += "%s      buildOptions);\n" % (t)

    elif solution["KernelLanguage"] == "Assembly":
      kernel = kernels[0]
      s += "%shipFunction_t hipFunction;\n" % (t)
      s += "%sstatic SolutionLock sl;\n" % (t)
      # if !CodeFromFiles then pass global _coba that points to code object
      s += "%sstatus = sl.getFunction(&hipFunction, deviceId, \"%s\", %s);;\n" \
              % (t, kernelName, "nullptr" if globalParameters["CodeFromFiles"] else kernelName+"_coba" )
      s += "%sif (status) return status;\n" % (t)

    typeName = solution["ProblemType"]["DataType"].toCpp()

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

    if kernel["EdgeType"] != "None":
      s += "%s// b/c single kernel, add extra work-group here if edge needed\n" % (t)
      s += "%sif (totalWorkGroups0*macroTile0 < sizeOfC0) { totalWorkGroups0++; }\n" % (t)
      s += "%sif (totalWorkGroups1*macroTile1 < sizeOfC1) { totalWorkGroups1++; }\n" % (t)
    if kernel["WorkGroupMappingType"] == "Z" and abs(kernel["WorkGroupMapping"]) == 2:
      s += "%sunsigned int totalWorkGroupsPow2 = totalWorkGroups0 > totalWorkGroups1 ? totalWorkGroups0 : totalWorkGroups1;\n" % (t)
      s += "%stotalWorkGroupsPow2--;\n" % (t)
      s += "%stotalWorkGroupsPow2 |= totalWorkGroupsPow2 >> 1;\n" % (t)
      s += "%stotalWorkGroupsPow2 |= totalWorkGroupsPow2 >> 2;\n" % (t)
      s += "%stotalWorkGroupsPow2 |= totalWorkGroupsPow2 >> 4;\n" % (t)
      s += "%stotalWorkGroupsPow2 |= totalWorkGroupsPow2 >> 8;\n" % (t)
      s += "%stotalWorkGroupsPow2 |= totalWorkGroupsPow2 >> 16;\n" % (t)
      s += "%stotalWorkGroupsPow2++;\n" % (t)
      s += "%stotalWorkGroups0 = totalWorkGroupsPow2;\n" % (t)
      s += "%stotalWorkGroups1 = totalWorkGroupsPow2;\n" % (t)

    if solution["GlobalSplitU"] > 1:
      s += "%stotalWorkGroups1 *= %u; // GlobalSplitU\n" % (t, solution["GlobalSplitU"])
    if solution["PersistentKernel"]:
      s += "%shipDeviceProp_t deviceProperties;\n" % (t)
      s += "%shipGetDeviceProperties( &deviceProperties, deviceId );\n" % (t)
      s += "%sglobalWorkSize[0][0] = deviceProperties.multiProcessorCount * %u;\n" \
              % (t, solution["PersistentKernel"])
      s += "%sglobalWorkSize[0][1] = 1;\n" % t
    else:
      s += "%sglobalWorkSize[0][0] = totalWorkGroups%u%s;\n" % (t, 0 if kernel["WorkGroupMapping"] > 0 else 1, "*localWorkSize[0]" if self.language == "OCL" else "")
      s += "%sglobalWorkSize[0][1] = totalWorkGroups%u%s;\n" % (t, 1 if kernel["WorkGroupMapping"] > 0 else 0, "*localWorkSize[1]" if self.language == "OCL" else "")

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

      # Tensor2DSizes - size excluding the batch dimension, accounts for cases where one of strides is 0
      problemType = solution["ProblemType"]
      #print "IndexAssignmentsA=", problemType["IndexAssignmentsA"], "Batch=", problemType["IndicesBatch"]
      firstStride = 0 if problemType["UseInitialStrides"] else 1
      del i

      numIdx = problemType["NumIndicesC"]
      printMe = printedFree = 0
      s += "%suint64_t tensor2dSizeC = %s" % \
          (t, "1" if firstStride==1 else "strideC%u%s"% (0,self.indexChars[0]))
      for idx in range(0,numIdx):
        # Multiply only by first free and first summation
        if idx in problemType["IndicesFree"] and printedFree<2:
          printedFree += 1
          printMe = True
        else:
          printMe = False

        if printMe:
          if idx < firstStride:
            strideIdx = problemType["IndexAssignmentsA"][idx+1]
            s += " * std::max(size%s, strideA%u%s)" % \
                (self.indexChars[idx], idx+1, self.indexChars[strideIdx])
          else:
            s += " * size%s" % (self.indexChars[idx])
      s += ";\n"

      numIdx = len(problemType["IndexAssignmentsA"])
      printMe = printedStride = printedFree = printedSum = False
      s += "%suint64_t tensor2dSizeA = %s" % (t, "1" if firstStride==1 else "strideA%u%s"% (0,self.indexChars[0]))
      for i in range(0,numIdx):
        idx = problemType["IndexAssignmentsA"][i]

        # Multiply only by first free and first summation
        if idx in problemType["IndicesFree"] and not printedFree:
          printMe = printedFree = True
        elif idx in problemType["IndicesSummation"] and not printedSum:
          printMe = printedSum = True
        else:
          printMe = False

        if printMe:
          if not printedStride:
            printedStride = True
            strideIdx = problemType["IndexAssignmentsA"][i+1]
            s += " * std::max(size%s, strideA%u%s)" % \
                (self.indexChars[idx], i+1, self.indexChars[strideIdx])
          else:
            s += " * size%s" % (self.indexChars[idx])
      s += ";\n"

      numIdx = len(problemType["IndexAssignmentsB"])
      printMe = printedStride = printedFree = printedSum = False
      s += "%suint64_t tensor2dSizeB = %s" % (t, "1" if firstStride==1 else "strideB%u%s"% (0,self.indexChars[0]))
      for i in range(0,numIdx):
        idx = problemType["IndexAssignmentsB"][i]

        # Multiply only by first free and first summation
        if idx in problemType["IndicesFree"] and not printedFree:
          printMe = printedFree = True
        elif idx in problemType["IndicesSummation"] and not printedSum:
          printMe = printedSum = True
        else:
          printMe = False

        if printMe:
          if not printedStride:
            printedStride = True
            strideIdx = problemType["IndexAssignmentsB"][i+1]
            s += " * std::max(size%s, strideB%u%s)" % \
                (self.indexChars[idx], i+1, self.indexChars[strideIdx])
          else:
            s += " * size%s" % (self.indexChars[idx])
      s += ";\n"


    #s += "printf(\"Launching with grid=%zu_%zu problemGrid=%u_%u mt=%u_%u\\n\", globalWorkSize[0][0], globalWorkSize[0][1], totalWorkGroups0, totalWorkGroups1, macroTile0, macroTile1);\n"
    s += "\n"

    ########################################
    # Enqueue Beta-Only Kernel
    ########################################
    if solution["GlobalSplitU"] > 1:
      kernelNamesBetaOnly = []
      numStridesC = solution["ProblemType"]["NumIndicesC"] - \
          (0 if solution["ProblemType"]["UseInitialStrides"] else 1)
      for beta in solution.getKernelsBetaOnly():
        kernelName = self.kernelWriter.getKernelNameBetaOnly(beta)
        kernelNamesBetaOnly.append(kernelName)
      s += "%s// enqueue Beta-Only kernel\n" % (t)

      # grid sizes
      s += "%ssize_t localWorkSizeBetaOnly[3] = { 8, 8, 1};\n" % (t)
      s += "%ssize_t globalWorkSizeBetaOnly[3];\n" % (t)
      #s += "%sunsigned int sizeOfC0 = size%s;\n" % (t, \
      #    self.indexChars[solution["ProblemType"]["Index0"]])
      #s += "%sunsigned int sizeOfC1 = size%s;\n" % (t, \
      #    self.indexChars[solution["ProblemType"]["Index1"]])
      s += "%ssize_t totalWorkGroupsBetaOnly0 = sizeOfC0 / localWorkSizeBetaOnly[0];\n" % (t)
      s += "%ssize_t totalWorkGroupsBetaOnly1 = sizeOfC1 / localWorkSizeBetaOnly[1];\n" % (t)
      s += "%s// b/c single kernel, add extra work-group here if edge needed\n" % (t)
      s += "%sif (totalWorkGroupsBetaOnly0*localWorkSizeBetaOnly[0] < sizeOfC0) { totalWorkGroupsBetaOnly0++; }\n" % (t)
      s += "%sif (totalWorkGroupsBetaOnly1*localWorkSizeBetaOnly[1] < sizeOfC1) { totalWorkGroupsBetaOnly1++; }\n" % (t)
      s += "%sglobalWorkSizeBetaOnly[0] = totalWorkGroupsBetaOnly0%s;\n" % (t, "*localWorkSizeBetaOnly[0]" if self.language == "OCL" else "")
      s += "%sglobalWorkSizeBetaOnly[1] = totalWorkGroupsBetaOnly1%s;\n" % (t, "*localWorkSizeBetaOnly[1]" if self.language == "OCL" else "")
      s += "%sglobalWorkSizeBetaOnly[2] = 1;\n" % (t)
      for i in range(0, solution["ProblemType"]["NumIndicesC"]):
        if i != solution["ProblemType"]["Index0"] and i != solution["ProblemType"]["Index1"]:
          s += "%sglobalWorkSizeBetaOnly[2] *= size%s;\n" % (t, self.indexChars[i])

      if solution["ProblemType"]["UseBeta"]:
        s += "%sbool betaZero = beta == 0;\n" % (t)
      if self.language == "OCL":
        if solution["ProblemType"]["UseBeta"]:
          s += "%scl_kernel kernelBetaOnly = betaZero ? kernel_%s : kernel_%s;\n" \
              % (t, kernelNamesBetaOnly[0], kernelNamesBetaOnly[1])
        else:
          #s += "%sbool betaZero = true;\n" % (t)
          s += "%scl_kernel kernelBetaOnly = kernel_%s;\n" \
              % (t, kernelNamesBetaOnly[0])
        argIdx = 0
        s += "%sstatus = clSetKernelArg( kernelBetaOnly, %u, sizeof(cl_mem), &dataC ); tensileStatusCheck(status);\n" % (t, argIdx); argIdx+=1
        s += "%sstatus = clSetKernelArg( kernelBetaOnly, %u, sizeof(unsigned int), &offsetC ); tensileStatusCheck(status);\n" % (t, argIdx); argIdx+=1
        # strides
        for i in range(0,numStridesC):
          s += "%sstatus = clSetKernelArg( kernelBetaOnly, %u, sizeof(unsigned int), &%s ); tensileStatusCheck(status);\n" % (t, argIdx, self.strideList[i]); argIdx+=1
        # sizes
        for i in range(0, solution["ProblemType"]["NumIndicesC"]):
          s += "%sstatus = clSetKernelArg( kernelBetaOnly, %u, sizeof(unsigned int), &size%s ); tensileStatusCheck(status);\n" % (t, argIdx, self.indexChars[i]); argIdx+=1
        # beta
        if solution["ProblemType"]["UseBeta"]:
          s += "%sif (!betaZero) {\n" % (t)
          s += "%s  status = clSetKernelArg( kernelBetaOnly, %u, sizeof(%s), &beta ); tensileStatusCheck(status);\n" % (t, argIdx, typeName); argIdx+=1
          s += "%s}\n" % (t)
        # enqueue
        s += "%scl_event kernelEventBetaOnly;\n" % (t)
        s += "%sstatus = clEnqueueNDRangeKernel(\n" % (t)
        t += "  "
        s += "%sstream,\n" % (t)
        s += "%skernelBetaOnly,\n" % (t)
        s += "%sworkDim,\n" % (t)
        s += "%sNULL, // globalWorkOffset\n" % (t)
        s += "%sglobalWorkSizeBetaOnly,\n" % (t)
        s += "%slocalWorkSizeBetaOnly,\n" % (t)
        s += "%snumInputEvents,\n" % (t)
        s += "%sinputEvents,\n" % (t)
        #s += "%soutputEvent );\n" % (t)
        s += "%s&kernelEventBetaOnly );\n" % (t)
        t = t[2:]
        s += "%stensileStatusCheck(status);\n" % (t)
        if solution["ProblemType"]["UseBeta"]:
          s += "%sbeta = %s;\n" % (t, solution["ProblemType"]["DataType"].zeroString(self.language, 1) )
        #s += "%sreturn tensileStatusSuccess;\n" % (t)
        s += "%sstatus = clFinish(stream);\n" % (t)
        s += "%stensileStatusCheck(status);\n" % (t)
        #s += " float tmp[128*128];\n"
        #s += "clEnqueueReadBuffer(stream, dataC, CL_TRUE, 0, 128*128*sizeof(float), tmp, 0, NULL, NULL);\n"
        #s += "for (unsigned int i = 0; i < 128*128; i++) { printf(\"%f\\n\", tmp[i]); }\n"


      else:
        s += "%sif( inputEvents != NULL )\n" % (t)
        t += "  "
        s += "%shipEventRecord(inputEvents[0], stream );\n" % (t)
        t += "  "
        s += "%stry {\n" % (t)
        if solution["ProblemType"]["UseBeta"]:
          s += "%sif (betaZero) {\n" % (t)
          t += "  "
        s += "%shipLaunchKernelGGL(\n" % (t)
        t += "  "
        s += "%sHIP_KERNEL_NAME(%s),\n" % (t, kernelNamesBetaOnly[0])
        s += "%sdim3(globalWorkSizeBetaOnly[0], globalWorkSizeBetaOnly[1], globalWorkSizeBetaOnly[2]),\n" % (t)
        s += "%sdim3(localWorkSizeBetaOnly[0], localWorkSizeBetaOnly[1], localWorkSizeBetaOnly[2]),\n" % (t)
        s += "%s0, // groupMemBytes\n" % (t)
        s += "%sstream,\n" % (t)
        s += "%sdataC,\n" % (t)
        s += "%soffsetC,\n" % (t)
        # strides
        for i in range(0,numStridesC):
          s += "%s%s,\n" % (t, self.strideList[i])
        # sizes
        for i in range(0, solution["ProblemType"]["NumIndicesC"]):
          s += "%ssize%s%s" % (t, self.indexChars[i], ",\n" if i < solution["ProblemType"]["NumIndicesC"]-1 else ");\n")

        if solution["ProblemType"]["UseBeta"]:
          s += "%s} else {\n" % (t)
          s += "%shipLaunchKernelGGL(\n" % (t)
          t += "  "
          s += "%sHIP_KERNEL_NAME(%s),\n" % (t, kernelNamesBetaOnly[1])
          s += "%sdim3(globalWorkSizeBetaOnly[0], globalWorkSizeBetaOnly[1], globalWorkSizeBetaOnly[2]),\n" % (t)
          s += "%sdim3(localWorkSizeBetaOnly[0], localWorkSizeBetaOnly[1], localWorkSizeBetaOnly[2]),\n" % (t)
          s += "%s0, // groupMemBytes\n" % (t)
          s += "%sstream,\n" % (t)
          s += "%sdataC,\n" % (t)
          s += "%soffsetC,\n" % (t)
          # strides
          for i in range(0,numStridesC):
            s += "%s%s,\n" % (t, self.strideList[i])
          # sizes
          for i in range(0, solution["ProblemType"]["NumIndicesC"]):
            s += "%ssize%s,\n" % (t, self.indexChars[i])
          s += "%sbeta);\n" % (t)
          s += "%s}\n" % (t)

        s += "%s} catch (const std::exception& e) {\n" % (t)
        s += "#ifdef DEBUG\n"
        s += "%s  std::cerr << e.what() << std::endl;\n" % (t)
        s += "#endif\n"
        s += "%s  return tensileStatusFailure;\n" % (t)
        s += "%s}\n" % (t)

    ########################################
    # Enqueue Kernels
    ########################################
    for kernelIdx in range(0, len(kernels)):
      kernel = kernels[kernelIdx]
      if kernel["KernelLanguage"] == "Source":
        kernel["ISA"] = (0, 0, 0) # HIP source kernels needs dummy ISA version
      kernelName = self.kernelWriter.getKernelName(kernel)
      s += "\n%s/* kernel %u: %s */\n" % (t, kernelIdx, kernelName)
      s += "%sunsigned int kernelIdx = %u;\n" % (t, kernelIdx)
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
        s += "%sprintf(\"  tensor2dSizeC== %%lu\\n\", tensor2dSizeC );\n" % (t)
        s += "%sprintf(\"  tensor2dSizeA== %%lu\\n\", tensor2dSizeA );\n" % (t)
        s += "%sprintf(\"  tensor2dSizeB== %%lu\\n\", tensor2dSizeB );\n" % (t)

      ########################################
      # OpenCL Runtime
      ########################################
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
        if False: # solution["GlobalSplitU"] > 1:
          s += "%s1,\n" % (t)
          s += "%s&kernelEventBetaOnly,\n" % (t)
        else:
          s += "%snumInputEvents,\n" % (t)
          s += "%sinputEvents,\n" % (t)
        s += "%soutputEvent );\n" % (t)
        s += "%stensileStatusCheck(status);\n" % (t)
        s += "%s}\n" % (t)

      ########################################
      # HIP Runtime
      ########################################
      else:
        if not globalParameters["PreciseKernelTime"] or solution["KernelLanguage"] == "Source":
          s += "%sif( inputEvents != NULL )\n" % (t)
          t += "  "
          s += "%shipEventRecord(inputEvents[enqueueIdx], stream );\n" % (t)
        t = t[2:]
        s += "%stry {\n" % (t)
        t += "  "
        # hip kernel
        if solution["KernelLanguage"] == "Source":
          s += "%shipLaunchKernelGGL(\n" % (t)
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
          if solution["PersistentKernel"]:
            s += "%s,totalWorkGroups%u\n" % (t, 0 if kernel["WorkGroupMapping"] > 0 else 1)
            s += "%s,totalWorkGroups%u\n" % (t, 1 if kernel["WorkGroupMapping"] > 0 else 0)
          s += "%s);\n" % (t)

        # assembly kernel
        else:
          if globalParameters["DebugKernel"]:
            s += "%sconst unsigned int debugBufferElementsPerThread = 16;\n" % t
            s += "%sunsigned int debugBufferNumElem = debugBufferElementsPerThread;\n" % (t)
            s += "%sdebugBufferNumElem *= max(1,globalWorkSize[kernelIdx][0]);\n" % (t)
            s += "%sdebugBufferNumElem *= max(1,globalWorkSize[kernelIdx][1]);\n" % (t)
            s += "%sdebugBufferNumElem *= max(1,globalWorkSize[kernelIdx][2]);\n" % (t)
            s += "%sdebugBufferNumElem *= localWorkSize[0];\n" % (t)
            s += "%sdebugBufferNumElem *= localWorkSize[1];\n" % (t)
            s += "%sdebugBufferNumElem *= localWorkSize[2];\n" % (t)
            s += "%s  printf(\"debugBufferNumElem: %%04i: \\n\", debugBufferNumElem);\n" % (t)
            s += "%ssize_t debugBufferSize = debugBufferNumElem * sizeof(unsigned int);\n" % (t)
            s += "%shipDevice_t device;\n" % t
            s += "%shipDeviceGet(&device, 0);\n" % t
            s += "%shipMalloc(&(hipFunctionArgs.debugBuffer), debugBufferSize);\n" % t
            s += "%sunsigned int *debugBufferHostPtr = new unsigned int[debugBufferNumElem];\n" % (t)
            s += "%smemset(debugBufferHostPtr,0,debugBufferSize);\n" % (t)
            s += "%shipMemcpyHtoD(hipFunctionArgs.debugBuffer, debugBufferHostPtr, debugBufferSize);\n" % (t)
            s += "%smemset(debugBufferHostPtr,1,debugBufferSize);\n" % (t)

          # hip assembly function
          s += "%shipFunctionArgs.dataC = dataC;\n" % (t)
          s += "%shipFunctionArgs.dataA = dataA;\n" % (t)
          s += "%shipFunctionArgs.dataB = dataB;\n" % (t)
          if solution["ProblemType"]["DataType"].isHalf():
            s += "%shipFunctionArgs.alpha[0] = alpha;\n" % (t)
            s += "%shipFunctionArgs.alpha[1] = alpha;\n" % (t)
          else:
            s += "%shipFunctionArgs.alpha = alpha;\n" % (t)
          if solution["ProblemType"]["UseBeta"]:
            if solution["ProblemType"]["DataType"].isHalf():
              s += "%shipFunctionArgs.beta[0] = beta;\n" % (t)
              s += "%shipFunctionArgs.beta[1] = beta;\n" % (t)
            else:
              s += "%shipFunctionArgs.beta = beta;\n" % (t)
          s += "%shipFunctionArgs.offsetC = offsets[kernelIdx][enqueueIdx][0];\n" % (t)
          s += "%shipFunctionArgs.offsetA = offsets[kernelIdx][enqueueIdx][1];\n" % (t)
          s += "%shipFunctionArgs.offsetB = offsets[kernelIdx][enqueueIdx][2];\n" % (t)
          # strides
          for stride in self.strideList:
            s += "%shipFunctionArgs.%s = %s;\n" % (t, stride, stride)
          # sizes
          for i in range(0, solution["ProblemType"]["TotalIndices"]):
            lastParam = i == solution["ProblemType"]["TotalIndices"]-1
            s += "%shipFunctionArgs.size%s = sizes[kernelIdx][enqueueIdx][%u];\n" \
                % (t, globalParameters["IndexChars"][i], i )

          s += "%shipFunctionArgs.tensor2dSizeC = tensor2dSizeC;\n" % (t)
          s += "%shipFunctionArgs.tensor2dSizeA = tensor2dSizeA;\n" % (t)
          s += "%shipFunctionArgs.tensor2dSizeB = tensor2dSizeB;\n" % (t)

          if solution["PersistentKernel"]:
            # pass in the number of groups since not available in WG
            s += "%shipFunctionArgs.numGroupTiles0 = totalWorkGroups0;\n" % (t)
            s += "%shipFunctionArgs.numGroupTiles1 = totalWorkGroups1;\n" % (t)

          s += "%shipHccModuleLaunchKernel(\n" % (t)
          t += "  "
          s += "%shipFunction,\n" % (t)
          s += "%sglobalWorkSize[kernelIdx][0]*localWorkSize[0],\n" % (t)
          s += "%sglobalWorkSize[kernelIdx][1]*localWorkSize[1],\n" % (t)
          s += "%sglobalWorkSize[kernelIdx][2]*localWorkSize[2],\n" % (t)
          s += "%slocalWorkSize[0],\n" % (t)
          s += "%slocalWorkSize[1],\n" % (t)
          s += "%slocalWorkSize[2],\n" % (t)
          s += "%s0, // groupMemBytes\n" % (t)
          s += "%sstream,\n" % (t)
          s += "%sNULL,\n" % (t)
          s += "%s(void**)hipLaunchParams\n" % (t)
          if globalParameters["PreciseKernelTime"]:
            s += "%s,inputEvents ? inputEvents[enqueueIdx]:nullptr\n" %(t)
            s += "%s,outputEvent ? outputEvent[enqueueIdx]:nullptr\n" % (t)

          s += "%s);\n" % (t)
          t = t[2:]
          if globalParameters["DebugKernel"]:
            # copy debug buffer
            s += "%shipMemcpyDtoH(debugBufferHostPtr, hipFunctionArgs.debugBuffer, debugBufferSize);\n" % (t)
            s += "%sfor(unsigned int i = 0; i < debugBufferNumElem/debugBufferElementsPerThread; i++) {\n" % (t)
            s += "%s  printf(\"%%04i\", i);\n" % (t)
            s += "%s  char u[debugBufferElementsPerThread] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};\n" % (t)
            #s += "%s  char u[debugBufferElementsPerThread] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};\n" % (t)
            #s += "%s  char u[debugBufferElementsPerThread] = {1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1};\n" % (t)
            s += "%s  for(unsigned int j = 0; j < debugBufferElementsPerThread; j++) {\n" % (t)
            s += "%s if (u[j]) printf(\",%%4u\", debugBufferHostPtr[i*debugBufferElementsPerThread+j]);\n" % (t)
            s += "%s else printf(\",%%4.0f\", ((float *)debugBufferHostPtr)[i*debugBufferElementsPerThread+j]);\n" % (t)

            s += "%s  }\n" % (t)
            s += "%s  printf(\"\\n\");\n" % (t)
            s += "%s}\n" % (t)


        t = t[2:]
        s += "%s} catch (const std::exception& e) {\n" % (t)
        s += "#ifdef DEBUG\n"
        s += "%s  std::cerr << e.what() << std::endl;\n" % (t)
        s += "#endif\n"
        s += "%s  return tensileStatusFailure;\n" % (t)
        s += "%s}\n" % (t)
        if not globalParameters["PreciseKernelTime"] or solution["KernelLanguage"] == "Source":
          s += "%sif( outputEvent != NULL )\n" % (t)
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
      for kernel in solution.getKernelsBetaOnly():
        kernelName = self.kernelWriter.getKernelNameBetaOnly(kernel)
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
  def getArgList(self, problemType, includeData, includeEvents, includeStream):
    self.strideList = []
    self.sizeList = []
    argList = []

    # data ptrs
    if includeData:
      typeName = problemType["DataType"].toCpp()
      if self.language == "HIP":
        argList.append(("%s *"%typeName, "dataC"))
        argList.append(("const %s *"%typeName, "dataA"))
        argList.append(("const %s *"%typeName, "dataB"))
      else:
        argList.append(("cl_mem", "dataC"))
        argList.append(("cl_mem", "dataA"))
        argList.append(("cl_mem", "dataB"))
      argList.append((typeName, "alpha"))
      if problemType["UseBeta"]:
        argList.append((typeName, "beta"))
      argList.append(("unsigned int", "offsetC"))
      argList.append(("unsigned int", "offsetA"))
      argList.append(("unsigned int", "offsetB"))

    # initial strides ?
    firstStride = 1
    if problemType["UseInitialStrides"]:
      firstStride = 0
    lastStrideC = problemType["NumIndicesC"]
    lastStrideA = len(problemType["IndexAssignmentsA"])
    lastStrideB = len(problemType["IndexAssignmentsB"])
    # c strides
    for i in range(firstStride,lastStrideC):
      self.strideList.append("strideC%u%s" % (i, self.indexChars[i]))
    # a strides
    for i in range(firstStride,lastStrideA):
      self.strideList.append("strideA%u%s" % (i, \
          self.indexChars[problemType["IndexAssignmentsA"][i]]))
    # b strides
    for i in range(firstStride,lastStrideB):
      self.strideList.append("strideB%u%s" % (i, \
          self.indexChars[problemType["IndexAssignmentsB"][i]]))
    # c sizes
    for i in range(0,problemType["TotalIndices"]):
      self.sizeList.append("size%s" % self.indexChars[i])
    for stride in self.strideList:
      argList.append(("unsigned int",stride))
    for size in self.sizeList:
      argList.append(("unsigned int", size))
    if includeStream:
      argList.append((self.streamName, "stream"))
    if includeEvents:
      argList.append(("unsigned int", "numInputEvents"))
      argList.append(("%s *"%self.eventName, "inputEvents"))
      argList.append(("%s *"%self.eventName, "outputEvent"))
    return argList

  ########################################
  # get function signature
  def getSolutionSignature(self, solution):
    t = "" # indent
    s = ""
    solutionName = self.getSolutionName(solution)
    s += "%s%s %s(\n" % (t, self.statusName, solutionName)
    t += "    "
    argList = self.getArgList(solution["ProblemType"], True, True, True)
    for i in range(0, len(argList)):
      argString = "%s %s" % argList[i]
      s += "%s%s%s" % (t, argString, ",\n" if i < len(argList)-1 else ")" )
    return s


  ########################################
  # get full source code
  # called from BenchmarkProblems
  def getSourceFileString(self, solution, kernelsWithBuildErrs):
    fileStr = "" # CHeader
    fileStr += self.getSourceString(solution, kernelsWithBuildErrs)
    return fileStr


  ########################################
  # get full header code
  # called from BenchmarkProblems
  def getHeaderFileString(self, solution):
    fileStr = "" # CHeader
    fileStr += self.getHeaderString(solution)
    return fileStr


