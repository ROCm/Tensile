################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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

from .SolutionStructs import Solution, isPackedIndex
from .KernelWriterSource import KernelWriterSource
from .Common import globalParameters

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
  def getProblemSourceString(self, problemType, solution, kernelsWithBuildErrs):
    gsu = solution["GlobalSplitU"]
    persistent = solution["PersistentKernel"]
    kernelLanguage = solution["KernelLanguage"]
    tt0 = solution["ThreadTile0"]
    tt1 = solution["ThreadTile1"]
    sg0 = solution["SubGroup0"]
    sg1 = solution["SubGroup1"]
    nt  = solution["NumThreads"]

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

    problemType = solution["ProblemType"] # shortcut

    if not globalParameters["MergeFiles"]:
      solutionName = self.getSolutionName(solution)
      s += "#include \"%s.h\"\n" % solutionName
      s += "\n"

    s += self.getSolutionSignature(solution)

    s += " {\n"
    if kernelBuildErr:
      s += "%s  return tensileStatusFailure; // One or more kernels had build failures (%s)\n" % (t, kernelNames)
      s += "%s}\n" % (t)
      return s

    t += "  "
    s += "%sTensileStatus status;\n" % (t)


    # hipFunction Struct
    if kernelLanguage == "Assembly":
      s += "\n"
      s += "%s/* module function args */\n" % (t)
      s += "%sstruct {\n" % t
      t += "  "
      if globalParameters["DebugKernel"]:
        s += "%sunsigned int *debugBuffer;\n" % t
      # Tensor sizes in elements, including only packed dims,
      # and accounting for zero or other strides < size
      # Place these first in the structure since they are 64-bits
      # and need to avoid any unneeded padding:
      s += "%s// Size of Tensor's packed dims, in elements\n" % t
      s += "%suint64_t tensor2dSizeC;\n" % t
      s += "%suint64_t tensor2dSizeA;\n" % t
      s += "%suint64_t tensor2dSizeB;\n" % t
      solutionArgs = self.getArgList(problemType, False, True, False, False, False, solution["_GlobalAccumulation"])
      for arg in solutionArgs:
        if arg[0] == "TensileHalf":
          s += "%s%s %s[2];\n" % (t, arg[0], arg[1])
        else:
          s += "%s%s %s;\n" % (t, arg[0], arg[1])
      for idxChar in solution["PackedC0IdxChars"][:-1]:
        s += "%sunsigned magicNumberSize%s;\n" % (t, idxChar)
        s += "%sunsigned magicShiftSize%s;\n" % (t, idxChar)
      for idxChar in solution["PackedC1IdxChars"][:-1]:
        s += "%sunsigned magicNumberSize%s;\n" % (t, idxChar)
        s += "%sunsigned magicShiftSize%s;\n" % (t, idxChar)

      # number of unroll loop iterations to stagger the start in "U" dim.
      s += "%sint staggerUIter;\n" % t

      # persistent
      s += "%sunsigned int problemNumGroupTiles0;\n" % t
      s += "%sunsigned int problemNumGroupTiles1;\n" % t
      s += "%sunsigned int magicNumberProblemNumGroupTiles0;\n" % t
      s += "%sunsigned int gridNumWorkGroups0;\n" % t
      s += "%sunsigned int numFullBlocks;\n" % t
      s += "%sunsigned int wgmRemainder1;\n" % t
      s += "%sunsigned int magicNumberWgmRemainder1;\n" % t

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

    # kernels
    s += "\n%s/* kernels */\n" % (t)
    s += "%sconst unsigned int numKernels = %u; // 1 or 4\n" % (t, len(kernels))

    if kernelLanguage == "Source" and globalParameters["RuntimeLanguage"] == "OCL":
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

      if gsu > 1:
        for ko in solution.getKernelBetaOlnyObjects():
          kernelName = ko.getKernelName(ko)
          s += "%scl_kernel kernel_%s;\n" % (t, kernelName)
          s += "%s  tensileGetCompiledOpenCLKernel(\n" % (t)
          s += "%s      &kernel_%s,\n" % (t, kernelName)
          s += "%s      %s_src,\n" % (t, kernelName)
          s += "%s      stream,\n" % (t)
          s += "%s      buildOptions);\n" % (t)

    elif kernelLanguage == "Assembly":
      kernel = kernels[0]
      s += "%shipFunction_t hipFunction;\n" % (t)
      # if !CodeFromFiles then pass global _coba that points to code object
      s += "%sstatus = solutionLock->getFunction(&hipFunction, deviceId, \"%s\", %s);;\n" \
              % (t, kernelName, "nullptr" if globalParameters["CodeFromFiles"] else kernelName+"_coba" )
      s += "%sif (status) return status;\n" % (t)

    typeName = problemType["DataType"].toCpp()

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
        % (t, tt0, tt1)
    s += "%sconst unsigned int groupSize[2] = { %u, %u };\n" \
        % (t, sg0, sg1)
    s += "%ssize_t localWorkSize[3] = { %3u, 1, 1 };\n" \
        % (t, nt)
    s += "%ssize_t globalWorkSize[numKernels][3];\n" % (t)
    # grid size [2]
    s += "%sglobalWorkSize[0][2] = 1;\n" % (t)
    for i in range(0, problemType["NumIndicesC"]):
      if i != problemType["Index0"] and i != problemType["Index1"] \
          and not isPackedIndex(solution,i):
        s += "%sglobalWorkSize[0][2] *= size%s;\n" % (t, self.indexChars[i])

    s += "%sunsigned int sizeOfC0 = " % (t)
    s += " * ".join(["size" + i for i in solution["PackedC0IdxChars"]])
    s += ";\n"

    s += "%sunsigned int sizeOfC1 = " % (t)
    s += " * ".join(["size" + i for i in solution["PackedC1IdxChars"]])
    s += ";\n"

    for idxChar in solution["PackedC0IdxChars"][:-1]:
      s += "%sunsigned magicShiftSize%s = 33; // bozo, review\n" % (t, idxChar)
      s += "%suint64_t magicNumberSize%s = (1L<<magicShiftSize%s) / size%s + 1;\n" \
          % (t, idxChar, idxChar, idxChar)
      s += "%sif (magicNumberSize%s >> 32) { magicShiftSize%s=31; magicNumberSize%s = (1L<<magicShiftSize%s) / size%s + 1;}\n" \
          % (t, idxChar, idxChar, idxChar, idxChar, idxChar)
    for idxChar in solution["PackedC1IdxChars"][:-1]:
      s += "%sunsigned magicShiftSize%s = 33; // bozo, review\n" % (t, idxChar)
      s += "%suint64_t magicNumberSize%s = (1L<<magicShiftSize%s) / size%s + 1;\n" \
              % (t, idxChar, idxChar, idxChar)
      s += "%sif (magicNumberSize%s >> 32) { magicShiftSize%s=31; magicNumberSize%s = (1L<<magicShiftSize%s) / size%s + 1;}\n" \
          % (t, idxChar, idxChar, idxChar, idxChar, idxChar)

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

    # persistent:
    s += "%sunsigned int problemNumGroupTiles0 = totalWorkGroups0;\n" % (t)
    s += "%sunsigned int problemNumGroupTiles1 = totalWorkGroups1;\n" % (t)
    s += "%sconst unsigned smallNumMagicShift = 31; // bozo, review\n" % (t)
    s += "%sunsigned magicNumberProblemNumGroupTiles0 = (1L<<smallNumMagicShift) / problemNumGroupTiles0 + 1; // bozo, review\n"  % (t)

    if kernel["WorkGroupMapping"] > 0:
        s += "%sunsigned numFullBlocks =  problemNumGroupTiles1 / %u; // divide by WorkGroupMapping\n" % (t, kernel["WorkGroupMapping"])
        s += "%sunsigned wgmRemainder1 =  problemNumGroupTiles1 %% %u;\n" % (t, kernel["WorkGroupMapping"])
        s += "%sif (wgmRemainder1 == 0) wgmRemainder1 = %u;\n" % (t, kernel["WorkGroupMapping"])
        s += "%sunsigned magicNumberWgmRemainder1 = ((1L<<smallNumMagicShift) / wgmRemainder1 + 1);\n"  % (t)
    else:
        s += "%sunsigned numFullBlocks =  problemNumGroupTiles1; // divide by WorkGroupMapping\n" % (t)
        s += "%sunsigned wgmRemainder1 =  0;\n" % (t)
        s += "%sunsigned magicNumberWgmRemainder1 = 0;\n"  % (t)

    #s += '  printf ("wgmRemainder1=%u \\n", wgmRemainder1);\n'
    #s += '  printf ("magicNumberWgmRemainder1=%u \\n", magicNumberWgmRemainder1);\n'

    if gsu> 1:
      s += "%stotalWorkGroups1 *= %u; // GlobalSplitU\n" % (t, gsu)
    if persistent:
      s += "%shipDeviceProp_t deviceProperties;\n" % (t)
      # TODO - should cache the device properties - expensive to call on each iteration here:
      s += "%shipGetDeviceProperties( &deviceProperties, deviceId );\n" % (t)
      s += "%sunsigned int numGroups = totalWorkGroups0 * totalWorkGroups1;\n" % (t)
      s += "%sglobalWorkSize[0][0] = (deviceProperties.multiProcessorCount * %u < numGroups) ? (deviceProperties.multiProcessorCount * %u) : numGroups;\n" \
              % (t, persistent, persistent)

      s += "%sglobalWorkSize[0][1] = 1;\n" % t
    else:
      s += "%sglobalWorkSize[0][0] = totalWorkGroups%u%s;\n" % (t, 0 if kernel["WorkGroupMapping"] >= 0 else 1, "*localWorkSize[0]" if self.language == "OCL" else "")
      s += "%sglobalWorkSize[0][1] = totalWorkGroups%u%s;\n" % (t, 1 if kernel["WorkGroupMapping"] >= 0 else 0, "*localWorkSize[1]" if self.language == "OCL" else "")

    # index sizes
    s += "\n%s/* index sizes */\n" % (t)
    s += "%sunsigned int sizes[numKernels][1][%u];\n" \
        % (t, problemType["TotalIndices"])
    for kernelIdx in range(0, len(kernels)):
      kernel = kernels[kernelIdx]
      kernelName = self.kernelWriter.getKernelName(kernel)
      # free index sizes
      for i in range(0,problemType["NumIndicesFree"] \
          + problemType["NumIndicesBatch"] ):
        s += "%ssizes[%u][0][%u] = size%s;\n" \
            % (t, kernelIdx, i, self.indexChars[i])
      # summation index sizes
      for i in range(problemType["NumIndicesC"], \
              problemType["TotalIndices"] ):
        lastParam = i == problemType["TotalIndices"]-1
        s += "%ssizes[%u][0][%u] = size%s;\n" \
            % (t, kernelIdx, i, self.indexChars[i])

      # Tensor2DSizes - size excluding the batch dimension, accounts for cases where one of strides is 0
      #print "IndexAssignmentsA=", problemType["IndexAssignmentsA"], "Batch=", problemType["IndicesBatch"]
      firstStride = 0 if problemType["UseInitialStridesCD"] else 1
      del i

      numIdx = problemType["NumIndicesC"]
      printMe = 0
      s += "%suint64_t tensor2dSizeC = %s" % \
          (t, "1" if firstStride==1 else "strideC%u%s"% (0,self.indexChars[0]))
      for idx in range(0,numIdx):
        # Multiply only by packed tensor dims
        if idx in problemType["IndicesFree"]:
          printMe = True
        else:
          printMe = False

        if printMe:
          if idx+1 < numIdx:
            strideIdx = idx+1
            s += " * std::max(size%s, strideC%u%s)" % \
                (self.indexChars[idx], idx+1, self.indexChars[strideIdx])
          else:
            s += " * size%s" % (self.indexChars[idx])
      s += ";\n"

      s += "%suint64_t tensor2dSizeA = 1;\n" % t
      s += "%suint64_t tensor2dSizeAStride = 0;\n" % t
      s += "%suint64_t tensor2dSizeAOffset = 0;\n" % t
      numIdx = len(problemType["IndexAssignmentsA"])

      printMe = False
      for i in range(0,numIdx):
        idx = problemType["IndexAssignmentsA"][i]

        # Don't multiple batch dimensions that will be backed into SRD:
        if idx in solution["PackedC0IndicesX"]:
          printMe = True
        elif idx in problemType["IndicesSummation"]:
          printMe = True
        else:
          printMe = False

        if printMe:
          if i+1 < numIdx:
            strideIdx = problemType["IndexAssignmentsA"][i+1]
            s += "%stensor2dSizeAStride = std::max(tensor2dSizeA*size%s, (uint64_t)strideA%u%s);\n" \
                % (t, self.indexChars[idx], i+1, self.indexChars[strideIdx])
            s += "%stensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size%s;\n" \
                % (t, self.indexChars[idx])
            s += "%stensor2dSizeA = tensor2dSizeAStride;\n" % (t)
          else:
            s += "%stensor2dSizeA = tensor2dSizeA * size%s;\n" % (t, self.indexChars[idx])

      s += "%stensor2dSizeA -= tensor2dSizeAOffset;\n" % t
      s += "\n"

      s += "%suint64_t tensor2dSizeB = 1;\n" % t
      s += "%suint64_t tensor2dSizeBStride = 0;\n" % t
      s += "%suint64_t tensor2dSizeBOffset = 0;\n" % t
      numIdx = len(problemType["IndexAssignmentsB"])
      printMe = False
      for i in range(0,numIdx):
        idx = problemType["IndexAssignmentsB"][i]

        # Multiply only by first free and first summation
        if idx in solution["PackedC1IndicesX"]:
          printMe = True
        elif idx in problemType["IndicesSummation"]:
          printMe = True
        else:
          printMe = False

        if printMe:
          if i+1 < numIdx:
            strideIdx = problemType["IndexAssignmentsB"][i+1]
            s += "%stensor2dSizeBStride = std::max(tensor2dSizeB*size%s, (uint64_t)strideB%u%s);\n" \
                % (t, self.indexChars[idx], i+1, self.indexChars[strideIdx])
            s += "%stensor2dSizeBOffset += tensor2dSizeBStride - tensor2dSizeB*size%s;\n" \
                % (t, self.indexChars[idx])
            s += "%stensor2dSizeB = tensor2dSizeBStride;\n" % (t)
          else:
            s += "%stensor2dSizeB = tensor2dSizeB * size%s;\n" % (t, self.indexChars[idx])

      s += "%stensor2dSizeB -= tensor2dSizeBOffset;\n" % t
      s += "\n"

    unrollChar = globalParameters["IndexChars"][problemType["IndexUnroll"]]

    s += "  unsigned int staggerUIter = %s; // how many stride-sized clicks to stagger start offset\n" \
        % (solution["StaggerU"])
    s += "  int unrollLoopIters = size%s/%u/%u; // /DepthU/GSU\n" % (unrollChar, solution["DepthU"], gsu)
    s += "  while (staggerUIter>1) {\n"
    s += "    if (unrollLoopIters >= (staggerUIter*%u)) {\n" % (1<<solution["_staggerStrideShift"])
    s += "      break;}\n"
    s += "    staggerUIter /= 2; // step down to smaller stagger\n"
    s += "  }\n"
    s += "  if (staggerUIter>=1) staggerUIter -= 1;\n" # convert to a mask
    #s += '  printf ("size%s=%%u StaggerU=%s unrollLoopIters=%%u, staggerUIter=%%d\\n", size%s, unrollLoopIters, staggerUIter);\n' % (unrollChar, solution["StaggerU"], unrollChar)



    #s += "printf(\"Launching with grid=%zu_%zu problemGrid=%u_%u mt=%u_%u\\n\", globalWorkSize[0][0], globalWorkSize[0][1], totalWorkGroups0, totalWorkGroups1, macroTile0, macroTile1);\n"
    s += "\n"
    s += "%sint kernelsLaunched=0;\n" % (t)

    ########################################
    # Enqueue Beta-Only Kernel
    ########################################
    if gsu > 1:
      kernelNamesBetaOnly = []
      numStridesC = problemType["NumIndicesC"] - \
          (0 if problemType["UseInitialStridesCD"] else 1)
      for ko in solution.getKernelBetaOlnyObjects():
        kernelName = ko.getKernelName()
        kernelNamesBetaOnly.append(kernelName)
      s += "%s// enqueue Beta-Only kernel\n" % (t)

      # grid sizes
      s += "%ssize_t localWorkSizeBetaOnly[3] = { 8, 8, 1};\n" % (t)
      s += "%ssize_t globalWorkSizeBetaOnly[3];\n" % (t)
      #s += "%sunsigned int sizeOfC0 = size%s;\n" % (t, \
      #    self.indexChars[problemType["Index0"]])
      #s += "%sunsigned int sizeOfC1 = size%s;\n" % (t, \
      #    self.indexChars[problemType["Index1"]])
      s += "%ssize_t totalWorkGroupsBetaOnly0 = sizeOfC0 / localWorkSizeBetaOnly[0];\n" % (t)
      s += "%ssize_t totalWorkGroupsBetaOnly1 = sizeOfC1 / localWorkSizeBetaOnly[1];\n" % (t)
      s += "%s// b/c single kernel, add extra work-group here if edge needed\n" % (t)
      s += "%sif (totalWorkGroupsBetaOnly0*localWorkSizeBetaOnly[0] < sizeOfC0) { totalWorkGroupsBetaOnly0++; }\n" % (t)
      s += "%sif (totalWorkGroupsBetaOnly1*localWorkSizeBetaOnly[1] < sizeOfC1) { totalWorkGroupsBetaOnly1++; }\n" % (t)
      s += "%sglobalWorkSizeBetaOnly[0] = totalWorkGroupsBetaOnly0%s;\n" % (t, "*localWorkSizeBetaOnly[0]" if self.language == "OCL" else "")
      s += "%sglobalWorkSizeBetaOnly[1] = totalWorkGroupsBetaOnly1%s;\n" % (t, "*localWorkSizeBetaOnly[1]" if self.language == "OCL" else "")
      s += "%sglobalWorkSizeBetaOnly[2] = 1;\n" % (t)
      for i in range(0, problemType["NumIndicesC"]):
        if i != problemType["Index0"] and i != problemType["Index1"]:
          s += "%sglobalWorkSizeBetaOnly[2] *= size%s;\n" % (t, self.indexChars[i])

      if problemType["UseBeta"]:
        s += "%sbool betaZero = beta == (%s)0;\n" % (t, typeName)
      if self.language == "OCL":
        if problemType["UseBeta"]:
          s += "%scl_kernel kernelBetaOnly = betaZero ? kernel_%s : kernel_%s;\n" \
              % (t, kernelNamesBetaOnly[0], kernelNamesBetaOnly[1])
        else:
          #s += "%sbool betaZero = true;\n" % (t)
          s += "%scl_kernel kernelBetaOnly = kernel_%s;\n" \
              % (t, kernelNamesBetaOnly[0])
        argIdx = 0
        s += "%sstatus = clSetKernelArg( kernelBetaOnly, %u, sizeof(cl_mem), &dataC ); tensileStatusCheck(status);\n" % (t, argIdx); argIdx+=1
        # strides
        for i in range(0,numStridesC):
          s += "%sstatus = clSetKernelArg( kernelBetaOnly, %u, sizeof(unsigned int), &%s ); tensileStatusCheck(status);\n" % (t, argIdx, self.strideList[i]); argIdx+=1
        # sizes
        for i in range(0, problemType["NumIndicesC"]):
          s += "%sstatus = clSetKernelArg( kernelBetaOnly, %u, sizeof(unsigned int), &size%s ); tensileStatusCheck(status);\n" % (t, argIdx, self.indexChars[i]); argIdx+=1
        # beta
        if problemType["UseBeta"]:
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
        if problemType["UseBeta"]:
          s += "%sbeta = %s;\n" % (t, problemType["DataType"].zeroString(self.language, 1) )
        #s += "%sreturn tensileStatusSuccess;\n" % (t)
        s += "%sstatus = clFinish(stream);\n" % (t)
        s += "%stensileStatusCheck(status);\n" % (t)
        #s += " float tmp[128*128];\n"
        #s += "clEnqueueReadBuffer(stream, dataC, CL_TRUE, 0, 128*128*sizeof(float), tmp, 0, NULL, NULL);\n"
        #s += "for (unsigned int i = 0; i < 128*128; i++) { printf(\"%f\\n\", tmp[i]); }\n"
      else:
        s += "%stry {\n" % (t)
        t += "  "
        # TODO - timing with beta kernels is somewhat pessimistic since it has this separate event only on the GSU path.
        # Introduces 2-3us of overhead ; may want to disable PreciseKernelTime so non-GSU have same overhead.
        # Long-term fix would be to launch the beta kernel with the hipHccModule* API and set start-event in that call
        if problemType["UseBeta"]:
          s += "%sif (betaZero) {\n" % (t)
          t += "  "
        s += "%sif( inputEvents != NULL )\n" % (t)
        s += "%s  hipEventRecord(inputEvents[0], stream );\n" % (t)
        s += "%skernelsLaunched++;\n" % (t)
        s += "%shipLaunchKernelGGL(\n" % (t)
        t += "  "
        s += "%sHIP_KERNEL_NAME(%s),\n" % (t, kernelNamesBetaOnly[0])
        s += "%sdim3(globalWorkSizeBetaOnly[0], globalWorkSizeBetaOnly[1], globalWorkSizeBetaOnly[2]),\n" % (t)
        s += "%sdim3(localWorkSizeBetaOnly[0], localWorkSizeBetaOnly[1], localWorkSizeBetaOnly[2]),\n" % (t)
        s += "%s0, // groupMemBytes\n" % (t)
        s += "%sstream,\n" % (t)
        s += "%sworkspace,\n" % (t) if solution["_GlobalAccumulation"] else ("%sdataD,\n" % (t))
        s += "%sdataC,\n" % (t)
        # strides
        for i in range(0,numStridesC*2):
          s += "%s%s,\n" % (t, self.strideList[i])
        # sizes
        for i in range(0, problemType["NumIndicesC"]):
          s += "%ssize%s%s" % (t, self.indexChars[i], ",\n" if i < problemType["NumIndicesC"]-1 else ");\n")
        t = t[:-2]

        if problemType["UseBeta"]:
          t = t[:-2]
          s += "%s} else {\n" % (t)
          t += "  "
          s += "%sif( inputEvents != NULL )\n" % (t)
          s += "%s  hipEventRecord(inputEvents[0], stream );\n" % (t)
          s += "%skernelsLaunched++;\n" % (t)
          s += "%shipLaunchKernelGGL(\n" % (t)
          t += "  "
          s += "%sHIP_KERNEL_NAME(%s),\n" % (t, kernelNamesBetaOnly[1])
          s += "%sdim3(globalWorkSizeBetaOnly[0], globalWorkSizeBetaOnly[1], globalWorkSizeBetaOnly[2]),\n" % (t)
          s += "%sdim3(localWorkSizeBetaOnly[0], localWorkSizeBetaOnly[1], localWorkSizeBetaOnly[2]),\n" % (t)
          s += "%s0, // groupMemBytes\n" % (t)
          s += "%sstream,\n" % (t)
          s += ("%sworkspace,\n") % (t) if solution["_GlobalAccumulation"] else ("%sdataD,\n" % (t))
          s += "%sdataC,\n" % (t)
          # strides
          for i in range(0,numStridesC*2):
            s += "%s%s,\n" % (t, self.strideList[i])
          # sizes
          for i in range(0, problemType["NumIndicesC"]):
            s += "%ssize%s,\n" % (t, self.indexChars[i])
          s += "%sbeta);\n" % (t)
          t = t[2:]
          t = t[:-2]
          s += "%s}\n" % (t)

        t = t[:-2]
        s += "%s} catch (const std::exception& e) {\n" % (t)
        t += "  "
        s += "#ifdef DEBUG\n"
        s += "%s  std::cerr << e.what() << std::endl;\n" % (t)
        s += "#endif\n"
        s += "%s  return tensileStatusFailure;\n" % (t)
        t = t[:-2]
        s += "%s}\n" % (t)

    ########################################
    # Enqueue Kernels
    ########################################
    for kernelIdx in range(0, len(kernels)):
      kernel = kernels[kernelIdx]
      if kernel["KernelLanguage"] == "Source":
        kernel["ISA"] = [0, 0, 0] # HIP source kernels needs dummy ISA version
      kernelName = self.kernelWriter.getKernelName(kernel)
      s += "\n%s/* kernel %u: %s */\n" % (t, kernelIdx, kernelName)
      s += "%sunsigned int kernelIdx = %u;\n" % (t, kernelIdx)
      if self.language == "OCL":
        # set kernel args same for all enqueues
        s += "%s// kernel args same for all enqueues\n" % (t)
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(cl_mem), &dataD ); tensileStatusCheck(status);\n" % (t, 0)
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(cl_mem), &dataC ); tensileStatusCheck(status);\n" % (t, 1)
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(cl_mem), &dataA ); tensileStatusCheck(status);\n" % (t, 2)
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(cl_mem), &dataB ); tensileStatusCheck(status);\n" % (t, 3)
        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(%s), &alpha ); tensileStatusCheck(status);\n" % (t, 4, typeName)
        s += "%s%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(%s), &beta ); tensileStatusCheck(status);\n" % (t, \
            "" if problemType["UseBeta"] else "//", 5, typeName)
        argIdx = 6 if problemType["UseBeta"] else 5
        for stride in self.strideList:
          s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(unsigned int), &%s ); tensileStatusCheck(status);\n" % (t, argIdx, stride)
          argIdx += 1
        for sizeIdx in range(0, problemType["TotalIndices"]):
          if sizeIdx not in [ problemType["Index0"],  problemType["Index1"], problemType["IndexUnroll"] ]:
            s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(unsigned int), &size%s ); tensileStatusCheck(status);\n" % (t, argIdx, self.indexChars[sizeIdx])
          argIdx += 1

        s += "%sstatus = clSetKernelArg( kernels[kernelIdx], %u, sizeof(staggerUIter), &staggerUIter ); tensileStatusCheck(status);\n" % (t, argIdx)
        argIdx += 1

      s += "%sfor (unsigned int enqueueIdx = 0; enqueueIdx < numEnqueues[%u]; enqueueIdx++) {\n" % (t, kernelIdx)
      t += "  "
      # debug print kernel dimensions
      if globalParameters["LibraryPrintDebug"]:
        s += "%sprintf(\"%s: g{ %%u, %%u, %%u } l{ %%u, %%u, %%u}\\n\", static_cast<unsigned int>(globalWorkSize[kernelIdx][0]), static_cast<unsigned int>(globalWorkSize[kernelIdx][1]), static_cast<unsigned int>(globalWorkSize[kernelIdx][2]), static_cast<unsigned int>(localWorkSize[0]), static_cast<unsigned int>(localWorkSize[1]), static_cast<unsigned int>(localWorkSize[2]) );\n" % (t, kernelName)
        # debug print kernel arguments
        # strides
        for stride in self.strideList:
          s += "%sprintf(\"  %s = %%u\\n\", %s);\n" % (t, stride, stride)
        # sizes
        for i in range(0, problemType["TotalIndices"]):
          s += "%sprintf(\"  sizes[kernelIdx][enqueueIdx][%u] = %%u\\n\", sizes[kernelIdx][enqueueIdx][%u] );\n" % (t, i, i )
        s += "%sprintf(\"  staggerUIter == %%u\\n\", staggerUIter );\n" % (t)
        s += "%sprintf(\"  problemNumGroupTiles0== %%u\\n\", problemNumGroupTiles0 );\n" % (t)
        s += "%sprintf(\"  problemNumGroupTiles1== %%u\\n\", problemNumGroupTiles1 );\n" % (t)
        s += "%sprintf(\"  tensor2dSizeC== %%lu\\n\", tensor2dSizeC );\n" % (t)
        s += "%sprintf(\"  tensor2dSizeA== %%lu\\n\", tensor2dSizeA );\n" % (t)
        s += "%sprintf(\"  tensor2dSizeB== %%lu\\n\", tensor2dSizeB );\n" % (t)
        for idxChar in solution["PackedC0IdxChars"][:-1]:
          s += "%sprintf(\"  magicNumberSize%s== 0x%%lx, magicShiftSize%s== %%u)\\n\",  magicNumberSize%s, magicShiftSize%s);\n" \
              % (t, idxChar, idxChar, idxChar, idxChar)
        for idxChar in solution["PackedC1IdxChars"][:-1]:
          s += "%sprintf(\"  magicNumberSize%s== 0x%%x, magicShiftSize%s== %%u)\\n\",  magicNumberSize%s, magicShiftSize%s);\n" \
              % (t, idxChar, idxChar, idxChar, idxChar)
        s += "%sprintf(\"  magicNumberProblemNumGroupTiles0==%%u\\n\", magicNumberProblemNumGroupTiles0);\n" % t

      ########################################
      # OpenCL Runtime
      ########################################
      if self.language == "OCL":
        # set kernel args different for all enqueues
        argIdx = 6 if problemType["UseBeta"] else 5
        argIdx += len(self.strideList)
        # sizes
        for sizeIdx in range(0, problemType["TotalIndices"]):
          if sizeIdx in [ problemType["Index0"],  problemType["Index1"], problemType["IndexUnroll"] ]:
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
        if False: # gsu > 1:
          s += "%s1,\n" % (t)
          s += "%s&kernelEventBetaOnly,\n" % (t)
        else:
          s += "%snumInputEvents,\n" % (t)
          s += "%sinputEvents,\n" % (t)
        s += "%soutputEvent );\n" % (t)
        s += "%stensileStatusCheck(status);\n" % (t)
        t = t[:-2]
        s += "%s}\n" % (t)

      ########################################
      # HIP Runtime
      ########################################
      else:

        if not globalParameters["PreciseKernelTime"] or kernelLanguage == "Source":
          s += "%sif( inputEvents != NULL )\n" % (t)
          t += "  "
          s += "%shipEventRecord(inputEvents[enqueueIdx], stream );\n" % (t)
        s += "%stry {\n" % (t)
        t += "  "
        # hip kernel
        if kernelLanguage == "Source":
          s += "%skernelsLaunched++;\n" % (t)
          s += "%shipLaunchKernelGGL(\n" % (t)
          t += "  "
          s += "%sHIP_KERNEL_NAME(%s),\n" % (t, kernelName)
          s += "%sdim3(globalWorkSize[kernelIdx][0], globalWorkSize[kernelIdx][1], globalWorkSize[kernelIdx][2]),\n" % (t)
          s += "%sdim3(localWorkSize[0], localWorkSize[1], localWorkSize[2]),\n" % (t)
          s += "%s0, // groupMemBytes\n" % (t)
          s += "%sstream,\n" % (t)
          s += "%sdataD,\n" % (t)
          s += "%sdataC,\n" % (t)
          s += "%sdataA,\n" % (t)
          s += "%sdataB,\n" % (t)
          s += "%salpha,\n" % (t)
          s += "%s%sbeta,\n" % (t, \
              "" if problemType["UseBeta"] else "//")
          # strides
          for stride in self.strideList:
            s += "%s%s,\n" % (t, stride)
          # sizes
          for i in range(0, problemType["TotalIndices"]):
            lastParam = i == problemType["TotalIndices"]-1
            s += "%ssizes[kernelIdx][enqueueIdx][%u]%s\n" \
                % (t, i, "" if lastParam else "," )
          for idxChar in solution["PackedC0IdxChars"][:-1]:
            s += "%s,static_cast<uint32_t>(magicNumberSize%s)\n" % (t, idxChar)
            s += "%s,magicShiftSize%s\n" % (t, idxChar)
          for idxChar in solution["PackedC1IdxChars"][:-1]:
            s += "%s,static_cast<uint32_t>(magicNumberSize%s)\n" % (t, idxChar)
            s += "%s,magicShiftSize%s\n" % (t, idxChar)
          s += "%s,staggerUIter\n" % (t)
          #persistent:
          s += "%s,problemNumGroupTiles0\n" % (t)
          s += "%s,problemNumGroupTiles1\n" % (t)
          s += "%s,magicNumberProblemNumGroupTiles0\n" % (t) # magic number to use when dividing by problemNumGroupTiles0
          s += "%s);\n" % (t)
          t = t[:-2]
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
          s += "%shipFunctionArgs.tensor2dSizeC = tensor2dSizeC;\n" % (t)
          s += "%shipFunctionArgs.tensor2dSizeA = tensor2dSizeA;\n" % (t)
          s += "%shipFunctionArgs.tensor2dSizeB = tensor2dSizeB;\n" % (t)

          if solution["_GlobalAccumulation"]:
            s += "%shipFunctionArgs.dataD = workspace;\n" % (t)
            s += "%shipFunctionArgs.dataC = workspace;\n" % (t)
          else:
            s += "%shipFunctionArgs.dataD = dataD;\n" % (t)
            s += "%shipFunctionArgs.dataC = dataC;\n" % (t)
          s += "%shipFunctionArgs.dataA = dataA;\n" % (t)
          s += "%shipFunctionArgs.dataB = dataB;\n" % (t)

          if problemType["DataType"].isHalf():
            s += "%shipFunctionArgs.alpha[0] = alpha;\n" % (t)
            s += "%shipFunctionArgs.alpha[1] = alpha;\n" % (t)
          else:
            s += "%shipFunctionArgs.alpha = alpha;\n" % (t)
          if problemType["UseBeta"]:
            if problemType["DataType"].isHalf():
              s += "%shipFunctionArgs.beta[0] = beta;\n" % (t)
              s += "%shipFunctionArgs.beta[1] = beta;\n" % (t)
            else:
              s += "%shipFunctionArgs.beta = beta;\n" % (t)
          # strides
          for stride in self.strideList:
            s += "%shipFunctionArgs.%s = %s;\n" % (t, stride, stride)
          # sizes
          for i in range(0, problemType["TotalIndices"]):
            lastParam = i == problemType["TotalIndices"]-1
            s += "%shipFunctionArgs.size%s = sizes[kernelIdx][enqueueIdx][%u];\n" \
                % (t, globalParameters["IndexChars"][i], i )

          s += "%shipFunctionArgs.tensor2dSizeC = tensor2dSizeC;\n" % (t)
          s += "%shipFunctionArgs.tensor2dSizeA = tensor2dSizeA;\n" % (t)
          s += "%shipFunctionArgs.tensor2dSizeB = tensor2dSizeB;\n" % (t)

          s += "%shipFunctionArgs.staggerUIter = staggerUIter;\n" % (t)
          # persistent - pass in the number of tiles in problem since not available in WG
          s += "\n"
          s += "%shipFunctionArgs.problemNumGroupTiles0 = problemNumGroupTiles0;\n" % (t)
          s += "%shipFunctionArgs.problemNumGroupTiles1 = problemNumGroupTiles1;\n" % (t)
          s += "%shipFunctionArgs.magicNumberProblemNumGroupTiles0 = magicNumberProblemNumGroupTiles0;\n" % (t)
          s += "%shipFunctionArgs.gridNumWorkGroups0 = globalWorkSize[kernelIdx][0];\n" % (t) #
          s += "%shipFunctionArgs.numFullBlocks = numFullBlocks;\n" % (t)
          s += "%shipFunctionArgs.wgmRemainder1 = wgmRemainder1;\n" % (t)
          s += "%shipFunctionArgs.magicNumberWgmRemainder1 = magicNumberWgmRemainder1;\n" % (t)

          # Magic numbers for packed indices:
          for idxChar in solution["PackedC0IdxChars"][:-1]:
            s += "%shipFunctionArgs.magicNumberSize%s = static_cast<uint32_t>(magicNumberSize%s);\n" % (t, idxChar, idxChar)
            s += "%shipFunctionArgs.magicShiftSize%s = magicShiftSize%s;\n" % (t, idxChar, idxChar)
          for idxChar in solution["PackedC1IdxChars"][:-1]:
            s += "%shipFunctionArgs.magicNumberSize%s = static_cast<uint32_t>(magicNumberSize%s);\n" % (t, idxChar, idxChar)
            s += "%shipFunctionArgs.magicShiftSize%s = magicShiftSize%s;\n" % (t, idxChar, idxChar)
          if globalParameters["LibraryPrintDebug"]:
            s += """
            std::vector<char> tmp(hipFunctionArgsSize);
            memcpy(tmp.data(), &hipFunctionArgs, hipFunctionArgsSize);
            for(int i = 0; i < hipFunctionArgsSize; i++)
            {
                if(i % 8 == 0) printf("\\n");

                printf("%02hhx", tmp[i]);
            }
            printf("\\n");
            """

          s += "%skernelsLaunched++;\n" % (t)
          s += "%shipExtModuleLaunchKernel(\n" % (t)
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
            if gsu > 1:
              s += "%s,nullptr\n" %(t)
            else:
              s += "%s,inputEvents ? inputEvents[enqueueIdx]:nullptr\n" %(t)
            if solution["_GlobalAccumulation"]:
              s += "%s,nullptr\n" % (t)
            else:
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
        if not globalParameters["PreciseKernelTime"] or kernelLanguage == "Source":
          s += "%sif( outputEvent != NULL )\n" % (t)
          s += "%s  hipEventRecord(outputEvent[enqueueIdx], stream );\n" % (t)
        s += "  }\n"
        t = t[2:]


    ###################################################
    # Enqueue Kernel for Global Accumultation Buffer
    ###################################################
    if solution["_GlobalAccumulation"]:
      numStridesC = problemType["NumIndicesC"] - (0 if problemType["UseInitialStridesCD"] else 1)
      for ko in solution.getKernelConversionObjects():
        kernelName = ko.getKernelName()
        s += "%s// enqueue GSU third kernel\n" % (t)

        # grid sizes
        s += "%ssize_t localWorkSizeGlobalAccum[3] = { 8, 8, 1};\n" % (t)
        s += "%ssize_t globalWorkSizeGlobalAccum[3];\n" % (t)
        #s += "%sunsigned int sizeOfC0 = size%s;\n" % (t, \
        #    self.indexChars[problemType["Index0"]])
        #s += "%sunsigned int sizeOfC1 = size%s;\n" % (t, \
        #    self.indexChars[problemType["Index1"]])
        s += "%ssize_t totalWorkGroupsGlobalAccum0 = sizeOfC0 / localWorkSizeGlobalAccum[0];\n" % (t)
        s += "%ssize_t totalWorkGroupsGlobalAccum1 = sizeOfC1 / localWorkSizeGlobalAccum[1];\n" % (t)
        s += "%s// b/c single kernel, add extra work-group here if edge needed\n" % (t)
        s += "%sif (totalWorkGroupsGlobalAccum0*localWorkSizeGlobalAccum[0] < sizeOfC0) { totalWorkGroupsGlobalAccum0++; }\n" % (t)
        s += "%sif (totalWorkGroupsGlobalAccum1*localWorkSizeGlobalAccum[1] < sizeOfC1) { totalWorkGroupsGlobalAccum1++; }\n" % (t)
        s += "%sglobalWorkSizeGlobalAccum[0] = totalWorkGroupsGlobalAccum0%s;\n" % (t, "*localWorkSizeGlobalAccum[0]" if self.language == "OCL" else "")
        s += "%sglobalWorkSizeGlobalAccum[1] = totalWorkGroupsGlobalAccum1%s;\n" % (t, "*localWorkSizeGlobalAccum[1]" if self.language == "OCL" else "")
        s += "%sglobalWorkSizeGlobalAccum[2] = 1;\n" % (t)
        for i in range(0, problemType["NumIndicesC"]):
          if i != problemType["Index0"] and i != problemType["Index1"]:
            s += "%sglobalWorkSizeGlobalAccum[2] *= size%s;\n" % (t, self.indexChars[i])

        if self.language == "OCL":
          #s += "%sbool betaZero = true;\n" % (t)
          s += "%scl_kernel kernelGlobalAccum = kernel_%s;\n" \
              % (t, kernelName)
          argIdx = 0
          s += "%sstatus = clSetKernelArg( kernelGlobalAccum, %u, sizeof(cl_mem), &dataC ); tensileStatusCheck(status);\n" % (t, argIdx); argIdx+=1
          # strides
          for i in range(0,numStridesC):
            s += "%sstatus = clSetKernelArg( kernelGlobalAccum, %u, sizeof(unsigned int), &%s ); tensileStatusCheck(status);\n" % (t, argIdx, self.strideList[i]); argIdx+=1
          # sizes
          for i in range(0, problemType["NumIndicesC"]):
            s += "%sstatus = clSetKernelArg( kernelGlobalAccum, %u, sizeof(unsigned int), &size%s ); tensileStatusCheck(status);\n" % (t, argIdx, self.indexChars[i]); argIdx+=1
          # enqueue
          s += "%scl_event kernelEventGlobalAccum;\n" % (t)
          s += "%sstatus = clEnqueueNDRangeKernel(\n" % (t)
          t += "  "
          s += "%sstream,\n" % (t)
          s += "%skernelGlobalAccum,\n" % (t)
          s += "%sworkDim,\n" % (t)
          s += "%sNULL, // globalWorkOffset\n" % (t)
          s += "%sglobalWorkSizeGlobalAccum,\n" % (t)
          s += "%slocalWorkSizeGlobalAccum,\n" % (t)
          s += "%snumInputEvents,\n" % (t)
          s += "%sinputEvents,\n" % (t)
          #s += "%soutputEvent );\n" % (t)
          s += "%s&kernelEventGlobalAccum );\n" % (t)
          t = t[2:]
          s += "%stensileStatusCheck(status);\n" % (t)
          #s += "%sreturn tensileStatusSuccess;\n" % (t)
          s += "%sstatus = clFinish(stream);\n" % (t)
          s += "%stensileStatusCheck(status);\n" % (t)
          #s += " float tmp[128*128];\n"
          #s += "clEnqueueReadBuffer(stream, dataC, CL_TRUE, 0, 128*128*sizeof(float), tmp, 0, NULL, NULL);\n"
          #s += "for (unsigned int i = 0; i < 128*128; i++) { printf(\"%f\\n\", tmp[i]); }\n"
        else:
          s += "%stry {\n" % (t)
          t += "  "
          # TODO - timing with beta kernels is somewhat pessimistic since it has this separate event only on the GSU path.
          # Introduces 2-3us of overhead ; may want to disable PreciseKernelTime so non-GSU have same overhead.
          # Long-term fix would be to launch the beta kernel with the hipHccModule* API and set start-event in that call
          s += "%skernelsLaunched++;\n" % (t)
          s += "%shipLaunchKernelGGL(\n" % (t)
          t += "  "
          s += "%sHIP_KERNEL_NAME(%s),\n" % (t, kernelName)
          s += "%sdim3(globalWorkSizeGlobalAccum[0], globalWorkSizeGlobalAccum[1], globalWorkSizeGlobalAccum[2]),\n" % (t)
          s += "%sdim3(localWorkSizeGlobalAccum[0], localWorkSizeGlobalAccum[1], localWorkSizeGlobalAccum[2]),\n" % (t)
          s += "%s0, // groupMemBytes\n" % (t)
          s += "%sstream,\n" % (t)
          s += "%sdataD,\n" % (t)
          s += "%sworkspace,\n" % (t)
          # strides
          for i in range(0,numStridesC):
            s += "%s%s,\n" % (t, self.strideList[i])
          # sizes
          for i in range(0, problemType["NumIndicesC"]):
            s += "%ssize%s%s" % (t, self.indexChars[i], ",\n" if i < problemType["NumIndicesC"]-1 else ");\n")
          t = t[:-2]
          s += "%sif( outputEvent != NULL )\n" % (t)
          s += "%s  hipEventRecord(outputEvent[0], stream );\n" % (t)
          t = t[:-2]
          s += "%s} catch (const std::exception& e) {\n" % (t)
          s += "#ifdef DEBUG\n"
          s += "%s  std::cerr << e.what() << std::endl;\n" % (t)
          s += "#endif\n"
          s += "%s  return tensileStatusFailure;\n" % (t)
          s += "%s}\n" % (t)

    s += "\n"
    s += "  return tensileStatusSuccess;\n"
    s += "}\n"
    s += "\n"
    s += "/* Solution Parameters\n"
    s += Solution.getParametersIndented(solution.getAttributes(), "  ")
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
      for ko in solution.getHelperKernelObjects():
        kernelName = ko.getKernelName()
        s += "#include \"" + kernelName + ".h\"\n"

      s += "\n"

    # function declaration
    s += self.getSolutionSignature(solution, header=True) + ";\n"
    s += "\n"
    #s += "#endif\n"
    s += "\n"
    return s

  ########################################
  # get solution arguments
  # includeData adds launch-time info including data pointers and solution index
  def getArgList(self, problemType, includeSolutionInfo, includeData, includeEvents, includeStream, includeGlobalAccumBuffer=False, GlobalAccumKernel=False):
    self.strideList = []
    self.sizeList = []
    argList = []

    if includeSolutionInfo:
      argList.append(("SolutionLock *", "solutionLock", "nullptr"))
    #  argList.append(("const char *", "kernelName2"))
    #  argList.append(("const unsigned char *", "kernelCoba"))

    # data ptrs
    if includeData:
      typeName = problemType["DataType"].toCpp()
      destTypeName = problemType["DestDataType"].toCpp()
      computeTypeName = problemType["ComputeDataType"].toCpp()
      if self.language == "HIP":
        if GlobalAccumKernel:
          argList.append(("float *", "dataD", "nullptr"))
          argList.append(("const float *", "dataC", "nullptr"))
        else:
          argList.append(("%s *"%destTypeName, "dataD", "nullptr"))
          argList.append(("const %s *"%destTypeName, "dataC", "nullptr"))
        argList.append(("const %s *"%typeName, "dataA", "nullptr"))
        argList.append(("const %s *"%typeName, "dataB", "nullptr"))
      else:
        argList.append(("cl_mem", "dataD", "nullptr"))
        argList.append(("cl_mem", "dataC", "nullptr"))
        argList.append(("cl_mem", "dataA", "nullptr"))
        argList.append(("cl_mem", "dataB", "nullptr"))
      argList.append((computeTypeName, "alpha", "%s()"%computeTypeName))
      if problemType["UseBeta"]:
        argList.append((computeTypeName, "beta", "%s()"%computeTypeName))

    # initial strides ?
    firstStrideAB = firstStrideCD = 1
    if problemType["UseInitialStridesAB"]:
      firstStrideAB = 0
    if problemType["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = problemType["NumIndicesC"]
    lastStrideA = len(problemType["IndexAssignmentsA"])
    lastStrideB = len(problemType["IndexAssignmentsB"])
    # d strides
    for i in range(firstStrideCD,lastStrideC):
      self.strideList.append("strideD%u%s" % (i, self.indexChars[i]))
    # c strides
    for i in range(firstStrideCD,lastStrideC):
      self.strideList.append("strideC%u%s" % (i, self.indexChars[i]))
    # a strides
    for i in range(firstStrideAB,lastStrideA):
      self.strideList.append("strideA%u%s" % (i, \
          self.indexChars[problemType["IndexAssignmentsA"][i]]))
    # b strides
    for i in range(firstStrideAB,lastStrideB):
      self.strideList.append("strideB%u%s" % (i, \
          self.indexChars[problemType["IndexAssignmentsB"][i]]))
    # c sizes
    for i in range(0,problemType["TotalIndices"]):
      self.sizeList.append("size%s" % self.indexChars[i])
    for stride in self.strideList:
      argList.append(("unsigned int", stride, 0))
    for size in self.sizeList:
      argList.append(("unsigned int", size, 0))
    if includeStream:
      argList.append((self.streamName, "stream", "nullptr"))
    if includeEvents:
      argList.append(("unsigned int", "numInputEvents", 0))
      argList.append(("%s *"%self.eventName, "inputEvents", "nullptr"))
      argList.append(("%s *"%self.eventName, "outputEvent", "nullptr"))

    if includeData:
      if self.language == "HIP":
        if includeGlobalAccumBuffer:
          argList.append(("float *", "workspace", "nullptr"))

    return argList

  ########################################
  # get function signature
  def getSolutionSignature(self, solution, header=False):
    t = "" # indent
    s = ""
    solutionName = self.getSolutionName(solution)
    s += "%s%s %s(\n" % (t, self.statusName, solutionName)
    t += "    "
    argList = self.getArgList(solution["ProblemType"], True, True, True, True, True)
    for i in range(0, len(argList)):
      argString = "%s %s = %s" % argList[i] if header else "%s %s" % (argList[i][0], argList[i][1])
      s += "%s%s%s" % (t, argString, ",\n" if i < len(argList)-1 else ")" )
    return s

  ########################################
  # get full header code
  # called from BenchmarkProblems
  def getHeaderFileString(self, solution):
    fileStr = "" # CHeader
    fileStr += self.getHeaderString(solution)
    return fileStr


