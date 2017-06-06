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
from Common import globalParameters
import abc

################################################################################
# Make OpenCL Kernel String
################################################################################
class KernelWriter:
  __metaclass__=abc.ABCMeta

  ##############################################################################
  # Make OpenCL Kernel String
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    self.language = globalParameters["KernelLanguage"]
    self.kernelMinNaming = kernelMinNaming
    self.kernelSerialNaming = kernelSerialNaming


  ##############################################################################
  # Kernel Body
  ##############################################################################
  def kernelBody( self, kernel ):

    ########################################
    # determine index chars
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.indexChars[kernel["ProblemType"]["Index0"]]
    self.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.indexChars[kernel["ProblemType"]["Index1"]]
    self.unrollIdx = kernel["ProblemType"]["NumIndicesSummation"]-1
    self.unrollChar = \
        self.indexChars[kernel["ProblemType"]["IndicesSummation"][\
        self.unrollIdx]]
    self.tileChar0 = self.indexChars[kernel["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[kernel["ProblemType"]["Index1"]]
    self.tileCharA = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==0) \
        else self.tileChar1
    self.tileCharB = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==1) \
        else self.tileChar1

    ########################################
    # derrive global-read-coalesce-group from local in config
    """
    if kernel["ProblemType"]["TLUA"]:
      self.globalReadCoalesceGroupA = kernel["LocalWriteCoalesceGroupA"]
    else:
      self.globalReadCoalesceGroupA = not kernel["LocalWriteCoalesceGroupA"]
    if kernel["ProblemType"]["TLUB"]:
      self.globalReadCoalesceGroupB = kernel["LocalWriteCoalesceGroupB"]
    else:
      self.globalReadCoalesceGroupB = not kernel["LocalWriteCoalesceGroupB"]
    """
    self.globalReadCoalesceGroupA = kernel["GlobalReadCoalesceGroupA"]
    self.globalReadCoalesceGroupB = kernel["GlobalReadCoalesceGroupB"]

    ########################################
    # read / write vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"]: # NT no transpose
      self.numReadsTileA = kernel["NumLoadsCoalescedA"]
      self.numReadsUnrollA = kernel["NumLoadsPerpendicularA"]
      if kernel["GlobalReadCoalesceVectorA"]:
        self.readTileDimComponentsA = False # Vector
        self.readTileDimVectorA = True # Vector
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        self.writeTileDimComponentsA = False # Vector
        self.writeUnrollDimComponentsA = False # Scalar
      else:
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorA = False # Components
        self.writeTileDimComponentsA = False # Scalar
        self.writeUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
    else:
      self.numReadsTileA = kernel["NumLoadsPerpendicularA"]
      self.numReadsUnrollA = kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]:
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = False # Vector
        self.readUnrollDimVectorA = True # Vector
        self.writeTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.writeUnrollDimComponentsA = False # Scalar
      else:
        self.readTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readTileDimVectorA = False # Components
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        self.writeTileDimComponentsA = False # Vector
        self.writeUnrollDimComponentsA = False # Scalar
    self.numReadVectorComponentsA = kernel["VectorWidth"] \
        if (self.readTileDimComponentsA \
        or self.readUnrollDimComponentsA) else 1
    self.numWriteVectorComponentsA = kernel["VectorWidth"] \
        if (self.writeTileDimComponentsA \
        or self.writeUnrollDimComponentsA) else 1
    self.numReadTileVectorComponentsA = kernel["VectorWidth"] \
        if self.readTileDimComponentsA else 1 # for branches

    ####################################
    # read / write vectors or vector components b
    ####################################
    if kernel["ProblemType"]["TLUB"]: # NT no transpose
      self.numReadsTileB = kernel["NumLoadsCoalescedB"]
      self.numReadsUnrollB = kernel["NumLoadsPerpendicularB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Vector
        self.readTileDimVectorB = True # Vector
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        self.writeTileDimComponentsB = False # Vector
        self.writeUnrollDimComponentsB = False # Scalar
      else:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorB = False # Components
        self.writeTileDimComponentsB = False # Scalar
        self.writeUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
    else:
      self.numReadsTileB = kernel["NumLoadsPerpendicularB"]
      self.numReadsUnrollB = kernel["NumLoadsCoalescedB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = False # Vector
        self.readUnrollDimVectorB = True # Vector
        self.writeTileDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.writeUnrollDimComponentsB = False # Scalar
      else:
        self.readTileDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.readTileDimVectorB = False # Components
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        self.writeTileDimComponentsB = False # Vector
        self.writeUnrollDimComponentsB = False # Scalar
    self.numReadVectorComponentsB = kernel["VectorWidth"] \
        if (self.readTileDimComponentsB \
        or self.readUnrollDimComponentsB) else 1
    self.numWriteVectorComponentsB = kernel["VectorWidth"] \
        if (self.writeTileDimComponentsB \
        or self.writeUnrollDimComponentsB) else 1
    self.numReadTileVectorComponentsB = kernel["VectorWidth"] \
        if self.readTileDimComponentsB else 1 # for branches

    ####################################
    # Begin String
    kStr = ""
    kStr += self.openString(kernel)

    ####################################
    # Function Prefix
    kStr += self.comment3("Function Prefix")
    kStr += self.functionPrefix(kernel)

    ####################################
    # Function Signature
    ####################################
    kStr += self.comment3("Begin Kernel")
    kStr += self.functionSignaturePrefix(kernel)
    kStr += self.functionSignature(kernel)
    kStr += self.functionSignatureSuffix(kernel)
    kStr += self.functionBegin(kernel)

    kStr += self.comment3("Allocate Resources")
    kStr += self.allocateResources(kernel)

    ####################################
    # Global Read Addresses
    ####################################
    kStr += self.comment3("Global Read Addresses")

    # work-group assignments
    kStr += self.comment("global read addresses: work-group")
    kStr += self.graWorkGroup(kernel)

    # subgroup assignments
    kStr += self.comment("global read addresses: subgroup")
    kStr += self.graSubgroup(kernel)

    # tile assignments
    kStr += self.comment("global read addresses: tile offset assignment a")
    kStr += self.graTileAssignmentA(kernel)
    kStr += self.comment("global read addresses: tile offset assignment b")
    kStr += self.graTileAssignmentB(kernel)

    # unroll assignments
    kStr += self.comment("global read addresses: unroll assignment a")
    kStr += self.graUnrollAssignmentA(kernel)
    kStr += self.comment("global read addresses: unroll assignment b")
    kStr += self.graUnrollAssignmentB(kernel)

    # other free indices
    if kernel["ProblemType"]["NumIndicesC"] > 2:
      kStr += self.comment("global read addresses: other free assignments")
      kStr += self.graOtherFreeAssignments(kernel)

    # other summation indices
    if kernel["ProblemType"]["NumIndicesSummation"] > 1:
      kStr += self.comment("global read addresses: other summation assignments")
      kStr += self.graOtherSummationAssignments(kernel)

    # tile offsets
    kStr += self.comment("global read addresses: tile offsets a")
    kStr += self.graTileOffsetsA(kernel)
    kStr += self.comment("global read addresses: tile offsets a")
    kStr += self.graTileOffsetsB(kernel)

    # unroll offsets
    kStr += self.comment("global read addresses: unroll offsets a")
    kStr += self.graUnrollOffsetsA(kernel)
    kStr += self.comment("global read addresses: unroll offsets b")
    kStr += self.graUnrollOffsetsB(kernel)

    # tile edges
    if kernel["EdgeType"] == "ShiftPtr":
      kStr += self.comment("global read addresses: shift a")
      kStr += self.graShiftA(kernel)
      kStr += self.comment("global read addresses: shift b")
      kStr += self.graShiftB(kernel)
    elif kernel["EdgeType"] == "Branch":
      kStr += self.comment("global read addresses: branch a")
      kStr += self.graBranchA(kernel)
      kStr += self.comment("global read addresses: branch b")
      kStr += self.graBranchB(kernel)

    # final offsets
    kStr += self.comment("global read addresses: final offsets a")
    kStr += self.graFinalOffsetsA(kernel)
    kStr += self.comment("global read addresses: final offsets b")
    kStr += self.graFinalOffsetsB(kernel)

    # user offsets
    kStr += self.comment("global read addresses: apply user offsets")
    kStr += self.graApplyUserOffsets(kernel)

    # addresses
    kStr += self.comment("global read addresses: addresses a")
    kStr += self.graAddressesA(kernel)
    kStr += self.comment("global read addresses: addresses b")
    kStr += self.graAddressesB(kernel)

    # increments
    kStr += self.comment("global read addresses: increments a")
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]):
      kStr += self.graIncrementsA(kernel, i)
    kStr += self.comment("global read addresses: increments b")
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]):
      kStr += self.graIncrementsB(kernel, i)

    ####################################
    # Local Write Addresses
    ####################################
    kStr += self.comment3("Local Write Addresses")

    # tile assignments
    kStr += self.comment("local write addresses: tile assignment a")
    kStr += self.lwaTileAssignmentA(kernel)
    kStr += self.comment("local write addresses: tile assignment b")
    kStr += self.lwaTileAssignmentB(kernel)

    # unroll assignments
    kStr += self.comment("local write addresses: unroll assignment a")
    kStr += self.lwaUnrollAssignmentA(kernel)
    kStr += self.comment("local write addresses: unroll assignment b")
    kStr += self.lwaUnrollAssignmentB(kernel)

    # first offsets
    kStr += self.comment("local write addresses: first offset a")
    kStr += self.lwaFirstOffsetA(kernel)
    kStr += self.comment("local write addresses: first offset b")
    kStr += self.lwaFirstOffsetB(kernel)

    # final offsets
    kStr += self.comment("local write addresses: final offsets a")
    kStr += self.lwaFinalOffsetsA(kernel)
    kStr += self.comment("local write addresses: final offsets b")
    kStr += self.lwaFinalOffsetsB(kernel)

    # declare addresses
    kStr += self.comment("local write addresses: declare addresses a")
    kStr += self.lwaDeclareAddressesA(kernel)
    kStr += self.comment("local write addresses: declare addresses b")
    kStr += self.lwaDeclareAddressesB(kernel)

    # init pointers
    kStr += self.comment("local write addresses: init pointers a")
    kStr += self.localWriteInitPointersA(kernel)
    kStr += self.comment("local write addresses: init pointers b")
    kStr += self.localWriteInitPointersB(kernel)

    ####################################
    # Local Read Addresses
    ####################################
    kStr += self.comment3("Local Read Addresses")

    # tile assignments
    kStr += self.comment("local read addresses: tile assignments a")
    kStr += self.lraTileAssignmentA(kernel)
    kStr += self.comment("local read addresses: tile assignments b")
    kStr += self.lraTileAssignmentB(kernel)


    # final offsets
    kStr += self.comment("local read addresses: final offsets a")
    kStr += self.lraFinalOffsetA(kernel)
    kStr += self.comment("local read addresses: final offsets b")
    kStr += self.lraFinalOffsetB(kernel)

    # declare addresses
    kStr += self.comment("local read addresses: declare addresses a")
    kStr += self.lraDeclareAddressesA(kernel)
    kStr += self.comment("local read addresses: declare addresses b")
    kStr += self.lraDeclareAddressesB(kernel)

    # init pointers
    kStr += self.comment("local read addresses: init pointers a")
    kStr += self.localReadInitPointersA(kernel)
    kStr += self.comment("local read addresses: init pointers b")
    kStr += self.localReadInitPointersB(kernel)

    ###########################################################################
    # summations loops: open
    ###########################################################################

    # declare loop num iter
    kStr += self.comment("declare loop num iterations")
    kStr += self.declareLoopNumIter(kernel)

    # open non-unrolled summation loops
    kStr += self.calculateLoopNumIter(kernel, i)
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      kStr += self.comment("summation loop %u"%i)
      kStr += self.openLoop(kernel, i)

    ####################################
    # prefetch: unrolled loop prefix
    ####################################
    if kernel["PrefetchGlobalRead"]:
      kStr += self.comment("prefetch: global -> local")
      kStr += self.openSumAtLeastUnroll(kernel)
      # global read
      kStr += self.comment("global read a")
      kStr += self.globalReadDoA(kernel, False)
      kStr += self.comment("global read b")
      kStr += self.globalReadDoB(kernel, False)
      # increment global
      kStr += self.comment("global read inc a")
      kStr += self.globalReadIncrementA(kernel, self.unrollIdx)
      kStr += self.comment("global read inc b")
      kStr += self.globalReadIncrementB(kernel, self.unrollIdx)
      # local write
      kStr += self.comment("local write a")
      kStr += self.localWriteDoA(kernel)
      kStr += self.comment("local write b")
      kStr += self.localWriteDoB(kernel)
      kStr += self.indent + self.syncStr + self.endLine
      # swap local ptrs
      kStr += self.comment("local write swap a")
      kStr += self.localWriteSwapOffsetsA(kernel)
      kStr += self.comment("local write swap b")
      kStr += self.localWriteSwapOffsetsB(kernel)
      kStr += self.comment("local write init pointers a")
      kStr += self.localWriteInitPointersA(kernel)
      kStr += self.comment("local write init pointers b")
      kStr += self.localWriteInitPointersB(kernel)
      # prefetch-local
      if kernel["PrefetchLocalRead"]:
        kStr += self.comment("local read prefetch a")
        kStr += self.localReadDoA(kernel, False)
        kStr += self.comment("local read prefetch b")
        kStr += self.localReadDoB(kernel, False)
        kStr += self.comment("local read inc a")
        kStr += self.localReadIncA(kernel)
        kStr += self.comment("local read inc b")
        kStr += self.localReadIncB(kernel)
      kStr += self.closeSumAtLeastUnroll(kernel)

    # open unrolled summation loop
    kStr += self.comment("unrolled summation loop")
    kStr += self.openLoop(kernel, self.unrollIdx)

    # unrolled loop: global read A, B
    kStr += self.comment("global read a")
    kStr += self.globalReadDoA(kernel, False)
    kStr += self.comment("global read b")
    kStr += self.globalReadDoB(kernel, False)

    # unrolled loop: increment global read addresses
    kStr += self.comment("global read inc a")
    kStr += self.globalReadIncrementA(kernel, self.unrollIdx)
    kStr += self.comment("global read inc b")
    kStr += self.globalReadIncrementB(kernel, self.unrollIdx)

    # if not prefetch global, localWrite before mac's
    if not kernel["PrefetchGlobalRead"]:
      # unrolled loop: local write A, B
      kStr += self.indent + self.syncStr + self.endLine
      kStr += self.comment("local write a")
      kStr += self.localWriteDoA(kernel)
      kStr += self.comment("local write b")
      kStr += self.localWriteDoB(kernel)
      kStr += self.indent + self.syncStr + self.endLine
      # debug Local state
      """
      kStr += "    /* print Local state */" + self.endLine
      kStr += "    for (unsigned int i = serial; i < LDS_NUM_ELEMENTS; i+=NUM_THREADS) {%s" % self.endLine
      kStr += "      printf(\\\"localMemory[%%06u] = %%.0f\\\\n\\\", i, localMemory[i]);%s" \
          % self.endLine
      kStr += "    }" + self.endLine
      """

    # unrolled loop: prefetch local
    if kernel["PrefetchLocalRead"] and not kernel["PrefetchGlobalRead"]:
      kStr += self.comment("prefetch local a")
      kStr += self.localReadDoA(kernel, False)
      kStr += self.comment("prefetch local b")
      kStr += self.localReadDoB(kernel, False)
      kStr += self.comment("local read increment a")
      kStr += self.localReadIncA(kernel)
      kStr += self.comment("local read increment b")
      kStr += self.localReadIncB(kernel)

    kStr += self.closeString(kernel)
    kStr += self.openString(kernel)

    ####################################
    # unrolled loop: mac iterations
    ####################################
    for u in range(0, kernel["LoopUnroll"]-2):
     # local read
      kStr += self.comment("iter %u"%u)
      readBlk = kernel["PrefetchLocalRead"] and u%2==0
      kStr += self.comment("local read a")
      kStr += self.localReadDoA(kernel, readBlk)
      kStr += self.comment("local read b")
      kStr += self.localReadDoB(kernel, readBlk)
      kStr += self.comment("local read increment a")
      kStr += self.localReadIncA(kernel)
      kStr += self.comment("local read increment b")
      kStr += self.localReadIncB(kernel)
      kStr += self.macIter(kernel, (kernel["PrefetchLocalRead"] and u%2==1) )

    kStr += self.closeString(kernel)
    kStr += self.openString(kernel)

    ####################################
    # unrolled loop: 2nd-to-last summation iter
    ####################################
    # if prefetch-local: read for last unroll,
    # local write, readSwap/Init, writeSwapInit
    # if no prefetch-local: read for current unroll of current iter
    unrollIter = kernel["LoopUnroll"]-2
    kStr += self.comment("iter %u"%unrollIter)
    if kernel["PrefetchLocalRead"] and kernel["PrefetchGlobalRead"]:
      # local read for last unroll
      kStr += self.comment("local read a")
      kStr += self.localReadDoA(kernel, True)
      kStr += self.comment("local read b")
      kStr += self.localReadDoB(kernel, True)
      # local write for next iter
      kStr += self.comment("local write a")
      kStr += self.localWriteDoA(kernel)
      kStr += self.comment("local write b")
      kStr += self.localWriteDoB(kernel)
      kStr += self.indent + self.syncStr + self.endLine
      # swap read and write pointers
      kStr += self.comment("local read swap offsets a")
      kStr += self.localReadSwapOffsetsA(kernel)
      kStr += self.comment("local read swap offsets b")
      kStr += self.localReadSwapOffsetsB(kernel)
      kStr += self.comment("local read init pointers a")
      kStr += self.localReadInitPointersA(kernel)
      kStr += self.comment("local read init pointers b")
      kStr += self.localReadInitPointersB(kernel)
      kStr += self.comment("local read swap offsets a")
      kStr += self.localWriteSwapOffsetsA(kernel)
      kStr += self.comment("local read swap offsets b")
      kStr += self.localWriteSwapOffsetsB(kernel)
      kStr += self.comment("local read init pointers a")
      kStr += self.localWriteInitPointersA(kernel)
      kStr += self.comment("local read init pointers b")
      kStr += self.localWriteInitPointersB(kernel)
    else:
      # local read
      readBlk = kernel["PrefetchLocalRead"] and unrollIter%2==0
      kStr += self.comment("local read a")
      kStr += self.localReadDoA(kernel, readBlk)
      kStr += self.comment("local read b")
      kStr += self.localReadDoB(kernel, readBlk)
      if kernel["PrefetchLocalRead"]:
        # local read init ptrs
        kStr += self.comment("local read init pointers a")
        kStr += self.localReadInitPointersA(kernel)
        kStr += self.comment("local read init pointers b")
        kStr += self.localReadInitPointersB(kernel)
      else:
        # local read inc
        kStr += self.comment("local read inc a")
        kStr += self.localReadIncA(kernel)
        kStr += self.comment("local read inc b")
        kStr += self.localReadIncB(kernel)
    kStr += self.macIter(kernel, False)

    ####################################
    # unrolled loop: last summation iter
    ####################################
    # if prefetch-local: read red for 1st unroll of next iter
    # if not prefetch-local: read for current unroll of current iter
    unrollIter = kernel["LoopUnroll"]-1
    kStr += self.comment("iter %u"%unrollIter)
    if not kernel["PrefetchLocalRead"] or kernel["PrefetchGlobalRead"]:
      # local read
      readBlk = kernel["PrefetchLocalRead"] and unrollIter%2==0
      kStr += self.comment("local read a")
      kStr += self.localReadDoA(kernel, readBlk)
      kStr += self.comment("local read b")
      kStr += self.localReadDoB(kernel, readBlk)
    if kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"]:
      # local read inc
      kStr += self.comment("local read inc a")
      kStr += self.localReadIncA(kernel)
      kStr += self.comment("local read inc b")
      kStr += self.localReadIncB(kernel)
    elif kernel["PrefetchGlobalRead"]:
      # local write
      kStr += self.comment("local write a")
      kStr += self.localWriteDoA(kernel)
      kStr += self.comment("local write b")
      kStr += self.localWriteDoB(kernel)
      kStr += self.indent + self.syncStr + self.endLine
      # swap read and write
      kStr += self.comment("local read swap offsets a")
      kStr += self.localReadSwapOffsetsA(kernel)
      kStr += self.comment("local read swap offsets b")
      kStr += self.localReadSwapOffsetsB(kernel)
      kStr += self.comment("local read init pointers a")
      kStr += self.localReadInitPointersA(kernel)
      kStr += self.comment("local read init pointers b")
      kStr += self.localReadInitPointersB(kernel)
      kStr += self.comment("local write swap offsets a")
      kStr += self.localWriteSwapOffsetsA(kernel)
      kStr += self.comment("local write swap offsets b")
      kStr += self.localWriteSwapOffsetsB(kernel)
      kStr += self.comment("local write init pointers a")
      kStr += self.localWriteInitPointersA(kernel)
      kStr += self.comment("local write init pointers b")
      kStr += self.localWriteInitPointersB(kernel)
    elif not kernel["PrefetchGlobalRead"] and not kernel["PrefetchLocalRead"]:
      # local read init ptrs
      kStr += self.comment("local read init pointers a")
      kStr += self.localReadInitPointersA(kernel)
      kStr += self.comment("local read init pointers b")
      kStr += self.localReadInitPointersB(kernel)
    kStr += self.macIter(kernel, kernel["PrefetchLocalRead"])

    # close unrolled loop
    kStr += self.closeLoop(kernel, self.unrollIdx)

    # prefetch: unrolled loop suffix
    if kernel["PrefetchGlobalRead"]:
      kStr += self.comment("prefetch: last unrolled iteration")
      kStr += self.openSumAtLeastUnroll(kernel)
      if kernel["PrefetchLocalRead"] and not kernel["PrefetchGlobalRead"]:
        kStr += self.comment("prefetch local read a")
        kStr += self.localReadDoA(kernel, False)
        kStr += self.comment("prefetch local read b")
        kStr += self.localReadDoB(kernel, False)
        kStr += self.comment("prefetch local read inc a")
        kStr += self.localReadIncA(kernel)
        kStr += self.comment("prefetch local read inc b")
        kStr += self.localReadIncB(kernel)
      for u in range(0, kernel["LoopUnroll"]):
        kStr += self.comment("iter %u"%u)
        readBlk = kernel["PrefetchLocalRead"] and u%2==0
        if u < kernel["LoopUnroll"]-1 or not kernel["PrefetchLocalRead"]:
          kStr += self.comment("local read a")
          kStr += self.localReadDoA(kernel, readBlk)
          kStr += self.comment("local read b")
          kStr += self.localReadDoB(kernel, readBlk)
          kStr += self.comment("local read inc a")
          kStr += self.localReadIncA(kernel)
          kStr += self.comment("local read inc b")
          kStr += self.localReadIncB(kernel)
        kStr += self.macIter(kernel, (kernel["PrefetchLocalRead"] and u%2==1) )
      kStr += self.closeSumAtLeastUnroll(kernel)

    ########################################
    # Tail Loop
    ########################################
    if kernel["LoopTail"]:
      kStr += self.comment3("Tail Loop")

      # tail: global read
      kStr += self.calculateLoopNumIter(kernel, -1)
      kStr += self.comment("global read a")
      kStr += self.globalReadDoA(kernel, True)
      kStr += self.comment("global read b")
      kStr += self.globalReadDoB(kernel, True)

      # tail: local write
      if kernel["PrefetchGlobalRead"]:
        kStr += self.comment("local write reset offsets a")
        kStr += self.localWriteResetOffsetsA(kernel)
        kStr += self.comment("local write reset offsets b")
        kStr += self.localWriteResetOffsetsB(kernel)
      kStr += self.comment("local write init pointers a")
      kStr += self.localWriteInitPointersA(kernel)
      kStr += self.comment("local write init pointers b")
      kStr += self.localWriteInitPointersB(kernel)
      kStr += self.indent + self.syncStr + self.endLine
      kStr += self.comment("local write a")
      kStr += self.localWriteDoA(kernel)
      kStr += self.comment("local write b")
      kStr += self.localWriteDoB(kernel)
      kStr += self.indent + self.syncStr + self.endLine

      # tail: re-init local read addresses
      if kernel["PrefetchGlobalRead"]:
        kStr += self.comment("local read reset offsets a")
        kStr += self.localReadResetOffsetsA(kernel)
        kStr += self.comment("local read reset offsets b")
        kStr += self.localReadResetOffsetsB(kernel)
        kStr += self.comment("local read init pointers a")
        kStr += self.localReadInitPointersA(kernel)
        kStr += self.comment("local read init pointers b")
        kStr += self.localReadInitPointersB(kernel)

      # tail: macs
      kStr += self.comment("tail loop: macs")
      kStr += self.openLoop(kernel, -1)
      kStr += self.comment("local read a")
      kStr += self.localReadDoA(kernel, False)
      kStr += self.comment("local read b")
      kStr += self.localReadDoB(kernel, False)
      kStr += self.comment("local read inc a")
      kStr += self.localReadIncA(kernel)
      kStr += self.comment("local read inc b")
      kStr += self.localReadIncB(kernel)
      kStr += self.macIter(kernel, False )

      # tail: close
      kStr += self.closeLoop(kernel, self.unrollIdx)

    # extra summation loops: global increment and close
    for i in reversed(range(0,kernel["ProblemType"]["NumIndicesSummation"]-1)):
      kStr += self.comment("global read inc a")
      kStr += self.globalReadIncrementA(kernel, i)
      kStr += self.comment("global read inc b")
      kStr += self.globalReadIncrementB(kernel, i)
      kStr += self.closeLoop(kernel, self.unrollIdx)

    ####################################
    # Shift Vector Components
    ####################################
    if kernel["EdgeType"] == "ShiftPtr" and kernel["VectorWidth"] > 1:
      # shift vector components d0
      if self.readTileDimVectorA:
        kStr += self.comment("shift vector components d0")
        kStr += self.shiftVectorComponents0(kernel)
      # shift vector components d1
      if self.readTileDimVectorB:
        kStr += self.comment("shift vector components d1")
        kStr += self.shiftVectorComponents1(kernel)

    # complex declare tmp registers
    kStr += self.complexDeclareTmpRegisters(kernel)


    ####################################
    # LocalSplitU reduction
    ####################################
    #if kernel["NumThreads"]%kernel["MacroTile0"] == 0:
    if kernel["LocalSplitU"] > 1:
      kStr += self.comment3("LocalSplitU Reduction")
      kStr += self.indent + self.syncStr + self.endLine

      # LocalSplitU: local write
      kStr += self.comment("LocalSplitU: local write")
      kStr += self.localSplitULocalWrite(kernel)

      # LocalSplitU: local read
      kStr += self.comment("LocalSplitU: local read")
      kStr += self.localSplitULocalRead(kernel)

      # LocalSplitU: local read
      kStr += self.comment("LocalSplitU: reduction")
      kStr += self.localSplitUReduction(kernel)

      # LocalSplitU: global write indices
      kStr += self.comment("LocalSplitU: global write indices")
      kStr += self.localSplitUGlobalWriteIndices(kernel)

      # LocalSplitU: global write
      kStr += self.comment("LocalSplitU: global write")
      kStr += self.localSplitUGlobalWrite(kernel)


    else:
      ####################################
      # NOT LocalSplitU
      ####################################

      # global write indices
      kStr += self.comment("not-LocalSplitU: global write indices")
      kStr += self.notLocalSplitUGlobalWriteIndices(kernel)

      # global write
      kStr += self.comment("not-LocalSplitU: global write")
      kStr += self.notLocalSplitUGlobalWrite(kernel)

    # function suffix
    kStr += self.functionEnd(kernel)
    kStr += self.functionSuffix(kernel)

    kStr += self.closeString(kernel)

    return kStr



  ##############################################################################
  #
  #   Functions to Write Kernel Segments
  #
  ##############################################################################

  ##############################################################################
  # single line comment
  ##############################################################################
  def comment1(self, text):
    s = ""
    s += self.indent
    s += self.commentPrefix
    s += " %s " % text
    s += self.commentSuffix
    s += self.endLine
    return s

  ##############################################################################
  # comment with prior newline
  ##############################################################################
  def comment(self, text):
    s = ""
    s += self.endLine
    s += self.comment1(text)
    return s

  ##############################################################################
  # 3-line comment
  ##############################################################################
  def comment3(self, text):
    s = ""
    s += self.endLine
    s += self.indent
    s += self.commentPrefix
    s += self.commentHR
    s += self.commentSuffix
    s += self.endLine

    s += self.indent
    s += self.commentPrefix
    s += " %-38s " % text
    s += self.commentSuffix
    s += self.endLine

    s += self.indent
    s += self.commentPrefix
    s += self.commentHR
    s += self.commentSuffix
    s += self.endLine
    return s

  ##############################################################################
  # Init Kernel
  ##############################################################################
  @abc.abstractmethod
  def initKernel(self, kernel):
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.indexChars[kernel["ProblemType"]["Index0"]]
    self.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.indexChars[kernel["ProblemType"]["Index1"]]
    self.unrollIdx = kernel["ProblemType"]["NumIndicesSummation"]-1
    self.unrollChar = \
        self.indexChars[kernel["ProblemType"]["IndicesSummation"][\
        self.unrollIdx]]
    self.tileChar0 = self.indexChars[kernel["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[kernel["ProblemType"]["Index1"]]
    self.tileCharA = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==0) \
        else self.tileChar1
    self.tileCharB = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==1) \
        else self.tileChar1

    if kernel["ProblemType"]["Tensor0"]==0:
      kernel["ThreadTileA"] = kernel["ThreadTile0"]
      kernel["ThreadTileB"] = kernel["ThreadTile1"]
      kernel["SubGroupA"] = kernel["SubGroup0"]
      kernel["SubGroupB"] = kernel["SubGroup1"]
      kernel["MacroTileA"] = kernel["MacroTile0"]
      kernel["MacroTileB"] = kernel["MacroTile1"]
    else:
      kernel["ThreadTileB"] = kernel["ThreadTile0"]
      kernel["ThreadTileA"] = kernel["ThreadTile1"]
      kernel["SubGroupB"] = kernel["SubGroup0"]
      kernel["SubGroupA"] = kernel["SubGroup1"]
      kernel["MacroTileB"] = kernel["MacroTile0"]
      kernel["MacroTileA"] = kernel["MacroTile1"]

    ########################################
    # derrive global-read-coalesce-group from local in config
    """
    if kernel["ProblemType"]["TLUA"]:
      self.globalReadCoalesceGroupA = kernel["LocalWriteCoalesceGroupA"]
    else:
      self.globalReadCoalesceGroupA = not kernel["LocalWriteCoalesceGroupA"]

    if kernel["ProblemType"]["TLUB"]:
      self.globalReadCoalesceGroupB = kernel["LocalWriteCoalesceGroupB"]
    else:
      self.globalReadCoalesceGroupB = not kernel["LocalWriteCoalesceGroupB"]
    """
    self.globalReadCoalesceGroupA = kernel["GlobalReadCoalesceGroupA"]
    self.globalReadCoalesceGroupB = kernel["GlobalReadCoalesceGroupB"]

    ########################################
    # read / write vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"]: # NT no transpose
      self.numReadsTileA = kernel["NumLoadsCoalescedA"]
      self.numReadsUnrollA = kernel["NumLoadsPerpendicularA"]
      self.numWritesCoalescedA = kernel["NumLoadsCoalescedA"]
      self.numWritesPerpendicularA = kernel["NumLoadsPerpendicularA"]
      if kernel["GlobalReadCoalesceVectorA"]:
        self.readTileDimComponentsA = False # Vector
        self.readTileDimVectorA = True # Vector
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        self.writeTileDimComponentsA = False # Vector
        self.writeUnrollDimComponentsA = False # Scalar
      else:
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorA = False # Components
        self.writeTileDimComponentsA = False # Scalar
        self.writeUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
    else:
      self.numReadsTileA = kernel["NumLoadsPerpendicularA"]
      self.numReadsUnrollA = kernel["NumLoadsCoalescedA"]
      self.numWritesCoalescedA = kernel["NumLoadsPerpendicularA"]
      self.numWritesPerpendicularA = kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]:
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = False # Vector
        self.readUnrollDimVectorA = True # Vector
        self.writeTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.writeUnrollDimComponentsA = False # Scalar
      else:
        self.readTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readTileDimVectorA = False # Components
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        self.writeTileDimComponentsA = False # Vector
        self.writeUnrollDimComponentsA = False # Scalar
    self.numReadVectorComponentsA = kernel["VectorWidth"] \
        if (self.readTileDimComponentsA \
        or self.readUnrollDimComponentsA) else 1
    self.numWriteVectorComponentsA = kernel["VectorWidth"] \
        if (self.writeTileDimComponentsA \
        or self.writeUnrollDimComponentsA) else 1
    self.numReadTileVectorComponentsA = kernel["VectorWidth"] \
        if self.readTileDimComponentsA else 1 # for branches
    # convert tile/unroll to para/perp
    if kernel["ProblemType"]["TLUA"]:
      self.readCoalescedComponentsA  = self.readTileDimComponentsA
      self.readCoalescedVectorA      = self.readTileDimVectorA
      self.readPerpendicularComponentsA  = self.readUnrollDimComponentsA
      self.readPerpendicularVectorA      = self.readUnrollDimVectorA
    else:
      self.readCoalescedComponentsA  = self.readUnrollDimComponentsA
      self.readCoalescedVectorA      = self.readUnrollDimVectorA
      self.readPerpendicularComponentsA  = self.readTileDimComponentsA
      self.readPerpendicularVectorA      = self.readTileDimVectorA

    ####################################
    # read / write vectors or vector components b
    ####################################
    if kernel["ProblemType"]["TLUB"]: # NT no transpose
      self.numReadsTileB = kernel["NumLoadsCoalescedB"]
      self.numReadsUnrollB = kernel["NumLoadsPerpendicularB"]
      self.numWritesCoalescedB = kernel["NumLoadsCoalescedB"]
      self.numWritesPerpendicularB = kernel["NumLoadsPerpendicularB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Vector
        self.readTileDimVectorB = True # Vector
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        self.writeTileDimComponentsB = False # Vector
        self.writeUnrollDimComponentsB = False # Scalar
      else:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorB = False # Components
        self.writeTileDimComponentsB = False # Scalar
        self.writeUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
    else:
      self.numReadsTileB = kernel["NumLoadsPerpendicularB"]
      self.numReadsUnrollB = kernel["NumLoadsCoalescedB"]
      self.numWritesCoalescedB = kernel["NumLoadsPerpendicularB"]
      self.numWritesPerpendicularB = kernel["NumLoadsCoalescedB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = False # Vector
        self.readUnrollDimVectorB = True # Vector
        self.writeTileDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.writeUnrollDimComponentsB = False # Scalar
      else:
        self.readTileDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.readTileDimVectorB = False # Components
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        self.writeTileDimComponentsB = False # Vector
        self.writeUnrollDimComponentsB = False # Scalar
    self.numReadVectorComponentsB = kernel["VectorWidth"] \
        if (self.readTileDimComponentsB \
        or self.readUnrollDimComponentsB) else 1
    self.numWriteVectorComponentsB = kernel["VectorWidth"] \
        if (self.writeTileDimComponentsB \
        or self.writeUnrollDimComponentsB) else 1
    self.numReadTileVectorComponentsB = kernel["VectorWidth"] \
        if self.readTileDimComponentsB else 1 # for branches
    # convert tile/unroll to para/perp
    if kernel["ProblemType"]["TLUB"]:
      self.readCoalescedComponentsB  = self.readTileDimComponentsB
      self.readCoalescedVectorB      = self.readTileDimVectorB
      self.readPerpendicularComponentsB  = self.readUnrollDimComponentsB
      self.readPerpendicularVectorB      = self.readUnrollDimVectorB
    else:
      self.readCoalescedComponentsB  = self.readUnrollDimComponentsB
      self.readCoalescedVectorB      = self.readUnrollDimVectorB
      self.readPerpendicularComponentsB  = self.readTileDimComponentsB
      self.readPerpendicularVectorB      = self.readTileDimVectorB

    ####################################
    # load sizes
    if kernel["ProblemType"]["TLUA"]:
      kernel["LSCA"] = kernel["MacroTileA"] \
          / kernel["NumLoadsCoalescedA"]
      kernel["LSPA"] = kernel["DepthU"] / kernel["NumLoadsPerpendicularA"]
    else:
      kernel["LSCA"] = kernel["DepthU"] / kernel["NumLoadsCoalescedA"]
      kernel["LSPA"] = kernel["MacroTileA"] \
          / kernel["NumLoadsPerpendicularA"]

    if kernel["ProblemType"]["TLUB"]:
      kernel["LSCB"] = kernel["MacroTileB"] \
          / kernel["NumLoadsCoalescedB"]
      kernel["LSPB"] = kernel["DepthU"] / kernel["NumLoadsPerpendicularB"]
    else:
      kernel["LSCB"] = kernel["DepthU"] / kernel["NumLoadsCoalescedB"]
      kernel["LSPB"] = kernel["MacroTileB"] \
          / kernel["NumLoadsPerpendicularB"]

    kernel["LVCA"] = kernel["LSCA"] / kernel["VectorWidth"]
    kernel["LVCB"] = kernel["LSCB"] / kernel["VectorWidth"]
    kernel["LVPA"] = kernel["LSPA"] / kernel["VectorWidth"]
    kernel["LVPB"] = kernel["LSPB"] / kernel["VectorWidth"]



  ##############################################################################
  # Open String
  ##############################################################################
  @abc.abstractmethod
  def openString(self, kernel):
    return ""

  ##############################################################################
  # Close String
  ##############################################################################
  @abc.abstractmethod
  def closeString(self, kernel):
    return ""

  ##############################################################################
  # Function Prefix
  ##############################################################################
  @abc.abstractmethod
  def functionPrefix(self, kernel):
    return ""

  ##############################################################################
  # Function Signature Prefix
  ##############################################################################
  @abc.abstractmethod
  def functionSignaturePrefix(self, kernel):
    return ""

  ##############################################################################
  # Function Signature
  ##############################################################################
  @abc.abstractmethod
  def functionSignature(self, kernel ):
    return ""

  ##############################################################################
  # Function Signature Suffix
  ##############################################################################
  @abc.abstractmethod
  def functionSignatureSuffix(self, kernel):
    return ""

  ##############################################################################
  # Function Begin
  ##############################################################################
  @abc.abstractmethod
  def functionBegin(self, kernel):
    return ""

  ##############################################################################
  # Allocate Resources
  ##############################################################################
  @abc.abstractmethod
  def allocateResources(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  @abc.abstractmethod
  def graWorkGroup(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Subgroup
  ##############################################################################
  @abc.abstractmethod
  def graSubgroup(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Tile Assignment A
  ##############################################################################
  @abc.abstractmethod
  def graTileAssignmentA(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Tile Assignment B
  ##############################################################################
  @abc.abstractmethod
  def graTileAssignmentB(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A
  ##############################################################################
  @abc.abstractmethod
  def graUnrollAssignmentA(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Assignment B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollAssignmentB(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherFreeAssignments(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherSummationAssignments(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Tile Offsets A
  ##############################################################################
  @abc.abstractmethod
  def graTileOffsetsA(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Tile Offsets B
  ##############################################################################
  @abc.abstractmethod
  def graTileOffsetsB(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A
  ##############################################################################
  @abc.abstractmethod
  def graUnrollOffsetsA(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Offsets B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollOffsetsB(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Branch A
  ##############################################################################
  @abc.abstractmethod
  def graBranchA(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Branch B
  ##############################################################################
  @abc.abstractmethod
  def graBranchB(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A
  ##############################################################################
  @abc.abstractmethod
  def graShiftA(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift B
  ##############################################################################
  @abc.abstractmethod
  def graShiftB(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Final Offsets A
  ##############################################################################
  @abc.abstractmethod
  def graFinalOffsetsA(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Final Offsets B
  ##############################################################################
  @abc.abstractmethod
  def graFinalOffsetsB(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Apply User Offsets
  ##############################################################################
  @abc.abstractmethod
  def graApplyUserOffsets(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Addresses A
  ##############################################################################
  @abc.abstractmethod
  def graAddressesA(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Addresses B
  ##############################################################################
  @abc.abstractmethod
  def graAddressesB(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Increments A
  ##############################################################################
  @abc.abstractmethod
  def graIncrementsA(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Global Read Addresses: Increments B
  ##############################################################################
  @abc.abstractmethod
  def graIncrementsB(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Local Write Addresses: Tile Assignment A
  ##############################################################################
  @abc.abstractmethod
  def lwaTileAssignmentA(self, kernel):
    return ""

  ##############################################################################
  # Local Write Addresses: Tile Assignment B
  ##############################################################################
  @abc.abstractmethod
  def lwaTileAssignmentB(self, kernel):
    return ""

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A
  ##############################################################################
  @abc.abstractmethod
  def lwaUnrollAssignmentA(self, kernel):
    return ""

  ##############################################################################
  # Local Write Addresses: Unroll Assignment B
  ##############################################################################
  @abc.abstractmethod
  def lwaUnrollAssignmentB(self, kernel):
    return ""

  ##############################################################################
  # Local Write Addresses: First Offset A
  ##############################################################################
  @abc.abstractmethod
  def lwaFirstOffsetA(self, kernel):
    return ""

  ##############################################################################
  # Local Write Addresses: First Offset B
  ##############################################################################
  @abc.abstractmethod
  def lwaFirstOffsetB(self, kernel):
    return ""

  ##############################################################################
  # Local Write Addresses: Final Offsets A
  ##############################################################################
  @abc.abstractmethod
  def lwaFinalOffsetsA(self, kernel):
    return ""

  ##############################################################################
  # Local Write Addresses: Final Offsets B
  ##############################################################################
  @abc.abstractmethod
  def lwaFinalOffsetsB(self, kernel):
    return ""

  ##############################################################################
  # Local Write Addresses: Declare Addresses A
  ##############################################################################
  @abc.abstractmethod
  def lwaDeclareAddressesA(self, kernel):
    return ""

  ##############################################################################
  # Local Write Addresses: Declare Addresses B
  ##############################################################################
  @abc.abstractmethod
  def lwaDeclareAddressesB(self, kernel):
    return ""

  ##############################################################################
  # Local Read Addresses: Tile Assignment A
  ##############################################################################
  @abc.abstractmethod
  def lraTileAssignmentA(self, kernel):
    return ""

  ##############################################################################
  # Local Read Addresses: Tile Assignment B
  ##############################################################################
  @abc.abstractmethod
  def lraTileAssignmentB(self, kernel):
    return ""

  ##############################################################################
  # Local Read Addresses: Final Offset A
  ##############################################################################
  @abc.abstractmethod
  def lraFinalOffsetA(self, kernel):
    return ""

  ##############################################################################
  # Local Read Addresses: Final Offset B
  ##############################################################################
  @abc.abstractmethod
  def lraFinalOffsetB(self, kernel):
    return ""

  ##############################################################################
  # Local Read Addresses: Declare Addresses A
  ##############################################################################
  @abc.abstractmethod
  def lraDeclareAddressesA(self, kernel):
    return ""

  ##############################################################################
  # Local Read Addresses: Declare Addresses B
  ##############################################################################
  @abc.abstractmethod
  def lraDeclareAddressesB(self, kernel):
    return ""

  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  @abc.abstractmethod
  def declareLoopNumIter(self, kernel):
    return ""

  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  @abc.abstractmethod
  def calculateLoopNumIter(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Open Loop
  ##############################################################################
  @abc.abstractmethod
  def openLoop(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Close Loop
  ##############################################################################
  @abc.abstractmethod
  def closeLoop(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  @abc.abstractmethod
  def macIter(self, kernel, black):
    return ""

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  @abc.abstractmethod
  def openSumAtLeastUnroll(self, kernel):
    return ""

  @abc.abstractmethod
  def closeSumAtLeastUnroll(self, kernel):
    return ""

  ##############################################################################
  # Global Read: Increment A
  ##############################################################################
  @abc.abstractmethod
  def globalReadIncrementA(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Global Read: Increment B
  ##############################################################################
  @abc.abstractmethod
  def globalReadIncrementB(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Global Read: Do It A
  ##############################################################################
  @abc.abstractmethod
  def globalReadDoA(self, kernel, guardK):
    return ""

  ##############################################################################
  # Global Gead: Do It B
  ##############################################################################
  @abc.abstractmethod
  def globalReadDoB(self, kernel, guardK):
    return ""

  ##############################################################################
  # Local Write: Swap Offsets A
  ##############################################################################
  @abc.abstractmethod
  def localWriteSwapOffsetsA(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Swap Offsets B
  ##############################################################################
  @abc.abstractmethod
  def localWriteSwapOffsetsB(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Reset Offsets A
  ##############################################################################
  @abc.abstractmethod
  def localWriteResetOffsetsA(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Reset Offsets B
  ##############################################################################
  @abc.abstractmethod
  def localWriteResetOffsetsB(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Init Pointers A
  ##############################################################################
  @abc.abstractmethod
  def localWriteInitPointersA(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Init Pointers B
  ##############################################################################
  @abc.abstractmethod
  def localWriteInitPointersB(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Do It A
  ##############################################################################
  @abc.abstractmethod
  def localWriteDoA(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Do It B
  ##############################################################################
  @abc.abstractmethod
  def localWriteDoB(self, kernel):
    return ""

  ##############################################################################
  # Local Read: Swap Offsets A
  ##############################################################################
  @abc.abstractmethod
  def localReadSwapOffsetsA(self, kernel):
    return ""

  ##############################################################################
  # Local Read: Wwap Offsets B
  ##############################################################################
  @abc.abstractmethod
  def localReadSwapOffsetsB(self, kernel):
    return ""

  ##############################################################################
  # Local Read: Reset Offsets A
  ##############################################################################
  @abc.abstractmethod
  def localReadResetOffsetsA(self, kernel):
    return ""

  ##############################################################################
  # Local Read: Reset Offsets B
  ##############################################################################
  @abc.abstractmethod
  def localReadResetOffsetsB(self, kernel):
    return ""

  ##############################################################################
  # Local Read: Init Pointers A
  ##############################################################################
  @abc.abstractmethod
  def localReadInitPointersA(self, kernel):
    return ""

  ##############################################################################
  # Local Read: Init Pointers B
  ##############################################################################
  @abc.abstractmethod
  def localReadInitPointersB(self, kernel):
    return ""

  ##############################################################################
  # Local Read: Increment A
  ##############################################################################
  @abc.abstractmethod
  def localReadIncA(self, kernel):
    return ""

  ##############################################################################
  # Local Read: Increment B
  ##############################################################################
  @abc.abstractmethod
  def localReadIncB(self, kernel):
    return ""

  ##############################################################################
  # Local Read: Do It A
  ##############################################################################
  @abc.abstractmethod
  def localReadDoA(self, kernel, black):
    return ""

  ##############################################################################
  # Local Read: Do It B
  ##############################################################################
  @abc.abstractmethod
  def localReadDoB(self, kernel, black):
    return ""

  ##############################################################################
  # Shift Vector Components d0
  ##############################################################################
  @abc.abstractmethod
  def shiftVectorComponents0(self, kernel):
    return ""

  ##############################################################################
  # Shift Vectors Components d1
  ##############################################################################
  @abc.abstractmethod
  def shiftVectorComponents1(self, kernel):
    return ""

  ##############################################################################
  # Complex Declare Tmp Registers
  ##############################################################################
  @abc.abstractmethod
  def complexDeclareTmpRegisters(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  @abc.abstractmethod
  def localSplitULocalWrite(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  @abc.abstractmethod
  def localSplitULocalRead(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  @abc.abstractmethod
  def localSplitUReduction(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWrite(self, kernel):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWrite(self, kernel):
    return ""

  ##############################################################################
  # Function End
  ##############################################################################
  @abc.abstractmethod
  def functionEnd(self, kernel):
    return ""

  ##############################################################################
  # Function Suffix
  ##############################################################################
  @abc.abstractmethod
  def functionSuffix(self, kernel):
    return ""

  ##############################################################################
  # Kernel Body Prefix
  ##############################################################################
  @abc.abstractmethod
  def kernelBodyPrefix(self, kernel):
    return ""

  ##############################################################################
  # Kernel Body Suffix
  ##############################################################################
  @abc.abstractmethod
  def kernelBodySuffix(self, kernel):
    return ""

  ##############################################################################
  #
  #   Entry Functions
  #
  ##############################################################################


  ##############################################################################
  # get kernel name
  ##############################################################################
  def getKernelName(self, kernel):
    if globalParameters["ShortNames"]:
      kernelName = Solution.getNameSerial(kernel, self.kernelSerialNaming)
    else:
      kernelName = Solution.getNameMin(kernel, self.kernelMinNaming)
    return kernelName


  ##############################################################################
  # source file string
  ##############################################################################
  def getSourceFileString(self, kernel):
    fileString = ""
    self.initKernel(kernel)
    fileString += self.kernelBodyPrefix( kernel )
    self.stringIdx = 0
    fileString += self.kernelBody( kernel )
    fileString += self.kernelBodySuffix( kernel )
    return fileString


  ##############################################################################
  # header file string
  ##############################################################################
  def getHeaderFileString(self, kernel):
    kernelName = self.getKernelName(kernel)
    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += "#pragma once\n\n"
      fileString += "\n"
      if self.language == "HIP":
        fileString += "#include <hip/hip_runtime.h>\n"
        fileString += "#include <hip/hip_fp16.h>\n"
        fileString += "\n"
      else:
        fileString += "#include <string>\n"
    if self.language == "OCL":
      fileString += "extern const char * const %s_src;\n" % kernelName
    else:
      fileString += self.functionSignature(kernel)
      fileString += ";\n"

    return fileString

  ##############################################################################
  #
  #   Beta-Only Kernels
  #
  # kernel dictionary has ProblemType for indices and Beta=True/False
  ##############################################################################

  ##############################################################################
  # Get Name
  ##############################################################################
  def getKernelNameBetaOnly(self, kernel):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += kernel["ProblemType"]["DataType"].toChar()
    if kernel["ProblemType"]["UseBeta"]: name += "B"
    if kernel["ProblemType"]["UseInitialStrides"]: name += "I"
    return name

  @abc.abstractmethod
  def functionSignatureBetaOnly(kernel):
    return ""

  @abc.abstractmethod
  def kernelBodyBetaOnly( self, kernel ):
    return ""

  def getSourceFileStringBetaOnly(self, kernel):
    fileString = ""
    kernelName = self.getKernelNameBetaOnly(kernel)
    if not globalParameters["MergeFiles"]:
      fileString += "\n"
      fileString += "#include \"%s.h\"\n" % kernelName
      fileString += "\n"
    if self.language == "OCL":
      fileString += "const char * const %s_src = \"\"\n\"" % kernelName
    fileString += self.functionSignatureBetaOnly( kernel )
    fileString += self.kernelBodyBetaOnly( kernel )
    if self.language == "OCL":
      fileString += "\";"
    return fileString

  def getHeaderFileStringBetaOnly(self, kernel):
    kernelName = self.getKernelNameBetaOnly(kernel)
    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += "#pragma once\n\n"
      fileString += "\n"
      if self.language == "HIP":
        fileString += "#include <hip/hip_runtime.h>\n"
        fileString += "#include <hip/hip_fp16.h>\n"
        fileString += "\n"
      else:
        fileString += "#include <string>\n"
    if self.language == "OCL":
      fileString += "extern const char * const %s_src;\n" % kernelName
    else:
      fileString += self.functionSignatureBetaOnly(kernel)
      fileString += ";\n"

    return fileString
