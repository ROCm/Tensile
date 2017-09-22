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
from Common import globalParameters, kernelLanguageIsSource, pushWorkingPath, popWorkingPath, printWarning, printExit, print1, print2, HR
import abc
from os import path, chmod
from os import name as osname
from subprocess import Popen

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

    self.enable = {}
    self.enable["PreLoop"]        = True
    self.enable["GlobalRead"]     = True
    self.enable["GlobalReadInc"]  = True
    self.enable["LocalWrite"]     = True
    self.enable["LocalRead"]      = True
    self.enable["Wait"]           = True
    self.enable["Sync"]           = True
    self.enable["MAC"]            = True
    self.enable["PostLoop"]       = True


  ##############################################################################
  # Kernel Body
  ##############################################################################
  def kernelBody( self, kernel, tensorParametersA, tensorParametersB ):


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

    if self.enable["PreLoop"]:
      ####################################
      # Global Read Addresses
      ####################################
      kStr += self.comment3("Global Read Addresses")

      # subgroup assignments
      kStr += self.comment("global read addresses: subgroup")
      kStr += self.graSubgroup(kernel)

      # work-group assignments
      kStr += self.comment("global read addresses: work-group")
      kStr += self.graWorkGroup(kernel)

      # tile assignments
      kStr += self.comment("global read addresses: tile offset assignment a")
      kStr += self.graTileAssignment(kernel, tensorParametersA)
      kStr += self.comment("global read addresses: tile offset assignment b")
      kStr += self.graTileAssignment(kernel, tensorParametersB)

      # unroll assignments
      kStr += self.comment("global read addresses: unroll assignment a")
      kStr += self.graUnrollAssignment(kernel, tensorParametersA)
      kStr += self.comment("global read addresses: unroll assignment b")
      kStr += self.graUnrollAssignment(kernel, tensorParametersB)

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
      kStr += self.graTileOffsets(kernel, tensorParametersA)
      kStr += self.comment("global read addresses: tile offsets b")
      kStr += self.graTileOffsets(kernel, tensorParametersB)

      # unroll offsets
      kStr += self.comment("global read addresses: unroll offsets a")
      kStr += self.graUnrollOffsets(kernel, tensorParametersA)
      kStr += self.comment("global read addresses: unroll offsets b")
      kStr += self.graUnrollOffsets(kernel, tensorParametersB)

      # tile edges
      if kernel["EdgeType"] == "ShiftPtr":
        kStr += self.comment("global read addresses: shift a")
        kStr += self.graShift(kernel, tensorParametersA)
        kStr += self.comment("global read addresses: shift b")
        kStr += self.graShift(kernel, tensorParametersB)
      elif kernel["EdgeType"] == "Branch":
        kStr += self.comment("global read addresses: branch a")
        kStr += self.graBranch(kernel, tensorParametersA)
        kStr += self.comment("global read addresses: branch b")
        kStr += self.graBranch(kernel, tensorParametersB)

      # final offsets
      kStr += self.comment("global read addresses: final offsets a")
      kStr += self.graFinalOffsets(kernel, tensorParametersA)
      kStr += self.comment("global read addresses: final offsets b")
      kStr += self.graFinalOffsets(kernel, tensorParametersB)

      # user offsets
      kStr += self.comment("global read addresses: apply user offsets")
      kStr += self.graApplyUserOffsets(kernel)

      # addresses
      kStr += self.comment("global read addresses: addresses a")
      kStr += self.graAddresses(kernel, tensorParametersA)
      kStr += self.comment("global read addresses: addresses b")
      kStr += self.graAddresses(kernel, tensorParametersB)

      # increments
      kStr += self.comment("global read addresses: increments a")
      for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]):
        kStr += self.graIncrements(kernel, i, tensorParametersA)
      kStr += self.comment("global read addresses: increments b")
      for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]):
        kStr += self.graIncrements(kernel, i, tensorParametersB)

      ####################################
      # Local Write Addresses
      ####################################
      kStr += self.comment3("Local Write Addresses")

      # tile assignments
      kStr += self.comment("local write addresses: tile assignment a")
      kStr += self.lwaTileAssignment(kernel, tensorParametersA)
      kStr += self.comment("local write addresses: tile assignment b")
      kStr += self.lwaTileAssignment(kernel, tensorParametersB)

      # unroll assignments
      kStr += self.comment("local write addresses: unroll assignment a")
      kStr += self.lwaUnrollAssignment(kernel, tensorParametersA)
      kStr += self.comment("local write addresses: unroll assignment b")
      kStr += self.lwaUnrollAssignment(kernel, tensorParametersB)

      # first offsets
      kStr += self.comment("local write addresses: first offset a")
      kStr += self.lwaFirstOffset(kernel, tensorParametersA)
      kStr += self.comment("local write addresses: first offset b")
      kStr += self.lwaFirstOffset(kernel, tensorParametersB)

      # final offsets
      kStr += self.comment("local write addresses: final offsets a")
      kStr += self.lwaFinalOffsets(kernel, tensorParametersA)
      kStr += self.comment("local write addresses: final offsets b")
      kStr += self.lwaFinalOffsets(kernel, tensorParametersB)

      # declare addresses
      kStr += self.comment("local write addresses: declare addresses a")
      kStr += self.lwaDeclareAddresses(kernel, tensorParametersA)
      kStr += self.comment("local write addresses: declare addresses b")
      kStr += self.lwaDeclareAddresses(kernel, tensorParametersB)

      # init pointers
      kStr += self.comment("local write addresses: init pointers a")
      kStr += self.localWriteInitPointers(kernel, tensorParametersA)
      kStr += self.comment("local write addresses: init pointers b")
      kStr += self.localWriteInitPointers(kernel, tensorParametersB)

      ####################################
      # Local Read Addresses
      ####################################
      kStr += self.comment3("Local Read Addresses")

      # tile assignments
      kStr += self.comment("local read addresses: tile assignments a")
      kStr += self.lraTileAssignmentA(kernel, tensorParametersA)
      kStr += self.comment("local read addresses: tile assignments b")
      kStr += self.lraTileAssignmentB(kernel, tensorParametersB)


      # final offsets
      kStr += self.comment("local read addresses: final offsets a")
      kStr += self.lraFinalOffset(kernel, tensorParametersA)
      kStr += self.comment("local read addresses: final offsets b")
      kStr += self.lraFinalOffset(kernel, tensorParametersB)

      # declare addresses
      kStr += self.comment("local read addresses: declare addresses a")
      kStr += self.lraDeclareAddresses(kernel, tensorParametersA)
      kStr += self.comment("local read addresses: declare addresses b")
      kStr += self.lraDeclareAddresses(kernel, tensorParametersB)

      # init pointers
      kStr += self.comment("local read addresses: init pointers a")
      kStr += self.localReadInitPointers(kernel, tensorParametersA)
      kStr += self.comment("local read addresses: init pointers b")
      kStr += self.localReadInitPointers(kernel, tensorParametersB)

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
      kStr += self.openSumAtLeastUnroll(kernel, True)
      if self.enable["GlobalRead"]:
        # global read
        kStr += self.comment("global read a")
        kStr += self.globalReadDo(kernel, False, tensorParametersA)
        kStr += self.comment("global read b")
        kStr += self.globalReadDo(kernel, False, tensorParametersB)
      if self.enable["GlobalReadInc"]:
        # increment global
        kStr += self.comment("global read inc a")
        kStr += self.globalReadIncrement(kernel, self.unrollIdx, \
            tensorParametersA)
        kStr += self.comment("global read inc b")
        kStr += self.globalReadIncrement(kernel, self.unrollIdx, \
            tensorParametersB)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for global read")
      if self.enable["LocalWrite"]:
        # local write
        kStr += self.comment("local write a")
        kStr += self.localWriteDo(kernel, tensorParametersA)
        kStr += self.comment("local write b")
        kStr += self.localWriteDo(kernel, tensorParametersB)
        # swap local ptrs
        kStr += self.comment("local write swap a")
        kStr += self.localWriteSwapOffsets(kernel, tensorParametersA)
        kStr += self.comment("local write swap b")
        kStr += self.localWriteSwapOffsets(kernel, tensorParametersB)
        kStr += self.comment("local write init pointers a")
        kStr += self.localWriteInitPointers(kernel, tensorParametersA)
        kStr += self.comment("local write init pointers b")
        kStr += self.localWriteInitPointers(kernel, tensorParametersB)
      # prefetch-local
      if kernel["PrefetchLocalRead"]:
        if self.enable["Wait"]:
          kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "wait for local write")
        if self.enable["Sync"]:
          kStr += self.syncThreads(kernel)
        if self.enable["LocalRead"]:
          kStr += self.comment("local read prefetch a")
          kStr += self.localReadDo(kernel, False, tensorParametersA)
          kStr += self.comment("local read prefetch b")
          kStr += self.localReadDo(kernel, False, tensorParametersB)
          kStr += self.comment("local read inc a")
          kStr += self.localReadInc(kernel, tensorParametersA)
          kStr += self.comment("local read inc b")
          kStr += self.localReadInc(kernel, tensorParametersB)
      kStr += self.closeSumAtLeastUnroll(kernel, True)

    # open unrolled summation loop
    kStr += self.comment3("Unrolled Loop - Begin")
    kStr += self.openLoop(kernel, self.unrollIdx)

    if self.enable["GlobalRead"]:
      # unrolled loop: global read A, B
      kStr += self.comment("global read a")
      kStr += self.globalReadDo(kernel, False, tensorParametersA)
      kStr += self.comment("global read b")
      kStr += self.globalReadDo(kernel, False, tensorParametersB)

    if self.enable["GlobalReadInc"]:
      # unrolled loop: increment global read addresses
      kStr += self.comment("global read inc a")
      kStr += self.globalReadIncrement(kernel, self.unrollIdx, \
          tensorParametersA)
      kStr += self.comment("global read inc b")
      kStr += self.globalReadIncrement(kernel, self.unrollIdx, \
          tensorParametersB)

    if kernel["PrefetchGlobalRead"] and not kernel["PrefetchLocalRead"]:
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "wait for local write")
      if self.enable["Sync"]:
        kStr += self.syncThreads(kernel)

    # if not prefetch global, localWrite before mac's
    if not kernel["PrefetchGlobalRead"]:
      # unrolled loop: local write A, B
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for global read")
      if self.enable["Sync"]:
        kStr += self.syncThreads(kernel) # prior iter done reading lds
      if self.enable["LocalWrite"]:
        kStr += self.comment("local write a")
        kStr += self.localWriteDo(kernel, tensorParametersA)
        kStr += self.comment("local write b")
        kStr += self.localWriteDo(kernel, tensorParametersB)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "wait for local write")
      if self.enable["Sync"]:
        kStr += self.syncThreads(kernel)
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
      if self.enable["LocalRead"]:
        kStr += self.comment("prefetch local a")
        kStr += self.localReadDo(kernel, False, tensorParametersA)
        kStr += self.comment("prefetch local b")
        kStr += self.localReadDo(kernel, False, tensorParametersB)
        kStr += self.comment("local read increment a")
        kStr += self.localReadInc(kernel, tensorParametersA)
        kStr += self.comment("local read increment b")
        kStr += self.localReadInc(kernel, tensorParametersB)

    kStr += self.closeString(kernel)
    kStr += self.openString(kernel)

    ############################################################################
    # unrolled loop: mac iterations
    ############################################################################
    for u in range(0, kernel["LoopUnroll"]-2):
     # local read
      kStr += self.comment("iter %u"%u)
      readBlk = kernel["PrefetchLocalRead"] and u%2==0
      if self.enable["LocalRead"]:
        kStr += self.comment("local read a")
        kStr += self.localReadDo(kernel, readBlk, tensorParametersA)
        kStr += self.comment("local read b")
        kStr += self.localReadDo(kernel, readBlk, tensorParametersB)
        kStr += self.comment("local read increment a")
        kStr += self.localReadInc(kernel, tensorParametersA)
        kStr += self.comment("local read increment b")
        kStr += self.localReadInc(kernel, tensorParametersB)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, 1 if (u==0 and kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"]) else -1, -1, 1 if kernel["PrefetchLocalRead"] else 0, "wait for prior local read")
      if self.enable["MAC"]:
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
      if self.enable["LocalRead"]:
        # local read for last unroll
        kStr += self.comment("local read a")
        kStr += self.localReadDo(kernel, True, tensorParametersA)
        kStr += self.comment("local read b")
        kStr += self.localReadDo(kernel, True, tensorParametersB)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for global read")
      if self.enable["LocalWrite"]:
        # local write for next iter
        kStr += self.comment("local write a")
        kStr += self.localWriteDo(kernel, tensorParametersA)
        kStr += self.comment("local write b")
        kStr += self.localWriteDo(kernel, tensorParametersB)
        kStr += self.comment("local write swap offsets a")
        kStr += self.localWriteSwapOffsets(kernel, tensorParametersA)
        kStr += self.comment("local write swap offsets b")
        kStr += self.localWriteSwapOffsets(kernel, tensorParametersB)
        kStr += self.comment("local write init pointers a")
        kStr += self.localWriteInitPointers(kernel, tensorParametersA)
        kStr += self.comment("local write init pointers b")
        kStr += self.localWriteInitPointers(kernel, tensorParametersB)
      if self.enable["LocalRead"]:
        # swap read and write pointers
        kStr += self.comment("local read swap offsets a")
        kStr += self.localReadSwapOffsets(kernel, tensorParametersA)
        kStr += self.comment("local read swap offsets b")
        kStr += self.localReadSwapOffsets(kernel, tensorParametersB)
        kStr += self.comment("local read init pointers a")
        kStr += self.localReadInitPointers(kernel, tensorParametersA)
        kStr += self.comment("local read init pointers b")
        kStr += self.localReadInitPointers(kernel, tensorParametersB)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, 1, 1, "wait for prior local read")
    else:
      if self.enable["LocalRead"]:
        # local read
        readBlk = kernel["PrefetchLocalRead"] and unrollIter%2==0
        kStr += self.comment("local read a")
        kStr += self.localReadDo(kernel, readBlk, tensorParametersA)
        kStr += self.comment("local read b")
        kStr += self.localReadDo(kernel, readBlk, tensorParametersB)
        if kernel["PrefetchLocalRead"]:
          # local read init ptrs
          kStr += self.comment("local read init pointers a")
          kStr += self.localReadInitPointers(kernel, tensorParametersA)
          kStr += self.comment("local read init pointers b")
          kStr += self.localReadInitPointers(kernel, tensorParametersB)
        else:
          # local read inc
          kStr += self.comment("local read inc a")
          kStr += self.localReadInc(kernel, tensorParametersA)
          kStr += self.comment("local read inc b")
          kStr += self.localReadInc(kernel, tensorParametersB)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 1 if kernel["PrefetchLocalRead"] else 0, "wait for prior local read")
    if self.enable["MAC"]:
      kStr += self.macIter(kernel, False)

    ####################################
    # unrolled loop: last summation iter
    ####################################
    # if prefetch-local: read red for 1st unroll of next iter
    # if not prefetch-local: read for current unroll of current iter
    unrollIter = kernel["LoopUnroll"]-1
    kStr += self.comment("iter %u"%unrollIter)
    if kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"]:
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "wait for local write")
      if self.enable["Sync"]:
        kStr += self.syncThreads(kernel)
    if not kernel["PrefetchLocalRead"] or kernel["PrefetchGlobalRead"]:
      if self.enable["LocalRead"]:
        # local read
        readBlk = kernel["PrefetchLocalRead"] and unrollIter%2==0
        kStr += self.comment("local read a")
        kStr += self.localReadDo(kernel, readBlk, tensorParametersA)
        kStr += self.comment("local read b")
        kStr += self.localReadDo(kernel, readBlk, tensorParametersB)
    if kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"]:
      if self.enable["LocalRead"]:
        # local read inc
        kStr += self.comment("local read inc a")
        kStr += self.localReadInc(kernel, tensorParametersA)
        kStr += self.comment("local read inc b")
        kStr += self.localReadInc(kernel, tensorParametersB)
    elif kernel["PrefetchGlobalRead"]:
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for global read")
      if self.enable["LocalWrite"]:
        # local write
        kStr += self.comment("local write a")
        kStr += self.localWriteDo(kernel, tensorParametersA)
        kStr += self.comment("local write b")
        kStr += self.localWriteDo(kernel, tensorParametersB)
        kStr += self.comment("local write swap offsets a")
        kStr += self.localWriteSwapOffsets(kernel, tensorParametersA)
        kStr += self.comment("local write swap offsets b")
        kStr += self.localWriteSwapOffsets(kernel, tensorParametersB)
        kStr += self.comment("local write init pointers a")
        kStr += self.localWriteInitPointers(kernel, tensorParametersA)
        kStr += self.comment("local write init pointers b")
        kStr += self.localWriteInitPointers(kernel, tensorParametersB)
      if self.enable["LocalRead"]:
        # swap read and write
        kStr += self.comment("local read swap offsets a")
        kStr += self.localReadSwapOffsets(kernel, tensorParametersA)
        kStr += self.comment("local read swap offsets b")
        kStr += self.localReadSwapOffsets(kernel, tensorParametersB)
        kStr += self.comment("local read init pointers a")
        kStr += self.localReadInitPointers(kernel, tensorParametersA)
        kStr += self.comment("local read init pointers b")
        kStr += self.localReadInitPointers(kernel, tensorParametersB)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, 1, 0, "wait for local read")
    elif not kernel["PrefetchGlobalRead"] and not kernel["PrefetchLocalRead"]:
      if self.enable["LocalRead"]:
        # local read init ptrs
        kStr += self.comment("local read init pointers a")
        kStr += self.localReadInitPointers(kernel, tensorParametersA)
        kStr += self.comment("local read init pointers b")
        kStr += self.localReadInitPointers(kernel, tensorParametersB)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "wait for local read")
    else:
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "wait for local read")
    # no wait needed here b/c we already waited for ds_write
    # which waited for this ds_read
    if self.enable["MAC"]:
      kStr += self.macIter(kernel, kernel["PrefetchLocalRead"])

    # close unrolled loop
    kStr += self.comment3("Unrolled Loop - End")
    kStr += self.closeLoop(kernel, self.unrollIdx)

    # prefetch: unrolled loop suffix
    if kernel["PrefetchGlobalRead"]:
      kStr += self.comment("prefetch: last unrolled iteration")
      kStr += self.openSumAtLeastUnroll(kernel, False)
      if not kernel["PrefetchLocalRead"]:
        if self.enable["Wait"]:
          kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "wait for local write")
        if self.enable["Sync"]:
          kStr += self.syncThreads(kernel)
      for u in range(0, kernel["LoopUnroll"]):
        kStr += self.comment("iter %u"%u)
        readBlk = kernel["PrefetchLocalRead"] and u%2==0
        if self.enable["LocalRead"]:
          if u < kernel["LoopUnroll"]-1 or not kernel["PrefetchLocalRead"]:
            kStr += self.comment("local read a")
            kStr += self.localReadDo(kernel, readBlk, tensorParametersA)
            kStr += self.comment("local read b")
            kStr += self.localReadDo(kernel, readBlk, tensorParametersB)
            kStr += self.comment("local read inc a")
            kStr += self.localReadInc(kernel, tensorParametersA)
            kStr += self.comment("local read inc b")
            kStr += self.localReadInc(kernel, tensorParametersB)
        if self.enable["Wait"]:
          kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, \
              1 if (u < kernel["LoopUnroll"]-1 and kernel["PrefetchLocalRead"]) else 0, "wait for local read")
        if self.enable["MAC"]:
          kStr += self.macIter(kernel, (kernel["PrefetchLocalRead"] and u%2==1) )
      kStr += self.closeSumAtLeastUnroll(kernel, False)

    ########################################
    # Tail Loop
    ########################################
    if kernel["LoopTail"]:
      kStr += self.comment3("Tail Loop")

      if self.enable["GlobalRead"]:
        # tail: global read
        kStr += self.calculateLoopNumIter(kernel, -1)
        kStr += self.comment("global read a")
        kStr += self.globalReadDo(kernel, True, tensorParametersA)
        kStr += self.comment("global read b")
        kStr += self.globalReadDo(kernel, True, tensorParametersB)
      if self.enable["Sync"]:
        kStr += self.syncThreads(kernel)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for global read")
      if self.enable["LocalWrite"]:
        # tail: local write
        if kernel["PrefetchGlobalRead"]:
          kStr += self.comment("local write reset offsets a")
          kStr += self.localWriteResetOffsets(kernel, tensorParametersA)
          kStr += self.comment("local write reset offsets b")
          kStr += self.localWriteResetOffsets(kernel, tensorParametersB)
        kStr += self.comment("local write init pointers a")
        kStr += self.localWriteInitPointers(kernel, tensorParametersA)
        kStr += self.comment("local write init pointers b")
        kStr += self.localWriteInitPointers(kernel, tensorParametersB)
        kStr += self.comment("local write a")
        kStr += self.localWriteDo(kernel, tensorParametersA)
        kStr += self.comment("local write b")
        kStr += self.localWriteDo(kernel, tensorParametersB)
      if self.enable["Sync"]:
        kStr += self.syncThreads(kernel)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "wait for local write")
      #kStr += self.dumpLds(kernel, 0, 8)

      # tail: re-init local read addresses
      if kernel["PrefetchGlobalRead"]:
        kStr += self.comment("local read reset offsets a")
        kStr += self.localReadResetOffsets(kernel, tensorParametersA)
        kStr += self.comment("local read reset offsets b")
        kStr += self.localReadResetOffsets(kernel, tensorParametersB)
        kStr += self.comment("local read init pointers a")
        kStr += self.localReadInitPointers(kernel, tensorParametersA)
        kStr += self.comment("local read init pointers b")
        kStr += self.localReadInitPointers(kernel, tensorParametersB)

      # tail: macs
      kStr += self.comment("tail loop: macs")
      kStr += self.openLoop(kernel, -1)
      if self.enable["LocalRead"]:
        kStr += self.comment("local read a")
        kStr += self.localReadDo(kernel, False, tensorParametersA)
        kStr += self.comment("local read b")
        kStr += self.localReadDo(kernel, False, tensorParametersB)
        kStr += self.comment("local read inc a")
        kStr += self.localReadInc(kernel, tensorParametersA)
        kStr += self.comment("local read inc b")
        kStr += self.localReadInc(kernel, tensorParametersB)
      if self.enable["Wait"]:
        kStr += self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "wait for local read")
      if self.enable["MAC"]:
        kStr += self.macIter(kernel, False )

      # tail: close
      kStr += self.closeLoop(kernel, -1)

    # extra summation loops: global increment and close
    for i in reversed(range(0,kernel["ProblemType"]["NumIndicesSummation"]-1)):
      kStr += self.comment("global read inc a")
      kStr += self.globalReadIncrement(kernel, i, tensorParametersA)
      kStr += self.comment("global read inc b")
      kStr += self.globalReadIncrement(kernel, i, tensorParametersB)
      kStr += self.closeLoop(kernel, i)

    kStr += self.endSummation()
    if self.enable["PostLoop"]:

      ####################################
      # Shift Vector Components
      ####################################
      if kernel["EdgeType"] == "ShiftPtr" and kernel["VectorWidth"] > 1:
        # shift vector components d0
        if self.readTileDimVectorA:
          kStr += self.comment("shift vector components d0")
          kStr += self.shiftVectorComponents(kernel, tensorParametersA)
        # shift vector components d1
        if self.readTileDimVectorB:
          kStr += self.comment("shift vector components d1")
          kStr += self.shiftVectorComponents(kernel, tensorParametersB)

      # complex declare tmp registers
      kStr += self.complexDeclareTmpRegisters(kernel)

      ####################################
      # LocalSplitU reduction
      ####################################
      #if kernel["NumThreads"]%kernel["MacroTile0"] == 0:
      if kernel["LocalSplitU"] > 1:
        kStr += self.comment3("LocalSplitU Reduction")
        if self.enable["Sync"]:
          kStr += self.syncThreads(kernel)

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
  def initKernel(self, kernel, tensorParametersA, tensorParametersB ):
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
    """
    # original parameters
    NumLoadsCoalesced -> NumLoadsPerpendicular

    # new intermediate parameters
    numReadsTile # nrt
    numReadsUnroll # nru
    numReadsTileVecComp # nrvt
    numReadsUnrollVecComp # nrvu

    numWritesCoal # nwc
    numWritesPerp # nwp
    numWritesCoalVecComp # nwvc
    numWritesPerpVecComp # nwvp

    readTileComponents (based on grcv)
    readTileVector
    """

    # TODO load sub-vector
    vwa = kernel["GlobalLoadVectorWidthA"]
    vwb = kernel["GlobalLoadVectorWidthB"]

    ########################################
    # read / write vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"]: # NT no transpose
      self.numReadsTileA = kernel["NumLoadsCoalescedA"]
      self.numReadsUnrollA = kernel["NumLoadsPerpendicularA"]
      self.numWritesCoalA = kernel["NumLoadsCoalescedA"]
      self.numWritesPerpA = kernel["NumLoadsPerpendicularA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vectors, write vectors
        self.readTileDimComponentsA = False # Vector
        self.readTileDimVectorA = True # Vector
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        self.writeTileDimComponentsA = False # Vector
        self.writeUnrollDimComponentsA = False # Scalar
        # NEW
        self.numReadsTileVecCompA = vwa
        self.numReadsUnrollVecCompA = 1
        self.numWritesCoalVecCompA = vwa
        self.numWritesPerpVecCompA = 1
      else: # read components, write components
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorA = False # Components
        self.writeTileDimComponentsA = False # Scalar
        self.writeUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
        # NEW
        self.numReadsTileVecCompA = 1
        self.numReadsUnrollVecCompA = vwa
        self.numWritesCoalVecCompA = 1
        self.numWritesPerpVecCompA = vwa
    else: # TN yes transpose
      self.numReadsTileA = kernel["NumLoadsPerpendicularA"]
      self.numReadsUnrollA = kernel["NumLoadsCoalescedA"]
      self.numWritesCoalA = kernel["NumLoadsPerpendicularA"]
      self.numWritesPerpA = kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vector, write components
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = False # Vector
        self.readUnrollDimVectorA = True # Vector
        self.writeTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.writeUnrollDimComponentsA = False # Scalar
        # NEW
        self.numReadsUnrollVecCompA = vwa
        self.numReadsTileVecCompA = 1
        self.numWritesCoalVecCompA = 1
        self.numWritesPerpVecCompA = vwa
      else: # read components, write vectors
        self.readTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readTileDimVectorA = False # Components
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        self.writeTileDimComponentsA = False # Vector
        self.writeUnrollDimComponentsA = False # Scalar
        # NEW
        self.numReadsUnrollVecCompA = 1
        self.numReadsTileVecCompA = vwa
        self.numWritesCoalVecCompA = vwa
        self.numWritesPerpVecCompA = 1


    self.numReadVectorComponentsA = kernel["GlobalLoadVectorWidthA"] \
        if (self.readTileDimComponentsA \
        or self.readUnrollDimComponentsA) else 1
    self.numWriteVectorComponentsA = kernel["GlobalLoadVectorWidthA"] \
        if (self.writeTileDimComponentsA \
        or self.writeUnrollDimComponentsA) else 1
    self.numReadTileVectorComponentsA = kernel["GlobalLoadVectorWidthA"] \
        if self.readTileDimComponentsA else 1 # for branches
    # convert tile/unroll to para/perp
    if kernel["ProblemType"]["TLUA"]:
      self.numReadsCoalVecCompA = self.numReadsTileVecCompA
      self.numReadsPerpVecCompA = self.numReadsUnrollVecCompA
      # for asm
      self.readCoalescedComponentsA  = self.readTileDimComponentsA
      self.readCoalescedVectorA      = self.readTileDimVectorA
      self.readPerpendicularComponentsA  = self.readUnrollDimComponentsA
      self.readPerpendicularVectorA      = self.readUnrollDimVectorA
    else:
      self.numReadsCoalVecCompA = self.numReadsUnrollVecCompA
      self.numReadsPerpVecCompA = self.numReadsTileVecCompA
      # for asm
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
      self.numWritesCoalB = kernel["NumLoadsCoalescedB"]
      self.numWritesPerpB = kernel["NumLoadsPerpendicularB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Vector
        self.readTileDimVectorB = True # Vector
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        self.writeTileDimComponentsB = False # Vector
        self.writeUnrollDimComponentsB = False # Scalar
        # NEW
        self.numReadsTileVecCompB = vwb
        self.numReadsUnrollVecCompB = 1
        self.numWritesCoalVecCompB = vwb
        self.numWritesPerpVecCompB = 1
      else:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorB = False # Components
        self.writeTileDimComponentsB = False # Scalar
        self.writeUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
        # NEW
        self.numReadsTileVecCompB = 1
        self.numReadsUnrollVecCompB = vwb
        self.numWritesCoalVecCompB = 1
        self.numWritesPerpVecCompB = vwb
    else:
      self.numReadsTileB = kernel["NumLoadsPerpendicularB"]
      self.numReadsUnrollB = kernel["NumLoadsCoalescedB"]
      self.numWritesCoalB = kernel["NumLoadsPerpendicularB"]
      self.numWritesPerpB = kernel["NumLoadsCoalescedB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = False # Vector
        self.readUnrollDimVectorB = True # Vector
        self.writeTileDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.writeUnrollDimComponentsB = False # Scalar
        # NEW
        self.numReadsUnrollVecCompB = vwb
        self.numReadsTileVecCompB = 1
        self.numWritesCoalVecCompB = 1
        self.numWritesPerpVecCompB = vwb
      else:
        self.readTileDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.readTileDimVectorB = False # Components
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        self.writeTileDimComponentsB = False # Vector
        self.writeUnrollDimComponentsB = False # Scalar
        # NEW
        self.numReadsUnrollVecCompB = 1
        self.numReadsTileVecCompB = vwb
        self.numWritesCoalVecCompB = vwb
        self.numWritesPerpVecCompB = 1

    self.numReadVectorComponentsB = kernel["GlobalLoadVectorWidthB"] \
        if (self.readTileDimComponentsB \
        or self.readUnrollDimComponentsB) else 1
    self.numWriteVectorComponentsB = kernel["GlobalLoadVectorWidthB"] \
        if (self.writeTileDimComponentsB \
        or self.writeUnrollDimComponentsB) else 1
    self.numReadTileVectorComponentsB = kernel["GlobalLoadVectorWidthB"] \
        if self.readTileDimComponentsB else 1 # for branches
    # convert tile/unroll to para/perp
    if kernel["ProblemType"]["TLUB"]:
      self.numReadsCoalVecCompB = self.numReadsTileVecCompB
      self.numReadsPerpVecCompB = self.numReadsUnrollVecCompB
      # for asm
      self.readCoalescedComponentsB  = self.readTileDimComponentsB
      self.readCoalescedVectorB      = self.readTileDimVectorB
      self.readPerpendicularComponentsB  = self.readUnrollDimComponentsB
      self.readPerpendicularVectorB      = self.readUnrollDimVectorB
    else:
      self.numReadsCoalVecCompB = self.numReadsUnrollVecCompB
      self.numReadsPerpVecCompB = self.numReadsTileVecCompB
      # for asm
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

    kernel["LVCA"] = kernel["LSCA"] / kernel["GlobalLoadVectorWidthA"]
    kernel["LVCB"] = kernel["LSCB"] / kernel["GlobalLoadVectorWidthB"]
    kernel["LVPA"] = kernel["LSPA"] / kernel["GlobalLoadVectorWidthA"]
    kernel["LVPB"] = kernel["LSPB"] / kernel["GlobalLoadVectorWidthB"]

    self.getTensorParameters(tensorParametersA, kernel, True)
    self.getTensorParameters(tensorParametersB, kernel, False)


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
  # Get Params For Tensor A/B
  ##############################################################################
  def getTensorParameters(self, tP, kernel, tA):
    if tA: # A
      tP["isA"] = True
      tP["isB"] = False
      tP["tensorChar"] = "A"
      tP["tensorIdx"] = 0
      tP["tileChar"] = self.tileCharA
      tP["tileIdx"] = kernel["ProblemType"]["TileA"]
      tP["lsc"] = "LSCA"
      tP["lsp"] = "LSPA"
      tP["lvc"] = "LVCA"
      tP["lvp"] = "LVPA"
      tP["rtv"] = self.readTileDimVectorA
      tP["rtc"] = self.readTileDimComponentsA
      #tP["ruv"] = self.readUnrollDimVectorA
      #tP["nlvc"] = self.numReadVectorComponentsA
      #tP["nwvc"] = self.numWriteVectorComponentsA
      tP["wg"] = "WorkGroup0"
      tP["sg"] = "SubGroup0"
      tP["tt"] = "ThreadTile0"
      tP["mt"] = "MacroTile0"
      tP["grcg"] = self.globalReadCoalesceGroupA
      tP["grcv"] = kernel["GlobalReadCoalesceVectorA"]
      tP["tlu"] = kernel["ProblemType"]["TLUA"]
      tP["ia"] = kernel["ProblemType"]["IndexAssignmentsA"]
      #tP["nlc"] = kernel["NumLoadsCoalescedA"]
      #tP["nlp"] = kernel["NumLoadsPerpendicularA"]
      #tP["nlcv"] = self.numReadsCoalVecCompA
      tP["nlpv"] = self.numReadsPerpVecCompA
      # NEW
      tP["nrt"] = self.numReadsTileA
      tP["nrtv"] = self.numReadsTileVecCompA
      tP["nru"] = self.numReadsUnrollA
      tP["nruv"] = self.numReadsUnrollVecCompA
      tP["nrc"] = kernel["NumLoadsCoalescedA"]
      tP["nrcv"] = self.numReadsCoalVecCompA
      tP["nrp"] = kernel["NumLoadsPerpendicularA"]
      tP["nrpv"] = self.numReadsPerpVecCompA
      tP["nwcv"] = self.numWritesCoalVecCompA
      tP["nwpv"] = self.numWritesPerpVecCompA
      tP["glvw"] = kernel["GlobalLoadVectorWidthA"]
      # asm
      tP["rcc"] = self.readCoalescedComponentsA
      tP["rcv"] = self.readCoalescedVectorA
      tP["rpc"] = self.readPerpendicularComponentsA
      tP["rpv"] = self.readPerpendicularVectorA
      tP["ruc"] = self.readUnrollDimComponentsA
      tP["wtc"] = self.writeTileDimComponentsA
      tP["wuc"] = self.writeUnrollDimComponentsA
    else: # B
      tP["isA"] = False
      tP["isB"] = True
      tP["tensorChar"] = "B"
      tP["tensorIdx"] = 1
      tP["tileChar"] = self.tileCharB
      tP["tileIdx"] = kernel["ProblemType"]["TileB"]
      tP["lsc"] = "LSCB"
      tP["lsp"] = "LSPB"
      tP["lvc"] = "LVCB"
      tP["lvp"] = "LVPB"
      tP["rtv"] = self.readTileDimVectorB
      tP["rtc"] = self.readTileDimComponentsB
      #tP["ruv"] = self.readUnrollDimVectorB
      #tP["nlvc"] = self.numReadVectorComponentsB
      #tP["nwvc"] = self.numWriteVectorComponentsB
      tP["wg"] = "WorkGroup1"
      tP["sg"] = "SubGroup1"
      tP["tt"] = "ThreadTile1"
      tP["mt"] = "MacroTile1"
      tP["grcg"] = self.globalReadCoalesceGroupB
      tP["grcv"] = kernel["GlobalReadCoalesceVectorB"]
      tP["tlu"] = kernel["ProblemType"]["TLUB"]
      tP["ia"] = kernel["ProblemType"]["IndexAssignmentsB"]
      # NEW
      tP["nrt"] = self.numReadsTileB
      tP["nrtv"] = self.numReadsTileVecCompB
      tP["nru"] = self.numReadsUnrollB
      tP["nruv"] = self.numReadsUnrollVecCompB
      tP["nrc"] = kernel["NumLoadsCoalescedB"]
      tP["nrcv"] = self.numReadsCoalVecCompB
      tP["nrp"] = kernel["NumLoadsPerpendicularB"]
      tP["nrpv"] = self.numReadsPerpVecCompB
      tP["nwcv"] = self.numWritesCoalVecCompB
      tP["nwpv"] = self.numWritesPerpVecCompB
      tP["glvw"] = kernel["GlobalLoadVectorWidthB"]
      # asm
      tP["rcc"] = self.readCoalescedComponentsB
      tP["rcv"] = self.readCoalescedVectorB
      tP["rpc"] = self.readPerpendicularComponentsB
      tP["rpv"] = self.readPerpendicularVectorB
      tP["ruc"] = self.readUnrollDimComponentsB
      tP["wtc"] = self.writeTileDimComponentsB
      tP["wuc"] = self.writeUnrollDimComponentsB

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollAssignment(self, kernel, tP):
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
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Branch A/B
  ##############################################################################
  @abc.abstractmethod
  def graBranch(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A/B
  ##############################################################################
  @abc.abstractmethod
  def graShift(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graFinalOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Apply User Offsets
  ##############################################################################
  @abc.abstractmethod
  def graApplyUserOffsets(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def graAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Increments A/B
  ##############################################################################
  @abc.abstractmethod
  def graIncrements(self, kernel, loopIdx, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaTileAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaUnrollAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaFirstOffset(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Final Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaFinalOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Declare Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaDeclareAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read Addresses: Tile Assignment A
  ##############################################################################
  @abc.abstractmethod
  def lraTileAssignmentA(self, kernel, tA):
    return ""

  ##############################################################################
  # Local Read Addresses: Tile Assignment B
  ##############################################################################
  @abc.abstractmethod
  def lraTileAssignmentB(self, kernel, tB):
    return ""

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lraFinalOffset(self, kernel, tA):
    return ""

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def lraDeclareAddresses(self, kernel, tP):
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
  # End Summation
  ##############################################################################
  @abc.abstractmethod
  def endSummation(self):
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
  def openSumAtLeastUnroll(self, kernel, prefetch):
    return ""

  @abc.abstractmethod
  def closeSumAtLeastUnroll(self, kernel, prefetch):
    return ""

  ##############################################################################
  # Global Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def globalReadIncrement(self, kernel, loopIdx, tP):
    return ""

  ##############################################################################
  # Global Read: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def globalReadDo(self, kernel, guardK, tP):
    return ""

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteSwapOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteResetOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write: Init Pointers A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteInitPointers(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteDo(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadSwapOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadResetOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInitPointers(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInc(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadDo(self, kernel, black, tP):
    return ""

  ##############################################################################
  # Shift Vector Components d0/1
  ##############################################################################
  @abc.abstractmethod
  def shiftVectorComponents(self, kernel, tP):
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
  def kernelBodyPrefix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # Kernel Body Suffix
  ##############################################################################
  @abc.abstractmethod
  def kernelBodySuffix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # WaitCnt
  ##############################################################################
  @abc.abstractmethod
  def wait(self, kernel, tPA, tPB, globalRead, localWrite, localRead, comment):
    return ""

  ##############################################################################
  # SyncThreads
  ##############################################################################
  @abc.abstractmethod
  def syncThreads(self, kernel):
    return self.indent + self.syncStr + self.endLine

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
    tensorParametersA = {}
    tensorParametersB = {}
    self.initKernel(kernel, tensorParametersA, tensorParametersB )
    fileString += self.kernelBodyPrefix( kernel, tensorParametersA, \
        tensorParametersB )
    self.stringIdx = 0
    fileString += self.kernelBody( kernel, tensorParametersA, tensorParametersB)
    fileString += self.kernelBodySuffix( kernel, tensorParametersA, \
        tensorParametersB )

    if not kernelLanguageIsSource():
      # write assembly file to assembly directory
      pushWorkingPath("assembly")
      pushWorkingPath(globalParameters["KernelLanguage"])
      kernelName = self.getKernelName(kernel)
      fileBase = path.join(globalParameters["WorkingPath"], kernelName )
      assemblyFileName = "%s.s" % fileBase
      objectFileName = "%s.o" % fileBase
      codeObjectFileName = "%s.co" % fileBase
      assemblyFile = open(assemblyFileName, "w")
      assemblyFile.write(fileString)
      assemblyFile.close()

      # assembler script
      assemblerFileName = path.join(globalParameters["WorkingPath"], \
          "asm.%s"%("bat" if osname=="nt" else "sh"))
      if not path.isfile(assemblerFileName):
        assemblerFile = open(assemblerFileName, "w")
        if osname == "nt":
          assemblerFile.write("echo Windows: Copying instead of Assembling\n")
          assemblerFile.write("copy %1.s %1.o\n")
          assemblerFile.write("copy %1.o %1.co\n")
        else:
          assemblerFile.write("#!/bin/sh\n")
          assemblerFile.write("ASM=%s\n"%globalParameters["AssemblerPath"])
          assemblerFile.write("${ASM} -x assembler -target amdgcn--amdhsa -mcpu=gfx%u%u%u -c -o $1.o $1.s\n" \
              % (self.versionMajor, self.versionMinor, self.versionStep))
          assemblerFile.write("${ASM} -target amdgcn--amdhsa $1.o -o $1.co\n")
        assemblerFile.close()
        chmod(assemblerFileName, 0777)

      # run assembler
      assemblerCommand = [assemblerFileName, kernelName]
      print2("# Assembling %s: %s" % (kernelName, assemblerCommand) )
      assemblerProcess = Popen(assemblerCommand, \
          cwd=globalParameters["WorkingPath"] )
      assemblerProcess.communicate()
      if assemblerProcess.returncode:
        printExit("Assembler process returned with code %u" \
            % assemblerProcess.returncode)

      # read code object file
      fileString = ""
      codeObjectFile = open(codeObjectFileName, "r")
      codeObjectByteArray = bytearray(codeObjectFile.read())
      codeObjectFile.close()

      # write code object byte array
      fileString += self.comment("code object byte array")
      fileString += "const unsigned char %s_coba[%u] = {\n" % (kernelName, len(codeObjectByteArray))
      for byteIdx in range(0, len(codeObjectByteArray)):
        byte = codeObjectByteArray[byteIdx]
        fileString += "0x%02x" % byte
        if byteIdx < len(codeObjectByteArray)-1:
          fileString += ","
        else:
          fileString += "};\n"
        if byteIdx % 16 == 15:
          fileString += "\n"


      popWorkingPath() # arch
      popWorkingPath() # assembly

      # read code-object file and convert to c++ representable uchar*
      # return string of code-object byte array
    return fileString


  ##############################################################################
  # header file string
  ##############################################################################
  def getHeaderFileString(self, kernel):
    kernelName = self.getKernelName(kernel)
    fileString = "" # CHeader
    if kernelLanguageIsSource():
      if not globalParameters["MergeFiles"]:
        fileString += "#pragma once\n\n"
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
    else:
      if not globalParameters["MergeFiles"]:
        fileString += "#pragma once\n\n"
      fileString += "extern const unsigned char %s_coba[]; // code object byte array\n" % kernelName

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
