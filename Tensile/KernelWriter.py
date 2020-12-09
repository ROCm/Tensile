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

from . import Code
from . import Common
from .Common import globalParameters, CHeader, roundUp
from .ReplacementKernels import ReplacementKernels
from .SolutionStructs import Solution

import abc
import os
import shutil
import subprocess
import copy
from itertools import zip_longest

################################################################################
# Kernel Writer
################################################################################
class KernelWriter(metaclass=abc.ABCMeta):
  #__metaclass__=abc.ABCMeta

  ##############################################################################
  # Init
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    self.kernelMinNaming = kernelMinNaming
    self.kernelSerialNaming = kernelSerialNaming
    self.overflowedResources = 0

  @property
  def asmCaps(self):
    """
    Assembler capabilities for the current ISA version.
    """
    return globalParameters["AsmCaps"][self.version]

  @property
  def archCaps(self):
    """
    Architectural capabilities for the current ISA version.
    """
    return globalParameters["ArchCaps"][self.version]

  @property
  def globalParams(self):
    """
    Global parameters for current configuration.
    """
    return globalParameters

  ##############################################################################
  # makeSchedule:  Schedule work into interations.

  # Tensile uses a two-level scheduler.  This the first-level, which
  # schedules global reads, global incs, and local writes into iteration.
  # Then makeSubIterSchedule schedules the instructions within the iteration.
  #
  # Inputs:
  #   localWriteEndIter: loop iteration where last writes should be inserted
  #      If scheduleLocalWrite=0, all writes will be be placed in this iteration.
  #      If scheduleLocalWrite=1, the scheduler will work backwards from this
  #      iteration.
  #
  # Outputs:
  #   self.unrollLoopHeaderCode:
  #      - Code module that should be added into the unroll loop header
  #        In unscheduled code this contains global loads and global address increment
  #   self.perIterGlobalReadCode[], self.perIterLocalWriteCode[]
  #      - List indexed by unroll iteration.
  #        Each entry in the list is a code module that should be added into that iteration.
  #        May be None, indicating no extra code for that iteration
  #   self.grEndMfmaIndex
  #   self.lwStartMfmaIndex
  #   self.lwEndMfmaIndex
  #   self.barrierMfmaIndex
  #   self.perIterLocalWriteCanSkip
  #   self.numGlobalReadInsPerMfma
  #   self.numReadsIterCoalescedB
  #   self.numMfmaForNextLoopLR
  # This routine is responsible for setting the schedule including determining
  # that all necessary dependency are met.  The driver code in kernelBody
  # blindly follows the plan set in unrollLoopHeaderCode and perIterCode
  ##############################################################################
  def makeSchedule(self, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu=0):
    # 0x2=print GR and LW code blocks, 0x1= print info messages
    schedDb = 0

    currentIsa = globalParameters["CurrentISA"]
    maxVmcnt = globalParameters["AsmCaps"][currentIsa]["MaxVmcnt"]

    self.unrollLoopHeaderCode = Code.Module()
    # schedule of work for each local_read iteration:
    self.perIterGlobalReadCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCodeNGLL = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
    assert([item.name for item in self.globalReadIncrements.itemList] == ['globalReadIncrementA', 'globalReadIncrementB'])

    globalReadIncACode  = self.globalReadIncrements.findNamedItem("globalReadIncrementA")
    globalReadIncBCode  = self.globalReadIncrements.findNamedItem("globalReadIncrementB")

    lastLoadIter = 0
    if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
      numMfmaPerIter = self.numMfmaPerIter
      ##################################
      # Can locally overrid these ######
      # number of mfma between last localWrite and barrier
      # mfma|lw|mfma|mfma|barrier, 2 mfma between last LW and barrier
      numMfmaBetweenLWandBarrier = 2 if kernel["MatrixInstM"] == 32 else 3
      # number of global read instructions between 2 mfma
      self.numGlobalReadInsPerMfma = 2 if kernel["MatrixInstM"] == 32 and not kernel["ProblemType"]["TLUA"] and not kernel["ProblemType"]["TLUB"] and kernel["TransposeLDS"] and not kernel["1LDSBuffer"] else 1
      # number of local write instructions between 2 mfma
      # we roughly peek number of localReads per mfma to decide number of localWrites per mfma
      readsLatencyA = self.numReadsPerIterA/numMfmaPerIter if self.numReadsIterCoalescedA == 1 else 0
      readsLatencyB = self.numReadsPerIterB/numMfmaPerIter if self.numReadsIterCoalescedB == 1 else 0
      readsLatency = roundUp(readsLatencyA+readsLatencyB)*2
      self.numLocalWriteModPerMfma = max((self.miLatency - readsLatency)//(self.tPA["localWriteInstruction"].IssueLatency*2+1),1)
      ##################################
      numGlobalReadInsPerIter = numMfmaPerIter * self.numGlobalReadInsPerMfma
      numLocalWriteModPerIter = numMfmaPerIter * self.numLocalWriteModPerMfma
      # if numGlobalReadInsPerMfma>1, we still want to schedule only 1 GlobalReadIncCode per mfma
      # inserting empty CodeModule so that generator will schedule 1 GlobalReadIncCode 1 empty CodeModule if numGlobalReadInsPerMfma=2
      # TODO: break globalReadIncCode
      numEmptyGlobalReadIncCode = self.numGlobalReadInsPerMfma - 1
      # assign parameter
      # 1. we calculate number of mfma to prefetch localReads for next loop
      # 2. we put barrier 1 mfma ahead that
      # 3. we put last localWrite 1~2 mfma ahead barrier
      # localReads followed following sequence to be scheduled
      # ds_read[A][0], ds_read[B][0], ds_read[A][1:], ds_read[B][1:]
      self.numMfmaForNextLoopLR = 1
      latencyLeft = self.miLatency
      # ds_read[A][0]
      for i in range(self.numReadPerVectorA):
        latencyLeft -= tensorParametersA["localReadInstruction"].IssueLatency*2
        if latencyLeft < 0:
          self.numMfmaForNextLoopLR += 1
          latencyLeft = self.miLatency - tensorParametersA["localReadInstruction"].IssueLatency*2
      # ds_read[B][0]
      for i in range(self.numReadPerVectorB):
        latencyLeft -= tensorParametersB["localReadInstruction"].IssueLatency*2
        if latencyLeft < 0:
          self.numMfmaForNextLoopLR += 1
          latencyLeft = self.miLatency - tensorParametersB["localReadInstruction"].IssueLatency*2
      # ds_read[A][1:]
      for i in range(self.numReadsPerIterA-self.numReadPerVectorA):
        latencyLeft -= tensorParametersA["localReadInstruction"].IssueLatency*2
        if latencyLeft < 0:
          self.numMfmaForNextLoopLR += 1
          latencyLeft = self.miLatency - tensorParametersA["localReadInstruction"].IssueLatency*2
      # ds_read[B][1:]
      for i in range(self.numReadsPerIterB-self.numReadPerVectorB):
        latencyLeft -= tensorParametersB["localReadInstruction"].IssueLatency*2
        if latencyLeft < 0:
          self.numMfmaForNextLoopLR += 1
          latencyLeft = self.miLatency - tensorParametersB["localReadInstruction"].IssueLatency*2
      # to calculate number of mfma we need to wait before data arrive from lds to vgpr.
      # TODO: latency: 40 quad-cycle for 4 word, 22 quad-cycle for 2 word, 10 quad-cycle for 1 word
      latencyForLR = tensorParametersB["localReadInstruction"].IssueLatency*18
      latencyForLR -= latencyLeft # remaining latency in mfma
      latencyForLR -= (self.miLatency+self.miLatencyBuffer+2) # last LR will have 1 mfma latency
      while latencyForLR > 0:
        latencyForLR -= (self.miLatency+self.miLatencyBuffer+2)
        self.numMfmaForNextLoopLR += 1
      # final index definition
      self.numMfmaForNextLoopLR = min(self.numMfmaForNextLoopLR,numMfmaPerIter-1)
      self.barrierMfmaIndex = numMfmaPerIter*(kernel["LoopIters"]-self.numItersPLR+1) - self.numMfmaForNextLoopLR - 1 if self.numItersPLR else 0
      self.nextLoopLRMfmaIndex = self.barrierMfmaIndex if self.numItersPLR else numMfmaPerIter*kernel["LoopIters"] - self.numMfmaForNextLoopLR - 1
      self.lwEndMfmaIndex = max(self.barrierMfmaIndex - numMfmaBetweenLWandBarrier,0) if self.numItersPLR else numMfmaPerIter*kernel["LoopIters"] - 1
      if kernel["DepthULdsDivisor"] > 1:
        # SplitLDS's schedule looks like: # (LW/GR) x N -> (GR address increment) x 2
        self.lwEndMfmaIndex = min(self.lwEndMfmaIndex + 2, numMfmaPerIter*kernel["LoopIters"] - 1)
      localWriteEndIter = self.lwEndMfmaIndex//numMfmaPerIter
      localWriteEndIter = min(kernel["LoopIters"] - 1, localWriteEndIter)
      assert localWriteEndIter < kernel["LoopIters"]
      assert self.lwEndMfmaIndex < numMfmaPerIter*kernel["LoopIters"]
    else:
      numGlobalReadInsPerIter = 1
      numLocalWriteModPerIter = 1
      numEmptyGlobalReadIncCode = 0

    if not self.scheduleGlobalRead:
      # put everything in the header:
      self.unrollLoopHeaderCode.addCode(self.dtlsM0UpdateACode)
      self.unrollLoopHeaderCode.addCode(self.globalReadACode)
      self.unrollLoopHeaderCode.addCode(self.dtlsM0UpdateBCode)
      self.unrollLoopHeaderCode.addCode(self.globalReadBCode)
      self.unrollLoopHeaderCode.addCode(globalReadIncACode)
      self.unrollLoopHeaderCode.addCode(globalReadIncBCode)
      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        self.grEndMfmaIndex = 0
      # TODO: if not schedule global read, just put everything in the first iteration
      # we can have more control to schedule global and sync at beginning of the loop
      # self.perIterGlobalReadCode[0].addCode(self.dtlsM0UpdateACode)
      # self.perIterGlobalReadCode[0].addCode(self.globalReadACode)
      # self.perIterGlobalReadCode[0].addCode(self.dtlsM0UpdateBCode)
      # self.perIterGlobalReadCode[0].addCode(self.globalReadBCode)
      # self.perIterGlobalReadCode[0].addCode(globalReadIncACode)
      # self.perIterGlobalReadCode[0].addCode(globalReadIncBCode)
    else:
      self.unrollLoopHeaderCode.addCode(self.globalReadACode.header)
      self.unrollLoopHeaderCode.addCode(self.globalReadBCode.header)


      # Add all loads from middle as individual schedulable items
      # when using PGR2, put global read instruction right after corresponding localWrite instruction
      if kernel["PrefetchGlobalRead"] == 2 or kernel["DepthULdsDivisor"] > 1:
        itemsGRToSched =  []
        itemsGRToSchedLater = list(self.globalReadACode.middle.items()) + \
                         list(self.globalReadBCode.middle.items())
      else:
        itemsGRToSched =  list(self.globalReadACode.middle.items()) + \
                        list(self.globalReadBCode.middle.items())
        itemsGRToSchedLater = []

      itemsGRToSched.append(globalReadIncACode)
      for i in range(numEmptyGlobalReadIncCode):
        imod = Code.Module()
        imod.addInst("//","globalReadIncA scheduler placeholder")
        itemsGRToSched.append(imod)
      itemsGRToSched.append(globalReadIncBCode)
      for i in range(numEmptyGlobalReadIncCode):
        imod = Code.Module()
        imod.addInst("//","globalReadIncB scheduler placeholder")
        itemsGRToSched.append(imod)

      if kernel["DepthULdsDivisor"] > 1:
        itemsGRToSchedLater.extend(itemsGRToSched) # GR addr inc after GR code
        itemsGRToSched.clear()

      readCnt = len(itemsGRToSched)

      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        # Loop in PGR1: GlobalRead -> GlobalReadInc -> LocalWrite
        # but GlobalReadInc shouldn't block LocalWrite so we count them out
        # Loop in PGR2: GlobalReadInc -> LocalWrite/GlobalRead pair
        # since LocalWrite/GlobalRead pair depends on GlobalReadInc, we count in only GlobalReadInc
        filter = [Code.GlobalReadInst]
        if kernel["PrefetchGlobalRead"] == 2:
          filter = [Code.Inst]
        loadsToSched = sum(1 for item in itemsGRToSched if item.countTypeList(filter))
        self.grEndMfmaIndex = max(0, roundUp(loadsToSched/self.numGlobalReadInsPerMfma) - 1)
        if self.grEndMfmaIndex > self.lwEndMfmaIndex:
          firstStep = (numMfmaPerIter + (self.grEndMfmaIndex - self.lwEndMfmaIndex)) * self.numGlobalReadInsPerMfma
          self.grEndMfmaIndex = self.lwEndMfmaIndex
        else:
          firstStep = numMfmaPerIter * self.numGlobalReadInsPerMfma
        endIter = roundUp((readCnt + 2 + 2*numEmptyGlobalReadIncCode)/numMfmaPerIter)
        if endIter > kernel["LoopIters"]:
          endIter = kernel["LoopIters"]
      else:
        # reads and incs are scheduled in iters range(0..endIter)
        endIter = readCnt + 2
        if endIter > localWriteEndIter:
          # Front-load some of the buffer loads if we don't have enough loop iters:
          # could use a different/smarter algorithm to space out the loads?
          firstStep = endIter-(localWriteEndIter) + 1
          endIter = localWriteEndIter
        else:
          # schedule b2b for readCnt > 2 (True for bigger TT)
          firstStep = 1

      if schedDb & 0x1:
        print("makeSchedule-gr, readCnt=", readCnt, "firstStep=", firstStep, "endIter=", endIter)

      # insert dtlsM0UpdateACode dtlsM0UpdateBCode code
      if self.globalReadACode.middle.items():
        self.globalReadACode.middle.items()[0].items().insert(0,self.dtlsM0UpdateACode)
      if self.globalReadBCode.middle.items():
        self.globalReadBCode.middle.items()[0].items().insert(0,self.dtlsM0UpdateBCode)
      # append 'n' global load at a time
      # append global load(S) first 'number of global load(s) determined by  firstStep
      for item in itemsGRToSched[:firstStep]:
        self.perIterGlobalReadCode[0].addCode(item)
      itemsGRToSched = itemsGRToSched[firstStep:]
      for u in range(1, endIter):
        itemPerIter = 1 * numGlobalReadInsPerIter
        try:
          for item in itemsGRToSched[:itemPerIter]:
            self.perIterGlobalReadCode[u].addCode(item)
            lastLoadIter = u
          itemsGRToSched = itemsGRToSched[itemPerIter:]
        except IndexError:
          break # no code left to schedule

      # here is to avoid globalReadInc code not add in
      # TODO: globalReadInc scheduling should not be blocked by localWriteInst
      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        for item in itemsGRToSched:
          self.perIterGlobalReadCode[endIter-1].addCode(item)
      else:
        assert not itemsGRToSched # should have scheduled everything already

      self.perIterGlobalReadCode[endIter-1].addCode(self.globalReadACode.footer)
      self.perIterGlobalReadCode[endIter-1].addCode(self.globalReadBCode.footer)

    if kernel["1LDSBuffer"]:
      barrier = Code.Module()
      barrier.addComment0("1 LDS buffer: read-sync-write")
      barrier.addInst("s_waitcnt lgkmcnt(0)","")
      barrier.addInst("s_barrier","")
      if self.localWriteACode.items():
        self.localWriteACode.items()[0].items().insert(0,barrier)

    # Now schedule the writes:
    if not self.scheduleLocalWrite:
      # if no scheduleLocalWrite - just add writes to localWritelocalWriteEndIter
      # If PGR=0, writes have to be done immediately following the loads - no opportunity to schedule
      #   so don't add to schedule, these will be added separately and before the first iter
      if kernel["PrefetchGlobalRead"]:
        # do we need a module here? That would prevent these from being scheduled
        imod = self.perIterLocalWriteCode[localWriteEndIter].addCode(Code.Module())
        if self.enable["Wait"]:
          imod.addCode(
              self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, \
              "1wait for global read"))
        imod.addComment1("local write A")
        imod.addCode(self.localWriteACode)
        imod.addComment1("local write B")
        imod.addCode(self.localWriteBCode)
      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        self.lwStartMfmaIndex = self.lwEndMfmaIndex
    else:
      # create a plan:
      itemsLWToSched = list(self.localWriteACode.items()) + list(self.localWriteBCode.items())
      # piggyback each global read instruction after each local write instruction to maximize G2L buffer utilization
      if kernel["PrefetchGlobalRead"] == 2 or kernel["DepthULdsDivisor"]>1:
        itemsLWToSched = list(zip_longest(itemsLWToSched, itemsGRToSchedLater, fillvalue=Code.Module("Placeholder")))
      try:
        # count how many items to schedule (local writes as well as global reads)
        _filterType = [Code.LocalWriteInst, Code.GlobalReadInst]
        writesToSched = sum(1 for p, q in itemsLWToSched if p.countTypeList(_filterType) or q.countTypeList(_filterType))
      except TypeError: # itemsLWToSched not tuple -> no piggybacked GR code
        if 1:
          # This counts the number of modules which contain a ds_write
          # Scheduler below keeps all writes in the same module in same iteration
          # so this is better match to what it is trying to do
          writesToSched = sum(1 for item in itemsLWToSched if item.countType(Code.LocalWriteInst))
        else:
          # count the number of writes, this doesn't match how they are
          # scheduled so pushes writes up too far
          writesToSched = self.localWriteACode.countType(Code.LocalWriteInst) + \
                      self.localWriteBCode.countType(Code.LocalWriteInst)
      # assign schedule index
      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        self.lwStartMfmaIndex = self.lwEndMfmaIndex - max(1,roundUp(writesToSched/self.numLocalWriteModPerMfma)) + 1
        if self.lwStartMfmaIndex < self.grEndMfmaIndex:
          self.lwStartMfmaIndex = self.grEndMfmaIndex
        startIter = self.lwStartMfmaIndex//numMfmaPerIter
        assert startIter < localWriteEndIter+1 # startIter should be at or before the endIter
      else:
        startIter = localWriteEndIter - writesToSched + 1
        # - can't move a write past the load it depends on
        #   as a simplificaton, don't move writes past any loads
        if startIter < lastLoadIter:
          startIter = lastLoadIter

      if schedDb & 0x2:
        print ("gra=", self.globalReadACode.middle.prettyPrint())
        print ("lwa=", self.localWriteACode.prettyPrint())

        print ("grb=", self.globalReadBCode.middle.prettyPrint())
        print ("lwb=", self.localWriteBCode.prettyPrint())
      if schedDb & 0x1:
        print ("makeSchedule-lw: writesToSched=", writesToSched, "lastLoadIter=", lastLoadIter, \
              "startIter=", startIter, "localWriteEndIter=", localWriteEndIter)

      readsToWait = len(list(self.localWriteACode.items())) + len(list(self.localWriteBCode.items()))
      readsToWaitNGLL = readsToWait
      if self.scheduleGlobalRead:
        # Number of write blocks should match number of reads.
        # Note for TLU=0 cases we will have multiple writes/load - but these are all in same write module
        # So number of moules should match:
        if 0:
            if not kernel["DirectToLdsA"]:
              assert self.globalReadACode.middle.countType(Code.GlobalReadInst) == \
                  len(list(self.localWriteACode.items()))
            if not kernel["DirectToLdsB"]:
              assert self.globalReadBCode.middle.countType(Code.GlobalReadInst) == \
                  len(list(self.localWriteBCode.items()))
      # Code.printItemList(itemsLWToSched)
      for u in range(startIter, localWriteEndIter+1):
        if u==(localWriteEndIter):
          itemPerIter = len(itemsLWToSched) # schedule all remaining activity
        else:
          itemPerIter = 1 * numLocalWriteModPerIter
          # if localwrite is not multiple of numLocalWriteModPerIter, fill last iteration first.
          # make sure numLocalWriteModPerIter is enough to schedule localwrite
          # TODO: if numLocalWriteModPerIter is not enough to schedule localwrite, need smarter way to distribute localWrite
          if u == startIter and kernel["ScheduleIterAlg"] == 3:
            itemPerIter = (numMfmaPerIter - self.lwStartMfmaIndex%numMfmaPerIter)*self.numLocalWriteModPerMfma

        for item in itemsLWToSched[:itemPerIter]:
          # Use a module to ensure these pieces stay together in the sub-iter scheduler
          imod = Code.Module("LocalWriteMod%u"%u)
          imodNGLL = Code.Module("LocalWriteMod%u"%u)

          # Prepend a waitcnt if needed. List item could be a tuple of Code.Module's depending on whether
          # other code modules (e.g., GR codes) are piggybacked on LW module
          try:
            writesPerItem = item[0].countType(Code.LocalWriteInst)
          except TypeError: # itemsLWToSched not tuple -> no piggybacked GR code
            writesPerItem = item.countType(Code.LocalWriteInst)
          imod.addComment0("sched write - iter %u writesPerItem=%u"%(u,writesPerItem))
          imodNGLL.addComment0("sched write - iter %u writesPerItem=%u"%(u,writesPerItem))
          if writesPerItem:
            # if writesPerItem>1 this indicates multiple LocalWrites in the same module
            # this happens in some transpose cases.  Here the first write needs to wait
            # for the associated global read to finish, then the remaining writes can flow
            # TODO - can schedule these writes across iters, should figure this out above
            writesToSched = writesToSched - writesPerItem
            readsToWait = readsToWait - 1
            readsToWaitNGLL = readsToWaitNGLL - 1
            # TODO - gfx9 supports higher max VMCNT
            if 1:
              if uDu < kernel["DepthULdsDivisor"]-1:
                imod.addComment0("no wait vmcnt except for in the last subLdsLoop")
              else:
                imod.addCode(Code.WaitCnt(self.version, -1, min(maxVmcnt, readsToWait), \
                  "wait for global read before writing to local"))
                imodNGLL.addCode(Code.WaitCnt(self.version, -1, min(maxVmcnt, readsToWaitNGLL), \
                  "wait for global read before writing to local"))
            else:
              print("warning - scheduleLocalWrite adding conservative vmcnt(0)")
              imod.addCode(Code.WaitCnt(self.version, -1, 0, "conservative waitcnt"))
          try:
            imod.addCode(item[0]) # original LW code
            imod.addCode(item[1]) # GR code piggybacked alongside LW code
            readsToWait = readsToWait + item[1].countType(Code.GlobalReadInst) # GR instruction increments vmcnt
          except TypeError: # itemsLWToSched not tuple -> no piggybacked GR code
            imod.addCode(item)
          self.perIterLocalWriteCode[u].addCode(imod)
          try:
            imodNGLL.addCode(copy.deepcopy(item[0])) # original LW code
            #imodNGLL.addCode(copy.deepcopy(item[1])) # omit GR code piggybacked alongside LW code
          except TypeError: # itemsLWToSched not tuple -> no piggybacked GR code
            imodNGLL.addCode(copy.deepcopy(item))
          self.perIterLocalWriteCodeNGLL[u].addCode(imodNGLL)
        itemsLWToSched = itemsLWToSched[itemPerIter:]

      # should never run out of items to schedule
      assert not itemsLWToSched # should have scheduled everthing already


  ##############################################################################
  # Schedule work into the each unroll loop iteration
  # localReadCode is the local reads for this loop iteration
  #  (returned by localReadDo). The instructions in localReadCode
  #  will retain their relative order, but may be interleaved
  #  with instructions from otherCode.

  # globalReadCode is the 'other' buffer loads and addr increments
  # localWriteCode is the 'other' local writes
  #  to schedule in with the ds reads.  The instructons
  #  will retain their relative order, but may be interleaved
  #  with instructions from localReadCode.

  # pointerCode contains local pointer changes (if needed)
  # waitCode contains s_waitcnt before macs.
  #   - Cannot be "" or None
  #   - may be empty Module if not waiting is desired (perhaps for debug)
  #   - may be multiple instructions (ConservativeWaitCnt)
  #   - typically is a single Code.WaitCnt.  This routine will
  #     modify the lgkmcnt to account for any scheduling decisions.
  #     If this is not desired, add the waitCnt to pointerCode and
  #     set waitCode to an empty module
  # macIterCode contains the mac iters.  May be a macro call.
  #
  # returns: a Module with the combined, optimally scheduled
  #  localReadCode + otherCode
  ##############################################################################
  def makeSubIterSchedule(self, kernel, localReadCode, iteration, pointerLWCode, pointerLRCode, waitCode, macIterCode, \
      waitLWCode = Code.Module(), syncCode = Code.Module(), packCode = Code.Module()):

    iterCode = Code.Module()
    globalReadCode = copy.deepcopy(self.perIterGlobalReadCode[iteration])
    localWriteCode = self.perIterLocalWriteCode[iteration]
    isBarrier = kernel["LoopIters"] - self.numItersPLR
    hasLocalRead = localReadCode.countType(Code.LocalReadInst)
    # Default schedule is other, local reads, then local writes:
    if self.scheduleIterAlg==0:
      # simple schedule, just add the modules in-order
      iterCode.addCode(globalReadCode)
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      iterCode.addCode(localReadCode)
      iterCode.addCode(localWriteCode)
      iterCode.addCode(pointerLWCode)
      iterCode.addCode(pointerLRCode)
      iterCode.addCode(waitCode)
      iterCode.addCode(packCode)
      iterCode.addCode(macIterCode)
    elif self.scheduleIterAlg == 1:
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      #import pdb
      #pdb.set_trace()
      # simple algorithm - do half the reads first:
      readsToSchedule = localReadCode.countType(Code.LocalReadInst) / 2
      #localReadCode.prettyPrint()
      readItems = localReadCode.flatitems()
      while readItems:
        item = readItems.pop(0)
        #print "readsToSchedule=", readsToSchedule, "item=", item
        iterCode.addCode(item)
        readsThisItem = item.countType(Code.LocalReadInst)
        if readsThisItem:
          assert readsThisItem==1, "Scheduler assumes 1 read per item"
          readsToSchedule = readsToSchedule - 1
          if readsToSchedule == 0:
            break

      iterCode.addCode(globalReadCode)

      # add rest of the reads here
      for item in readItems:
        iterCode.addCode(item)

      #move down write to be the last
      iterCode.addCode(localWriteCode)
      # tack on the pointer and mac code:
      iterCode.addCode(pointerLWCode)
      iterCode.addCode(pointerLRCode)
      iterCode.addCode(waitCode)
      iterCode.addCode(packCode)
      iterCode.addCode(macIterCode)
    elif self.scheduleIterAlg == 2:
    # SIA2 use only 1 iteration and separate compute and fetch by raising compute priority
    # 2 workgroup interleave, while WG0/WG1 doing compute, WG1/WG0 doing fetch
    # EPS need to be 1, or valu instruction will break interleave
      iterCode.addCode(globalReadCode)
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      iterCode.addCode(localReadCode)
      iterCode.addCode(waitCode)

      # interleave pack code
      instPerPack = 1 if kernel["ProblemType"]["DataType"].isBFloat16() else 2
      packItems = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)) ]
        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Code.Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Code.Module()
        packBItems = packB.flatitems()
        if packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)):
          packItems += packINtems.pop(0)

      macIterItem = macIterCode.flatitems()
      # pop the first code which is s_nop 1 for packing
      item = macIterItem.pop(0)

      numMfmaPerIter = self.numMfmaPerIter
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0

      for i in range(numMfmaPerIter):
        if packItems:
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          packAIdx += instPerPack if i//(kernel["MIWaveTile"][0]+kernel["MIWaveTile"][0]*kernel["MIWaveTile"][1]*(i//(kernel["MIWaveTile"][0]*kernel["MIWaveTile"][1]))) == 0 else 0
          packBIdx += instPerPack if i % kernel["MIWaveTile"][0] == 0 else 0
          packAIdx = packAIdx if self.tPA["localReadInstruction"].blockWidth == 0.5 else 0
          packBIdx = packBIdx if self.tPB["localReadInstruction"].blockWidth == 0.5 else 0
          numPack = (packAIdx + packBIdx)
          iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))
          # we put 2 pack in each mfma
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPack):
                iterCode.addCode(packItems.pop(0))
                curPackIdx += 1
            else:
              iterCode.addInst("s_nop ","0","VALU packing writes to be consumed by matrix instruction")
              curPackIdx += 1
        if i == 0:
          if not packItems:
            tmpVgpr = self.vgprPool.checkOut(1)
            iterCode.addInst("v_mov_b32 ","v%u"%(tmpVgpr),"0x0","valu operation to have different priority")
            self.vgprPool.checkIn(tmpVgpr)
          iterCode.addInst("s_setprio ","3","Raise priority while processing macs")
        item = macIterItem.pop(0)
        iterCode.addCode(item)

      iterCode.addInst("s_setprio ","1","Raise priority while processing macs")
      iterCode.addCode(localWriteCode)
      iterCode.addCode(pointerLWCode)
      iterCode.addCode(pointerLRCode)
      iterCode.addInst("s_setprio ","2","Raise priority while processing macs")
      pass
    elif self.scheduleIterAlg == 3:
      iterCode.addComment0(" grEndMfmaIndex:%u, lwStartMfmaIndex:%u, lwEndMfmaIndex:%u " %(self.grEndMfmaIndex,self.lwStartMfmaIndex,self.lwEndMfmaIndex))
      iterCode.addComment0(" numMfmaForLR:%u, barrierMfmaIndex:%u " %(self.numMfmaForNextLoopLR,self.barrierMfmaIndex))
      #####
      # Prepare and Assign parameter
      ####
      numMfmaPerIter = self.numMfmaPerIter
      isBarrier = kernel["LoopIters"] - self.numItersPLR
      writeItems = list(localWriteCode.items())
      macIterItems = macIterCode.flatitems()
      readsPerIter = localReadCode.countType(Code.LocalReadInst)
      skipLocalWriteWaitcnt = 0
      localReadsWaitcnt = 0
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0

      #####
      # Prepare localReadCode
      ####
      localReadCodeAB = Code.Module()
      for iui in range(kernel["InnerUnroll"]):
        localReadCodeA = localReadCode.findNamedItem("LocalReadDoA_I%s"%(iui))
        localReadCodeB = localReadCode.findNamedItem("LocalReadDoB_I%s"%(iui))
        # In case localReadDo not generate localReadCode Module
        # and findNamedItem will return None type
        # TODO: findNamedItem return Code.Module() if not found
        if not localReadCodeA:
          localReadCodeA = Code.Module()
        if not localReadCodeB:
          localReadCodeB = Code.Module()
        if localReadCodeA.items():
          localReadCodeAB.addCode(localReadCodeA.items().pop(0))
        if localReadCodeB.items():
          localReadCodeAB.addCode(localReadCodeB.items().pop(0))
        while localReadCodeA.items():
          localReadCodeAB.addCode(localReadCodeA.items().pop(0))
        while localReadCodeB.items():
          localReadCodeAB.addCode(localReadCodeB.items().pop(0))
      localReadItems = localReadCodeAB.flatitems()
      localReadItemsThisLoop = localReadItems if iteration < isBarrier else []
      localReadItemsNextLoop = localReadItems if iteration >= isBarrier else []

      #####
      # Prepare pack Code                for A:
      # since the mfma reuse B first =>    for B: mfma[A][B]
      # we need 1 vector A and 1 vector B for first mfma
      # then we prepare remaining A, then remaining B
      ####
      instPerPack = 1 if kernel["ProblemType"]["DataType"].isBFloat16() else 2
      packItems = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)) ]
        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Code.Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Code.Module()
        packBItems = packB.flatitems()
        if packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)):
          packItems += packINtems.pop(0)

      # remove s_nop for packing
      # we will add s_nop if needed
      if macIterItems:
        macIterItems.pop(0)

      for i in range(numMfmaPerIter):
        mfmaIndex = iteration * numMfmaPerIter + i
        iterCode.addComment0(" mfmaIndex:%u " %(mfmaIndex))

        ####
        # scheduled local read
        ####
        readLeft = readsPerIter
        # with PrefetchLocalRead, localreads can interleave with mfma
        if self.numItersPLR:
          latencyLeft = self.miLatency
          # take ds_write into account to schedule ds_read, assume A and B localwrite have same width (TLDS=1)
          if (mfmaIndex >= self.lwStartMfmaIndex) and not globalReadCode.countType(Code.GlobalReadInst):
            for j in range(len(writeItems)):
              latencyLeft -= (self.tPA["localWriteInstruction"].IssueLatency*2+1)
          readLeftLROPT = 0
          for j in range(len(localReadItems)):
            latencyLeft -= localReadItems[j].IssueLatency*2
            readLeftLROPT += 1 if latencyLeft >= 0 else 0
          # evenly schedule localread with each mfma
          readLeftLREven = readsPerIter // numMfmaPerIter
          if (readsPerIter % (numMfmaPerIter)) > i:
            readLeftLREven += 1
          # we want no localreads at first and barrier mfma
          if (iteration == 0 or iteration == isBarrier):
            numMfmaForLR = numMfmaPerIter - 1
            if iteration == isBarrier:
              numMfmaForLR = self.numMfmaForNextLoopLR
            if i < numMfmaPerIter - numMfmaForLR or numMfmaPerIter == 1:
              readLeftLREven = 0
              readLeftLROPT = 0
            # rest mfma help to schedule those localReads
            else:
              readLeftLREven = readsPerIter // (numMfmaPerIter-1)
              if (readsPerIter % (numMfmaPerIter-1)) >= i:
                readLeftLREven += 1
          # if there are too many localreads, change strategy to even.
          readLeft = max(readLeftLREven,readLeftLROPT)
        # if start to schedule localwrite, but still have localreads not scheduled yet,
        # reject to use 1LDSB, since it will write and read same lds buffer at same time.
        # TODO: force to schedule all remaining localreads before start to schedule localwrite.
        if mfmaIndex >= self.lwStartMfmaIndex and mfmaIndex <= max(self.lwEndMfmaIndex,self.barrierMfmaIndex) and \
          localReadItemsThisLoop and kernel["1LDSBuffer"]:
          self.overflowedResources = 5
        while localReadItemsThisLoop:
          if readLeft == 0 and (i != numMfmaPerIter - 1):
            break
          item = localReadItemsThisLoop.pop(0)
          iterCode.addCode(item)
          readsThisItem = item.countType(Code.LocalReadInst)
          if readsThisItem:
            readLeft = readLeft - 1
            # because waitCode is scheduled at first mfma, we only need to skip localreads at first mfma.
            if (i == 0):
              localReadsWaitcnt += 1

        ####
        # scheduled global read
        ####
        for j in range(self.numGlobalReadInsPerMfma):
          if globalReadCode.items():
            iterCode.addCode(globalReadCode.items().pop(0))
        # schedule remaining globalReadInst
        if mfmaIndex == self.grEndMfmaIndex:
          while globalReadCode.items() and \
              (globalReadCode.countType(Code.GlobalReadInst) or kernel["PrefetchGlobalRead"] == 2):
            iterCode.addCode(globalReadCode.items().pop(0))
        # schedule remaining globalReadIncInst
        if i == numMfmaPerIter - 1:
          while globalReadCode.items():
            iterCode.addCode(globalReadCode.items().pop(0))

        ####
        # scheduled local write
        ####
        if (mfmaIndex >= self.lwStartMfmaIndex):
          for j in range(self.numLocalWriteModPerMfma):
            # in case there are localWrite and globalread in same iteration
            # we need to make sure globalRead before localWrite
            if writeItems and not globalReadCode.countType(Code.GlobalReadInst):
              writeItem = writeItems.pop(0)
              iterCode.addCode(writeItem)
              # if there is localWrite at first mfma, need to skip it in waitcnt.
              if i == 0:
                skipLocalWriteWaitcnt += writeItem.countType(Code.LocalWriteInst)
              if not localReadItemsThisLoop:
                self.perIterLocalWriteCanSkip[iteration] += writeItem.countType(Code.LocalWriteInst)
        if mfmaIndex == self.lwEndMfmaIndex:
          while writeItems:
            writeItem = writeItems.pop(0)
            iterCode.addCode(writeItem)
            if i == 0:
              skipLocalWriteWaitcnt += writeItem.countType(Code.LocalWriteInst)
            if not localReadItemsThisLoop:
              self.perIterLocalWriteCanSkip[iteration] += writeItem.countType(Code.LocalWriteInst)

        if mfmaIndex == self.lwEndMfmaIndex:
          iterCode.addCode(pointerLWCode)
        if i == numMfmaPerIter - 1:
          iterCode.addCode(pointerLRCode)

        ####
        # scheduled sync
        ####
        if mfmaIndex == self.barrierMfmaIndex and self.numItersPLR:
          iterCode.addCode(waitLWCode)
          iterCode.addCode(syncCode)

        ####
        # scheduled local read for next loop
        # localReads for next loop should after barrier
        ####
        while localReadItemsNextLoop:
          if readLeft == 0 and (i != numMfmaPerIter - 1):
            break
          item = localReadItemsNextLoop.pop(0)
          iterCode.addCode(item)
          readsThisItem = item.countType(Code.LocalReadInst)
          if readsThisItem:
            readLeft = readLeft - 1
            if (i == 0):
              localReadsWaitcnt += 1

        if i == 0:
          iterCode.addCode(waitCode)

        ####
        # scheduled pack
        ####
        if packItems:
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          packAIdx += instPerPack if i//(kernel["MIWaveTile"][0]+kernel["MIWaveTile"][0]*kernel["MIWaveTile"][1]*(i//(kernel["MIWaveTile"][0]*kernel["MIWaveTile"][1]))) == 0 else 0
          packBIdx += instPerPack if i % kernel["MIWaveTile"][0] == 0 else 0
          packAIdx = packAIdx if self.tPA["localReadInstruction"].blockWidth == 0.5 else 0
          packBIdx = packBIdx if self.tPB["localReadInstruction"].blockWidth == 0.5 else 0
          numPack = (packAIdx + packBIdx)
          iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))
          # we put 2 pack in each mfma
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPack):
                iterCode.addCode(packItems.pop(0))
                curPackIdx += 1
            else:
              iterCode.addInst("s_nop ","0","VALU packing writes to be consumed by matrix instruction")
              curPackIdx += 1
        if i == numMfmaPerIter - 1:
          while packItems:
            iterCode.addCode(packItems.pop(0))

        ####
        # scheduled mfma
        ####
        iterCode.addCode(macIterItems.pop(0) if macIterItems else Code.Module())
    else:
      assert 0, "Unsupported scheduleIterAlg=%u"%self.scheduleIterAlg

    if isinstance(waitCode, Code.WaitCnt):
      # Set the waitCount, based on the new iter schedule
      lgkmcnt = 0 # most conservative
      localReads = 0
      localWrites = 0
      if kernel["EnableMatrixInstruction"]:
        # dataAtIter      : the data we wait is read at which iteration
        # numReadsIter    : in this loop, number of iteration we have read (data used in current loop)
        dataAtIterA = iteration//self.numIterPerCoalescedReadA - self.numItersPLR
        dataAtIterB = iteration//self.numIterPerCoalescedReadB - self.numItersPLR
        numReadsIterA = min(iteration+1, kernel["LoopIters"]//self.numIterPerCoalescedReadA - self.numItersPLR)
        numReadsIterB = min(iteration+1, kernel["LoopIters"]//self.numIterPerCoalescedReadB - self.numItersPLR)
        skipReadsIterA = numReadsIterA - dataAtIterA - 1 if not dataAtIterA < max(dataAtIterA,dataAtIterB) else 0
        skipReadsIterB = numReadsIterB - dataAtIterB - 1 if not dataAtIterB < max(dataAtIterA,dataAtIterB) else 0
        # numPrefetchIter : in this loop, number of prefetch iteration we have read (data used in next loop)
        # currently we have localReadA and localReadB if iteration >= isBarrier
        # some case will not have localReads if PGR=0 or NoLoadLoop
        # known bug: wider localread + numItersPLR>1 may have chance to fail.
        numPrefetchIter = (iteration//(kernel["LoopIters"]-self.numItersPLR))*((iteration+1)-(kernel["LoopIters"]-self.numItersPLR)) if kernel["PrefetchGlobalRead"] else 0
        numPrefetchIter = 0 if iteration >= isBarrier and not hasLocalRead else numPrefetchIter
        skipReadsIterA += numPrefetchIter
        skipReadsIterB += numPrefetchIter
        # here the reads are prefetches so can skip them in the waitcnt
        # how many localreads can skip is based on how many iterations we prefetch.
        localReads += self.numReadsPerIterA * skipReadsIterA + localReads + self.numReadsPerIterB * skipReadsIterB
        # some of localReads is interleaved after waitcnt in SIA3
        if kernel["ScheduleIterAlg"] == 3 and \
          (iteration < numReadsIterA or iteration < numReadsIterB or numPrefetchIter) and \
          self.enable["LocalRead"]:
          if (iteration < numReadsIterA and not dataAtIterA < max(dataAtIterA,dataAtIterB)) or numPrefetchIter:
            localReads -= self.numReadsPerIterA
          if (iteration < numReadsIterB and not dataAtIterB < max(dataAtIterA,dataAtIterB)) or numPrefetchIter:
            localReads -= self.numReadsPerIterB
          localReads += localReadsWaitcnt
        lgkmcnt += localReads
        iterCode.addComment0("numPrefetchIter=%u" % numPrefetchIter)
        iterCode.addComment0("dataAtIterA=%u numReadsIterA=%u skipReadsIterA=%u readsPerIterA=%u" % (dataAtIterA, numReadsIterA, skipReadsIterA, self.numReadsPerIterA))
        iterCode.addComment0("dataAtIterB=%u numReadsIterB=%u skipReadsIterB=%u readsPerIterB=%u" % (dataAtIterB, numReadsIterB, skipReadsIterB, self.numReadsPerIterB))
        if kernel["ScheduleIterAlg"] == 0 or kernel["ScheduleIterAlg"] == 1:
          for i in range (max(dataAtIterA,dataAtIterB),iteration+1):
            localWrites += self.perIterLocalWriteCode[i].countType(Code.LocalWriteInst)
        # ScheduleIterAlg=2, localwrite is after waitCnt, no need to count it's current iteration.
        if kernel["ScheduleIterAlg"] == 3:
          for i in range (max(dataAtIterA,dataAtIterB)+1,iteration):
            localWrites += self.perIterLocalWriteCode[i].countType(Code.LocalWriteInst)
          if kernel["ScheduleLocalWrite"] > 0:
            # current iteration localWrite count
            localWrites += skipLocalWriteWaitcnt
            # dataAtIter iteration localWrite count
            if self.numItersPLR:
              skipPreIterLW = self.perIterLocalWriteCanSkip[max(dataAtIterA,dataAtIterB)]
              localWrites += skipPreIterLW
        lgkmcnt += localWrites
      else:
        for item in list(iterCode.items()):
          localReads  = item.countType(Code.LocalReadInst)
          localWrites = item.countType(Code.LocalWriteInst)
          if self.numVgprBuffer:
            # here the reads are prefetches so can skip them in the waitcnt
            lgkmcnt += localReads
            # and the writes are targetting another section of LDS and are
            # synchronized through a different waitnct than this one
            # (which is always just before the macs)
            lgkmcnt += localWrites
          else:
            # if UnrollLoopEfficiencyEnable == True  use waitCode passed lgkmCnt
            # else:
            # we need to wait for all preceding reads before the macs
            # so only opportunity for optimization is if the writes are at the end
            if globalParameters["UnrollLoopEfficiencyEnable"]:
              lgkmcnt = waitCode.lgkmcnt
            else:
              if localReads:
                lgkmcnt = 0 # reset to wait for all reads
              else:
                lgkmcnt = localWrites  # this only survives if writes are at the end

      lgkmcnt = min(lgkmcnt, 15)
      waitCode.comment += " old=%u, new=%u newLW=%u newLR=%u" % (waitCode.lgkmcnt, lgkmcnt,localWrites,localReads)
      waitCode.lgkmcnt = lgkmcnt

    return iterCode

  ##############################################################################
  # returns list of modules or text
  # papIter indicates this is the setup for the "prefetchAcrossPersistent"
  # (aka pap) iteration
  ##############################################################################
  def setupNewTile(self, kernel, tensorParametersA, tensorParametersB, isPap, isOptNLL=False):
    kl = []

    if self.enable["PreLoop"]:
      ####################################
      # Global Read Addresses
      ####################################
      kl.append(self.comment3("Begin setupNewTile, isPap=%s") % isPap)

      # work-group assignments
      kl.append(self.comment("global read addresses: work-group"))
      kl.append(self.graWorkGroup(kernel, isPap))

      needShift = False
      if (kernel["EdgeType"] == "ShiftPtr") and \
         (not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"])) or \
         (not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialB"])):
        needShift = True

      # some case (PAP), we don't have to append the code for duplicated calculation
      # only those calculation related to WorkGroupID need to be generated. otherwise it's just redundant
      # default dontAppendCode = False, means need to append code
      self.dontAppendCode = False

      # 1. during isPap, this is actually no needed, so we can skip this.
      #    but since there are some vgpr value is used in the later lwaFirstOffset (when not OptNLL, such as "lwoT")
      #    so we still do this part when "isPap & not OptNLL"
      # 2. if tile edge, then we still need to add all these codes even isPap
      self.dontAppendCode = isPap and isOptNLL and (not needShift)
      # tile assignments
      kl.append(self.comment("global read addresses: tile offset assignment a"))
      kl.append(self.graTileAssignment(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: tile offset assignment b"))
      kl.append(self.graTileAssignment(kernel, tensorParametersB))

      self.dontAppendCode = isPap and (not needShift)
      # unroll assignments
      kl.append(self.comment("global read addresses: unroll assignment a"))
      kl.append(self.graUnrollAssignment(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: unroll assignment b"))
      kl.append(self.graUnrollAssignment(kernel, tensorParametersB))
      self.dontAppendCode = False

      # other free indices
      if kernel["ProblemType"]["NumIndicesC"] > 2:
        kl.append(self.comment("global read addresses: other free assignments"))
        kl.append(self.graOtherFreeAssignments(kernel))

      # other summation indices
      if self.otherSummations:
        kl.append(self.comment("global read addresses: other summation assignments"))
        kl.append(self.graOtherSummationAssignments(kernel))

      self.dontAppendCode = isPap and (not needShift)
      # tile offsets
      kl.append(self.comment("global read addresses: tile offsets a"))
      kl.append(self.graTileOffsets(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: tile offsets b"))
      kl.append(self.graTileOffsets(kernel, tensorParametersB))

      # unroll offsets
      kl.append(self.comment("global read addresses: unroll offsets a"))
      kl.append(self.graUnrollOffsets(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: unroll offsets b"))
      kl.append(self.graUnrollOffsets(kernel, tensorParametersB))
      self.dontAppendCode = False

      # tile edges
      if kernel["EdgeType"] == "ShiftPtr":
        # Shift here has two purposes:
        #  1. Ensure the loads are in-bounds to prevent fault.
        #     BufferLoad uses the buffer limit hardware and does not require bounds checking for this case
        #  2. Shift-left a wide vector load to ensure it is completely in-bounds.
        #     If this occurs we need to 'unshift' the C values (see shiftVectorComponents)
        #     BufferLoad does support this shifting, but if GuaranteeNoPartial=1 then
        #     it can be guaranteed that no shifting is required.
        if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"]):
          kl.append(self.comment("global read addresses: shift a"))
          kl.append(self.graShift(kernel, tensorParametersA))
        if not (kernel["BufferLoad"] and  kernel["GuaranteeNoPartialB"]):
          kl.append(self.comment("global read addresses: shift b"))
          kl.append(self.graShift(kernel, tensorParametersB))
      elif kernel["EdgeType"] == "Branch":
        kl.append(self.comment("global read addresses: branch a"))
        kl.append(self.graBranch(kernel, tensorParametersA))
        kl.append(self.comment("global read addresses: branch b"))
        kl.append(self.graBranch(kernel, tensorParametersB))

      self.dontAppendCode = isPap and (not needShift)
      # final offsets
      kl.append(self.comment("global read addresses: final offsets a"))
      kl.append(self.graFinalOffsets(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: final offsets b"))
      kl.append(self.graFinalOffsets(kernel, tensorParametersB))
      self.dontAppendCode = False

      # addresses
      kl.append(self.comment("global read addresses: addresses a"))
      kl.append(self.graAddresses(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: addresses b"))
      kl.append(self.graAddresses(kernel, tensorParametersB))

      self.dontAppendCode = isPap
      # increments
      kl.append(self.comment("global read addresses: increments a"))
      for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
        kl.append(self.graIncrements(kernel, i, tensorParametersA))
      kl.append(self.comment("global read addresses: increments b"))
      for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
        kl.append(self.graIncrements(kernel, i, tensorParametersB))
      self.dontAppendCode = False

      ####################################
      # Local Write Addresses
      ####################################
      kl.append(self.comment3("Local Write Addresses"))

      # tile assignments
      kl.append(self.lwaTileAssignment(kernel, tensorParametersA))
      kl.append(self.lwaTileAssignment(kernel, tensorParametersB))

      # unroll assignments
      kl.append(self.lwaUnrollAssignment(kernel, tensorParametersA))
      kl.append(self.lwaUnrollAssignment(kernel, tensorParametersB))

      # if PAP, no need to reset LWA, but if not OptNLL, we still do this (due to TailLoop)
      self.dontAppendCode = isPap and isOptNLL
      # first offsets
      kl.append(self.comment("local write addresses: first offset a"))
      kl.append(self.lwaFirstOffset(kernel, tensorParametersA))
      kl.append(self.comment("local write addresses: first offset b"))
      kl.append(self.lwaFirstOffset(kernel, tensorParametersB))
      self.dontAppendCode = False

      # final offsets
      kl.append(self.lwaFinalOffsets(kernel, tensorParametersA))
      kl.append(self.lwaFinalOffsets(kernel, tensorParametersB))

      # declare addresses
      kl.append(self.lwaDeclareAddresses(kernel, tensorParametersA))
      kl.append(self.lwaDeclareAddresses(kernel, tensorParametersB))

      # init pointers
      kl.append(self.localWriteInitPointers(kernel, tensorParametersA))
      kl.append(self.localWriteInitPointers(kernel, tensorParametersB))

    ###########################################################################
    # summations loops: open
    ###########################################################################

    # declare loop num iter
    kl.append(self.comment1("declare loop num iterations"))
    kl.append(self.declareLoopNumIter(kernel))

    # perform initC in the shadow of the prefetch
    # Prefetch occurs at start of unroll loop
    # If we have multiple summation indices (otherSummationLoops>0),
    # we can't init in shadow of this prefetch
    # since that would initC inside the other summation loops

    if self.doShadowInit != 2:
      kl.append(self.initC(kernel))

    # open non-unrolled summation loops
    for i in range(kernel["ProblemType"]["NumIndicesSummation"]-1):
      kl.append(self.comment("summation loop %u"%i))
      kl.append(self.calculateLoopNumIter(kernel, i, isPap))
      if self.actualSummationLoops>1:
        kl.append(self.openLoop(kernel, i))
    kl.append(self.calculateLoopNumIter(kernel, self.unrollIdx, isPap))

    if self.staggerU:
      kl.append(self.declareStaggerParms(kernel))
      kl.append(self.calculateStagger(kernel, tensorParametersA))
      kl.append(self.calculateStagger(kernel, tensorParametersB))

    # isPap don't init the read pointers - we want to continue to use the double-buffer
    # LRO and LWA as assigned
    if self.enable["PreLoop"] and not isPap:
      # init lds read pointers before each unrolled loop
      kl.append(self.comment1("local read addresses: init pointers a"))
      kl.append(self.localReadInitPointers(kernel, tensorParametersA))
      kl.append(self.comment1("local read addresses: init pointers b"))
      kl.append(self.localReadInitPointers(kernel, tensorParametersB))

    if isPap and not isOptNLL:
      # init lds read pointers before each unrolled loop
      kl.append(self.comment1("local read addresses: reset offset a"))
      kl.append(self.localReadResetOffsets(kernel, tensorParametersA))
      kl.append(self.comment1("local read addresses: reset offset b"))
      kl.append(self.localReadResetOffsets(kernel, tensorParametersB))

    ####################################
    # prefetch: unrolled loop prefix
    ####################################
    if kernel["PrefetchGlobalRead"]:
      pfi = 1
      kl.append(self.comment("prefetch: global -> local"))
      kl.append(self.openSumAtLeastUnroll(kernel, prefetch=True, isPap=isPap, isOptNLL=isOptNLL))
      if isPap and isOptNLL:
        if self.enable["GlobalRead"]:
          self.dtlsM0UpdateACode = self.directToLdsM0Update(kernel, 0, tensorParametersA)
          self.globalReadACode = self.globalReadDo(kernel, 0, tensorParametersA)
          self.dtlsM0UpdateBCode = self.directToLdsM0Update(kernel, 0, tensorParametersB)
          self.globalReadBCode = self.globalReadDo(kernel, 0, tensorParametersB)
        else:
          self.dtlsM0UpdateACode = Code.StructuredModule()
          self.globalReadACode = Code.StructuredModule() # empty
          self.dtlsM0UpdateBCode = Code.StructuredModule()
          self.globalReadBCode = Code.StructuredModule() # empty

        if self.enable["GlobalReadInc"]:
          self.globalReadIncrements = self.globalReadIncrementAB(kernel, self.unrollIdx, pfi)
        else:
          self.globalReadIncrements = Code.Module()
          self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
          self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))

      else:
        if self.enable["GlobalRead"]:
          kl.append(str(self.directToLdsM0Update(kernel, 0, tensorParametersA)))
          kl.append(str(self.globalReadDo(kernel, 0, tensorParametersA)))
          kl.append(str(self.directToLdsM0Update(kernel, 0, tensorParametersB)))
          kl.append(str(self.globalReadDo(kernel, 0, tensorParametersB)))
        if self.enable["GlobalReadInc"]:
          kl.append(self.globalReadIncrementAB(kernel, self.unrollIdx, pfi))

    kl.append(self.comment3("End setupNewTile, isPap=%s") % isPap)

    return kl

  ##############################################################################
  # noLoadLoop
  # Create the no load loop (NLL)
  #
  # isOptNLL : the NLL is to be optimized for the alpha=1 and non-edge case
  ##############################################################################
  def noLoadLoop( self, kernel, tensorParametersA, tensorParametersB, isOptNLL, isNGLL, pack ):
    kl = []
    pflr     = self.numItersPLR
    localWriteEndIter = kernel["LoopIters"] - self.numItersPLR - 1

    if isNGLL:
      kl.append(self.comment3(" NoGlobalLoadLoop - Begin"))
      self.perIterLocalWriteCode = self.perIterLocalWriteCodeNGLL
      self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
    else:
      kl.append(self.comment3("%s NoLoadLoop - Begin") % ("Opt." if isOptNLL else "Ord."))
      self.dtlsM0UpdateACode = Code.StructuredModule()
      self.globalReadACode = Code.StructuredModule() # empty
      self.dtlsM0UpdateBCode = Code.StructuredModule()
      self.globalReadBCode = Code.StructuredModule() # empty
      self.globalReadIncrements = Code.Module()
      self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
      self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))
      self.localWriteACode = Code.Module()
      self.localWriteBCode = Code.Module()

    # the scheduled GlobalRead,Inc code of PAP is inside openSumAtLeastUnroll (if PAP=on)
    kl.append(self.openSumAtLeastUnroll(kernel, prefetch=False, isPap=False, isOptNLL=isOptNLL))

    if not self.numItersPLR:
      if self.enable["Wait"]:
        if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "10wait for global read"))
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "4wait for local write"))
      if self.enable["Sync"]:
        kl.append(self.syncThreads(kernel))

    for uIdx in range(0, kernel["LoopIters"]*kernel["DepthULdsDivisor"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      uDu = uIdx // kernel["LoopIters"] # uDu: index of compute loop
      isLastLoop = (uDu == kernel["DepthULdsDivisor"] -1 ) and not isNGLL
      if u == 0:
        if uDu > 0:
          if self.enable["GlobalRead"]:
            assert len(self.globalReadACode.items()) > 0 and len(self.globalReadBCode.items()) > 0 # already issued in first uDu
            self.globalReadACode = Code.StructuredModule() # empty
            self.globalReadBCode = Code.StructuredModule() # empty
          if self.enable["GlobalReadInc"]:
            self.globalReadIncrements = Code.Module() # empty
            self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
            self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))
        if not isLastLoop:
          self.localWriteACode = self.localWriteDo(kernel, tensorParametersA, (uDu+1)%kernel["DepthULdsDivisor"])  # local write in loopcnt N targets data for loopcnt N+1
          self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB, (uDu+1)%kernel["DepthULdsDivisor"])
        else:
          self.localWriteACode = Code.Module()
          self.localWriteBCode = Code.Module()

        # TODO schedule waitcnt/barrier in makeSubIterSchedule()
        if kernel["PrefetchGlobalRead"] and kernel["LoopIters"] in [1, 2] and uDu > 0:
          if self.enable["Wait"]:
            kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "wait for local write"))
          if self.enable["Sync"]:
            kl.append(self.syncThreads(kernel, "sync for local read after write"))

        if not isNGLL:
          # PAP would have GlobalRead and GlobalInc, but no localWrite
          # Get the perIterGlobalReadCode code for PAP (if PAP=On), else would be empty
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu)
          kl.append(str(self.unrollLoopHeaderCode))

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = (u == localWriteEndIter)
      isSwapAndResetLwoIter = isResetLroIter
      isSwapLroIter = isResetLroIter
      if kernel["ScheduleIterAlg"] == 3:
          # isSwapLroIter = (u == self.nextLoopLRMfmaIndex//(self.numMfmaPerIter))
          isSwapAndResetLwoIter = (u == self.lwEndMfmaIndex//(self.numMfmaPerIter))

      extraComment = ""
      if isLastLoop:
        extraComment += " (last unrolled loop)"
      else:
        if isResetLroIter:
            extraComment += " (reset local read pointers iteration) "
        if isSwapAndResetLwoIter:
            extraComment += " (swap and reset local write pointers iteration) "
        if isSwapLroIter:
            extraComment += " (swap local read pointers iteration) "

      kl.append(self.comment("iter %u%s"%(u,extraComment)))
      plrIdx = ((u+pflr) % (self.numVgprBuffer+1)) % kernel["LoopIters"]
      localReads = Code.Module()

      pointerLWCode = Code.Module()
      pointerLRCode = Code.Module()
      waitCode = Code.Module()  # may be overwritten (not added to) below
      macIterCode = Code.Module()
      waitLWCode = Code.Module()
      syncCode = Code.Module()

      if self.enable["LocalRead"]:
        hasLiveLdsData = kernel["PrefetchGlobalRead"] or (uDu < kernel["DepthULdsDivisor"]-1)
        hasLiveLdsData = hasLiveLdsData and not isLastLoop
        # reads for current loop are done in previous iteration because of wider local read
        doReadA = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadA - self.numItersPLR)
        doReadB = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadB - self.numItersPLR)
        # reads for next loop
        doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
        doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
        for iui in range(0,kernel["InnerUnroll"]):
          doReadA = doReadA and iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]
          doReadB = doReadB and iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]
          if doReadA:
            localReads.addText(self.comment("local read a"))
            localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
            localReads.addCode(localReadCodeA)
            pack[plrIdx*self.numIterPerCoalescedReadA].addCode(packCodeA)
          if doReadB:
            localReads.addText(self.comment("local read b"))
            localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
            localReads.addCode(localReadCodeB)
            pack[plrIdx*self.numIterPerCoalescedReadB].addCode(packCodeB)
          if (not isResetLroIter or iui != kernel["InnerUnroll"]-1):
            if doReadA:
              localReads.addText(self.comment("local read increment a"))
              localReads.addText(self.localReadInc(kernel, iui, tensorParametersA))
            if doReadB:
              localReads.addText(self.comment("local read increment b"))
              localReads.addText(self.localReadInc(kernel, iui, tensorParametersB))

      if not isLastLoop:
        if kernel["PrefetchGlobalRead"]:
          # put barrier at localWriteEndIter+1
          if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
            if self.enable["Wait"]:
              waitLWCode.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
            if self.enable["Sync"]:
              syncCode.addCode(self.syncThreads(kernel))

          if isSwapAndResetLwoIter: # ResetLroIter
            if self.enable["LocalWrite"]:
              # local write for next iter, used to have local writes here
              pointerLWCode.addText(self.comment("local write swap offsets a"))
              pointerLWCode.addText(self.localWriteSwapOffsets(kernel, tensorParametersA))
              pointerLWCode.addText(self.comment("local write swap offsets b"))
              pointerLWCode.addText(self.localWriteSwapOffsets(kernel, tensorParametersB))
              pointerLWCode.addText(self.localWriteInitPointers(kernel, tensorParametersA))
              pointerLWCode.addText(self.localWriteInitPointers(kernel, tensorParametersB))

          if isSwapLroIter: # ResetLroIter
            if self.enable["LocalRead"]:
              # Swap, reset, or increment the LRO:
              pointerLRCode.addText(self.comment("local read swap offsets a"))
              pointerLRCode.addText(self.localReadSwapOffsets(kernel, kernel["ExpandPointerSwap"], tensorParametersA))
              pointerLRCode.addText(self.comment("local read swap offsets b"))
              pointerLRCode.addText(self.localReadSwapOffsets(kernel, kernel["ExpandPointerSwap"], tensorParametersB))

        if isResetLroIter: # ResetLroIter
          if self.enable["LocalRead"]:
            pointerLRCode.addText(self.comment("local read init pointers a"))
            pointerLRCode.addText(self.localReadInitPointers(kernel, tensorParametersA))
            pointerLRCode.addText(self.comment("local read init pointers b"))
            pointerLRCode.addText(self.localReadInitPointers(kernel, tensorParametersB))

      if self.enable["Wait"]:
        dataAtIter = u - self.numItersPLR
        numReadsIter = min(u+1,kernel["LoopIters"] - self.numItersPLR)
        skipReadsIter = numReadsIter - dataAtIter - 1
        waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, \
            skipReadsIter, \
            "7wait for local read")
      luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
      if self.enable["MAC"]:
        if kernel["EnableMatrixInstruction"]:
          macIterCode.addCode(self.mfmaIter(kernel, luIdx, kernel["InnerUnroll"]))
        else:
          macIterCode.addCode(self.macIter(kernel, luIdx, kernel["InnerUnroll"], True ))

      subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                      u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx])
      kl.append(subIterCode)
      for item in list(pack[luIdx].items()):
        if item.tempVgpr != None:
          self.vgprPool.checkIn(item.tempVgpr)
          item.tempVgpr = None
      pack[luIdx] = Code.Module()

    kl.append(self.closeSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=isOptNLL, isNGLL=isNGLL))

    return kl

  ##############################################################################
  # Kernel Body
  ##############################################################################
  def kernelBody( self, kernel, tensorParametersA, tensorParametersB ):

    ####################################
    # Begin String
    kl = []
    kl.append(self.openString(kernel))

    ####################################
    # Function Prefix
    kl.append(self.comment3("Function Prefix"))
    kl.append(self.functionPrefix(kernel))

    ####################################
    # Function Signature
    ####################################
    kl.append(self.comment3("Begin Kernel"))
    kl.append(self.functionSignaturePrefix(kernel))

    beforeFunctionSignature = '\n'.join([str(x) for x in kl])
    kl = []

    kl.append(self.functionSignatureSuffix(kernel))
    kl.append(self.functionBegin(kernel))

    kl.append(self.comment3("Allocate Resources"))
    kl.append(self.allocateResources(kernel))

    if self.enable["PreLoop"]:
      ####################################
      # Local Read Addresses
      ####################################
      kl.append(self.comment3("Local Read Addresses"))

      # tile assignments
      kl.append(self.comment("local read addresses: tile assignments a"))
      kl.append(self.lraTileAssignmentA(kernel, tensorParametersA))
      kl.append(self.comment("local read addresses: tile assignments b"))
      kl.append(self.lraTileAssignmentB(kernel, tensorParametersB))


      # final offsets
      kl.append(self.comment("local read addresses: final offsets a"))
      kl.append(self.lraFinalOffset(kernel, tensorParametersA))
      kl.append(self.comment("local read addresses: final offsets b"))
      kl.append(self.lraFinalOffset(kernel, tensorParametersB))

      # declare addresses
      kl.append(self.comment("local read addresses: declare addresses a"))
      kl.append(self.lraDeclareAddresses(kernel, tensorParametersA))
      kl.append(self.comment("local read addresses: declare addresses b"))
      kl.append(self.lraDeclareAddresses(kernel, tensorParametersB))

    # doShadowInit perfoms initialization in the 'shadow' of the global mem prefetch
    self.doShadowInit = 0
    if kernel["PrefetchGlobalRead"]:
      if self.actualSummationLoops == 1:
        self.doShadowInit = 2 # 2 is both store setup and initC
      else:
        # can't do shadow initC with multiple summation since this resets the ValuC counters
        # on each unroll iteration.
        self.doShadowInit = 1 # 1 is just store setup

    # for PersistentKernel with HPA mode, we can do the alpha, beta conversion f16->f32 only once outside the PK-loop
    if kernel["PersistentKernel"]:
      kl.append( self.checkAlphaBetaForHPA(kernel))

    if self.prefetchAcrossPersistent:
      # first prefetch is outside persistent loop, subsequent prefetch will
      # be integrated into no-load-loop
      kl += self.setupNewTile(kernel, tensorParametersA, tensorParametersB, isPap=False, isOptNLL=False)
      kl.append(self.openPersistentLoop(kernel))
    else:
      # prefetch is inside persistent loop
      kl.append(self.openPersistentLoop(kernel))
      kl += self.setupNewTile(kernel, tensorParametersA, tensorParametersB, isPap=False, isOptNLL=False)

    pack = [ Code.Module() for i in range (self.numVgprBuffer+1) ]

    if kernel["PrefetchGlobalRead"]:
      pkWithStoreRemap = kernel["PersistentKernel"] and kernel["StoreRemapVectorWidth"]
      if self.doShadowInit:
        kl.append(self.openShadowInit(kernel))
        kl.append(self.globalWriteWorkGroupInit(kernel))
        if self.doShadowInit == 2:
          kl.append(self.initC(kernel)) # initC while waiting for global reads
        kl.append(self.closeShadowInit(kernel))
      if self.enable["Wait"]:
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "8wait for global read"))
        # These cases loop back and run the prefetch loop again
        # we need an extra barrier to ensure that the ds_reads from previous iteration
        # have finished before we generate the prefetch for the next summation index.
        if pkWithStoreRemap or self.actualSummationLoops>1:
          kl.append( self.indent + self.syncStr + "// for PersistentKernel with SRVW " + self.endLine )


      if self.enable["LocalWrite"]:
        # local write
        kl.append(self.comment("local write a"))
        kl.append(self.localWriteDo(kernel, tensorParametersA, 0))
        kl.append(self.comment("local write b"))
        kl.append(self.localWriteDo(kernel, tensorParametersB, 0))
        # swap local ptrs
        kl.append(self.comment("local write swap a"))
        kl.append(self.localWriteSwapOffsets(kernel, tensorParametersA))
        kl.append(self.comment("local write swap b"))
        kl.append(self.localWriteSwapOffsets(kernel, tensorParametersB))
        kl.append(self.localWriteInitPointers(kernel, tensorParametersA))
        kl.append(self.localWriteInitPointers(kernel, tensorParametersB))

      if kernel["PrefetchGlobalRead"] == 2:
        kl.append(self.openPrefetchGlobalRead2(kernel))
        if self.enable["GlobalRead"]:
          kl.append(str(self.directToLdsM0Update(kernel, 0, tensorParametersA)))
          kl.append(str(self.globalReadDo(kernel, 0, tensorParametersA)))
          kl.append(str(self.directToLdsM0Update(kernel, 0, tensorParametersB)))
          kl.append(str(self.globalReadDo(kernel, 0, tensorParametersB)))
        kl.append(self.closePrefetchGlobalRead2(kernel))

      # prefetch-local
      if self.numItersPLR:
        if self.enable["Wait"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "0prefetch wait for local write"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel))

        # in some cases need an extra copy of the LDS read with appropriate double buffer offsets
        if self.enable["LocalRead"]:
          for plrIdx in range(0, self.numItersPLR):
            pack[plrIdx] = Code.Module()
            for espi in range(0, (self.prefetchAcrossPersistent and kernel["ExpandPointerSwap"])+1):
              for iui in range(0,kernel["InnerUnroll"]):
                if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]:
                  kl.append(self.comment("local read prefetch a"))
                  localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, espi, tensorParametersA)
                  kl.append(localReadCodeA)
                  pack[plrIdx].addCode(packCodeA)
                if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]:
                  kl.append(self.comment("local read prefetch b"))
                  localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, espi, tensorParametersB)
                  kl.append(localReadCodeB)
                  pack[plrIdx].addCode(packCodeB)
                if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]:
                  kl.append(self.comment("local read inc a"))
                  kl.append(self.localReadInc(kernel, iui, tensorParametersA))
                if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]:
                  kl.append(self.comment("local read inc b"))
                  kl.append(self.localReadInc(kernel, iui, tensorParametersB))
      kl.append(self.closeSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=False, isNGLL=False))

    # open unrolled summation loop
    kl.append(self.comment3("Unrolled Loop(s) - Begin"))
    kl.append(self.openLoop(kernel, self.unrollIdx))

    expand = kernel["ExpandPointerSwap"]
    loopCopies = 2 if expand else 1
    for lc in range(0, loopCopies):
      finalLoop = not expand or lc==loopCopies-1
      kl.append(self.comment3("Unrolled Loop %u/%u - Begin" % (lc+1, loopCopies)))
      kl.append(self.openLoopCopy(kernel, lc))
      if kernel["PrefetchGlobalRead"] and not self.numItersPLR and not kernel["ScheduleIterAlg"] == 2:
        if self.enable["Wait"]:
          if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
            kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "11wait for global read"))
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "1wait for local write"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel, "4sync for global read"))

      if self.enable["GlobalRead"]:
        # unrolled loop: global read A, B
        # M0 update for directToLds
        self.dtlsM0UpdateACode = self.directToLdsM0Update(kernel, 1, tensorParametersA)
        self.globalReadACode = self.globalReadDo(kernel, 1, tensorParametersA)
        self.dtlsM0UpdateBCode = self.directToLdsM0Update(kernel, 1, tensorParametersB)
        self.globalReadBCode = self.globalReadDo(kernel, 1, tensorParametersB)
      else:
        self.dtlsM0UpdateACode = Code.StructuredModule()
        self.globalReadACode = Code.StructuredModule() # empty
        self.dtlsM0UpdateBCode = Code.StructuredModule()
        self.globalReadBCode = Code.StructuredModule() # empty

      if self.enable["GlobalReadInc"]:
        # unrolled loop: increment global read addresses
        self.globalReadIncrements = self.globalReadIncrementAB(kernel, self.unrollIdx, 0)
      else:
        self.globalReadIncrements = Code.Module()
        self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
        self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))

      if self.enable["LocalWrite"]:
        self.localWriteACode = self.localWriteDo(kernel, tensorParametersA)
        self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB)
      else:
        self.localWriteACode = Code.Module()
        self.localWriteBCode = Code.Module()

      # localWriteEndIter is used to determine which iteration to put sync
      # if PGR=0, GR,LW,sync,LR will put at front of loop.
      localWriteEndIter = kernel["LoopIters"] - self.numItersPLR - 1

      # Schedule the global read, global read inc, and writes:
      unrollLoopHeaderCodeScheduled = False
      if not kernel["PrefetchGlobalRead"]:
        unrollLoopHeaderCodeScheduled = True
        self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter)
        kl.append(str(self.unrollLoopHeaderCode))

      # if not prefetch global, localWrite before mac's
      if not kernel["PrefetchGlobalRead"]:
        # unrolled loop: local write A, B
        if self.enable["Wait"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "5wait for global read"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel, "PGR=0, prior iter done reading lds"))
        if self.enable["LocalWrite"]:
          kl.append(self.comment("local write a"))
          kl.append(self.localWriteDo(kernel, tensorParametersA))
          kl.append(self.comment("local write b"))
          kl.append(self.localWriteDo(kernel, tensorParametersB))
        if self.enable["Wait"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel))
          # debug Local state
          """
          kl.append("    /* print Local state */" + self.endLine)
          kl.append("    for (unsigned int i = serial; i < LDS_NUM_ELEMENTS; i+=NUM_THREADS) {%s" % self.endLine)
          kl.append("      printf(\\\"localMemory[%%06u] = %%.0f\\\\n\\\", i, localMemory[i]);%s" )
              % self.endLine
          kl.append("    }" + self.endLine)
          """

      # unrolled loop: prefetch local
      if self.numItersPLR and not kernel["PrefetchGlobalRead"]:
        if self.enable["LocalRead"]:
          for plrIdx in range(0, self.numItersPLR):
            pack[plrIdx] = Code.Module()
            for iui in range(0,kernel["InnerUnroll"]):
              if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]:
                kl.append(self.comment("prefetch local a"))
                localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
                kl.append(localReadCodeA)
                pack[plrIdx].addCode(packCodeA)
              if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]:
                kl.append(self.comment("prefetch local b"))
                localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
                kl.append(localReadCodeB)
                pack[plrIdx].addCode(packCodeB)
              if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]:
                kl.append(self.comment1("local read increment a"))
                kl.append(self.localReadInc(kernel, iui, tensorParametersA))
              if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]:
                kl.append(self.comment1("local read increment b"))
                kl.append(self.localReadInc(kernel, iui, tensorParametersB))

      kl.append(self.closeString(kernel))
      kl.append(self.openString(kernel))

      pflr     = self.numItersPLR  # how many pf already done above

      ############################################################################
      # unrolled loop: mac iterations
      ############################################################################

      # double/quadruple the number of compute loop for each DepthU's worth of data read
      for uIdx in range(0, kernel["LoopIters"]*kernel["DepthULdsDivisor"]):
        u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
        uDu = uIdx // kernel["LoopIters"] # uDu: index of compute loop
        if u==0: # if at start of subloop...
          # ...do not issue global reads if not in first uDu
          if uDu > 0:
            if self.enable["GlobalRead"]:
              assert len(self.globalReadACode.items()) > 0 and len(self.globalReadBCode.items()) > 0 # already issued in first uDu
              self.globalReadACode = Code.StructuredModule() # empty
              self.globalReadBCode = Code.StructuredModule() # empty
            if self.enable["GlobalReadInc"]:
              self.globalReadIncrements = Code.Module()
              self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
              self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))

          # ...update local write code
          if self.enable["LocalWrite"]:
            self.localWriteACode = self.localWriteDo(kernel, tensorParametersA, (uDu+1)%kernel["DepthULdsDivisor"])  # local write in loopcnt N targets data for loopcnt N+1
            self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB, (uDu+1)%kernel["DepthULdsDivisor"])
          else:
            self.localWriteACode = Code.Module()
            self.localWriteBCode = Code.Module()

          # TODO schedule waitcnt/barrier in makeSubIterSchedule()
          if kernel["PrefetchGlobalRead"] and kernel["LoopIters"] in [1, 2] and uDu > 0:
            if self.enable["Wait"]:
              kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "wait for local write"))
            if self.enable["Sync"]:
              kl.append(self.syncThreads(kernel, "sync for local read after write"))

          if not unrollLoopHeaderCodeScheduled:
            self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu)
            kl.append(str(self.unrollLoopHeaderCode))

        # for PGR=0 where generator can't schedule the instructions (yet),
        # we duplicate the local write codegen and append to string list directly
        if not kernel["PrefetchGlobalRead"]:
          doWrite = False
          if uDu<kernel["DepthULdsDivisor"]-1 and u==kernel["LoopIters"]-self.numItersPLR:
            doWrite = True
            writeForNextLoop = 1
          if uDu>0 and self.numItersPLR==0 and u==0:
            assert doWrite==False # should be exclusive with the previous condition
            doWrite = True
            writeForNextLoop = 0
          # unrolled loop: local write A, B
          if doWrite:
            if self.enable["Wait"]:
              kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "5wait for local read"))
            if self.enable["Sync"]:
              kl.append(self.syncThreads(kernel, "PGR=0, prior iter done reading lds"))
            if self.enable["LocalWrite"]:
              kl.append(self.comment("local write a"))
              kl.append(self.localWriteDo(kernel, tensorParametersA, (uDu+writeForNextLoop)%kernel["DepthULdsDivisor"]))
              kl.append(self.comment("local write b"))
              kl.append(self.localWriteDo(kernel, tensorParametersB, (uDu+writeForNextLoop)%kernel["DepthULdsDivisor"]))
            if self.enable["Wait"]:
              kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
            if self.enable["Sync"]:
              kl.append(self.syncThreads(kernel))

        # which loop iteration to reset the LRO,
        # note if PLR=0, isResetLroIter is False for all u
        isResetLroIter = (u == localWriteEndIter)
        isSwapAndResetLwoIter = isResetLroIter
        isSwapLroIter = isResetLroIter
        if kernel["ScheduleIterAlg"] == 3:
          # isSwapLroIter = (u == self.nextLoopLRMfmaIndex//(self.numMfmaPerIter))
          isSwapAndResetLwoIter = (u == self.lwEndMfmaIndex//(self.numMfmaPerIter))
        extraComment = ""
        if isResetLroIter:
          extraComment += " (reset local read pointers iteration) "
        if isSwapAndResetLwoIter:
          extraComment += " (swap and reset local write pointers iteration) "
        if isSwapLroIter:
          extraComment += " (swap local read pointers iteration) "

        kl.append(self.comment("iter %u%s"%(u,extraComment)))
        plrIdx = ((u+pflr) % (self.numVgprBuffer+1)) % kernel["LoopIters"]

        localReads = Code.Module()
        localReadsA = Code.Module()
        localReadsB = Code.Module()

        pointerLWCode = Code.Module()
        pointerLRCode = Code.Module()
        waitCode = Code.Module()  # may be overwritten (not added to) below
        macIterCode = Code.Module()
        waitLWCode = Code.Module()
        syncCode = Code.Module()

        if self.enable["LocalRead"]:
          hasLiveLdsData = kernel["PrefetchGlobalRead"] or (uDu < kernel["DepthULdsDivisor"]-1)
          # reads for current loop are done in previous iteration because of wider local read
          doReadA = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadA - self.numItersPLR)
          doReadB = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadB - self.numItersPLR)
          # reads for next loop
          doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
          doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
          for iui in range(0,kernel["InnerUnroll"]):
            doReadA = doReadA and iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]
            doReadB = doReadB and iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]
            if doReadA:
              localReads.addText(self.comment("local read a"))
              localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
              localReads.addCode(localReadCodeA)
              localReadsA.addCode(localReadCodeA)
              pack[plrIdx*self.numIterPerCoalescedReadA].addCode(packCodeA)
            if doReadB:
              localReads.addText(self.comment("local read b"))
              localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
              localReads.addCode(localReadCodeB)
              localReadsB.addCode(localReadCodeB)
              pack[plrIdx*self.numIterPerCoalescedReadB].addCode(packCodeB)
            # Don't increment the LRO if we are going to reset them below:
            if not isResetLroIter or iui != kernel["InnerUnroll"]-1:
              if doReadA:
                localReads.addText(self.comment("local read increment a"))
                localReads.addText(self.localReadInc(kernel, iui, tensorParametersA))
              if doReadB:
                localReads.addText(self.comment("local read increment b"))
                localReads.addText(self.localReadInc(kernel, iui, tensorParametersB))

        if kernel["PrefetchGlobalRead"]:
          # put barrier at localWriteEndIter+1
          if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
            if self.enable["Wait"]:
              if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
                kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "12wait for global read"))
              waitLWCode.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
            if self.enable["Sync"]:
              syncCode.addCode(self.syncThreads(kernel))

          if isSwapAndResetLwoIter: # ResetLroIter
            if self.enable["LocalWrite"]:
              # local write for next iter, used to have local writes here
              pointerLWCode.addText(self.comment("local write swap offsets a"))
              pointerLWCode.addText(self.localWriteSwapOffsets(kernel, tensorParametersA))
              pointerLWCode.addText(self.comment("local write swap offsets b"))
              pointerLWCode.addText(self.localWriteSwapOffsets(kernel, tensorParametersB))
              pointerLWCode.addText(self.localWriteInitPointers(kernel, tensorParametersA))
              pointerLWCode.addText(self.localWriteInitPointers(kernel, tensorParametersB))

          if isSwapLroIter: # ResetLroIter
            if self.enable["LocalRead"]:
              # Swap, reset, or increment the LRO:
              pointerLRCode.addText(self.comment("local read swap offsets a"))
              pointerLRCode.addText(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
              pointerLRCode.addText(self.comment("local read swap offsets b"))
              pointerLRCode.addText(self.localReadSwapOffsets(kernel, expand, tensorParametersB))

        if isResetLroIter: # ResetLroIter
          if self.enable["LocalRead"]:
            pointerLRCode.addText(self.comment("local read init pointers a"))
            pointerLRCode.addText(self.localReadInitPointers(kernel, tensorParametersA))
            pointerLRCode.addText(self.comment("local read init pointers b"))
            pointerLRCode.addText(self.localReadInitPointers(kernel, tensorParametersB))

          waitGlobalRead = -1
          if kernel["PrefetchGlobalRead"] and isSwapAndResetLwoIter:
            waitLocalWrite = 1
          else:
            waitLocalWrite = -1

        else: # not isResetLroIter
          waitGlobalRead = 1 if u==0 and kernel["PrefetchGlobalRead"] and self.numVgprBuffer else -1
          waitLocalWrite = -1

        if self.enable["Wait"]:
          dataAtIter = u - self.numItersPLR
          numReadsIter = min(u+1,kernel["LoopIters"] - self.numItersPLR)
          numPrefetchIter = (u//(kernel["LoopIters"]-self.numItersPLR))*((u+1)-(kernel["LoopIters"]-self.numItersPLR)) if kernel["PrefetchGlobalRead"] else 0
          skipReadsIter = numReadsIter + numPrefetchIter - dataAtIter - 1
          waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, \
              waitGlobalRead, waitLocalWrite, skipReadsIter, \
              "wait for prior local read local write")

        luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
        if self.enable["MAC"]:
          if kernel["EnableMatrixInstruction"]:
            macIterCode.addCode(self.mfmaIter(kernel, luIdx, kernel["InnerUnroll"]))
          else:
            macIterCode.addCode(self.macIter(kernel, luIdx, kernel["InnerUnroll"], True ))

        ###### unroll loop efficiency implementation######################################
        # unroll loop efficiency implementation
        ## split A&B fetch&MAC code into mutiple groups
        ## splitting strategy   based on TT size
        ## 6x4 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
        ##        LDS fetch(es) into group of A{1-2)B(0) , A(3),B(1) (not implemented yet)
        ## 4x6 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
        ##        LDS fetch(es) into group of B{1-2)A(0) , B(3),A(1)
        ## 4x4 -> split into group of 8 and 8  MAC(s)
        ## 6x6 -> split into group of 12 MAC(s)
        ## 8x4/4x8 -> split into group of 16 and 16  MAC(s)
        ## 8x8 -> split into group of 16 MAC(s)
        ## supports only PLR=0
        ###############################################################################
        if self.numItersPLR or (not globalParameters["UnrollLoopEfficiencyEnable"]):
          subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                          u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx])
          kl.append(subIterCode) # add scheduled "other", local reads, local writes
          for item in list(pack[luIdx].items()):
            if item.tempVgpr != None:
              self.vgprPool.checkIn(item.tempVgpr)
              item.tempVgpr = None
          pack[luIdx] = Code.Module()
        else:
          macIterCode = Code.Module()
          MacitemsReorder = []
          if self.enable["MAC"]:
            luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
            macIterCode.addCode(self.macCode(kernel, luIdx, kernel["InnerUnroll"] ))
          MacIteritems = macIterCode.flatitems()
          #remove last and second entry from list if AggressiveMode is set
          # re-insert them back later
          if (kernel["AggressivePerfMode"]):
            MacIteritems = MacIteritems[:-1]
            MacIteritems.pop(1)
          #print("number MacItems\n",len(MacIteritems))
          waitGlobalRead = 1 if u==0 and kernel["PrefetchGlobalRead"] and self.numVgprBuffer else -1
          blockWidth = tensorParametersA["localReadInstruction"].blockWidth
          numVectorsPerTileA = (kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]]/kernel["VectorWidth"])
          numReadsPerVectorA = (kernel["VectorWidth"] * tensorParametersA["bpe"] ) / (blockWidth*4)
          numVectorsPerTileB = (kernel["ThreadTile%u"%tensorParametersB["tensorIdx"]]/kernel["VectorWidth"])
          TotalnumLdsFetches = numVectorsPerTileA*numReadsPerVectorA + numVectorsPerTileB*numReadsPerVectorA
          waitLocalWrite = -1
          ## Rules for applying kernel["UnrollLoopEfficiencyEnable"]
          ## if A+B fetches <= 3 no split approach
          if not TotalnumLdsFetches > 3:
            subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                         u, pointerLWCode, pointerLRCode, waitCode, macIterCode)
          else:
            if ((kernel["ThreadTile0"] == 6 and kernel["ThreadTile1"] == 4) or
               (kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 6)):
              numGroups = 2   #group0 = 8 MAC(s)  #group1 = 16 MAC(s) (6x4 - 4x2)
              # ldsItems for splitting lds(s)
              ldsItems = ([[4,2],[2,2]]) if kernel["ThreadTile0"] == 6 else ([[2,4],[2,2]])
              macItems = [8,16]
              waitCntItems = [0,0]
            elif (kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 4):
              numGroups = 2   #group0 = 8 MAC(s)  #group1 = 8  MAC(s) 2)
              ldsItems = ([[4,2],[0,2]])
              macItems = [8,8]
              waitCntItems = [0,0]
            elif (kernel["ThreadTile0"] == 6 and kernel["ThreadTile1"] == 6):
              numGroups = 2   #group0 = 8 MAC(s)  #group1 = 16 MAC(s) 2)
              ldsItems = ([[4,4],[2,2]])
              macItems = [16,20]
              waitCntItems = [0,0]
            elif ((kernel["ThreadTile0"] == 8 and kernel["ThreadTile1"] == 4) or
               (kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 8)):
              numGroups = 2   #group0 = 16 MAC(s)  #group1 = 16 MAC(s) 2)
              ldsItems = ([[4,4],[4,0]]) if kernel["ThreadTile0"] == 8 else ([[4,4],[0,4]])
              macItems = [16,16]
              waitCntItems = [0,0]
            elif (kernel["ThreadTile0"] == 8 and kernel["ThreadTile1"] == 8):
              numGroups = 2   #group0 = 8 MAC(s)  #group1 = 8 MAC(s) 2)
              #ldsItems = ([[4,4],[4,4]])
              macItems = [16,48]
              waitCntItems = [0,0]
            AitemsToReorder = localReadsA.flatitems()
            BitemsToReorder = localReadsB.flatitems()
            ##reorder code?? based on LDS fetch
            ## works for 2 groups.. needs fix for moer than 2 groups
            for iter in range(0,numGroups):
              endIdx   = ldsItems[iter][0] if iter == 0 else kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]]
              startIdx = 0  if iter == 0 else ldsItems[iter-1][1]
              for Bitems in range(startIdx, startIdx+ldsItems[iter][1]):
                for Aitems in range(0, endIdx):
                  idx = Aitems+(kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]]*Bitems)
                  MacitemsReorder.append(MacIteritems[idx])
                if (iter != 0):
                  for Bitems in range(0, ldsItems[iter-1][1]):
                    for Aitems in range(ldsItems[iter-1][0], kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]]):
                       MacitemsReorder.append(MacIteritems[Aitems+((kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]])*Bitems)])
            #print("Total number mac items A(%u)\n",len(MacitemsReorder))
            #print("Total number ds items A(%u)\n"%(TotalnumLdsFetches))
            #print("number ds items A_B(%u .. %u)\n"%(len(AitemsToReorder),len(BitemsToReorder)))
            #reorder LDS fetches so order in which A+B fetches matches MAC blob
            #e.g 8x4 original order in DGEMM case A[0-1]A[2-3]A[4-5]A[6-7]B[0-1][2-3]
            #we want to re-order them into A[0-1][2-3]B[0-1]B[2-3];  In all other except
            #DGEMM type, number of LDS fetches <=4 so no need for LDS re-order
            if self.enable["LocalRead"] and TotalnumLdsFetches > 4:
              localReads = Code.Module()
              for iter in range(0,numGroups):
                if len(AitemsToReorder):
                  localReads.addText(self.comment("local read a"))
                  numLocalReads = roundUp((ldsItems[iter][0])/kernel["VectorWidth"])
                  ##print("number ds items A(%u..%u)\n"%(iter,numLocalReads))
                  for idx in range(0,numLocalReads):
                    localReads.addCode(AitemsToReorder[0])
                    AitemsToReorder = AitemsToReorder[1:]
                if len(BitemsToReorder):
                  numLocalReads = roundUp(ldsItems[iter][1]/kernel["VectorWidth"])
                  ##print("number ds items B(%u..%u)\n"%(iter,numLocalReads))
                  localReads.addText(self.comment("local read b"))
                  for items in range(0,numLocalReads):
                    localReads.addCode(BitemsToReorder[0])
                    BitemsToReorder = BitemsToReorder[1:]
                if iter == 0:
                  waitCntItems[iter] = TotalnumLdsFetches - ((ldsItems[iter][0])/kernel["VectorWidth"] + (ldsItems[iter][1])/kernel["VectorWidth"])
                elif iter+1 != numGroups:
                  waitCntItems[iter] = TotalnumLdsFetches - ((ldsItems[iter][0])/kernel["VectorWidth"] + (ldsItems[iter][1])/kernel["VectorWidth"] + waitCntItems[iter-1])
                else:
                  waitCntItems[iter] = 0
                #print("Waitcnt(%u..%u)\n"%(iter,waitCntItems[iter]))
            for iter in range(0,numGroups):
              #Mac Code
              #place holder for future work Instruction class for generting MAC instruction
              #FMAInstruction = MacInstruction(globalParameters["CurrentISA"])
              subIterCode = Code.Module()
              waitCode = Code.Module()
              macIterCodeGrp = Code.Module()
              doOnce = False
              if self.enable["MAC"]:
                numMacItems = macItems[iter]
                for Items in range(0,numMacItems):
                  macItem = MacitemsReorder.pop(0)
                  macIterCodeGrp.addCode(macItem)
                  ## add s_setprio 1 when AggressivePerfMode ==1 as second instruction for second-last blob macCode
                  if (kernel["AggressivePerfMode"] and not doOnce):
                      macIterCodeGrp.addInst("s_setprio ","1","Raise priority while processing macs")
                      doOnce = True
                ## add s_setprio 0 when AggressivePerfMode ==1 as last instruction
                if (kernel["AggressivePerfMode"]):
                  macIterCodeGrp.addInst("s_setprio ","0","Reset priority after macs")
              #print("ReadWaitcnt(%u..%u)\n"%(iter,waitCntItems[iter]))
              #print("WriteCodeCount(%d..%u)\n",u,self.perIterLocalWriteCode[u].count())
              if (iter == 0):
                if self.enable["Wait"]:
                  #calculate lgkmcnt value including read+write for first iteration
                  waitCntVal = waitCntItems[iter] + 1 if (self.perIterLocalWriteCode[u].count()>0) else waitCntItems[iter]
                  # read + write instructions lgkmcnt (1=> for write)
                  # build waitCnt using new lgkmcnt
                  waitCode = Code.WaitCnt(self.version, waitCntVal,-1,"wait for prior local read")
                subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                         u, pointerLWCode, pointerLRCode, waitCode, macIterCodeGrp)
              else:
                  #last group only pointer + localWrite Code
                if self.enable["Wait"]:
                  waitCode = Code.WaitCnt(self.version, waitCntItems[iter],-1,"wait for prior local read & local writes")
                subIterCode.addCode(waitCode)
                subIterCode.addCode(macIterCodeGrp)
              kl.append(subIterCode) # add scheduled "other", local reads, local writes
      kl.append(self.closeString(kernel))
      kl.append(self.openString(kernel))

      # close unrolled loop
      if expand:
        if not finalLoop:
          kl.append(self.comment3("Unrolled Loop - End %u/%u"%(lc+1, loopCopies)))
        else:
          kl.append(self.comment3("Unrolled Loop - End %u/%u (final)"%(lc+1, loopCopies)))
      else:
        kl.append(self.comment3("Unrolled Loop - End"))

      kl.append(self.closeLoop(kernel, self.unrollIdx, finalLoop))

    if kernel["PrefetchGlobalRead"] == 2:
      kl += self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=True, pack=pack)

    # This "NoLoad" loop is a copy of the unroll loop but with global loads + LDS writes removed
    # doShadowInit is required since this pushes up the store SRD initialization before the NLL
    # OptNLL only allowed for single summation index  - for multiple summation we (currently)
    # execute the NLL inside each unroll iteration not just once at the end.
    if kernel["PrefetchGlobalRead"]:
      if not kernel["SuppressNoLoadLoop"]:

        if kernel["KernelLanguage"] == "Assembly" and kernel["OptNoLoadLoop"] and \
           kernel["BufferLoad"] and kernel["BufferStore"] and self.doShadowInit and \
           kernel["LocalSplitU"]==1 and kernel["GlobalSplitU"] == 1 and \
           self.actualSummationLoops==1:
          self.saveLocalPointers(kernel)

          # deepCopy packCode for OptNLL noLoadLoop
          deepCopyPack = copy.deepcopy(pack)
          kl += self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=True, isNGLL=False, pack=deepCopyPack)
          self.restoreLocalPointers(kernel)

        kl += self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isNGLL=False, pack=pack)
      # if PGR, last few iterations will have PLR,
      # and those PLR will not be used(register not checkIn) if without NoLoadLoop
      else:
        for i in range(self.numVgprBuffer):
          for item in list(pack[i].items()):
            if item.tempVgpr != None:
              self.vgprPool.checkIn(item.tempVgpr)
              item.tempVgpr = None

    if self.staggerU and self.actualSummationLoops>1:
      kl.append(self.comment("remove stagger offsets"))
      kl.append(self.removeStagger(kernel, tensorParametersA))
      kl.append(self.removeStagger(kernel, tensorParametersB))


    ########################################
    # Tail Loop
    # PackSummationDims=1 requires that the tile slice does not cross DepthU
    # which means tail loop not needed.
    ########################################
    self.inTailLoop = True
    if kernel["LoopTail"] and not kernel["PackSummationDims"]:
      kl.append(self.comment3("Tail Loop"))

      # Update local write pointers in case the upcoming global reads are writing directly to LDS:
      if self.enable["LocalWrite"]:
        if kernel["PrefetchGlobalRead"]:
          kl.append(self.comment("local write reset offsets a"))
          kl.append(self.localWriteResetOffsets(kernel, tensorParametersA))
          kl.append(self.comment("local write reset offsets b"))
          kl.append(self.localWriteResetOffsets(kernel, tensorParametersB))

      if self.enable["GlobalRead"]:
        # tail: global read
        kl.append(self.calculateLoopNumIter(kernel, -1, False))
        if self.staggerU and self.actualSummationLoops==1:
          kl.append(self.comment("remove stagger offsets for tail loop"))
          kl.append(self.removeStagger(kernel, tensorParametersA))
          kl.append(self.removeStagger(kernel, tensorParametersB))

        kl.append(self.comment("Update M0 for DTLDS"))
        kl.append(str(self.directToLdsM0Update(kernel, 1, tensorParametersA)))
        kl.append(self.comment("global read a"))
        kl.append(str(self.globalReadDo(kernel, 2, tensorParametersA)))
        kl.append(self.comment("Update M0 for DTLDS"))
        kl.append(str(self.directToLdsM0Update(kernel, 1, tensorParametersB)))
        kl.append(self.comment("global read b"))
        kl.append(str(self.globalReadDo(kernel, 2, tensorParametersB)))
      if self.enable["Wait"]:
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "2wait for global read"))
      if self.enable["Sync"]:
        kl.append(self.syncThreads(kernel))

      # the following read/write addresses could be modified in recalcLocal(Read|Write)Addresses due to policy change
      self.oriLraA = None # back up original local read address vgpr
      self.oriLraB = None
      self.oriLwaA = None # back up original local write address vgpr
      self.oriLwaB = None
      for uDu in range(0, kernel["DepthULdsDivisor"]):
        if kernel["DepthULdsDivisor"] > 1:
          # change local write poilcy from interleave-K to fractional as tail loop
          # iterate LDS read address one unit of K at a time
          kl.append(self.comment("Recalc local write offsets"))
          kl.append(self.recalcLocalWriteAddresses(kernel, tensorParametersA, uDu))
          kl.append(self.recalcLocalWriteAddresses(kernel, tensorParametersB, uDu))
        if self.enable["Sync"]:
          if uDu > 0:
            kl.append(self.comment("sync before local write"))
            kl.append(self.syncThreads(kernel))
        if self.enable["LocalWrite"]:
          # tail: local write
          kl.append(self.localWriteInitPointers(kernel, tensorParametersA))
          kl.append(self.localWriteInitPointers(kernel, tensorParametersB))
          kl.append(self.comment("local write a"))
          kl.append(self.localWriteDo(kernel, tensorParametersA, None))
          kl.append(self.comment("local write b"))
          kl.append(self.localWriteDo(kernel, tensorParametersB, None))
        # change local read policy from wider local read to one unit of K at a time
        kl.append(self.comment("Recalc local read offsets"))
        kl.append(self.recalcLocalReadAddressesAB(kernel))
        if self.enable["Wait"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "5wait for local write"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel))
        #kl.append(self.dumpLds(kernel, 0, 8))

        # tail: re-init local read addresses
        if kernel["PrefetchGlobalRead"]:
          kl.append(self.comment("local read reset offsets a"))
          kl.append(self.localReadResetOffsets(kernel, tensorParametersA))
          kl.append(self.comment("local read reset offsets b"))
          kl.append(self.localReadResetOffsets(kernel, tensorParametersB))
          kl.append(self.comment("local read init pointers a"))
          kl.append(self.localReadInitPointers(kernel, tensorParametersA))
          kl.append(self.comment("local read init pointers b"))
          kl.append(self.localReadInitPointers(kernel, tensorParametersB))
        # tail: macs
        kl.append(self.comment("tail loop: macs"))
        kl.append(self.openLoop(kernel, -1, uDu if kernel["DepthULdsDivisor"]>1 else None))

        # Try to use InnerUnroll in the tail loop if allowed:
        KinInnerUnroll = kernel["InnerUnroll"]
        if kernel["EnableMatrixInstruction"]:
          KinInnerUnroll *= kernel["MatrixInstK"]
        tailLoopInnerUnroll = kernel["InnerUnroll"] if (kernel["AssertSummationElementMultiple"] % KinInnerUnroll == 0) else 1

        pack[0] = Code.Module()
        for iui in range(0,tailLoopInnerUnroll):
          if self.enable["LocalRead"]:
            # Reading 16-bit data from LDS requires packing when ECC enabled
            kl.append(self.comment("local read a"))
            localReadCodeA, packCodeA = self.localReadDo(kernel, 0, iui, 0, tensorParametersA)
            kl.append(localReadCodeA)
            kl.append(self.comment("local read b"))
            localReadCodeB, packCodeB = self.localReadDo(kernel, 0, iui, 0, tensorParametersB)
            kl.append(localReadCodeB)
            pack[0].addCode(packCodeA)
            pack[0].addCode(packCodeB)

            kl.append(self.comment("local read inc a"))
            kl.append(self.localReadInc(kernel, iui, tensorParametersA))
            kl.append(self.comment("local read inc b"))
            kl.append(self.localReadInc(kernel, iui, tensorParametersB))
        if self.enable["Wait"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "4wait for local read"))

        if kernel["EnableMatrixInstruction"]:
          kl.append(pack[0])
          for item in list(pack[0].items()):
            if item.tempVgpr != None:
              self.vgprPool.checkIn(item.tempVgpr)
              item.tempVgpr = None
          pack[0] = Code.Module()
        if self.enable["MAC"]:
          if kernel["EnableMatrixInstruction"]:
            kl.append(self.mfmaIter(kernel, 0, tailLoopInnerUnroll, True))
          else:
            kl.append(self.macIter(kernel, 0, tailLoopInnerUnroll, True))

        kl.append(self.closeLoop(kernel, -1, True, uDu if kernel["DepthULdsDivisor"]>1 else None))
    # always emit the skip-tail-loop label
    kl.append(self.closeLoop(kernel, -1, None, emitEndLabelOnly=True))
    # tail: close
    self.inTailLoop = False

    # extra summation loops: global increment and close
    for i in reversed(range(self.otherSummationLoops)):
      kl.append(self.comment("global read inc AB"))
      kl.append(self.globalReadIncrementAB(kernel, i, 0))
      kl.append(self.closeLoop(kernel, i, True))

    if self.prefetchAcrossPersistent:
      kl.append(str(self.openPrefetchAcrossPersistent(kernel, isOptNLL=False)))
      kl += self.setupNewTile(kernel, self.tPA, self.tPB, isPap=True, isOptNLL=False)
      kl.append(str(self.closePrefetchAcrossPersistent(kernel, isOptNLL=False)))

    kl.append(self.endSummation(kernel))
    if self.enable["PostLoop"]:
      if not self.doShadowInit:
        kl.append(self.globalWriteWorkGroupInit(kernel))

      ####################################
      # Shift Vector Components
      ####################################
      if kernel["EdgeType"] == "ShiftPtr":
        # GuaranteeNoPartial means each component in the vector loads is always valid.  In this case we
        # don't need the unshift code

        # shift vector components d0
        if not kernel["GuaranteeNoPartialA"] and self.readTileDimVectorA:
          kl.append(self.comment("shift vector components d0"))
          kl.append(self.shiftVectorComponents(kernel, tensorParametersA))

        # shift vector components d1
        if not kernel["GuaranteeNoPartialB"] and self.readTileDimVectorB and not kernel["EnableMatrixInstruction"]:
          kl.append(self.comment("shift vector components d1"))
          kl.append(self.shiftVectorComponents(kernel, tensorParametersB))

      # complex declare tmp registers
      kl.append(self.complexDeclareTmpRegisters(kernel))

      ####################################
      # LocalSplitU reduction
      ####################################
      #if kernel["NumThreads"]%kernel["MacroTile0"] == 0:
      if kernel["LocalSplitU"] > 1:
        kl.append(self.comment3("LocalSplitU Reduction"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel))

        # LocalSplitU: local write
        kl.append(self.comment("LocalSplitU: local write"))
        kl.append(self.localSplitULocalWrite(kernel))

        # LocalSplitU: local read
        kl.append(self.comment("LocalSplitU: local read"))
        kl.append(self.localSplitULocalRead(kernel))

        # LocalSplitU: local read
        kl.append(self.comment("LocalSplitU: reduction"))
        kl.append(self.localSplitUReduction(kernel))

        # LocalSplitU: global write indices
        kl.append(self.comment("LocalSplitU: global write indices"))
        kl.append(self.localSplitUGlobalWriteIndices(kernel))

        # LocalSplitU: global write
        kl.append(self.comment("LocalSplitU: global write"))
        kl.append(self.localSplitUGlobalWrite(kernel))


      else:
        ####################################
        # NOT LocalSplitU
        ####################################

        # global write indices
        kl.append(self.comment("not-LocalSplitU: global write indices"))
        kl.append(self.notLocalSplitUGlobalWriteIndices(kernel))

        # global write
        kl.append(self.comment("not-LocalSplitU: global write"))
        kl.append(self.notLocalSplitUGlobalWrite(kernel))

    # function suffix
    kl.append(self.functionEnd(kernel, True))
    kl.append(self.functionSuffix(kernel))

    kl.append(self.closeString(kernel))
    kStr = '\n'.join([str(x) for x in kl])
    afterFunctionSignature = kStr

    error = self.overflowedResources

    # function signature last since it needs to know how many gprs were actually used
    kStr = beforeFunctionSignature + self.functionSignature(kernel) + afterFunctionSignature
    return (error,kStr)



  ##############################################################################
  #
  #   Functions to Write Kernel Segments
  #
  ##############################################################################

  def comment1(self, text):
    """
    single line comment
    """

    s = ""
    s += self.indent
    s += self.commentPrefix
    s += " %s " % text
    s += self.commentSuffix
    s += self.endLine
    return s

  def comment(self, text):
    """
    comment with prior newline
    """

    s = ""
    s += self.endLine
    s += self.comment1(text)
    return s

  def comment3(self, text):
    """
    3-line comment
    """

    s = ""
    s += self.endLine
    s += self.indent
    s += self.commentPrefix
    s += self.commentHR
    s += self.commentSuffix
    s += self.endLine

    for line in text.split("\n"):
      s += self.indent
      s += self.commentPrefix
      s += " %-38s " % line
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

    self.staggerU = kernel["StaggerU"] and (kernel["KernelLanguage"]=="Source" or kernel["BufferLoad"])
    self.tPA = tensorParametersA
    self.tPB = tensorParametersB

    # Only assembly supports scheduling
    self.canSchedule = (kernel["KernelLanguage"] == "Assembly")

    if self.canSchedule:
      self.scheduleGlobalRead = kernel["ScheduleGlobalRead"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"] # flat updates lgkmcnt counts = hard to schedule flat loads
    else:
      self.scheduleGlobalRead = 0

    if self.canSchedule:
      self.scheduleLocalWrite = kernel["ScheduleLocalWrite"] \
          and kernel["PrefetchGlobalRead"] \
          and kernel["BufferLoad"]  # flat updates lgkmcnt counts = hard to schedule writes and loads?
    else:
      self.scheduleLocalWrite = 0

    if self.canSchedule:
      self.scheduleIterAlg = kernel["ScheduleIterAlg"]
    else:
      self.scheduleIterAlg = 0

    self.prefetchAcrossPersistent = \
        kernel["KernelLanguage"] == "Assembly" and \
        kernel["PersistentKernel"] and \
        kernel["PrefetchGlobalRead"] and \
        not kernel["SuppressNoLoadLoop"] and \
        kernel["PrefetchAcrossPersistent"]


    self.actualSummationLoops = 1 if kernel["PackSummationDims"] else kernel["ProblemType"]["NumIndicesSummation"]
    self.otherSummationLoops  = self.actualSummationLoops-1
    self.otherSummations      = kernel["ProblemType"]["NumIndicesSummation"]-1 # not loops but summations vars

    # If 0, unroll loop is decremented by 1 each iteration and scaled by DEPTHU when number of summation elements
    # is required.
    # if 1, unroll loop starts at 0 and increments by DEPTHU.  No scaling is required.  This mode is required
    # for pack summation dims, but can also be used independently and this is useful for isolation and testing.
    self.unrollIncIsDepthU = kernel["UnrollIncIsDepthU"] or kernel["PackSummationDims"] \
                             or bool(kernel["ProblemType"]["ZeroPadA"]) or bool(kernel["ProblemType"]["ZeroPadB"])

    # turn on parts of prefetchAcrossPersistent code for testing
    self.prefetchAcrossPersistent0 = 0 or self.prefetchAcrossPersistent
    self.prefetchAcrossPersistent2 = 0 and self.prefetchAcrossPersistent

    self.enable = {}
    dkp = kernel["DisableKernelPieces"]
    # Can locally overrid these by changing True to False or
    # use the DisableKernelPieces for a quick search (see Common.py)
    self.enable["PreLoop"]        = True and not (dkp>0 and dkp >= 7) and not dkp == -7
    self.enable["GlobalRead"]     = True and not (dkp>0 and dkp >= 2) and not dkp == -2
    self.enable["GlobalReadInc"]  = True and not (dkp>0 and dkp >= 7) and not dkp == -7
    self.enable["LocalWrite"]     = True and not (dkp>0 and dkp >= 3) and not dkp == -3
    self.enable["LocalRead"]      = True and not (dkp>0 and dkp >= 4) and not dkp == -4
    self.enable["Wait"]           = True and not (dkp>0 and dkp >= 5) and not dkp == -5
    self.enable["Sync"]           = True and not (dkp>0 and dkp >= 5) and not dkp == -5
    self.enable["MAC"]            = True and not (dkp>0 and dkp >= 6) and not dkp == -6
    self.enable["PostLoop"]       = True and not (dkp>0 and dkp >= 1) and not dkp == -1

    #if dkp:
    #  print "\nKernelWriter enable:", self.enable

    if kernel["KernelLanguage"] == "Source":
      self.language = globalParameters["RuntimeLanguage"]
    else:
      self.language = "ASM"
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

    """
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
    """

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

    self.numItersPLR = kernel["PrefetchLocalRead"]%kernel["LoopIters"]
    self.numVgprBuffer = kernel["PrefetchLocalRead"]
    # merge N iteration's read into 1 iteration if can't coalesce read
    # ex, A can coalesce read, B can't
    # MergeRead 0: ds_readAx1 ds_readBx1 mfma | ds_readAx1 ds_readBx1 mfma | => ds_readAx2 ds_readBx1 mfma | ds_readBx1 mfma |
    # MergeRead 1: ds_readAx1 ds_readBx1 mfma | ds_readAx1 ds_readAx1 mfma | => ds_readAx2 ds_readBx1 ds_readBx1 mfma | mfma |
    MergeRead = 0
    self.lrvwA = kernel["LocalReadVectorWidth"] if not kernel["ProblemType"]["TLUA"] or MergeRead else kernel["ProblemType"]["DataType"].numMIInput()
    self.lrvwB = kernel["LocalReadVectorWidth"] if not kernel["ProblemType"]["TLUB"] or MergeRead else kernel["ProblemType"]["DataType"].numMIInput()

    # Wider LocalRead
    if kernel["EnableMatrixInstruction"]:
      self.numReadsIterCoalescedA = self.lrvwA//kernel["ProblemType"]["DataType"].numMIInput()
      self.numReadsIterCoalescedB = self.lrvwB//kernel["ProblemType"]["DataType"].numMIInput()
    else:
      self.numReadsIterCoalescedA  = 1
      self.numReadsIterCoalescedB  = 1

    self.numIterPerCoalescedReadA = max(1,self.numReadsIterCoalescedA//kernel["InnerUnroll"])
    self.numIterPerCoalescedReadB = max(1,self.numReadsIterCoalescedB//kernel["InnerUnroll"])

    if kernel["ScheduleIterAlg"] == 3 or kernel["ScheduleIterAlg"] == 2:
      self.numMfmaPerIter = kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1] * kernel["InnerUnroll"]
      if kernel["ProblemType"]["DataType"].isComplex(): self.numMfmaPerIter *= 4

    ########################################
    # read vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"]: # NT no transpose
      self.numReadsTileA = kernel["NumLoadsCoalescedA"]
      self.numReadsUnrollA = kernel["NumLoadsPerpendicularA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vectors
        self.readTileDimComponentsA = False # Vector
        self.readTileDimVectorA = True # Vector
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        self.numReadsTileVecCompA = vwa
        self.numReadsUnrollVecCompA = 1
      else: # read components, write components
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorA = False # Components
        self.numReadsTileVecCompA = 1
        self.numReadsUnrollVecCompA = vwa
    else: # TN yes transpose
      self.numReadsTileA = kernel["NumLoadsPerpendicularA"]
      self.numReadsUnrollA = kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vector
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = False # Vector
        self.readUnrollDimVectorA = True # Vector
        self.numReadsUnrollVecCompA = vwa
        self.numReadsTileVecCompA = 1
      else: # read components, write vectors
        self.readTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readTileDimVectorA = False # Components
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        # NEW
        self.numReadsUnrollVecCompA = 1
        self.numReadsTileVecCompA = vwa

    ########################################
    # write vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"] != kernel["UnrollMajorLDSA"]: # NT no transpose
      self.numWritesCoalA = kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vectors, write vectors
        self.writeUnrollDimComponentsA = False # Scalar
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsA = kernel["GlobalReadVectorWidth"] > 1 # Components
          writeCoal = False
        else:
          self.writeTileDimComponentsA = False # Vector
          writeCoal = True
      else: # read components, write components
        self.writeTileDimComponentsA = False # Scalar
        self.writeUnrollDimComponentsA = kernel["GlobalReadVectorWidth"] > 1 # Components
        writeCoal = False
    else: # TN yes transpose
      self.numWritesCoalA = kernel["NumLoadsPerpendicularA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vector, write components
        self.writeUnrollDimComponentsA = False # Scalar
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsA = kernel["GlobalReadVectorWidth"] > 1 # Components
          # LDS writes with LDL>1 will never be coalesced
          writeCoal = False
        else:
          self.writeTileDimComponentsA = kernel["GlobalReadVectorWidth"] > 1 # Components
          writeCoal = False
      else: # read components, write vectors
        self.writeTileDimComponentsA = False # Vector
        self.writeUnrollDimComponentsA = False # Scalar
        writeCoal = True

    # writeCoal indicates writes should be done in the coal dim
    # else in perp
    if writeCoal:
      self.numWritesCoalVecCompA = vwa // kernel["DepthULdsDivisor"]
      self.numWritesPerpVecCompA = 1
    else:
      self.numWritesCoalVecCompA = 1
      self.numWritesPerpVecCompA = vwa
    del writeCoal

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
    # read vectors or vector components b
    ####################################
    if kernel["ProblemType"]["TLUB"]: # NT no transpose
      self.numReadsTileB = kernel["NumLoadsCoalescedB"]
      self.numReadsUnrollB = kernel["NumLoadsPerpendicularB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Vector
        self.readTileDimVectorB = True # Vector
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        self.numReadsTileVecCompB = vwb
        self.numReadsUnrollVecCompB = 1
      else:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorB = False # Components
        # NEW
        self.numReadsTileVecCompB = 1
        self.numReadsUnrollVecCompB = vwb
    else: # TN yes transpose
      self.numReadsTileB = kernel["NumLoadsPerpendicularB"]
      self.numReadsUnrollB = kernel["NumLoadsCoalescedB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = False # Vector
        self.readUnrollDimVectorB = True # Vector
        self.numReadsUnrollVecCompB = vwb
        self.numReadsTileVecCompB = 1
      else:
        self.readTileDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.readTileDimVectorB = False # Components
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        # NEW
        self.numReadsUnrollVecCompB = 1
        self.numReadsTileVecCompB = vwb

    ####################################
    # write vectors or vector components b
    ####################################
    if kernel["ProblemType"]["TLUB"] != kernel["UnrollMajorLDSB"]: # NT no transpose
      self.numWritesCoalB = kernel["NumLoadsCoalescedB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.writeUnrollDimComponentsB = False # Vector
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsB = kernel["GlobalReadVectorWidth"] > 1 # Components
          writeCoal = False
        else:
          self.writeTileDimComponentsB = False # Vector
          writeCoal = True
      else:
        self.writeTileDimComponentsB = False # Scalar
        self.writeUnrollDimComponentsB = kernel["GlobalReadVectorWidth"] > 1 # Components
        # NEW
        self.numWritesCoalVecCompB = 1
        self.numWritesPerpVecCompB = vwb
    else: # TN yes transpose
      self.numWritesCoalB = kernel["NumLoadsPerpendicularB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.writeUnrollDimComponentsB = False
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsB = kernel["GlobalReadVectorWidth"] > 1 # Components
          # LDS writes with LDL>1 will never be coalesced
          writeCoal = False
        else:
          self.writeTileDimComponentsB = kernel["GlobalReadVectorWidth"] > 1 # Components
          writeCoal = False
      else:
        self.writeTileDimComponentsB = False # Vector
        self.writeUnrollDimComponentsB = False # Scalar
        # NEW
        self.numWritesCoalVecCompB = vwb
        self.numWritesPerpVecCompB = 1

    # writeCoal indicates writes should be done in the coal dim
    # else in perp
    if writeCoal:
      self.numWritesCoalVecCompB = vwb // kernel["DepthULdsDivisor"]
      self.numWritesPerpVecCompB = 1
    else:
      self.numWritesCoalVecCompB = 1
      self.numWritesPerpVecCompB = vwb
    del writeCoal

    # numReadVectorComponentsB is refers to global reads
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
    """
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
    """

    self.getTensorParameters(tensorParametersA, kernel, True)
    self.getTensorParameters(tensorParametersB, kernel, False)

    tensorParametersA["PackBatchDims"] = kernel["PackBatchDims"] if kernel["PackBatchDims"] & 0x1 else 0
    tensorParametersB["PackBatchDims"] = kernel["PackBatchDims"] if kernel["PackBatchDims"] & 0x2 else 0
    tensorParametersA["PackedIndices"] = kernel["PackedC0IndicesX"]
    tensorParametersB["PackedIndices"] = kernel["PackedC1IndicesX"]

  @staticmethod
  def zpForSumIdx(sumIdx, zeroPad):
     """ Returns zero-pad for specified sumIdx if it matches or None if not """
     return next((zpi for zpi in zeroPad if zpi[1] == sumIdx), None)


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
  # Open Persistent Loop
  # init iteration counter, define loop target
  ##############################################################################
  @abc.abstractmethod
  def openPersistentLoop(self, kernel):
    return ""


  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  @abc.abstractmethod
  def graWorkGroup(self, kernel, isPap):
    return ""

  ##############################################################################
  # Get Params For Tensor A/B
  ##############################################################################
  def getTensorParameters(self, tP, kernel, tA):
    tP["mirror"] = bool(kernel["ProblemType"]["MirrorDims%s" % ("A" if tA else "B")])
    if tA: # A
      tP["isA"] = True                                      # is this tensor A
      tP["isB"] = False                                     # is this tensor B
      tP["bpe"] = int(4*kernel["ProblemType"]["DataType"].numRegisters())
      tP["tensorChar"] = "A"                                # tensor character A/B
      tP["tensorIdx"] = 0                                   # tensor index A=0, B=1
      tP["tileChar"] = self.tileCharA                       # tile char I0 or J1
      tP["tileIdx"] = kernel["ProblemType"]["Index01A"]     # is the tile dimension of A the 0th or 1th index, i.e. Aki, tileIdx=0
      tP["lsc"] = "LSCA"                                    # load size coalesced A, number of elements that get loaded along coalesced dimension with each load
      tP["lsp"] = "LSPA"                                    # load size perpendicular A, number of elements that get loaded along non-coalesced dimension with each load
      tP["lvc"] = "LVCA"                                    # "load size" in terms of number of short-vectors and not elements
      tP["lvp"] = "LVPA"                                    # "load size" in terms of number of short-vectors and not elements
      tP["rtv"] = self.readTileDimVectorA                   # bool in the tile dimension, reads will read vectors
      tP["rtc"] = self.readTileDimComponentsA               # bool in the tile dimension, reads will read vector components
      #tP["ruv"] = self.readUnrollDimVectorA
      #tP["nlvc"] = self.numReadVectorComponentsA
      #tP["nwvc"] = self.numWriteVectorComponentsA
      tP["wg"] = "WorkGroup0"                               # these are storing the actual strong to lookup the number from kernel dictionary
      tP["prevWg"] = "PrevWorkGroup0"                       # used for prefetch-across-persistent
      tP["sg"] = "SubGroup0"
      tP["tt"] = "ThreadTile0"
      tP["mt"] = "MacroTile0"
      tP["grcg"] = self.globalReadCoalesceGroupA            # global reads are coalesced along threads
      tP["grcv"] = kernel["GlobalReadCoalesceVectorA"]      # global reads are vector reads, and lds writes will be components if transposing
      tP["tlu"] = kernel["ProblemType"]["TLUA"]             # thread stride is less than unroll stride, i.e., not transposing matrix
      tP["ia"] = kernel["ProblemType"]["IndexAssignmentsA"] # array of index assignments
      #tP["nlc"] = kernel["NumLoadsCoalescedA"]
      #tP["nlp"] = kernel["NumLoadsPerpendicularA"]
      #tP["nlcv"] = self.numReadsCoalVecCompA
      tP["nlpv"] = self.numReadsPerpVecCompA                # num vector components perpendicular to coalesced; =1 or VW
      # NEW
      tP["nrt"] = self.numReadsTileA                        # number of reads along tile dimension
      tP["nrtv"] = self.numReadsTileVecCompA                # number of vector components along tile dimension; =1 or VW
      tP["nru"] = self.numReadsUnrollA                      # number of reads along unroll dimension
      tP["nruv"] = self.numReadsUnrollVecCompA              # number of vector components along unroll dimension; =1 or VW
      tP["nrc"] = kernel["NumLoadsCoalescedA"]              # number of reads along coalesced dimension
      tP["nrcv"] = self.numReadsCoalVecCompA                # number of vector components along coalesced dimension
      tP["nrp"] = kernel["NumLoadsPerpendicularA"]          # number of reads along perpendicular dimension
      tP["nrpv"] = self.numReadsPerpVecCompA                # number of vector components along perpendicular dimension
      tP["nwcv"] = self.numWritesCoalVecCompA               # number of vector component writes along coalesced dimension
      tP["nwpv"] = self.numWritesPerpVecCompA               # number of vector component writes along perpendicular dimension
      tP["glvw"] = kernel["GlobalLoadVectorWidthA"]
      # asm
      tP["rcc"] = self.readCoalescedComponentsA             # read vector components along coalesced dimensions
      tP["rcv"] = self.readCoalescedVectorA                 # read vector along coalesced dimension
      tP["rpc"] = self.readPerpendicularComponentsA         # read vector components along perpendicular dimension
      tP["rpv"] = self.readPerpendicularVectorA             # read vector along perpendicular dimension
      tP["ruc"] = self.readUnrollDimComponentsA             # read vector components along unroll dimension
      tP["wtc"] = self.writeTileDimComponentsA              # write vector components along tile dimension
      tP["wuc"] = self.writeUnrollDimComponentsA            # write vector components along unroll dimension
      tP["idx"] = kernel["ProblemType"]["Index0"]           # index 0 is tile dimension belonging to A. Note 'idx' may not be in tP['ia'].
      tP["rc"] = kernel["ProblemType"]["IndexAssignmentsA"][0] \
          in [kernel["ProblemType"]["Index01A"], \
          kernel["ProblemType"]["IndexUnroll"]]             # can read coalesced
      tP["NonTemporal"] = kernel["NonTemporalA"]            # non-temporal read type
    else: # B
      tP["isA"] = False
      tP["isB"] = True
      tP["bpe"] = int(4*kernel["ProblemType"]["DataType"].numRegisters())
      tP["tensorChar"] = "B"
      tP["tensorIdx"] = 1
      tP["tileChar"] = self.tileCharB
      tP["tileIdx"] = kernel["ProblemType"]["Index01B"]
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
      tP["prevWg"] = "PrevWorkGroup1"
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
      tP["idx"] = kernel["ProblemType"]["Index1"]
      tP["rc"] = kernel["ProblemType"]["IndexAssignmentsB"][0] \
          in [kernel["ProblemType"]["Index01B"], \
          kernel["ProblemType"]["IndexUnroll"]] # can read coalesced
      tP["NonTemporal"] = kernel["NonTemporalB"]

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
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def graAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Increments A/B
  # This function declares the increments
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
  def lwaFirstOffset(self, kernel, tP, uDu):
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
  # Recalculate local read addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalReadAddressesAB(self, kernel):
    return ""

  ##############################################################################
  # Recalculate local write addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalWriteAddresses(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  @abc.abstractmethod
  def declareLoopNumIter(self, kernel):
    return ""

  ##############################################################################
  # Define stagger parms that will be used in calculateStagger
  ##############################################################################
  @abc.abstractmethod
  def declareStaggerParms(self, kernel):
    return ""


  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  @abc.abstractmethod
  def calculateStagger(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Remove stagger offset (before tail loop)
  ##############################################################################
  @abc.abstractmethod
  def removeStagger(self, kernel):
    return ""

  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  @abc.abstractmethod
  def calculateLoopNumIter(self, kernel, loopIdx, isPap):
    return ""


  ##############################################################################
  # openShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def openShadowInit(self, kernel):
    return ""

  ##############################################################################
  # closeShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def closeShadowInit(self, kernel):
    return ""

  ##############################################################################
  # Initialize C
  ##############################################################################
  @abc.abstractmethod
  def initC(self, kernel):
    return ""

  ##############################################################################
  # Open Loop
  # loopIdx<0 : tail loop
  ##############################################################################
  @abc.abstractmethod
  def openLoop(self, kernel, loopIdx, uDu):
    return ""

  ##############################################################################
  # Close Loop
  ##############################################################################
  @abc.abstractmethod
  def closeLoop(self, kernel, loopIdx, finalLoop, uDu, emitEndLabelOnly):
    return ""

  ##############################################################################
  # Open Loop Copy
  ##############################################################################
  @abc.abstractmethod
  def openLoopCopy(self, kernel, lc):
      return ""

  ##############################################################################
  # End Summation
  ##############################################################################
  @abc.abstractmethod
  def endSummation(self, kernel):
    return ""

  ##############################################################################
  # Convert Alpha, Beta from F16 to F32 for HPA
  ##############################################################################
  @abc.abstractmethod
  def checkAlphaBetaForHPA(self, kernel):
    return ""

  ##############################################################################
  # MAC Iteration
  # useMacro : if true, call the MAC* macro. If False, inline the MACs
  ##############################################################################
  @abc.abstractmethod
  def macIter(self, kernel, bufferIdx, iuiCount, useMacro):
    return ""

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  @abc.abstractmethod
  def openSumAtLeastUnroll(self, kernel, prefetch, isPap, isOptNLL):
    return ""

  @abc.abstractmethod
  def closeSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isNGLL):
    return ""

  ##############################################################################
  # Global Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def globalReadIncrementAB(self, kernel, loopIdx, prefetchIndex, incs=1):
    return ""

  ##############################################################################
  # Global Read: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def globalReadDo(self, kernel, mode, tP):
    return ""

  ##############################################################################
  # directToLds m0 update: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def directToLdsM0Update(self, kernel, mode, tP):
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
  def localWriteDo(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
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
  def localReadDo(self, kernel, bufferIdx, innerUnrollIndex, epsi, tP):
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
  # globalWriteWorkGroupInit:
  # Perform work-group granularity init
  ##############################################################################
  @abc.abstractmethod
  def globalWriteWorkGroupInit(self, kernel):
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

  @abc.abstractmethod
  def openPrefetchAcrossPersistent(self, kernel, isOptNLL=False):
    return ""

  @abc.abstractmethod
  def closePrefetchAcrossPersistent(self, kernel, isOptNLL=False):
    return ""

  ##############################################################################
  # PrefetchGlobalRead2
  ##############################################################################
  @abc.abstractmethod
  def openPrefetchGlobalRead2(self, kernel):
    return ""

  @abc.abstractmethod
  def closePrefetchGlobalRead2(self, kernel):
    return ""

  ##############################################################################
  # Function End
  ##############################################################################
  @abc.abstractmethod
  def functionEnd(self, kernel, addLabel=True):
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
  # MapAcctoArch
  ##############################################################################
  @abc.abstractmethod
  def MapAcctoArchRegs(self, kernel, option):
    return ""

  ##############################################################################


  ##############################################################################
  #
  #   Entry Functions
  #
  ##############################################################################


  ##############################################################################
  # get kernel name
  ##############################################################################
  def getKernelFileBase(self, kernel):
    rv = self.getKernelName(kernel)
    return self.shortenFileBase(rv)

  def getKernelName(self, kernel):
    if globalParameters["ShortNames"]:
      kernelName = Solution.getNameSerial(kernel, self.kernelSerialNaming)
    else:
      kernelName = Solution.getNameMin(kernel, self.kernelMinNaming)
    return kernelName

  def getKernelSource(self, kernel):
    """
    Returns the source of the kernel, either C++ or assembly.
    """


    fileString = ""
    self.tPA = tensorParametersA = {}
    self.tPB = tensorParametersB = {}
    self.initKernel(kernel, tensorParametersA, tensorParametersB )
    fileString += self.kernelBodyPrefix( kernel, tensorParametersA, \
        tensorParametersB )
    self.stringIdx = 0
    (error, kb) = self.kernelBody( kernel, tensorParametersA, tensorParametersB)

    fileString += kb
    fileString += self.kernelBodySuffix( kernel, tensorParametersA, \
        tensorParametersB )

    if error != 0:
      if globalParameters["ForceGenerateKernel"]:
        print ("warning: Generating kernel source resulted in error {}, but ForceGenerateKernel=1 so saving source".format(error))
      else:
        raise RuntimeError("Generating kernel source resulted in error {}".format(error))

    return fileString

  def getAssemblyDirectory(self):
      return Common.ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly"))

  def byteArrayScriptSource(self):
    return """
#!/usr/bin/env python

fileString = ""
fileString += "/*******************************************************************************\\n"
fileString += "* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.\\n"
fileString += "*\\n"
fileString += "* Permission is hereby granted, free of charge, to any person obtaining a copy\\n"
fileString += '* of this software and associated documentation files (the \"Software\"), to deal\\n'
fileString += "* in the Software without restriction, including without limitation the rights\\n"
fileString += "* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-\\n"
fileString += "* ies of the Software, and to permit persons to whom the Software is furnished\\n"
fileString += "* to do so, subject to the following conditions:\\n"
fileString += "*\\n"
fileString += "* The above copyright notice and this permission notice shall be included in all\\n"
fileString += "* copies or substantial portions of the Software.\\n"
fileString += "*\\n"
fileString += '* THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-\\n'
fileString += "* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS\\n"
fileString += "* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR\\n"
fileString += "* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER\\n"
fileString += "* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-\\n"
fileString += "* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\\n"
fileString += "*******************************************************************************/\\n\\n"
fileString += "/**************************************************\\n"
fileString += "* This file was generated by Tensile:             *\\n"
fileString += "* https://github.com/ROCmSoftwarePlatform/Tensile *\\n"
fileString += "**************************************************/\\n\\n\\n"
import os.path
fileString += '#include "Kernels.h"\\n\\n'
fileString += "/* code object byte array */\\n\\n"
codeObjectFileNames = [f for f in os.listdir(".") if (os.path.isfile(f) and f.endswith(".co"))]
for codeObjectFileName in codeObjectFileNames:
  print codeObjectFileName
  print "\\n"
  kernelName=os.path.splitext(codeObjectFileName)[0]
  codeObjectFile = open(codeObjectFileName, "r")
  codeObjectByteArray = bytearray(codeObjectFile.read())
  codeObjectFile.close()
# write code object byte array for asm
  fileString += "const unsigned char %s_coba[%u] = {\\n" % (kernelName, len(codeObjectByteArray))
  for byteIdx in range(0, len(codeObjectByteArray)):
    byte = codeObjectByteArray[byteIdx]
    fileString += "0x%02x" % byte
    if byteIdx < len(codeObjectByteArray)-1:
      fileString += ","
    else:
      fileString += "};\\n"
    if byteIdx % 16 == 15:
      fileString += "\\n"
  text_file = open("Kernels.cpp", "w")
  text_file.write("%s" % fileString)
  text_file.close()
"""

  def writeByteArrayScript(self):
    asmPath = self.getAssemblyDirectory()

    bytearrayFileName = os.path.join(asmPath,"insert_byte_array.py")
    if not os.path.isfile(bytearrayFileName):
      with open(bytearrayFileName, 'w') as bytearrayFile:
        bytearrayFile.write(self.byteArrayScriptSource())
      os.chmod(bytearrayFileName, 0o777)
    return bytearrayFileName

  def getReplacementKernelPath(self, kernel):
    if not kernel["ReplacementKernel"]:
      return None

    kernelName = self.getKernelName(kernel)
    return ReplacementKernels.Get(kernelName)

  def shortenFileBase(self, base):
    if len(base) <= globalParameters["MaxFileName"]:
      return base

    import hashlib
    import base64

    pivot = globalParameters["MaxFileName"] * 3 // 4
    firstPart = base[:pivot]
    secondPart = base[pivot:]

    secondHash = hashlib.sha256(secondPart.encode()).digest()
    secondPart = base64.b64encode(secondHash, b'_-').decode()

    return firstPart + secondPart

  def getKernelObjectAssemblyFile(self, kernel):
    asmPath = self.getAssemblyDirectory()
    # write assembly file to assembly directory
    kernelName = self.shortenFileBase(self.getKernelName(kernel))
    fileBase = os.path.join(asmPath, kernelName )
    assemblyFileName = "%s.s" % fileBase

    replacementKernel = self.getReplacementKernelPath(kernel)

    if replacementKernel is not None:
      self.tPA = tensorParametersA = {}
      self.tPB = tensorParametersB = {}
      self.initKernel(kernel, tensorParametersA, tensorParametersB )

      shutil.copyfile(replacementKernel, assemblyFileName)
      if globalParameters["PrintLevel"] >= 1:
        print("replacement_assemblyFilename %s" % assemblyFileName)
    else:
      kernelSource = self.getKernelSource(kernel)

      if globalParameters["PrintLevel"] >= 2:
        print("write_assemblyFilename %s" % assemblyFileName)

      with open(assemblyFileName, 'w') as assemblyFile:
        assemblyFile.write(kernelSource)

    return assemblyFileName

  def getAssembledKernelObjectFile(self, kernel):
    assemblyFileName = self.getKernelObjectAssemblyFile(kernel)

    base, ext = os.path.splitext(assemblyFileName)
    objectFileName = base + '.o'

    args = self.getCompileArgs(assemblyFileName, objectFileName)
    if globalParameters["PrintCodeCommands"]:
      print (' '.join(args), " && ")

    subprocess.check_call(args, cwd=self.getAssemblyDirectory())

    return objectFileName

  def getSingleCodeObjectFile(self, kernel):
    objectFileName = self.getAssembledKernelObjectFile(kernel)

    base, ext = os.path.splitext(objectFileName)
    coFileName = base + '.co'

    args = self.getLinkCodeObjectArgs([objectFileName], coFileName)
    if globalParameters["PrintCodeCommands"]:
      print (' '.join(args))

    subprocess.check_call(args, cwd=self.getAssemblyDirectory())

    return coFileName

  def getByteArrayCobaDefinition(self, varName, byteArray):
    s = self.comment("code object byte array")
    s += "const unsigned char %s_coba[%u] = {\n" % (varName, len(byteArray))

    if len(byteArray) != 0:
      s += "0x%02x" % byteArray[0]
      for byteIdx, byte in enumerate(byteArray[1:]):
        if byteIdx % 16 == 15:
          s += ",\n0x%02x" % byte
        else:
          s += ",0x%02x" % byte

    s += '};\n'
    return s

  def getFileCobaDefinition(self, varName, fileName):
    with open(fileName, 'rb') as f:
      byteArray = bytearray(f.read())
    return self.getByteArrayCobaDefinition(varName, byteArray)

  ##############################################################################
  def getSourceFileString(self, kernel):
    """
    Returns a string suitable for placing in Kernels.cpp.  This means the actual kernel source in the case
    of a source kernel, or an assembled code object byte array definition in the case of an assembly kernel,
    or an empty string in the case that CodeFromFiles is true.

    In the case of an assembly kernel, this function has the side effect of creating the following files:
     * An assembly source file
     * An object file
     * A code object file
     * A Python script which can create byte array variable definitions.
    """

    try:
      if kernel["KernelLanguage"] == "Assembly":
        asmPath = self.getAssemblyDirectory()
        kernelName = self.getKernelName(kernel)

        if globalParameters["GenerateSourcesAndExit"]:
          # only create the assembly file.
          self.getKernelObjectAssemblyFile(kernel)
          return (0, "")
        else:
          self.writeByteArrayScript()
          coFile = self.getSingleCodeObjectFile(kernel)

          if globalParameters["CodeFromFiles"] or globalParameters["NewClient"] > 1:
            # I guess in this case we are making sure that the code object file exists by executing the code
            # above but we aren't placing it into the source.
            return (0, "")

          return (0, self.getFileCobaDefinition(kernelName, os.path.join(asmPath, coFile)))

      else:
        return (0, self.getKernelSource(kernel))

    except subprocess.CalledProcessError as exc:
      print(exc)
      return (-1, "")
    except RuntimeError as exc:
      print(exc)
      return (-2, "")

  ##############################################################################
  # header file string
  ##############################################################################
  def getHeaderFileString(self, kernel):
    kernelName = self.getKernelName(kernel)
    fileString = "" # CHeader
    if self.language == "HIP" or self.language == "OCL":
      if not globalParameters["MergeFiles"]:
        fileString += CHeader
        fileString += "#pragma once\n\n"
        if self.language == "HIP":
          fileString += "#include <hip/hip_runtime.h>\n"
          fileString += "#include <hip/hip_fp16.h>\n"
          fileString += "#include <KernelHeader.h>\n"
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
      if not globalParameters["CodeFromFiles"]:
        fileString += "extern const unsigned char %s_coba[]; // code object byte array\n" % kernelName

    return fileString

