################################################################################
# Copyright (C) 2016-2019 Advanced Micro Devices, Inc. All rights reserved.
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
  #   self.perIterCode[]
  #      - List indexed by unroll iteration.
  #        Each entry in the list is a code module that should be added into that iteration.
  #        May be None, indicating no extra code for that iteration
  #
  # This routine is responsible for setting the schedule including determining
  # that all necessary dependency are met.  The driver code in kernelBody
  # blindly follows the plan set in unrollLoopHeaderCode and perIterCode
  ##############################################################################
  def makeSchedule(self, kernel, tensorParametersA, tensorParametersB, localWriteEndIter):
    # 0x2=print GR and LW code blocks, 0x1= print info messages
    schedDb = 0

    currentIsa = globalParameters["CurrentISA"]
    maxVmcnt = globalParameters["AsmCaps"][currentIsa]["MaxVmcnt"]

    self.unrollLoopHeaderCode = Code.Module()
    # schedule of work for each local_read iteration:
    self.perIterGlobalReadCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    assert([item.name for item in self.globalReadIncrements.itemList] == ['globalReadIncrementA', 'globalReadIncrementB'])

    globalReadIncACode  = self.globalReadIncrements.findNamedItem("globalReadIncrementA")
    globalReadIncBCode  = self.globalReadIncrements.findNamedItem("globalReadIncrementB")

    lastLoadIter = 0
    if kernel["MatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
      numMfmaPerIter = self.numMfmas * kernel["InnerUnroll"]
    else:
      numMfmaPerIter = 1
    if not self.scheduleGlobalRead:
      # put everything in the header:
      self.unrollLoopHeaderCode.addCode(self.dtlsM0UpdateACode)
      self.unrollLoopHeaderCode.addCode(self.globalReadACode)
      self.unrollLoopHeaderCode.addCode(self.dtlsM0UpdateBCode)
      self.unrollLoopHeaderCode.addCode(self.globalReadBCode)
      self.unrollLoopHeaderCode.addCode(globalReadIncACode)
      self.unrollLoopHeaderCode.addCode(globalReadIncBCode)
    else:
      self.unrollLoopHeaderCode.addCode(self.globalReadACode.header)
      self.unrollLoopHeaderCode.addCode(self.globalReadBCode.header)

      readCnt = self.globalReadACode.middle.countType(Code.GlobalReadInst) + \
                self.globalReadBCode.middle.countType(Code.GlobalReadInst)
      # reads and incs are scheduled in iters range(0..endIter)
      endIter = (readCnt + 2);
      endIter = roundUp(endIter/numMfmaPerIter)

      if endIter > localWriteEndIter:
        # Front-load some of the buffer loads if we don't have enough loop iters:
        # could use a different/smarter algorithm to space out the loads?
        firstStep = (endIter-(localWriteEndIter) + 1)*numMfmaPerIter
        endIter = localWriteEndIter
      else:
	# schedule b2b for readCnt > 2 (True for bigger TT)
        firstStep = 1 * numMfmaPerIter
        

      # Add all loads from middle as individual schedulable items
      itemsToSched =  list(self.globalReadACode.middle.items()) + \
                      list(self.globalReadBCode.middle.items())
      itemsToSched.append(globalReadIncACode)
      itemsToSched.append(globalReadIncBCode)


      if schedDb & 0x1:
        print("makeSchedule-gr, readCnt=", readCnt, "firstStep=", firstStep, "endIter=", endIter)

      # insert dtlsM0UpdateACode dtlsM0UpdateBCode code
      if self.globalReadACode.middle.items():
        self.globalReadACode.middle.items()[0].items().insert(0,self.dtlsM0UpdateACode)
      if self.globalReadBCode.middle.items():
        self.globalReadBCode.middle.items()[0].items().insert(0,self.dtlsM0UpdateBCode)
      # append 'n' global load at a time 
      # append global load(S) first 'number of global load(s) determined by  firstStep
      for item in itemsToSched[:firstStep]:
        self.perIterGlobalReadCode[0].addCode(item)
      itemsToSched = itemsToSched[firstStep:]
      for u in range(1, endIter):
        itemPerIter = 1 * numMfmaPerIter
        try:
          for item in itemsToSched[:itemPerIter]:
            self.perIterGlobalReadCode[u].addCode(item)
            lastLoadIter = u
          itemsToSched = itemsToSched[itemPerIter:]
        except IndexError:
          break # no code left to schedule

      assert not itemsToSched # should have scheduled everything already
      
      self.perIterGlobalReadCode[endIter-1].addCode(self.globalReadACode.footer)
      self.perIterGlobalReadCode[endIter-1].addCode(self.globalReadBCode.footer)

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
    else:
      # create a plan:
      itemsToSched = list(self.localWriteACode.items()) + list(self.localWriteBCode.items())
      if 1:
        # This counts the number of modules which contain a ds_write
        # Scheduler below keeps all writes in the same module in same iteration
        # so this is better match to what it is trying to do
        writesToSched = sum(1 for item in itemsToSched if item.countType(Code.LocalWriteInst))
      else:
        # count the number of writes, this doesn't match how they are
        # scheduled so pushes writes up too far
        writesToSched = self.localWriteACode.countType(Code.LocalWriteInst) + \
                     self.localWriteBCode.countType(Code.LocalWriteInst)
      startIter = kernel["LoopIters"] - writesToSched
      # to offset last LW 1 mfma with barrier
      # TODO: how many mfma to offset based on mfma latency
      if kernel["MatrixInstruction"]:
        writesToSched += 1
      startIter = localWriteEndIter - roundUp(writesToSched/numMfmaPerIter) + 1
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
      for u in range(startIter, localWriteEndIter+1):
        if u==(localWriteEndIter):
          itemPerIter = len(itemsToSched) # schedule all remaining activity
        else:
          itemPerIter = 1 * numMfmaPerIter
          # if localwrite is not multiple of numMfmaPerIter, fill last iteration first.
          # make sure numMfmaPerIter is enough to schedule localwrite
          # TODO: if numMfmaPerIter is not enough to schedule localwrite, need smarter way to distribute localWrite
          if kernel["MatrixInstruction"] and \
            u == startIter and \
            not (startIter + writesToSched//numMfmaPerIter > localWriteEndIter) and \
            kernel["ScheduleIterAlg"] == 3:
            itemPerIter = writesToSched % numMfmaPerIter

        for item in itemsToSched[:itemPerIter]:
          # Use a module to ensure these pieces stay together in the sub-iter scheduler
          imod = Code.Module("LocalWriteMod%u"%u)

          # Prepend a waitcnt if needed
          writesPerItem = item.countType(Code.LocalWriteInst)
          imod.addComment0("sched write - iter %u writesPerItem=%u"%(u,writesPerItem))
          if writesPerItem:
            # if writesPerItem>1 this indicates multiple LocalWrites in the same module
            # this happens in some transpose cases.  Here the first write needs to wait
            # for the associated global read to finish, then the remaining writes can flow
            # TODO - can schedule these writes across iters, should figure this out above
            writesToSched = writesToSched - writesPerItem
            readsToWait = readsToWait - 1
            # TODO - gfx9 supports higher max VMCNT
            if 1:
              imod.addCode(Code.WaitCnt(self.version, -1, min(maxVmcnt, readsToWait), \
                  "wait for global read before writing to local"))
            else:
              print("warning - scheduleLocalWrite adding conservative vmcnt(0)")
              imod.addCode(Code.WaitCnt(self.version, -1, 0, "conservative waitcnt"))
          imod.addCode(item)
          self.perIterLocalWriteCode[u].addCode(imod)
        itemsToSched = itemsToSched[itemPerIter:]

      # should never run out of items to schedule
      assert not itemsToSched # should have scheduled everthing already


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
  def makeSubIterSchedule(self, kernel, localReadCode, iteration, pointerCode, waitCode, macIterCode, \
      waitLWCode = Code.Module(), syncCode = Code.Module(), packCode = Code.Module()):

    iterCode = Code.Module()
    globalReadCode = self.perIterGlobalReadCode[iteration]
    localWriteCode = self.perIterLocalWriteCode[iteration]
    # Default schedule is other, local reads, then local writes:
    if self.scheduleIterAlg==0:
      # simple schedule, just add the modules in-order
      iterCode.addCode(globalReadCode)
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      iterCode.addCode(localReadCode)
      iterCode.addCode(localWriteCode)
      iterCode.addCode(pointerCode)
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
      iterCode.addCode(pointerCode)
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
      packItem = packCode.flatitems()
      packCount = 0
      packInstPerIter = (1 + self.numRowInsts * self.numColInsts)*(kernel["MatrixInstK"]//2)
      while packItem:
        if packCount < packInstPerIter:
          item = packItem.pop(0)
          iterCode.addCode(item)
          packCount = packCount + 1
        else:
          break
      
      if packCount == 0:
        tmpVgpr = self.vgprPool.checkOut(1)
        iterCode.addInst("v_mov_b32 ","v%u"%(tmpVgpr),"0x0","valu operation to have different priority")
        self.vgprPool.checkIn(tmpVgpr)

      iterCode.addInst("s_setprio ","3","Raise priority while processing macs")
      macIterItem = macIterCode.flatitems()
      # pop the first code which is s_nop 1 for packing
      if kernel["MatrixInstruction"] and not kernel["TransposeLDS"] and \
      (kernel["ProblemType"]["DataType"].isBFloat16() or kernel["ProblemType"]["DataType"].isHalf()):
        item = macIterItem.pop(0)
        iterCode.addCode(item)

      while macIterItem:
        item = macIterItem.pop(0)
        iterCode.addCode(item)
        packCount = 0
        while packItem:
          if packCount < 8:
            item = packItem.pop(0)
            iterCode.addCode(item)
            packCount = packCount + 1
          else:
            break
        if packCount > 0:
          iterCode.addInst("s_nop ","1","VALU packing writes to be consumed by matrix instruction")  

      iterCode.addInst("s_setprio ","1","Raise priority while processing macs")
      iterCode.addCode(localWriteCode)
      iterCode.addCode(pointerCode)
      iterCode.addInst("s_setprio ","2","Raise priority while processing macs")
      pass
    elif self.scheduleIterAlg == 3:
      # mfma interleave
      if kernel["MatrixInstruction"]:
        numMfmaPerIter = self.numMfmas * kernel["InnerUnroll"]
        writesPerIter = localWriteCode.countType(Code.LocalWriteInst)
        localWriteEndIter = kernel["LoopIters"] - kernel["PrefetchLocalRead"] - 1
        isBarrier = localWriteEndIter + 1
        writeItems = list(localWriteCode.items())
        macIterItems = macIterCode.flatitems()
        readsPerIter = localReadCode.countType(Code.LocalReadInst)
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
          while localReadCodeA.items() or localReadCodeB.items():
            if localReadCodeA.items():
              localReadCodeAB.addCode(localReadCodeA.items().pop(0))
            if localReadCodeB.items():
              localReadCodeAB.addCode(localReadCodeB.items().pop(0))
        localReadItems = localReadCodeAB.flatitems()
        packAB = Code.Module()
        for iui in range(kernel["InnerUnroll"]):
          packA = packCode.findNamedItem("packA_I%s"%(iui))
          packB = packCode.findNamedItem("packB_I%s"%(iui))
          # In case localReadDo not generate pack Module
          # and findNamedItem will return None type
          # TODO: let all type have pack Module
          if not packA:
            packA = Code.Module()
          if not packB:
            packB = Code.Module()
          while packA.items() or packB.items():
            if packA.items():
              packAB.addCode(packA.items().pop(0))
            if packB.items():
              packAB.addCode(packB.items().pop(0))
        # remove s_nop for packing if TLDS
        if (kernel["ProblemType"]["DataType"].isBFloat16() or kernel["ProblemType"]["DataType"].isHalf()) and not kernel["TransposeLDS"]:
          macIterItems.pop(0)
        iterCode.addCode(waitLWCode)
        iterCode.addCode(syncCode)
        skipLocalWriteWaitcnt = 0
        localReadsWaitcnt = 0
        for i in range(numMfmaPerIter):
          readLeft = readsPerIter
          # with PrefetchLocalRead, localreads can interleave with mfma
          if kernel["PrefetchLocalRead"]:
            # we want no localreads at first and barrier mfma
            if (iteration == 0 or iteration == isBarrier):
              if i == 0 or numMfmaPerIter == 1:
                readLeft = 0
              # rest mfma help to schedule those localReads
              else:
                readLeft = readsPerIter // (numMfmaPerIter-1)
                if (readsPerIter % (numMfmaPerIter-1)) >= i:
                  readLeft += 1
            else:
              # if readsPerIter is not multiple of numMfmaPerIter, fill remainder localReads from first mfma
              readLeft = readsPerIter // numMfmaPerIter
              if (readsPerIter % (numMfmaPerIter)) > i:
                readLeft += 1
          while localReadItems:
            if readLeft == 0 and (i != numMfmaPerIter - 1):
              break
            item = localReadItems.pop(0)
            iterCode.addCode(item)
            readsThisItem = item.countType(Code.LocalReadInst)
            if readsThisItem:
              readLeft = readLeft - 1
              # because waitCode is scheduled at first mfma, we only need to skip localreads at first mfma.
              if (i == 0):
                localReadsWaitcnt += 1
          if globalReadCode.items():
            iterCode.addCode(globalReadCode.items().pop(0))
          # let localWrite to be scheduled from last mfma
          # mfma|lw|mfma|lw|mfma|mfma => mfma|mfma|lw|mfma|lw|mfma
          # except for localWriteEndIter, because in that iteration, we only have numMfmaPerIter-1 localWrite
          if (i >= numMfmaPerIter - writesPerIter) or (iteration == localWriteEndIter):
            # in case there are localWrite and globalread in same iteration
            # we need to make sure globalRead before localWrite
            if writeItems and not globalReadCode.countType(Code.GlobalReadInst):
              writeItem = writeItems.pop(0)
              iterCode.addCode(writeItem)
              # if there is localWrite at first mfma, need to skip it in waitcnt.
              if i == 0:
                skipLocalWriteWaitcnt += writeItem.countType(Code.LocalWriteInst)
          if i == numMfmaPerIter - 1:
            while globalReadCode.items():
              iterCode.addCode(globalReadCode.items().pop(0))
            while writeItems:
              writeItem = writeItems.pop(0)
              iterCode.addCode(writeItem)
              if i == 0:
                skipLocalWriteWaitcnt += writeItem.countType(Code.LocalWriteInst)
            iterCode.addCode(pointerCode)
          if i == 0:
            iterCode.addCode(waitCode)
          if packAB.items():
            if packAB.items(): 
              iterCode.addCode(packAB.items().pop(0))
            if packAB.items():
              iterCode.addCode(packAB.items().pop(0))
            iterCode.addInst("s_nop ","1","VALU packing writes to be consumed by matrix instruction")  
          iterCode.addCode(macIterItems.pop(0))
    else:
      assert 0, "Unsupported scheduleIterAlg=%u"%self.scheduleIterAlg

    for item in list(packCode.items()):
      self.vgprPool.checkIn(item.tempVgpr)

    if isinstance(waitCode, Code.WaitCnt):
      # Set the waitCount, based on the new iter schedule
      lgkmcnt = 0 # most conservative
      if kernel["MatrixInstruction"]:
        localReads  = localReadCode.countType(Code.LocalReadInst)
        localWrites = 0
        if kernel["ScheduleIterAlg"] == 0 or kernel["ScheduleIterAlg"] == 1:
          for i in range (kernel["PrefetchLocalRead"]+1):
            curIter = ((iteration-i)+kernel["LoopIters"])%kernel["LoopIters"]
            localWrites += self.perIterLocalWriteCode[curIter].countType(Code.LocalWriteInst)
        # ScheduleIterAlg=2, localwrite is after waitCnt, no need to count it's current iteration.
        if kernel["ScheduleIterAlg"] == 3:
          localReads = readsPerIter
          for i in range (1,kernel["PrefetchLocalRead"]):
            curIter = ((iteration-i)+kernel["LoopIters"])%kernel["LoopIters"]
            localWrites += self.perIterLocalWriteCode[curIter].countType(Code.LocalWriteInst)
        # here the reads are prefetches so can skip them in the waitcnt
        # how many localreads can skip is based on how many iterations we prefetch.
        lgkmcnt += localReads * kernel["PrefetchLocalRead"]
        # and the writes are targetting another section of LDS and are
        # synchronized through a different waitnct than this one
        # (which is always just before the macs)
        lgkmcnt += localWrites
        # some of localReads is interleaved after waitcnt
        if kernel["ScheduleIterAlg"] == 3 and kernel["PrefetchLocalRead"] != 0:
          lgkmcnt -= (localReads - localReadsWaitcnt)
        if kernel["ScheduleIterAlg"] == 3 and kernel["ScheduleLocalWrite"] == 1:
          # current localWrite iteration
          lgkmcnt += skipLocalWriteWaitcnt
          # first localWrite iteration
          # in this iteration, we only have to skip localwrite schedule at last mfma
          if self.perIterLocalWriteCode[iteration-kernel["PrefetchLocalRead"]].countType(Code.LocalWriteInst) and kernel["PrefetchLocalRead"]:
            preIterLocalWrites = self.perIterLocalWriteCode[iteration-kernel["PrefetchLocalRead"]].countType(Code.LocalWriteInst)
            preIterGlobalReads = self.perIterGlobalReadCode[iteration-kernel["PrefetchLocalRead"]].countType(Code.GlobalReadInst)
            skipPreIterLW = max(preIterLocalWrites - (numMfmaPerIter - preIterGlobalReads - 1),1) + max(numMfmaPerIter - readsPerIter, 0)
            lgkmcnt += skipPreIterLW
      else:
        for item in list(iterCode.items()):
          localReads  = item.countType(Code.LocalReadInst)
          localWrites = item.countType(Code.LocalWriteInst)
          if kernel["PrefetchLocalRead"]:
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
      waitCode.comment += " old=%u new=%u (Local write no wait)" % (waitCode.lgkmcnt, lgkmcnt)
      waitCode.lgkmcnt = lgkmcnt

    return iterCode

  ##############################################################################
  # returns list of modules or text
  # papIter indicates this is the setup for the "prefetchAcrossPersistent"
  # (aka pap) iteration
  ##############################################################################
  def setupNewTile(self, kernel, tensorParametersA, tensorParametersB, isPap):
    kl = []

    if self.enable["PreLoop"]:
      ####################################
      # Global Read Addresses
      ####################################
      kl.append(self.comment3("Begin setupNewTile"))

      # work-group assignments
      kl.append(self.comment("global read addresses: work-group"))
      kl.append(self.graWorkGroup(kernel, isPap))

      # tile assignments
      kl.append(self.comment("global read addresses: tile offset assignment a"))
      kl.append(self.graTileAssignment(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: tile offset assignment b"))
      kl.append(self.graTileAssignment(kernel, tensorParametersB))

      # unroll assignments
      kl.append(self.comment("global read addresses: unroll assignment a"))
      kl.append(self.graUnrollAssignment(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: unroll assignment b"))
      kl.append(self.graUnrollAssignment(kernel, tensorParametersB))

      # other free indices
      if kernel["ProblemType"]["NumIndicesC"] > 2:
        kl.append(self.comment("global read addresses: other free assignments"))
        kl.append(self.graOtherFreeAssignments(kernel))

      # other summation indices
      if self.otherSummations:
        kl.append(self.comment("global read addresses: other summation assignments"))
        kl.append(self.graOtherSummationAssignments(kernel))

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

      # final offsets
      kl.append(self.comment("global read addresses: final offsets a"))
      kl.append(self.graFinalOffsets(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: final offsets b"))
      kl.append(self.graFinalOffsets(kernel, tensorParametersB))

      # addresses
      kl.append(self.comment("global read addresses: addresses a"))
      kl.append(self.graAddresses(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: addresses b"))
      kl.append(self.graAddresses(kernel, tensorParametersB))

      # increments
      kl.append(self.comment("global read addresses: increments a"))
      for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
        kl.append(self.graIncrements(kernel, i, tensorParametersA))
      kl.append(self.comment("global read addresses: increments b"))
      for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
        kl.append(self.graIncrements(kernel, i, tensorParametersB))

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

      # first offsets
      kl.append(self.comment("local write addresses: first offset a"))
      kl.append(self.lwaFirstOffset(kernel, tensorParametersA))
      kl.append(self.comment("local write addresses: first offset b"))
      kl.append(self.lwaFirstOffset(kernel, tensorParametersB))

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

    ####################################
    # prefetch: unrolled loop prefix
    ####################################
    if kernel["PrefetchGlobalRead"]:
      pfi = 1
      kl.append(self.comment("prefetch: global -> local"))
      kl.append(self.openSumAtLeastUnroll(kernel, prefetch=True, isPap=isPap, isOptNLL=False))
      if self.enable["GlobalRead"]:
        kl.append(str(self.directToLdsM0Update(kernel, 0, tensorParametersA)))
        kl.append(str(self.globalReadDo(kernel, 0, tensorParametersA)))
        kl.append(str(self.directToLdsM0Update(kernel, 0, tensorParametersB)))
        kl.append(str(self.globalReadDo(kernel, 0, tensorParametersB)))
      if self.enable["GlobalReadInc"]:
        kl.append(self.globalReadIncrementAB(kernel, self.unrollIdx, pfi))

    kl.append(self.comment3("End setupNewTile"))

    return kl

  ##############################################################################
  # noLoadLoop
  # Create the no load loop (NLL)
  #
  # isOptNLL : the NLL is to be optimized for the alpha=1 and non-edge case
  ##############################################################################
  def noLoadLoop( self, kernel, tensorParametersA, tensorParametersB, isOptNLL, pack ):
    kl = []
    pflr     = kernel["PrefetchLocalRead"]

    kl.append(self.comment3("%s NoLoadLoop - Begin") % "Opt" if isOptNLL else "")
    if self.prefetchAcrossPersistent:
      kl.append(self.openPrefetchAcrossPersistent(kernel))
      kl += self.setupNewTile(kernel, self.tPA, self.tPB, True)
      kl.append(self.closePrefetchAcrossPersistent(kernel))

    kl.append(self.openSumAtLeastUnroll(kernel, prefetch=False, isPap=False, \
        isOptNLL=isOptNLL))
    if not kernel["PrefetchLocalRead"]:
      if self.enable["Wait"]:
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "4wait for local write"))
      if self.enable["Sync"]:
        kl.append(self.syncThreads(kernel))
    for u in range(0, kernel["LoopIters"]):
      kl.append(self.comment("iter %u"%u))
      plrIdx = (u+pflr) % (kernel["PrefetchLocalRead"] + 1)
      pack[plrIdx] = Code.Module()
      for iui in range(0,kernel["InnerUnroll"]):
        if self.enable["LocalRead"]:
          if u < kernel["LoopIters"] - kernel["PrefetchLocalRead"]:
            if kernel["MatrixInstruction"] and \
              (kernel["ProblemType"]["DataType"].isBFloat16() or kernel["ProblemType"]["DataType"].isHalf()) and \
              not kernel["TransposeLDS"] :
              # Reading 16-bit data from LDS requires packing when ECC enabled
              kl.append(self.comment("local read a"))
              localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx, iui, 0, ((u+pflr)*kernel["InnerUnroll"] + iui), tensorParametersA)
              kl.append(localReadCodeA)
              kl.append(self.comment("local read b"))
              localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx, iui, 0, ((u+pflr)*kernel["InnerUnroll"] + iui), tensorParametersB)
              kl.append(localReadCodeB)
              pack[plrIdx].addCode(packCodeA)
              pack[plrIdx].addCode(packCodeB)
            else:
              kl.append(self.comment("local read a"))
              kl.append(self.localReadDo(kernel, plrIdx, iui, 0,((u+pflr)*kernel["InnerUnroll"] + iui), tensorParametersA))
              kl.append(self.comment("local read b"))
              kl.append(self.localReadDo(kernel, plrIdx, iui, 0,((u+pflr)*kernel["InnerUnroll"] + iui), tensorParametersB))
            kl.append(self.comment("local read inc a"))
            kl.append(self.localReadInc(kernel, iui, tensorParametersA))
            kl.append(self.comment("local read inc b"))
            kl.append(self.localReadInc(kernel, iui, tensorParametersB))
      if self.enable["Wait"]:
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, \
            1 if (u < kernel["LoopIters"]-1 and kernel["PrefetchLocalRead"]) else 0, \
            "7wait for local read"))
      if self.enable["MAC"]:
        luIdx = (u) % (kernel["PrefetchLocalRead"] + 1)
        if kernel["MatrixInstruction"]:
          kl.append(pack[luIdx])
          for item in list(pack[luIdx].items()):
            self.vgprPool.checkIn(item.tempVgpr)
          kl.append(self.mfmaIter(kernel, luIdx, kernel["InnerUnroll"]))
        else:
          kl.append(self.macIter(kernel, luIdx, kernel["InnerUnroll"], useMacro=not isOptNLL ))
    kl.append(self.closeSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=isOptNLL))
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
    if self.prefetchAcrossPersistent:
      # first prefetch is outside persistent loop, subsequent prefetch will
      # be integrated into no-load-loop
      kl += self.setupNewTile(kernel, tensorParametersA, tensorParametersB, False)
      kl.append(self.openPersistentLoop(kernel))
    else:
      # prefetch is inside persistent loop
      kl.append(self.openPersistentLoop(kernel))
      kl += self.setupNewTile(kernel, tensorParametersA, tensorParametersB, False)
    
    pack = [ Code.Module() for i in range (kernel["PrefetchLocalRead"]+1) ]
    
    if kernel["PrefetchGlobalRead"]:
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
        if kernel["PrefetchAcrossPersistent"] or self.actualSummationLoops>1:
          kl.append(self.syncThreads(kernel))

      if self.enable["LocalWrite"]:
        # local write
        kl.append(self.comment("local write a"))
        kl.append(self.localWriteDo(kernel, tensorParametersA))
        kl.append(self.comment("local write b"))
        kl.append(self.localWriteDo(kernel, tensorParametersB))
        # swap local ptrs
        kl.append(self.comment("local write swap a"))
        kl.append(self.localWriteSwapOffsets(kernel, tensorParametersA))
        kl.append(self.comment("local write swap b"))
        kl.append(self.localWriteSwapOffsets(kernel, tensorParametersB))
        kl.append(self.localWriteInitPointers(kernel, tensorParametersA))
        kl.append(self.localWriteInitPointers(kernel, tensorParametersB))
      # prefetch-local
      if kernel["PrefetchLocalRead"]:
        if self.enable["Wait"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "0prefetch wait for local write"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel))

        # in some cases need an extra copy of the LDS read with appropriate double buffer offsets
        if self.enable["LocalRead"]:
          for plrIdx in range(0, kernel["PrefetchLocalRead"]):
            pack[plrIdx] = Code.Module()
            for espi in range(0, (self.prefetchAcrossPersistent and kernel["ExpandPointerSwap"])+1):
              for iui in range(0,kernel["InnerUnroll"]):
                if kernel["MatrixInstruction"] and \
                  (kernel["ProblemType"]["DataType"].isBFloat16() or kernel["ProblemType"]["DataType"].isHalf()) and \
                  not kernel["TransposeLDS"]:
                  # Reading 16-bit data from LDS requires packing when ECC enabled
                  kl.append(self.comment("local read prefetch a"))
                  localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx, iui, espi, (plrIdx*kernel["InnerUnroll"]+iui), tensorParametersA)
                  kl.append(localReadCodeA)
                  kl.append(self.comment("local read prefetch b"))
                  localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx, iui, espi, (plrIdx*kernel["InnerUnroll"]+iui), tensorParametersB)
                  kl.append(localReadCodeB)
                  pack[plrIdx].addCode(packCodeA)
                  pack[plrIdx].addCode(packCodeB)
                else:
                  kl.append(self.comment("local read prefetch a"))
                  kl.append(self.localReadDo(kernel, plrIdx, iui, espi,(plrIdx*kernel["InnerUnroll"]+iui), tensorParametersA))
                  kl.append(self.comment("local read prefetch b"))
                  kl.append(self.localReadDo(kernel, plrIdx, iui, espi,(plrIdx*kernel["InnerUnroll"]+iui), tensorParametersB))
                kl.append(self.comment("local read inc a"))
                kl.append(self.localReadInc(kernel, iui, tensorParametersA))
                kl.append(self.comment("local read inc b"))
                kl.append(self.localReadInc(kernel, iui, tensorParametersB))
      kl.append(self.closeSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=False))

    # open unrolled summation loop
    kl.append(self.comment3("Unrolled Loop(s) - Begin"))
    kl.append(self.openLoop(kernel, self.unrollIdx))

    expand = kernel["ExpandPointerSwap"]
    loopCopies = 2 if expand else 1
    for lc in range(0, loopCopies):
      finalLoop = not expand or lc==loopCopies-1
      kl.append(self.comment3("Unroll Loop %u/%u - Begin" % (lc+1, loopCopies)))
      kl.append(self.openLoopCopy(kernel, lc))
      if kernel["PrefetchGlobalRead"] and not kernel["PrefetchLocalRead"]:
        if self.enable["Wait"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, \
              "1wait for local write"))
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

      # which iteration to perform the local writes
      # if scheduleLocalWrite=0, all local writes performed in this iteration
      # if scheduleLocalWrite=1, writes are scheduled backwards from this iteration
      # If PLR=0, the writes are placed in the last loop iteration
      localWriteEndIter = kernel["LoopIters"] - kernel["PrefetchLocalRead"] - 1

      # Schedule the global read, global read inc, and writes:
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
      if kernel["PrefetchLocalRead"] and not kernel["PrefetchGlobalRead"]:
        if self.enable["LocalRead"]:
          for plrIdx in range(0, kernel["PrefetchLocalRead"]):
            pack[plrIdx] = Code.Module()
            for iui in range(0,kernel["InnerUnroll"]):
              if kernel["MatrixInstruction"] and \
                (kernel["ProblemType"]["DataType"].isBFloat16() or kernel["ProblemType"]["DataType"].isHalf()) and \
                not kernel["TransposeLDS"] :
                # Reading 16-bit data from LDS requires packing when ECC enabled
                kl.append(self.comment("prefetch local a"))
                localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx, iui, 0, (plrIdx*kernel["InnerUnroll"]+iui), tensorParametersA)
                kl.append(localReadCodeA)
                kl.append(self.comment("prefetch local b"))
                localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx, iui, 0, (plrIdx*kernel["InnerUnroll"]+iui), tensorParametersB)
                kl.append(localReadCodeB)
                pack[plrIdx].addCode(packCodeA)
                pack[plrIdx].addCode(packCodeB)
              else:
                kl.append(self.comment("prefetch local a"))
                kl.append(self.localReadDo(kernel, plrIdx, iui, 0, (plrIdx*kernel["InnerUnroll"]+iui), tensorParametersA))
                kl.append(self.comment("prefetch local b"))
                kl.append(self.localReadDo(kernel, plrIdx, iui, 0, (plrIdx*kernel["InnerUnroll"]+iui), tensorParametersB))
              kl.append(self.comment1("local read increment a"))
              kl.append(self.localReadInc(kernel, iui, tensorParametersA))
              kl.append(self.comment1("local read increment b"))
              kl.append(self.localReadInc(kernel, iui, tensorParametersB))

      kl.append(self.closeString(kernel))
      kl.append(self.openString(kernel))

      pflr     = kernel["PrefetchLocalRead"]  # how many pf already done above

      ############################################################################
      # unrolled loop: mac iterations
      # Includes handling for the 2nd-to-last iteration:
      ############################################################################
      for u in range(0, kernel["LoopIters"]-1):
        # which loop iteration to reset the LRO,
        # note if PLR=0, isResetLroIter is False for all u
        isResetLroIter = (u == localWriteEndIter)
        extraComment = ""
        if isResetLroIter:
          extraComment = " (localWrite + swap local pointers iteration)"
        kl.append(self.comment("iter %u%s"%(u,extraComment)))
        plrIdx = (u+pflr) % (kernel["PrefetchLocalRead"]+1)
        pack[plrIdx] = Code.Module()

        localReads = Code.Module()
        localReadsA = Code.Module()
        localReadsB = Code.Module()
        # if not PrefetchGlobalRead, PrefetchLocalRead is at start of the loop
        if (u < kernel["LoopIters"] - kernel["PrefetchLocalRead"]) or kernel["PrefetchGlobalRead"]:
          for iui in range(0,kernel["InnerUnroll"]):
            if self.enable["LocalRead"]:
              if kernel["MatrixInstruction"] and \
                (kernel["ProblemType"]["DataType"].isBFloat16() or kernel["ProblemType"]["DataType"].isHalf()) and \
                not kernel["TransposeLDS"] :
                # Reading 16-bit data from LDS requires packing when ECC enabled
                localReads.addText(self.comment("local read a"))
                localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx, iui, 0, ((u+pflr)*kernel["InnerUnroll"]+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersA)
                localReads.addCode(localReadCodeA)
                localReads.addText(self.comment("local read b"))
                localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx, iui, 0, ((u+pflr)*kernel["InnerUnroll"]+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersB)
                localReads.addCode(localReadCodeB)
                localReadsA.addCode(localReadCodeA)
                localReadsB.addCode(localReadCodeB)
                pack[plrIdx].addCode(packCodeA)
                pack[plrIdx].addCode(packCodeB)
              else:
                localReads.addText(self.comment("local read a"))
                localReads.addCode(self.localReadDo(kernel, plrIdx, iui, 0,((u+pflr)*kernel["InnerUnroll"]+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersA))
                localReads.addText(self.comment("local read b"))
                localReads.addCode(self.localReadDo(kernel, plrIdx, iui, 0,((u+pflr)*kernel["InnerUnroll"]+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersB))
                #container for holding local read A & B elements for later re-ordering
                localReadsA.addCode(self.localReadDo(kernel, plrIdx, iui, 0,((u+pflr)*kernel["InnerUnroll"]+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersA))
                localReadsB.addCode(self.localReadDo(kernel, plrIdx, iui, 0,((u+pflr)*kernel["InnerUnroll"]+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersB))

              # Don't increment the LRO if we are going to reset them below:
              if not isResetLroIter or iui != kernel["InnerUnroll"]-1:
                localReads.addText(self.comment("local read increment a"))
                localReads.addText(self.localReadInc(kernel, iui, tensorParametersA))
                localReads.addText(self.comment("local read increment b"))
                localReads.addText(self.localReadInc(kernel, iui, tensorParametersB))

        pointerCode = Code.Module()
        waitCode = Code.Module()  # may be overwritten (not added to) below
        macIterCode = Code.Module()
        waitLWCode = Code.Module()
        syncCode = Code.Module()
        # put barrier at localWriteEndIter+1
        if u == localWriteEndIter+1:
          if self.enable["Wait"]:
            waitLWCode.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, \
                "3wait for local write"))
          if self.enable["Sync"]:
            syncCode.addCode(self.syncThreads(kernel))

        if isResetLroIter: # ResetLroIter
          if kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"]:
            if self.enable["LocalWrite"]:
              # local write for next iter, used to have local writes here
              pointerCode.addText(self.comment("local write swap offsets a"))
              pointerCode.addText(self.localWriteSwapOffsets(kernel, tensorParametersA))
              pointerCode.addText(self.comment("local write swap offsets b"))
              pointerCode.addText(self.localWriteSwapOffsets(kernel, tensorParametersB))
              pointerCode.addText(self.localWriteInitPointers(kernel, tensorParametersA))
              pointerCode.addText(self.localWriteInitPointers(kernel, tensorParametersB))

          if self.enable["LocalRead"]:
            # Swap, reset, or increment the LRO:
            if kernel["PrefetchGlobalRead"]:
              pointerCode.addText(self.comment("local read swap offsets a"))
              pointerCode.addText(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
              pointerCode.addText(self.comment("local read swap offsets b"))
              pointerCode.addText(self.localReadSwapOffsets(kernel, expand, tensorParametersB))

            pointerCode.addText(self.comment("local read init pointers a"))
            pointerCode.addText(self.localReadInitPointers(kernel, tensorParametersA))
            pointerCode.addText(self.comment("local read init pointers b"))
            pointerCode.addText(self.localReadInitPointers(kernel, tensorParametersB))
          else:
            # local read inc
            pointerCode.addText(self.comment("local read inc a"))
            pointerCode.addText(self.localReadInc(kernel, iui, tensorParametersA))
            pointerCode.addText(self.comment("local read inc b"))
            pointerCode.addText(self.localReadInc(kernel, iui, tensorParametersB))

          waitGlobalRead = -1
          if kernel["PrefetchGlobalRead"] and isResetLroIter:
            waitLocalWrite = 1
          else:
            waitLocalWrite = -1
          waitLocalRead  = 1 if isResetLroIter else 0

        else: # not isResetLroIter
          waitGlobalRead = 1 if u==0 and kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"] else -1
          waitLocalWrite = -1
          waitLocalRead  = 1 if kernel["PrefetchLocalRead"] else 0

        if self.enable["Wait"]:
          waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, \
              waitGlobalRead, waitLocalWrite, waitLocalRead, \
              "wait for prior local read local write")

        if self.enable["MAC"]:
          luIdx = (u) % (kernel["PrefetchLocalRead"]+1) # local to use for MACs
          if kernel["MatrixInstruction"]:
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
        if kernel["PrefetchLocalRead"] or (not globalParameters["UnrollLoopEfficiencyEnable"]):
          subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                          u, pointerCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx])
          kl.append(subIterCode) # add scheduled "other", local reads, local writes
        else:
          macIterCode = Code.Module()
          MacitemsReorder = []
          if self.enable["MAC"]:
            luIdx = (u) % (kernel["PrefetchLocalRead"]+1) # local to use for MACs
            macIterCode.addCode(self.macCode(kernel, luIdx, kernel["InnerUnroll"] ))
          MacIteritems = macIterCode.flatitems()
          #remove last and second entry from list if AggressiveMode is set
          # re-insert them back later
          if (kernel["AggressivePerfMode"]):
            MacIteritems = MacIteritems[:-1]
            MacIteritems.pop(1)
          #print("number MacItems\n",len(MacIteritems))
          waitGlobalRead = 1 if u==0 and kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"] else -1
          blockWidth = tensorParametersA["localReadInstruction"].blockWidth
          numVectorsPerTileA = (kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]]/kernel["VectorWidth"])
          numReadsPerVectorA = (kernel["VectorWidth"] * tensorParametersA["bpe"] ) / (blockWidth*4)
          numVectorsPerTileB = (kernel["ThreadTile%u"%tensorParametersB["tensorIdx"]]/kernel["VectorWidth"])
          TotalnumLdsFetches = numVectorsPerTileA*numReadsPerVectorA + numVectorsPerTileB*numReadsPerVectorA
          waitLocalWrite = -1
          waitLocalRead  = 0
          ## Rules for applying kernel["UnrollLoopEfficiencyEnable"]
          ## if A+B fetches <= 3 no split approach
          if not TotalnumLdsFetches > 3:
            subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                         u, pointerCode, waitCode, macIterCode)
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
                         u, pointerCode, waitCode, macIterCodeGrp)
              else:
                  #last group only pointer + localWrite Code
                if self.enable["Wait"]:
                  waitCode = Code.WaitCnt(self.version, waitCntItems[iter],-1,"wait for prior local read & local writes")
                subIterCode.addCode(waitCode)
                subIterCode.addCode(macIterCodeGrp)
              kl.append(subIterCode) # add scheduled "other", local reads, local writes
      kl.append(self.closeString(kernel))
      kl.append(self.openString(kernel))

      ####################################
      # unrolled loop: last summation iter
      ####################################
      # if prefetch-local: read red for 1st unroll of next iter
      # if not prefetch-local: read for current unroll of current iter
      unrollIter = kernel["LoopIters"]-1
      kl.append(self.comment("iter %u (last)"%unrollIter))
      if kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"] == 1:
        if self.enable["Wait"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, \
              "3wait for local write"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel))

      localReads = Code.Module()
      if not kernel["PrefetchLocalRead"] or kernel["PrefetchGlobalRead"]:
        plrIdx = (unrollIter+pflr) % (kernel["PrefetchLocalRead"] + 1)
        pack[plrIdx] = Code.Module()
        for iui in range(0,kernel["InnerUnroll"]):
          if self.enable["LocalRead"]:
            if kernel["MatrixInstruction"] and \
              (kernel["ProblemType"]["DataType"].isBFloat16() or kernel["ProblemType"]["DataType"].isHalf()) and \
              not kernel["TransposeLDS"] :
              # Reading 16-bit data from LDS requires packing when ECC enabled
              localReads.addText(self.comment("local read a"))
              localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx, iui, 0, (((unrollIter+pflr)*kernel["InnerUnroll"])+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersA)
              localReads.addCode(localReadCodeA)
              localReads.addText(self.comment("local read b"))
              localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx, iui, 0, (((unrollIter+pflr)*kernel["InnerUnroll"])+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersB)
              localReads.addCode(localReadCodeB)
              pack[plrIdx].addCode(packCodeA)
              pack[plrIdx].addCode(packCodeB)
            else:
              localReads.addText(self.comment("local read a"))
              localReads.addCode(self.localReadDo(kernel, plrIdx, iui, 0, (((unrollIter+pflr)*kernel["InnerUnroll"])+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersA))
              localReads.addText(self.comment("local read b"))
              localReads.addCode(self.localReadDo(kernel, plrIdx, iui, 0, (((unrollIter+pflr)*kernel["InnerUnroll"])+iui)%(kernel["LoopIters"]*kernel["InnerUnroll"]), tensorParametersB))
            if kernel["InnerUnroll"] and iui != kernel["InnerUnroll"]-1:
              localReads.addText(self.comment("unroll increments:"))
              localReads.addText(self.comment("local read inc a"))
              localReads.addText(self.localReadInc(kernel, iui, tensorParametersA))
              localReads.addText(self.comment("local read inc b"))
              localReads.addText(self.localReadInc(kernel, iui, tensorParametersB))

      pointerCode = Code.Module()
      waitCode = Code.Module()  # may be overwritten (not added to) below
      macIterCode = Code.Module()
      waitLWCode = Code.Module()
      syncCode = Code.Module()
      if kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"]:
        if self.enable["LocalRead"]:
          # local read inc
          pointerCode.addText(self.comment("local read inc a"))
          pointerCode.addText(self.localReadInc(kernel, iui, tensorParametersA))
          pointerCode.addText(self.comment("local read inc b"))
          pointerCode.addText(self.localReadInc(kernel, iui, tensorParametersB))
      elif kernel["PrefetchGlobalRead"] and not kernel["PrefetchLocalRead"]:
        # For PGR=1 PLR=0, writes are scheduled above through the subIter scheduler
        if self.enable["LocalWrite"]:
          pointerCode.addText(self.comment("local write swap offsets a"))
          pointerCode.addText(self.localWriteSwapOffsets(kernel, tensorParametersA))
          pointerCode.addText(self.comment("local write swap offsets b"))
          pointerCode.addText(self.localWriteSwapOffsets(kernel, tensorParametersB))
          pointerCode.addText(self.localWriteInitPointers(kernel, tensorParametersA))
          pointerCode.addText(self.localWriteInitPointers(kernel, tensorParametersB))
        if self.enable["LocalRead"]:
          # swap read and write
          pointerCode.addText(self.comment("local read swap offsets a"))
          pointerCode.addText(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
          pointerCode.addText(self.comment("local read swap offsets b"))
          pointerCode.addText(self.localReadSwapOffsets(kernel, expand, tensorParametersB))
          pointerCode.addText(self.comment("local read init pointers a"))
          pointerCode.addText(self.localReadInitPointers(kernel, tensorParametersA))
          pointerCode.addText(self.comment("local read init pointers b"))
          pointerCode.addText(self.localReadInitPointers(kernel, tensorParametersB))
        if self.enable["Wait"]:
          waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, -1, 1, 0, \
              "6wait for local read")
      elif not kernel["PrefetchGlobalRead"] and not kernel["PrefetchLocalRead"]:
        if self.enable["LocalRead"]:
          # local read init ptrs
          pointerCode.addText(self.comment("local read init pointers a"))
          pointerCode.addText(self.localReadInitPointers(kernel, tensorParametersA))
          pointerCode.addText(self.comment("local read init pointers b"))
          pointerCode.addText(self.localReadInitPointers(kernel, tensorParametersB))
        if self.enable["Wait"]:
          waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "1wait for local read")
      elif not kernel["PrefetchGlobalRead"] and kernel["PrefetchLocalRead"]:
        if self.enable["Wait"]:
          waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "2wait for local read")
      else:
        assert(0) # unknown PGR/PLR pattern
      # no wait needed here b/c we already waited for ds_write
      # which waited for this ds_read
      if self.enable["MAC"]:
        luIdx = (unrollIter) % (kernel["PrefetchLocalRead"] + 1)
        if kernel["MatrixInstruction"]:
          macIterCode.addCode(self.mfmaIter(kernel, luIdx, kernel["InnerUnroll"]))
        else:
          macIterCode.addCode(self.macIter(kernel, luIdx, kernel["InnerUnroll"], True))

      subIterCode = self.makeSubIterSchedule(kernel, localReads,
                            unrollIter, pointerCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx])
      kl.append(subIterCode)

      # close unrolled loop
      if expand:
        if not finalLoop:
          kl.append(self.comment3("Unrolled Loop - End %u/%u"%(lc+1, loopCopies)))
        else:
          kl.append(self.comment3("Unrolled Loop - End %u/%u (final)"%(lc+1, loopCopies)))
      else:
        kl.append(self.comment3("Unrolled Loop - End"))

      kl.append(self.closeLoop(kernel, self.unrollIdx, finalLoop))

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
          import copy
          deepCopyPack = copy.deepcopy(pack)
          kl += self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=True, pack=deepCopyPack)
          self.restoreLocalPointers(kernel)

        kl += self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, pack=pack)
      # if PGR, last few iterations will have PLR,
      # and those PLR will not be used(register not checkIn) if without NoLoadLoop
      else:
        for i in range(kernel["PrefetchLocalRead"]):
          for item in list(pack[i].items()):
            self.vgprPool.checkIn(item.tempVgpr)

    if self.staggerU and self.actualSummationLoops>1:
      kl.append(self.comment("remove stagger offsets"))
      kl.append(self.removeStagger(kernel, tensorParametersA))
      kl.append(self.removeStagger(kernel, tensorParametersB))


    ########################################
    # Tail Loop
    # PackSummationDims=1 requires that the tile slice does not cross DepthU
    # which means tail loop not needed.
    ########################################
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
      if self.enable["LocalWrite"]:
        # tail: local write
        kl.append(self.localWriteInitPointers(kernel, tensorParametersA))
        kl.append(self.localWriteInitPointers(kernel, tensorParametersB))
        kl.append(self.comment("local write a"))
        kl.append(self.localWriteDo(kernel, tensorParametersA))
        kl.append(self.comment("local write b"))
        kl.append(self.localWriteDo(kernel, tensorParametersB))
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
      kl.append(self.openLoop(kernel, -1))
      # Try to use InnerUnroll in the tail loop if allowed:
      k = kernel["MatrixInstK"] if kernel["MatrixInstruction"] else 1
      tailLoopInnerUnroll = \
        kernel["InnerUnroll"] if (kernel["AssertSummationElementMultiple"] % (kernel["InnerUnroll"]*k)==0) else 1
      
      pack[0] = Code.Module()
      for iui in range(0,tailLoopInnerUnroll):
        if self.enable["LocalRead"]:
          if kernel["MatrixInstruction"] and \
            (kernel["ProblemType"]["DataType"].isBFloat16() or kernel["ProblemType"]["DataType"].isHalf()) and \
            not kernel["TransposeLDS"] :
            # Reading 16-bit data from LDS requires packing when ECC enabled
            kl.append(self.comment("local read a"))
            localReadCodeA, packCodeA = self.localReadDo(kernel, 0, iui, 0, 0, tensorParametersA)
            kl.append(localReadCodeA)
            kl.append(self.comment("local read b"))
            localReadCodeB, packCodeB = self.localReadDo(kernel, 0, iui, 0, 0, tensorParametersB)
            kl.append(localReadCodeB)
            pack[0].addCode(packCodeA)
            pack[0].addCode(packCodeB)
          else:
            kl.append(self.comment("local read a"))
            kl.append(self.localReadDo(kernel, 0, iui, 0, 0, tensorParametersA))
            kl.append(self.comment("local read b"))
            kl.append(self.localReadDo(kernel, 0, iui, 0, 0, tensorParametersB))
          kl.append(self.comment("local read inc a"))
          kl.append(self.localReadInc(kernel, iui, tensorParametersA))
          kl.append(self.comment("local read inc b"))
          kl.append(self.localReadInc(kernel, iui, tensorParametersB))
      if self.enable["Wait"]:
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "4wait for local read"))
      if self.enable["MAC"]:
        if kernel["MatrixInstruction"]:
          kl.append(pack[0])
          for item in list(pack[0].items()):
            self.vgprPool.checkIn(item.tempVgpr)
          kl.append(self.mfmaIter(kernel, 0, tailLoopInnerUnroll))
        else:
          kl.append(self.macIter(kernel, 0, tailLoopInnerUnroll, True))

      # tail: close
      kl.append(self.closeLoop(kernel, -1, True))

    # extra summation loops: global increment and close
    for i in reversed(range(self.otherSummationLoops)):
      kl.append(self.comment("global read inc AB"))
      kl.append(self.globalReadIncrementAB(kernel, i, 0))
      kl.append(self.closeLoop(kernel, i, True))

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
          if kernel["MatrixInstruction"]:
            kl.append(self.shiftVectorComponentsForMatrixInst(kernel, tensorParametersA))
          else:
            kl.append(self.shiftVectorComponents(kernel, tensorParametersA))
        # shift vector components d1
        if not kernel["GuaranteeNoPartialB"] and self.readTileDimVectorB and not kernel["MatrixInstruction"]:
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

    ########################################
    # read / write vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"]: # NT no transpose
      self.numReadsTileA = kernel["NumLoadsCoalescedA"]
      self.numReadsUnrollA = kernel["NumLoadsPerpendicularA"]
      self.numWritesCoalA = kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vectors, write vectors
        self.readTileDimComponentsA = False # Vector
        self.readTileDimVectorA = True # Vector
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        self.numReadsTileVecCompA = vwa
        self.numReadsUnrollVecCompA = 1

        self.writeUnrollDimComponentsA = False # Scalar
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsA = kernel["GlobalReadVectorWidth"] > 1 # Components
          writeCoal = False
        else:
          self.writeTileDimComponentsA = False # Vector
          writeCoal = True
      else: # read components, write components
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorA = False # Components
        self.numReadsTileVecCompA = 1
        self.numReadsUnrollVecCompA = vwa

        self.writeTileDimComponentsA = False # Scalar
        self.writeUnrollDimComponentsA = kernel["GlobalReadVectorWidth"] > 1 # Components
        writeCoal = False
    else: # TN yes transpose
      self.numReadsTileA = kernel["NumLoadsPerpendicularA"]
      self.numReadsUnrollA = kernel["NumLoadsCoalescedA"]
      self.numWritesCoalA = kernel["NumLoadsPerpendicularA"] if not kernel["TransposeLDS"]  else kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vector, write components
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = False # Vector
        self.readUnrollDimVectorA = True # Vector
        self.numReadsUnrollVecCompA = vwa
        self.numReadsTileVecCompA = 1

        self.writeUnrollDimComponentsA = False # Scalar
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsA = kernel["GlobalReadVectorWidth"] > 1 # Components
          # LDS writes with LDL>1 will never be coalesced
          writeCoal = False
        else:
          self.writeTileDimComponentsA = kernel["GlobalReadVectorWidth"] > 1 if not kernel["TransposeLDS"] else False# Components
          self.writeUnrollDimComponentsA = False  if not kernel["TransposeLDS"] else kernel["GlobalReadVectorWidth"] > 1 # Scalar
          writeCoal = False if not kernel["TransposeLDS"] else True
      else: # read components, write vectors
        self.readTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        self.readTileDimVectorA = False # Components
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        # NEW
        self.numReadsUnrollVecCompA = 1
        self.numReadsTileVecCompA = vwa
        self.writeTileDimComponentsA = False # Vector
        self.writeUnrollDimComponentsA = False # Scalar
        writeCoal = True

    # writeCoal indicates writes should be done in the coal dim
    # else in perp
    if writeCoal:
      self.numWritesCoalVecCompA = vwa
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
    # read / write vectors or vector components b
    ####################################
    if kernel["ProblemType"]["TLUB"]: # NT no transpose
      self.numReadsTileB = kernel["NumLoadsCoalescedB"]
      self.numReadsUnrollB = kernel["NumLoadsPerpendicularB"]
      self.numWritesCoalB = kernel["NumLoadsCoalescedB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Vector
        self.readTileDimVectorB = True # Vector
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        self.numReadsTileVecCompB = vwb
        self.numReadsUnrollVecCompB = 1
        self.writeUnrollDimComponentsB = False # Vector
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsB = kernel["GlobalReadVectorWidth"] > 1 # Components
          writeCoal = False
        else:
          self.writeTileDimComponentsB = False # Vector
          writeCoal = True
      else:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
        self.readUnrollDimVectorB = False # Components
        self.writeTileDimComponentsB = False # Scalar
        self.writeUnrollDimComponentsB = kernel["GlobalReadVectorWidth"] > 1 # Components
        # NEW
        self.numReadsTileVecCompB = 1
        self.numReadsUnrollVecCompB = vwb
        self.numWritesCoalVecCompB = 1
        self.numWritesPerpVecCompB = vwb
    else: # TN yes transpose
      self.numReadsTileB = kernel["NumLoadsPerpendicularB"]
      self.numReadsUnrollB = kernel["NumLoadsCoalescedB"]
      self.numWritesCoalB = kernel["NumLoadsPerpendicularB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = False # Vector
        self.readUnrollDimVectorB = True # Vector
        self.numReadsUnrollVecCompB = vwb
        self.numReadsTileVecCompB = 1
        self.writeUnrollDimComponentsB = False
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsB = kernel["GlobalReadVectorWidth"] > 1 # Components
          # LDS writes with LDL>1 will never be coalesced%
          writeCoal = False
        else:
          self.writeTileDimComponentsB = kernel["GlobalReadVectorWidth"] > 1 if not kernel["TransposeLDS"] else False # Components
          self.writeUnrollDimComponentsB = False if not kernel["TransposeLDS"] else kernel["GlobalReadVectorWidth"] > 1
          writeCoal = False if not kernel["TransposeLDS"] else True
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

    # writeCoal indicates writes should be done in the coal dim
    # else in perp
    if writeCoal:
      self.numWritesCoalVecCompB = vwb
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
  def openLoop(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Close Loop
  ##############################################################################
  @abc.abstractmethod
  def closeLoop(self, kernel, loopIdx, finalLoop):
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
  def closeSumAtLeastUnroll(self, kernel, prefetch, isOptNLL):
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
  def localWriteDo(self, kernel, tP):
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
  def localReadDo(self, kernel, bufferIdx, innerUnrollIndex, epsi, uIdx, tP):
    return ""

  ##############################################################################
  # Shift Vector Components d0/1
  ##############################################################################
  @abc.abstractmethod
  def shiftVectorComponents(self, kernel, tP):
    return ""

  @abc.abstractmethod
  def shiftVectorComponentsForMatrixInst(self, kernel, tP):
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
  def openPrefetchAcrossPersistent(self, kernel):
    return ""

  @abc.abstractmethod
  def closePrefetchAcrossPersistent(self, kernel):
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
        self.writeByteArrayScript()

        asmPath = self.getAssemblyDirectory()
        coFile = self.getSingleCodeObjectFile(kernel)
        kernelName = self.getKernelName(kernel)

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

  ##############################################################################
  #
  #   Beta-Only Kernels
  #
  # kernel dictionary has ProblemType for indices and Beta=True/False
  ##############################################################################

  ##############################################################################
  # Get Name
  ##############################################################################
  @staticmethod
  def getKernelNameBetaOnly(kernel):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += kernel["ProblemType"]["DestDataType"].toChar()
    if kernel["ProblemType"]["UseBeta"]: name += "B"

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
    return (0,fileString)

  def getHeaderFileStringBetaOnly(self, kernel):
    kernelName = self.getKernelNameBetaOnly(kernel)
    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += CHeader
      fileString += "#pragma once\n\n"
      fileString += "\n"
      fileString += "#include <KernelHeader.h>\n\n"
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
