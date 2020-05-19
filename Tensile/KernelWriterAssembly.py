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
from .Common import globalParameters, printExit, printWarning, roundUp
from .KernelWriter import KernelWriter
from .SolutionStructs import isPackedIndex
from .Utils import ceil_divide, roundUpToNearestMultiple

from warnings import warn
from math import log, ceil, trunc, modf
from copy import deepcopy
import collections
import traceback
from enum import Enum

################################################################################
# Memory Instruction
################################################################################
class MemoryInstruction:
  def __init__(self, name, numAddresses, numOffsets, \
      offsetMultiplier, blockWidth, formatting):
    self.name = name
    self.formatting = formatting
    self.numAddresses = numAddresses
    self.numOffsets = numOffsets
    self.offsetMultiplier = offsetMultiplier
    self.blockWidth = blockWidth
    self.numBlocks = 2 if self.numAddresses > 1 or self.numOffsets > 1 else 1
    self.totalWidth = self.blockWidth * self.numBlocks
    #in Quad-Cycle
    if (name == "ds_read_b128"):
      self.IssueLatency = 2
    elif (name == "ds_write_b128"):
      self.IssueLatency = 5
    elif (name == "ds_write2_b64"):
      self.IssueLatency = 3
    elif (name == "ds_write_b64"):
      self.IssueLatency = 3
    elif (name == "ds_write2_b32"):
      self.IssueLatency = 3
    elif (name == "ds_write_b32"):
      self.IssueLatency = 2
    elif (name == "ds_write_u16") :
      self.IssueLatency = 2
    else:
      self.IssueLatency = 1
    self.endLine = "\n"
  ########################################
  # write in assembly format
  def toString(self, params, comment, nonTemporal=0, highBits=0):
    name = self.name
    if highBits:
      name += "_d16_hi"
    instStr = "%s %s" % (name, (self.formatting % params) )
    if nonTemporal%2==1:
      instStr += " glc"
    if nonTemporal//2==1:
      instStr += " slc"
    line = "%-50s // %s%s" % (instStr, comment, self.endLine)
    return line

  # Like toString, but don't add a comment or newline
  # Designed to feed into Code.Inst constructors, somewhat
  def toCodeInst(self, params, nonTemporal=0, highBits=0):
    name = self.name
    if highBits:
      name += "_d16_hi"
    instStr = "%s %s" % (name, (self.formatting % params) )
    if nonTemporal%2==1:
      instStr += " glc"
    if nonTemporal//2==1:
      instStr += " slc"
    line = "%-50s" % (instStr)
    return line


  def __str__(self):
    return self.name

################################################################################
# RegisterPool
# Debugging register performance problems:
# - Enable self.db["PrintRP" to see messages as vgprPool state changes.
# - Search for 'overlow' to see when pool grows dynamically - typically this
#   indicates growth for temps or other cases.
# - checkIn, checkout take optional tag but this is not widely used in tensile.
# - checkout returns vgpr index that was returned - can search disasm to see where
#   this vgpr is used.
################################################################################
class RegisterPool:
  statusUnAvailable = 0
  statusAvailable = 1
  statusInUse = 2

  class Register:
    def __init__(self, status, tag):
      self.status = status
      self.tag = tag

  ########################################
  # Init
  # reservedAtEnd: reserve a register at the end of the poool, used for vserial
  # defaultPreventOverflow: control behavior of checkout and checkoutAligned when preventOverflow is not explicitly specificed.
  def __init__(self, size, type, reservedAtEnd, defaultPreventOverflow, printRP=0):
    self.printRP=printRP
    self.type = type
    self.reservedAtEnd = reservedAtEnd
    self.defaultPreventOverflow = defaultPreventOverflow
    self.pool = [self.Register(self.statusUnAvailable, "init") for i in range(0,size)]
    self.checkOutSize = {}

  ########################################
  # Adds registers to the pool so they can be used as temps
  # Convenience function that takes a range and returns it in string form
  def addRange(self, start, stop, tag=""):
    self.add(start, stop-start+1, tag)
    if (start == stop):
      return "%d"%(start)
    else:
      return "%d-%d" % (start, stop)

  ########################################
  # Adds registers to the pool so they can be used as temps
  # Add
  def add(self, start, size, tag=""):
    # reserve space
    if self.printRP:
      print("RP::add(%u..%u for '%s')"%(start,start+size-1,tag))
    newSize = start + size
    oldSize = len(self.pool)
    if newSize > oldSize:
      for i in range(0, newSize-oldSize):
        self.pool.addInst(self.Register(self.statusUnAvailable,tag))
    # mark as available
    for i in range(start, start+size):
      if self.pool[i].status == self.statusUnAvailable:
        self.pool[i].status = self.statusAvailable
      elif self.pool[i].status == self.statusAvailable:
        warn("RegisterPool::add(%u,%u) pool[%u] already available" % (start, size, i))
      elif self.pool[i].status == self.statusInUse:
        warn("RegisterPool::add(%u,%u) pool[%u] already in use" % (start, size, i))
      else:
        raise RuntimeError("RegisterPool::add(%u,%u) pool[%u] = %s" % (start, size, i, self.pool[i].status))
    if self.printRP:
      print(self.state())
  ########################################
  # Remove
  # Removes registers from the pool so they cannot be subsequently allocated for tmps
  def remove(self, start, size, tag=""):
    if self.printRP:
      print("RP::remove(%u..%u) for %s"%(start,size-1,tag))
    # reserve space
    newSize = start + size
    oldSize = len(self.pool)
    if newSize > oldSize:
      printWarning("RegisterPool::remove(%u,%u) but poolSize=%u" % (start, size, oldSize))
    # mark as unavailable
    for i in range(start, start+size):
      if  self.pool[i].status == self.statusAvailable:
        self.pool[i].status = self.statusUnAvailable
      elif self.pool[i].status == self.statusUnAvailable:
        printWarning("RegisterPool::remove(%u,%u) pool[%u] already unavailable" % (start, size, i))
      elif  self.pool[i].status == self.statusInUse:
        printWarning("RegisterPool::remove(%u,%u) pool[%u](%s) still in use" % (start, size, i, self.pool[i].tag))
      else:
        printExit("RegisterPool::remove(%u,%u) pool[%u] = %s" % (start, size, i, self.pool[i].status))

  ########################################
  # Check Out
  def checkOut(self, size, tag="_untagged_", preventOverflow=-1):
    return self.checkOutAligned(size, 1, tag, preventOverflow)

  def checkOutAligned(self, size, alignment, tag="_untagged_aligned_", preventOverflow=-1):
    if preventOverflow == -1:
      preventOverflow = self.defaultPreventOverflow
    assert(size > 0)
    found = -1
    for i in range(0, len(self.pool)):
      # alignment
      if i % alignment != 0:
        continue
      # enough space
      if i + size > len(self.pool):
        continue
      # all available
      allAvailable = True
      for j in range(0, size):
        if self.pool[i+j].status != self.statusAvailable:
          allAvailable = False
          i = j+1
          break
      if allAvailable:
        found = i
        break
      else:
        continue

    # success without overflowing
    if found > -1:
      #print "Found: %u" % found
      for i in range(found, found+size):
        self.pool[i].status = self.statusInUse
        self.pool[i].tag = tag
      self.checkOutSize[found] = size
      if self.printRP:
        print("RP::checkOut '%s' (%u,%u) @ %u avail=%u"%(tag, size,alignment, found, self.available()))
        #print self.state()
      return found
    # need overflow
    else:
      #print "RegisterPool::checkOutAligned(%u,%u) overflowing past %u" % (size, alignment, len(self.pool))
      # where does tail sequence of available registers begin
      assert (not preventOverflow)
      start = len(self.pool)
      for i in range(len(self.pool)-1, 0, -1):
        if self.pool[i].status == self.statusAvailable:
          self.pool[i].tag = tag
          start = i
          continue
        else:
          break
      #print "Start: ", start
      # move forward for alignment

      start = roundUpToNearestMultiple(start,alignment)
      #print "Aligned Start: ", start
      # new checkout can begin at start
      newSize = start + size
      oldSize = len(self.pool)
      overflow = newSize - oldSize
      #print "Overflow: ", overflow
      for i in range(start, len(self.pool)):
        self.pool[i].status = self.statusInUse
        self.pool[i].tag = tag
      for i in range(0, overflow):
        if len(self.pool) < start:
          # this is padding to meet alignment requirements
          self.pool.append(self.Register(self.statusAvailable,tag))
        else:
          self.pool.append(self.Register(self.statusInUse,tag))
      self.checkOutSize[start] = size
      if self.printRP:
        print(self.state())
        print("RP::checkOut' %s' (%u,%u) @ %u (overflow)"%(tag, size, alignment, start))
      return start

  def initTmps(self, initValue, start=0, stop=-1):
    kStr = ""
    stop= len(self.pool) if stop== -1 or stop>len(self.pool) else stop+1
    for i in range(start, stop):
      #if self.type == 's':
      #  print i, self.pool[i].status
      if self.pool[i].status==self.statusAvailable:
        if self.type == 's':
          kStr += inst("s_mov_b32", sgpr(i), hex(initValue), "init tmp in pool")
        elif self.type == 'v':
          kStr += inst("v_mov_b32", vgpr(i), hex(initValue), "init tmp in pool")
        else:
          assert(0) # bad regpool type

    return kStr

  ########################################
  # Check In
  def checkIn(self, start):
    if start in self.checkOutSize:
      size = self.checkOutSize[start]
      for i in range(start, start+size):
        self.pool[i].status = self.statusAvailable
      self.checkOutSize.pop(start)
      if self.printRP:
        print("RP::checkIn('%s') @ %u +%u"%(self.pool[i].tag, start,size))
    else:
      if 0:
        traceback.print_stack(None)
        import pdb; pdb.set_trace()
      printWarning("RegisterPool::checkIn('%s',%s) but it was never checked out"%(self.pool[start].tag, start))
    #traceback.print_stack(None)

  ########################################
  # Size
  def size(self):
    return len(self.pool) + self.reservedAtEnd


  ########################################
  # Number of available registers
  def available(self):
    numAvailable = 0
    for s in self.pool:
      if s.status == self.statusAvailable:
        numAvailable += 1
    return numAvailable

  ########################################
  # Size of registers of at least specified blockSize
  def availableBlock(self, blockSize):
    if blockSize ==0:
      blockSize = 1
    blocksAvail = 0
    consecAvailable = 0
    for s in self.pool:
      if s.status == self.statusAvailable:
        consecAvailable += 1
      else:
        blocksAvail += consecAvailable // blockSize
        consecAvailable = 0
    blocksAvail += consecAvailable // blockSize
    #print self.state()
    #print "available()=", self.available(), "availableBlock()=",maxAvailable
    return blocksAvail * blockSize

  def availableBlockAtEnd(self):
    availCnt = 0
    for s in reversed(self.pool):
      if s.status == self.statusAvailable:
        availCnt += 1
      else:
        break

    return availCnt


  ########################################
  def checkFinalState(self):
    for si in range(0,len(self.pool)):
      if self.pool[si].status == self.statusInUse:
        if self.printRP:
          print(self.state())
        raise RuntimeError("RegisterPool::checkFinalState: temp (%s, '%s') was never checked in." \
            %(si, self.pool[si].tag))

  ########################################
  # State
  def state(self):
    stateStr = ""
    placeValues = [1000, 100, 10, 1]
    for placeValueIdx in range(1, len(placeValues)):
      placeValue = placeValues[placeValueIdx]
      priorPlaceValue = placeValues[placeValueIdx-1]
      if len(self.pool) >= placeValue:
        pvs = "" # place value string
        for i in range(0, len(self.pool)):
          if i % placeValue==0:
            pvs += "%u"%((i%priorPlaceValue)//placeValue)
          else:
            pvs += " "
        stateStr += pvs + "\n"
    for i in range(0, len(self.pool)):
      if self.pool[i].status == self.statusUnAvailable:
        stateStr += "." # 'removed', this indicates a fixed assignment from "remove", ie a non-tmp allocation 
      elif self.pool[i].status == self.statusAvailable:
        stateStr += "|" # Can be allocated
      elif self.pool[i].status == self.statusInUse:
        stateStr += "#" # Checked out
    return stateStr



class ZeroPadReg:
  class State(Enum):
    Allocated=0
    MacroDef=1
    CalculatedAddr=2
  def __init__(self, zp, regName, vgprIdx, perp, sPerp, para, sPara):
    self.zp = zp
    self.state = ZeroPadReg.State.Allocated
    self.regName = regName
    self.vgprIdx= vgprIdx
    self.perp = perp
    self.sPerp = sPerp
    self.para = para
    self.sPara = sPara

  def isMatch(self, perp, sPerp, para, sPara):
    return self.perp==perp and self.sPerp==sPerp and self.para==para and self.sPara==sPara



################################################################################
# Assembly Kernel
################################################################################
class KernelWriterAssembly(KernelWriter):

  ##############################################################################
  # Init
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    super(KernelWriterAssembly, self).__init__( \
        kernelMinNaming, kernelSerialNaming)
    self.do = {}

    self.do["PreLoop"]     = True
    self.do["GlobalReadA"] = True
    self.do["GlobalReadB"] = True
    self.do["GlobalInc"]   = True
    self.do["LocalWrite"]  = True
    self.do["LocalReadA"]  = True
    self.do["LocalReadB"]  = True
    self.do["Wait"]        = True
    self.do["Sync"]        = True
    self.do["MAC"]         = True
    self.do["PostLoop"]    = True
    self.do["ApplyAlpha"]  = True
    self.do["GlobalWrite"] = True

    self.do["EdgeWrite"]   = True

    self.do["KeepDirectToLdsAlloc"] = False  # If true, keep regs used for LDS alloc even if not used

    # Remove me if 906 can work with beta in SGPR
    # Also can push alpha/beta recalc back to host for HPA mode
    self.betaInSgpr = True

    # Various debug flags and modes
    self.db = {}
    self.db["EnableAsserts"]     = False  # Enable assertion codegen. Requires 2 SGPR.
    self.db["DebugKernelMaxItems"] = 16  # Capture first N(=16) print values, ignore subsequent.  If -1, debug writing is faster but writing more than 16 values is undefined.

    # Chicken bit to add conservative synchronization at strategic points:
    # 0x01 = waitcnt + barrier after vector load
    # 0x02 = waitcnt at self.wait() for globalRead
    # 0x04 = waitcnt at self.wait() for localWrite
    # 0x08 = waitcnt at self.wait() for localRead
    # 0x10 = waitcnt after summation iteration, this can catch lingering ds or vm activity from summation loop
    # 0x20 = waitcnt before each write batch
    # 0x40 = waitcnt after each write batch
    self.db["ConservativeWaitCnt"] = 0x00

    self.db["InitLds"]     = False  # Initialize LDS at start of kernel
    self.printedAssertCnt  = 0
    self.initLdsValue     = 0xFFFFFFFF  # Value to use for LDS Init, if enabled

    # InitSgpr and InitVgpr can initialize at various points:
    #  0x1: Init at kernel start
    #  0x2: Init at end of summation loop (after tail too) - this is just before store loop
    self.db["InitSgpr"]   = 0x0  # init SGPRs
    self.initSgprValue    = 0x0  # Value to use for Sgpr Init, if enabled

    self.db["InitVgpr"]   = 0x0  # init VGPRs
    self.initVgprValue    = 0xFFFFFFFF  # Value to use for Sgpr Init, if enabled

    # Debug and Check flags:
    # Check A and B values loaded from memory to ensure they are 1
    # Requires DataInitTypeAB=1.
    # Only works if the problem uses full tiles (no edges)
    # Mismatches will assert (generate GPUVM fault)
    self.db["CheckValue1A"] = False
    self.db["CheckValue1B"] = False

    # Check value in C matrix.
    # Caveats:
    #  - Only works for single.
    #  - Checks after alpha calc for each element.  Later elements (in the TT) will not yet have applied their alpha.
    #  - Only works if matrix is integral multiple of macro-tile (no edges) - check is dumb so doesn't know
    #    which work-items are outside the valid edge.
    #  - Does not work in OptNoLoadLoop
    self.db["CheckValueC"]  = False
    # value expected if CheckValueC is set. Use '.' for FP.
    # For example could be 16.0 if U=8 and alpha=2
    self.db["ValueCExpectedValue"] = 16.0

    # Force an expected value for all C outputs.
    # May be useful for checking store path
    # See same caveats as CheckValueC
    self.db["ForceExpectedValue"]  = False

    # Force VSerial value into the output, this will
    # not match reference but can be useful to see which work-items are
    # storing which values
    # See same caveats as CheckValueC
    self.db["ForceVSerial"] = False

    # can't do both of these since they both override output
    assert (not (self.db["ForceExpectedValue"] and self.db["ForceVSerial"]))


    self.db["ForceInputValueA"] = False
    self.db["ForceInputValueB"] = False
    self.db["ForceValueA"] = 1.0
    self.db["ForceValueB"] = 1.0

    self.db["CheckStoreC"] = -1 # -1 disables, reload and verify output data.  Specify expected constant value.
    #self.db["CheckStoreC"] = 1024.0 # possible value

    self.db["ForceEdgeStores"] = 0 # 1=force use of edge store path for all tiles,  2=add assert in non-edge stores
    self.db["AssertNoEdge"] = 0 # Add assert in edge store code so crashes if executed

    # print vgpr register pool checkins and checkouts
    self.db["PrintRP"] = 0
    self.db["AssertOnSgprOverflow"] = False
    self.db["PrintStoreRegisterDb"] = False

    # Number of times localReadDo(localWriteDo) has been called by the code-generator.
    # Used to control debug enablement.
    # Note this increments as the assembly code is generated not as it executes
    # so it can be used to determine which iteration of the unroll is being generated
    self.localReadDoCnt   = 0
    self.localWriteDoCnt  = 0

    self.maxVgprs = 256
    # max allowed is 112 out of 112 , 6 is used by hardware 4 SGPRs are wasted
    self.maxSgprs = 102

    self.endLine = "\n"
    self.syncStr = "s_barrier"
    self.commentPrefix = "/*"
    self.commentSuffix = "*/"
    self.commentHR = "*"*40
    self.indent = ""
    self.labels = {}
    self.localReadOffsetA = 0
    self.localReadOffsetB = 0
    self.inTailLoop = False

    self.vgprOccupancy = [0]*(256+1)
    for i in range(0,   24+1): self.vgprOccupancy[i] = 10
    for i in range(25,  28+1): self.vgprOccupancy[i] = 9
    for i in range(29,  32+1): self.vgprOccupancy[i] = 8
    for i in range(33,  36+1): self.vgprOccupancy[i] = 7
    for i in range(37,  40+1): self.vgprOccupancy[i] = 6
    for i in range(41,  48+1): self.vgprOccupancy[i] = 5
    for i in range(49,  64+1): self.vgprOccupancy[i] = 4
    for i in range(65,  84+1): self.vgprOccupancy[i] = 3
    for i in range(85, 128+1): self.vgprOccupancy[i] = 2
    for i in range(129,256+1): self.vgprOccupancy[i] = 1

  def getCompileArgs(self, sourceFileName, objectFileName, *moreArgs):
    isa = self.version
    archHasV3 = globalParameters["AsmCaps"][isa]["HasCodeObjectV3"]

    rv = [globalParameters['AssemblerPath'],
          '-x', 'assembler',
          '-target', 'amdgcn-amd-amdhsa']

    if archHasV3:
      rv += ['-mno-code-object-v3' if globalParameters["CodeObjectVersion"] == "V2" else '-mcode-object-v3']

    rv += ['-mcpu=gfx' + ''.join(map(str,isa))]
    if isa == (10,1,0):
      rv += ['-mwavefrontsize64']

    rv += moreArgs

    rv += ['-c', '-o', objectFileName, sourceFileName]

    return rv

  def getLinkCodeObjectArgs(self, objectFileNames, coFileName, *moreArgs):
    rv = [globalParameters['AssemblerPath'],
          '-target', 'amdgcn-amd-amdhsa']

    rv += moreArgs

    rv += ['-o', coFileName] + objectFileNames

    return rv

  ########################################
  def getOccupancy(self, kernel, vgprs):
    multiplier = int(ceil(max(kernel["NumThreads"], 256) / 256.0))
    # example: wg=512 multiplier=2, 1024=4

    maxLds = 65536
    ldsSize = kernel["LdsNumElements"] * kernel["ProblemType"]["DataType"].numBytes()
    ldsSize = (ldsSize + 255) & 0x1ff00 # 256-byte granularity
    ldsLimitedOccupancy = int(ceil(maxLds / float(ldsSize)))

    vgprs *= multiplier
    vgprLimitedOccupancy =  self.vgprOccupancy[vgprs] if vgprs <= 256 else 0

    return min(ldsLimitedOccupancy, vgprLimitedOccupancy)

  def getMaxRegsForOccupancy(self, vgprs):
    initOccupancy =  self.vgprOccupancy[vgprs]
    lastVgprs = vgprs
    while vgprs < len(self.vgprOccupancy):
      vgprs += 1
      if self.vgprOccupancy[vgprs] == initOccupancy:
        lastVgprs = vgprs
        next
      else:
        break
    return lastVgprs

  ########################################
  ########################################
  def sizeRef(self, idx):
    """
    Return sgpr() or const with the specified size
    See above definitions for how these are mapped to Free or Sum sizes
    based on the problem definition.
    """
    idxChar= globalParameters["IndexChars"][idx]
    return sgpr("Size%s"%idxChar)

  def loopChar(self, kernel, loopIdx):
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    return globalParameters["IndexChars"][loopDim]


  def loopSizeRef(self, kernel, loopIdx):
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    return self.sizeRef(loopDim)

  def loopCounterName(self, kernel, loopIdx):
    return "LoopCounter%s"%(self.loopChar(kernel, loopIdx))

  def loopCounter(self, kernel, loopIdx):
    """
    Return loopCounter for loopIdx wrapped in "SGPR" syntax
    loop idx is 0...unrollIdx
    """
    return sgpr(self.loopCounterName(kernel,loopIdx))

  def checkLastIter(self, kernel, comment="at last iteration?"):
    """ Return last iteration of unroll loop. """
    if self.unrollIncIsDepthU:
      return inst("s_cmp_gt_u32", "DepthU", \
          sgpr("UnrollLoopLastIter"), comment)
    else:
      return inst("s_cmp_eq_u32", self.loopCounter(kernel, self.unrollIdx), \
          0, comment)

  def isConstUnitStride(self, stride):
      return stride.startswith("const")

  ########################################
  ########################################
  def strideRef(self, tc, dim):
    """
    Return sgpr with specified stride or define starting with const if constant.
    dim is index 0...max indices and is in global index space.
    """
    problemType = self.kernel["ProblemType"]
    if tc in ['A','B']:
      if not problemType["UseInitialStridesAB"] and \
          dim == problemType["IndexAssignments%s"%tc][0]:
        return ("constStride%s%s"%(tc,self.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.indexChars[dim]))
    elif tc in ['D','C']:
      if not problemType["UseInitialStridesCD"] and dim == 0:
        return ("constStride%s%s"%(tc,self.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.indexChars[dim]))
    else:
      raise ValueError("unexpected tensorChar='%s' in stride function"%tc)


  ########################################
  # Get Label
  # return label number - create new if it doesn't already exist
  ########################################
  def getLabelNum(self, name):
    if name not in self.labels:
      self.labels[name] = len(self.labels)
    return self.labels[name]

  ########################################
  # return label name including a unique number
  # create new if it doesn't already exist
  ########################################
  def getNamedLabel(self, name):
    if name not in self.labels:
      self.labels[name] = "%s_%u" % (name, len(self.labels))
    return self.labels[name]

  ########################################
  # return string that defines a unique named name_number
  ########################################
  def getNamedLabelDef(self, name, labelComment=""):
    t = "%s: // %s\n" % (self.getNamedLabel(name), labelComment)
    return t

  ########################################
  # return string that defines a unique numeric label
  # labelComment is a comment string if this is a label definition
  ##############################################################################
  def getLabelDef(self,name,labelComment=""):
    t = "label_%04u: // %s %s\n" % (self.getLabelNum(name), name, labelComment)
    return t

  ##############################################################################
  # define a label and return undecorated label_%4u - suitable for using as jump target
  ##############################################################################
  def getLabelTarget(self,name,labelDef=None):
    t = "label_%04u" % (self.getLabelNum(name))
    return t

  ##############################################################################
  ##############################################################################
  def getUniqLabel(self):
    name = "uniq_label_" + str(len(self.labels))
    return self.getLabelNum(name)

  ##############################################################################
  # Find Memory Instruction For Width and Stride
  ##############################################################################
  def findMemoryInstructionForWidthStride(self, width, strides, combine, \
      instructions):
    for i in range(0, len(instructions)):
      instruction = instructions[i]
      numAddresses = instruction.numAddresses
      numOffsets = instruction.numOffsets
      offsetMultiplier = instruction.offsetMultiplier
      blockWidth = instruction.blockWidth
      valid = True
      if width < blockWidth:
        valid = False
      if combine: # try to combine ops
        if numOffsets > 0: # if inst combines using offsets
          for stride in strides:
            if stride % offsetMultiplier != 0:
              valid = False
      else: # don't try to combine ops
        if numOffsets > 1 or numAddresses > 1:
          valid = False
      if valid:
        return i
      else:
        continue

    printWarning("Could not find valid memory instruction for width=%f" % width)
    return len(instructions)


  ##############################################################################
  # Select Memory Instruction
  # when selecting instruction, need to support stride in both dims
  ##############################################################################
  def selectMemoryInstruction(self,
      operation, # ReadGlobal, WriteLocal, ReadLocal
      width, # num registers 1 chunk
      write2, # Para, Perp, None
      para2, # NumLoadsPara >= 2
      perp2, # NumLoadsPerp >= 2
      strides ):

    #instructions = self.memoryArchitecture[operation]
    instructions = self.memoryInstructions[self.version][operation]
    # try to combine
    if (write2 == "Coalesced" and para2) \
        or (write2 == "Perpendicular" and perp2):
      instructionIdx = self.findMemoryInstructionForWidthStride( \
          width, strides, True, instructions)
      if instructionIdx < len(instructions): # found combined
        return instructionIdx

    # don't or can't combine
    return self.findMemoryInstructionForWidthStride( \
        width, strides, False, instructions)

  class TmpSgpr:
    """ A temporary register which is automatically returned to sgpr pool when class is destroyed. """
    def __init__(self, regPool, num, align, tag=None):
      self.regPool = regPool
      self.regIdx = regPool.checkOutAligned(num, align, tag=tag, preventOverflow=False)

    def idx(self):
      return self.regIdx

    def __int__(self):
      return self.idx()

    def __del__(self):
      self.regPool.checkIn(self.regIdx)

  def getTmpSgpr(self, num, align=None, tag=None):
    if align==None:
      align = 1 if num==1 else 2
    if tag==None:
      tag = "getTmpSgpr(%d)"%num

    t = self.TmpSgpr(self.sgprPool, num, align, tag)
    if t.idx()+num > self.maxSgprs:
      self.overflowedResources = 2
      if self.db["AssertOnSgprOverflow"]:
        assert(t.idx()+num <= self.maxSgprs)
    return t

  def dumpSgpr(self, sgprStore):
    kStr = ""
    if globalParameters["DebugKernel"]:
      afterDump = -1
      if self.db["DebugKernelMaxItems"] != -1:
        afterDump = self.getUniqLabel()
        kStr += inst("s_cmp_lt_u32", sgpr("DebugKernelItems"), 16,  "")
        kStr += inst("s_cbranch_scc0", "label_%04u"%afterDump, \
                     "skip if already wrote enough work-items" )
        kStr += inst("s_add_u32", sgpr("DebugKernelItems"), \
                     sgpr("DebugKernelItems"), \
                     hex(1), "inc items written" )

      tmp = self.vgprPool.checkOut(1,"tmp")
      kStr += inst("v_mov_b32", vgpr(tmp), sgprStore, "Debug")
      kStr += inst("flat_store_dword", vgpr("AddressDbg", 2), \
          vgpr(tmp), "debug dump sgpr store" )
      kStr += inst("_v_add_co_u32", vgpr("AddressDbg"), "vcc", vgpr("AddressDbg"), \
          hex(4), "debug dump inc" )
      self.vgprPool.checkIn(tmp)

      if self.db["DebugKernelMaxItems"] != -1:
        kStr += "label_%04u:%s  %s" % (afterDump, "// skip debug target", self.endLine)

    return kStr


  def defineSgpr(self, name, numSgprs, align=1):
    if numSgprs == 0: return

    sgprIdx = self.sgprPool.checkOutAligned(numSgprs, align, tag=name, preventOverflow=0)
    #self.sgprIdx = roundUpToNearestMultiple(self.sgprIdx,align)
    #print (name, "->", self.sgprIdx, "+", numSgprs)
    self.sgprs[name] = sgprIdx

    return sgprIdx

  def undefineSgpr(self, name):
    self.sgprPool.checkIn(self.sgprs[name])
    # later references will result in compile-time error (with odd 'error: expected relocatable expression')
    # and 'Kernel ... not found in any loaded module'
    return ".set %s, UNDEF\n" % name

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def initKernel(self, kernel, tPA, tPB ):
    super(KernelWriterAssembly, self).initKernel(kernel, tPA, tPB)
    problemType = kernel["ProblemType"]

    dkp = kernel["DisableKernelPieces"]
    self.do["NullKernel"]  = dkp >= 9 or dkp == -9

    self.kernel = kernel

    # init these here in case some kernel pieces are disabled for performance exploration:
    tPA["localReadOffset"] = 0
    tPB["localReadOffset"] = 0

    self.sgprs=collections.OrderedDict()

    self.LdsOOB = 0xF00000

    #---
    # Internal optimization and debug controls.
    # These have a default which is almost always faster so don't make a full-blown YAML parm
    # But have a control here so we can disable for debugging and also easily tell
    # which parts of the code were changed to support the new mode.
    self.globalReadIncsUseVgpr = False if kernel["BufferLoad"] else True

    # If True, GRO are expressed as offsets from the beginning of the macro-tile, and the SRD
    # is set to the beginning of the macro-tile.
    # If False, GRO are expressed as offsets from the beginning of the lowest 2 dimensions
    # in the tensor.
    # True can allow Buffer-Based logic to have significantly higher range and handle larger tensors
    # groOffsetInMacroTile doesn't work with pointer-shift because it sets the SRD to point to the
    # start of the macro-tile - if we overhang by small number of elements (<GRVW) then can't shift
    # back to get all the data.
    # groOffsetInMacroTile doesn't work with packed dims since these need to set SRD to the tensor base
    # then extract the packed dimensions from the flattened index (including the workgroup) and scale by strides
    # - the index is per-work-item so can't put work-group into the SRD
    # ZeroPad requires groOffsetInMacroTile since it needs the gro offsets in each dimension to include
    # the tile components, since those same vars are used to compute the ZP offsets used for edge comparisons.
    if problemType["ZeroPadA"] == [] and problemType["ZeroPadB"] == [] and \
       len(kernel["PackedC0IndicesX"])==1 and len(kernel["PackedC1IndicesX"])==1 and kernel["BufferLoad"]:
      self.groOffsetInMacroTile = 1
    else:
      self.groOffsetInMacroTile = 0


    self.use64bProductOfSums = 0
    self.use64bPackSumOffset = 0  # use 2 SGPR for extracting packed summation dims.  Not supported, but this marks eventual required changes

    # use 64-bit buffer limit shadow register
    # PackSummationDims does not support shadow limit - the address calc code would need to restore the shadow limit, which is possible
    # but not implemented or tested
    self.use64bShadowLimit = kernel["Use64bShadowLimit"] and kernel["BufferLoad"] and not kernel["PackSummationDims"]


    # Check if the address setup code for LWA and GRO causes register growth.
    # This is not an error condition but bears further investigation.
    # In particular if PrefetchAcrossPersistent=1 then the NewTile setup code
    # will be run before the no-load-loop iteration where registers are still
    # tight.  Realistically we just have the GlobalToLocal VGPRs, all else is
    # growth.
    self.preventVgprOverflowDuringNewTile = 0 and not globalParameters["ForceGenerateKernel"]

    # For Beta:
    # Rather than waiting for all loads to finish with s_waitcnt vmcnt(0), interleave
    # appropriate vmwnts into the stores so they issue as loads become available
    self.interleaveStoreVmcnt = 1 and kernel["BufferStore"]


    # if >0, shift the start of the SRD left by specified #elements (not bytes)
    # Gives pointer shift some room to move left, even into the previous macro-tile
    # This slightly reduces the range of the GRO since they have to include the offset
    # Pointer shift still cannot be used with very small matrices < GRVW
    self.srdShiftLeft = {}
    self.srdShiftLeft["A"] = kernel["GlobalLoadVectorWidthA"]
    self.srdShiftLeft["B"] = kernel["GlobalLoadVectorWidthB"]

    self.checkGRO = False
    # checkGRO requires useSgprForGRO=0 so that code allocates and uses
    # the VGPRs that are used for the GRO offset checking
    assert not (kernel["_UseSgprForGRO"] and self.checkGRO)

    # Debug mode to explore combining VGPRs.
    # Saves VGPRs but doesn't generate correct answer
    self.combineLocalAddresses = 0

    # ISA version, such as 803
    self.version = globalParameters["CurrentISA"]
    if "ISA" in kernel:
      self.version = tuple(kernel["ISA"])
    if not globalParameters["AsmCaps"][self.version]["SupportedISA"]:
      defaultIsa = (9,0,0)
      print("warning: ISA:", self.version, " is not supported; overriding with ", defaultIsa)
      self.version = defaultIsa
    
    if kernel["EnableMatrixInstruction"] and not self.version == (9,0,8):
      printExit("MatrixInstruction not supported for {0}".format(self.version))

    self.AsmBugs = {}
    self.AsmBugs["ExplicitCO"] = globalParameters["AsmCaps"][self.version]["HasExplicitCO"]
    self.AsmBugs["ExplicitNC"] = globalParameters["AsmCaps"][self.version]["HasExplicitNC"]

    if not globalParameters["AsmCaps"][self.version]["HasDirectToLds"]:
      kernel["DirectToLdsA"] = False
      kernel["DirectToLdsB"] = False
      kernel["LocalWriteUseSgprA"] = False # Requires DirectToLdsA
      kernel["LocalWriteUseSgprB"] = False # Requires DirectToLdsB

    #######################################L
    # Available Memory Instructions
    ########################################

    # name, numAddresses, numOffsets, offsetMultiplier, blockWidth, formatting):
    ########################################
    # Local Read
    ds_read_b128 = MemoryInstruction("ds_read_b128",  1, 1, 4, 4, \
        "%s, %s offset:%s" )
    ds_read2_b64 = MemoryInstruction("ds_read2_b64",  1, 2, 2, 2, \
        "%s, %s offset0:%s, offset1:%s" )
    ds_read_b64 = MemoryInstruction("ds_read_b64",    1, 1, 2, 2, \
        "%s, %s offset:%s" )
    ds_read2_b32 = MemoryInstruction("ds_read2_b32",  1, 2, 1, 1, \
        "%s, %s offset0:%s offset1:%s" )
    ds_read_b32 = MemoryInstruction("ds_read_b32",    1, 1, 1, 1, \
        "%s, %s offset:%s" )
    ds_read_u16 = MemoryInstruction("ds_read_u16",    1, 1, 1, 0.5, \
        "%s, %s offset:%s" )
    ########################################
    # Local Write
    ds_write_b128 = MemoryInstruction("ds_write_b128",  1, 1, 4, 4, \
        "%s, %s offset:%s" )
    ds_write2_b64 = MemoryInstruction("ds_write2_b64",  1, 2, 2, 2, \
        "%s, %s, %s offset0:%s, offset1:%s" )
    ds_write_b64 = MemoryInstruction("ds_write_b64",    1, 1, 2, 2, \
        "%s, %s offset:%s" )
    ds_write2_b32 = MemoryInstruction("ds_write2_b32",  1, 2, 1, 1, \
        "%s, %s, %s offset0:%s offset1:%s" )
    ds_write_b32 = MemoryInstruction("ds_write_b32",    1, 1, 1, 1, \
        "%s, %s offset:%s" )
    ds_write_b16 = MemoryInstruction("ds_write_b16",    1, 1, 1, 0.5, \
        "%s, %s offset:%s" )
    ########################################
    # Global Read
    flat_load_dwordx4 = MemoryInstruction("flat_load_dwordx4",  1, 0, 0, 4, \
        "UNUSED %s, %s" )
    flat_load_dwordx2 = MemoryInstruction("flat_load_dwordx2",  1, 0, 0, 2, \
        "UNUSED %s, %s" )
    flat_load_dword = MemoryInstruction("flat_load_dword",      1, 0, 0, 1, \
        "UNUSED %s, %s" )

    buffer_load_dwordx4 = MemoryInstruction("buffer_load_dwordx4", 1, 0, 0, 4, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s" )
    buffer_load_dwordx2 = MemoryInstruction("buffer_load_dwordx2", 1, 0, 0, 2, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s" )
    buffer_load_dword = MemoryInstruction("buffer_load_dword", 1, 0, 0, 1, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s" )
    # generate half directly w/o using the format string to handle hi/lo correctly
    buffer_load_short = MemoryInstruction("buffer_load_short_d16", 1, 0, 0, 0.5, \
        "UNUSED %s, %s, %s, %s offen offset:0 %s" )

    ########################################
    # Global Write
    flat_store_dwordx4 = MemoryInstruction("flat_store_dwordx4",  1, 0, 0, 4, \
        "%s, %s" )
    flat_store_dwordx2 = MemoryInstruction("flat_store_dwordx2",  1, 0, 0, 2, \
        "%s, %s" )
    flat_store_dword = MemoryInstruction("flat_store_dword",      1, 0, 0, 1, \
        "%s, %s" )

    ########################################
    # Available Memory Instructions per Architecture
    # gfx701 "Hawaii"
    # gfx801 "Carrizo"
    # gfx802 "Tonga"
    # gfx803 "Fiji"
    # gfx900
    ########################################
    if (kernel["BufferLoad"]):
      chosen_load_dwordx4 = buffer_load_dwordx4
      chosen_load_dwordx2 = buffer_load_dwordx2
      chosen_load_dword   = buffer_load_dword
      chosen_load_short    = buffer_load_short
    else:
      chosen_load_dwordx4 = flat_load_dwordx4
      chosen_load_dwordx2 = flat_load_dwordx2
      chosen_load_dword   = flat_load_dword
      chosen_load_short    = flat_load_dword # not supported

    chosen_store_dwordx4 = flat_store_dwordx4
    chosen_store_dwordx2 = flat_store_dwordx2
    chosen_store_dword   = flat_store_dword

    self.memoryInstructions = {
        (9,0,0): {
          "GlobalRead": [ chosen_load_dwordx4, chosen_load_dwordx2,
            chosen_load_dword, chosen_load_short ],
          "GlobalWrite": [ chosen_store_dwordx4, chosen_store_dwordx2,
            chosen_store_dword ],
          "LocalRead": [ ds_read_b128, ds_read2_b64,
            ds_read_b64, ds_read2_b32, ds_read_b32, ds_read_u16 ],
          "LocalWrite": [ ds_write_b128, ds_write2_b64,
            ds_write_b64, ds_write2_b32, ds_write_b32, ds_write_b16 ]
          }, # 900
        }
    self.memoryInstructions[(8,0,3)] = self.memoryInstructions[(9,0,0)]
    self.memoryInstructions[(9,0,6)] = self.memoryInstructions[(9,0,0)]
    self.memoryInstructions[(9,0,8)] = self.memoryInstructions[(9,0,0)]
    self.memoryInstructions[(10,1,0)] = self.memoryInstructions[(9,0,0)]

    if self.version == (9,0,0):
      self.mixinst = "v_mad_mix_f32"
    elif self.version == (9,0,6) or self.version == (9,0,8):
      self.mixinst = "v_fma_mix_f32"
    else:
      self.mixinst = "NOT_SUPPORTED"

    self.overflowedResources = 0 # if true, comment out whole kernel

    self.kernelName = self.getKernelName(kernel)
    self.inTailLoop = False

    # registers per element
    self.bpr = 4 # all registers are 32bit
    self.bpeAB = int(self.bpr*\
        kernel["ProblemType"]["DataType"].numRegisters())
    self.bpeCexternal = int(self.bpr*\
        kernel["ProblemType"]["DataType"].numRegisters())
    #jgolds Need to check device for support
    if kernel["ProblemType"]["HighPrecisionAccumulate"]:
        if kernel["ProblemType"]["DataType"].isHalf():
            self.bpeCinternal = int(self.bpr*1)
        elif kernel["ProblemType"]["DataType"].isBFloat16():
            self.bpeCinternal = int(self.bpr*1)
            # print("need_replacement_kernel %s" % self.kernelName)
        elif kernel["ProblemType"]["DataType"].isInt8x4():
            # numRegisters for Int8x4 = numRegisters for float = 1
            self.bpeCinternal = int(self.bpr* kernel["ProblemType"]["DataType"].numRegisters())
        else:
            print("HighPrecisionAccumulate only valid when DataType is half, Int8x4.")
            self.bpeCinternal = int(self.bpr*\
                kernel["ProblemType"]["DataType"].numRegisters())
            kernel["ProblemType"]["HighPrecisionAccumulate"] = False
    else:
        self.bpeCinternal = int(self.bpr*\
            kernel["ProblemType"]["DataType"].numRegisters())
    assert self.bpeAB == tPA["bpe"]
    assert self.bpeAB == tPB["bpe"]
    # registers per global address
    self.rpga = 2 # 64-bit
    # registers per local address
    self.rpla = 1 # 32-bit
    # registers per global 32-bit offset (some intructions only support 32-bit offset)
    self.rpgo = 1 # 32-bit

    ####################################
    # choose memory instructions
    ####################################

    ########################################
    # globalReadA instruction; no flat_load2_*
    self.globalReadWidthA = float(tPA["nrcv"]*tPA["bpe"])/self.bpr
    self.globalRead2CoalescedA = kernel["NumLoadsCoalescedA"]>1 \
        or self.readCoalescedComponentsA
    self.globalRead2PerpendicularA = kernel["NumLoadsPerpendicularA"] > 1 \
        or self.readPerpendicularComponentsA
    self.globalReadInstructionIdxA = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthA, \
        kernel["GlobalRead2A"], \
        self.globalRead2CoalescedA, self.globalRead2PerpendicularA, [] )
    ########################################
    # globalReadB instruction; no flat_load2_
    self.globalReadWidthB = float(tPB["nrcv"]*tPB["bpe"])/self.bpr
    self.globalRead2CoalescedB = kernel["NumLoadsCoalescedB"]>1 \
        or self.readCoalescedComponentsB
    self.globalRead2PerpendicularB = kernel["NumLoadsPerpendicularB"] > 1 \
        or self.readPerpendicularComponentsB
    self.globalReadInstructionIdxB = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthB, \
        kernel["GlobalRead2B"], \
        self.globalRead2CoalescedB, self.globalRead2PerpendicularB, [] )

    ########################################
    # localWriteA instruction
    # for local, tile->para, unroll->perp
    #self.localWriteWidthA = 1 if (self.writeTileDimComponentsA \
    #    or self.writeUnrollDimComponentsA) else kernel["VectorWidth"]
    self.localWriteWidthA = tPA["nwcv"]*tPA["bpe"]//self.bpr
    if self.localWriteWidthA < 1:
      self.localWriteWidthA = (1.0*tPA["nwcv"]*tPA["bpe"])/self.bpr
    self.localWrite2CoalescedA = tPA["nrc"]>1 \
        or self.writeTileDimComponentsA
    self.localWrite2PerpendicularA = tPA["nrp"]>1 \
        or self.writeUnrollDimComponentsA
    # localWriteA stride tile
    if kernel["ProblemType"]["TLUA"]:
      if self.writeTileDimComponentsA:
        self.localWriteStrideTileA = 1
        self.localWriteJoinTileA = "Components"
      else:
        self.localWriteStrideTileA = kernel["LSCA"]
        self.localWriteJoinTileA = "Coalesced"
    else:
      if self.writeUnrollDimComponentsA:
        self.localWriteStrideTileA = 1
        self.localWriteJoinTileA = "Components"
      else:
        self.localWriteStrideTileA = kernel["LSPA"]
        self.localWriteJoinTileA = "Perpendicular"
    self.localWriteStrideTileA = self.localWriteStrideTileA*tPA["bpe"]//self.bpr
    # localWriteA stride unroll
    if kernel["ProblemType"]["TLUA"]:
      if self.writeUnrollDimComponentsA:
        self.localWriteStrideUnrollA = 1*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Components"
      else:
        self.localWriteStrideUnrollA = kernel["LSCA"]*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Perpendicular"
    else:
      if self.writeTileDimComponentsA:
        self.localWriteStrideUnrollA = 1*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Components"
      else:
        self.localWriteStrideUnrollA = kernel["LSCA"]*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Coalesced"
    self.localWriteStrideUnrollA = \
        (self.localWriteStrideUnrollA*tPA["bpe"])//self.bpr
    self.localWriteInstructionIdxA = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthA, \
        kernel["LocalWrite2A"], \
        self.localWrite2CoalescedA, self.localWrite2PerpendicularA,
        [self.localWriteStrideTileA, self.localWriteStrideUnrollA] )

    ########################################
    # localWriteB instruction
    # for local, tile->para, unroll->perp
    #self.localWriteWidthB = 1 if (self.writeTileDimComponentsB \
    #    or self.writeUnrollDimComponentsB) else kernel["VectorWidth"]
    self.localWriteWidthB = tPB["nwcv"]*tPB["bpe"]//self.bpr
    if self.localWriteWidthB < 1:
      self.localWriteWidthB = (1.0*tPB["nwcv"]*tPB["bpe"])/self.bpr
    self.localWrite2CoalescedB = tPB["nrc"]>1 \
        or self.writeTileDimComponentsB
    self.localWrite2PerpendicularB = tPB["nrp"]>1 \
        or self.writeUnrollDimComponentsB
    # localWriteB stride tile
    if kernel["ProblemType"]["TLUB"]:
      if self.writeTileDimComponentsB:
        self.localWriteStrideTileB = 1
        self.localWriteJoinTileB = "Components"
      else:
        self.localWriteStrideTileB = kernel["LSCB"]
        self.localWriteJoinTileB = "Coalesced"
    else:
      if self.writeUnrollDimComponentsB:
        self.localWriteStrideTileB = 1
        self.localWriteJoinTileB = "Components"
      else:
        self.localWriteStrideTileB = kernel["LSPB"]
        self.localWriteJoinTileB = "Perpendicular"
    self.localWriteStrideTileB = (self.localWriteStrideTileB*tPB["bpe"])//self.bpr
    # localWriteB stride unroll
    if kernel["ProblemType"]["TLUB"]:
      if self.writeUnrollDimComponentsB:
        self.localWriteStrideUnrollB = 1*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Components"
      else:
        self.localWriteStrideUnrollB = kernel["LSCB"]*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Perpendicular"
    else:
      if self.writeTileDimComponentsB:
        self.localWriteStrideUnrollB = 1*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Components"
      else:
        self.localWriteStrideUnrollB = kernel["LSCB"]*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Coalesced"
    self.localWriteStrideUnrollB = \
        (self.localWriteStrideUnrollB*tPB["bpe"])//self.bpr
    self.localWriteInstructionIdxB = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthB, \
        kernel["LocalWrite2B"], \
        self.localWrite2CoalescedB, self.localWrite2PerpendicularB,
        [self.localWriteStrideTileB, self.localWriteStrideUnrollB] )

    ########################################
    # localRead A
    localReadWidth = (kernel["VectorWidth"] * tPA["bpe"]) // self.bpr
    if kernel["EnableMatrixInstruction"]:
      localReadWidth = tPA["bpe"] / self.bpr
    if kernel["UnrollMajorLDSA"]:
      localReadWidth *= kernel["ProblemType"]["DataType"].numMIInput()

    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedA = \
        kernel["ThreadTile0"] * tPA["bpe"]//self.bpr
    self.localRead2CoalescedA = kernel["ThreadTile0"]//kernel["VectorWidth"] > 1
    self.localReadInstructionIdxA = \
        self.selectMemoryInstruction("LocalRead", localReadWidth, \
        kernel["LocalRead2A"], \
        self.localRead2CoalescedA, localRead2Perpendicular,
        [self.localReadStrideCoalescedA] )
    tPA["localReadSwapByteOffset"] = 0
    tPB["localReadSwapByteOffset"] = 0
    tPA["localWriteSwapByteOffset"] = 0
    tPB["localWriteSwapByteOffset"] = 0


    ########################################
    # localRead B
    localReadWidth = (kernel["VectorWidth"] * tPB["bpe"]) // self.bpr
    if kernel["EnableMatrixInstruction"]:
      localReadWidth = tPB["bpe"] / self.bpr
    if kernel["UnrollMajorLDSB"]:
      localReadWidth *= kernel["ProblemType"]["DataType"].numMIInput()

    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedB = \
    kernel["ThreadTile1"] * tPB["bpe"]//self.bpr
    self.localRead2CoalescedB = kernel["ThreadTile1"]//kernel["VectorWidth"] > 1
    self.localReadInstructionIdxB = \
        self.selectMemoryInstruction("LocalRead", localReadWidth, \
        kernel["LocalRead2B"], \
        self.localRead2CoalescedB, localRead2Perpendicular,
        [self.localReadStrideCoalescedB] )

    instructions = self.memoryInstructions[self.version]
    self.globalReadInstructionA = instructions["GlobalRead"][ \
        self.globalReadInstructionIdxA]
    self.globalReadInstructionB = instructions["GlobalRead"][ \
        self.globalReadInstructionIdxB]
    self.localWriteInstructionA = instructions["LocalWrite"][ \
        self.localWriteInstructionIdxA]
    self.localWriteInstructionB = instructions["LocalWrite"][ \
        self.localWriteInstructionIdxB]
    self.localReadInstructionA = instructions["LocalRead"][ \
        self.localReadInstructionIdxA]
    self.localReadInstructionB = instructions["LocalRead"][ \
        self.localReadInstructionIdxB]
    # global reads per instruction
    tPA["nrcvpi"] = int((self.globalReadInstructionA.totalWidth*self.bpr)/tPA["bpe"])
    tPB["nrcvpi"] = int((self.globalReadInstructionB.totalWidth*self.bpr)/tPB["bpe"])
    tPA["nwcvpi"] = int((self.localWriteInstructionA.totalWidth*self.bpr)/tPA["bpe"])
    tPB["nwcvpi"] = int((self.localWriteInstructionB.totalWidth*self.bpr)/tPB["bpe"])
    ####################################
    # VGPR Allocation
    ####################################

    ####################################
    # num vgprs: valu
    #jgolds bpeCinternal because we are allocating accumulation registers here
    self.numVgprValuC = (kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.bpeCinternal)//self.bpr

    valuBlocks = (1+kernel["PrefetchLocalRead"]) * kernel["InnerUnroll"]
    if kernel["EnableMatrixInstruction"]:
      self.numVgprValuAPerBlock = kernel["MIWaveTile"][0] * kernel["ProblemType"]["DataType"].numMIInput() * tPA["bpe"] // self.bpr
      self.numVgprValuBPerBlock = kernel["MIWaveTile"][1] * kernel["ProblemType"]["DataType"].numMIInput() * tPA["bpe"] // self.bpr
    else:
      self.numVgprValuAPerBlock = kernel["ThreadTileA"]*tPA["bpe"]//self.bpr
      self.numVgprValuBPerBlock = kernel["ThreadTileB"]*tPB["bpe"]//self.bpr
      if kernel["ProblemType"]["DataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        self.numVgprValuAPerBlock = self.numVgprValuAPerBlock * 2
        self.numVgprValuBPerBlock = self.numVgprValuBPerBlock * 2

    numVgprValuA = self.numVgprValuAPerBlock * valuBlocks
    numVgprValuB = self.numVgprValuBPerBlock * valuBlocks

    ####################################
    # num vgprs: global -> local elements
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      self.numVgprG2LA = roundUp((kernel["NumLoadsCoalescedA"] * kernel["NumLoadsPerpendicularA"] *\
        kernel["GlobalLoadVectorWidthA"] * tPA["bpe"])/(float)(self.bpr))
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      self.numVgprG2LB = roundUp((kernel["NumLoadsCoalescedB"]*kernel["NumLoadsPerpendicularB"]* \
        kernel["GlobalLoadVectorWidthB"] * tPB["bpe"])/(float)(self.bpr))

    ####################################
    # num vgprs: local read addresses
    numVgprLocalReadAddressesA = 1 * self.rpla
    numVgprLocalReadAddressesB = 1 * self.rpla

    ####################################
    # num vgprs: local write addresses
    #numLocalWritesA = kernel["NumLoadsCoalescedA"] \
    #    * nlp * self.numWriteVectorComponentsA
    #numLocalWriteInstructionsA = numLocalWritesA \
    #    / self.localWriteInstructionA[self.instructionIdxNumOffsets]
    self.numVgprLocalWriteAddressesA = 0 if kernel["LocalWriteUseSgprA"] else 1 * self.rpla
    # TODO - if we only have one local write - can just map the overhang register to the LWO
    if kernel["FractionalLoad"]==1 and kernel["fractionalPerpOverhangA"]:
      self.numVgprLocalWriteAddressesA += 1*self.rpla

    #numLocalWritesB = kernel["NumLoadsCoalescedB"] \
    #    * nlp * self.numWriteVectorComponentsB
    #numLocalWriteInstructionsB = numLocalWritesB \
    #    / self.localWriteInstructionB[self.instructionIdxNumOffsets]
    self.numVgprLocalWriteAddressesB = 0 if kernel["LocalWriteUseSgprB"] else 1 * self.rpla
    if kernel["FractionalLoad"]==1 and kernel["fractionalPerpOverhangB"]:
      self.numVgprLocalWriteAddressesB += 1*self.rpla

    ####################################
    # num vgprs: global read addresses
    numGlobalReadsA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["GlobalLoadVectorWidthA"] \
        * self.numReadVectorComponentsA
    numGlobalReadInstructionsA = (numGlobalReadsA * tPA["bpe"])//\
        (self.globalReadInstructionA.blockWidth * 4)

    if kernel["BufferLoad"]:
      numGlobalReadOffsetsA = roundUp(numGlobalReadInstructionsA * self.rpgo)
    else:
      numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.rpga

    numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["GlobalLoadVectorWidthB"] \
        * self.numReadVectorComponentsB
    numGlobalReadInstructionsB = (numGlobalReadsB * tPB["bpe"])// \
        (self.globalReadInstructionB.blockWidth * 4)
    if kernel["BufferLoad"]:
      numGlobalReadOffsetsB = roundUp(numGlobalReadInstructionsB * self.rpgo)
    else:
      numVgprGlobalReadAddressesB = numGlobalReadInstructionsB * self.rpga
    if self.globalReadIncsUseVgpr:
      numVgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga
      numVgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga
    else:
      numVgprGlobalReadIncsA = 0
      numVgprGlobalReadIncsB = 0

    numVgprAddressDbg = self.rpga if globalParameters["DebugKernel"] else 0

    ####################################
    # num vgprs: c write address
    # 1 address where to write first value
    # 1 tmp address where to write current value


    ####################################
    # VGPR Assignment
    ####################################
    vgprIdx = 0

    self.startVgprValuC = vgprIdx; vgprIdx += self.numVgprValuC

    self.startVgprValuA = vgprIdx; vgprIdx += numVgprValuA

    valuBlocks = (1+kernel["PrefetchLocalRead"]) * kernel["InnerUnroll"]
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      if kernel["PrefetchGlobalRead"]:
        self.startVgprG2LA = vgprIdx; vgprIdx += self.numVgprG2LA
      else: # g2l can overlap valu
        self.startVgprG2LA = self.startVgprValuA
        vgprIdx = self.startVgprValuA \
            + max(self.numVgprValuAPerBlock*valuBlocks, self.numVgprG2LA)

    self.startVgprValuB = vgprIdx; vgprIdx += numVgprValuB
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      if kernel["PrefetchGlobalRead"]:
        self.startVgprG2LB = vgprIdx; vgprIdx += self.numVgprG2LB
      else: # g2l can overlap valu
        self.startVgprG2LB = self.startVgprValuB
        vgprIdx = self.startVgprValuB \
            + max(self.numVgprValuBPerBlock*valuBlocks, self.numVgprG2LB)

    # Registers allocated above this point can be used as temps during setup
    # Registers above here are reserved in initC, near the end of the setup
    # code
    self.lastValuAB = vgprIdx
    #----------------------------------

    if not kernel["LocalWriteUseSgprA"]:
      if self.combineLocalAddresses:
        self.startVgprLocalWriteAddressesA = self.startVgprLocalReadAddressesA
      else:
        self.startVgprLocalWriteAddressesA = vgprIdx
        vgprIdx += self.numVgprLocalWriteAddressesA

    if not kernel["LocalWriteUseSgprB"]:
      if self.combineLocalAddresses:
        self.startVgprLocalWriteAddressesB = self.startVgprLocalReadAddressesA
      else:
        self.startVgprLocalWriteAddressesB = vgprIdx
        vgprIdx += self.numVgprLocalWriteAddressesB

    # BufferLoad:
    # Uses a resource descriptor (SRD) which is stored in 4 SGPRs and thus shared by all work-items.
    # Each work-item also uses  a unique 32-bit offset into vgprGlobalReadOffset.  These offsets are set when 
    # the tile is initialized and stay constant through the execution of the kernel.
    # The base address in the SRD is updated when the algoritm moves to a new tile
    # BufferLoad disables the gptGlobalReadAddr used in flat addressing.
    if kernel["BufferLoad"]:
       self.startVgprGlobalReadOffsetA = vgprIdx
       vgprIdx += 1 if kernel["_UseSgprForGRO"] else numGlobalReadOffsetsA
       self.startVgprGlobalReadOffsetB = vgprIdx
       vgprIdx += 1 if kernel["_UseSgprForGRO"] else numGlobalReadOffsetsB

    else:
      self.startVgprGlobalReadAddressesA = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesA
      self.startVgprGlobalReadAddressesB = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesB

    self.zeroPadRegs={}
    self.zeroPadRegs['A'] = collections.OrderedDict()
    self.zeroPadRegs['B'] = collections.OrderedDict()
    for (tc,tP) in (('A',self.tPA),('B',self.tPB)):
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
              for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
                (freeDim, sumDim) = zp[:2]
                freeDimChar = globalParameters["IndexChars"][freeDim]
                sumDimChar  = globalParameters["IndexChars"][sumDim]
                zpName = "GlobalReadOffset%s_ZP%s%s_%d_%d_%d_%d" % \
                          (tc, freeDimChar, sumDimChar, para, sPara, perp, sPerp)

                assert (zpName not in self.zeroPadRegs[tc])
                self.zeroPadRegs[tc][zpName] = ZeroPadReg(zp, zpName, vgprIdx, \
                                                        perp, sPerp, para, sPara)
                vgprIdx += 1

    self.startVgprGlobalReadIncsA = vgprIdx
    vgprIdx += numVgprGlobalReadIncsA
    self.startVgprGlobalReadIncsB = vgprIdx
    vgprIdx += numVgprGlobalReadIncsB

    # Point at last VGPR that can be reclaimed for use in the summation loop
    # If more VGPRs are added here be aware of the register reclaim code in
    # endSummation - registers that should be preserved after lastVgprForReads
    self.lastVgprForReads = vgprIdx
    #-----------

    self.startVgprLocalReadAddressesA = vgprIdx
    vgprIdx += numVgprLocalReadAddressesA
    if self.combineLocalAddresses:
      self.startVgprLocalReadAddressesB = self.startVgprLocalReadAddressesA
    else:
      self.startVgprLocalReadAddressesB = vgprIdx
      vgprIdx += numVgprLocalReadAddressesB

    self.startVgprAddressDbg = vgprIdx
    vgprIdx += numVgprAddressDbg

    # tmp vgprs
    #minVgprTmp = 1
    #if kernel["LoopTail"]:
    #  minVgprTmp += 4
    #if globalParameters["DebugKernel"]:
    #  minVgprTmp += 2
    #vgprIdx += minVgprTmp
    #print2("%3u vgprs <- %s" % (vgprIdx, self.kernelName) )

    self.totalVgprs = vgprIdx
    if self.totalVgprs < kernel["MinVgprNumber"] or self.totalVgprs > kernel["MaxVgprNumber"]:
      raise RuntimeError("Generating asm kernel error: total vgpr: %u not in [%u, %u].\n" % (self.totalVgprs, kernel["MinVgprNumber"], kernel["MaxVgprNumber"]))

    ########################################
    # SGPR Allocation
    ########################################

    ####################################
    # num sgprs: initial kernel state
    self.sgprPool = RegisterPool(0, 's', 0, defaultPreventOverflow=True, printRP=0)
    numSgprAddressD = self.rpga # til end
    numSgprAddressC = self.rpga # til end
    numSgprAddressA = self.rpga # til read offsets
    numSgprAddressB = self.rpga # til read offsets
    numSgprAlpha = max(1,int(tPA["bpe"]/4))
    numSgprBeta  = max(1,int(self.bpeCexternal/4)) if kernel["ProblemType"]["UseBeta"] else 0
    self.numSgprStridesD = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprStridesC = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprStridesA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    self.numSgprStridesB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    if not kernel["ProblemType"]["UseInitialStridesCD"]:
      self.numSgprStridesD -= 1
      self.numSgprStridesC -= 1
    if not kernel["ProblemType"]["UseInitialStridesAB"]:
      self.numSgprStridesA -= 1
      self.numSgprStridesB -= 1
    self.numSgprSizesSum = kernel["ProblemType"]["NumIndicesSummation"]

    # Number of D strides to actually load into SGPR
    # Save some SGPR if LdcEqualsLdd
    self.numSgprStridesDToLoad = 0 if kernel["LdcEqualsLdd"] else self.numSgprStridesD

    self.numSgprSizesFree = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprAddressDbg = self.rpga if globalParameters["DebugKernel"] else 0

    ####################################
    # num sgprs: global read increments
    if self.globalReadIncsUseVgpr:
      numSgprGlobalReadIncsA = 0
      numSgprGlobalReadIncsB = 0
    else:
      numSgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpgo
      numSgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpgo



    ########################################
    # SGPR Assignment according to AMDGPU-ABI
    ########################################

    self.defineSgpr("KernArgAddress", self.rpga)
    assert(self.sgprs["KernArgAddress"] ==  0) # kernarg is passed to kernel as SGPR0

    if kernel["WorkGroupMapping"]>=0 :
      self.defineSgpr("WorkGroup0", 1)
      self.defineSgpr("WorkGroup1", 1)
    else:
      self.defineSgpr("WorkGroup1", 1)
      self.defineSgpr("WorkGroup0", 1)

    wg=2
    for idx in kernel["ProblemType"]["IndicesBatch"]:
      if not isPackedIndex(kernel,idx):
        self.defineSgpr("WorkGroup%u"%wg, 1)
        wg+=1

    # SGPR above are user SGPR which are set by GPU hardware when the kernel is launched
    self.firstInitSgpr = self.sgprPool.size()
    
    # To avoid corrupting tmp sgprs that may be used around the assert,
    # reserve some sgprs to save/restore the execmask
    if self.db["EnableAsserts"]:
      self.defineSgpr("SaveExecMask", 2, 2)

    self.defineSgpr("GSUSumIdx", 2 if kernel["GlobalSplitU"] > 1 else 0)

    self.sumMagicParms = []
    if kernel["PackSummationDims"]:
      self.magicSumChars = [globalParameters["IndexChars"][c] for c in kernel["ProblemType"]["IndicesSummation"][1:]]

      self.sumMagicParms=["%s"%idxChar for idxChar in self.magicSumChars]
      if kernel["PackSummationDims"] and kernel["GlobalSplitU"] > 1 and self.sumMagicParms:
          self.sumMagicParms.append("%s_GsuRemainder"%self.unrollChar)

      for magicName in self.sumMagicParms:
        if kernel["MagicDivAlg"]==2:
          self.defineSgpr("MagicAbitSize%s"%magicName, 1)

    # for packed batches without stride restrictions need to do something different here
    assert sorted(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]) == \
           sorted(set(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]))
    for idxChar in kernel["PackedC0IdxChars"][:-1]:
      if kernel["MagicDivAlg"]==2:
        self.defineSgpr("MagicAbitSize%s"%idxChar, 1)
    for idxChar in kernel["PackedC1IdxChars"][:-1]:
      if kernel["MagicDivAlg"]==2:
        self.defineSgpr("MagicAbitSize%s"%idxChar, 1)

    # product of all packed dims in the 0 or 1 dimensions:
    if len(kernel["PackedC0IndicesX"]) > 1:
      self.defineSgpr("PackedSize0", 1)
    if len(kernel["PackedC1IndicesX"]) > 1:
      self.defineSgpr("PackedSize1", 1)
    
    if kernel["PackSummationDims"]:
      self.defineSgpr(self.loopCounterName(kernel,self.unrollIdx), 1)
    else:
      # contractions with multiple summations will use multiple LoopCounters, if PSD=0
      for i in range(kernel["ProblemType"]["NumIndicesSummation"]):
        self.defineSgpr(self.loopCounterName(kernel,i), 1)
        # numRemainderSumElements is required by multi-k matrix product in the multiply-accumulate loop. 
        # It means in the final iteration of each tail loop, the last N elem along summation dimension 
        # should be filled with 0's (numRemainderSumElements = numIterK % MatrixInstK)
        if kernel["EnableMatrixInstruction"] and \
          kernel["AssertSummationElementMultiple"] % kernel["MatrixInstK"] != 0:
          self.defineSgpr("NumRemainderSumElements%s" % self.loopChar(kernel, i), 1)

    self.defineSgpr("OrigLoopCounter", 1)

    if self.prefetchAcrossPersistent0:
      if kernel["ExpandPointerSwap"]:
        # For ExpandPointerSwap + PAP, track which expanded loop iter to start on
        # global prefetches bounce between two LDS buffers, and the bounce state
        # must be maintained across PK boundaries.
        # If the no-load-loop is present it counts as one iteration and
        # So if K is even multiple of unroll then we exit at odd iteration
        # and each PK loop will start on the second expanded pointer swap
        self.defineSgpr("EvenIterStart", 1)
      self.defineSgpr("TailLoopCounter", 1)
    if globalParameters["DebugKernel"]:
      self.defineSgpr("AddressDbg", self.numSgprAddressDbg)
      self.defineSgpr("DebugKernelItems", 1)

    if kernel["BufferLoad"]:
       # resource descriptor (SRD) A and B, must be aligned on 4-SGPR boundary
      self.defineSgpr("SrdA", 4, 4)
      self.defineSgpr("SrdB", 4, 4)
    if kernel["BufferStore"]:
      self.defineSgpr("SrdD", 4, 4)
      self.defineSgpr("SrdC", 4, 4)

    # fill empty Sgpr slot caused by Sgpr alignment, 
    # because we need following defineSgpr use continous sgpr
    SgprSlot = []
    currentSize = self.sgprPool.size()
    while (1):
      tempSgpr = self.sgprPool.checkOut(1,"fill empty slot temporarily",preventOverflow=0)
      if tempSgpr >= currentSize:
        self.sgprPool.checkIn(tempSgpr)
        break
      SgprSlot.append(tempSgpr)

    ###################################
    # Get kernel argument start here
    self.defineSgpr("Tensor2dSizeA", 2,4)
    self.defineSgpr("Tensor2dSizeB", 2,2)
    self.defineSgpr("AddressD", numSgprAddressD)
    self.defineSgpr("AddressC", numSgprAddressC)
    self.defineSgpr("AddressA", numSgprAddressA)
    self.defineSgpr("AddressB", numSgprAddressB)
    self.defineSgpr("Alpha", numSgprAlpha, numSgprAlpha)
    if kernel["ProblemType"]["UseBeta"]:
      self.defineSgpr("Beta", numSgprBeta, numSgprBeta)
    self.defineSgpr("StridesD", self.numSgprStridesD)
    self.defineSgpr("StridesC", self.numSgprStridesC)
    self.defineSgpr("StridesA", self.numSgprStridesA)
    self.defineSgpr("StridesB", self.numSgprStridesB)
    self.defineSgpr("SizesFree", self.numSgprSizesFree)
    self.defineSgpr("SizesSum", self.numSgprSizesSum)
    self.sumMagicParms = []
    if kernel["PackSummationDims"]:
      self.magicSumChars = [globalParameters["IndexChars"][c] for c in kernel["ProblemType"]["IndicesSummation"][1:]]
      self.sumMagicParms=["%s"%idxChar for idxChar in self.magicSumChars]
      if kernel["PackSummationDims"] and kernel["GlobalSplitU"] > 1 and self.sumMagicParms:
          self.sumMagicParms.append("%s_GsuRemainder"%self.unrollChar)
      for magicName in self.sumMagicParms:
        self.defineSgpr("MagicNumberSize%s"%magicName, 1)
        self.defineSgpr("MagicShiftSize%s"%magicName, 1)
    # for packed batches without stride restrictions need to do something different here
    assert sorted(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]) == \
           sorted(set(kernel["PackedC0IdxChars"]+kernel["PackedC1IdxChars"]))
    for idxChar in kernel["PackedC0IdxChars"][:-1]:
      self.defineSgpr("MagicNumberSize%s"%idxChar, 1)
      self.defineSgpr("MagicShiftSize%s"%idxChar, 1)
    for idxChar in kernel["PackedC1IdxChars"][:-1]:
      self.defineSgpr("MagicNumberSize%s"%idxChar, 1)
      self.defineSgpr("MagicShiftSize%s"%idxChar, 1)
    for tc in ('A', 'B'):
      for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
        (freeDim, sumDim, padStart, padEnd) = zp
        freeDimChar = globalParameters["IndexChars"][freeDim]
        sumDimChar  = globalParameters["IndexChars"][sumDim]
        # These will eventually be read as kernel args:
        self.defineSgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar),1)
        self.defineSgpr("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar),1)
    self.defineSgpr("OrigStaggerUIter", 1)  # Original stagger register.  Only needed for Persistent
    self.defineSgpr("NumWorkGroups0", 1)
    self.defineSgpr("NumWorkGroups1", 1)
    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    self.lastPostLoopSgpr = self.sgprPool.size()
    self.defineSgpr("MagicNumberProblemNumGroupTiles0", 1) # Magic number to use for division
    self.defineSgpr("GridNumWorkGroups0", 1) # Magic number to use for division
    self.defineSgpr("NumFullBlocks", 1) # Magic number to use for div by (NumWorkGroups1 % WGM)
    self.defineSgpr("WgmRemainder1", 1) # Magic number to use for div by (NumWorkGroups1 % WGM)
    self.defineSgpr("MagicNumberWgmRemainder1", 1) # Magic number to use for div by (NumWorkGroups1 % WGM)
    
    self.numSgprToLoad = 2 + 2 + numSgprAddressD + numSgprAddressC + numSgprAddressA + numSgprAddressB + numSgprAlpha + \
      (numSgprBeta if kernel["ProblemType"]["UseBeta"] else 0) + self.numSgprStridesD + self.numSgprStridesC + self.numSgprStridesA + \
      self.numSgprStridesB + self.numSgprSizesFree + self.numSgprSizesSum + len(self.sumMagicParms)*2 + len(kernel["PackedC0IdxChars"][:-1])*2 + \
      len(kernel["PackedC1IdxChars"][:-1])*2 + len(kernel["ProblemType"]["ZeroPadA"])*2 + len(kernel["ProblemType"]["ZeroPadB"])*2 + 1 + 2 + 2 + 3
    # Get kernel argument end here
    ###################################
    
    # put unused Sgpr back to SgprPool
    while SgprSlot:
      tempSgpr = SgprSlot.pop(0)
      self.sgprPool.checkIn(tempSgpr)
    if not self.numSgprStridesDToLoad:
      self.undefineSgpr("StridesD")
    if not self.staggerU:
      self.undefineSgpr("OrigStaggerUIter")  # Original stagger register.  Only needed for Persistent
    if not kernel["PersistentKernel"]:
      self.undefineSgpr("MagicNumberProblemNumGroupTiles0") # Magic number to use for division
      self.undefineSgpr("GridNumWorkGroups0") # Magic number to use for division

    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    # self.lastPostLoopSgpr = self.sgprPool.size()

    if self.unrollIncIsDepthU:
      # product of all summation dimensions, this also will be divided if GSU is enabled
      self.defineSgpr("UnrollLoopLastIter", 1)

    if kernel["PackSummationDims"] and kernel["GlobalSplitU"]>1:
      self.defineSgpr("GsuNumIter%s"%self.loopChar(kernel,self.unrollIdx), 1)

    for tc in ('A', 'B'):
      for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
        (freeDim, sumDim, padStart, padEnd) = zp
        freeDimChar = globalParameters["IndexChars"][freeDim]
        sumDimChar  = globalParameters["IndexChars"][sumDim]
        # These will eventually be read as kernel args:
        self.defineSgpr("ElementEdge%s%s"%(tc, sumDimChar),1)
        if kernel["PackSummationDims"]:
          self.defineSgpr("Iter%s"%(sumDimChar),1)

    if kernel["FractionalLoad"] == 2:
      if kernel["fractionalPerpOverhangA"]:
        self.defineSgpr("PerpOverhangVccA", 2, 2)
      if kernel["fractionalPerpOverhangB"]:
        self.defineSgpr("PerpOverhangVccB", 2, 2)
    if self.use64bShadowLimit:
      # If need more SGPR could overlap this with the Tensor2dSize regs
      self.defineSgpr("ShadowLimitA", 2, 2)
      self.defineSgpr("ShadowLimitB", 2, 2)

    if kernel["PackSummationDims"]:
      for tc in ('A','B'):
        self.defineSgpr("InitialSrd%sBase"%tc, 2)
        self.defineSgpr("InitialSrd%sLimit"%tc, 2 if self.use64bShadowLimit else 1)

    if self.staggerU:
      self.defineSgpr("StaggerUIter", 1)  # stagger loop iterations, used for various iter counts in the code
      self.defineSgpr("WrapUA", 2)  # Bytes to add to SrdA to reset address from N-1 iter to AddressA
      self.defineSgpr("WrapUB", 2)  # Bytes to add to SrdB to reset address from N-1 iter to AddressB

    if kernel["PersistentKernel"]:
      self.defineSgpr("SerialWorkGroupIter", 1) # Track sequential persistent wg
    if self.prefetchAcrossPersistent0:
      self.defineSgpr("PrevWorkGroup0", 1) # WorkGroup0 from prev iteration, use for stores
      self.defineSgpr("PrevWorkGroup1", 1) # WorkGroup0 from prev iteration, use for stores


    self.defineSgpr("GlobalReadIncsA", numSgprGlobalReadIncsA)
    self.defineSgpr("GlobalReadIncsB", numSgprGlobalReadIncsB)

    if kernel["LocalWriteUseSgprA"]:
        self.defineSgpr("LocalWriteAddrA", 1)
    if kernel["LocalWriteUseSgprB"]:
        self.defineSgpr("LocalWriteAddrB", 1)

    if kernel["_UseSgprForGRO"]:
      self.defineSgpr("ScalarGlobalReadOffsetA", numGlobalReadOffsetsA-1)
      self.defineSgpr("ScalarGlobalReadOffsetB", numGlobalReadOffsetsB-1)

    # debug flag to allocate dummy / unused sgpr
    # useful when comparing code that adds new kernel arguments to see what
    # was actually changed
    numDummySgpr= 0
    for i in range(numDummySgpr):
      self.defineSgpr("DummySgpr%d"%i, 1)

    if self.sgprPool.size() >= self.maxSgprs:
      print ("warning: Number of defined SGPRS (%d) overflowed max SGPRS (%d)." \
               % (self.sgprPool.size(), self.maxSgprs))

    # TODO-persistent - likely recompute some of the registers above.
    if kernel["PersistentKernel"]:
      self.lastPostLoopSgpr = self.sgprPool.size()

    ########################################
    # AGPR Allocation
    ########################################
    self.totalAgprs = 0
    if "MatrixInstM" in kernel:
      self.destAgprs  = kernel["MatrixInstM"] * kernel["MatrixInstN"] * kernel["MatrixInstB"] // globalParameters["WavefrontWidth"]
      self.totalAgprs = self.destAgprs * kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1]

    ########################################
    # Register Pools
    ########################################
    #print "TotalVgprs", self.totalVgprs
    self.vgprPool = RegisterPool(self.totalVgprs, 'v', reservedAtEnd=1, defaultPreventOverflow=False,
                                 printRP=self.db["PrintRP"])
    #print self.vgprPool.state()
    self.savedVgprPool = None

    # C regs are not used during initialization so mark them as available - 
    # we will claim then just before the start of the unroll loop:
    self.vgprPool.add(self.startVgprValuC, \
        self.numVgprValuC, "ValuC-Block") # Add as available
    self.vgprPool.add(self.startVgprValuA, \
        self.lastValuAB - self.startVgprValuA, "ValuAB") # Add as available
    #print self.vgprPool.state()

    self.agprPool = RegisterPool(self.totalAgprs, 'a', 0, defaultPreventOverflow=False, printRP=0)
    # C regs are not used during initialization so mark them as available - 
    # we will claim then just before the start of the unroll loop:
    self.agprPool.add(0, self.totalAgprs, "ValuC-Block")

    # place any of these gpr inst values into tPA, tPB for later reference
    tPA["globalReadInstruction"] = self.globalReadInstructionA
    tPA["localWriteInstruction"] = self.localWriteInstructionA
    tPA["localReadInstruction"] = self.localReadInstructionA
    tPA["gpr"] = {}

    tPB["globalReadInstruction"] = self.globalReadInstructionB
    tPB["localWriteInstruction"] = self.localWriteInstructionB
    tPB["localReadInstruction"] = self.localReadInstructionB
    tPB["gpr"] = {}

    # pre-determine labels in order
    unrollChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
    self.labels = {}
    #self.getLabelNum("PrefetchGlobalBegin")
    self.getLabelNum("PrefetchGlobalEnd")
    self.getLabelNum("LoopBegin%s"%(unrollChar))
    self.getLabelNum("LoopEnd%s"%(unrollChar))
    self.getLabelNum("LoopEnd%s_oddexit"%(unrollChar))
    self.getLabelNum("PrefetchGlobalLastIterEnd")
    self.getLabelNum("TailLoopBegin%s"%(unrollChar))
    self.getLabelNum("TailLoopEnd%s"%(unrollChar))
    self.getLabelNum("KernelEnd%s"%(unrollChar))
    # shift vectors determined later

    assert not self.db["CheckValueC"] or kernel["ProblemType"]["DataType"].isSingle()

    if self.db["InitLds"] : print ("\n***WARNING: InitLds enabled, may impact performance\n")
    if self.db["InitSgpr"] : print ("\n***WARNING: InitSgpr enabled, may impact performance\n")
    if self.db["InitVgpr"] : print ("\n***WARNING: InitVgpr enabled, may impact performance\n")
    if self.db["ConservativeWaitCnt"] : print ("\n***WARNING: ConservativeWaitCnt enabled, may impact performance\n")
    if self.do["KeepDirectToLdsAlloc"] : print ("\n***WARNING: KeepDirectToLdsAlloc enabled, may impact performance\n")
    if not kernel["LoopTail"] : print ("\n***WARNING: LoopTail disabled, kernel may not function correctly for all inputs\n")
    if self.db["CheckValue1A"] : print ("\n***WARNING: CheckValue1A enabled, may impact performance\n")
    if self.db["CheckValue1B"] : print ("\n***WARNING: CheckValue1B enabled, may impact performance\n")
    if self.db["CheckValueC"] : print ("\n***WARNING: CheckValueC enabled, may impact performance\n")
    if self.db["ForceExpectedValue"] : print ("\n***WARNING: ForceExpectedValue enabled, may impact functionality\n")
    if self.db["ForceVSerial"] : print ("\n***WARNING: ForceVSerial enabled, will impact functionality\n")
    if self.db["ForceInputValueA"] : print ("\n***WARNING: ForceInputValueA enabled, may impact functionality\n")
    if self.db["ForceInputValueB"] : print ("\n***WARNING: ForceInputValueB enabled, may impact functionality\n")
    if self.db["CheckStoreC"] >=0  : print ("\n***WARNING: CheckStoreC enabled, may impact performance\n")
    if self.db["ForceEdgeStores"] : print ("\n***WARNING: ForceEdgeStores enabled, may impact performance\n")
    if self.db["AssertNoEdge"] : print ("\n***WARNING: AssertNoEdge enabled, may impact functionality and performance\n")
    if self.db["PrintRP"] : print ("\n***WARNING: PrintRP enabled, may generate verbose output\n")
    if kernel["CheckTensorDimAsserts"] : print ("\n***WARNING: CheckTensorDimAsserts enabled, may impact performance\n")
    if kernel["CheckDimOverflow"] : print ("\n***WARNING: CheckDimOverflow enabled, may impact performance\n")


  ##############################################################################
  # format macro
  def macroRegister(self, name, value):
    return ".set %s, %s%s" % (name, value, self.endLine)

  def v2Argument(self, name, size, align, valueKind, valueType, AddrSpaceQual = None):
    kStr = ""
    kStr += "      - Name:            %s\n" % name
    kStr += "        Size:            %s\n" % size
    kStr += "        Align:           %s\n" % align
    kStr += "        ValueKind:       %s\n" % valueKind
    kStr += "        ValueType:       %s\n" % valueType
    if AddrSpaceQual != None:
      kStr += "        AddrSpaceQual:   %s\n" % AddrSpaceQual
    return kStr

  def v3Argument(self, name, size, offset, valueKind, valueType, AddrSpaceQual = None):
    kStr = ""
    kStr += "      - .name:            %s\n" % name
    kStr += "        .size:            %s\n" % size
    kStr += "        .offset:          %s\n" % offset
    kStr += "        .value_kind:      %s\n" % valueKind
    kStr += "        .value_type:      %s\n" % valueType
    if AddrSpaceQual != None:
      kStr += "        .address_space:   %s\n" % AddrSpaceQual
    return kStr

  ##############################################################################
  # Function Prefix
  ##############################################################################
  def functionPrefix(self, kernel):
    kStr = ""

    return kStr

  def defineMACs(self, kernel, m, innerUnroll):

    kStr = ""
    beAggressive = kernel["AggressivePerfMode"]

    doOnce = False
    macIdx = 0
    # half precision
    if kernel["ProblemType"]["DataType"].isHalf():
      for blockB in range(0, kernel["ThreadTile1"]//2):
        for blockA in range(0, kernel["ThreadTile0"]//2):
          if self.version == (8,0,3):
            for b in range(blockB*2, (blockB+1)*2):
              for a in range(blockA*2, (blockA+1)*2):
                for iui in range(0, innerUnroll):
                  # v_mac_f16 or v_fma_f16
                  cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"])
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iui), blockA)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iui), blockB)
                  kStr += "v_mac_f16 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine) # FIXME op_sel
                  if beAggressive and not doOnce:
                    kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
                    doOnce = True
          elif self.version == (9,0,0):
            if kernel["ProblemType"]["HighPrecisionAccumulate"]:
              # we treat HighPrecisionAccumulate as expanded packed math
              b = blockB*2
              a = blockA*2
              if kernel["LocalDotLayout"] > 1 and innerUnroll == 2:    # Only supports LocalDotLayout == 2 for now
                cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 0
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I0"%m, blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I0"%m, blockB)
                kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                if beAggressive and not doOnce:
                  kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
                  doOnce = True
                kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I1"%m, blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I0"%m, blockB)
                cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I0"%m, blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I1"%m, blockB)
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I1"%m, blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I1"%m, blockB)
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                #kStr += self.bomb(-13)
                """
                ignore this, not quite correct for mixed precision
                D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                C[0] = A[0]*B[0]+D[0]
                C[1] = A[1]*B[1]+D[1]
                """
              else:
                for iui in range(0, innerUnroll):
                  cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 0
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iui), blockA)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iui), blockB)
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] iui=%u%s" % (cStr, aStr, bStr, cStr, cidx, iui, self.endLine)
                  if beAggressive and not doOnce:
                    kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
                    doOnce = True
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                  cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                  cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                  cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  """
                  ignore this, not quite correct for mixed precision
                  D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                  D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                  C[0] = A[0]*B[0]+D[0]
                  C[1] = A[1]*B[1]+D[1]
                  """
            else:
              b = blockB*2
              a = blockA*2
              for iui in range(0, innerUnroll):
                cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # /2 b/c of 2 f16's per 32-bit vgpr
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I%u"%(m,iui), blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I%u"%(m,iui), blockB)
                kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,0,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
                if beAggressive and not doOnce:
                  kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
                  doOnce = True
                cStr = "v[%s+%u+%u*%u+%u]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
                """
                D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                C[0] = A[0]*B[0]+D[0]
                C[1] = A[1]*B[1]+D[1]
                """
          elif self.version == (9,0,6) or self.version == (9,0,8):
            if kernel["ProblemType"]["HighPrecisionAccumulate"]:
              # we treat HighPrecisionAccumulate as expanded packed math
              b = blockB*2
              a = blockA*2
              if kernel["LocalDotLayout"] > 1 and innerUnroll == 2:    # Only supports LocalDotLayout == 2 for now
                cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 0
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I0"%m, blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I0"%m, blockB)
                kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                if beAggressive and not doOnce:
                  kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
                  doOnce = True
                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I1"%m, blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I0"%m, blockB)
                cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I0"%m, blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I1"%m, blockB)
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I1"%m, blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I1"%m, blockB)
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                #kStr += self.bomb(-13)
                """
                ignore this, not quite correct for mixed precision
                D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                C[0] = A[0]*B[0]+D[0]
                C[1] = A[1]*B[1]+D[1]
                """
              else:
                for iui in range(0, innerUnroll):
                  cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 0
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iui), blockA)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iui), blockB)
                  kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] iui=%u%s" % (cStr, aStr, bStr, cStr, cidx, iui, self.endLine)
                  if beAggressive and not doOnce:
                    kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
                    doOnce = True
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                  cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                  kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[1,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                  cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                  kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                  cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                  kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  """
                  ignore this, not quite correct for mixed precision
                  D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                  D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                  C[0] = A[0]*B[0]+D[0]
                  C[1] = A[1]*B[1]+D[1]
                  """
                #kStr += self.bomb(-13)
            else:
              b = blockB*2
              a = blockA*2
              for iui in range(0, innerUnroll):
                cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # /2 b/c of 2 f16's per 32-bit vgpr
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I%u"%(m,iui), blockA)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I%u"%(m,iui), blockB)
                kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,0,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
                if beAggressive and not doOnce:
                  kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
                  doOnce = True
                cStr = "v[%s+%u+%u*%u+%u]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
                """
                D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                C[0] = A[0]*B[0]+D[0]
                C[1] = A[1]*B[1]+D[1]
                """
          else:
            printExit("Half-precision not supported for arch=%u" % self.version )
      if beAggressive:
        kStr += "s_setprio 0 // Reset priority after macs%s" % self.endLine

    # bfloat16
    elif kernel["ProblemType"]["DataType"].isBFloat16():
      if self.version == (9,0,8) and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        for iui in range(0, innerUnroll):
          for blockA in range(kernel["ThreadTile0"]//2-1, -1, -1):
            kStr += "v_and_b32     v[vgprValuA_X%u_I%u+%u], 0xffff0000, v[vgprValuA_X%u_I%u+%u]%s" % (m, iui, blockA*2+1, m, iui, blockA, self.endLine)
            kStr += "v_lshlrev_b32 v[vgprValuA_X%u_I%u+%u], 16,         v[vgprValuA_X%u_I%u+%u]%s" % (m, iui, blockA*2,   m, iui, blockA, self.endLine)

          for blockB in range(kernel["ThreadTile1"]//2-1, -1, -1):
            kStr += "v_and_b32     v[vgprValuB_X%u_I%u+%u], 0xffff0000, v[vgprValuB_X%u_I%u+%u]%s" % (m, iui, blockB*2+1, m, iui, blockB, self.endLine)
            kStr += "v_lshlrev_b32 v[vgprValuB_X%u_I%u+%u], 16,         v[vgprValuB_X%u_I%u+%u]%s" % (m, iui, blockB*2,   m, iui, blockB, self.endLine)

        for blockB in range(0, kernel["ThreadTile1"]//2):
          for blockA in range(0, kernel["ThreadTile0"]//2):
            if kernel["ProblemType"]["HighPrecisionAccumulate"]:
              # we treat HighPrecisionAccumulate as expanded packed math
              b = blockB*2
              a = blockA*2
              for iui in range(0, innerUnroll):
                aStr0 = "v[%s+%u]" % ("vgprValuA_X%u_I%u"%(m,iui), blockA*2+0)
                aStr1 = "v[%s+%u]" % ("vgprValuA_X%u_I%u"%(m,iui), blockA*2+1)
                bStr0 = "v[%s+%u]" % ("vgprValuB_X%u_I%u"%(m,iui), blockB*2+0)
                bStr1 = "v[%s+%u]" % ("vgprValuB_X%u_I%u"%(m,iui), blockB*2+1)

                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 0
                cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                kStr += "v_fma_f32 %s, %s, %s, %s //ValuC[%u]%s" % (cStr, aStr0, bStr0, cStr, cidx, self.endLine)

                if beAggressive and not doOnce:
                  kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
                  doOnce = True

                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                kStr += "v_fma_f32 %s, %s, %s, %s //ValuC[%u]%s" % (cStr, aStr1, bStr0, cStr, cidx, self.endLine)

                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                kStr += "v_fma_f32 %s, %s, %s, %s //ValuC[%u]%s" % (cStr, aStr0, bStr1, cStr, cidx, self.endLine)

                cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                kStr += "v_fma_f32 %s, %s, %s, %s //valuC[%u]%s" % (cStr, aStr1, bStr1, cStr, cidx, self.endLine)
                """
                ignore this, not quite correct for mixed precision
                D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                C[0] = A[0]*B[0]+D[0]
                C[1] = A[1]*B[1]+D[1]
                """
                #kStr += self.bomb(-13)
      else:
        printExit("Bfloat16 not supported for arch=%u" % self.version )


    # integer i8
    elif kernel["ProblemType"]["DataType"].isInt8x4():
      for b in range(0, kernel["ThreadTile1"]):
        for a in range(0, kernel["ThreadTile0"]):
          if self.version == (8,0,3):
            kStr += self.comment3("int8 not implemented yet for gfx803:")
          elif self.version == (9,0,0):
            kStr += self.comment3("int8 not implemented yet for gfx900:")
          elif self.version == (9,0,6) or self.version == (9,0,8):
            for iui in range(0, innerUnroll):
              cidx = a + b*kernel["ThreadTile0"] + 0
              cStr = "v[%s+%u+%u*%u]" % ("vgprValuC", a, b, kernel["ThreadTile0"])
              aStr = "v[%s+%u]"       % ("vgprValuA_X%u_I%u"%(m,iui), a)
              bStr = "v[%s+%u]"       % ("vgprValuB_X%u_I%u"%(m,iui), b)
              kStr += "v_dot4_i32_i8  %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              if beAggressive and not doOnce:
                kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
                doOnce = True
      if beAggressive:
        kStr += "s_setprio 0 // Reset priority after macs %s" % self.endLine

    # single precision
    elif kernel["ProblemType"]["DataType"].isSingle():
      for b in range(0, kernel["ThreadTile1"]):
        for a in range(0, kernel["ThreadTile0"]):
          for iui in range(0, innerUnroll):
            cStr = "v[%s+%u+%u*%u]" % ("vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u]" \
                % ("vgprValuA_X%u_I%u"%(m,iui), a)
            bStr = "v[%s+%u]" \
                % ("vgprValuB_X%u_I%u"%(m,iui), b)
            #if a==0 and b==0:
            #  kStr += dump(aStr)
            kStr += "v_mac_f32 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine)
            if beAggressive and not doOnce:
              kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
              doOnce = True
            if macIdx == kernel["PerformanceWaitLocation"]:
                kStr += "s_waitcnt lgkmcnt(%u) // extra wait for performance%s" \
                    % (kernel["PerformanceWaitCount"], self.endLine)
            if macIdx == kernel["PerformanceSyncLocation"]:
                kStr += "s_barrier // extra barrier for performance%s" \
                    % (self.endLine)
            macIdx += 1
      if beAggressive:
        kStr += "s_setprio 0 // Reset priority after macs %s" % self.endLine

    # double precision
    elif kernel["ProblemType"]["DataType"].isDouble():
      for b in range(0, kernel["ThreadTile1"]):
        for a in range(0, kernel["ThreadTile0"]):
          for iui in range(0, innerUnroll):
            cStr = "v[%s+(%u+%u*%u)*2:(%s+%u+%u*%u)*2+1]" % ("vgprValuC", a, b, kernel["ThreadTile0"], "vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*2:%s+%u*2+1]" \
                % ("vgprValuA_X%u_I%u"%(m,iui) , a, "vgprValuA_X%u_I%u"%(m,iui), a)
            bStr = "v[%s+%u*2:%s+%u*2+1]" \
                % ("vgprValuB_X%u_I%u"%(m,iui) , b, "vgprValuB_X%u_I%u"%(m,iui), b)
            kStr += "v_fma_f64 %s, %s, %s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)
            if beAggressive and not doOnce:
              kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
              doOnce = True
      if beAggressive:
        kStr += "s_setprio 0 // Reset priority after macs %s" % self.endLine

    # single precision complex
    elif kernel["ProblemType"]["DataType"].isSingleComplex():
      for b in range(0, kernel["ThreadTile1"]):
        for a in range(0, kernel["ThreadTile0"]):
          for iui in range(0, innerUnroll):
            cStr = "v[%s+(%u+%u*%u)*2]" % ("vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*2]" % ("vgprValuA_X%u_I%u"%(m,iui) , a)
            bStr = "v[%s+%u*2]" % ("vgprValuB_X%u_I%u"%(m,iui) , b)
            kStr += "v_mac_f32 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine)

            cStr = "v[%s+(%u+%u*%u)*2]" % ("vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*2+1]" % ("vgprValuA_X%u_I%u"%(m,iui) , a)
            bStr = "v[%s+%u*2+1]" % ("vgprValuB_X%u_I%u"%(m,iui) , b)
            if (not kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]) or \
               (kernel["ProblemType"]["ComplexConjugateA"] and kernel["ProblemType"]["ComplexConjugateB"]):
              kStr += "v_mac_f32 %s, -%s, %s%s" % (cStr, aStr, bStr, self.endLine)
            else:
              kStr += "v_mac_f32 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine)

            cStr = "v[%s+(%u+%u*%u)*2+1]" % ("vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*2]" % ("vgprValuA_X%u_I%u"%(m,iui) , a)
            bStr = "v[%s+%u*2+1]" % ("vgprValuB_X%u_I%u"%(m,iui) , b)
            if kernel["ProblemType"]["ComplexConjugateB"]:
              kStr += "v_mac_f32 %s, %s, -%s%s" % (cStr, aStr, bStr, self.endLine)
            else:
              kStr += "v_mac_f32 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine)

            cStr = "v[%s+(%u+%u*%u)*2+1]" % ("vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*2+1]" % ("vgprValuA_X%u_I%u"%(m,iui) , a)
            bStr = "v[%s+%u*2]" % ("vgprValuB_X%u_I%u"%(m,iui) , b)
            if kernel["ProblemType"]["ComplexConjugateA"]:
              kStr += "v_mac_f32 %s, -%s, %s%s" % (cStr, aStr, bStr, self.endLine)
            else:
              kStr += "v_mac_f32 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine)

            if beAggressive and not doOnce:
              kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
              doOnce = True
      if beAggressive:
        kStr += "s_setprio 0 // Reset priority after macs %s" % self.endLine

    # double precision complex
    elif kernel["ProblemType"]["DataType"].isDoubleComplex():
      for b in range(0, kernel["ThreadTile1"]):
        for a in range(0, kernel["ThreadTile0"]):
          for iui in range(0, innerUnroll):
            # c.real += a.real * b.real 
            cStr = "v[%s+(%u+%u*%u)*4+0:(%s+%u+%u*%u)*4+1]" % ("vgprValuC", a, b, kernel["ThreadTile0"], "vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*4+0:%s+%u*4+1]" % ("vgprValuA_X%u_I%u"%(m,iui) , a, "vgprValuA_X%u_I%u"%(m,iui), a)
            bStr = "v[%s+%u*4+0:%s+%u*4+1]" % ("vgprValuB_X%u_I%u"%(m,iui) , b, "vgprValuB_X%u_I%u"%(m,iui), b)
            kStr += "v_fma_f64 %s, %s, %s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)
            # c.real -= a.imag * b.imag
            cStr = "v[%s+(%u+%u*%u)*4+0:(%s+%u+%u*%u)*4+1]" % ("vgprValuC", a, b, kernel["ThreadTile0"], "vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*4+2:%s+%u*4+3]" % ("vgprValuA_X%u_I%u"%(m,iui) , a, "vgprValuA_X%u_I%u"%(m,iui), a)
            bStr = "v[%s+%u*4+2:%s+%u*4+3]" % ("vgprValuB_X%u_I%u"%(m,iui) , b, "vgprValuB_X%u_I%u"%(m,iui), b)
            if kernel["ProblemType"]["ComplexConjugateA"] and kernel["ProblemType"]["ComplexConjugateB"]:
              kStr += "v_fma_f64 %s, %s, -%s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)
            elif kernel["ProblemType"]["ComplexConjugateA"] or kernel["ProblemType"]["ComplexConjugateB"]:
              kStr += "v_fma_f64 %s, %s, %s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)
            else:
              kStr += "v_fma_f64 %s, %s, -%s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)
            # c.imag += a.real * b.imag
            cStr = "v[%s+(%u+%u*%u)*4+2:(%s+%u+%u*%u)*4+3]" % ("vgprValuC", a, b, kernel["ThreadTile0"], "vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*4+0:%s+%u*4+1]" % ("vgprValuA_X%u_I%u"%(m,iui) , a, "vgprValuA_X%u_I%u"%(m,iui), a)
            bStr = "v[%s+%u*4+2:%s+%u*4+3]" % ("vgprValuB_X%u_I%u"%(m,iui) , b, "vgprValuB_X%u_I%u"%(m,iui), b)
            if kernel["ProblemType"]["ComplexConjugateB"]:
              kStr += "v_fma_f64 %s, %s, -%s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)
            else:
              kStr += "v_fma_f64 %s, %s, %s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)
            # c.imag += a.imag * b.real
            cStr = "v[%s+(%u+%u*%u)*4+2:(%s+%u+%u*%u)*4+3]" % ("vgprValuC", a, b, kernel["ThreadTile0"], "vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*4+2:%s+%u*4+3]" % ("vgprValuA_X%u_I%u"%(m,iui) , a, "vgprValuA_X%u_I%u"%(m,iui), a)
            bStr = "v[%s+%u*4+0:%s+%u*4+1]" % ("vgprValuB_X%u_I%u"%(m,iui) , b, "vgprValuB_X%u_I%u"%(m,iui), b)
            if kernel["ProblemType"]["ComplexConjugateA"]:
              kStr += "v_fma_f64 %s, -%s, %s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)
            else:
              kStr += "v_fma_f64 %s, %s, %s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)
              
            if beAggressive and not doOnce:
              kStr += "s_setprio 1 // Raise priority while processing macs%s" % self.endLine
              doOnce = True
      if beAggressive:
        kStr += "s_setprio 0 // Reset priority after macs %s" % self.endLine
        
      # other precision
    else:
      printExit("Assembly doesn't support %s" % kernel["ProblemType"]["DataType"])

    return kStr


  def defineMACMacro(self, kernel, innerUnroll, useMacro):

    kStr = ""
    # Create a macro version that processes just one U iter
    # (used in tail loop in some cases)
    oneIUI = kernel["InnerUnroll"] > 1 and innerUnroll==1

    ########################################
    # MACs
    kStr += self.comment3("%dx%d thread-tile" \
        % (kernel["ThreadTile0"], kernel["ThreadTile1"]) )
    for m in range(0, 1+kernel["PrefetchLocalRead"]):
      # Create a special macro that does one K iter if needed:
      ext = "_OneIUI" if oneIUI else ""
      if useMacro:
        kStr += ".macro MAC_%ux%u_X%u%s" \
            % (kernel["ThreadTile0"], kernel["ThreadTile1"], m, ext)
      kStr += self.endLine

      kStr += self.defineMACs(kernel, m, innerUnroll)


      if useMacro:
        kStr += ".endm%s" % self.endLine


    return kStr

  def defineCMPXMacros(self, kernel):
    """
    Navi's cmpx instruction writes only to EXEC, not to SGPRs or to VCC.
    For now, replicate old behaviour with two instructions.
    """
    def macro(op, dtype):
      dict = {'op': op, 'dtype': dtype}
      mStr = ".macro _v_cmpx_{op}_{dtype} dst, src0, src1=".format(**dict) + self.endLine
      if self.archCaps["CMPXWritesSGPR"]:
        mStr += r"   v_cmpx_{op}_{dtype} \dst \src0 \src1 ".format(**dict) + self.endLine
      else:
        mStr += r"   v_cmp_{op}_{dtype} \dst \src0 \src1".format(**dict) + self.endLine
        mStr += r"   s_mov_b64 exec \dst" + self.endLine
      mStr += ".endm" + self.endLine
      return mStr

    ops = ['lt', 'eq', 'le', 'gt', 'lg', 'ge', 'o', 'u']
    dtypes = list([sg + ln
              for sg in ['i','u']
              for ln in ['16', '32', '64']])

    return self.endLine + \
           self.endLine.join([macro(op, dtype)
                              for op in ops
                              for dtype in dtypes])


  ##############################################################################
  # Function Signature
  # called after rest of code
  ##############################################################################
  def functionSignature(self, kernel ):
    kStr = ""

    # begin kernel descriptor
    if globalParameters["CodeObjectVersion"] == "V2":
      kStr += ".hsa_code_object_version %s,0%s" \
        % (globalParameters["CodeObjectVersion"][1], self.endLine)
      kStr += ".hsa_code_object_isa %u, %u, %u, \"AMD\", \"AMDGPU\" %s" \
        % (self.version[0], self.version[1], self.version[2], self.endLine)
    if globalParameters["CodeObjectVersion"] == "V3":
      kStr += ".amdgcn_target \"amdgcn-amd-amdhsa--gfx%s%s\"%s" \
        % ("".join(map(str,self.version)), \
        "+sram-ecc" if self.version == (9,0,8) else "",  self.endLine)

    kStr += ".text%s" % self.endLine
    kStr += ".protected %s%s" % (self.kernelName, self.endLine)
    kStr += ".globl %s%s" % (self.kernelName, self.endLine)
    kStr += ".p2align 8%s" % self.endLine
    kStr += ".type %s,@function%s" % (self.kernelName, self.endLine)

    if globalParameters["CodeObjectVersion"] == "V3":
        kStr += ".section .rodata,#alloc%s" % self.endLine
        kStr += ".p2align 6%s" % self.endLine
    tWord = "amdgpu_hsa_kernel" if globalParameters["CodeObjectVersion"] == "V2" else "amdhsa_kernel"
    kStr += ".%s %s%s" % (tWord, self.kernelName, self.endLine)
    if globalParameters["CodeObjectVersion"] == "V2":
        kStr += "%s:%s" % (self.kernelName, self.endLine)
        kStr += ".amd_kernel_code_t%s" % self.endLine
        kStr += "  is_ptr64 = 1%s" % self.endLine
    tWord = "enable_sgpr_kernarg_segment_ptr =" if globalParameters["CodeObjectVersion"] == "V2" \
        else ".amdhsa_user_sgpr_kernarg_segment_ptr"
    kStr += "  %s 1%s" % (tWord, self.endLine)

    # kern arg size
    kernArgReg = 0
    kernArgReg += 3*self.rpga
    kernArgReg += max(1,int(self.bpeAB/4)) # alpha
    if kernel["ProblemType"]["UseBeta"]:
      kernArgReg += max(1,int(self.bpeCexternal/4)) # beta
    kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
    kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
    kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsA"]) # strides
    kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsB"]) # strides
    if not kernel["ProblemType"]["UseInitialStridesAB"]:
      kernArgReg -= 2 # strides
    if not kernel["ProblemType"]["UseInitialStridesCD"]:
      kernArgReg -= 2 # strides
    kernArgReg += kernel["ProblemType"]["NumIndicesSummation"]
    kernArgReg += kernel["ProblemType"]["NumIndicesC"]
    if globalParameters["DebugKernel"]:
      kernArgReg += self.rpga # debug buffer
    kernArgBytes = kernArgReg * 4 # bytes/reg
    if globalParameters["CodeObjectVersion"] == "V2":
      kStr += "  kernarg_segment_byte_size = %u // bytes of kern args%s" \
        % (kernArgBytes, self.endLine)
    # kernArgReg = 0
    # kernArgReg += 3*self.rpga
    # kernArgReg += max(1,int(self.bpeAB/4)) # alpha
    # if kernel["ProblemType"]["UseBeta"]:
      # kernArgReg += max(1,int(self.bpeCexternal/4)) # beta
    # kernArgReg += 3 # offsets
    # kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
    # kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsA"]) # strides
    # kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsB"]) # strides
    # if not kernel["ProblemType"]["UseInitialStrides"]:
      # kernArgReg -= 3 # strides
    # kernArgReg += kernel["ProblemType"]["NumIndicesSummation"]
    # kernArgReg += kernel["ProblemType"]["NumIndicesC"]
    # if globalParameters["DebugKernel"]:
      # kernArgReg += self.rpga # debug buffer
    # kernArgBytes = kernArgReg * 4 # bytes/reg
    # kStr += "  kernarg_segment_byte_size = %u // bytes of kern args%s" \
        # % (kernArgBytes, self.endLine)

    # register allocation
    totalVgprs = self.vgprPool.size()
    totalSgprs = self.sgprPool.size()
    tWord = "workitem_vgpr_count =" if globalParameters["CodeObjectVersion"] == "V2" \
        else ".amdhsa_next_free_vgpr"
    kStr += "  %s %u // vgprs%s" \
        % (tWord, totalVgprs, self.endLine)
    tWord = "wavefront_sgpr_count =" if globalParameters["CodeObjectVersion"] == "V2" \
        else ".amdhsa_next_free_sgpr"
    kStr += "  %s %u // sgprs%s" \
        % (tWord, totalSgprs, self.endLine)

    if globalParameters["CodeObjectVersion"] == "V2":
      kStr += "  compute_pgm_rsrc1_vgprs = %u // floor((%u-1)/4)%s" \
        % ( (totalVgprs-1)//4, totalVgprs, self.endLine)
      kStr += "  compute_pgm_rsrc1_sgprs = %u // floor((%u-1)/8)%s" \
        % ( 1+(totalSgprs-1)//8, totalSgprs, self.endLine)

      # work-group dimensions
      kStr += "  compute_pgm_rsrc2_tidig_comp_cnt = 0 // 1D wg%s" % self.endLine

      # grid dimensions
      kStr += "  compute_pgm_rsrc2_tgid_x_en = 1 // wg.x%s" % self.endLine
      kStr += "  compute_pgm_rsrc2_tgid_y_en = 1 // wg.y%s" % self.endLine
      if kernel["ProblemType"]["NumIndicesC"] > 2:
        kStr += "  compute_pgm_rsrc2_tgid_z_en = %u // wg.z%s" % (1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0, self.endLine)
      #if abs(kernel["WorkGroupMapping"]) > 1:
      #  kStr += "  enable_sgpr_grid_workgroup_count_x = 1 // nwg0%s" % self.endLine
      #  kStr += "  enable_sgpr_grid_workgroup_count_y = 1 // nwg1%s" % self.endLine

      # lds size
      #kStr += "  compute_pgm_rsrc2_lds_size = 1 // ?%s" % self.endLine # don't use, it eats up 512 bytes of LDS
      #jgolds HACK
      # only want to enable this for cases we know it helps: 4x4 TT size and 16x16 WG size. Feel free to add more
      # cases after validating performance
    tWord = "workgroup_group_segment_byte_size =" if globalParameters["CodeObjectVersion"] == "V2" \
        else ".amdhsa_group_segment_fixed_size"
    if kernel["AggressivePerfMode"]>=2 and kernel["ProblemType"]["DataType"].isDouble() and \
      kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 4 and kernel["WorkGroup"] == [16,16,1]:
      group_segment_size = 32768 # Pad LDS to ensure we run exactly two waves
    else:
      group_segment_size = kernel["LdsNumElements"] * self.bpeAB
    kStr += "  %s %u // lds bytes%s" \
      % ( tWord, group_segment_size, self.endLine )

    if self.archCaps["HasWave32"]:
      if globalParameters["CodeObjectVersion"] == "V2":
        kStr += "  wavefront_size = 6 // 64-thread wavefronts%s" % self.endLine
      else:
        kStr += "  .amdhsa_wavefront_size32 0 // 64-thread wavefronts%s" % self.endLine

    if globalParameters["CodeObjectVersion"] == "V2":
      # other
      kStr += "  compute_pgm_rsrc2_user_sgpr = 2 // vcc%s" % self.endLine
      kStr += "  kernarg_segment_alignment = 4%s" % self.endLine
      kStr += "  group_segment_alignment = 4%s" % self.endLine
      kStr += "  private_segment_alignment = 4%s" % self.endLine
      kStr += ".end_amd_kernel_code_t%s" % self.endLine
    if globalParameters["CodeObjectVersion"] == "V3":
      kStr += "  .amdhsa_private_segment_fixed_size 0%s" % self.endLine
      kStr += "  .amdhsa_system_sgpr_workgroup_id_x 1%s" % self.endLine
      kStr += "  .amdhsa_system_sgpr_workgroup_id_y 1%s" % self.endLine
      kStr += "  .amdhsa_system_sgpr_workgroup_id_z %u%s" % (1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0, self.endLine)
      kStr += "  .amdhsa_system_vgpr_workitem_id 0%s" % self.endLine
      kStr += ".end_amdhsa_kernel%s" % self.endLine
      kStr += ".text%s" % self.endLine

    kStr += self.comment3("Optimizations and Config:")
    kStr += self.comment1("ThreadTile= %u x %u" % (kernel["ThreadTile0"], kernel["ThreadTile1"]))
    kStr += self.comment1("SubGroup= %u x %u" % (kernel["SubGroup0"], kernel["SubGroup1"]))
    kStr += self.comment1("VectorWidth=%u" % (kernel["VectorWidth"]))
    kStr += self.comment1("GlobalLoadVectorWidthA=%u, GlobalLoadVectorWidthB=%u" % (kernel["GlobalLoadVectorWidthA"], kernel["GlobalLoadVectorWidthB"]))
    kStr += self.comment1("DirectToLdsA=%s" % kernel["DirectToLdsA"])
    kStr += self.comment1("DirectToLdsB=%s" % kernel["DirectToLdsB"])
    kStr += self.comment1("UseSgprForGRO=%s" % kernel["_UseSgprForGRO"])

    if kernel["ProblemType"]["DataType"].isHalf():
      if kernel["ProblemType"]["HighPrecisionAccumulate"]:
        if globalParameters["CodeObjectVersion"] == "V2": srcValueType = "Struct"
        if globalParameters["CodeObjectVersion"] == "V3": srcValueType = "struct"
        if globalParameters["CodeObjectVersion"] == "V2": dstValueType = "Struct"
        if globalParameters["CodeObjectVersion"] == "V3": dstValueType = "struct"
        cptSize = "4"
        cptAlign = "4"
        cptByte  = 4
        cptValueType = "F32"
      else:
        if globalParameters["CodeObjectVersion"] == "V2": srcValueType = "F16"
        if globalParameters["CodeObjectVersion"] == "V3": srcValueType = "f16"
        if globalParameters["CodeObjectVersion"] == "V2": dstValueType = "F16"
        if globalParameters["CodeObjectVersion"] == "V3": dstValueType = "f16"
        cptSize = "2"
        cptAlign = "2"
        cptByte  = 2
      if globalParameters["CodeObjectVersion"] == "V2": cptValueType = "F16"
      if globalParameters["CodeObjectVersion"] == "V3": cptValueType = "f16"
    
    elif kernel["ProblemType"]["DataType"].isInt8x4():
      if globalParameters["CodeObjectVersion"] == "V2": srcValueType = "I8"
      if globalParameters["CodeObjectVersion"] == "V3": srcValueType = "i8"
      if globalParameters["CodeObjectVersion"] == "V2": dstValueType = "I32"
      if globalParameters["CodeObjectVersion"] == "V3": dstValueType = "i32"
      cptSize = "4"
      cptAlign = "4"
      cptByte  = 4
      if globalParameters["CodeObjectVersion"] == "V2": cptValueType = "I32"
      if globalParameters["CodeObjectVersion"] == "V3": cptValueType = "i32"
    
    elif kernel["ProblemType"]["DataType"].isSingle():
      if globalParameters["CodeObjectVersion"] == "V2": srcValueType = "F32"
      if globalParameters["CodeObjectVersion"] == "V3": srcValueType = "f32"
      if globalParameters["CodeObjectVersion"] == "V2": dstValueType = "F32"
      if globalParameters["CodeObjectVersion"] == "V3": dstValueType = "f32"
      cptSize = "4"
      cptAlign = "4"
      cptByte  = 4
      if globalParameters["CodeObjectVersion"] == "V2": cptValueType = "F32"
      if globalParameters["CodeObjectVersion"] == "V3": cptValueType = "f32"
    
    elif kernel["ProblemType"]["DataType"].isDouble() or \
         kernel["ProblemType"]["DataType"].isSingleComplex():
      if globalParameters["CodeObjectVersion"] == "V2": srcValueType = "F64"
      if globalParameters["CodeObjectVersion"] == "V3": srcValueType = "f64"
      if globalParameters["CodeObjectVersion"] == "V2": dstValueType = "F64"
      if globalParameters["CodeObjectVersion"] == "V3": dstValueType = "f64"
      cptSize = "8"
      cptAlign = "8"
      cptByte  = 8
      if globalParameters["CodeObjectVersion"] == "V2": cptValueType = "F64"
      if globalParameters["CodeObjectVersion"] == "V3": cptValueType = "f64"
    elif kernel["ProblemType"]["DataType"].isDoubleComplex():
      if globalParameters["CodeObjectVersion"] == "V2": srcValueType = "F64"
      if globalParameters["CodeObjectVersion"] == "V3": srcValueType = "f64"
      if globalParameters["CodeObjectVersion"] == "V2": dstValueType = "F64"
      if globalParameters["CodeObjectVersion"] == "V3": dstValueType = "f64"
      cptSize = "16"
      cptAlign = "16"
      cptByte  = 16
      if globalParameters["CodeObjectVersion"] == "V2": cptValueType = "Struct"
      if globalParameters["CodeObjectVersion"] == "V3": cptValueType = "struct"
    elif kernel["ProblemType"]["DataType"].isBFloat16():
      if globalParameters["CodeObjectVersion"] == "V2": srcValueType = "Struct"
      if globalParameters["CodeObjectVersion"] == "V3": srcValueType = "struct"
      if globalParameters["CodeObjectVersion"] == "V2": dstValueType = "Struct"
      if globalParameters["CodeObjectVersion"] == "V3": dstValueType = "struct"
      cptSize = "4"
      cptAlign = "4"
      cptByte  = 4
      if globalParameters["CodeObjectVersion"] == "V2": cptValueType = "F32"
      if globalParameters["CodeObjectVersion"] == "V3": cptValueType = "f32"

    if globalParameters["CodeObjectVersion"] == "V2":
      # Codeobject V2 metadata
      kStr += ".amd_amdgpu_hsa_metadata\n"
      kStr += "Version: [ 1, 0 ]\n"
      kStr += "Kernels:\n"
      kStr += "  - Name: %s%s" % (self.kernelName, self.endLine)
      kStr += "    SymbolName: '%s@kd'%s" % (self.kernelName, self.endLine)
      kStr += "    Language: OpenCL C\n"
      kStr += "    LanguageVersion: [ 2, 0 ]\n"
      kStr += "    Args:\n"
      ka_size = 0

      if globalParameters["DebugKernel"]:
        kStr += self.v2Argument(                    'AddressDbg',     '8',      '8', "GlobalBuffer",     "Struct", "Generic"); ka_size += 8

      kStr += self.v2Argument(                           'sizeC',     '8',      '8',      "ByValue",        "I64"); ka_size += 8
      kStr += self.v2Argument(                           'sizeA',     '8',      '8',      "ByValue",        "I64"); ka_size += 8
      kStr += self.v2Argument(                           'sizeB',     '8',      '8',      "ByValue",        "I64"); ka_size += 8

      kStr += self.v2Argument(                               'D',     '8',      '8', "GlobalBuffer", dstValueType, "Generic"); ka_size += 8
      kStr += self.v2Argument(                               'C',     '8',      '8', "GlobalBuffer", dstValueType, "Generic"); ka_size += 8
      kStr += self.v2Argument(                               'A',     '8',      '8', "GlobalBuffer", srcValueType, "Generic"); ka_size += 8
      kStr += self.v2Argument(                               'B',     '8',      '8', "GlobalBuffer", srcValueType, "Generic"); ka_size += 8

      if kernel["ProblemType"]["DataType"].isHalf() or \
         kernel["ProblemType"]["DataType"].isBFloat16() or \
         kernel["ProblemType"]["DataType"].isSingle() or \
         kernel["ProblemType"]["DataType"].isInt8x4():
        kStr += self.v2Argument(                         "alpha",     '4',      '4',      "ByValue", cptValueType); ka_size += 4
      elif kernel["ProblemType"]["DataType"].isDouble() or \
           kernel["ProblemType"]["DataType"].isSingleComplex() or \
           kernel["ProblemType"]["DataType"].isDoubleComplex():
        kStr += self.v2Argument(                         "alpha", cptSize, cptAlign,      "ByValue", cptValueType); ka_size += cptByte

      if kernel["ProblemType"]["UseBeta"]:
        if kernel["ProblemType"]["DataType"].isHalf() or \
           kernel["ProblemType"]["DataType"].isBFloat16() or \
           kernel["ProblemType"]["DataType"].isSingle() or \
           kernel["ProblemType"]["DataType"].isInt8x4():
          kStr += self.v2Argument(                        "beta",     '4',      '4',      "ByValue", cptValueType); ka_size += 4
        elif kernel["ProblemType"]["DataType"].isDouble() or \
             kernel["ProblemType"]["DataType"].isSingleComplex() or \
             kernel["ProblemType"]["DataType"].isDoubleComplex():
          kStr += self.v2Argument(                        "beta", cptSize, cptAlign,      "ByValue", cptValueType); ka_size += cptByte

      for i in range(0, self.numSgprStridesD):
        kStr += self.v2Argument(                   "strideD%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      for i in range(0, self.numSgprStridesC):
        kStr += self.v2Argument(                   "strideC%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      for i in range(0, self.numSgprStridesA):
        kStr += self.v2Argument(                   "strideA%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      for i in range(0, self.numSgprStridesB):
        kStr += self.v2Argument(                   "strideB%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4


      for i in range(0, self.numSgprSizesFree):
        kStr += self.v2Argument(                 "SizesFree%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      for i in range(0, self.numSgprSizesSum):
        kStr += self.v2Argument(                  "SizesSum%u"%i,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      for magicName in self.sumMagicParms:
        kStr += self.v2Argument(     "MagicNumberSize%s"%magicName,     '4',      '4',      "ByValue",        "U32"); ka_size += 4
        kStr += self.v2Argument(      "MagicShiftSize%s"%magicName,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      for idxChar in kernel["PackedC0IdxChars"][:-1]:
        kStr += self.v2Argument(     "MagicNumberSize%s"%idxChar,     '4',      '4',      "ByValue",        "U32"); ka_size += 4
        kStr += self.v2Argument(      "MagicShiftSize%s"%idxChar,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      for idxChar in kernel["PackedC1IdxChars"][:-1]:
        kStr += self.v2Argument(     "MagicNumberSize%s"%idxChar,     '4',      '4',      "ByValue",        "U32"); ka_size += 4
        kStr += self.v2Argument(      "MagicShiftSize%s"%idxChar,     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      kStr += self.v2Argument(                "OrigStaggerUIter",     '4',      '4',      "ByValue",        "I32"); ka_size += 4

      kStr += self.v2Argument(                  "NumWorkGroups0",     '4',      '4',      "ByValue",        "U32"); ka_size += 4
      kStr += self.v2Argument(                  "NumWorkGroups1",     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      kStr += self.v2Argument("MagicNumberProblemNumGroupTiles0",     '4',      '4',      "ByValue",        "U32"); ka_size += 4
      kStr += self.v2Argument(              "GridNumWorkGroups0",     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      kStr += self.v2Argument(                   "NumFullBlocks",     '4',      '4',      "ByValue",        "U32"); ka_size += 4
      kStr += self.v2Argument(                   "WgmRemainder1",     '4',      '4',      "ByValue",        "U32"); ka_size += 4
      kStr += self.v2Argument(        "MagicNumberWgmRemainder1",     '4',      '4',      "ByValue",        "U32"); ka_size += 4
      kStr += self.v2Argument(                         "padding",     '4',      '4',      "ByValue",        "U32"); ka_size += 4

      kStr += "    CodeProps:\n"
      kStr += "      KernargSegmentSize: %u%s" % (ka_size, self.endLine)
      kStr += "      GroupSegmentFixedSize: %u%s" % ( group_segment_size, self.endLine )
      kStr += "      PrivateSegmentFixedSize: %u%s" % ( 0, self.endLine )
      kStr += "      KernargSegmentAlign:  %u%s" % ( 8, self.endLine )
      kStr += "      WavefrontSize:        %u%s" % ( 64, self.endLine )
      kStr += "      NumSGPRs:             %u%s" % ( totalSgprs, self.endLine )
      kStr += "      NumVGPRs:             %u%s" % ( totalVgprs, self.endLine )
      kStr += "      MaxFlatWorkGroupSize: %u%s" % ( kernel["SubGroup0"] * kernel["SubGroup1"] * kernel["LocalSplitU"], self.endLine )
      kStr += ".end_amd_amdgpu_hsa_metadata\n"
    else:
      # Codeobject V3 metadata
      kStr += ".amdgpu_metadata\n"
      kStr += "---\n"
      kStr += "amdhsa.version:\n"
      kStr += "  - 1\n"
      kStr += "  - 0\n"
      kStr += "amdhsa.kernels:\n"
      kStr += "  - .name: %s%s" % (self.kernelName, self.endLine)
      kStr += "    .symbol: '%s.kd'%s" % (self.kernelName, self.endLine)
      kStr += "    .language:                   %s%s" % ("OpenCL C", self.endLine)
      kStr += "    .language_version:%s" % self.endLine
      kStr += "      - 2%s" % self.endLine
      kStr += "      - 0%s" % self.endLine
      kStr += "    .args:%s" % self.endLine
      offset = 0

      if globalParameters["DebugKernel"]:
        kStr += self.v3Argument(                    'AddressDbg',     '8', offset, "global_buffer","struct", "generic"); offset += 8

      kStr += self.v3Argument(                           'sizeC',     '8', offset,      "by_value",        "u64"); offset += 8
      kStr += self.v3Argument(                           'sizeA',     '8', offset,      "by_value",        "u64"); offset += 8
      kStr += self.v3Argument(                           'sizeB',     '8', offset,      "by_value",        "u64"); offset += 8

      kStr += self.v3Argument(                               'D',     '8', offset, "global_buffer", dstValueType, "generic"); offset += 8
      kStr += self.v3Argument(                               'C',     '8', offset, "global_buffer", dstValueType, "generic"); offset += 8
      kStr += self.v3Argument(                               'A',     '8', offset, "global_buffer", srcValueType, "generic"); offset += 8
      kStr += self.v3Argument(                               'B',     '8', offset, "global_buffer", srcValueType, "generic"); offset += 8

      if kernel["ProblemType"]["DataType"].isHalf() or \
         kernel["ProblemType"]["DataType"].isBFloat16() or \
         kernel["ProblemType"]["DataType"].isSingle() or \
         kernel["ProblemType"]["DataType"].isInt8x4():
        kStr += self.v3Argument(                         "alpha",       4, offset,      "by_value", cptValueType); offset += 4
      elif kernel["ProblemType"]["DataType"].isDouble() or \
           kernel["ProblemType"]["DataType"].isSingleComplex() or \
           kernel["ProblemType"]["DataType"].isDoubleComplex():
        kStr += self.v3Argument(                         "alpha", cptSize, offset,      "by_value", cptValueType); offset += cptByte

      if kernel["ProblemType"]["UseBeta"]:
        if kernel["ProblemType"]["DataType"].isHalf() or \
           kernel["ProblemType"]["DataType"].isBFloat16() or \
           kernel["ProblemType"]["DataType"].isSingle() or \
           kernel["ProblemType"]["DataType"].isInt8x4():
          kStr += self.v3Argument(                        "beta",       4, offset,      "by_value", cptValueType); offset += 4
        elif kernel["ProblemType"]["DataType"].isDouble() or \
             kernel["ProblemType"]["DataType"].isSingleComplex() or \
             kernel["ProblemType"]["DataType"].isDoubleComplex():
          kStr += self.v3Argument(                        "beta", cptSize, offset,      "by_value", cptValueType); offset += cptByte

      for i in range(0, self.numSgprStridesD):
        kStr += self.v3Argument(                   "strideD%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

      for i in range(0, self.numSgprStridesC):
        kStr += self.v3Argument(                   "strideC%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

      for i in range(0, self.numSgprStridesA):
        kStr += self.v3Argument(                   "strideA%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

      for i in range(0, self.numSgprStridesB):
        kStr += self.v3Argument(                   "strideB%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4


      for i in range(0, self.numSgprSizesFree):
        kStr += self.v3Argument(                 "SizesFree%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

      for i in range(0, self.numSgprSizesSum):
        kStr += self.v3Argument(                  "SizesSum%u"%i,     '4', offset,      "by_value",        "u32"); offset += 4

      for magicName in self.sumMagicParms:
        kStr += self.v3Argument(     "MagicNumberSize%s"%magicName,     '4', offset,      "by_value",        "u32"); offset += 4
        kStr += self.v3Argument(      "MagicShiftSize%s"%magicName,     '4', offset,      "by_value",        "u32"); offset += 4

      for idxChar in kernel["PackedC0IdxChars"][:-1]:
        kStr += self.v3Argument(     "MagicNumberSize%s"%idxChar,     '4', offset,      "by_value",        "u32"); offset += 4
        kStr += self.v3Argument(      "MagicShiftSize%s"%idxChar,     '4', offset,      "by_value",        "u32"); offset += 4

      for idxChar in kernel["PackedC1IdxChars"][:-1]:
        kStr += self.v3Argument(     "MagicNumberSize%s"%idxChar,     '4', offset,      "by_value",        "u32"); offset += 4
        kStr += self.v3Argument(      "MagicShiftSize%s"%idxChar,     '4', offset,      "by_value",        "u32"); offset += 4

      kStr += self.v3Argument(              "OrigStaggerUIter",       '4', offset,      "by_value",        "i32"); offset += 4

      kStr += self.v3Argument(                  "NumWorkGroups0",     '4', offset,      "by_value",        "u32"); offset += 4
      kStr += self.v3Argument(                  "NumWorkGroups1",     '4', offset,      "by_value",        "u32"); offset += 4

      kStr += self.v3Argument("MagicNumberProblemNumGroupTiles0",     '4', offset,      "by_value",        "u32"); offset += 4
      kStr += self.v3Argument(              "GridNumWorkGroups0",     '4', offset,      "by_value",        "u32"); offset += 4

      kStr += self.v3Argument(                   "NumFullBlocks",     '4', offset,      "by_value",        "u32"); offset += 4
      kStr += self.v3Argument(                   "WgmRemainder1",     '4', offset,      "by_value",        "u32"); offset += 4
      kStr += self.v3Argument(        "MagicNumberWgmRemainder1",     '4', offset,      "by_value",        "u32"); offset += 4

      kStr += self.v3Argument(                         "padding",     '4', offset,      "by_value",        "u32"); offset += 4
      kStr += "    .group_segment_fixed_size:   %u%s" % ( group_segment_size, self.endLine ) #XXXXXX
      kStr += "    .kernarg_segment_align:      %u%s" % ( 8, self.endLine )
      kStr += "    .kernarg_segment_size:       %u%s" % (((offset+7)//8)*8, self.endLine) # round up to .kernarg_segment_align
      kStr += "    .max_flat_workgroup_size:    %u%s" % ( kernel["SubGroup0"] * kernel["SubGroup1"] * kernel["LocalSplitU"], self.endLine )
      kStr += "    .private_segment_fixed_size: %u%s" % ( 0, self.endLine )
      kStr += "    .sgpr_count:                 %u%s" % ( totalSgprs, self.endLine )
      kStr += "    .sgpr_spill_count:           %u%s" % ( 0, self.endLine )
      kStr += "    .vgpr_count:                 %u%s" % ( totalVgprs, self.endLine )
      kStr += "    .vgpr_spill_count:           %u%s" % ( 0, self.endLine )
      kStr += "    .wavefront_size:             %u%s" % ( 64, self.endLine )

      kStr += "...\n"

      kStr += ".end_amdgpu_metadata\n"

    if globalParameters["CodeObjectVersion"] == "V3":
        kStr += "%s:%s" % (self.kernelName, self.endLine)
    kStr += self.comment3("Asm syntax workarounds")
    kStr += ".macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_add_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    # add w/o carry-out.  On older arch, vcc is still written
    kStr += "\n"
    kStr += ".macro _v_add_u32 dst:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitNC"]:
        kStr += r"   v_add_nc_u32 \dst, \src0 \src1 \dpp" + self.endLine
    elif self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_add_u32 \dst, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_add_u32 \dst, vcc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += "\n"
    kStr += ".macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_sub_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += "\n"
    # sub w/o carry-out.  On older arch, vcc is still written.
    kStr += ".macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_sub_u32 \dst, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_sub_u32 \dst, vcc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += "\n"
    kStr += ".macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=" + self.endLine
    if self.AsmBugs["ExplicitNC"]:
        kStr += r"   v_add_co_ci_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp" + self.endLine
    elif self.AsmBugs["ExplicitCO"]:
        kStr += r"   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp" + self.endLine
    else:
        kStr += r"   v_addc_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    # Use combined add+shift, where available:
    kStr += "\n"
    kStr += ".macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req" + self.endLine
    if globalParameters["AsmCaps"][self.version]["HasAddLshl"]:
      kStr += r"    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt" + self.endLine
    else:
      if self.AsmBugs["ExplicitCO"]:
        kStr += r"    v_add_co_u32 \dst, vcc, \src0, \src1" + self.endLine
      else:
        kStr += r"    v_add_u32 \dst, vcc, \src0, \src1" + self.endLine
      kStr += r"    v_lshlrev_b32 \dst, \shiftCnt, \dst" + self.endLine
    kStr += ".endm" + self.endLine


    # Use combined shift+add, where available:
    kStr += "\n"
    kStr += ".macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req" + self.endLine
    if globalParameters["AsmCaps"][self.version]["HasAddLshl"]:
      kStr += r"    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt" + self.endLine
    else:
      kStr += r"    v_lshlrev_b32 \dst, \shiftCnt, \dst" + self.endLine
      if self.AsmBugs["ExplicitCO"]:
        kStr += r"    v_add_co_u32 \dst, vcc, \src0, \src1" + self.endLine
      else:
        kStr += r"    v_add_u32 \dst, vcc, \src0, \src1" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += self.defineCMPXMacros(kernel)



    # Performs a division using 'magic number' computed on host
    # Argument requirements:
    #   - dstIdx must be two consecutive registers ; on exit the lower one will contain the quotient.  The upper is used as a temp.
    #   - First parm is passed as an integer vgpr index ; remaining are vgpr or sgpr symbolic names
    #   - dstIdx+1 cannot be same as dividend.  dividend+0 can be same as dividend and this may be useful for chaining divides.
    kStr += self.comment3("Magic div and mod functions")
    if kernel["MagicDivAlg"]==1: # TODO: remove me
        kStr += ".macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req" + self.endLine
        kStr += "    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber" + self.endLine
        kStr += "    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicNumber" + self.endLine
        kStr += "    v_lshrrev_b64 v[\dstIdx:\dstIdx+1], \magicShift, v[\dstIdx:\dstIdx+1]" + self.endLine
        kStr += ".endm" + self.endLine
    elif kernel["MagicDivAlg"]==2:
        kStr += ".macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req" + self.endLine
        kStr += "    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber" + self.endLine
        kStr += "    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA" + self.endLine
        kStr += "    v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]" + self.endLine
        kStr += "    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]" + self.endLine
        kStr += ".endm" + self.endLine

    ########################################
    # VGPR Macros
    ########################################
    kStr += self.comment3("VGPR Assignments")
    kStr += self.macroRegister("vgprValuC", self.startVgprValuC)

    kStr += self.comment1("ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx")
    ri = 0
    for bi in range(0,kernel["PrefetchLocalRead"]+1): # buffer indices
      for iui in range(0, kernel["InnerUnroll"]):
        kStr += self.macroRegister("vgprValuA_X%u_I%u"%(bi,iui), self.startVgprValuA+ri)
        ri += self.numVgprValuAPerBlock
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
        kStr += self.macroRegister("vgprG2LA", self.startVgprG2LA)

    ri = 0
    for bi in range(0,kernel["PrefetchLocalRead"]+1): # buffer indices
      for iui in range(0, kernel["InnerUnroll"]):
        kStr += self.macroRegister("vgprValuB_X%u_I%u"%(bi,iui), self.startVgprValuB+ri)
        ri += self.numVgprValuBPerBlock
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
        kStr += self.macroRegister("vgprG2LB", self.startVgprG2LB)
    if not kernel["LocalWriteUseSgprA"]:
      kStr += self.macroRegister("vgprLocalWriteAddrA", \
          self.startVgprLocalWriteAddressesA)
      if self.numVgprLocalWriteAddressesA > 1:
        kStr += self.macroRegister("vgprLocalWriteAddrOverhangA", \
            self.startVgprLocalWriteAddressesA+1)
    if not kernel["LocalWriteUseSgprB"]:
      kStr += self.macroRegister("vgprLocalWriteAddrB", \
          self.startVgprLocalWriteAddressesB)
      if self.numVgprLocalWriteAddressesB > 1:
        kStr += self.macroRegister("vgprLocalWriteAddrOverhangB", \
            self.startVgprLocalWriteAddressesB+1)
    if kernel["BufferLoad"]:
      kStr += self.macroRegister("vgprGlobalReadOffsetA", \
          self.startVgprGlobalReadOffsetA)
      kStr += self.macroRegister("vgprGlobalReadOffsetB", \
          self.startVgprGlobalReadOffsetB)
    else:
      kStr += self.macroRegister("vgprGlobalReadAddrA", \
          self.startVgprGlobalReadAddressesA)
      kStr += self.macroRegister("vgprGlobalReadAddrB", \
          self.startVgprGlobalReadAddressesB)

    for tc in ('A','B'):
      for zpr in self.zeroPadRegs[tc].values():
        kStr += self.macroRegister("vgpr" + zpr.regName, zpr.vgprIdx)
        self.zpr = ZeroPadReg.State.MacroDef
    if self.globalReadIncsUseVgpr:
      kStr += self.macroRegister("vgprGlobalReadIncsA", \
          self.startVgprGlobalReadIncsA)
      kStr += self.macroRegister("vgprGlobalReadIncsB", \
          self.startVgprGlobalReadIncsB)
    kStr += self.macroRegister("vgprLocalReadAddrA", \
        self.startVgprLocalReadAddressesA)
    kStr += self.macroRegister("vgprLocalReadAddrB", \
        self.startVgprLocalReadAddressesB)

    # Serial is always the last register in the pool so the store
    # code doesn't have to deal with fragmentation
    startSerial = self.vgprPool.size()-1
    kStr += self.macroRegister("vgprSerial", startSerial)

    if globalParameters["DebugKernel"]:
      kStr += self.macroRegister("vgprAddressDbg", \
          self.startVgprAddressDbg)
    #kStr += self.comment1("Occu: %u waves/simd" % self.numWavesPerSimd )
    kStr += self.comment1("Num VGPR=%u"%self.vgprPool.size())


    ########################################
    # SGPR Macros
    ########################################
    kStr += self.comment3("SGPR Assignments")


    # Emit declarations for all sgprs allocated with defineSgpr
    # in the order they were declared
    for skey in self.sgprs:
      kStr += self.macroRegister("sgpr"+skey, self.sgprs[skey])
    kStr += self.comment1("max SGPR=%u"%self.sgprPool.size())

    kStr += "\n"
    kStr += self.comment1("Size Assignments")
    problemType = kernel["ProblemType"]
    for idx in range(max(problemType["IndexAssignmentsA"] + problemType["IndexAssignmentsB"])+1):
      idxChar= globalParameters["IndexChars"][idx]
      if idx in problemType["IndicesFree"] or idx in problemType["IndicesBatch"]:
        idxType="Free"
      elif idx in problemType["IndicesSummation"]:
        idxType="Sum"
        idx = idx - problemType["NumIndicesC"]
      else:
        raise ValueError("unexpected index type in size assignments")

      kStr += self.macroRegister("sgprSize%s"%(idxChar), \
                  "sgprSizes%s+%u"%(idxType, idx))


    kStr += "\n"
    kStr += self.comment1("Stride Assignments")
    for tc in ('D','C'):
      if tc=='D' and not kernel["LdcEqualsLdd"]:
        sourceTc = 'D'
      else:
        sourceTc = 'C'
      for idx in range(0, problemType["NumIndicesC"]):
        i = idx
        idxChar= self.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesCD"]:
          kStr += self.macroRegister("constStride%s%s"%(tc,idxChar), 1)
        else:
          if not kernel["ProblemType"]["UseInitialStridesCD"]:
            i = i-1
          kStr += self.macroRegister("sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s+%u"%(sourceTc, i))

    for tc in ('A','B'):
      for i, idx in enumerate(problemType["IndexAssignments%s"%tc]):
        idxChar= self.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesAB"]:
          kStr += self.macroRegister("constStride%s%s"%(tc,idxChar), 1)
        else:
          if not kernel["ProblemType"]["UseInitialStridesAB"]:
            i = i-1
          kStr += self.macroRegister("sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s+%u"%(tc, i))

    kStr += "\n"
    kStr += self.macroRegister("MT0", kernel["MacroTile0"])
    kStr += self.macroRegister("MT1", kernel["MacroTile1"])
    kStr += self.macroRegister("DepthU", kernel["DepthU"])
    kStr += self.macroRegister("GSU", kernel["GlobalSplitU"])
    kStr += self.macroRegister("BpeA", self.tPA["bpe"])
    kStr += self.macroRegister("BpeALog2", log2(self.tPA["bpe"]))
    kStr += self.macroRegister("BpeB", self.tPB["bpe"])
    kStr += self.macroRegister("BpeBLog2", log2(self.tPB["bpe"]))
    kStr += self.comment1("Number of elements to shift-left SRD")
    kStr += self.macroRegister("SrdShiftLeftA", self.srdShiftLeft['A'])
    kStr += self.macroRegister("SrdShiftLeftB", self.srdShiftLeft['B'])

    if kernel["BufferLoad"] or kernel["BufferStore"]:
      kStr += self.comment1("2GB limit - set offsets to -1 to exceed this and clamp")
      kStr += self.macroRegister("BufferLimit", "0x80000000")
      #TODO-64 : This is max 32-bit negative value, the tail loop
      # does incrementally step through the GRO and increment GRO
      # which are initialized with this value
      kStr += self.macroRegister("BufferOOB", "0x80000000")

      srdUpperValue = Code.SrdUpperValue(self.version)
      kStr += self.comment3("Bits 127:96 of SRD.\n" + srdUpperValue.desc())
      kStr += self.macroRegister("Srd127_96", str(srdUpperValue))

    ########################################
    # Global Offsets
    ########################################
    # justOffset32 means we should only write the 32-bit offset
    # This is used in Buffer addressing modes.
    # Flat addressing modes expect the GLOBAL_OFFSET to initialize a full 64-bit address
    for (tc, indices, justOffset32, tP) in [ \
        ("C", list(range(0, kernel["ProblemType"]["NumIndicesC"])), kernel["BufferStore"], None), \
        ("A", kernel["ProblemType"]["IndexAssignmentsA"], kernel["BufferLoad"], self.tPA), \
        ("B", kernel["ProblemType"]["IndexAssignmentsB"], kernel["BufferLoad"], self.tPB) ]:

      # BufferStore does not use this macro so don't generate it:
      if tc == "C" and kernel["BufferStore"]:
        continue

      kStr += self.comment("Global Offset %s"%tc)
      numDim = len(indices)
      idxChars = []
      for i in indices:
        idxChars.append(self.indexChars[i])

      packBatchDims = tP["PackBatchDims"] if tP != None else 0x3

      # macro declaration
      kStr += ".macro GLOBAL_OFFSET_%s vgprAddr:req"%tc
      calcDims = [] # dimensions which are participating in the address calc (ignores other summation)
      for i in range(0, numDim):
        if tc == 'C':
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesCD"]
          idxChar = self.indexChars[i]
        else:
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesAB"]
          idxChar = self.indexChars[tP['ia'][i]]

        # tile index or unroll vgpr or summation
        # other summation (other than unroll) are included in the GLOBAL_OFFSET macro but not used in address calc
        if     tc in ('A','C') and indices[i] == kernel["ProblemType"]["Index0"] \
            or tc in ('B','C') and indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          kStr += " vgprOffset%s:req" % idxChars[i]
          calcDims.append(i)
        elif indices[i] in kernel["ProblemType"]["IndicesSummation"]:
          # other summation index (not unroll)
          continue
        else:
          # other batch or free index
          if isPackedIndex(kernel, indices[i], packBatchDims):
            calcDims.append(i)
            kStr += " vgprOffset%s:req" % idxChars[i]
          elif not justOffset32: # buffer/justOffset32 scalars are included in SRD not the offset, so skip here
            calcDims.append(i)
            kStr += " sgprOffset%s:req" % idxChars[i]
      kStr += " vgprTmp:req" + self.endLine

      # Each index may be skipped, scaled by stride, or unscaled
      # If destLo is unset, no accumulation is necessary.

      # if the first index (i==0) is unscaled (UseInitialStrides),
      # it can be combined at the next update or moved at end
      # (if there is no next update)

      pendingOffset = None # offset pending for accumulation
      offsetIsVgpr = False # True if the source is VGPR ; False if SGPR
      destLo = None

      # true for first addr calc. In this case, we can directly write addr
      # rather than accumulating through a tmp
      writeDirectToAddr = 1
      for i in calcDims:
        # should have eliminated these above
        idx = indices[i]
        assert not (idx in kernel["ProblemType"]["IndicesSummation"] and idx != kernel["ProblemType"]["IndexUnroll"])

        if indices[i] == kernel["ProblemType"]["Index0"] \
            or indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          offsetIsVgpr = True
        # other c index sgpr (free or batch)
        elif indices[i] < kernel["ProblemType"]["NumIndicesC"]:
          if isPackedIndex(kernel, indices[i], packBatchDims):
            offsetIsVgpr = True
          else:
            offsetIsVgpr = False
        else:
          assert(0) # no other type allowed

        # offset is VGPR or SGPR string to use for the offset
        if offsetIsVgpr:
          offset = "v[\\vgprOffset%s]" % idxChars[i]
        else:
          offset = "s[\\sgprOffset%s]" % idxChars[i]

        #kStr += self.comment1("dim%s pendingOffset=%s offset=%s offsetIsVgpr=%s" \
        #    % (self.indexChars[indices[i]], pendingOffset, offset, offsetIsVgpr))

        needAdd = 0
        # should be indices[i]??
        if i==0 and not useInitialStrides:
          # slide into next address calc - can do addr = pendingOffset + nextAddrCalc
          pendingOffset = offset
          writeDirectToAddr = 0
        else:
          # tile index or unroll vgpr
          if offsetIsVgpr:
            if writeDirectToAddr:
              destLo = "v[\\vgprAddr+0]"
              destHi = "v[\\vgprAddr+1]"
              needAdd = 0 # don't need add since writing address directly.
              writeDirectToAddr = 0
            else:
              destLo = "v[\\vgprTmp+0]"
              destHi = "v[\\vgprTmp+1]"
              needAdd = 1

            # offset * stride
            kStr += inst("v_mul_lo_u32", \
                destLo,
                self.strideRef(tc, indices[i]), \
                offset, \
                "mul d%u lower"%i)
            if not justOffset32:
              kStr += inst("v_mul_hi_u32", \
                  destHi,
                  self.strideRef(tc, indices[i]), \
                  offset, \
                  "mul d%u upper"%i)
          else: # offset is SGPR:
            if not justOffset32:
              # buffer mode (aka justOffset32) does scalars into SRD not offset
              kStr += inst("v_mov_b32", \
                  "v[\\vgprTmp+2]", \
                  "s[\\sgprOffset%s]"%idxChars[i], \
                  "sgprOffset -> vgprTmp+2")
              # offset * stride
              kStr += inst("v_mul_lo_u32", \
                  "v[\\vgprTmp+0]", \
                  self.strideRef(tc, indices[i]), \
                  "v[\\vgprTmp+2]",  \
                  "other stride mul d%u lower"%i)
              kStr += inst("v_mul_hi_u32", \
                  "v[\\vgprTmp+1]", \
                  self.strideRef(tc, indices[i]), \
                  "v[\\vgprTmp+2]",  \
                  "mul d%u upper"%i)
              needAdd = 1

        if needAdd:
          writeDirectToAddr = 0 # safety net, once we write address can't directly overwrite it later
          destLo = "v[\\vgprAddr+0]"
          destHi = "v[\\vgprAddr+1]"
          # addr += offset * stride (lo) : accumulate just-computed address term into addr

          srcLo = pendingOffset if pendingOffset else destLo
          srcHi = 0 if pendingOffset else destHi
          kStr += inst("_v_add_co_u32", \
            destLo, \
            "vcc", \
            srcLo, \
            "v[\\vgprTmp+0]", \
            "accumulate %s lower"%idxChar)

          # addr += offset * stride (hi)
          if not justOffset32:
            kStr += inst("_v_addc_co_u32", \
                "v[\\vgprAddr+1]", \
                "vcc", \
                "v[\\vgprTmp+1]",  \
                srcHi, \
                "vcc", \
                "accumulate %s upper"%idxChar)
          pendingOffset = None

      # pendingOffset but never got a chance to apply it,
      # need to just add an explicit move or add:
      # this can happen for small-order tensors
      if pendingOffset != None:
        destLo = "v[\\vgprAddr+0]"
        if writeDirectToAddr:
          kStr += inst("v_mov_b32", destLo, offset, "setup d0 lower")
          if not justOffset32:
            kStr += inst("v_mov_b32", "v[\\vgprAddr+1]", hex(0), "d0 upper")
        else:
          kStr += inst("_v_add_co_u32", \
            destLo, \
            "vcc", \
            destLo, \
            pendingOffset, \
            "accumulate final pendingOffset")


      if tP != None and kernel["BufferLoad"] and self.srdShiftLeft[tc]:
        kStr += inst("_v_add_u32", \
            "v[\\vgprAddr+0]", \
            hex(self.srdShiftLeft[tc]), \
            "v[\\vgprAddr+0]", \
            "add prepad for pointer shift")

      # addr *= bytes/element
      if justOffset32:
        kStr += inst("v_lshlrev_b32", \
            "v[\\vgprAddr+0]", \
            hex(log2(self.bpeAB)), \
            "v[\\vgprAddr+0]", \
            "offset *= bytes/element")
      else:
        kStr += inst("v_lshlrev_b64", \
            "v[\\vgprAddr+0:\\vgprAddr+1]", \
            hex(log2(self.bpeAB)), \
            "v[\\vgprAddr+0:\\vgprAddr+1]", \
            "offset *= bytes/element")
      #kStr += "s_endpgm\n"
      kStr += ".endm%s" % self.endLine

    ########################################
    # Dynamic Scalar Divide
    kStr += self.comment3("Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor;")
    kStr += ".macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp%s" % self.endLine
    kStr += inst("v_cvt_f32_u32", "v[\\vQuotient]",  "v[\\vDivisor]",  "" )
    kStr += inst("v_rcp_f32",     "v[\\vQuotient]",  "v[\\vQuotient]", "" )
    kStr += inst("v_mul_f32",     "v[\\vQuotient]",  "0x4f800000",     "v[\\vQuotient]", "" )
    kStr += inst("v_cvt_u32_f32", "v[\\vQuotient]",  "v[\\vQuotient]", "" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vDivisor]", "v[\\vQuotient]", "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vTmp0]",      "v[\\vDivisor]", "v[\\vQuotient]", "" )
    kStr += inst("_v_sub_co_u32",     "v[\\vTmp1]",      "vcc", hex(0),    "v[\\vRemainder]", "" )
    kStr += inst("v_cmp_ne_i32",  "s[\\sTmp:\\sTmp+1]", hex(0),        "v[\\vTmp0]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vRemainder]", "v[\\vTmp1]",     "v[\\vRemainder]", "s[\\sTmp:\\sTmp+1]", "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vRemainder]", "v[\\vRemainder]", "v[\\vQuotient]", "" )
    kStr += inst("_v_sub_co_u32",     "v[\\vTmp0]",      "vcc",            "v[\\vQuotient]", "v[\\vRemainder]", "" )
    kStr += inst("_v_add_co_u32",     "v[\\vQuotient]",  "vcc",            "v[\\vQuotient]", "v[\\vRemainder]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vTmp0]", "s[\\sTmp:\\sTmp+1]", "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vDividend]", "" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
    kStr += inst("_v_sub_co_u32",     "v[\\vTmp0]",      "vcc",            "v[\\vDividend]", "v[\\vRemainder]", "" )
    kStr += inst("v_cmp_ge_u32",  "s[\\sTmp:\\sTmp+1]", "v[\\vDividend]", "v[\\vRemainder]", "" )
    kStr += inst("_v_add_co_u32",     "v[\\vRemainder]", "vcc",            hex(1), "v[\\vQuotient]", "" )
    kStr += inst("_v_add_co_u32",     "v[\\vTmp1]",      "vcc", -1,        "v[\\vQuotient]", "" )
    kStr += inst("v_cmp_le_u32",  "vcc",             "v[\\vDivisor]", "v[\\vTmp0]", "" )
    kStr += inst("s_and_b64",     "vcc",             "s[\\sTmp:\\sTmp+1]", "vcc", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vRemainder]", "vcc", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vTmp1]",     "v[\\vQuotient]", "s[\\sTmp:\\sTmp+1]", "" )
    kStr += inst("v_cmp_ne_i32",  "vcc", hex(0),     "v[\\vDivisor]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  -1, "v[\\vQuotient]", "vcc", "final result" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
    kStr += inst("_v_sub_co_u32",     "v[\\vRemainder]", "vcc",            "v[\\vDividend]", "v[\\vRemainder]", "final result" )
    kStr += ".endm%s" % self.endLine

    if not kernel["EnableMatrixInstruction"]:
      kStr += self.defineMACMacro(kernel, kernel["InnerUnroll"], True)
      if kernel["InnerUnroll"] > 1:
        kStr += self.defineMACMacro(kernel, 1, True) # define OneIter case

    if self.overflowedResources:
      print("")
      if self.overflowedResources == 1:
        msg = "too many vgprs"
      elif self.overflowedResources == 2:
        msg = "too many sgprs"
      elif self.overflowedResources == 3:
        msg = "half store requires at lesat two elements per batch"
      elif self.overflowedResources == 4:
        msg = "Occupancy limit"
      else:
        msg = "unknown"

      printWarning("%s overflowed resources.  errorCode=%d, msg=\"%s\", vgprs=%u, sgprs=%u" \
          % (self.kernelName, self.overflowedResources, msg, \
          self.vgprPool.size(), self.sgprPool.size()))
      kStr += "s_endpgm // overflowed resources\n"
      kStr += ".if 0\n"


    return kStr


  ##############################################################################
  # Function Beginning
  ##############################################################################
  def functionSignaturePrefix(self, kernel): return ""
  def functionSignatureSuffix(self, kernel): return ""
  def functionBegin(self, kernel): return ""

  ##############################################################################
  # getKernArg
  # Write an argument to specified SGPR and move the kernArgOffset
  # if writeSgpr==0, just move the kernArgOffset - this is used to skip
  # unused parms
  ##############################################################################
  def getKernArg(self, parmName, writeSgpr=1):
    kStr = ""
    size = 1*4
    if writeSgpr:
      kStr += inst("s_load_dword", sgpr(parmName), \
          sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
    self.kernArgOffset += size
    return kStr

  ##############################################################################
  ##############################################################################
  def allocateResources(self, kernel):
    kStr = ""
    if self.do["NullKernel"]:
      kStr += inst("s_endpgm", "Skip the whole kernel")

    if self.do["PreLoop"]: 
      if self.db["InitSgpr"] & 0x1:
        kStr += self.comment("Init SGPRs")
        for i in range(self.firstInitSgpr, self.sgprPool.size()):
          kStr += inst("s_mov_b32", sgpr(i), hex(self.initSgprValue), "InitSgpr&0x1")
        kStr += "\n"

      if self.db["InitVgpr"] & 0x1:
        kStr += self.comment("Init VGPRs")
        for i in range(1, self.totalVgprs):
          kStr += inst("v_mov_b32", vgpr(i), hex(self.initVgprValue), "InitVgpr&0x1")
        kStr += "\n"

      # set m0
      kStr += inst("s_mov_b32", "m0", hex(kernel["LdsNumElements"] \
          * self.bpeAB), "LDS clamp at %u bytes" \
          %(kernel["LdsNumElements"] * self.bpeAB) )

      kStr += inst("v_mov_b32", vgpr("Serial"), vgpr(0), "thread serial id")

      ########################################
      # load kernel args
      kStr += self.comment("Load Kernel Args")
      self.kernArgOffset = 0
      if globalParameters["DebugKernel"]:
        kStr += self.getKernArg("AddressDbg")
        kStr += self.getKernArg("AddressDbg+1")

      kStr += self.getKernArg("Tensor2dSizeC+0",0)
      kStr += self.getKernArg("Tensor2dSizeC+1",0)
      
      load = self.numSgprToLoad
      sgprStart = self.sgprs["Tensor2dSizeA"]
      while load > 0:
        if load >= 16:
          load -= 16
          kStr += inst("s_load_dwordx16", sgpr(sgprStart,16), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 16
          self.kernArgOffset += 16 * 4
          continue
        if load >= 8:
          load -= 8
          kStr += inst("s_load_dwordx8", sgpr(sgprStart,8), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 8
          self.kernArgOffset += 8 * 4
          continue
        if load >= 4:
          load -= 4
          kStr += inst("s_load_dwordx4", sgpr(sgprStart,4), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 4
          self.kernArgOffset += 4 * 4
          continue
        if load >= 2:
          load -= 2
          kStr += inst("s_load_dwordx2", sgpr(sgprStart,2), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 2
          self.kernArgOffset += 2 * 4
          continue
        if load >= 1:
          load -= 1
          kStr += inst("s_load_dword", sgpr(sgprStart), sgpr("KernArgAddress",2), hex(self.kernArgOffset), "")
          sgprStart += 1
          self.kernArgOffset += 1 * 4
          continue
      # currently align sgpr to kernel argument memory, and use s_load_dwordxN to load argument as large as possible in one instruction
      # however, in order to match sgpr to kernel argument memory, some unnecessarily sgpr will also be defined, and caused wasting of sgpr.
      # TODO: more efficient way is to organize both sgpr and kernel argument memory in API
      
      # comment out original getKernArg, in case we need it back 
      """
      kStr += self.getKernArg("Tensor2dSizeA+0")
      kStr += self.getKernArg("Tensor2dSizeA+1")
      kStr += self.getKernArg("Tensor2dSizeB+0")
      kStr += self.getKernArg("Tensor2dSizeB+1")

      kStr += self.getKernArg("AddressD")
      kStr += self.getKernArg("AddressD+1")
      kStr += self.getKernArg("AddressC")
      kStr += self.getKernArg("AddressC+1")
      kStr += self.getKernArg("AddressA")
      kStr += self.getKernArg("AddressA+1")
      kStr += self.getKernArg("AddressB")
      kStr += self.getKernArg("AddressB+1")

      # for half precision or smaller, data is padded to fill up 32-bits
      if kernel["ProblemType"]["DataType"].isHalf() or \
         kernel["ProblemType"]["DataType"].isBFloat16() or \
         kernel["ProblemType"]["DataType"].isSingle() or \
         kernel["ProblemType"]["DataType"].isInt8x4():
        kStr += self.getKernArg("Alpha")
      elif kernel["ProblemType"]["DataType"].isDouble() or \
           kernel["ProblemType"]["DataType"].isSingleComplex():
        kStr += self.getKernArg("Alpha+0")
        kStr += self.getKernArg("Alpha+1")
      elif kernel["ProblemType"]["DataType"].isDoubleComplex():
        kStr += self.getKernArg("Alpha+0")
        kStr += self.getKernArg("Alpha+1")
        kStr += self.getKernArg("Alpha+2")
        kStr += self.getKernArg("Alpha+3")

      if kernel["ProblemType"]["UseBeta"]:
        if kernel["ProblemType"]["DataType"].isHalf() or \
           kernel["ProblemType"]["DataType"].isBFloat16() or \
           kernel["ProblemType"]["DataType"].isSingle() or \
           kernel["ProblemType"]["DataType"].isInt8x4():
          kStr += self.getKernArg("Beta")
        elif kernel["ProblemType"]["DataType"].isDouble() or \
             kernel["ProblemType"]["DataType"].isSingleComplex():
          kStr += self.getKernArg("Beta+0")
          kStr += self.getKernArg("Beta+1")
        elif kernel["ProblemType"]["DataType"].isDoubleComplex():
          kStr += self.getKernArg("Beta+0")
          kStr += self.getKernArg("Beta+1")
          kStr += self.getKernArg("Beta+2")
          kStr += self.getKernArg("Beta+3")
      for i in range(0, self.numSgprStridesD):
        kStr += self.getKernArg("StridesD+%u"%i, not kernel["LdcEqualsLdd"])
      for i in range(0, self.numSgprStridesC):
        kStr += self.getKernArg("StridesC+%u"%i)
      for i in range(0, self.numSgprStridesA):
        kStr += self.getKernArg("StridesA+%u"%i)
      for i in range(0, self.numSgprStridesB):
        kStr += self.getKernArg("StridesB+%u"%i)
      for i in range(0, self.numSgprSizesFree):
        kStr += self.getKernArg("SizesFree+%u"%i)
      for i in range(0, self.numSgprSizesSum):
        kStr += self.getKernArg("SizesSum+%u"%i)
      for magicName in self.sumMagicParms:
        kStr += self.getKernArg("MagicNumberSize%s"%magicName)
        kStr += self.getKernArg("MagicShiftSize%s"%magicName)

      for idxChar in kernel["PackedC0IdxChars"][:-1]:
        kStr += self.getKernArg("MagicNumberSize%s"%idxChar)
        kStr += self.getKernArg("MagicShiftSize%s"%idxChar)
      for idxChar in kernel["PackedC1IdxChars"][:-1]:
        kStr += self.getKernArg("MagicNumberSize%s"%idxChar)
        kStr += self.getKernArg("MagicShiftSize%s"%idxChar)

      for idx in kernel["ProblemType"]["IndicesSummation"]:
        for tc in ('A','B'):
          for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
            (freeDim, sumDim, padStart, padEnd) = zp
            if sumDim == idx:
              freeDimChar = globalParameters["IndexChars"][freeDim]
              sumDimChar  = globalParameters["IndexChars"][sumDim]
              kStr += self.getKernArg("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar))
              kStr += self.getKernArg("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar))

      kStr += self.getKernArg("OrigStaggerUIter", self.staggerU)

      kStr += self.getKernArg("NumWorkGroups0")
      kStr += self.getKernArg("NumWorkGroups1")
      kStr += self.getKernArg("MagicNumberProblemNumGroupTiles0", kernel["PersistentKernel"])
      kStr += self.getKernArg("GridNumWorkGroups0", kernel["PersistentKernel"])
      kStr += self.getKernArg("NumFullBlocks")
      kStr += self.getKernArg("WgmRemainder1")
      kStr += self.getKernArg("MagicNumberWgmRemainder1")
      """

      kStr += inst("s_waitcnt", "lgkmcnt(0)", \
          "wait for %u bytes of kern args" % self.kernArgOffset )
    else:
      kStr += ".if 0\n"

    for tc in ('A', 'B'):
      for zp in kernel["ProblemType"]["ZeroPad%s"%tc]:
        (freeDim, sumDim) = zp[:2]
        freeDimChar = globalParameters["IndexChars"][freeDim]
        sumDimChar  = globalParameters["IndexChars"][sumDim]
        kStr += inst("s_lshl_b32", \
                     sgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                     sgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                     "Bpe%sLog2"%tc, "")
        kStr += inst("s_lshl_b32", \
                     sgpr("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                     sgpr("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                     "Bpe%sLog2"%tc, "")

    if kernel["PersistentKernel"]:
      kStr += inst("s_mov_b32", sgpr("SerialWorkGroupIter"), sgpr("WorkGroup0"), "init SerialWorkGroupIter")
    if self.prefetchAcrossPersistent0 and kernel["ExpandPointerSwap"]:
      kStr += inst("s_mov_b32", sgpr("EvenIterStart"), 0, "init SerialWorkGroupIter")

    if kernel["MagicDivAlg"]==2:
      for magicName in self.sumMagicParms:
          kStr += inst("s_lshr_b32", sgpr("MagicAbitSize%s"%magicName), sgpr("MagicShiftSize%s"%magicName), 31,"extract abit")
          kStr += inst("s_and_b32",  sgpr("MagicShiftSize%s"%magicName), sgpr("MagicShiftSize%s"%magicName), hex(0x7fffffff), "remove abit")

      for idxChar in sorted(set(kernel["PackedC0IdxChars"][:-1] + kernel["PackedC1IdxChars"][:-1])):
          kStr += inst("s_lshr_b32", sgpr("MagicAbitSize%s"%idxChar), sgpr("MagicShiftSize%s"%idxChar), 31,"extract abit")
          kStr += inst("s_and_b32",  sgpr("MagicShiftSize%s"%idxChar), sgpr("MagicShiftSize%s"%idxChar), hex(0x7fffffff), "remove abit")

    ########################################
    # Debug Buffer
    if globalParameters["DebugKernel"]:
      kStr += self.comment("Debug Buffer")

      # nwg0 FIXME use NumWorkGroups0
      #kStr += self.assert_eq(vgpr(nwg0), sgpr("NumWorkGroups0")) # "bozo, remove me")
      nwg0 = self.vgprPool.checkOut(1)
      tmpVgpr = self.vgprPool.checkOut(2)
      tmpSgpr = self.getTmpSgpr(1).idx()
      kStr += "// nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile0"]-1), "MT0-1")
      kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr(tmpSgpr), "MT0-1")
      kStr += inst("_v_add_co_u32", vgpr(nwg0), "vcc", sgpr("SizesFree+0"), \
          vgpr(tmpVgpr), "%s = size0+MT0-1"%vgpr(nwg0))
      kStr += vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpVgpr, tmpSgpr)
      self.vgprPool.checkIn(tmpVgpr)
      self.nipt = 16 # num integers per thread
      v = self.vgprPool.checkOut(3)
      kStr += inst("v_mov_b32", vgpr(v), sgpr("WorkGroup0"), "%s=wg0"%vgpr(v) )
      kStr += inst("v_mov_b32", vgpr(v+1), sgpr("WorkGroup1"), "%s=wg1"%vgpr(v+1) )
      kStr += inst("v_mul_lo_u32", vgpr(v+1), vgpr(v+1), vgpr(nwg0), \
          "%s=wg1*nwg0"%vgpr(v+1) )
      kStr += inst("_v_add_co_u32", vgpr(v), "vcc", vgpr(v), vgpr(v+1), \
          "%s=wg1*nwg0+wg0"%vgpr(v) )
      kStr += staticMultiply(vgpr(v), vgpr(v), kernel["NumThreads"], sgpr(tmpSgpr))
      kStr += inst("_v_add_co_u32", vgpr(v), "vcc", vgpr(v), vgpr("Serial"), \
          "%s=tid+NT*(wg1*nwg0+wg0)=serial"%vgpr(v) )
      kStr += inst("v_mul_lo_u32", vgpr(v), hex(self.nipt*4), vgpr(v), \
          "%s=serial*nipt*4"%vgpr(v) )
      kStr += inst("v_mov_b32", vgpr(v+1), 0, "")
      kStr += inst("_v_add_co_u32", vgpr("AddressDbg"), "vcc", sgpr("AddressDbg"), \
          vgpr(v), "%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") )
      kStr += inst("v_mov_b32", vgpr(v+2), sgpr("AddressDbg+1"), "%s=AddressD1"%vgpr(v+2) )
      kStr += inst("_v_addc_co_u32", vgpr("AddressDbg+1"), "vcc", vgpr(v+2), \
          vgpr(v+1), "vcc", "%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") )
      kStr += inst("s_mov_b32", sgpr("DebugKernelItems"), 0, "")
      self.vgprPool.checkIn(v)
      self.vgprPool.checkIn(nwg0)


    if self.db["InitLds"]:
      kStr += self.initLds(kernel, self.initLdsValue)

    if kernel["CheckTensorDimAsserts"]:
      kStr += self.assert_multiple_b32(sgpr("SizesSum+%u"%(self.numSgprSizesSum-1)),
                kernel["AssertSummationElementMultiple"], 0x1001)
      kStr += self.assert_multiple_b32(sgpr("SizesFree+0"),
                kernel["AssertFree0ElementMultiple"], 0x1002)
      kStr += self.assert_multiple_b32(sgpr("SizesFree+1"),
                kernel["AssertFree1ElementMultiple"], 0x1003)

    return kStr


  ##############################################################################
  # Perform a magic division (mul by magic number and shift)
  # dest is two consec SGPR, used for intermediate temp as well as final result
  # result quotient returned in sgpr(dest,1)
  ##############################################################################
  def sMagicDiv(self, kernel, dest, dividend, magicNumber, magicShift):
    kStr = ""
    kStr += self.s_mul_u64_u32(sgpr(dest), sgpr(dest+1), dividend, magicNumber, "s_magic mul")
    kStr += inst("s_lshr_b64", sgpr(dest,2), sgpr(dest,2), magicShift, "sMagicDiv")
    return kStr

  ##############################################################################
  # Open Persistent Loop
  # init iteration counter, define loop target
  ##############################################################################
  def openPersistentLoop(self, kernel):
    kStr = ""
    if kernel["PersistentKernel"]:
      kStr += self.comment3("Persistent Loop Start")
      kStr += self.getLabelDef("PersistentLoopStart")
      #kStr += str(Code.WaitCnt(self.version, 0,0,"wait for outstanding stores"))

    return kStr

  ##############################################################################
  # Global Read Addresses: WorkGroup
  ##############################################################################
  def graWorkGroup(self, kernel, isPap):
    kStr = ""

    if kernel["PersistentKernel"]:
      stmp = self.getTmpSgpr(2).idx()
      # Always reset pointers to handle odd-exit case which moves LRO to the upper bank
      if not self.prefetchAcrossPersistent and kernel["PrefetchGlobalRead"]:
        kStr += self.localReadResetOffsets(kernel, self.tPA)
        kStr += self.localReadResetOffsets(kernel, self.tPB)
      kStr += self.comment1("compute SerialWorkGroupIter / problemNumGroupTiles0 (aka numWorkGroups0)")
      kStr += self.sMagicDiv(kernel, stmp, sgpr("SerialWorkGroupIter"), sgpr("MagicNumberProblemNumGroupTiles0"), 31)
      kStr += inst("s_mov_b32", sgpr("WorkGroup1"), sgpr(stmp), "wg1 = SerialWorkGroupIter / problemNumGroupTiles0")
      kStr += inst("s_mul_i32", sgpr("WorkGroup0"), sgpr(stmp), sgpr("NumWorkGroups0"), "remainder part 1 : quotient * divisor")
      kStr += inst("s_sub_u32", sgpr("WorkGroup0"), sgpr("SerialWorkGroupIter"), sgpr("WorkGroup0"), "wg0 = SerialWorkGroupIter % problemNumGroupTiles0")

      #kStr += self.assert_ne(sgpr("SerialWorkGroupIter"), 2)
      kStr += "\n"


    kStr += self.comment1("graWorkGroup mapping")
    if kernel["GlobalSplitU"] > 1:
      if kernel["GlobalSplitUWorkGroupMappingRoundRobin"]:
        # gsuSumIdx = wg1 / nwg1
        # wg1       = wg1 % nwg1

        # nwg1
        nwg1 = self.vgprPool.checkOut(1, "nwg1", self.preventVgprOverflowDuringNewTile)
        tmpVgpr = self.vgprPool.checkOut(2, "tmpVgpr", self.preventVgprOverflowDuringNewTile)
        quotient = self.vgprPool.checkOut(1, "quotient", self.preventVgprOverflowDuringNewTile)
        tmpSgpr = self.getTmpSgpr(1).idx()
        kStr += "// GSU-WGMapRR :nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
            % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
        kStr += inst("v_mov_b32", vgpr(nwg1), sgpr("SizesFree+1"), "")
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile1"]-1), "")
        kStr += inst("_v_add_co_u32", vgpr(nwg1), "vcc", sgpr(tmpSgpr), vgpr(nwg1), \
            "%s = size1+MT1-1"%vgpr(nwg1))
        kStr += vectorStaticDivide(quotient, nwg1, kernel["MacroTile1"], tmpVgpr, tmpSgpr)
        self.vgprPool.checkIn(nwg1)
        nwg1 = quotient

        # wg1
        wg1 = self.vgprPool.checkOut(1, "wg1", self.preventVgprOverflowDuringNewTile)
        kStr += inst("v_mov_b32", vgpr(wg1), sgpr("WorkGroup1"), "wg1")

        # gsuSumIdx = wg1 / nwg1
        # wg1       = wg1 % nwg1
        quotient = self.vgprPool.checkOut(1, "quotient", self.preventVgprOverflowDuringNewTile)
        remainder = self.vgprPool.checkOut(1, "remainer", self.preventVgprOverflowDuringNewTile)
        tmpVgpr1 = self.vgprPool.checkOut(1, "tmpVgpr1", self.preventVgprOverflowDuringNewTile)
        dividend = wg1
        divisor = nwg1
        kStr += "DYNAMIC_VECTOR_DIVIDE %s %s %s %s %s %s %s%s" \
            % ( quotient, remainder, dividend, divisor, \
            tmpVgpr, tmpVgpr1, tmpSgpr, self.endLine )

        # move vgprs into sgprs
        kStr += inst("v_readfirstlane_b32", sgpr("GSUSumIdx"), \
            vgpr(quotient), "")
        kStr += inst("v_readfirstlane_b32", sgpr("WorkGroup1"), \
            vgpr(remainder), "")
        self.vgprPool.checkIn(tmpVgpr)
        self.vgprPool.checkIn(tmpVgpr1)
        self.vgprPool.checkIn(nwg1)
        self.vgprPool.checkIn(wg1)
        self.vgprPool.checkIn(quotient)
        self.vgprPool.checkIn(remainder)
      else:
        kStr += "// GSU-not-WGMapRR :nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
            % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)

        # gsuSumIdx = wg1 % GSU
        # wg1       = wg1 / GSU
        tmpSgpr = self.getTmpSgpr(3).idx() # needs 3
        divisor = tmpSgpr+2
        kStr += inst("s_mov_b32", sgpr(divisor), sgpr("WorkGroup1"), \
            "copying for divisor")

        #tmp = self.vgprPool.checkOut(1)

        #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("WorkGroup1"), "wg1")
        #kStr += dump(vgpr(tmp)) # numerator

        kStr += scalarStaticDivideAndRemainder("WorkGroup1", "GSUSumIdx", \
            divisor, kernel["GlobalSplitU"], tmpSgpr, 1)

        #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("WorkGroup1"), "wg1")
        #kStr += dump(vgpr(tmp)) # quotient
        #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("GSUSumIdx"), "gsusumidx")
        #kStr += dump(vgpr(tmp)) # remainder
        #self.vgprPool.checkIn(tmp)
        #kStr += "s_endpgm\n"

    ########################################
    # Blocked rows or columns
    absWgm = abs(kernel["WorkGroupMapping"])
    if kernel["WorkGroupMappingType"] == "B" and abs(kernel["WorkGroupMapping"]) > 1:
      smallNumMagicShift = 31
      magicNumberWgm = ((1<<smallNumMagicShift) // absWgm + 1)

      tmpSgpr = self.getTmpSgpr(4).idx()
      blockId2  = tmpSgpr+0
      wgSerial2 = tmpSgpr+1
      wgmDivisor = tmpSgpr+2
      wgmDivisorMagicNumber = tmpSgpr+3

      kStr += inst("s_mov_b32", sgpr(wgmDivisorMagicNumber), hex(magicNumberWgm)+'L', \
          "magic number for WGM==%u"%absWgm)
      # blockId and serial within block

      # note this overwrites blockId2+1
      kStr += self.sMagicDiv(kernel, dest=blockId2, dividend=sgpr("WorkGroup1"), \
          magicNumber=sgpr(wgmDivisorMagicNumber), magicShift=smallNumMagicShift)
      kStr += inst("s_mul_i32", sgpr(wgSerial2), sgpr(blockId2), absWgm, "quotient * non-magic divisor")
      kStr += inst("s_sub_u32", sgpr(wgSerial2), sgpr("WorkGroup1"), sgpr(wgSerial2), "WorkGroup1=remainder")
      kStr += inst("s_mul_i32", sgpr(wgSerial2), sgpr(wgSerial2), sgpr("NumWorkGroups0"), "(wg1 % WGM)*nwg0")
      kStr += inst("s_add_u32", sgpr(wgSerial2), sgpr(wgSerial2), sgpr("WorkGroup0"), "wgSerial = wg0 + (wg1 % WGM)*nwg0")

      kStr += inst("s_cmp_ge_u32", sgpr(blockId2), sgpr("NumFullBlocks"), "blockId >= numFullBlocks ?")
      # reuse wgmDivisorMagicNumber - may override with remainder here:
      kStr += inst("s_cmov_b32", sgpr(wgmDivisorMagicNumber), sgpr("MagicNumberWgmRemainder1"),  "")
      kStr += inst("s_cselect_b32", sgpr(wgmDivisor), sgpr("WgmRemainder1"), absWgm,  "")

      if kernel["WorkGroupMapping"]>=0 :
        firstWg = "WorkGroup0"
        secondWg = "WorkGroup1"
      else:
        firstWg = "WorkGroup1"
        secondWg = "WorkGroup0"

      assert(self.sgprs[firstWg] & 0x1 == 0) # must be even and ...
      assert(self.sgprs[firstWg]+1 == self.sgprs[secondWg] ) # must be consecutive (for magic div below)
      kStr += self.sMagicDiv(kernel, dest=self.sgprs[firstWg], dividend=sgpr(wgSerial2), \
          magicNumber=sgpr(wgmDivisorMagicNumber), magicShift=smallNumMagicShift)
      if kernel["WorkGroupMapping"]<0 :
        kStr += inst("s_mov_b32", sgpr("WorkGroup0"), sgpr(firstWg), "")
      kStr += inst("s_mul_i32", sgpr("WorkGroup1"), sgpr("WorkGroup0"), sgpr(wgmDivisor), "quotient * non-magic divisor")
      kStr += inst("s_sub_u32", sgpr("WorkGroup1"), sgpr(wgSerial2), sgpr("WorkGroup1"), "WorkGroup1=remainder")

      kStr += inst("s_mul_i32", sgpr(blockId2), sgpr(blockId2), \
          abs(kernel["WorkGroupMapping"]), "blockId * WGM")

      kStr += inst("s_add_u32", sgpr(secondWg), sgpr(secondWg), \
          sgpr(blockId2), "wg1 += blockId * WGM")

    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  # global read addresses: tile offset assignment (message from .s)
  ##############################################################################
  def graTileAssignment(self, kernel, tP):
    kStr = ""
    if tP["grcg"]:
      if tP["grcv"]:
        divisorName = tP["lvc"]
      else:
        # Fractional load use the more accurate lsc, multiply by VW later
        divisorName = tP["lsc"]
    else:
      if tP["grcv"]:
        divisorName = tP["lsp"]
      else:
        divisorName = tP["lvp"]
    divisor = kernel[divisorName]

    if tP["grcg"] == tP["tlu"]:
      rReg = self.vgprPool.checkOut(1, "graTA rReg0", self.preventVgprOverflowDuringNewTile) # gro-tile = serial%divisor
      qReg = self.vgprPool.checkOut(1, "graTA qReg0", self.preventVgprOverflowDuringNewTile) # gro-unroll = serial/divisor
      tReg = rReg
      uReg = qReg
      tOpStr = "%"
      uOpStr = "/"
    else:
      qReg = self.vgprPool.checkOut(1, 'graTA qReg1', self.preventVgprOverflowDuringNewTile) # gro-tile = serial/divisor
      rReg = self.vgprPool.checkOut(1, 'graTA rReg1', self.preventVgprOverflowDuringNewTile) # gro-unroll = serial%divisor
      tReg = qReg
      uReg = rReg
      tOpStr = "/"
      uOpStr = "%"

    kStr += self.comment1("%s = %u" % (divisorName, kernel[divisorName]))
    if self.groOffsetInMacroTile:
      tReg2 = tReg
      # treg2 and treg same register and value - we store the 'static'
      # part of the address calculation in the SRD to maximize the
      # range of the 32-bit GRO
      kStr += self.comment1("%s = (local)gro%s-tile = serial%s%s (note (wg%s*MT%s) will be added to SRD)" \
          % (vgpr(tReg2), tP["tensorChar"], tOpStr, divisorName, tP["tensorChar"], tP["tensorChar"]) )
    else:
      tReg2 = self.vgprPool.checkOut(1, 'treg2', self.preventVgprOverflowDuringNewTile)
      kStr += self.comment1("%s = gro%s-tile = serial%s%s + (wg%s*MT%s)" \
          % (vgpr(tReg2), tP["tensorChar"], tOpStr, divisorName, tP["tensorChar"], tP["tensorChar"]) )

    kStr += self.comment1("%s = gro%s-unroll = serial%s%s" \
        % (vgpr(uReg), tP["tensorChar"], uOpStr, divisorName) )
    dividendReg = "Serial" # local serial
    tmpVgpr = self.vgprPool.checkOut(2, 'graTA vgpr', self.preventVgprOverflowDuringNewTile)
    tmpSgpr = self.getTmpSgpr(1).idx()
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)

    if tP["glvw"] > 1:
      if tP["grcv"] == tP["tlu"]:
        kStr += self.comment1("gro-tile *= glvw")
        kStr += staticMultiply(vgpr(tReg), vgpr(tReg), tP["glvw"], sgpr(tmpSgpr))
      else:
        kStr += self.comment1("gro-unroll *= glvw")
        kStr += staticMultiply(vgpr(uReg), vgpr(uReg), tP["glvw"], sgpr(tmpSgpr))
    if not self.groOffsetInMacroTile:
      # Buffer Load will set the SRD to start of the MacroTile
      # So don't add the static wg-related component here - save for later.
      kStr += staticMultiply(vgpr(tmpVgpr), sgpr(tP["wg"]), kernel[tP["mt"]])  # workgroup
      kStr += inst("_v_add_co_u32", vgpr(tReg2), "vcc", vgpr(tmpVgpr), \
          vgpr(tReg), "gro%s-tile = serial%s%s*VW + (wg%s*MT%s)" \
          % (tP["tensorChar"], tOpStr, divisorName, tP["tensorChar"], tP["tensorChar"]) )

    if kernel["GlobalSplitU"] > 1:
      uReg2 = self.vgprPool.checkOut(1, "uReg2", self.preventVgprOverflowDuringNewTile)
      kStr += inst("v_mov_b32", vgpr(uReg2), vgpr(uReg), "copy for GlobalSplitU")
      tP["gpr"]["uReg2"] = uReg2
    tP["gpr"]["lwoT"] = tReg
    tP["gpr"]["tReg"] = tReg2
    tP["gpr"]["uReg"] = uReg
    self.vgprPool.checkIn(tmpVgpr)

    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Assignment
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    kStr = ""
    # note groOffsetInMacroTile rolls these into SRD so don't change here:
    if not self.groOffsetInMacroTile and kernel["GlobalSplitU"] > 1:
      gsuOffset = self.vgprPool.checkOut(1, "gsuOffset", self.preventVgprOverflowDuringNewTile)
      kStr += inst("v_mov_b32", vgpr(gsuOffset), sgpr("GSUSumIdx"), "=gsuSumIdx")
      if kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        # graUnrollAssignment += gsuSumIdx*DepthU
        tmpSgpr = self.getTmpSgpr(1).idx()
        kStr += staticMultiply(vgpr(gsuOffset), vgpr(gsuOffset), kernel["DepthU"], sgpr(tmpSgpr))
      else:
        # graUnrollAssignment += gsuSumIdx*(SizeU/GSU)
        sizeU = self.vgprPool.checkOut(1, "sizeU", self.preventVgprOverflowDuringNewTile)
        kStr += inst("v_mov_b32", vgpr(sizeU), sgpr("SizesSum+0"), \
            "=Size%s"%self.unrollChar)
        quotient = self.vgprPool.checkOut(1, "quotient", self.preventVgprOverflowDuringNewTile)
        dummy = self.vgprPool.checkOut(1, "dummy", self.preventVgprOverflowDuringNewTile)
        tmpVgpr = self.vgprPool.checkOut(2, "tmpVgpr", self.preventVgprOverflowDuringNewTile)
        tmpSgpr = self.getTmpSgpr(1).idx()
        kStr += vectorStaticDivideAndRemainder(quotient, dummy, sizeU, \
            kernel["GlobalSplitU"], tmpVgpr, tmpSgpr)
        self.vgprPool.checkIn(sizeU)
        self.vgprPool.checkIn(dummy)
        self.vgprPool.checkIn(tmpVgpr)
        #kStr += " + (size%s/GLOBAL_SPLITU)*" % self.unrollChar
        kStr += inst("v_mul_lo_u32", vgpr(gsuOffset), vgpr(quotient), \
            vgpr(gsuOffset), "gsuOffset=gsuSumIdx*(SizeU/GSU)")
        self.vgprPool.checkIn(quotient)

      kStr += inst("_v_add_co_u32", vgpr(tP["gpr"]["uReg"]), "vcc", \
          vgpr(gsuOffset), vgpr(tP["gpr"]["uReg"]), \
          "graUnrollAssignment += gsuOffset")
      self.vgprPool.checkIn(gsuOffset)
    else:
      kStr += self.comment1(vgpr(tP["gpr"]["uReg"]))

    return kStr

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  def graOtherFreeAssignments(self, kernel):
    return self.comment1(sgpr("WorkGroup2"))

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  def graOtherSummationAssignments(self, kernel):
    kStr = ""
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      index = i
      kStr += ".set globalReadOffsetA%s,  0%s" \
          % (self.indexChars[index], self.endLine)
      kStr += ".set globalReadOffsetB%s,  0%s" \
          % (self.indexChars[index], self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  def graTileOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    tP["vgprPackedOffsets"] = None
    if kernel["_UseSgprForGRO"]:
      # Let the vgprTileOffsets checkin handle tReg later since these are same vgpr
      tP["vgprTileOffsets"] = tP["gpr"]["tReg"]
    else:
      numTileOffsets = tP["nrt"]
      if tP["rtc"]:
        numTileOffsets *= tP["glvw"]
      tP["vgprTileOffsets"] = self.vgprPool.checkOut(numTileOffsets, "vgprTileOffsets", self.preventVgprOverflowDuringNewTile)
      v = tP["vgprTileOffsets"]
      numExtraPackedOffsetsPerTile = len(tP["PackedIndices"])-1
      if numExtraPackedOffsetsPerTile:
        tP["vgprPackedOffsets"] = self.vgprPool.checkOut(numExtraPackedOffsetsPerTile * numTileOffsets, "vgprPackedOffsets", self.preventVgprOverflowDuringNewTile)
      strideIdx = tP["lsc"] if tP["tlu"] else tP["lsp"]
      stride = kernel[strideIdx]

      if tP["rtc"]:
        assert(numExtraPackedOffsetsPerTile == 0) # not supported here
        # l=0, s=0
        kStr += inst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["tReg"]), "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], 0, 0) )
        # l=0, s>0
        for s in range(1, tP["glvw"]):
          kStr += inst("_v_add_co_u32", vgpr(v+s), "vcc", 1, \
              vgpr(v+s-1), "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], 0, s) )
        for l in range(1, tP["nrt"]):
          # l>0, s=0
          kStr += inst("_v_add_co_u32", vgpr(v+l*tP["glvw"]), "vcc", stride, \
              vgpr(v+(l-1)*tP["glvw"]), \
              "gro%s%s_%u_s%u + %s"%(tP["tensorChar"], tP["tileChar"], l, 0, strideIdx) )
          # l>0, s>0
          for s in range(1, tP["glvw"]):
            kStr += inst("_v_add_co_u32", vgpr(v+l*tP["glvw"]+s), "vcc", \
                1, vgpr(v+l*tP["glvw"]+(s-1)), \
                "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], l, s) )

      else:
        kStr += inst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["tReg"]), "gro%s%s_%u"%(tP["tensorChar"], tP["tileChar"], 0) )
        for l in range(1, tP["nrt"]):
          kStr += inst("_v_add_co_u32", vgpr(v+l), "vcc", stride, \
              vgpr(v+l-1), "gro%s%s_%u += %s"%(tP["tensorChar"], tP["tileChar"], l, strideIdx) )
        if numExtraPackedOffsetsPerTile:
          tmpV = self.vgprPool.checkOutAligned(2,2,"packTmp", self.preventVgprOverflowDuringNewTile)

          for l in range(0, tP["nrt"]):
            lastGroVgpr = vgpr(v+l)
            lastGroIdx = tP["PackedIndices"][0]
            kStr += "\n"
            for p in range(0, numExtraPackedOffsetsPerTile):
              groIdx  = tP["PackedIndices"][p+1]
              groChar = globalParameters["IndexChars"][tP["PackedIndices"][p+1]]
              groVgpr = vgpr(tP["vgprPackedOffsets"] + l*numExtraPackedOffsetsPerTile + p)
              pChar = globalParameters["IndexChars"][tP["PackedIndices"][p]]
              kStr += "V_MAGIC_DIV %s, %s, %s, %s, %s\n" \
                  % (tmpV, lastGroVgpr, sgpr("MagicNumberSize%s"%pChar), \
                  sgpr("MagicShiftSize%s"%pChar), sgpr("MagicAbitSize%s"%pChar) if kernel["MagicDivAlg"]==2 else "0")
              kStr += inst("v_mov_b32", groVgpr, vgpr(tmpV), "extract gro%s%s_%u (%s)"%(tc,groChar,l,groVgpr))
              kStr += inst("v_mul_lo_u32", vgpr(tmpV), groVgpr, sgpr("SizesFree+%u"%lastGroIdx), "remainder part 1")
              kStr += inst("v_sub_u32", lastGroVgpr, lastGroVgpr, vgpr(tmpV), \
                  "remove extracted bits from gro%s%s_%u (%s)"%(tc, globalParameters["IndexChars"][lastGroIdx], l, lastGroVgpr))
              lastGroVgpr = groVgpr
              lastGroIdx = groIdx
          self.vgprPool.checkIn(tmpV)

      # groOffsetInMacroTile uses same register for both of these, don't free it here:
      if tP["gpr"]["lwoT"] != tP["gpr"]["tReg"] :
        self.vgprPool.checkIn(tP["gpr"]["tReg"])
        tP["gpr"]["tReg"] = None
    return kStr


  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  def graUnrollOffsets(self, kernel, tP):
    kStr = ""
    if kernel["_UseSgprForGRO"]:
      tP["gpr"]["unrollOffsets"] = tP["gpr"]["uReg"]
    else:
      numUnrollOffsets = tP["nru"]
      if tP["ruc"]:
        numUnrollOffsets *= tP["glvw"]
      tP["gpr"]["unrollOffsets"] = self.vgprPool.checkOut(numUnrollOffsets, "unrollOffsets", self.preventVgprOverflowDuringNewTile)
      v = tP["gpr"]["unrollOffsets"]
      strideIdx = (tP["lsp"] if tP["tlu"] else tP["lsc"])
      stride = kernel[strideIdx]
      if tP["ruc"]:
        # l=0, s=0
        kStr += inst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["uReg"]), "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, 0) )
        # l=0, s>0
        for s in range(1, tP["glvw"]):
          kStr += inst("_v_add_co_u32", vgpr(v+s), "vcc", 1, \
              vgpr(v+s-1), "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, s) )
        for l in range(1, tP["nru"]):
          # l>0, s=0
          kStr += inst("_v_add_co_u32", vgpr(v+l*tP["glvw"]), "vcc", stride, \
              vgpr(v+(l-1)*tP["glvw"]), \
              "gro%s%s_%u_s%u + %s"%(tP["tensorChar"], self.unrollChar, l, 0, strideIdx) )
          # l>0, s>0
          for s in range(1, tP["glvw"]):
            kStr += inst("_v_add_co_u32", vgpr(v+l*tP["glvw"]+s), "vcc", \
                1, vgpr(v+l*tP["glvw"]+(s-1)), \
                "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, s) )
      else:
        kStr += inst("v_mov_b32", vgpr(v), \
            vgpr(tP["gpr"]["uReg"]), "gro%s%s_%u"%(tP["tensorChar"], self.unrollChar, 0) )
        for l in range(1, tP["nru"]):
          kStr += inst("_v_add_co_u32", vgpr(v+l), "vcc", stride, \
              vgpr(v+l-1), "gro%s%s_%u + %s"%(tP["tensorChar"], self.unrollChar, l, strideIdx) )
      #self.vgprPool.checkIn(tP["gpr"]["uReg"])
    return kStr


  ##############################################################################
  # Global Read Addresses: Branch A/B
  ##############################################################################
  def graBranch(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A/B
  # See if the load (including vw) will extend past the 'free' dim of the 
  # tensor.  If so clip to the last legal value which is inside the array

  ##############################################################################
  def graShift(self, kernel, tP):
    # FractionalLoad maps addresses in a different way?

    # graShift requires a vgpr for each address component (so each component
    # can be examined and shifted if necessary) - therefore does not work
    # with UseSgprForGRO.
    assert(not kernel["_UseSgprForGRO"])

    kStr = ""
    tc = tP["tensorChar"]
    # edge value
    margin = tP["glvw"] if tP["rtv"] else 1
    edge = self.vgprPool.checkOut(1, "edge", self.preventVgprOverflowDuringNewTile)

    if self.groOffsetInMacroTile:
      # Subtract the static component from SizesFree:
      tmpSgpr = self.getTmpSgpr(1).idx()
      kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr(tP["wg"]), kernel[tP["mt"]], "WorkGroup[01] * MT")
      kStr += inst("s_sub_u32", sgpr(tmpSgpr), self.sizeRef(tP["idx"]), sgpr(tmpSgpr), \
                "edge = Size%s - WG*MT"%(tP["tileChar"]))
      # use math here to use unsigned (to increase range)
      #  - add srdShiftLeft to tmpSgpr - ensure it is always positive
      #  - below add srdShiftLeft to a tmp copy of the offset used for the compare
      kStr += inst("s_sub_u32", sgpr(tmpSgpr), sgpr(tmpSgpr), margin, "edge -= margin")
      kStr += inst("v_mov_b32", vgpr(edge), sgpr(tmpSgpr), \
          "edge vgpr = Size%s-%u"%(tP["tileChar"], margin) )
      shiftedEdge = self.vgprPool.checkOut(1, "shiftedEdge", self.preventVgprOverflowDuringNewTile)
      kStr += inst("_v_add_co_u32", vgpr(shiftedEdge), "vcc", vgpr(edge), self.srdShiftLeft[tc], "add srdShiftLift")
    else:
      tmpSgpr = self.getTmpSgpr(1).idx()
      kStr += inst("s_sub_u32", sgpr(tmpSgpr), self.sizeRef(tP["idx"]), margin, \
          "edge = Size%s-%u"%(tP["tileChar"], margin) )
      kStr += inst("v_mov_b32", vgpr(edge), sgpr(tmpSgpr), \
          "edge vgpr = Size%s-%u"%(tP["tileChar"], margin) )

    if kernel["CheckDimOverflow"]:
      # if tensor is really skinnty (SizesFree is less then glvw) then shifting fails-
      # can detect here if the computed edge after subtracting marging is <0
      kStr += self.assert_ge_i32(vgpr(edge), 0)
    #kStr += self.assert_ne(sgpr("WorkGroup0"),1)

    # shift offsets
    v = tP["vgprTileOffsets"]
    tmpSgpr = self.getTmpSgpr(2).idx()
    for l in range(0, tP["nrt"]):
      # compare
      if self.groOffsetInMacroTile:
        shiftedOffset = self.vgprPool.checkOut(1, "shiftedOffset", self.preventVgprOverflowDuringNewTile)
        kStr += inst("_v_add_co_u32", vgpr(shiftedOffset), "vcc", vgpr(v+l), self.srdShiftLeft[tc], "")
        # int cmp since if we are near the front of the tile this may go negative:
        kStr += inst("v_cmp_lt_u32", sgpr(tmpSgpr,2), vgpr(shiftedOffset), vgpr(shiftedEdge), "offset < edge" )
        self.vgprPool.checkIn(shiftedOffset)
      else:
        kStr += inst("v_cmp_lt_u32", sgpr(tmpSgpr,2), vgpr(v+l), vgpr(edge), "offset < edge" )
      # shift
      kStr += inst("v_cndmask_b32", vgpr(v+l), vgpr(edge), vgpr(v+l), sgpr(tmpSgpr,2), "offset = (offset < edge) ? offset : edge" )
    self.vgprPool.checkIn(edge)
    if self.groOffsetInMacroTile:
      self.vgprPool.checkIn(shiftedEdge)

    #if tP["isB"]:
    #  kStr += "s_endpgm\n"

    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  def graFinalOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    problemType = kernel["ProblemType"]
    tVW = 1
    tVS = 0
    uVW = 1
    uVS = 0
    if tP["rtc"]:
      tVW = tP["glvw"]
      tVS = 1
    elif tP["ruc"]:
      uVW = tP["glvw"]
      uVS = 1
    tmp = self.vgprPool.checkOut(3, "tmp", self.preventVgprOverflowDuringNewTile)
    graIdx = 0
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
            # vgpr assignments
            if tP["tlu"]:
              vgprTile   = tP["vgprTileOffsets"]   + para*tVW + sPara*tVS
              vgprUnroll = tP["gpr"]["unrollOffsets"] + perp*uVW + sPerp*uVS
            else:
              vgprTile   = tP["vgprTileOffsets"]   + perp*tVW + sPara*tVS
              vgprUnroll = tP["gpr"]["unrollOffsets"] + para*uVW + sPerp*uVS

            if graIdx==0 or not kernel["_UseSgprForGRO"]:
              # emit global offset macro
              # TODO -refactor this and macro def to pass all indices, use the ones we need
              if kernel["BufferLoad"]:
                kStr += "GLOBAL_OFFSET_%s vgprGlobalReadOffset%s+%u"%(tP["tensorChar"], tP["tensorChar"], graIdx)
              else:
                kStr += "GLOBAL_OFFSET_%s vgprGlobalReadAddr%s+%u"%(tP["tensorChar"], tP["tensorChar"], graIdx)
              packedIter = 0 #iterator through ia
              iaToGpr = [None] * problemType["TotalIndices"]
              for i in tP["ia"]:
                if i < problemType["NumIndicesC"]:
                  if i == tP["tileIdx"]:
                    iaToGpr[i] = vgprTile
                    kStr += ", %2u" % iaToGpr[i]
                  else:
                    if isPackedIndex(kernel,i, tP["PackBatchDims"]):
                      iaToGpr[i] = tP["vgprPackedOffsets"] + \
                                    (vgprTile-tP["vgprTileOffsets"])*(len(tP["PackedIndices"])-1) + \
                                    packedIter
                      kStr += ", %2u" % (iaToGpr[i])
                      packedIter += 1
                    else:
                      # just a group index
                      if not kernel["BufferLoad"]:  # buffer load adds these to SRD not the GLOBAL_OFFSET here
                        kStr += ", sgprWorkGroup%u"%i
                else: # summation index
                  if i == problemType["IndexUnroll"]:
                    iaToGpr[i] = vgprUnroll
                    kStr += ", %2u" % iaToGpr[i]
                  # other summation indices are ignored

              kStr += ", %u // gRO%s_%u_%u_%u_%u%s" % (tmp, tP["tensorChar"], \
                  para, sPara, perp, sPerp, self.endLine)

              for zpr in [zpr for zpr in self.zeroPadRegs[tc].values() if zpr.isMatch(perp, sPerp, para, sPara)]:
                assert(zpr.state == ZeroPadReg.State.Allocated) # only calc address once
                zpr.state = ZeroPadReg.State.CalculatedAddr
                kStr += self.comment1(zpr.regName)
                (freeDim,sumDim) = zpr.zp[:2]
                freeDimChar = globalParameters["IndexChars"][freeDim]
                sumDimChar  = globalParameters["IndexChars"][sumDim]
                assert(iaToGpr[freeDim] != None)
                kStr += inst("v_mul_lo_u32", \
                          vgpr(zpr.regName), \
                          vgpr(iaToGpr[freeDim]), \
                          self.strideRef(tc, freeDim), \
                          "zp.freeDim * strideFree")
                #iaToGpr[sumDim] will be 0 for other summation dims
                kStr += inst("v_mul_lo_u32", \
                          vgpr(tmp), \
                          vgpr(iaToGpr[sumDim]) if vgpr(iaToGpr[sumDim]) else 0, \
                          self.strideRef(tc, sumDim), \
                          "zp.sumDim * strideSum")
                kStr += inst("_v_add_u32", \
                          vgpr(zpr.regName), \
                          vgpr(zpr.regName), \
                          vgpr(tmp),
                          "zp.freeDim * strideFree + zp.sumDim * strideSum")
                kStr += inst("v_lshlrev_b32", \
                             vgpr(zpr.regName), \
                             "Bpe%sLog2"%tc, \
                             vgpr(zpr.regName), \
                             "scale to bpe")
                kStr += inst("v_sub_u32",
                          vgpr(zpr.regName), \
                          vgpr(zpr.regName), \
                          sgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                          "zp.freeDim * strideFree + zp.sumDim * strideSum PadStart")

              if kernel["BufferLoad"] and kernel["FractionalLoad"]:
                tmpSgpr = self.getTmpSgpr(2).idx()
                lastValidThread = kernel[tP["lsc"]]*kernel[tP["lsp"]]//tP["glvw"]
                if lastValidThread < kernel["NumThreads"]:
                  kStr += "// Offset only valid for %u/%u threads inside the PerLoadTile\n" \
                       % (lastValidThread, kernel["NumThreads"])
                  kStr += inst("s_mov_b32", sgpr(tmpSgpr), lastValidThread, "" )
                  kStr += inst("v_cmp_lt_u32", \
                      "vcc", \
                      vgpr("Serial"), \
                      sgpr(tmpSgpr), \
                      "tid < valid-tid")
                  boundsVgpr = self.vgprPool.checkOut(3)
                  kStr += inst("s_mov_b32", sgpr(tmpSgpr), "BufferOOB", "" )
                  kStr += inst("v_mov_b32", vgpr(boundsVgpr), sgpr(tmpSgpr), "" )
                  kStr += inst("v_cndmask_b32", \
                       vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx)), \
                       vgpr(boundsVgpr), \
                       vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx)), \
                       "vcc",
                       "Mask load so OOB will return 0")
                  self.vgprPool.checkIn(boundsVgpr)

            if graIdx >0 and (kernel["_UseSgprForGRO"] or self.checkGRO):
              # compute offsets for scalar global read offsets:
              if kernel["_UseSgprForGRO"]:
                scalarGro = "ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1)
              else:
                scalarGro = self.getTmpSgpr(1).idx()

              # this needs unroll stride in some cases and free stride in others
              # if we have multiple free strides - what is expected behavior?
              # could just extract the first free dimension from A?
              stride1 = "Stride%s%s"%(tc,self.indexChars[tP["idx"]])
              if tP["tlu"]:
                tileStride   = kernel[tP["lsc"]] * (para*tVW + sPara*tVS)
                unrollStride = kernel[tP["lsp"]] * (perp*uVW + sPerp*uVS)
                unrollSummation = [ i for i in tP["ia"] if i in problemType["IndicesSummation"] ]
                strideU = "Stride%s%s"%(tc,self.indexChars[unrollSummation[-1]])
                kStr += inst("s_mul_i32", sgpr(scalarGro), sgpr(strideU), unrollStride, \
                             "compute offset diff (scaled unrollDim)")
                if tileStride:
                  kStr += inst("s_add_u32", sgpr(scalarGro), sgpr(scalarGro), tileStride, \
                             "compute offset diff (tileDim)")
              else:
                tileStride   = kernel[tP["lsp"]] * (perp*tVW + sPara*tVS)
                unrollStride = kernel[tP["lsc"]] * (para*uVW + sPerp*uVS)
                strideF = "Stride%s%s"%(tc,self.indexChars[tP['tileIdx']])
                kStr += inst("s_mul_i32", sgpr(scalarGro), sgpr(strideF), tileStride, \
                             "compute offset diff (scaled tileDim)")
                if unrollStride:
                  kStr += inst("s_add_u32", sgpr(scalarGro), sgpr(scalarGro), unrollStride, \
                             "compute offset diff (unrollDim)")

              # Using offsets so GRO holds a byte offset not an element offset
              # So scale here before comparison:
              kStr += inst("s_lshl_b32", \
                  sgpr(scalarGro), \
                  sgpr(scalarGro), \
                  hex(log2(tP["bpe"])), \
                  "scalar offset *= bytes/element")
              if self.checkGRO:
                # Debug mode to verify that the computed offsets are offset by the expected scalar
                print(tc, "tileStride=", tileStride, "unrollStride=", unrollStride, \
                      "stride=%s"%(stride1))

                kStr += self.assert_vector_diff(vgpr("GlobalReadOffset%s+%u"%(tc,0)), \
                                                vgpr("GlobalReadOffset%s+%u"%(tc,graIdx)), \
                                                sgpr(scalarGro))

              #-- End UseSgprForGRO
            # dump final offsets
            # BufferLoad flavor:
            #if tP["isA"]:
            #  kStr += self.dump(vgpr("GlobalReadOffset%s+%u+0"%(tP["tensorChar"], graIdx)))
            # Flat load flavor:
            #kStr += dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)))
            #kStr += dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)))
            graIdx += self.rpgo if kernel["BufferLoad"] else self.rpga

    if not self.groOffsetInMacroTile or not kernel["_UseSgprForGRO"]:
      self.vgprPool.checkIn(tP["vgprTileOffsets"])
      tP["vgprTileOffsets"] = None
      # _UseSgprForGRO uses same vgpr for ureg and tP["gpr"]["unrollOffsets"] so
      # let checkin(ureg) do the checkin
      # vgprTileOffsets is renamed version of treg/lwo so checkin here

    if not kernel["_UseSgprForGRO"]:
      self.vgprPool.checkIn(tP["gpr"]["unrollOffsets"])
      tP["gpr"]["unrollOffsets"] = None

    if tP["vgprPackedOffsets"] != None:
      self.vgprPool.checkIn(tP["vgprPackedOffsets"])
      tP["vgprPackedOffsets"] = None

    self.vgprPool.checkIn(tmp)
    #if tP["isB"]:
    #  kStr += self.bomb(0x100)

    # ensure we computed all the required addresses above
    for zpr in self.zeroPadRegs[tc].values():
      assert(zpr.state == ZeroPadReg.State.CalculatedAddr)

    return kStr

  ##############################################################################
  # Global Read Addresses: Apply User Offsets
  ##############################################################################
  def graApplyUserOffsets(self, kernel):
    kStr = ""
    kStr += self.comment1("moved earlier")
    return kStr


  ##############################################################################
  # Add the constant offsets to the specified srd.
  # Srd is set to point to the base of the tile. All offsets except lowest-order
  # 2d dims are computed into the SRD.
  # GRO are offset from the tile SRD and the first GRO will be 0
  # Only called for BufferLoad=1 (or eventually BufferStore=1)
  ##############################################################################
  def computeLoadSrd(self, kernel, tP, tc, indices, bpe):
    kStr = ""

    stmp = self.getTmpSgpr(2+2+1).idx()
    tileStart = stmp+2
    wroteTileStart = False

    #---
    # Compute tileStart #elements from the 2D array start
    # Add tile (and unroll if GSU) component into SRD - SRD will point to beginning of the macro-tile:
    if self.groOffsetInMacroTile:
      # packed modes can't use this mode, and code here assumes 1 index.
      assert(len(kernel["PackedC0IndicesX"])==1)
      assert(len(kernel["PackedC1IndicesX"])==1)

      wroteTileStart = True
      #tP['ia'][1]

      # This is guaranteed to fit in 32-bit since the WG*MT is a number of elements in some unsigned direction:
      kStr += self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(tP["wg"]), kernel[tP["mt"]], "WorkGroup[01] * MT")
      if kernel["CheckDimOverflow"] >=2:
        kStr += self.assert_eq(sgpr(tileStart+1),0)
      strideF = self.strideRef(tc, tP['tileIdx'])
      if not self.isConstUnitStride(strideF):
        kStr += self.s_mul_u64_u32(sgpr(tileStart), sgpr(tileStart+1), sgpr(tileStart+0), \
                   strideF, "tlu=0, scaled tile-offset by stride")

      if kernel["GlobalSplitU"] > 1:
        # Only GlobalSplitUSummationAssignmentRoundRobin supported for groOffsetInMacroTile - would need different math here for start:
        assert(kernel["GlobalSplitUSummationAssignmentRoundRobin"])

        kStr += self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), kernel["DepthU"], sgpr("GSUSumIdx"), "gsuOffset = DepthU*bpe*GSUSumIdx")
        if kernel["CheckDimOverflow"] >=2:
          kStr += self.assert_eq(sgpr(stmp+1),0)
        # TODO - PackSummationDims handling needs to handle multiple sum dims
        unrollSummation = [ i for i in tP["ia"] if i in kernel["ProblemType"]["IndicesSummation"] ]
        stride = self.strideRef(tc,unrollSummation[-1])
        if tP["tlu"] and not self.isConstUnitStride(stride):
          # non-transpose case, unroll is in perp dim and should be scaled by unroll Stride
          kStr += self.s_mul_u64_u32(sgpr(stmp), sgpr(stmp+1), sgpr(stmp+0), \
                    stride, "tlu=1, scaled unroll-offset by stride")

        kStr += inst("s_add_u32",  sgpr(tileStart+0), sgpr(tileStart+0), sgpr(stmp+0), "accum GsuOffet term to tilestart")
        kStr += inst("s_addc_u32", sgpr(tileStart+1), sgpr(tileStart+1), sgpr(stmp+1), "accum GsuOffet term to tilestart")


    # Output : tileStart[0:1] have offset in elements from the 2D start of the tile.
    # if groOffsetInMacroTile=1, 2DStart + tileStart gives the the start of the macro-tile;
    # This is used to compute the limit.
    # Later we modify tileStart to include batch and higher-order dims and add this to SRD.

    #---
    # Compute BUFFER Limit:
    prePad = prePadConst = self.srdShiftLeft[tc] * tP["bpe"] # leave room in case we have to pointer shift
    # subtract the zeropad(s) from the SRD base
    # this causes small offsets (<pad) to result in large negative offsets and thus report as OOB
    for i,zp in enumerate(kernel["ProblemType"]["ZeroPad%s"%tc]):
      (freeDim,sumDim) = zp[:2]
      freeDimChar = globalParameters["IndexChars"][freeDim]
      sumDimChar  = globalParameters["IndexChars"][sumDim]
      # override the const pre-pad with an SGPR based on the leading/trailing items:
      prePad = prePadSgpr = sgpr(stmp+4)
      if i==0:
        kStr += inst("s_add_u32", prePadSgpr, \
                 sgpr("PadStart%s%s%s"%(tc, freeDimChar,sumDimChar)), \
                 prePadConst, "prePadSgpr = PadStart + ptr-shift-pad")
      else:
        kStr += inst("s_add_u32", prePadSgpr, \
                 prePadSgpr, sgpr("PadStart%s%s%s"%(tc,freeDimChar, sumDimChar)), \
                 "prepadSgpr += PadStart")


    if not wroteTileStart:
      kStr += inst("s_mov_b32", sgpr(tileStart+0), 0, "set default tileStart")
      kStr += inst("s_mov_b32", sgpr(tileStart+1), 0, "set default tileStart")

    if self.use64bShadowLimit:
      limitTmp0 = "ShadowLimit%s+0"%tc
      limitTmp1 = "ShadowLimit%s+1"%tc
    else:
      limitTmp0 = stmp+0
      limitTmp1 = stmp+1

    kStr += inst("s_sub_u32",  sgpr(limitTmp0), sgpr("Tensor2dSize%s"%tc), sgpr(tileStart+0), "sub tileStart")
    kStr += inst("s_subb_u32", sgpr(limitTmp1), sgpr("Tensor2dSize%s+1"%tc), sgpr(tileStart+1), "sub tileStart")

    if self.use64bShadowLimit:
      # Set initial buffer limit
      # if the limit is >64bit, incrementSrd decrements the shadow as the SRD increments,
      # and when we get within 32-bit we start to step down the SRD
      # if the limit is <32bits, set it accurately here:
      # Note lshl_b64 the higher-numbered SGPR has the upper 32-bits
      kStr += inst("s_lshl_b64", sgpr("ShadowLimit%s"%tc,2),  sgpr("ShadowLimit%s"%tc,2), \
          hex(log2(tP["bpe"])), "Set limit to use bytes")
      if prePad:
        kStr += inst("s_add_u32",  sgpr("ShadowLimit%s+0"%tc), sgpr("ShadowLimit%s+0"%tc), prePad, "extend limit for pre-pad")
        kStr += inst("s_addc_u32", sgpr("ShadowLimit%s+1"%tc), sgpr("ShadowLimit%s+1"%tc), 0, "extend limit for pre-pad")

      kStr += inst("s_cmp_eq_u32", sgpr("ShadowLimit%s+1"%tc), 0, "are we within 2^32?")
      kStr += inst("s_cselect_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "BufferLimit", "Move shadow to real if we are within 2^32")
    else:
      # put limit directly into SRD:
      kStr += inst("s_lshl_b32", sgpr("Srd%s+2"%tc), sgpr(stmp+0), hex(log2(tP["bpe"])), "Set limit to use bytes")
      kStr += inst("s_add_u32",  sgpr("Srd%s+2"%tc), sgpr("Srd%s+2"%tc), prePad, "extend limit for pre-pad")

    # Apply any high-order address components to the tileStart and eventually the SRD - batch idx for batched gemm
    numDim = len(indices)
    wg=2 # TODO - refactor since only WG2 is supported and this is always batch
    for i in range(1, numDim):
      idx = indices[i]
      if idx == kernel["ProblemType"]["Index0"] \
          or idx == kernel["ProblemType"]["Index1"] \
          or idx in kernel["ProblemType"]["IndicesSummation"] \
          or isPackedIndex(kernel, idx):
            continue # these will be captured in GRO not the SRD (or other summations are always 0)
      else:
        assert(wg==2) # can only have one wg2 with a batch. Other dimensions should be packed into wg0/wg1
        stride = "Stride%s%s"%(tc,self.indexChars[tP['ia'][i]])
        if not wroteTileStart:
          kStr += self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(stride), sgpr("WorkGroup2"), "Stride*WG")
          wroteTileStart = True
        else:
          kStr += self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), sgpr(stride), sgpr("WorkGroup2"), "Stride*WG")
          kStr += inst("s_add_u32",  sgpr(tileStart+0), sgpr(tileStart+0), sgpr(stmp+0), "accum wg term to tilestart")
          kStr += inst("s_addc_u32", sgpr(tileStart+1), sgpr(tileStart+1), sgpr(stmp+1), "accum wg term to tilestart")
        wg+=1

    # Add the tile start to the SRD
    if wroteTileStart:
      kStr += inst("s_lshl_b64", sgpr(tileStart,2), sgpr(tileStart,2), log2(bpe), "tileStart *= BPE")
      kStr += inst("s_add_u32",  sgpr("Srd%s+0"%tc), sgpr("Address%s+0"%tc), sgpr(tileStart+0), "SRD base = Address+ tileStart0")
      kStr += inst("s_addc_u32", sgpr("Srd%s+1"%tc), sgpr("Address%s+1"%tc), sgpr(tileStart+1), "SRD base = Address+ tileStart1")
    else:
      kStr += inst("s_mov_b32", sgpr("Srd%s+0"%tc), sgpr("Address%s+0"%tc), "init SRD base address (lower )" )
      kStr += inst("s_mov_b32", sgpr("Srd%s+1"%tc), sgpr("Address%s+1"%tc), "init SRD base address (upper) + other fields" )

    if prePad:
      kStr += inst("s_sub_u32",  sgpr("Srd%s+0"%tc), sgpr("Srd%s+0"%tc), prePad, "pre-pad to make room for possible pointer shift")
      kStr += inst("s_subb_u32",  sgpr("Srd%s+1"%tc), sgpr("Srd%s+1"%tc), 0, "pre-pad to make room for possible pointer shift")

    kStr += inst("s_mov_b32", sgpr("Srd%s+3"%tc), "Srd127_96", "Set bits 127_96 in SRD")

    #if tP["isB"]:
   #   kStr += self.assert_ne(sgpr("WorkGroup1"), 0xA)

    if kernel["CheckDimOverflow"]>=2:
      # double-check to make sure the SRD limit is inside the allowed tensor:
      #   - compute size of tensor in elements (including all dimensions)
      #   - subtract the SRD base and SRD buffer limit
      #   - Make sure the 64bit result is >0
      kStr += inst("s_lshl_b64", sgpr(stmp,2), sgpr("Tensor2dSize%s"%tc,2), log2(bpe), "tensor size in bytes")
      kStr += inst("s_add_u32",  sgpr(stmp+0), sgpr(stmp+0), sgpr("Address%s+0"%tc), "add start ptr to compute tensor%s bot-right"%tc)
      kStr += inst("s_addc_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("Address%s+1"%tc), "add start ptr to compute tensor%s bot-right"%tc)
      kStr += inst("s_sub_u32",  sgpr(stmp+0), sgpr(stmp+0), sgpr("Srd%s+0"%tc), "sub SRD base")
      kStr += inst("s_subb_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("Srd%s+1"%tc), "sub SRD base")
      if self.use64bShadowLimit:
        kStr += inst("s_sub_u32", sgpr(stmp+0), sgpr(stmp+0), sgpr("ShadowLimit%s+0"%tc), "sub buffer size")
        kStr += inst("s_subb_u32", sgpr(stmp+1), sgpr(stmp+1), sgpr("ShadowLimit%s+1"%tc), "sub buffer size")
      else:
        kStr += inst("s_sub_u32",  sgpr(stmp+0), sgpr(stmp+0), sgpr("Srd%s+2"%tc), "sub buffer limit")

      kStr += self.assert_eq(sgpr(stmp+1), 0)  # must be 0 or we are way OOB
      kStr += self.assert_ge_u32(sgpr(stmp+0), 0) # diff greater than zero
      if 0 and tP["isB"]:
        t = self.vgprPool.checkOut(1, "t", self.preventVgprOverflowDuringNewTile)
        kStr += inst("s_add_u32", sgpr(stmp+0), sgpr("WorkGroup1"), sgpr("WorkGroup2"), "bozo, debug")
        kStr += inst("v_mov_b32", vgpr(t), 0x54, "")
        kStr += self.assert_ne(sgpr(stmp+0), vgpr(t) )
        self.vgprPool.checkIn(t)

    if kernel["PackSummationDims"]:
      kStr += self.comment("Save the initial SRD and limit for later address calculation")
      kStr += inst("s_mov_b32", sgpr("InitialSrd%sBase+0"%tc), sgpr("Srd%s+0"%tc), "save base")
      kStr += inst("s_mov_b32", sgpr("InitialSrd%sBase+1"%tc), sgpr("Srd%s+1"%tc), "save base")
      if self.use64bShadowLimit:
        kStr += inst("s_mov_b32", sgpr("InitialSrd%sLimit+0"%tc), sgpr("ShadowLimit%s+0"%tc), "save shadow limit")
        kStr += inst("s_mov_b32", sgpr("InitialSrd%sLimit+1"%tc), sgpr("ShadowLimit%s+1"%tc), "save shadow limit")
      else:
        kStr += inst("s_mov_b32", sgpr("InitialSrd%sLimit"%tc), sgpr("Srd%s+2"%tc), "save limit")

    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  def graAddresses(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    graIdx = 0

    if kernel["BufferLoad"]:
      # maxAddrSgpr = size[n] * stride[n-1]
      kStr += self.comment1("max read offset = size[n] * stride[n-1]")

      kStr += self.computeLoadSrd(kernel, tP, tc, kernel["ProblemType"]["IndexAssignments%s"%tc], tP["bpe"])

      #kStr += self.bomb(0x13) # after addresses and SRD set
    else:
      tmp = self.vgprPool.checkOut(2, "tmp", self.preventVgprOverflowDuringNewTile)
      kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("Address%s+0"%tP["tensorChar"]), "" )
      kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("Address%s+1"%tP["tensorChar"]), "" )
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):

              comment = "gRA%s_%u_%u_%u_%u = addr%s+grO%s_%u_%u_%u_%u" \
                  % (tP["tensorChar"], para, sPara, perp, sPerp, \
                  tP["tensorChar"], tP["tensorChar"], \
                  para, sPara, perp, sPerp )
              kStr += inst("_v_add_co_u32", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                  vgpr(tmp+0), \
                  comment+" (lower)")
              kStr += inst("_v_addc_co_u32", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  vgpr(tmp+1), \
                  "vcc", \
                  comment+" (upper)")
              #kStr += dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)))
              #kStr += dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)))
              graIdx += self.rpga
      #kStr += "s_endpgm\n"
      self.vgprPool.checkIn(tmp)

    return kStr

  ##############################################################################
  # Global Read Addresses: Increments
  # Define graIncrements, called once for each summation
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    kStr = ""
    tc = tP["tensorChar"]

    dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
    loopChar = self.indexChars[dimIdx]

    stride = self.strideRef(tc, dimIdx)

    #print (tc, ": loopIdx=", loopIdx, "dimIdx=", dimIdx, "strideIdx=", strideIdx)

    gsu = 1
    if kernel["GlobalSplitU"] > 1 \
        and kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
      gsu = kernel["GlobalSplitU"]

    assert(self.unrollIdx == kernel["ProblemType"]["NumIndicesSummation"]-1)
    if loopIdx==self.unrollIdx:
      if self.globalReadIncsUseVgpr:
        if kernel["PackSummationDims"]:
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+%u+0"%(tc, 2*loopIdx)), \
              stride, \
              "" )
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+%u+1"%(tc, 2*loopIdx)), \
              0,
              "" )
        else:
          tmpSgpr = self.getTmpSgpr(2).idx()
          kStr += inst("s_mul_i32", sgpr(tmpSgpr+0), \
              "DepthU*%d"%(gsu*tP["bpe"]), stride, \
              "incr%s%s = %s*DepthU*bpe (unrollIdx)"%(tc, loopChar, stride) )
          # TODO - this should be mul-H??
          kStr += inst("s_mov_b32", \
              sgpr(tmpSgpr+1), \
              hex(0), \
              "(carry)")
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+%u+0"%(tc, 2*loopIdx)), \
              sgpr(tmpSgpr+0), \
              "" )
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+%u+1"%(tc, 2*loopIdx)), \
              sgpr(tmpSgpr+1), \
              "" )
      else: # not globalReadIncsUseVgpr, ie use SGPR

        if kernel["PackSummationDims"]:
          m = "Bpe%s"%(tc)
        else:
          m = "DepthU*Bpe%s"%(tc)
        if gsu>1:
          m += "*%d"%gsu

        # multiply by stride, optimizing if unit stride
        if self.isConstUnitStride(stride):
          kStr += inst("s_mov_b32", sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), m, \
              "incr%s (unrollIdx)"%(tc) )
        else:
          kStr += inst("s_mul_i32", sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), \
              m, stride, \
              "incr%s unrollIdx)"%(tc) )
    else:
      # other summation
      if self.globalReadIncsUseVgpr:
        printExit("NumIndicesSummation=%u not yet supported in assembly unless globalReadIncsUseVgpr==0" \
            % kernel["ProblemType"]["NumIndicesSummation"] )
      else:
        graInc = "GlobalReadIncs%s+%u"%(tc, loopIdx)
        if kernel["PackSummationDims"]:
          # simpler address calculation here - don't need to subtract prev iteration increments
          # since only one iteration
          kStr += inst("s_lshl_b32", \
              sgpr(graInc), \
              stride, \
              "Bpe%sLog2"%tc,
              "<- scale by bpe")
        else:
          # subtract increments done by the inner iterations
          # may be negative:
          loopIdxPrev = loopIdx + 1
          dimIdxPrev    = kernel["ProblemType"]["IndicesSummation"][loopIdxPrev] # dimension index
          loopCharPrev  = self.indexChars[dimIdxPrev]
          stridePrev = self.strideRef(tc, dimIdxPrev)

          kStr += self.comment("compute globalReadInc for higher-level loop")

          tmpSgpr = self.getTmpSgpr(3).idx()
          # Summations always appear in both A and B, can compute number of iterations just once:
          if loopIdxPrev==self.unrollIdx:
            loopCounterName= self.loopCounterName(kernel, self.unrollIdx)
            if tP["isA"]:
              quotient = loopCounterName
              dividend = "SizesSum+%u"%self.unrollIdx
              divisor = kernel["DepthU"]
              kStr += scalarStaticDivideAndRemainder(quotient, None, dividend, \
                          divisor, tmpSgpr+2, 0)

              if kernel["GlobalSplitU"] > 1:
                kStr += self.calculateLoopNumIterGsu(kernel, loopCounterName, tmpSgpr)

              kStr += inst("s_mul_i32", sgpr(loopCounterName), sgpr(loopCounterName), \
                        kernel["GlobalSplitU"]*kernel["DepthU"], \
                        "=loopCounterName*DepthU")
            kStr += inst("s_mul_i32", sgpr(graInc), stridePrev, sgpr(loopCounterName), \
                  "tmp <- stride%s%s * myWgUnrollIters" %(tc, loopCharPrev))
          else:
            kStr += inst("s_mul_i32", sgpr(graInc), stridePrev, self.sizeRef(dimIdxPrev), \
                  "tmp <- stride%s%s * size%s%s" %(tc, loopCharPrev, tc, loopCharPrev))

          # subtract amount that previous inner loop will have already incremented:
          # graInc is used as temp for the prev loop calc
          kStr += inst("s_sub_i32", sgpr(graInc), \
              stride, \
              sgpr(graInc), \
              "incr%s%s = stride%s%s - <prev-incs>"%(tc, loopChar, tc, loopChar) )

          kStr += inst("s_lshl_b32", \
              sgpr(graInc), \
              sgpr(graInc), \
              "Bpe%sLog2"%tc,
              "<- scale by bpe")

        if 0 and tP["isB"] and loopIdx==0:
          kStr += self.bomb()
          #kStr += self.assert_ne(sgpr("WorkGroup1"),0)

    #kStr += dump(vgpr("GlobalReadIncs%s"%tP["tensorChar"]))
    #kStr += "s_endpgm\n"
    #if tP["isB"]:
    #  kStr += self.bomb(0x100)
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  def lwaTileAssignment(self, kernel, tP):
    return self.comment1("lwaTileAssignment%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["lwoT"])))

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    return self.comment1("lwaUnrollAssignment%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"])))

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    #"lwFOA = lwA%s + lwA%s*MT%s" \
    #    % (tP["tileChar"], self.unrollChar, tP["tileChar"])
    uReg = tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"]
    if kernel["LocalWriteUseSgpr%s"%tc]:
      destVgpr = self.vgprPool.checkOut(1, "destVgpr", self.preventVgprOverflowDuringNewTile)
    else:
      destVgpr = "LocalWriteAddr%s"%tc

    dotInterleave = kernel["LocalDotLayout"]

    if dotInterleave == 1:
      if kernel["UnrollMajorLDS%s" % tc]:
        lds_stride = kernel["DepthU"] + LdsPad
        kStr += inst("v_mul_u32_u24", vgpr(destVgpr), hex(lds_stride), vgpr(tP["gpr"]["lwoT"]), \
            "lw%s%s**(MT%s + PAD)"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
        kStr += inst("_v_add_lshl_u32", vgpr(destVgpr), vgpr(uReg), vgpr(destVgpr), hex(log2(tP["bpe"])), \
            "lwFO%s = (lw%s%s + lw%s%s*(MT%s+PAD))*bpe" % (tc, tc, tc, tc, self.unrollChar, tP["tileChar"]) )
      else:
        lds_stride = kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad
        kStr += inst("v_mul_u32_u24", vgpr(destVgpr), hex(lds_stride), vgpr(uReg), \
            "lw%s%s**(MT%s + PAD)"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
        kStr += inst("_v_add_lshl_u32", vgpr(destVgpr), vgpr(tP["gpr"]["lwoT"]), vgpr(destVgpr), hex(log2(tP["bpe"])), \
            "lwFO%s = (lw%s%s + lw%s%s*(MT%s+PAD))*bpe" % (tc, tc, tc, tc, self.unrollChar, tP["tileChar"]) )

      # LdsBlockSizePerPad: add padding
      if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
        tmpVgpr = self.vgprPool.checkOut(2)
        tmpSgpr = self.getTmpSgpr(1).idx()
        kStr += vectorStaticDivide(uReg, destVgpr, kernel["LdsBlockSizePerPad%s"%tc], tmpVgpr, tmpSgpr)
        kStr += staticMultiply(vgpr(uReg), vgpr(uReg), kernel["LdsPad%s"%tc] * tP["bpe"], sgpr(tmpSgpr))
        kStr += inst("v_add_u32", vgpr(destVgpr), vgpr(uReg), vgpr(destVgpr), "")
        self.vgprPool.checkIn(tmpVgpr)
    else:
      ldlOffsetVgpr = self.vgprPool.checkOut(1, "ldlOffsetVgpr", self.preventVgprOverflowDuringNewTile)
      uRegScrap = self.vgprPool.checkOut(1, "uRegScrap", self.preventVgprOverflowDuringNewTile)
      # likely broken for dot4, revisit
      # odd tiles will write to MT, even tiles to normal location
      kStr += inst("v_and_b32", \
          vgpr(destVgpr), \
          ~(kernel["LocalDotLayout"]-1), \
          vgpr(tP["gpr"]["lwoT"]), \
          "lwoT & ~(LDL-1)")
      # uReg bit 1 maps to LDS offset bit 1 (calculateLdsWriteOffset) or LocalWriteAddr (here)
      kStr += inst("v_and_b32", \
          vgpr(uRegScrap), \
          kernel["LocalDotLayout"]-1, \
          vgpr(uReg), \
          "uReg & LDL-1")
      kStr += inst("v_and_b32", \
          vgpr(uReg), \
          ~(kernel["LocalDotLayout"]-1), \
          vgpr(uReg), \
          "uReg & LDL-1")
      kStr += inst("v_and_b32", \
          vgpr(ldlOffsetVgpr), \
          kernel["LocalDotLayout"]-1, \
          vgpr(tP["gpr"]["lwoT"]), \
          "lwoT & LDL-1")
      kStr += inst("_v_lshl_add_u32", \
          vgpr(uReg), \
          vgpr(ldlOffsetVgpr), \
          #log2(kernel["LocalDotLayout"]), \
          0, \
          vgpr(uReg), \
          "shift scrap by LDL")
      kStr += inst("v_mul_u32_u24", \
          vgpr(uReg), \
          hex(kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad), \
          vgpr(uReg), \
          "lw%s%s**(MT%s + PAD)"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
      kStr += inst("_v_add_co_u32", \
          vgpr(uReg), \
          "vcc", \
          vgpr(uRegScrap), \
          vgpr(uReg), \
          "add scraps from LDL masking")
      kStr += inst("_v_add_lshl_u32", \
          vgpr(destVgpr), \
          vgpr(uReg), \
          vgpr(destVgpr), \
          hex(log2(tP["bpe"])), \
          " *= bpe")
      self.vgprPool.checkIn(uRegScrap)
      self.vgprPool.checkIn(ldlOffsetVgpr)

    if tP["isB"]:
      kStr += inst("_v_add_co_u32", \
          vgpr(destVgpr), \
          "vcc", \
          hex(kernel["LdsOffsetB"]*tP["bpe"]), \
          vgpr(destVgpr), \
          "lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u*%u" % (tP["tileChar"], \
          self.unrollChar, tP["tileChar"], kernel["LdsOffsetB"], self.bpeAB) )

    self.vgprPool.checkIn(tP["gpr"]["lwoT"])
    tP["gpr"]["lwoT"] = None
    self.vgprPool.checkIn(tP["gpr"]["uReg"])
    if kernel["GlobalSplitU"] > 1:
      self.vgprPool.checkIn(tP["gpr"]["uReg2"])

    #LSC_ * LSP_
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    validWIPerLoad = kernel[tP["lsc"]] * kernel[tP["lsp"]]//tP["glvw"]
    validBytesPerLoad = kernel[tP["lsc"]] * kernel[tP["lsp"]] * numBytesPerElement
    maxBytesPerLoad = kernel["NumThreads"] * tP["glvw"] * numBytesPerElement

    assert (validBytesPerLoad <= maxBytesPerLoad)
    assert (kernel[tP["lsc"]] * kernel[tP["lsp"]] % tP["glvw"] == 0)

    if validBytesPerLoad != maxBytesPerLoad:
      tmpSgpr = self.getTmpSgpr(1).idx()
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), validWIPerLoad, \
          "lsc*lsp=%u*%u"%(kernel[tP["lsc"]],kernel[tP["lsp"]] ))
      kStr += inst("v_cmp_lt_u32", \
          "vcc", \
          vgpr("Serial"), \
          sgpr(tmpSgpr), \
          "fractional: ensure tid < global read tile elements")
      tmpVgpr = self.vgprPool.checkOut(1, "tmpVgpr", self.preventVgprOverflowDuringNewTile)
      kStr += inst("v_mov_b32", vgpr(tmpVgpr), hex(self.LdsOOB), "")
      kStr += inst("v_cndmask_b32", \
                  vgpr(destVgpr), \
                  vgpr(tmpVgpr), \
                  vgpr(destVgpr), \
                   "vcc", \
                   "Mask load so out-of-gr-tile bounds returns 0")
      self.vgprPool.checkIn(tmpVgpr)

    if kernel["LocalWriteUseSgpr%s"%tc]:
      # TODO: Can refactor code above to Compute this directly:
      kStr += inst("v_readfirstlane_b32", \
          sgpr("LocalWriteAddr%s"%tc), \
          vgpr(destVgpr), \
          "Copy lds write address VGPR to SGPR")
      self.vgprPool.checkIn(destVgpr)

    if kernel["FractionalLoad"] and kernel["fractionalPerpOverhang%s"%tc]:
      overhang = kernel["fractionalPerpOverhang%s"%tc]
      validWI = overhang*kernel[tP["lsc"]]//tP["glvw"]
      if kernel["FractionalLoad"] == 2:
        mask = "PerpOverhangVcc%s"%tc
      else:
        mask = self.getTmpSgpr(2).idx()
      kStr += self.comment1("Compute fractional overhang")
      kStr += inst("s_mov_b32", sgpr(mask), validWI, \
          "overhang=%u, validWI=%u" % (overhang, validWI))
      kStr += inst("v_cmp_lt_u32", \
          sgpr(mask,2),
          vgpr("Serial"), \
          sgpr(mask,1),
          "fractional-overhang: some wi write to harmless LDS location")
      if kernel["FractionalLoad"] == 1:
        kStr += inst("v_cndmask_b32", \
                    vgpr("LocalWriteAddrOverhang%s"%tc), \
                    1.0, \
                    vgpr("LocalWriteAddr%s"%tc), \
                    sgpr(mask,2), \
                    "Mask load so out-of-gr-tile bounds returns 0. Note 1.0f=0x3f80000 which is large non-neg int")


    # dump lds write offsets
    #if tP["isA"]:
      #kStr += self.dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
      #kStr += self.bomb(-40)
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A/B
  ##############################################################################
  def lwaFinalOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Declare Addresses A/B
  ##############################################################################
  def lwaDeclareAddresses(self, kernel, tP):
    return ""


  ##############################################################################
  # Local Read Addresses: Tile Assignment A
  ##############################################################################
  def lraTileAssignmentVALUA(self, kernel, tP):
    kStr = ""

    # allocate resource
    qReg = self.vgprPool.checkOut(1,"qReg") # quotient
    rReg = self.vgprPool.checkOut(1,"rReg") # remainder
    tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
    tmpSgpr = self.getTmpSgpr(1).idx()

    # constant
    dividendReg = "Serial" # local serial
    divisor = kernel["SubGroup0"]

    # generate instruction
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgpr, tmpSgpr)

    # release and return resource
    tP["gpr"]["lro"] = rReg
    self.tmplroB = qReg
    self.vgprPool.checkIn(tmpVgpr)

    return kStr


  ##############################################################################
  # Local Read Addresses: Tile Assignment A
  ##############################################################################
  def lraTileAssignmentMFMAA(self, kernel, tP):
    kStr = ""

    # alloc vgpr
    tReg    = self.vgprPool.checkOut(1,"tReg") # remainder
    wReg    = self.vgprPool.checkOut(1,"wReg") # quotient
    kReg    = self.vgprPool.checkOut(1,"kReg") # remainder
    tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
    dummy   = self.vgprPool.checkOut(1,"dummy")

    # alloc sgpr
    tmpSgpr = self.getTmpSgpr(1).idx()

    # get constant parameter
    dividendReg    = "Serial" # local serial
    LdsPad         = kernel["LdsPadA"] if kernel["LdsBlockSizePerPadA"] == 0 else 0
    MIBShape0      = kernel["MatrixInstM"] * kernel["MatrixInstBM"] # matrix instruction MN shape for M
    dividendForK   = kernel["MatrixInstM"] * kernel["MatrixInstB"]
    inputPerThread = kernel["MatrixInstM"] * kernel["MatrixInstK"] * kernel["MatrixInstB"] // globalParameters["WavefrontWidth"]
    strideM        = 1
    strideK        = (kernel["MacroTile0"] + LdsPad) * inputPerThread
    strideWave     = kernel["MatrixInstM"] * kernel["MatrixInstBM"]

    # adjust stride according to lds orientation
    if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      strideM    = kernel["DepthU"] + LdsPad
      strideK    = inputPerThread
      strideWave = kernel["MatrixInstM"] * kernel["MatrixInstBM"] * strideM

    # thread offset
    kStr += vectorStaticRemainder(dummy, kReg, "Serial", globalParameters["WavefrontWidth"], tmpVgpr, tmpSgpr)
    kStr += vectorStaticRemainder(dummy, tReg, kReg, MIBShape0, tmpVgpr, tmpSgpr)
    kStr += staticMultiply(vgpr(tReg), vgpr(tReg), strideM, sgpr(tmpSgpr))

    kStr += vectorStaticDivide(kReg, kReg, dividendForK, tmpVgpr, tmpSgpr)
    kStr += staticMultiply(vgpr(kReg), vgpr(kReg), strideK, sgpr(tmpSgpr))
    kStr += inst("v_add_u32", vgpr(tReg), vgpr(kReg), vgpr(tReg), "")

    # wave offset
    if True:
    # if kernel["MIWaveGroup"][0] > 1:
      kStr += vectorStaticDivide(wReg, "Serial", globalParameters["WavefrontWidth"], tmpVgpr, tmpSgpr)
      kStr += vectorStaticRemainder(dummy, wReg, wReg, kernel["MIWaveGroup"][0], tmpVgpr, tmpSgpr)
      kStr += staticMultiply(vgpr(wReg), vgpr(wReg), strideWave, sgpr(tmpSgpr))
      kStr += inst("v_add_u32", vgpr(tReg), vgpr(wReg), vgpr(tReg), "")

    # release register
    tP["gpr"]["lro"] = tReg
    self.vgprPool.checkIn(wReg)
    self.vgprPool.checkIn(kReg)
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(dummy)

    return kStr


  ##############################################################################
  # Local Read Addresses: Tile Assignment A
  ##############################################################################
  def lraTileAssignmentA(self, kernel, tP):
    kStr = ""

    kStr += "%slr%s = serial %% SG%s%s%s" \
        % (self.commentPrefix, tP["tileChar"], tP["tileChar"], \
        self.commentSuffix, self.endLine)

    if kernel["EnableMatrixInstruction"]:
      kStr += self.lraTileAssignmentMFMAA(kernel, tP)
    else:
      kStr += self.lraTileAssignmentVALUA(kernel, tP)

    return kStr


  ##############################################################################
  # Local Read Addresses: Tile Assignment B
  ##############################################################################
  def lraTileAssignmentVALUB(self, kernel, tP):
    kStr = ""

    # allocate resources
    qReg    = self.vgprPool.checkOut(1,"qReg") # quotient
    rReg    = self.vgprPool.checkOut(1,"rReg") # remainder
    tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
    tmpSgpr = self.getTmpSgpr(1).idx()

    # constant
    divisor = kernel["SubGroup1"]
    dividendReg = self.tmplroB

    # generate instruction
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgpr, tmpSgpr)

    # release and return resource
    tP["gpr"]["lro"] = rReg

    self.vgprPool.checkIn(self.tmplroB) # old
    self.vgprPool.checkIn(qReg)
    self.vgprPool.checkIn(tmpVgpr)

    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment B
  ##############################################################################
  def lraTileAssignmentMFMAB(self, kernel, tP):
    kStr = ""

    # alloc vgpr
    wReg    = self.vgprPool.checkOut(1,"wReg") # quotient
    tReg    = self.vgprPool.checkOut(1,"tReg") # remainder
    kReg    = self.vgprPool.checkOut(1,"kReg") # remainder
    tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
    dummy   = self.vgprPool.checkOut(1,"dummy")

     # alloc sgpr
    tmpSgpr = self.getTmpSgpr(1).idx()

    # get constant parameter
    dividendReg    = "Serial" # local serial
    LdsPad         = kernel["LdsPadB"] if kernel["LdsBlockSizePerPadB"] == 0 else 0
    dividendForK   = kernel["MatrixInstN"] * kernel["MatrixInstB"]
    inputPerThread = kernel["MatrixInstN"] * kernel["MatrixInstK"] * kernel["MatrixInstB"] // globalParameters["WavefrontWidth"]
    strideN        = 1
    strideK        = (kernel["MacroTile1"] + LdsPad) * inputPerThread
    strideBN       =  kernel["MatrixInstN"]
    strideWave     = kernel["MatrixInstM"] * kernel["MatrixInstBN"]

    # adjust stride according to lds orientation
    if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      strideN    = kernel["DepthU"] + LdsPad
      strideK    = inputPerThread
      strideBN   = kernel["MatrixInstN"] * strideN
      strideWave = kernel["MatrixInstM"] * kernel["MatrixInstBN"] * strideN

    # thread offset
    kStr += vectorStaticRemainder(dummy, kReg, "Serial", globalParameters["WavefrontWidth"], tmpVgpr, tmpSgpr)
    kStr += vectorStaticRemainder(dummy, tReg, kReg, kernel["MatrixInstN"], tmpVgpr, tmpSgpr)
    kStr += staticMultiply(vgpr(tReg), vgpr(tReg), strideN, sgpr(tmpSgpr))

    kStr += vectorStaticDivide(wReg, kReg, (kernel["MatrixInstN"] * kernel["MatrixInstBM"]), tmpVgpr, tmpSgpr)
    kStr += vectorStaticRemainder(dummy, wReg, wReg, kernel["MatrixInstBN"], tmpVgpr, tmpSgpr)
    kStr += staticMultiply(vgpr(wReg), vgpr(wReg), strideBN, sgpr(tmpSgpr))
    kStr += inst("v_add_u32", vgpr(tReg), vgpr(wReg), vgpr(tReg), "")

    kStr += vectorStaticDivide(kReg, kReg, dividendForK, tmpVgpr, tmpSgpr)
    kStr += staticMultiply(vgpr(kReg), vgpr(kReg), strideK, sgpr(tmpSgpr))
    kStr += inst("v_add_u32", vgpr(tReg), vgpr(kReg), vgpr(tReg), "")

    # wave offset
    if True:
    # if kernel["MIWaveGroup"][0] > 1:
      kStr += vectorStaticDivide(wReg, "Serial", globalParameters["WavefrontWidth"], tmpVgpr, tmpSgpr)
      kStr += vectorStaticDivide(wReg, wReg, kernel["MIWaveGroup"][0], tmpVgpr, tmpSgpr)
      kStr += staticMultiply(vgpr(wReg), vgpr(wReg), strideWave, sgpr(tmpSgpr))
      kStr += inst("v_add_u32", vgpr(tReg), vgpr(wReg), vgpr(tReg), "")

    # release register
    tP["gpr"]["lro"] = tReg
    self.vgprPool.checkIn(wReg)
    self.vgprPool.checkIn(kReg)
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(dummy)

    return kStr


  ##############################################################################
  # Local Read Addresses: Tile Assignment B
  ##############################################################################
  def lraTileAssignmentB(self, kernel, tP):
    kStr = ""

    kStr += "%slr%s = (serial / SG%s) %% SG%s%s%s" \
        % (self.commentPrefix, tP["tileChar"], tP["tileChar"], \
        tP["tileChar"], self.commentSuffix, self.endLine)

    if kernel["EnableMatrixInstruction"]:
      kStr += self.lraTileAssignmentMFMAB(kernel, tP)
    else:
      kStr += self.lraTileAssignmentVALUB(kernel, tP)

    return kStr


  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  def lraFinalOffsetVALU(self, kernel, tP):
    kStr = ""

    # allocate resources
    qReg = self.vgprPool.checkOut(1) # quotient
    rReg = self.vgprPool.checkOut(1) # remainder, unused here
    tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
    tmpSgpr = self.getTmpSgpr(1).idx()

    # constant
    sgid        = qReg
    dividendReg = "Serial"
    tIdx        = tP["tensorIdx"]
    tc          = tP["tensorChar"]
    LdsPad      = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    divisor     = kernel["SubGroup0"] * kernel["SubGroup1"]

    # generate instruction
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)

    kStr += inst("s_mov_b32", \
        sgpr(tmpSgpr), \
        hex(kernel["MacroTile%u"%tP["tensorIdx"]] + LdsPad), \
        "MT%u+PAD"%tP["tensorIdx"] )

    kStr += inst("v_mul_lo_u32", \
        vgpr(sgid), \
        sgpr(tmpSgpr), \
        vgpr(sgid), \
        "sgid=sgid*(MT%u+PAD)"%tIdx )

    if kernel["VectorWidth"] > 1:
      kStr += staticMultiply(vgpr(tP["gpr"]["lro"]), vgpr(tP["gpr"]["lro"]), \
          kernel["VectorWidth"], sgpr(tmpSgpr))

    kStr += inst("_v_add_lshl_u32", \
        vgpr("LocalReadAddr%s"%tc), \
        vgpr(sgid), \
        vgpr(tP["gpr"]["lro"]), \
        hex(log2(tP["bpe"])), \
        "o = (lro%s*VW+sgid*MT%u)*bpe"%(tc, tIdx) )

    # release resources
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(qReg)
    self.vgprPool.checkIn(rReg)
    self.vgprPool.checkIn(tP["gpr"]["lro"])

    return kStr


  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  def lraFinalOffsetMFMA(self, kernel, tP):
    kStr = ""

    # allocate resources
    sgid    = self.vgprPool.checkOut(1) # quotient
    rReg    = self.vgprPool.checkOut(1) # remainder, unused here
    tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
    tmpSgpr = self.getTmpSgpr(1).idx()

    # constants
    dividendReg = "Serial"
    tc          = tP["tensorChar"]
    tIdx        = tP["tensorIdx"]
    LdsPad      = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    divisor     = kernel["SubGroup0"] * kernel["SubGroup1"]
    mtAddPad    = kernel["MacroTile%u"%tP["tensorIdx"]] + LdsPad

    # LSU offset
    kStr += vectorStaticDivideAndRemainder(sgid, rReg, dividendReg, divisor, tmpVgpr, tmpSgpr)
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(mtAddPad), "MT%u+PAD"%tP["tensorIdx"] )
    kStr += inst("v_mul_lo_u32", vgpr(sgid), sgpr(tmpSgpr), vgpr(sgid), "sgid=sgid*(MT%u+PAD)"%tIdx )

    # final offset
    kStr += inst("_v_add_lshl_u32", vgpr("LocalReadAddr%s"%tc), vgpr(sgid), vgpr(tP["gpr"]["lro"]), hex(log2(tP["bpe"])), "o = (lro%s*VW+sgid*MT%u)*bpe"%(tc, tIdx) )

    # LdsBlockSizePerPad: add padding
    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] !=0:
      kStr += vectorStaticDivide(rReg, "LocalReadAddr%s"%tc, kernel["LdsBlockSizePerPad%s"%tc], tmpVgpr, tmpSgpr)
      kStr += staticMultiply(vgpr(rReg), vgpr(rReg), kernel["LdsPad%s"%tc] * tP["bpe"], sgpr(tmpSgpr))
      kStr += inst("v_add_u32", vgpr("LocalReadAddr%s"%tc), vgpr(rReg), vgpr("LocalReadAddr%s"%tc), "")

    # release resources
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(sgid)
    self.vgprPool.checkIn(rReg)
    self.vgprPool.checkIn(tP["gpr"]["lro"])

    return kStr


  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    kStr = ""

    if kernel["EnableMatrixInstruction"]:
      kStr += self.lraFinalOffsetMFMA(kernel, tP)
    else:
      kStr += self.lraFinalOffsetVALU(kernel, tP)

    return kStr


  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    if tP["isA"]:
      return self.comment1("N/A")
    else:
      return inst("_v_add_co_u32", \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          "vcc", \
          hex(kernel["LdsOffset%s"%tP["tensorChar"]]*tP["bpe"]), \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          " += LdsOffset%s (lower)"%tP["tensorChar"])

  ##############################################################################
  # openShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def openShadowInit(self, kernel):
    kStr = ""
    kStr += self.getNamedLabelDef("ShadowInitStart")
    return kStr

  ##############################################################################
  # closeShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def closeShadowInit(self, kernel):
    kStr = ""
    assert(self.doShadowInit and kernel["PrefetchGlobalRead"])

    kStr += self.checkLastIter(kernel)
    if kernel["SuppressNoLoadLoop"]:
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
      lastIterEnd = self.getLabelNum("LoopEnd%s"%loopChar)
    else:
      lastIterEnd = self.getLabelNum("PrefetchGlobalLastIterEnd")
    kStr += inst("s_cbranch_scc1 label_%04u"\
          % lastIterEnd, \
          "after InitC, skip to end of prefetch last iter b/c numIter==0")

    return kStr


  ##############################################################################
  # Initialize C
  ##############################################################################
  def initC(self, kernel):
    kStr = ""
    if kernel["EnableMatrixInstruction"]:
      pass # we can delay initializing them until before the writeback, where
           # alpha/beta operations and packing are about to be performed
    else:
      # remove the C regs from the pool since we are about to write them here:
      kStr += self.comment("initC: remove C-tile %u-%u from pool"%(self.startVgprValuC, self.startVgprValuC+self.numVgprValuC))
      self.vgprPool.remove(self.startVgprValuC, self.numVgprValuC, "ValuC")

    self.vgprPool.remove(self.startVgprValuA, \
        self.lastValuAB - self.startVgprValuA, "ValuAB")

    for i in range(0, self.numVgprValuC):
      kStr += inst("v_mov_b32", vgpr("ValuC+%u"%i), hex(0), "initC")

    # if using MFMAs, initialize ACC VGPRS as well
    if "MatrixInstM" in kernel:
      kStr = ""
      self.agprPool.remove(0, self.totalAgprs, "ValuC")
      for i in range(0, self.totalAgprs):
        kStr += inst("v_accvgpr_write", "acc%u"%i, hex(0), "init Acc vgprs")

      # TODO: Remove debug code when finished
      # for debug, write 42 and check results
      # write 42 into vgprs
      #for i in range(0, self.totalAgprs):
      #  kStr += inst("v_mov_b32", vgpr("ValuC+%u"%i), "0x"+struct.pack('>f', 42.0).hex(), "write 42")
      # copy over to agprs
      #for i in range(0, self.totalAgprs):
      #  kStr += inst("v_accvgpr_write", "acc%u"%i, vgpr("ValuC+%u"%i), "write 42 into agprs")
      # restore vgprs
      #for i in range(0, self.totalAgprs):
      #  kStr += inst("v_mov_b32", vgpr("ValuC+%u"%i), hex(0), "restore 0")
      #kStr += "s_barrier // debug\n"
      #kStr += "s_waitcnt lgkmcnt(0) & vmcnt(0)\n"
      #kStr += self.bomb()

    if kernel["PersistentKernel"]:
      # Move to next serial wg early since SerialWorkGroupIter is checked in several places below including tail loop which has multiple entry points
      # As a result be aware for much of the loop SerialWorkGroupIter points to the next tile not the current one
      kStr += self.comment1("move to next serial WG")
      kStr += inst("s_add_u32", sgpr("SerialWorkGroupIter"), \
          sgpr("SerialWorkGroupIter"), sgpr("GridNumWorkGroups0"), \
          "Move Serial forward by numworkgroups - will map to new wg0/wg1 later")
      if self.prefetchAcrossPersistent0:
        kStr += self.comment1("save PrevWorkGroup for stores here")
        kStr += inst("s_mov_b32", sgpr("PrevWorkGroup0"), sgpr("WorkGroup0"), "save for store code")
        kStr += inst("s_mov_b32", sgpr("PrevWorkGroup1"), sgpr("WorkGroup1"), "save for store code")

    return kStr


  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  def declareLoopNumIter(self, kernel):
    kStr =""
    if self.unrollIncIsDepthU:
      if kernel["GlobalSplitU"] > 1:
        tmpSgpr = self.getTmpSgpr(2).idx()
        quotient = "UnrollLoopLastIter"
        dividend = self.loopSizeRef(kernel, self.unrollIdx) # sumSize
        divisor = kernel["DepthU"]
        kStr += scalarStaticDivideAndRemainder(quotient, None, dividend, divisor, tmpSgpr, 0)
        tmpSgpr = self.getTmpSgpr(3).idx()
        kStr += self.calculateLoopNumIterGsu(kernel, "UnrollLoopLastIter", tmpSgpr)
        kStr += inst ("s_mul_i32", sgpr("UnrollLoopLastIter"), sgpr("UnrollLoopLastIter"), "DepthU", "scale")

      else:
        kStr += inst ("s_mov_b32", sgpr("UnrollLoopLastIter"), self.loopSizeRef(kernel, self.unrollIdx), "init")

      if kernel["PackSummationDims"]:
        if kernel["GlobalSplitU"]>1:
          kStr += inst ("s_mov_b32", sgpr("GsuNumIter%s"%self.loopChar(kernel,self.unrollIdx)), \
                        sgpr("UnrollLoopLastIter"), "save innermost iters for later unpacking")
        for idx in range(self.otherSummations):
          assert not self.use64bProductOfSums
          kStr += inst ("s_mul_i32", sgpr("UnrollLoopLastIter"), sgpr("UnrollLoopLastIter"), \
                            self.loopSizeRef(kernel, idx), "")

    return kStr


  ##############################################################################
  # Calculate and apply stagger offsets and edge
  # Output: Sets sgpr(StaggerRowMask)
  ##############################################################################
  def declareStaggerParms(self, kernel):

    kStr=""
    tmpSgpr = self.getTmpSgpr(2).idx()
    if self.staggerU:
      # this coud be dynamic?
      if kernel["StaggerUMapping"] == 0:
        staggerInput = sgpr("WorkGroup0")
      elif kernel["StaggerUMapping"] == 1:
        staggerInput = sgpr("WorkGroup1")
      elif kernel["StaggerUMapping"] == 2:
        staggerInput = sgpr("WorkGroup2")
      elif kernel["StaggerUMapping"] == 3:
        # wgSerial = (nwg0*ngw1)*wg2 + (nwg0)*wg1 + wg0
        wgSerial = tmpSgpr
        tmp = tmpSgpr+1
        kStr += inst("s_mul_i32", sgpr(wgSerial), sgpr("NumWorkGroups0"), sgpr("NumWorkGroups1"), \
          "wgSerial = (nwg0*ngw1)*wg2 + (nwg0)*wg1 + wg0")
        kStr += inst("s_mul_i32", sgpr(wgSerial), sgpr(wgSerial), sgpr("WorkGroup2"), "")
        kStr += inst("s_mul_i32", sgpr(tmp), sgpr("NumWorkGroups0"), sgpr("WorkGroup1"), "")
        kStr += inst("s_add_u32", sgpr(wgSerial), sgpr(wgSerial), sgpr(tmp), "")
        kStr += inst("s_add_u32", sgpr(wgSerial), sgpr(wgSerial), sgpr("WorkGroup0"), "")
        staggerInput = sgpr(wgSerial)
      elif kernel["StaggerUMapping"] == 4:
        staggerInput = -1

      kStr += inst("s_and_b32", sgpr("StaggerUIter"), sgpr("OrigStaggerUIter"), \
                    staggerInput, \
                    "Compute actual stagger start for this tile")
      kStr += inst("s_lshl_b32", sgpr("StaggerUIter"), sgpr("StaggerUIter"), \
                kernel["_staggerStrideShift"], "shift by StaggerUStride")
    return kStr

  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  def calculateStagger(self, kernel, tP):

    kStr=""
    tc = tP["tensorChar"]

    if self.staggerU:
      assert (kernel["BufferLoad"])
      staggerTmp = self.getTmpSgpr(1).idx()

      #---
      kStr += self.comment1("SRDs += (StaggerUIter) * GlobalReadIncs%s+%u"% (tc, self.unrollIdx))

      kStr += inst("s_mul_i32", \
        sgpr(staggerTmp),\
        sgpr("StaggerUIter"),\
        sgpr("GlobalReadIncs%s+%u"%(tc, self.unrollIdx)), \
        " stagger byte offset")

      # Amount of bytes to add to get back to start.
      # on the llop iteration which matches StaggerUIter, this offset added instead of GlobalReadInc
      kStr += self.s_mul_i64_i32(sgpr("WrapU%s+0"%tc), sgpr("WrapU%s+1"%tc), \
                    self.loopCounter(kernel, self.unrollIdx), sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                    "Number of bytes accessed by the unroll loop")

      kStr += inst("s_sub_u32", sgpr("WrapU%s+0"%tc),  \
                sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                sgpr("WrapU%s+0"%tc), \
                "remove one iteration")
      kStr += inst("s_subb_u32", sgpr("WrapU%s+1"%tc), \
                sgpr("WrapU%s+1"%tc), \
                0, \
                "remove one iteration")

      kStr += self.incrementSrd(kernel, tP, sgpr(staggerTmp), 0)

      if tP["isB"]:
        # Convert passed in S' to S for easy loop comparison.  S=S-(PGR-1)'
        kStr += inst("s_add_u32", sgpr("StaggerUIter"), sgpr("StaggerUIter"), \
                  (kernel["PrefetchGlobalRead"]+1), \
                  "Subtract (PGR-1); StaggerUIter now contains target iteration to wrap")
    return kStr

  ##############################################################################
  # Remove stagger offset (before tail loop)
  # |          |           |   |
  # |-- S'*I --|
  # |---------- W' --------|-I-|
  #           ^ current SRD pos
  # ^unrollLoopStart           ^tailLoopStart   (in summation0 dimension)

  #
  # S = sgprStaggerUIter = S+(PGR+1)'
  # W = sgprWrapU
  # PGR = kernel["PrefetchGlobalRead"]
  #
  # S' = StaggUIter that is passed into the kernel = -PGR+1+S
  # S'*I is also the global read offset (from unrollLoopStart) at unroll loop exit ?
  # I = GlobalReadIncs
  # W' = W

  # Need to move it to tailLoopStart

  # To compute position where tail loop should start:
  #  = W' - S'*I + I
  #  = W - (S+PGR+1)*I) + I
  #  = W - (S+PGR+1)*I + I
  #  = W - (S+2+PGR)*I
  ##############################################################################
  def removeStagger(self, kernel, tP):
    kStr = ""
    if self.staggerU:
      tc = tP["tensorChar"]
      tmp = self.getTmpSgpr(1).idx()
      # might be able to refactor this to eliminate signed math
      kStr += inst("s_sub_i32", sgpr(tmp), 2+kernel["PrefetchGlobalRead"], \
                  sgpr("StaggerUIter"), "")
      kStr += inst("s_mul_i32", sgpr(tmp), sgpr(tmp),
                    sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                    "start offset S in bytes")
      kStr += inst("s_sub_u32", sgpr(tmp), sgpr(tmp), sgpr("WrapU%s"%tc), "S - WrapU")

      kStr += self.incrementSrd(kernel, tP, sgpr(tmp), 0)

    return kStr


  ##############################################################################
  # Emit code to compute loop iterations for GSU.
  # See same function in KernelWriterSource.py for background explanation
  # This function is used to compute number of loop iters and also
  # for computing the global read increment for GSU case.
  # For multiple summation, the number of loop iterations needs to be reset
  # for each iteration so replicate the code in addr inc and at open of unroll loop

  # tmpSgpr is allocation of at least 3 tmpSgpr

  # Output: SGPR(destName) contains the number of unroll iterations for
  # this workgroup.
  ##############################################################################
  def calculateLoopNumIterGsu(self, kernel, destName, tmpSgpr):
    kStr = ""

    loopCounter = sgpr(destName)
    quotient = destName
    remainder = "GSUSumIdx+1" # numIterPerWgRemainder
    dividend = tmpSgpr+2 # numIterMyWg
    divisor = kernel["GlobalSplitU"]
    kStr += inst("s_mov_b32", sgpr(dividend), loopCounter, "copy for divide IterGsu" )
    kStr += scalarStaticDivideAndRemainder(quotient, remainder, dividend, divisor, tmpSgpr, 1)

    # if gsuSumIdx < numIterPerWgRemainder
    kStr += inst("s_add_u32", sgpr(tmpSgpr), "1", \
                  loopCounter, "tmp<-numIterMyWg+" )
    kStr += inst("s_cmp_lt_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
        "gsuSumIdx < numIterPerWgRemainder" )
    kStr += inst("s_cmov_b32", loopCounter, sgpr(tmpSgpr), "numIterMyWg++ if needed" )

    return kStr


  ##############################################################################
  # Calculate Loop Num Iter
  # loopIdx is the index of the loop (used for contractions with multiple summations)
  # 0 is outermost; self.unrollIdx is the unroll index.
  # -1 is tail loop (used only for the unroll loop)
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx, isPap):
    kStr = ""

    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    loopChar = self.indexChars[loopDim]

    ########################################
    # Tail Loop
    numRemainderSumElements = None
    if tailLoop:
      tmpSgpr = self.getTmpSgpr(4).idx()
      if self.prefetchAcrossPersistent0:
        loopCounterName = "TailLoopCounter"
        if kernel["EnableMatrixInstruction"] and \
           kernel["AssertSummationElementMultiple"] % kernel["MatrixInstK"] != 0:
          numRemainderSumElements = "NumRemainderSumElements"
      else:
        loopCounterName = self.loopCounterName(kernel, loopIdx)
        if kernel["EnableMatrixInstruction"] and \
           kernel["AssertSummationElementMultiple"] % kernel["MatrixInstK"] != 0:
          numRemainderSumElements = "NumRemainderSumElements%s"%loopChar
      kStr += "\n"
      if kernel["SuppressNoLoadLoop"]:
        # If the tail loop is suppressed, then final iterations will have moved the Srd base forward
        # (and also moved back the srd shadow limit) and slammed Limit to 0, so need to 'undo'
        # those increments - see setTailSrd
        assert(kernel["PrefetchGlobalRead"] == 1) #if >1 would need a multiply here
        kStr += inst("s_cmp_eq_u32", sgpr("OrigLoopCounter"), 0, "completely skipped unroll loop?")
        kStr += inst("s_cselect_b32", sgpr(tmpSgpr+0), 0, sgpr("GlobalReadIncsA"), "force to 0?")
        kStr += inst("s_cselect_b32", sgpr(tmpSgpr+1), 0, sgpr("GlobalReadIncsB"), "force to 0?")
        kStr += self.setTailSrd(kernel, self.tPA, sgpr(tmpSgpr+0))
        kStr += "\n"
        kStr += self.setTailSrd(kernel, self.tPB, sgpr(tmpSgpr+1))
        kStr += "\n"
        #kStr += self.bomb()

      kStr += "%s//numIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)%s" \
          % (self.indent, self.unrollChar, self.unrollChar, self.endLine)
      # size % DepthU
      kStr += scalarStaticDivideAndRemainder(tmpSgpr, loopCounterName, "SizesSum+%u"%loopIdx, kernel["DepthU"], tmpSgpr+2, 2)
      loopCounter = sgpr(loopCounterName)
      if numRemainderSumElements:
        kStr += scalarStaticDivideAndRemainder(None, numRemainderSumElements, loopCounter, kernel["MatrixInstK"], tmpSgpr+2, 2)

      if kernel["LocalSplitU"] > 1:
        # (size % DepthU) + LSU - 1
        kStr += inst("s_add_u32", loopCounter, hex(kernel["LocalSplitU"]-1), loopCounter, "(size % DepthU) + LSU - 1" )
        dividend = tmpSgpr+2
        kStr += inst("s_mov_b32", sgpr(dividend), loopCounter, "copy for divide LSU" )
        kStr += scalarStaticDivideAndRemainder( loopCounterName, None, dividend, kernel["LocalSplitU"], tmpSgpr, 0)

      # if GSU numIter=0 if gsuSumIdx != remainder
      if kernel["GlobalSplitU"] > 1:
        kStr += inst("s_cmp_lg_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
            "gsuSumIdx == numIterPerWgRemainder" )
        kStr += inst("s_cmov_b32", loopCounter, hex(0), "numIter=0 if gsuSimIdx!=remainder")

      # if tail numIter == 0 skip altogether
      tailLoopLabelEnd = self.getLabelNum("TailLoopEnd%s"%(loopChar) )
      kStr += inst("s_cmp_eq_u32", loopCounter, \
          hex(0), "numIter%s == 0"%loopChar )
      kStr += inst("s_cbranch_scc1 label_%04u"\
          % tailLoopLabelEnd, \
          "skip to end of tail loop b/c numIter==0")

    ########################################
    # Unrolled Loop
    elif loopIdx == self.unrollIdx:
      loopCounterName = self.loopCounterName(kernel, loopIdx)
      loopCounter = sgpr(loopCounterName)
      if not self.do["PreLoop"]: kStr += ".endif\n"

      sumSize = "SizesSum+%u"%loopIdx
      #sumSize = self.sumSize(kernel, loopIdx)
      if self.unrollIncIsDepthU:
        kStr += inst("s_mov_b32", loopCounter, 0,\
                  "init loop counter, unrollIncIsDepthU mode")

      else:
        # TODO - use named arguments
        tmpSgpr = self.getTmpSgpr(2).idx()
        quotient = loopCounterName
        dividend = sumSize
        divisor = kernel["DepthU"]
        kStr += scalarStaticDivideAndRemainder(quotient, None, dividend, divisor, tmpSgpr, 0)
        # if GSU numIter++ if gsuSumIdx < remainder
        if kernel["GlobalSplitU"] > 1:
          kStr += self.calculateLoopNumIterGsu(kernel, loopCounterName, tmpSgpr)

      kStr += inst("s_mov_b32", sgpr("OrigLoopCounter"), \
                loopCounter, \
                "copy loop counter")
      if self.prefetchAcrossPersistent0 and kernel["ExpandPointerSwap"] and isPap:
        kStr += inst("s_and_b32", sgpr("EvenIterStart"), sgpr("OrigLoopCounter"), \
                  0x1,
                  "save unroll loop start position - copy1 or copy2")
    elif not kernel["PackSummationDims"]:
      # other summation, not unroll loop
      #printExit("no assembly support for 2+ dimensional summation")
      kStr += self.comment("%sother summation, numIter%s = size%s" \
          % (self.indent, loopChar, loopChar))
      loopCounter = self.loopCounter(kernel, loopIdx)
      kStr += inst("s_mov_b32", loopCounter, \
                sgpr("SizesSum+%u"%loopIdx), \
                "init loop counter")

    if not tailLoop:
      # compute element edge:
      problemType = kernel["ProblemType"]
      zpB = next((zpi for zpi in problemType["ZeroPadB"] if zpi[1] == loopDim), None)
      assert zpB==None # not supported

      zpA = next((zpi for zpi in problemType["ZeroPadA"] if zpi[1] == loopDim), None)
      if zpA:
        tc = 'A'
        (freeDim,sumDim) = zpA[:2]
        freeDimChar = globalParameters["IndexChars"][freeDim]
        sumDimChar  = globalParameters["IndexChars"][sumDim]
        elementEdge = "ElementEdge%s%s" % (tc,sumDimChar)
        tmpSgpr = self.getTmpSgpr(2).idx()
        kStr += "\n"
        kStr += self.comment1("ElementEdge%s%s" % (tc, sumDimChar))
        kStr += inst("s_mul_i32", sgpr(elementEdge), \
                  self.strideRef('A', freeDim), \
                  self.sizeRef(freeDim), \
                  "strideFree*sizeFree")

        kStr += inst("s_sub_u32", sgpr(tmpSgpr), \
                   self.sizeRef(sumDim), 1, \
                   "strideSum*(sizeSum-1), step1")
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), \
                  self.strideRef('A', sumDim), \
                  sgpr(tmpSgpr),\
                   "strideSum*(sizeSum-1), step2")

        kStr += inst("s_add_u32", sgpr(elementEdge), \
                  sgpr(elementEdge), \
                  sgpr(tmpSgpr), \
                   "strideFree*sizeFree + strideSum*(sizeSum-1)")

        kStr += inst("s_lshl_b32", \
                  sgpr("ElementEdge%s%s"%(tc, sumDimChar)), \
                  sgpr("ElementEdge%s%s"%(tc, sumDimChar)), \
                  "Bpe%sLog2"%tc, "scale by bpe")

        kStr += inst("s_sub_u32", sgpr(elementEdge), \
                  sgpr(elementEdge), \
                  sgpr("PadStart%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                  "sub PadStart*Bpe")
        kStr += inst("s_sub_u32", sgpr(elementEdge), \
                  sgpr(elementEdge), \
                  sgpr("PadEnd%s%s%s"%(tc, freeDimChar, sumDimChar)), \
                  "Final0: (strideFree*sizeFree - strideSum*(sizeSum-1))*BPE - padStart - padEnd")


        kStr += inst("s_mov_b32", sgpr("Iter"+sumDimChar), 0, "init iterX")

        #assert(self.groOffsetInMacroTile==0)

    return kStr

  ##############################################################################
  # Open Loop
  ##############################################################################
  def openLoop(self, kernel, loopIdx):
    kStr = ""
    # TODO - rewrite this function to simplify control-flow between tail-loop / unroll loop
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
      self.inTailLoop = True
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if not tailLoop:
      kStr += "%s:\n" % self.getNamedLabel("openLoop%s"%loopChar)
    loopLabelBegin = self.getLabelNum("%sLoopBegin%s"%("Tail" if tailLoop else "", loopChar) )
    loopLabelEnd = self.getLabelNum("%sLoopEnd%s"%("Tail" if tailLoop else "", loopChar) )

    # is numIter at least 1? otherwise skip to end
    # PGL needs a skip-check here if not bufferload
    # If kernel["SuppressNoLoadLoop"] we don't have a special loop for the 'last iter'
    loopCounter = self.loopCounter(kernel, loopIdx)
    if tailLoop:
      if self.prefetchAcrossPersistent0:
        loopCounter = "TailLoopCounter"
      endCounter = 0
    elif kernel["PrefetchGlobalRead"]:
      if kernel["SuppressNoLoadLoop"]:
        endCounter =  0
      else:
        endCounter = 1
    else:
      endCounter =  0

    if tailLoop:
      kStr += inst("s_cmp_le_u32", \
          loopCounter, \
          hex(endCounter), \
          "LoopCounter%s < EndCounter"%(loopChar) )
      kStr += inst("s_cbranch_scc1 label_%04u"%loopLabelEnd, \
          "don't enter Loop%s"%loopChar )

      kStr += inst("s_mov_b32", sgpr("OrigLoopCounter"), 0, \
          "repurpose to count each localRead increment")

      # LSU not all threads will do summation
      if kernel["LocalSplitU"] > 1:
        tmpSgpr = self.getTmpSgpr(2).idx()
        kStr += self.comment("apply exec mask for LSU")
        tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
        dummy = self.vgprPool.checkOut(1,"dummy")
        sgId = self.vgprPool.checkOut(1,"sgId")
        divisor = kernel["SubGroup0"]*kernel["SubGroup1"]
        kStr += vectorStaticDivide(sgId, "Serial", divisor, tmpVgpr, tmpSgpr)
        numIter = self.vgprPool.checkOut(1,"numIter")
        kStr += inst("v_mov_b32", vgpr(numIter), sgpr("SizesSum+0"), "sizeU to vgpr")
        divisor = kernel["DepthU"]
        kStr += vectorStaticDivideAndRemainder(dummy, numIter, numIter, divisor, tmpVgpr, tmpSgpr)
        self.vgprPool.checkIn(dummy)
        #kStr += dump(vgpr(sgId))
        #kStr += dump(vgpr(numIter))
        kStr += inst("v_cmpx_lt_u32", "vcc", \
            vgpr(sgId), vgpr(numIter), "sgId < numIter")
        self.vgprPool.checkIn(tmpVgpr)
        #self.tailNumIter = numIter
        #self.vgprPool.checkIn(numIter)
        # thread is active is sgId < numIter % LocalSplitU

      # begin loop
      kStr += "label_%04u:%s" % (loopLabelBegin, self.endLine)

      # LSU mask for this iteration
      if kernel["LocalSplitU"] > 1:
        kStr += inst("v_cmpx_lt_u32", "vcc", \
            vgpr(sgId), vgpr(numIter), "sgId < numIter")
        kStr += inst("_v_add_co_u32", vgpr(sgId), "vcc", hex(kernel["LocalSplitU"]), \
            vgpr(sgId), "sgId+=LSU")
        self.vgprPool.checkIn(sgId)
        self.vgprPool.checkIn(numIter)
        #kStr += dump(vgpr(sgId))

    else: # not tailloop:

      if loopIdx == self.unrollIdx:
        if self.unrollIncIsDepthU:
          kStr += inst("s_cmp_ge_u32", \
              loopCounter, \
              sgpr("UnrollLoopLastIter"), \
              "LoopCounter%s > EndCounter"%(loopChar) )
        else:
          kStr += inst("s_cmp_le_u32", \
              loopCounter, \
              hex(endCounter), \
              "LoopCounter%s < EndCounter"%(loopChar) )
        kStr += inst("s_cbranch_scc1 label_%04u"%loopLabelEnd, \
            "don't enter Loop%s"%loopChar )

      if self.prefetchAcrossPersistent and kernel["ExpandPointerSwap"]:
        kStr += inst("","compare if odd-iter return")
        #kStr += inst("s_cbranch_scc1", self.getLabelTarget("LoopCopy2"), "start at oddIter?")

      kStr += "label_%04u:%s" % (loopLabelBegin, self.endLine)

      if loopIdx != self.unrollIdx:
        # reset LRO since these may have changed due to odd-iter exit ?
        if kernel["PrefetchGlobalRead"]:
          kStr += self.comment1("openLoop - reset LRO for possible odd-iter exit")
          kStr += self.localReadResetOffsets(kernel, self.tPA)
          kStr += self.localReadResetOffsets(kernel, self.tPB)

    return kStr


  ##############################################################################
  # Close Loop
  # finalLoop : final unroll loop
  ##############################################################################
  def closeLoop(self, kernel, loopIdx, finalLoop):
    kStr = ""
    #kStr += self.indent + self.syncStr + self.endLine
    #kStr += "s_endpgm\n"
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      loopLabelBegin = self.getLabelNum("TailLoopBegin%s"%(loopChar) )
      loopLabelEnd = self.getLabelNum("TailLoopEnd%s"%(loopChar) )
      loopLabelEndOddExit = self.getLabelNum("TailLoopEnd%s_oddexit"%(loopChar) )
      if self.prefetchAcrossPersistent0:
        loopCounter = "TailLoopCounter"
      else:
        loopCounter = self.loopCounter(kernel, loopIdx)

      unrollInc      = 1
      KinInnerUnroll = kernel["InnerUnroll"]
      if kernel["EnableMatrixInstruction"]:
        unrollInc      *= kernel["MatrixInstK"]
        KinInnerUnroll *= kernel["MatrixInstK"]
      if kernel["AssertSummationElementMultiple"] % KinInnerUnroll == 0:
        unrollInc *= kernel["InnerUnroll"]

      kStr += self.comment("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))

      kStr += inst("s_sub_i32", \
          loopCounter, \
          loopCounter, \
          hex(unrollInc), \
          "dec counter%s (tailLoop)"%(loopChar) )

      # Track # LDS reads?
      kStr += inst("s_add_u32", \
        sgpr("OrigLoopCounter"), \
        sgpr("OrigLoopCounter"), \
        hex(unrollInc),
        "inc counter%s"%(loopChar) )

      endCounter = 0
      kStr += inst("s_cmp_le_i32", \
          loopCounter, \
          hex(endCounter), \
        "counter%s==%d"%(loopChar,endCounter) )
    else: # not tailloop
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      loopLabelBegin = self.getLabelNum("LoopBegin%s"%(loopChar) )
      loopLabelEnd = self.getLabelNum("LoopEnd%s"%(loopChar) )
      loopLabelEndOddExit = self.getLabelNum("LoopEnd%s_oddexit"%(loopChar) )
      loopCounter = self.loopCounter(kernel, loopIdx)
      kStr += self.comment("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))


      if self.unrollIncIsDepthU and loopIdx==self.unrollIdx:
        assert (not kernel["SuppressNoLoadLoop"]) # not accounting for end-of-loop iteration change here in deprecated mode

        kStr += inst("s_cmp_ge_u32", \
            loopCounter, \
            sgpr("UnrollLoopLastIter"), \
          "counter%s==0"%(loopChar) )
      else:
        kStr += inst("s_sub_u32", \
            loopCounter, loopCounter, \
            1, \
            "dec counter%s"%(loopChar) )

        # If PrefetchGlobalRead=1 the loads in the loop prefetch next macro-tile
        # For the final trip through the unroll loop we need to ensure those loads stay in bounds.

        # One technique is to create a copy of the unroll loop with all loads removed.
        # However buffer load doesn't need this loop copy since we OOB loads can be supressed by buffer limit hardware
        # So can do one more iteration (endCounter==0) in the main unroll loop, and adjust the pointer
        # increments appropriately.
        # Also sum idx other than unroll always compare against 0 (there is no PGR to account for)
        if kernel["PrefetchGlobalRead"] and not kernel["SuppressNoLoadLoop"] and loopIdx == self.unrollIdx:
          endCounter = 1
        else:
          endCounter = 0

        kStr += inst("s_cmp_eq_i32", \
            loopCounter, \
            hex(endCounter), \
          "counter%s==%d"%(loopChar,endCounter) )

    if not finalLoop:
      # just an exit check, else fall through to the next loop copy
      kStr += inst("s_cbranch_scc1 label_%04u"%(loopLabelEndOddExit), "exit Loop%s"%loopChar )
    else: #finalLoop:
      kStr += inst("s_cbranch_scc0 label_%04u"%loopLabelBegin, \
          "restart Loop%s"%(loopChar ))

      if tailLoop:
        if kernel["PersistentKernel"] or len(kernel["ProblemType"]["IndicesSummation"]) > 1:
          # recover the 'damage' done to LRO:
          stmp = self.getTmpSgpr(1).idx()
          for tP in [self.tPA, self.tPB]:
            tc     = tP["tensorChar"]
            LdsPad = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
            inc    = kernel["LocalSplitU"]*(kernel["MacroTile%u"%tP["tensorIdx"]]+LdsPad)*tP["bpe"]
            kStr += inst("s_mov_b32", sgpr(stmp), inc, "tailloop lds offset")
            kStr += inst("s_mul_i32", sgpr(stmp), sgpr("OrigLoopCounter"), sgpr(stmp), "scale by mul")
            kStr += inst("v_sub_u32", vgpr("LocalReadAddr%s"%tc), vgpr("LocalReadAddr%s"%tc), sgpr(stmp), "remove lro damage")
      elif loopIdx == self.unrollIdx:
        oddIterCode = Code.Module()
        if not kernel["SuppressNoLoadLoop"] and kernel["ExpandPointerSwap"]:
          # In this case we kept the 'no-load' loop which has LDS offsets assuming first bank of LDS
          # if we exit the main loop at an odd iter - need to swap LDS read pointers
          # so the ds_reads read from the 'high' buffer of LDS
          oddIterCode.addComment1("Select high bank of LDS")
          oddIterCode.addText(self.localReadSwapOffsets(kernel, False, self.tPA))
          oddIterCode.addText(self.localReadSwapOffsets(kernel, False, self.tPB))

        if oddIterCode.count():
          kStr += inst("s_branch label_%04u"%loopLabelEnd, \
              "exit unroll loop%s (and skip oddexit)"%(loopChar ))
        kStr += "label_%04u: // unroll loop odditer exit\n" % (loopLabelEndOddExit)
        kStr += str(oddIterCode)

      kStr += "label_%04u:%s" % (loopLabelEnd, self.endLine)

    # restore all threads
    if tailLoop and kernel["LocalSplitU"] > 1:
      kStr += self.comment("restore full exec mask")
      fullExec = self.getTmpSgpr(2).idx()
      kStr += inst("s_mov_b64", sgpr(fullExec,2), \
          "0xFFFFFFFFFFFFFFFF", "restore all threads active")
      kStr += inst("s_or_saveexec_b64",  sgpr(fullExec,2), sgpr(fullExec,2), "full mask -> exec" )
    return kStr

  ##############################################################################
  ##############################################################################
  def openLoopCopy(self, kernel, lc):
    kStr = ""

    kStr += self.getLabelDef("LoopCopy%u"%(lc+1) )


    return kStr

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self, kernel):
    kStr = ""

    kStr += "%s:\n" % self.getNamedLabel("Summation_End")

    kStr += self.comment1("endSummation: add vgpr %u...%u to pool" % \
            (self.startVgprValuA, self.lastVgprForReads))

    self.vgprPool.add(self.startVgprValuA, \
        self.lastVgprForReads - self.startVgprValuA, "endSummation")

    lastRegTag=None
    for i in range(self.lastPostLoopSgpr, self.sgprPool.size()):
      regTag = self.sgprPool.pool[i].tag
      if regTag != lastRegTag:
        lastRegTag = regTag
        if self.sgprPool.pool[i].status == RegisterPool.statusInUse:
          kStr += self.undefineSgpr(regTag)

    if self.db["InitVgpr"] & 0x2:
      #kStr += self.vgprPool.initTmps(self.initVgprValue)
      kStr += self.vgprPool.initTmps(self.initVgprValue,start=0, stop=100)
    if 0:
      for i in range(0,16+1):
         #kStr += inst("v_mov_b32", vgpr(21), hex(self.initVgprValue), "hack tmp in pool")
         kStr += inst("v_mov_b32", vgpr(21), vgpr(21), "hack tmp in pool")

    # this doesn't seem to do anything - not being aggressive with lastPostLoopSgpr
    if self.db["InitSgpr"] & 0x2:
      kStr += self.sgprPool.initTmps(self.initSgprValue)

    if self.db["ConservativeWaitCnt"] & 0x10:
      kStr += "s_barrier // debug" + self.endLine
      kStr += "s_waitcnt lgkmcnt(0) & vmcnt(0)" + self.endLine
      if self.archCaps["SeparateVscnt"]:
        kStr += "s_waitcnt_vscnt null, 0" + self.endLine

    if kernel["SuppressNoLoadLoop"]:
      kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "wait for all summation activity")
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

    # copy accumulated C from agpr to vgpr
    if "MatrixInstM" in kernel:
      #for i in range(0, self.totalAgprs):
      #  kStr += inst("v_accvgpr_read_b32", vgpr("ValuC+%u"%i), "acc%u"%i, "copy areg to vreg")
      #TODO avoid s_nop if its possible
      instCycles = kernel["MatrixInstM"] // 2 # 32x32 is 64 cycles, 16x16 is 32 cycles, 4x4 is 8 cycles
      kStr += "s_nop %u\n" % instCycles
      kStr += self.MapAcctoArchRegs(kernel,option=0)

    return kStr


  ##############################################################################
  # MFMA Iteration
  ##############################################################################
  def mfmaIter(self, kernel, m, innerUnroll, tail=False):

    imod = Code.Module("mi")
    shiftK = Code.Module("shiftK")

    # calculate constant
    numRegisters     = kernel["ProblemType"]["DataType"].numRegisters()
    loopCounterName  = self.loopCounterName(kernel, self.unrollIdx)
    accs_per_wave    = kernel["MatrixInstM"] * kernel["MatrixInstN"] * kernel["MatrixInstB"] / globalParameters["WavefrontWidth"]
    dividerFortidInK = kernel["MatrixInstN"] * kernel["MatrixInstB"]
    numMIInput       = kernel["ProblemType"]["DataType"].numMIInput()
    vgprPerInput     = int(numMIInput * numRegisters)
    shiftPerElement  = int(numRegisters * 32)
    elementPerVgpr   = int(1 / numRegisters)
    s_nop            = 0

    # alloc vgpr
    kReg    = self.vgprPool.checkOut(1,"kReg") # remainder
    abReg   = self.vgprPool.checkOut(vgprPerInput,"abReg")
    tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
    dummy   = self.vgprPool.checkOut(1,"dummy")

    # alloc sgpr
    tmpSgpr = self.getTmpSgpr(3).idx()

    if (numRegisters == 0.5) and ((kernel["UnrollMajorLDSA"] == False) or (kernel["UnrollMajorLDSB"] == False)):
      s_nop = 2

    # handle multiple K element in MFMA instruction
    if tail and kernel["MatrixInstK"] > 1:
      shiftK.addCode(vectorStaticRemainder(dummy, kReg, "Serial", globalParameters["WavefrontWidth"], tmpVgpr, tmpSgpr))
      shiftK.addCode(vectorStaticDivide(kReg, kReg, dividerFortidInK, tmpVgpr, tmpSgpr))
      shiftK.addCode(staticMultiply(vgpr(kReg), vgpr(kReg), numMIInput, sgpr(tmpSgpr)))

      # replace 0 for differnet thread
      shiftK.addCode(inst("v_cmp_ge_i32", sgpr(tmpSgpr, 2), vgpr(kReg), sgpr(loopCounterName), "check K index >= Size L"))
      for bk in range(0, vgprPerInput):
        for a in range(0, kernel["MIWaveTile"][0]):
          for iui in range(0, innerUnroll):
            aStr  = vgpr("ValuA_X%u_I%u+%u+%u" % (m, iui, a*vgprPerInput, bk), 1)
            shiftK.addCode(inst("v_cndmask_b32", aStr, aStr, hex(0), sgpr(tmpSgpr, 2), "set 0 if K_idx >= sizeL"))
        for b in range(0, kernel["MIWaveTile"][1]):
          for iui in range(0, innerUnroll):
            bStr  = vgpr("ValuB_X%u_I%u+%u+%u" % (m, iui, b*vgprPerInput, bk), 1)
            shiftK.addCode(inst("v_cndmask_b32", bStr, bStr, hex(0), sgpr(tmpSgpr, 2), "set 0 if K_idx >= sizeL"))

      # replace 0 for same thread
      if numMIInput > 1:
        shiftK.addCode(inst("v_sub_u32",    vgpr(kReg), sgpr(loopCounterName), vgpr(kReg), "get distance between size and k index"))
        shiftK.addCode(inst("v_cmp_lt_i32", sgpr(tmpSgpr,2), vgpr(kReg), numMIInput, "set partial 0 if distance less than input per thread"))
        shiftK.addCode(inst("s_and_b32",    sgpr(tmpSgpr+2), sgpr(loopCounterName), numMIInput-1, "get inputs for edge thread"))
        shiftK.addCode(inst("s_sub_u32",    sgpr(tmpSgpr+2), numMIInput, sgpr(tmpSgpr+2), "use shift to fill 0 for outside element"))
        shiftK.addCode(inst("s_lshl_b32",   sgpr(tmpSgpr+2), sgpr(tmpSgpr+2), log2(shiftPerElement), "use shift to fill 0 for outside element"))
        for a in range(0, kernel["MIWaveTile"][0]):
          for iui in range(0, innerUnroll):
            aStr  = vgpr("ValuA_X%u_I%u+%u" % (m, iui, a * vgprPerInput), vgprPerInput)
            shiftK.addCode(inst("v_lshlrev_b%u" % (vgprPerInput*32), vgpr(abReg, vgprPerInput), sgpr(tmpSgpr+2), aStr, ""))
            for bk in range(0, vgprPerInput):
              aStr  = vgpr("ValuA_X%u_I%u+%u+%u" % (m, iui, a * vgprPerInput, bk), 1)
              shiftK.addCode(inst("v_cndmask_b32", aStr, aStr, vgpr(abReg+bk), sgpr(tmpSgpr, 2), ""))
        for b in range(0, kernel["MIWaveTile"][1]):
          for iui in range(0, innerUnroll):
            bStr     = vgpr("ValuB_X%u_I%u+%u" % (m, iui, b*vgprPerInput), vgprPerInput)
            shiftK.addCode(inst("v_lshlrev_b%u" % (vgprPerInput*32), vgpr(abReg, vgprPerInput), sgpr(tmpSgpr+2), bStr, ""))
            for bk in range(0, vgprPerInput):
              bStr  = vgpr("ValuB_X%u_I%u+%u+%u" % (m, iui, b*vgprPerInput, bk), 1)
              shiftK.addCode(inst("v_cndmask_b32", bStr, bStr, vgpr(abReg+bk), sgpr(tmpSgpr, 2), ""))

      s_nop = 2

    if s_nop != 0:
      imod.addCode("s_nop %u\n" % (s_nop - 1))
    else:
      imod.addCode("")

    for iui in range(0, innerUnroll):
      for b in range(0, kernel["MIWaveTile"][1]):
        for a in range(0, kernel["MIWaveTile"][0]):
          accIdx   = b * kernel["MIWaveTile"][0] + a
          accStart = accIdx * accs_per_wave
          accEnd   = accStart + accs_per_wave - 1
          aStr     = vgpr("ValuA_X%u_I%u+%u" % (m, iui, a*vgprPerInput), vgprPerInput)
          bStr     = vgpr("ValuB_X%u_I%u+%u" % (m, iui, b*vgprPerInput), vgprPerInput)

          imod.addCode("v_mfma_f32_%ux%ux%u%s a[%u:%u], %s, %s, a[%u:%u]%s" \
                     % (kernel["MatrixInstM"], kernel["MatrixInstN"], kernel["MatrixInstK"], kernel["ProblemType"]["DataType"].toNameAbbrev(),
                        accStart, accEnd, aStr, bStr, accStart, accEnd, self.endLine))

    # release register
    self.vgprPool.checkIn(kReg)
    self.vgprPool.checkIn(abReg)
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(dummy)

    mfmaMod = Code.Module("mfmaCode")
    mfmaMod.addCode(shiftK)
    mfmaMod.addCode(imod)

    return mfmaMod

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  def macIter(self, kernel, bufferIdx, iuiCount, useMacro):
    if not self.do["MAC"]: return ""
    imod = Code.Module("macIter_X%u_I%u"%(bufferIdx, iuiCount))

    if kernel["ProblemType"]["DataType"].isHalf():
      imod.addInst(".align32 8, 0xbf800001", "align v_pk_fma")   # Align v_pk_fma instructions used in MAC_ blocks

    if kernel["InnerUnroll"] > 1 and iuiCount==1:
      # This it tail-loop case where we just want one IUI,
      imod.addText("MAC_%ux%u_X%u_OneIUI" % (kernel["ThreadTile0"],kernel["ThreadTile1"], bufferIdx))
    else:
      if useMacro:
        imod.addText("MAC_%ux%u_X%u" % (kernel["ThreadTile0"],kernel["ThreadTile1"], bufferIdx))
      else:
        # Generate MAC calls inline
        imod.addText(self.defineMACs(kernel, bufferIdx, kernel["InnerUnroll"]))

    return imod

  ##############################################################################
  # MAC Iteration -alternate version
  ##############################################################################
  def macCode(self, kernel, bufferIdx, iuiCount):
    if not self.do["MAC"]: return ""
    imod = Code.Module("macIter_X%u_I%u"%(bufferIdx, iuiCount))

    if kernel["ProblemType"]["DataType"].isHalf():
      imod.addInst(".align32 8, 0xbf800001", "align v_pk_fma")   # Align v_pk_fma instructions used in MAC_ blocks

    doOnce = False
    beAggressive = kernel["AggressivePerfMode"]
    macIdx = 0

    # half precision
    if kernel["ProblemType"]["DataType"].isHalf():
      for blockB in range(0, kernel["ThreadTile1"]//2):
        for blockA in range(0, kernel["ThreadTile0"]//2):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # bf16 precision
    elif kernel["ProblemType"]["DataType"].isBFloat16():
      for blockB in range(0, kernel["ThreadTile1"]//2):
        for blockA in range(0, kernel["ThreadTile0"]//2):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # integer i8
    elif kernel["ProblemType"]["DataType"].isInt8x4():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # single precision
    elif kernel["ProblemType"]["DataType"].isSingle():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True
          if macIdx == kernel["PerformanceWaitLocation"]:
            imod.addCode(Code.WaitCnt(self.version, kernel["PerformanceWaitCount"],"extra wait for performance"))
          if macIdx == kernel["PerformanceSyncLocation"]:
            imod.addInst("s_barrier ","extra barrier for performance")
          macIdx += 1

    # double precision
    elif kernel["ProblemType"]["DataType"].isDouble():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # single precision complex
    elif kernel["ProblemType"]["DataType"].isSingleComplex():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True

    # double precision complex
    elif kernel["ProblemType"]["DataType"].isDoubleComplex():
      for blockB in range(0, kernel["ThreadTile1"]):
        for blockA in range(0, kernel["ThreadTile0"]):
          imod.addCode(Code.MacInst(kernel,blockA,blockB,bufferIdx,iuiCount))
          if beAggressive and not doOnce:
            imod.addInst("s_setprio ","1","Raise priority while processing macs")
            doOnce = True
            
    else:
      printExit("Assembly doesn't support %s" % kernel["ProblemType"]["DataType"])

    if beAggressive and doOnce:
      imod.addInst("s_setprio ","0","Reset priority after macs")

    return imod
 
  ##############################################################################
  # At Least 1 Unroll
  # prefetch means this is in the prefetch code, either before unroll loop
  # or in the PAP code.
  # isPap means this is the PAP iteration, need to adjust the loop exit
  # isOptNLL : this is for the store-interleaved NLL optimization
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel, prefetch, isPap, isOptNLL):
    kStr = ""
    if prefetch:
      kStr += self.checkLastIter(kernel)
      if not isPap:
        if self.doShadowInit:
          kStr += inst("s_cbranch_scc1 %s"\
              % self.getNamedLabel("ShadowInitStart"), \
              "skip to ShadowInitStart iter b/c numIter==0")
        else:
          loopChar = self.indexChars[ \
              kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
          labelName = "label_%04d" % self.getLabelNum("LoopEnd%s"%loopChar)
          kStr += inst("s_cbranch_scc1 %s" % labelName,
              "skip to unrollLoop end loop%s iter b/c numIter==0" % loopChar)
      else:
        kStr += inst("s_cbranch_scc1 label_%04u"\
            % self.getLabelNum("SkipPrefetchAcrossPersistent"), \
            "skip prefetch loads since numIter==0")
    elif isOptNLL:
      skipOptNLL = self.getNamedLabel("OptNLL_End")
      tmpSgpr = self.getTmpSgpr(2).idx()

      kStr += self.checkIsBetaZero(kernel, tmpSgpr, skipOptNLL)

      # check alpha
      if self.do["ApplyAlpha"]:
        if kernel["ProblemType"]["DataType"].isHalf():
          if not kernel["ProblemType"]["HighPrecisionAccumulate"] or kernel["ProblemType"]["ComputeDataType"].isHalf():
            kStr += inst("s_mov_b32", sgpr(tmpSgpr), "0x3c003c00", "Packed alpha==1.0")
            kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), sgpr(tmpSgpr), "alpha == 1.0?")
          else: # HPA
            kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), "1.0", "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["DataType"].isBFloat16():
          kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), "1.0", "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["DataType"].isInt8x4():
          kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), "1.0", "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["DataType"].isSingle():
            #kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
          kStr += inst("s_cmp_eq_u32", sgpr("Alpha"), "1.0", "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["DataType"].isDouble():
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+0), 0x00000000, "Low part of double 1.0")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), "0x3ff00000", "High part of double 1.0")
          kStr += inst("s_cmp_eq_u64", sgpr("Alpha",2), sgpr(tmpSgpr,2), "Alpha == 1.0 ?")

        elif kernel["ProblemType"]["DataType"].isSingleComplex():
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+0), "1.0", "Real part of 1.0")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), "0.0", "Imaginary part of 1.0")
          kStr += inst("s_cmp_eq_u64", sgpr("Alpha",2), sgpr(tmpSgpr,2), "Alpha == 1.0 ?")
          
        elif kernel["ProblemType"]["DataType"].isDoubleComplex():
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+0), "0x00000000", "lsb of real part of 1.0")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), "0x3ff00000", "msb of real part of 1.0")
          kStr += inst("s_cmp_eq_u64", sgpr("Alpha",2), sgpr(tmpSgpr,2), "Alpha.real == 1.0 ?")
          kStr += inst("s_cbranch_scc0 %s"%skipOptNLL, "branch if alpha.real != 1")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+0), "0x00000000", "lsb of imag part of 0.0")
          kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), "0x00000000", "msb of imag part of 0.0")
          kStr += inst("s_cmp_eq_u64", sgpr("Alpha+2",2), sgpr(tmpSgpr,2), "Alpha.imag == 0.0 ?")

        kStr += inst("s_cbranch_scc0 %s"%skipOptNLL, "branch if alpha != 1")
        kStr += "\n"

      kStr += self.checkIsEdge(kernel, tmpSgpr, skipOptNLL)
      kStr += "\n"

      # Check tail loop required:
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][self.unrollIdx]]
      kStr += scalarStaticDivideAndRemainder(tmpSgpr, tmpSgpr+1, "SizesSum+%u"%self.unrollIdx, \
                kernel["DepthU"], tmpSgpr+2, 2)
      kStr += inst("s_cmp_eq_u32", sgpr(tmpSgpr+1), \
          hex(0), "numIter%s == 0"%loopChar )
      kStr += inst("s_cbranch_scc0 %s"%skipOptNLL, \
          "skip if tail loop required")

      # OptNLL has no tail loop so can reclaim some regs here -
      # we need these to do the address calcs for the stores, etc
      # save the vgprPool for generating the normal path.
      self.savedVgprPool = deepcopy(self.vgprPool)

      added = [] # track registers added to pool
      if kernel["PrefetchGlobalRead"]:
        if not kernel["DirectToLdsA"]:
          added.append(self.vgprPool.addRange(self.startVgprG2LA, \
              self.startVgprG2LA+self.numVgprG2LA-1, "startOptNLL"))
          added.append(self.vgprPool.addRange(self.startVgprLocalWriteAddressesA, \
                       self.startVgprLocalWriteAddressesA, "startOptNLL"))
        if not kernel["DirectToLdsB"]:
          added.append(self.vgprPool.addRange(self.startVgprG2LB, \
              self.startVgprG2LB+self.numVgprG2LB-1, "startOptNLL"))
          added.append(self.vgprPool.addRange(self.startVgprLocalWriteAddressesB, \
                       self.startVgprLocalWriteAddressesB, "startOptNLL"))

      if kernel["BufferLoad"]:
        added.append(self.vgprPool.addRange(self.startVgprGlobalReadOffsetA, \
            self.startVgprGlobalReadOffsetB, "startOptNLL"))
      else:
        added.append(self.vgprPool.addRange(self.startVgprGlobalReadAddressesA, \
            self.startVgprGlobalReadAddressesB, "startOptNLL"))
      kStr += self.comment("reclaim VGPRS: " + ", ".join(added))

    return kStr

  ##############################################################################
  ##############################################################################
  def closeSumAtLeastUnroll(self, kernel, prefetch, isOptNLL):
    kStr = ""
    if not prefetch:
      if isOptNLL:
        kStr += self.comment1("Stores for OptNLL")
        self.getNamedLabel("Summation_End")

        # add copyback if required
        if kernel["EnableMatrixInstruction"]:
          instCycles = kernel["MatrixInstM"] // 2 # 32x32 is 64 cycles, 16x16 is 32 cycles, 4x4 is 8 cycles
          kStr += "s_nop %u\n" % instCycles
          ##for i in range(0, self.totalAgprs):
          ##  kStr += inst("v_accvgpr_read_b32", vgpr("ValuC+%u"%i), "acc%u"%i, "copy areg to vreg")
          kStr += self.MapAcctoArchRegs(kernel,option=0) # begin locking c-tile vregs from register pool

        # perhaps could work with LSU>1 by adding other indices here, but not tested
        # had to move it below MapAcctoArchRegs(), else computeStoreVgprs() will check out c-tile vregs as  
        # temp vregs which soon gets utilized by c-tile and guarantees aliasing troubles
        assert (kernel["LocalSplitU"] == 1)
        kStr += self.notLocalSplitUGlobalWriteIndices(kernel)

        # add stores for opt NLL
        (fullVw, elements) = self.notLocalFullTileElements(kernel, False)
        # optimized NLL has edge=beta=atomic=False by design
        ss = self.StoreState(self, kernel, fullVw, edge=False, beta=False, \
            atomic=False, elements=elements)

        # TODO - maybe this should be encapulated in the SS setup code
        tmpSgpr = self.getTmpSgpr(2).idx()
        if ss.optSingleColVgpr:
          tmpVgpr= None # don't need tmps for other uses?
        else:
          tmpVgpr= self.vgprPool.checkOut(3, "NLL address tmps")
        assert(len(kernel["PackedC1IndicesX"]) == 1)

        ss.setupStoreElementsForBatch(kernel, fullVw, elements, None, isOptNLL=True)
        if not kernel["EnableMatrixInstruction"]:
          kStr += inst("_v_add_lshl_u32", \
              vgpr(ss.sharedColVgprs), \
              vgpr(self.cinRowPtr), \
              vgpr(self.coord0), \
              hex(log2(self.bpeCexternal)), \
             "NLL: init cb addr <-  cinRowStart + coord0, scaled by BPE")

        if kernel["ProblemType"]["DataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
          vgprBf16Temp = self.vgprPool.checkOut(4,"vgprBf16Mask")
          vgprBf16Mask = vgprBf16Temp + 1
          vgprFp32Nan = vgprBf16Temp + 2
          vgprBf16Inc = vgprBf16Temp + 3
          kStr += inst("v_mov_b32", vgpr(vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" )
          kStr += inst("v_mov_b32", vgpr(vgprFp32Nan), "0x7fff0000", "fp32 Nan" )
          kStr += inst("v_mov_b32", vgpr(vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" )

        for elementIdx in range(0, len(elements)):
          kStr += self.comment("store element %d : %s" % (elementIdx, str(elements[elementIdx])))
          # pack stores, beta and non-beta reach here:
          for vi in range(0, fullVw):
            sumIdxV = ss.elementSumIdx[elementIdx] + vi
            if kernel["ProblemType"]["DataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
              assert (fullVw % 2 == 0)
              kStr += inst("v_cvt_f16_f32", vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), "convert C to fp16" )
              if vi%2 == 1:
                d = ss.elementSumIdx[elementIdx] + vi//2
                kStr += inst("v_pack_b32_f16", vgpr(d), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr("ValuC+%u"%sumIdxV), "Pack with neighbor")
            elif kernel["ProblemType"]["DataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
              assert (fullVw % 2 == 0)
              kStr += inst("v_cmp_u_f32", sgpr(tmpSgpr,2), vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), "check Nan" )
              kStr += inst("v_bfe_u32", vgpr(vgprBf16Temp), vgpr("ValuC+%u"%sumIdxV), "16", "1", "Non-Nan case: store lsb of bf16" )
              kStr += inst("v_add3_u32", vgpr(vgprBf16Temp), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprBf16Temp), vgpr(vgprBf16Inc), "Non-Nan case: add lsb and the increment for rounding" )
              kStr += inst("v_cndmask_b32", vgpr("ValuC+%u"%sumIdxV), vgpr(vgprBf16Temp), vgpr(vgprFp32Nan), sgpr(tmpSgpr,2), "" )
              if vi%2 == 0:
                kStr += inst("v_lshrrev_b32", vgpr("ValuC+%u"%sumIdxV), "16", vgpr("ValuC+%u"%sumIdxV), "convert C to bf16" )
              elif vi%2 == 1:
                d = ss.elementSumIdx[elementIdx] + vi//2
                kStr += inst("v_and_or_b32", vgpr(d), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprBf16Mask), vgpr("ValuC+%u"%(sumIdxV-1)), "pack two bf16 to dword")

          addrCalc = ss.elementAddr[elementIdx]
          sumIdx = ss.elementSumIdx[elementIdx]

          kStr += addrCalc.emitAddressSetupCode(kernel, ss, tmpVgpr01=tmpVgpr, tmpS01=tmpSgpr, \
                              edge=False, beta=False, atomic=False, mask=None, elementIdx=elementIdx)
          kStr += self.addStore(kernel, ss, addrCalc, sumIdx, tmpSgpr, edge=False)

        if kernel["ProblemType"]["DataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
          self.vgprPool.checkIn(vgprBf16Temp)
        if tmpVgpr:
          self.vgprPool.checkIn(tmpVgpr)

        kStr += "\n"
        kStr += str(self.functionEnd(kernel, False))
        #kStr += inst("s_branch %s"%summationEnd, "skip the OptNLL")

        label = self.getNamedLabel("OptNLL_End")
        kStr += "%s:%s" % (label, self.endLine)
        del ss # release store state before swapping back saved vgpr pool
               # so the destructor of ss can refer to the correct vgpr pool
      else:
        label = self.getLabelNum("PrefetchGlobalLastIterEnd")
        kStr += "label_%04u:%s" % (label, self.endLine)
    if self.savedVgprPool != None:
      # in case pool size in current path is larger than pool size in main path
      # and it will miss allocate vgpr since allocating vgpr is based on pool size in main path
      oldSize = self.savedVgprPool.size()
      newSize = self.vgprPool.size()
      if newSize > self.savedVgprPool.size():
        for i in range(oldSize-1,newSize):
          self.savedVgprPool.pool.append(self.savedVgprPool.Register(self.savedVgprPool.statusAvailable,"restore vgprPool"))
      self.vgprPool = self.savedVgprPool # restore vgprPool before alternate path 
      self.savedVgprPool = None
    return kStr

  ##############################################################################
  ##############################################################################
  # incLower must be constant or SGRP unsigned value
  def incrementSrd(self, kernel, tP, incLower, incUpper, checkShadowLimitCopy=True):
    tc = tP["tensorChar"]
    kStr = ""

    kStr += inst("s_add_u32", \
         sgpr("Srd%s+0"%(tc)), \
         sgpr("Srd%s+0"%(tc)), \
         incLower, \
        "gra SRD += inc(lower)" )
    kStr += inst("s_addc_u32 ", \
         sgpr("Srd%s+1"%(tc)), \
         sgpr("Srd%s+1"%(tc)), \
         incUpper, \
        "gra SRD += inc(upper)" )

    # also have to move the boundary since we change the base
    # so less buffers to the edge:
    if self.use64bShadowLimit:
      kStr += inst("s_sub_u32", \
          sgpr("ShadowLimit%s+0"%tc), \
          sgpr("ShadowLimit%s+0"%tc), \
           incLower, \
            "limit -= inc)")
      kStr += inst("s_subb_u32", \
          sgpr("ShadowLimit%s+1"%tc), \
          sgpr("ShadowLimit%s+1"%tc), \
           incUpper, \
            "limit -= inc)" )
      if checkShadowLimitCopy:
        kStr += inst("s_cmp_eq_u32", sgpr("ShadowLimit%s+1"%tc), 0, "are we within 2^32?")
        kStr += inst("s_cmov_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "Move shadow to real if we are within 2^32")
    else:
      kStr += inst("s_sub_u32", \
           sgpr("Srd%s+2"%(tc)), \
           sgpr("Srd%s+2"%(tc)), \
           incLower, \
            "limit -= inc)" )

    return kStr


  ##############################################################################
  ##############################################################################
  # incLower must be constant or SGRP unsigned value
  def setTailSrd(self, kernel, tP, incLower):
    # In SuppressNoLoadLoop, the final loop iteration moves the SRD base forward
    # and the ShadowLimit backwards by one extra 'click' of GlobalReadIncs[AB].
    # Note the ShadowLimit may become negative - for example edge tiles where the
    # increment is > tile width.
    # The SuppressNoLoadLoop mode also forces the SRD limit to 0 on the final iteration.
    # The code here undoes the final click step by moving the base backwards and the
    # limit forwards (reading from the ShadowLimit).
    # It only works if use64bShadowLimit is enabled (since this enables use of the ShadowLimit)

    tc = tP["tensorChar"]
    kStr = ""
    incUpper = 0

    kStr += inst("s_sub_u32 ", \
         sgpr("Srd%s+0"%(tc)), \
         sgpr("Srd%s+0"%(tc)), \
         incLower, \
        "gra SRD -= inc(lower)" )
    kStr += inst("s_subb_u32 ", \
         sgpr("Srd%s+1"%(tc)), \
         sgpr("Srd%s+1"%(tc)), \
         incUpper, \
        "gra SRD -= inc(upper)" )

    # using Shadow limit here which only works with 64-bit PBC:
    assert(self.use64bShadowLimit)

    kStr += inst("s_add_u32", \
        sgpr("ShadowLimit%s+0"%tc), \
        sgpr("ShadowLimit%s+0"%tc), \
         incLower, \
          "limit -= inc)")
    kStr += inst("s_addc_u32", \
        sgpr("ShadowLimit%s+1"%tc), \
        sgpr("ShadowLimit%s+1"%tc), \
         incUpper, \
          "limit -= inc)" )
    kStr += inst("s_cmp_eq_u32", sgpr("ShadowLimit%s+1"%tc), 0, "are we within 2^32?")
    kStr += inst("s_cmov_b32", sgpr("Srd%s+2"%tc), sgpr("ShadowLimit%s+0"%tc), "Move shadow to real if we are within 2^32")

    return kStr

  ##############################################################################
  # Global Read: Increment A/B
  # loopIdx is summation idx:
  #   self.unrollIdx, or an idx from 0..NumIndicesSummation
  # prefetchIndex is >0 (1...PrefetchGlobalRead) if this increment follows a
  #   global prefetch or 0 otherwise
  # incs is number of increments to perform
  ##############################################################################
  def globalReadIncrement(self, kernel, imod, loopIdx, tP, prefetchIndex, incs=1):
    if not self.do["GlobalInc"]: return ""
    tc = tP["tensorChar"]
    loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]

    imod.addComment1("global read inc %s loop%s"%(tc,loopChar))

    if kernel["BufferLoad"]:
      # TODO - does this handle N-dim tensors correctly?
      #if tP["isB"]:
      #  kStr += inst("s_mov_b32", sgpr("OffsetB"), sgpr("SrdB+0"), "hack to save")
      if self.staggerU and loopIdx == self.unrollIdx:
        # add a wrap increment, if needed:
        incLower = self.getTmpSgpr(3).idx()
        incUpper = incLower + 1
        tmpS =    incLower + 2
        if prefetchIndex:
          imod.addInst("s_add_u32", sgpr(tmpS), self.loopCounter(kernel, self.unrollIdx), prefetchIndex, "remove pf(%u)"%prefetchIndex)
          imod.addInst("s_cmp_eq_u32",  sgpr("StaggerUIter"), sgpr(tmpS), "Is this wrapIter? (pf)")
        else:
          imod.addInst("s_cmp_eq_u32",  self.loopCounter(kernel, self.unrollIdx), \
                    sgpr("StaggerUIter"), "Is this the wrapIter?")
        #kStr += self.assert_scc_is_1() # break at the wrap iteration
        imod.addInst("s_cselect_b32", sgpr(incLower), sgpr("WrapU%s+0"%tc), sgpr("GlobalReadIncs%s+%u"%(tc,self.unrollIdx)), \
                    "incLower <- ?")
        imod.addInst("s_cselect_b32", sgpr(incUpper), sgpr("WrapU%s+1"%tc), 0,
                    "incUpper <- ?")
        imod.addText(self.incrementSrd(kernel, tP, sgpr(incLower), sgpr(incUpper), checkShadowLimitCopy=True))
      else:
        if loopIdx != self.unrollIdx:
          incUpper = sgpr(self.getTmpSgpr(1).idx())
          # GRO may be negative for other summation if stride-other < stride-unroll.
          imod.addInst("s_ashr_i32", incUpper, sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), 31, "sign-extend")
        else:
          incUpper = 0 # GRO is positive for loop unroll
        imod.addText( self.incrementSrd(kernel, tP, sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), incUpper))
    else:
      graIdx = 0
      #for perp in range(0, tP["nrp"]):
      #  for para in range(0, tP["nrc"]):
      #    for s in range(0, tP["nrcv"]):
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
              if self.globalReadIncsUseVgpr:
                imod.addInst("_v_add_co_u32 ", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    vgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], 2*loopIdx)), \
                    "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
                imod.addInst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    vgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], 2*loopIdx)), \
                    "vcc", \
                    "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
              else:
                imod.addInst("_v_add_co_u32 ", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    sgpr("GlobalReadIncs%s+%u"%(tP["tensorChar"], loopIdx)), \
                    "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
                imod.addInst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    0,
                    "vcc", \
                    "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
              graIdx += self.rpga
      #kStr += dump(vgpr("GlobalReadAddrA+0"))
      #kStr += dump(vgpr("GlobalReadAddrA+1"))
      #kStr += "s_endpgm\n"


  def globalReadIncrementAB(self, kernel, loopIdx, prefetchIndex, incs=1):
    imod = Code.Module("globalReadIncrementAB%s")
    problemType = self.kernel["ProblemType"]
    unrollLoopCounter = self.loopCounter(kernel, self.unrollIdx)

    incCodeA = imod.addCode(Code.Module("globalReadIncrementA"))
    incCodeB = imod.addCode(Code.Module("globalReadIncrementB"))

    if self.unrollIncIsDepthU and loopIdx==self.unrollIdx:
      loopCounter = self.loopCounter(kernel, self.unrollIdx)
      incCodeA.addInst("s_add_u32",
                   loopCounter, loopCounter,
                   "DepthU",  "increment psdIter")

    if loopIdx==self.unrollIdx and kernel["PackSummationDims"] and self.actualSummationLoops==1:
      incSize = 2 if self.use64bPackSumOffset else 1
      tmpSgpr = self.getTmpSgpr(3 + 2*incSize + (3 if kernel["GlobalSplitU"]>1 else 0)).idx()
      inc ={}
      inc['A'] = tmpSgpr + 3
      inc['B'] = inc['A'] + incSize
      gsuMagic = inc['B'] + incSize

      psdPackedBits = "DepthU" if prefetchIndex>0 else unrollLoopCounter
      incCodeA.addComment1("extract indices here from %s"%psdPackedBits)
      for os in reversed(range(problemType["NumIndicesSummation"])):
        sumDim  = problemType["IndicesSummation"][os]
        sumChar = self.indexChars[sumDim]
        firstIter = (os==problemType["NumIndicesSummation"]-1)
        lastIter  = (os==0)

        incCodeA.addComment1("extract index %s"%sumChar)

        if not lastIter:
          if os==self.unrollIdx and kernel["GlobalSplitU"] > 1:
            # GSU divides the first loop counter size by some amount
            size = "GsuNumIter%s"%sumChar
          else:
            size = "Size%s"%sumChar

          if firstIter:
            psdPackedBits2 = psdPackedBits
          else:
            psdPackedBits2 = sgpr(tmpSgpr+2)
            incCodeA.addInst("s_mov_b32", psdPackedBits2, psdPackedBits, "copy psdPackedBits")

          if os==self.unrollIdx and kernel["GlobalSplitU"] > 1:
            # compare GSUA
            # cmov into temps for Size,Abit,Shift
            # divide and go.
            # need more temps for this, need divide routine to take 3 parms
            incCodeA.addInst("s_cmp_lt_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
                "gsuSumIdx < numIterPerWgRemainder" )
            incCodeA.addInst("s_cselect_b32", sgpr(gsuMagic+0), sgpr("MagicNumberSize%s_GsuRemainder"%sumChar),
                              sgpr("MagicNumberSize%s"%sumChar), "Use alternate divisor")
            incCodeA.addInst("s_cselect_b32", sgpr(gsuMagic+1), sgpr("MagicAbitSize%s_GsuRemainder"%sumChar),
                              sgpr("MagicAbitSize%s"%sumChar), "Use alternate divisor")
            incCodeA.addInst("s_cselect_b32", sgpr(gsuMagic+2), sgpr("MagicShiftSize%s_GsuRemainder"%sumChar),
                              sgpr("MagicShiftSize%s"%sumChar), "Use alternate divisor")
            incCodeA.addText(self.scalarMagicDivExplicit(tmpSgpr, psdPackedBits,
                              magicNumber=gsuMagic+0, magicAbit=gsuMagic+1, magicShift=gsuMagic+2))
          else:
            incCodeA.addText(self.scalarMagicDiv(tmpSgpr, psdPackedBits, sumChar))

          # TODO-64
          incCodeA.addInst("s_mul_i32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+0), sgpr(size), "remainder step 1")
          incCodeA.addInst("s_sub_u32", sgpr(tmpSgpr+1), psdPackedBits2, sgpr(tmpSgpr+1), "remainder step 2")
          iterX=sgpr(tmpSgpr+1)
        elif firstIter and lastIter:
          # just one iter, use loop counter directly not remainder
          iterX = psdPackedBits
        else:
          iterX=sgpr(tmpSgpr+0)


        for tc in ('A','B'):
          zp = next((zpi for zpi in problemType["ZeroPad"+tc] if zpi[1] == sumDim), None)
          if zp:
            incCodeA.addInst("s_mov_b32", sgpr("Iter"+sumChar), iterX, "save iterX")

        # update psdOffset. Inputs:
        #   - tmpSgpr+0== packedBits, and must be preserved for next iteration
        #   - iterX, number of iterations for this dim.  Used in A/B increment loop below
        for tc in ('A','B'):
          assert(not self.use64bPackSumOffset)
          if firstIter:
            #incCodeA.addText(self.s_mul_u64_u32(inc{'A'}+0, inc{'A'}+1, tmpSgpr+1, sgpr["GlobalReadIncs%s+%d"]))
            incCodeA.addInst("s_mul_i32", sgpr(inc[tc]), iterX, sgpr("GlobalReadIncs%s+%d"%(tc,os)),
                              "psdOffset%s += scale iter%s"%(tc,sumChar))
          else:
            incCodeA.addInst("s_mul_i32", sgpr(tmpSgpr+2), iterX, sgpr("GlobalReadIncs%s+%d"%(tc,os)), "Scale iter%s"%sumChar)
            incCodeA.addInst("s_add_u32", sgpr(inc[tc]+0), sgpr(inc[tc]+0), sgpr(tmpSgpr+2), "psdOffset%s += scale iter%s"%(tc,sumChar))
            #incCodeA.addText(self.s_mul_u64_u32(tmp+0, inc{'A'}+1, tmpSgpr+1, sgpr["GlobalReadIncsA"]))

          psdPackedBits = sgpr(tmpSgpr+0)

        if 0 and lastIter:
          incCodeA.addText(self.assert_ne(sgpr("LoopCounterM"), 8))

      assert(kernel["BufferLoad"])

      incCodeA.addText("\n");
      incCodeA.addComment1("Reset and increment SRDs")
      for tc in ('A','B'):
        incCodeA.addInst("s_mov_b32", sgpr("Srd%s+0"%tc), sgpr("InitialSrd%sBase+0"%tc), "restore base")
        incCodeA.addInst("s_mov_b32", sgpr("Srd%s+1"%tc), sgpr("InitialSrd%sBase+1"%tc), "restore base")
        if self.use64bShadowLimit:
          incCodeA.addInst("s_mov_b32", sgpr("ShadowLimit%s+0"%tc), sgpr("InitialSrd%sLimit+0"%tc), "restore shadow limit")
          incCodeA.addInst("s_mov_b32", sgpr("ShadowLimit%s+1"%tc), sgpr("InitialSrd%sLimit+1"%tc), "restore shadow limit")
          assert(0) # not tested, would maybe need to restore base too if limit 0
        else:
          incCodeA.addInst("s_mov_b32", sgpr("Srd%s+2"%tc), sgpr("InitialSrd%sLimit"%tc), "restore limit")


      # TODO - this skips over the stagger-u wrap codes
      incCodeA.addText("\n");
      incCodeA.addText(self.incrementSrd(kernel, self.tPA, sgpr(inc['A']), sgpr(inc['A']+1) if self.use64bPackSumOffset else 0))
      incCodeA.addText("\n");
      incCodeA.addText(self.incrementSrd(kernel, self.tPB, sgpr(inc['B']), sgpr(inc['B']+1) if self.use64bPackSumOffset else 0))
    else:
      self.globalReadIncrement(kernel, incCodeA, loopIdx, self.tPA, prefetchIndex, incs)
      self.globalReadIncrement(kernel, incCodeB, loopIdx, self.tPB, prefetchIndex, incs)
    return imod


  ##############################################################################
  # Global Read:
  # globalReadGuardK is called for loads in the tail loop
  # Must ensure each load is in bounds - either using buffer bounds 
  # or exec-mask checks.
  ##############################################################################
  def globalReadGuardK(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    problemType = self.kernel["ProblemType"]
    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth

    ########################################
    # Calculate Max Addr
    ########################################
    maxAddrSgpr = self.getTmpSgpr(4).idx()
    tmpSgpr = maxAddrSgpr + 2

    if not kernel["BufferLoad"]:
      kStr += self.comment1("flat addressing - max read address = size[n] * stride[n-1]")
      dim = len(tP["ia"])-1 # dim
      sizeIdx = tP["ia"][dim]
      sizeIdxIsSum = sizeIdx in kernel["ProblemType"]["IndicesSummation"]
      if sizeIdxIsSum:
        sizeIdx -= kernel["ProblemType"]["NumIndicesC"]
      # TODO-multiply by largest stride
      kStr += self.s_mul_u64_u32(sgpr(maxAddrSgpr+0), sgpr(maxAddrSgpr+1),  \
                  sgpr("Sizes%s+%u"%("Sum" if sizeIdxIsSum else "Free", sizeIdx)),  \
                  sgpr("Stride%s%s"%(tc, self.indexChars[tP['ia'][-1]])), \
                  "64b tensor%s size in elements"%tc)
      kStr += inst("s_lshl_b64", \
        sgpr(maxAddrSgpr,2), \
        sgpr(maxAddrSgpr,2), \
        hex(log2(tP["bpe"])), "<- tensor%s size in bytes"%tc)

      kStr += inst("s_add_u32", \
          sgpr(maxAddrSgpr+0), \
          sgpr(self.sgprs["AddressA"] if tP["isA"] else self.sgprs["AddressB"]), \
          sgpr(maxAddrSgpr+0), \
          "prepend address lower")
      kStr += inst("s_addc_u32", \
          sgpr(maxAddrSgpr+1), \
          sgpr((self.sgprs["AddressA"] if tP["isA"] else self.sgprs["AddressB"])+1), \
          sgpr(maxAddrSgpr+1), \
          "prepend address upper")
      # sgpr->vgpr
      maxAddrVgpr = self.vgprPool.checkOut(2, "maxAddrVgpr")
      kStr += inst("v_mov_b32", vgpr(maxAddrVgpr+0), sgpr(maxAddrSgpr+0), "sgpr->vgpr")
      kStr += inst("v_mov_b32", vgpr(maxAddrVgpr+1), sgpr(maxAddrSgpr+1), "sgpr->vgpr")

      # full exec mask
      fullExec = tmpSgpr
      kStr += inst("s_mov_b64", sgpr(fullExec,2), \
          "0xFFFFFFFFFFFFFFFF", "to restore all threads active")
      bpeVgpr = self.vgprPool.checkOut(1, "bpeVgpr")
      kStr += inst("v_mov_b32", vgpr(bpeVgpr), hex(tP["bpe"]), "bpe")

      # can remove this?
      zeroVgpr = self.vgprPool.checkOut(1,"zeroVgpr")
      kStr += inst("v_mov_b32", vgpr(zeroVgpr), hex(0), "zero")

    extraFields = ""
    if tP["NonTemporal"]%2==1:
      extraFields += " glc"
    if tP["NonTemporal"]//2==1:
      extraFields += " slc"
    if kernel["DirectToLds%s"%tc]:
      extraFields += " lds"

    directToLdsLoads = 0
    # print("tc={}, nrp={}, nrpv={}, nrc={}, nrcv/nrcvpi={}, zeroPad={}, sgprforGRO={}".format(tc, tP["nrp"], tP["nrpv"], tP["nrc"], tP["nrcv"]//tP["nrcvpi"], problemType["ZeroPad%s"%tc], kernel["UseSgprForGRO"]))
    if problemType["ZeroPad%s"%tc]:
      addrV = self.vgprPool.checkOut(1,"addrV")
    loopCnt = -1
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
            i = sPara + (tP["nrcv"] // tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            loopCnt += 1
            graIdx = i * self.rpgo if kernel["BufferLoad"] else i * self.rpga
            g2lIdx = i * loadWidth

            destVgprHi = None

            r = 0
            # for each component in vector
            while r < loadWidth*self.bpr//tP["bpe"]:
              numElementsPerLoad = 1
              if kernel["ProblemType"]["DataType"].isHalf() or \
                 kernel["ProblemType"]["DataType"].isBFloat16():
                if tP["glvw"]>1 and kernel["AssertSummationElementMultiple"] % 2 == 0:
                # Pack two FP16 values into a single load dword x2
                  numElementsPerLoad = 2
                elif self.archCaps["HasEccHalf"]:
                  # In some cards, loading half types into register will zero out
                  # the other half. Therefore we need to load into a separate register 
                  # then pack 2 registers into one
                  destVgprHi = self.vgprPool.checkOut(1, 'destVgprHi') 
                regIdx = r // 2
              elif kernel["ProblemType"]["DataType"].isInt8x4() or \
                   kernel["ProblemType"]["DataType"].isSingle():
                regIdx = r
              elif kernel["ProblemType"]["DataType"].isDouble() or \
                   kernel["ProblemType"]["DataType"].isSingleComplex():
                regIdx = r*2
              elif kernel["ProblemType"]["DataType"].isDoubleComplex() :
                regIdx = r*4
              else:
                printWarning("DataType unsupported")
              kStr += self.comment1("g2l=%u, load component %u"%(g2lIdx, r))

              offset = 0

              if kernel["BufferLoad"]:
                # Use buffer limit to stay in-bounds - the limit was set to edge when SRD initialized
                # and each increment of SRD base in the unroll loop does a corresponding decrement
                # of the srd limit - so base+limit stays constant and also points at maximum
                # element that should be accessed.
                if kernel["_UseSgprForGRO"]:
                  offsetVgpr = "GlobalReadOffset%s+0"%(tc)
                  if graIdx==0:
                    soffset = "0"
                  else:
                    soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))
                else:
                  offsetVgpr = "GlobalReadOffset%s+%u"%(tc, graIdx)
                  soffset = "0"

                if problemType["ZeroPad%s"%tc]:
                  codeMod = Code.Module("guardZeroPad%u"%loopCnt)
                  offsetVgpr = self.guardZeroPad(kernel, tP, codeMod, offsetVgpr, soffset, tmpSgpr, addrV, perp, sPerp, para, sPara)
                  kStr += str(codeMod)


                if kernel["DirectToLds%s"%tc]:
                  if directToLdsLoads != 0:
                    ldsInc = kernel["NumThreads"]*4
                    kStr += inst("s_add_u32", "m0", "m0", ldsInc, \
                        "Move LDS write address to next line" )
                  directToLdsLoads+=1

                  # Assembler expects a destination VGPR even though not written
                  destVgpr=0
                else:
                  destVgpr="G2L%s+%u+%u"%(tc, g2lIdx, regIdx)

                offset = r * tP["bpe"]
                hi16 = 0
                if kernel["ProblemType"]["DataType"].isHalf() or \
                   kernel["ProblemType"]["DataType"].isBFloat16():
                  if numElementsPerLoad==2:
                    # Pack two FP16 values into a single load dword x2
                    r += 1 # skip next element since we loaded 2X here
                    comment="load packed 2X half buffer value"
                  elif not kernel["DirectToLds%s"%tc]:
                    hi16=loopCnt%2 if tP["glvw"]==1 else r%2
                    comment="load half buffer value"
                else:
                  comment="load one buffer value"

                bpl = numElementsPerLoad*self.bpeAB # bytesPerLoad

                kStr += self.chooseGlobalRead(True, \
                          bpl, destVgpr=destVgprHi if (hi16 and destVgprHi != None) else destVgpr, \
                          addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 4), \
                          soffset=soffset, offset=offset, \
                          extraFields=extraFields, \
                          hi16=hi16, \
                          comment=comment).toStr()
                # print("  bpl={}, destVgpr={}, soffset={}, offset={}, hi16={}".format(bpl, destVgpr, soffset, offset, hi16))

              else: # Not buffer load, ie 'flat' load
                # mask if current address if in bounds
                kStr += inst("_v_cmpx_lt_u64", "vcc", \
                    vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), \
                    vgpr(maxAddrVgpr,2), \
                    "addr < maxAddr")
                hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and r%2==1
                destVgpr="G2L%s+%u+%u"%(tc, g2lIdx, regIdx)
                # load one element from address
                kStr += self.chooseGlobalRead(False, \
                          self.bpeAB, destVgpr=destVgprHi if (hi16 and destVgprHi != None) else destVgpr, \
                          addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                          soffset=0, offset=0, \
                          extraFields=extraFields, \
                          hi16=hi16, \
                          comment="load one flat value").toStr()

                # restore full exec mask
                kStr += inst("s_or_saveexec_b64", "vcc", sgpr(fullExec,2), \
                    "all threads active")

                # increment address by 1 element (BPE)
                kStr += inst("_v_add_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    vgpr(bpeVgpr), "gra += 1 (lower)")
                kStr += inst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    vgpr(zeroVgpr), \
                    "vcc", \
                    "gra += 1 (upper)")
              # pack 2x registers containing 16-bit each into one 32-bit packed half-dword
              if destVgprHi != None and r % 2 == 1:
                kStr += "s_waitcnt vmcnt(0)\n"
                kStr += "v_or_b32 " + vgpr(destVgpr) + ", " + vgpr(destVgpr) + ", " + vgpr(destVgprHi) + " // HasEccHalf: pack\n"
              r += 1 # next component (for half)
              if destVgprHi != None: self.vgprPool.checkIn(destVgprHi)
            # end R loop

    if self.db["ConservativeWaitCnt"] & 0x1:
        kStr += "s_barrier // debug\n"
        kStr += "s_waitcnt lgkmcnt(0) & vmcnt(0)\n"
        if self.archCaps["SeparateVscnt"]:
          kStr += "s_waitcnt_vscnt null, 0\n"
        kStr += "s_barrier // debug\n"
        #kStr += self.assert_lt(vgpr("Serial"), 64) # examine second wavefront

    if problemType["ZeroPad%s"%tc]:
      self.vgprPool.checkIn(addrV)

    # TODO - can remove one of these m0 restores if A and B both TLU
    if kernel["DirectToLds%s"%tP["tensorChar"]]:
      kStr += inst("s_mov_b32", "m0", \
          hex(kernel["LdsNumElements"] * tP["bpe"]), \
          "Restore LDS clamp at %u bytes"%(kernel["LdsNumElements"] * tP["bpe"]))

    if not kernel["BufferLoad"]:
      self.vgprPool.checkIn(maxAddrVgpr)
      self.vgprPool.checkIn(bpeVgpr)
      self.vgprPool.checkIn(zeroVgpr)

    return kStr


  ##############################################################################
  # guardZeroPad
  # add to code module the code to guard subsequent load
  # Inputs:
  #  - offsetVgpr contains GlobalReadOffset
  # Outputs:
  #  - addrV is temp vgpr, returns the guarded address (OOB lanes return -1)
  ##############################################################################
  def guardZeroPad(self, kernel, tP, codeMod, offsetVgpr, soffset, tmpSgpr, addrV, perp, sPerp, para, sPara):
    tc = tP["tensorChar"]
    zps = [zpr for zpr in self.zeroPadRegs[tc].values() if zpr.isMatch(perp, sPerp, para, sPara)]
    for i, zpr in enumerate(zps):
      #zpTmp = tmpSgpr + i + 1
      (freeDim,sumDim) = zpr.zp[:2]
      sumChar = self.indexChars[sumDim]

      codeMod.addComment1("guardZeroPad: "+zpr.regName)
      iterX = "Iter"+sumChar if kernel["PackSummationDims"] else "LoopCounter"+sumChar
      codeMod.addInst("s_mul_i32", sgpr(tmpSgpr), sgpr(iterX), \
                        self.strideRef(tc,sumDim), "LoopCounterZp*strideSum")
      codeMod.addInst("s_lshl_b32", sgpr(tmpSgpr), sgpr(tmpSgpr), \
                        "Bpe%sLog2"%tc, "")
      if soffset != "0":
        assert (soffset == "0") # need to add to scalar above.  Can't happen with UseSgprForGRO=0
        codeMod.addInst("s_add_u32", sgpr(tmpSgpr), sgpr(tmpSgpr), soffset, "add soffset ")

      codeMod.addInst("_v_add_u32", vgpr(addrV), vgpr(zpr.regName), sgpr(tmpSgpr), \
                        "<- GRO + scaled elementCounter")

      cmpDest = "vcc" if i==0 else sgpr(tmpSgpr,2) # first one writes vcc
      codeMod.addInst("v_cmp_ge_u32", cmpDest, vgpr(addrV), \
                        sgpr("ElementEdge%s%s"%(tc,sumChar)), \
                        "loopCounter*strideSum >= ElementEdge ?")

      if i>0:
        codeMod.addInst("s_or_b64", "vcc", "vcc", sgpr(tmpSgpr,2),"combine elementEdge masks")

      if i==len(zps)-1:
        codeMod.addInst("v_cndmask_b32", vgpr(addrV), vgpr(offsetVgpr), -1, "vcc", \
                          "Set addresses in pad to large OOB value")

      #if soffset != "0":
      #  assert(sumChar == self.unrollChar) # don't think we need this for non-unroll dims
      #  #codeMod.addText(self.assert_ne(sgpr("WorkGroup0"),1))
      #codeMod.addText(self.bomb())

    return addrV

  ##############################################################################
  # DirectToLds M0 update: Do It A/B
  ##############################################################################
  def directToLdsM0Update(self, kernel, mode, tP):
    tc = tP["tensorChar"]
    imod = Code.Module("directToLdsM0Update%s_%u"%(tc,mode))
    DtldsModule = imod.addCode(Code.Module("dtls_offset%s"%tP["tensorChar"]))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return imod
    if kernel["DirectToLds%s"%tP["tensorChar"]]:
      # DirectToLds only enabled for TLU=1 cases, where the registers are directly copied into LDS
      # for cases both A&B are DTLS, updating m0 for each GlobalRead requires instruction schedule 
      # along with global reads
      assert (kernel["LocalWriteUseSgpr%s"%tc])
      if kernel["ExpandPointerSwap"]:
        DtldsModule.addInst("s_add_u32", "m0", sgpr("LocalWriteAddr%s"%tc), \
                      tP["localWriteSwapByteOffset"], "m0 <- LDS write address")
      else:
        DtldsModule.addInst("s_mov_b32", "m0", sgpr("LocalWriteAddr%s"%tc), "m0 <- LDS write address")

    return imod



  ##############################################################################
  # Global Read: Do It A/B
  ##############################################################################
  def globalReadDo(self, kernel, mode, tP):
    tc = tP["tensorChar"]
    problemType = self.kernel["ProblemType"]
    imod = Code.StructuredModule("globalReadDo%s_%u"%(tc,mode))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return imod

    # sizeK % LOCAL_DEPTHU
    guardK = (mode==2)

    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth # load width in elements?
    bpl = self.bpeAB * tP["glvw"] # bytes per load
    ldsOffset = 0
    instOffset = 0

    loopIdx = self.unrollIdx # TODO - does this handle multiple summation indices?
    if kernel["SuppressNoLoadLoop"]:
      if mode==1 and tP["isA"]:
        imod.header.addInst("s_cmp_eq_i32", \
              self.loopCounter(kernel, loopIdx), \
              "%u"% 1, \
              "%s"%"is this the last iteration")
        imod.header.addInst("s_cmov_b32", \
              sgpr("SrdA+2"), \
              0,
              "Set limit to 0 for last iteration")
        imod.header.addInst("s_cmov_b32", \
              sgpr("SrdB+2"), \
              0,
              "Set limit to 0 for last iteration")

    tmpSgpr = self.getTmpSgpr(2+len(problemType["ZeroPad%s"%tc])).idx()
    # TODO - clean up here:
    # +0,+1 - general purpose tmp. i + 2 is the offset for zero-pad index X
    #for i, zp in enumerate(problemType["ZeroPad%s"%tc]):
    #  zpTmp = tmpSgpr + i + 2
    #  imod.header.addComment1("Zeropad check:")
    #  (freeDim,sumDim)= zp[:2]
    #  sumChar = self.indexChars[sumDim]
    #  loopIdx = problemType["IndicesSummation"].index(sumDim)
    #  # TODO - fix for GSU, need LOCAL_DEPTHU*GSUp
    #  if guardK:
    #    imod.header.addInst("s_sub_u32", sgpr(zpTmp), self.sizeRef(freeDim), \
    #      self.loopCounter(kernel,loopIdx), "compute elementCounter%s, step2"%(sumChar))
    #  else:
    #    imod.header.addInst("s_mul_i32", sgpr(zpTmp), self.loopCounter(kernel,loopIdx), \
    #      "DepthU", "compute elementCounter%s, step1"%(sumChar))
    #    imod.header.addInst("s_sub_u32", sgpr(zpTmp), self.sizeRef(freeDim), \
    #      sgpr(zpTmp), "compute elementCounter%s, step2"%(sumChar))
    #  imod.header.addInst("s_mul_i32", sgpr(zpTmp), self.strideRef(tc,freeDim), sgpr(zpTmp), "scale by stride")
    #  imod.header.addInst("s_lshl_b32", sgpr(zpTmp), sgpr(zpTmp), log2(self.bpeAB), "scale by bpe")

    if tP["isA"] and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
      imod.header.addText(self.comment1("before DirectToLds load, ensure prior ds_reads have finished"))
      imod.header.addText(self.syncThreads(kernel))


    if guardK:
      imod.middle.addText(self.globalReadGuardK(kernel, tP))
      return imod

    # else not-guardK below:

    extraFields = ""
    if tP["NonTemporal"]%2==1:
      extraFields += " glc"
    if tP["NonTemporal"]//2==1:
      extraFields += " slc"
    if kernel["DirectToLds%s"%tc]:
      extraFields += " lds"

    directToLdsLoads = 0

    loopCnt = -1
    if problemType["ZeroPad%s"%tc]:
      addrV = self.vgprPool.checkOut(1,"addrV")
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
            i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            loopCnt += 1
            graIdx = i * self.rpgo if kernel["BufferLoad"] else i * self.rpga
            g2lIdx = i * loadWidth
            # Each load may contains a small bundle of instructions, package them together in loadModule:
            loadModule = Code.Module("load%u"%loopCnt)
            imod.middle.addCode(loadModule)

            if kernel["BufferLoad"]:
              if graIdx==0 or not kernel["_UseSgprForGRO"]:
                offsetVgpr= "GlobalReadOffset%s+%u"%(tc, graIdx)
                soffset = "0"
              else:
                offsetVgpr= "GlobalReadOffset%s+0"%(tc)
                soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))

              if problemType["ZeroPad%s"%tc]:
                codeMod = Code.Module("guardZeroPad%u"%loopCnt)
                offsetVgpr = self.guardZeroPad(kernel, tP, codeMod, offsetVgpr, soffset, tmpSgpr, addrV, perp, sPerp, para, sPara)
                loadModule.addCode(codeMod)

              if kernel["DirectToLds%s"%tc]:

                # Get offset (for checking, see comment below) and comment:
                (checkOffset, iDummy, comment) = \
                    self.calculateLdsWriteOffset(perp, para, sPerp, sPara, kernel, tP, 0)

                # Direct to LDS always writes consecutive LDS locations at m0 + 4 * TidInWave
                # Therefore we double-check here to ensure the desired LDS write offset
                # is moving at NumThreads*4.  This should already be guaranteed since
                # we only use direct-to-lds for non-transpose cases but double-check here.
                ldsInc = kernel["NumThreads"]*4
                #print ("checkOffset=", checkOffset, "ldsOffset=", ldsOffset, "ldsInc=", ldsInc)


                if directToLdsLoads != 0:
                  LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
                  if not kernel["UseInstOffsetForGRO"]:
                    ldsOffset = ldsInc + (ldsInc//256) * LdsPad * tP["bpe"]
                    loadModule.addInst("s_add_u32", "m0", "m0", hex(ldsOffset), \
                      "Move LDS write address to next line" )
                  else:
                    instOffset += ldsInc + (ldsInc//256) * LdsPad * tP["bpe"]
                directToLdsLoads+=1
                #ldsOffset += ldsInc
                destVgpr=0
              else:
                destVgpr="G2L%s+%u"%(tc, g2lIdx)

              loadModule.addCode( self.chooseGlobalRead(kernel["BufferLoad"], \
                        bpl, destVgpr=destVgpr, \
                        addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 4), \
                        soffset=soffset, offset=instOffset, \
                        extraFields=extraFields, \
                        hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and loopCnt%2==1, \
                        comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp)))
              #print "IM=", type(imod.instList[-1]), imod.instList[-1], 
            else: # not buffer load
              # load one element from address
              loadModule.addCode( self.chooseGlobalRead(False, \
                        bpl, \
                        destVgpr="G2L%s+%u"%(tc, g2lIdx), \
                        addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                        soffset=0, offset=0, \
                        extraFields=extraFields, \
                        hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and loopCnt%2==1, \
                        comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp )))

    if self.db["ConservativeWaitCnt"] & 0x1:
        imod.footer.addInst( "s_barrier", "debug")
        imod.footer.addInst( "s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "conservative wait")
        if self.archCaps["SeparateVscnt"]:
          imod.footer.addInst( "s_waitcnt_vscnt", "null", "0", "stores")
        imod.footer.addInst( "s_barrier", "debug")
        #kStr += self.assert_lt(vgpr("Serial"), 64) # examine second wavefront

    # TODO - can remove one of these m0 restores if A and B both TLU
    if kernel["DirectToLds%s"%tP["tensorChar"]]:
      imod.footer.addInst("s_mov_b32", "m0", \
          hex(kernel["LdsNumElements"] * tP["bpe"]), \
          "Restore LDS clamp at %u bytes"%(kernel["LdsNumElements"] * tP["bpe"]))

    if problemType["ZeroPad%s"%tc]:
      self.vgprPool.checkIn(addrV)


    return imod

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, tP):
    if not self.do["LocalWrite"]: return ""
    kStr = ""
    tc = tP["tensorChar"]
#fixme-iui  need to use wrapping increment for double or triple buffering:
    if kernel["ExpandPointerSwap"]:
      tP["localWriteSwapByteOffset"] = 0 if tP["localWriteSwapByteOffset"] else kernel["LdsOffsetA_Blk"]*tP["bpe"]
    else:
      if kernel["LocalWriteUseSgpr%s"%tc]:
        kStr += inst("s_xor_b32", \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "swap Red Blk SGPR")
      else:
        numLwa = self.numVgprLocalWriteAddressesA if tP["isA"] else self.numVgprLocalWriteAddressesB
        for i in range(0,numLwa):
          kStr += inst("v_xor_b32", \
              vgpr("LocalWriteAddr%s+%u"%(tc,i)), \
              hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
              vgpr("LocalWriteAddr%s+%u"%(tc,i)), \
              "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets A/B
  # used for global-read + tail-loop to reset to writing in red
  ##############################################################################
  def localWriteResetOffsets(self, kernel, tP):
    if not self.do["LocalWrite"]: return ""
    kStr = ""
    resetMask = hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]-1 | self.LdsOOB)
    tc = tP["tensorChar"]
    if kernel["ExpandPointerSwap"]:
      tP["localWriteSwapByteOffset"] = 0
    else:
      if kernel["LocalWriteUseSgpr%s"%tc]:
        kStr += inst("s_and_b32", \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            resetMask, \
            sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "reset to Red")
      else:
        kStr += inst("v_and_b32", \
            vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            resetMask, \
            vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "reset to Red")
    return kStr

  ##############################################################################
  # Local Write: Init Pointers A/B
  ##############################################################################
  def localWriteInitPointers(self, kernel, tP):
    return ""


  ##############################################################################
  # Calculate offset to use for LDS write
  # Intro:
  #   Each WI has a 2D tile index (coal, perp).
  #     - Code above computes global mem address by scaling one dim by the
  #       lda and adding the other.
  #     - Here we compute a linear LDS offset by scaling one dim by the MT
  #       dim and adding the other.
  #   Result is we map a tile from global memory into LDS.  Consecutive LDS
  #   locations contain elements from different summation 'rows' - therefore
  #   loading a row of LDS will feed computations for different C tile indices.
  #   LocalDotLayout>1 will place N elements from same summation 'row' in
  #   adjacent dims, which is handy for feeding dot instructions.
  # Notes:
  #   Total load insts is nrc * nrp which load the macro-tile.
  #   Par and coalesced are ~synonyms referring to same dimension
  #   Either nrpv or nrvc must be 1 - can't have vectors in both dimensions.
  #     Thus either sPerp or sPara is 0.
  # Inputs:
  #   perp : index of the load in perp dimension (0...nrp)
  #   par  : index of the load in the para dim (0...nrc)
  #   sPerp : component index of the perp vector (0...nrpv)
  #   sPara : component index of the par vector (0...nrcv)
  # Outputs:
  #   offsetBytes : Offset in bytes for the ds_write instruction
  #   i : ?
  #   comment : Comment with the text version of the formula
  #############################################################################
  def calculateLdsWriteOffset(self, perp, para, sPerp, sPara, kernel, tP, localWriteCnt):
    tc = tP["tensorChar"]
    ldl = kernel["LocalDotLayout"]
    mask = ldl-1
    #print "tc ", tc, " perp ", perp, " para ", para, " sPerp ", sPerp, " sPara ", sPara
    lscaOffset = para * kernel[tP["lsc"]]
    perp_masked = perp
    perp_rem = 0
    if (ldl > 1):
      if (kernel[tP["mt"]] >= kernel["SubGroup0"] * kernel["SubGroup1"] * tP["glvw"]):
        # Since it will take multiple fetches to get a full MT, we map low bits of perp to small,
        # horizontal shift to fill in gaps we made by spacing out the data for LDL.
        # Other cases will be handled by low bits of uReg in lwaFirstOffset().
        perp_masked = perp & ~mask
        perp_rem = perp & mask
    lspaOffset = perp_masked * kernel[tP["lsp"]]
    rem = 0

    # Add component offset to interleave from different regs
    # and compute mysterious "i"
    assert(sPerp==0 or sPara==0)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset += sPerp & mask
      lscaOffset += sPara
      rem = (sPerp & ~mask) >> log2(ldl)
      if ldl > 1:
        #i = sPara + (tP["nrcv"]/tP["nrcvpi"]) * (para * tP["glvw"] + tP["nrc"] * (sPerp + tP["glvw"] * tP["nrpv"] * perp ))
        i = localWriteCnt
      else:
        i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp_masked))
      #print "nrcv ", tP["nrcv"], " nrcvpi ", tP["nrcvpi"], " nrc ", tP["nrc"], " nrpv ", tP["nrpv"]
    else:
      lscaOffset += (sPara // ldl) * ldl
      lspaOffset += sPerp
      rem = sPara % ldl
      i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para * tP["glvw"] + tP["nrc"] * (sPerp + tP["glvw"] * tP["nrpv"] * perp ))

    #if not tP["tlu"]:
    #  tmp = sPara
    #  sPara = sPerp
    #  sPerp = tmp
    # print("0lspaOffset", lspaOffset)
    # print("0lscaOffset", lscaOffset)

    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    lds_stride = (kernel["DepthU"] + LdsPad) if kernel["UnrollMajorLDS%s" % tP["tensorChar"]] \
            else (kernel[tP["mt"]] + LdsPad)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset *= lds_stride
      lspaOffset += rem * ldl + perp_rem
    else:
      lscaOffset *= lds_stride
      lscaOffset += rem

    # print("1lspaOffset", lspaOffset)
    # print("1lscaOffset", lscaOffset)
    #if tP["tlu"] == tP["grcv"]:
    #  lspaOffset *= tP["glvw"]
    #  lscaOffset *= tP["glvw"]

    # print("2lspaOffset", lspaOffset)
    # print("2lscaOffset", lscaOffset)
    offsetElements = (lspaOffset + lscaOffset)
    # print("offsetElements", offsetElements)
    offsetBytes = offsetElements*tP["bpe"]

    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
      offsetBytes = offsetBytes + (offsetBytes // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]

    offsetBytes += tP["localWriteSwapByteOffset"]

    #print("offsetBytes", offsetBytes)
    #print "offset", offset

    comment = "lwo%s_%u_%u_%u_%u = (%s%d*%s)" \
        % (tP["tensorChar"], \
        para, sPara, perp, sPerp, \
        (("%u + "%sPara) if tP["wtc"] else ""), \
        para, tP["lsc"] )
    if not tP["tlu"]:
      comment += "*(MT%s+PAD)" % (tP["tileChar"])
    comment += " + (%s%d*%s)" % (
        (("%u + "%sPerp) if tP["wuc"] else ""), perp, \
        tP["lsp"])
    if tP["tlu"]:
      comment += "(*MT%s+PAD)" % (tP["tileChar"])
    comment += " = %u" % (offsetBytes)

    return (offsetBytes, i, comment)


  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  def localWriteDo(self, kernel, tP):
    if not self.do["LocalWrite"]: return ""
    tc = tP["tensorChar"]
    self.localWriteDoCnt += 1
    imod = Code.Module()
    if not kernel["DirectToLds%s"%tc]:
      instruction = tP["localWriteInstruction"]
      numBlocks = instruction.numBlocks
      numOffsets = instruction.numOffsets
      blockWidth = instruction.blockWidth
      #offsetMultiplier = instruction.offsetMultiplier
      g2lIdx = 0
      #kStr += dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
      if 0:
        print("\nLocalWrite", tP["tensorChar"])
        print("tlu", tP["tlu"])
        print("lsc", kernel[tP["lsc"]])
        print("lsp", kernel[tP["lsp"]])
        print("grcv", tP["grcv"])
        print("wtc", tP["wtc"])
        print("wuc", tP["wuc"])
        print("nrc", tP["nrc"])
        print("nrp", tP["nrp"])
        print("nwcv", tP["nwcv"])
        print("nwpv", tP["nwpv"])
        print("nrcvpi", tP["nrcvpi"])
        print("nwcvpi", tP["nwcvpi"])

      tmpLocalWriteAddr = -1

      loopCnt = 0
      # if transposing, positions of sPerp and sPara are transposed
      instructionCnt = -1
      for perp in range(0, tP["nrp"]):
        instructionCnt += 1
        localWriteCode = imod.addCode(Code.Module("LocalWrite%u perp=%d"%(instructionCnt,perp)))
        lwa = "LocalWriteAddr%s"%tc  # default
        if kernel["FractionalLoad"] and perp==tP["nrp"]-1:
          # add inline here:
          overhang = kernel["fractionalPerpOverhang%s"%tc]
          if overhang:
            if kernel["FractionalLoad"]==1:
              # Use already-computed vpr:
              lwa = "LocalWriteAddrOverhang%s"%tc
            elif kernel["FractionalLoad"]==2:
              if tmpLocalWriteAddr == -1:
                tmpLocalWriteAddr = self.vgprPool.checkOut(1,"tmpLocalWriteAddr")

              validWI = overhang*kernel[tP["lsc"]]//tP["glvw"]
              #print "%s: overhang=%u element validWI=%u" % (tc, overhang, validWI)
              localWriteCode.addText(self.comment1("LastPerp.  overhang=%u, mask WI>%u" % (overhang, validWI)))
              localWriteCode.addInst("v_cndmask_b32", \
                          vgpr(tmpLocalWriteAddr), \
                          1.0, \
                          vgpr("LocalWriteAddr%s"%tc), \
                          sgpr("PerpOverhangVcc%s"%tc,2), \
                          "Mask load so out-of-gr-tile bounds returns 0. Note 1.0f=0x3f80000 which is large non-neg int")
              lwa = tmpLocalWriteAddr
        for para in range(0, tP["nrc"]):
          if para>=1:
            localWriteCode = imod.addCode(Code.Module("LocalWrite%u perp=%d para=%d"%(instructionCnt,perp,para)))
          for s in range(0, max(tP["nwcv"],tP["nwpv"])//tP["nwcvpi"]):

            sPerp = 0
            sPara = 0
            if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
              if tP["wtc"] == tP["grcv"]:
                sPerp = s
              elif tP["wuc"] == tP["grcv"]:
                sPara = s
            else:
              if tP["wtc"] == tP["grcv"]:
                sPara = s
              elif tP["wuc"] == tP["grcv"]:
                sPerp = s

            # print("perp:{0} para:{1} sPerp:{2} sPara:{3} loopCnt:{4}".format(perp,para,sPerp,sPara,loopCnt))
            (offset, i, comment) = self.calculateLdsWriteOffset(perp, para, sPerp, sPara, kernel, tP, loopCnt)
            # print("offset: %u"%(offset))
            g2lIdx = i*blockWidth

            paramList = []
            paramList.append(vgpr(lwa))
            for blockIdx in range(0, numBlocks):
              if blockWidth == 1:
                paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx)))
              else:
                paramList.append( vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), \
                    blockWidth))
              if self.db["ForceInputValue%s"%tc]:
                localWriteCode.addInst("v_mov_b32", vgpr("G2L%s+%u"%(tc, g2lIdx)), self.db["ForceValue%s"%tc], "ForceInputValue")
            for oIdx in range(0, numOffsets):
              paramList.append(offset)

            #print "offset", offset

            paramTuple = tuple(paramList)
            #comment = "Reg -> L %u_%u_%u_%u"%(para, sPara, perp, sPerp)
            #comment += " #%u"%self.localWriteDoCnt
            nonTemporal = 0
            highBits = False
            if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
              if s%2==1:
                highBits = True
              if tP["glvw"]==1 and instructionCnt%2==1:
                highBits = True
            localWriteCode.addCode(Code.LocalWriteInst( \
                tP["localWriteInstruction"].toCodeInst(paramTuple, \
                nonTemporal, highBits),comment))

            loopCnt+=1
      if tmpLocalWriteAddr != -1:
        self.vgprPool.checkIn(tmpLocalWriteAddr)


    # localWriteDoCnt<=2 is prefetch if PrefetchGlobalRead:
    if 0 and tP["isB"]: # post-lds-write
    #if 0 and self.localWriteDoCnt >= 0:
      localWriteCode.addInst( "s_waitcnt lgkmcnt(0) & vmcnt(0)", "")
      if self.archCaps["SeparateVscnt"]:
        localWriteCode.addInst( "s_waitcnt_vscnt", "null", "0", "")
      localWriteCode.addInst("s_barrier", "dump LDS" )
      localWriteCode.addText(self.assert_ne(sgpr("WorkGroup0"),1))
      #localWriteCode.addText(self.bomb())

    return imod

  ##############################################################################
  # Local Read: Swap Offsets A/B
  # internalPointerSwap: swap internally tracked offsets - rather than
  #    emit specific instructions to do the pointer swap
  ##############################################################################
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return ""
    kStr = ""

    if internalPointerSwap:
      tP["localReadSwapByteOffset"] = 0 if tP["localReadSwapByteOffset"] else kernel["LdsOffsetA_Blk"]*tP["bpe"]
      kStr += self.comment("local read swap internal offset -> %u" % tP["localReadSwapByteOffset"])
    else:
      kStr += inst("v_xor_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A/B
  # x % n == n & (n-1) for n power of 2
  # tP[localReadOffset] maintains running count of offsets
  # This is called from the tail loop to reset read offsets?
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return ""
    kStr = ""
    if tP["localReadInstruction"].numOffsets == 1:
      tP["localReadSwapByteOffset"] = 0
      kStr += self.comment("localReadResetOffsets")
      tP["localReadOffset"] = 0
      kStr += self.comment1("handled internally")
    kStr += inst("v_and_b32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]-1), \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "reset Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  def localReadInitPointers(self, kernel, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return ""
    kStr = ""
    if self.localReadInstructionA.numOffsets == 1:
      kStr += self.comment("localReadInitPointers")
      tP["localReadOffset"] = 0
    else:
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffset%s_Blk"%tP["tensorChar"]]*tP["bpe"]-1), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "init Red,Blk -> Red")
    return kStr


  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  def localReadInc(self, kernel, iui, tP):
    if not self.do["LocalRead%s" % tP["tensorChar"]]:
      return ""

    kStr = ""

    tc = tP["tensorChar"]
    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0

    if self.inTailLoop:
      inc = kernel["LocalSplitU"] * (kernel["MacroTile%u" % tP["tensorIdx"]] + LdsPad) * tP["bpe"]
      if kernel["EnableMatrixInstruction"]:
        if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
          inc = kernel["LocalSplitU"] * tP["bpe"]
        inc *= kernel["MatrixInstK"]
      tmpSgpr = self.getTmpSgpr(1).idx()
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(inc), "inc")
      kStr += inst("_v_add_co_u32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "vcc", \
          sgpr(tmpSgpr), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "lr%s += %u (LSU*(MT+PAD)*bpe)"%(tP["tensorChar"], inc) )
    else:
      if tP["localReadInstruction"].numOffsets == 1:
        if kernel["EnableMatrixInstruction"]:
          if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
            tP["localReadOffset"] += kernel["LocalSplitU"] * kernel["MatrixInstK"]
          else:
            tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%u"%tP["tensorIdx"]] + LdsPad) * kernel["MatrixInstK"]
        else:
          tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%u"%tP["tensorIdx"]] + LdsPad)
        kStr += self.comment1("N/A, lro->%d" % tP["localReadOffset"])
      else:
        inc = kernel["LocalSplitU"] * (kernel["MacroTile%u" % tP["tensorIdx"]] + LdsPad)
        kStr += inst("_v_add_co_u32", \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            "vcc", \
            hex(inc), \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            "lr%s += %u (LSU+(MT+Pad)*bpe"%(tP["tensorChar"], inc) )

    return kStr


  ##############################################################################
  # Local Read: Do It A/B
  # iui = Inner Unroll Idx
  # epsi = expand pointer swap index. Only used for PAP
  ##############################################################################
  def localReadDoVALU(self, kernel, bufferIdx, iui, epsi, tP):

    self.localReadDoCnt += 1

    tc                = tP["tensorChar"]
    imod              = Code.Module("LocalReadDo%s_I%s"%(tc,iui))
    pack              = Code.Module("pack%s_I%s"%(tc,iui))
    instruction       = tP["localReadInstruction"]
    numOffsets        = instruction.numOffsets
    blockWidth        = instruction.blockWidth
    offsetMultiplier  = 1 # instruction.offsetMultiplier
    #totalReads       = (kernel["ThreadTile%u"%tP["tensorIdx"]]/blockWidth) / numOffsets
    valuIdx           = 0
    numVectorsPerTile = (kernel["ThreadTile%u"%tP["tensorIdx"]]//kernel["VectorWidth"])
    numReadsPerVector = (kernel["VectorWidth"] * tP["bpe"]) // (blockWidth*4) # bytes/register
    loopIdx           = self.unrollIdx
    loopDim           = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    loopChar          = self.indexChars[loopDim]

    for vIdx in range(0, numVectorsPerTile):
      for rIdx in range(0, numReadsPerVector):
        localReadCode = imod.addCode (Code.Module("LocalRead%s Valu%u"%(tc,valuIdx)))
        paramList     = []
        destVgpr      = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuIdx), blockWidth)

        paramList.append(destVgpr)
        paramList.append(vgpr("LocalReadAddr%s"%tc))

        for oIdx in range(0, numOffsets):
          paramList.append(((rIdx*blockWidth + kernel["SubGroup%u"%tP["tensorIdx"]] * (vIdx*numOffsets+oIdx)*kernel["VectorWidth"] \
            + tP["localReadOffset"]) * tP["bpe"] + tP["localReadSwapByteOffset"]) // offsetMultiplier)
          # print("Debug: Matrix{}, rIdx offset {}, vIdx offset {}, bpe {}, net offset {}".format( \
          #     tP["tensorChar"], \
          #     rIdx * blockWidth, \
          #     kernel["SubGroup%u" % tP["tensorIdx"]] * (vIdx * numOffsets + oIdx) * kernel["VectorWidth"] + tP["localReadOffset"], \
          #     tP["bpe"], \
          #     paramList[-1]))
        paramTuple = tuple(paramList)
        comment = "L -> Reg lro=%d swapByteOffset=%u ti=%u vIdx=%u rIdx=%u oIdx=%u buffer=%u iui=%u"\
            %(tP["localReadOffset"],tP["localReadSwapByteOffset"],kernel["SubGroup%u"%tP["tensorIdx"]], vIdx, rIdx, oIdx, bufferIdx, iui)
        localReadCode.addCode(Code.LocalReadInst(instruction.toCodeInst(paramTuple), comment))
        valuIdx += blockWidth

        # TODO - handle vector-load
        if self.db["CheckValue1%s" % tc]:
            localReadCode.addInst("s_waitcnt lgkmcnt(0)", "CheckValue1 wait for LDS read")
            if self.archCaps["SeparateVscnt"]:
              localReadCode.addInst( "s_waitcnt_vscnt", "null", "0", "")
            if kernel["ProblemType"]["DataType"].isHalf():
              localReadCode.append(self.assert_eq(destVgpr, hex(0x3c003c00))) # packed 1s
            elif kernel["ProblemType"]["DataType"].isBFloat16():
              localReadCode.append(self.assert_eq(destVgpr, hex(0x3f803f80))) # packed 1s
            elif kernel["ProblemType"]["DataType"].isInt8x4() or \
                kernel["ProblemType"]["DataType"].isSingle():
              localReadCode.addText(self.assert_eq(destVgpr, 1.0))

    return imod, pack


  ##############################################################################
  # Local Read: Do It A/B
  # iui = Inner Unroll Idx
  # epsi = expand pointer swap index. Only used for PAP
  ##############################################################################
  def localReadDoMFMA(self, kernel, bufferIdx, iui, epsi, tP):

    self.localReadDoCnt += 1

    imod = Code.Module("LocalReadDo%s_I%s" % (tP["tensorChar"],iui))

    tc               = tP["tensorChar"]
    tIdx             = tP["tensorIdx"]
    instruction      = tP["localReadInstruction"]

    numOffsets       = instruction.numOffsets
    blockWidth       = instruction.blockWidth
    MIWaveGropuShape = [ kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0], \
                         kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] ]

    LdsPad           = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    tileStride       = 1
    UnrollStride     = kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad
    if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      tileStride     = kernel["DepthU"] + LdsPad
      UnrollStride   = 1

    numVectorsPerTile = kernel["MIWaveTile"][tIdx]
    numReadsPerVector = tP["bpe"] * kernel["ProblemType"]["DataType"].numMIInput() // int(blockWidth * 4) # bytes/register
    numVgpr           = int(ceil(blockWidth))

    # pack register
    needPack = blockWidth < 1
    pack     = Code.Module("pack%s_I%s"%(tc,iui))
    if needPack:
      tmpVgprIdx = self.vgprPool.checkOut(self.numVgprValuAPerBlock if tc == 'A' else self.numVgprValuBPerBlock)
      pack.addTempVgpr(tmpVgprIdx)

    valufIdx = 0
    for vIdx in range(0, numVectorsPerTile):
      valuiIdx = int(valufIdx)
      localReadCode = imod.addCode (Code.Module("LocalRead%s Valu%u"%(tc,valuiIdx)))
      if needPack:
        packCode = pack.addCode (Code.Module("packCode"))

      for rIdx in range(0, numReadsPerVector):
        valuiIdx = int(valufIdx)
        destVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx), numVgpr)

        # pack for blockWidth 0.5 type
        highBits = (blockWidth == 0.5) and ((rIdx % 2) == 1)
        if needPack and highBits:
          highVgpr = vgpr(tmpVgprIdx + valuiIdx)
          packCode.addInst("v_or_b32", destVgpr, destVgpr, highVgpr, "pack two half Vgpr to one Vgpr")
          destVgpr = highVgpr

        valufIdx += blockWidth

        # load read instrution
        paramList = []
        paramList.append(destVgpr)
        paramList.append(vgpr("LocalReadAddr%s"%tc))

        for oIdx in range(0, numOffsets):
          offset_val = (vIdx * numOffsets+oIdx) * MIWaveGropuShape[tIdx] * tileStride
          offset_val = (rIdx * UnrollStride + offset_val + tP["localReadOffset"]) * tP["bpe"]
          if kernel["LdsBlockSizePerPad%s"%tc] != 0:
            offset_val = offset_val + (offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
          offset_val = offset_val + tP["localReadSwapByteOffset"]
          paramList.append(int(offset_val))

        paramTuple = tuple(paramList)
        comment = "L -> Reg lro=%d swapByteOffset=%u ti=%u vIdx=%u rIdx=%u oIdx=%u buffer=%u iui=%u" \
            % (tP["localReadOffset"], tP["localReadSwapByteOffset"], MIWaveGropuShape[tIdx], vIdx, rIdx, oIdx, bufferIdx, iui)

        localReadCode.addCode(Code.LocalReadInst(instruction.toCodeInst(paramTuple, 0, highBits), comment))

        if self.db["CheckValue1%s"%tc]:
            localReadCode.addInst("s_waitcnt lgkmcnt(0)", "CheckValue1 wait for LDS read")
            if self.archCaps["SeparateVscnt"]:
              localReadCode.addInst( "s_waitcnt_vscnt", "null", "0", "")
            if kernel["ProblemType"]["DataType"].isHalf():
              localReadCode.append(self.assert_eq(destVgpr, hex(0x3c003c00))) # packed 1s
            elif kernel["ProblemType"]["DataType"].isBFloat16():
              localReadCode.append(self.assert_eq(destVgpr, hex(0x3f803f80))) # packed 1s
            elif kernel["ProblemType"]["DataType"].isInt8x4() or kernel["ProblemType"]["DataType"].isSingle():
              localReadCode.addText(self.assert_eq(destVgpr, 1.0))

    return imod, pack



  ##############################################################################
  # Local Read: Do It A/B
  # iui = Inner Unroll Idx
  # uIdx - Unroll Idx
  # epsi = expand pointer swap index. Only used for PAP
  ##############################################################################
  def localReadDo(self, kernel, bufferIdx, iui, epsi, uIdx, tP):

    if not self.do["LocalRead%s" % tP["tensorChar"]]:
      imod = Code.Module("LocalReadDo%s_I%s" % (tP["tensorChar"], iui))
      pack = Code.Module("pack%s_I%s" % (tP["tensorChar"], iui))
      return imod, pack

    if kernel["EnableMatrixInstruction"]:
      return self.localReadDoMFMA(kernel, bufferIdx, iui, epsi, tP)
    else:
      return self.localReadDoVALU(kernel, bufferIdx, iui, epsi, tP)


  ##############################################################################
  # Save the local read pointers, for example when creating a duplicated
  # optimized path (like optNLL)
  ##############################################################################
  def saveLocalPointers(self, kernel):
    self.tPA["savedLocalReadOffset"] = self.tPA["localReadOffset"]
    self.tPB["savedLocalReadOffset"] = self.tPB["localReadOffset"]

  ##############################################################################
  # Restore the saved local read pointers
  # Must be paired with an earlier call to savePointers
  ##############################################################################
  def restoreLocalPointers(self, kernel):
    self.tPA["localReadOffset"] = self.tPA["savedLocalReadOffset"]
    self.tPB["localReadOffset"] = self.tPB["savedLocalReadOffset"]
    del self.tPA["savedLocalReadOffset"]
    del self.tPB["savedLocalReadOffset"]

  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponentsVALU(self, kernel, tP):
    kStr = ""

    # glvw
    vw = tP["glvw"]
    numVectors = kernel[tP["tt"]]//vw
    # labels
    svrLabels = []
    sviLabels = []
    for i in range(0, vw):
      r = (i+1) % vw
      label = self.getLabelNum("ShiftVectorComponents%u_R%u"%(tP["idx"], r) )
      svrLabels.append(label)
      tmpLabels = []
      for v in range(0, numVectors):
        label = self.getLabelNum("ShiftVectorComponents%u_R%u_V%u"%(tP["idx"], r, v) )
        tmpLabels.append(label)
      sviLabels.append(tmpLabels)

    # wgMT value
    tmpSgpr = self.getTmpSgpr(2).idx()
    tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
    wgMT = self.vgprPool.checkOut(1,"wgMT")
    kStr += inst("v_mov_b32", vgpr(wgMT), sgpr(tP["wg"]), "")
    kStr += inst("v_mul_i32_i24", vgpr(wgMT), hex(-kernel[tP["mt"]]), vgpr(wgMT), \
        "wg*MT")
    kStr += inst("_v_add_co_u32", vgpr(wgMT), "vcc", sgpr("SizesFree+%u"%tP["idx"]), \
        vgpr(wgMT), "wgMT = Size - wg*MT")
    kStr += inst("v_mov_b32", vgpr(tmpVgpr), hex(kernel[tP["mt"]]), "MT")
    kStr += inst("v_cmp_lt_u32", sgpr(tmpSgpr,2), vgpr(wgMT), \
        vgpr(tmpVgpr), "wgMT < MT" )
    kStr += inst("v_cndmask_b32", vgpr(wgMT), vgpr(tmpVgpr), \
        vgpr(wgMT), sgpr(tmpSgpr,2), "wgMT = (wgMT < MT) ? wgMT : MT" )
    dummy = self.vgprPool.checkOut(1,"dummy")

    # qReg
    qReg = self.vgprPool.checkOut(1,"qReg")
    divisor = kernel["VectorWidth"] # vw
    kStr += vectorStaticDivide(qReg, wgMT, divisor, \
        tmpVgpr, tmpSgpr)

    # rReg
    rReg = self.vgprPool.checkOut(1,"rReg")
    divisor = vw
    kStr += vectorStaticRemainder(dummy, rReg, wgMT, divisor, \
        tmpVgpr, tmpSgpr)

    # qReg %/ SG
    sReg = self.vgprPool.checkOut(1,"sReg")
    eReg = self.vgprPool.checkOut(1,"eReg")
    divisor = kernel[tP["sg"]]
    kStr += vectorStaticDivideAndRemainder(sReg, eReg, qReg, divisor, \
        tmpVgpr, tmpSgpr)

    if tP["isA"]:
      # thread = serial % SG0
      thread = self.vgprPool.checkOut(1,"thread")
      divisor = kernel["SubGroup0"]
      kStr += vectorStaticRemainder(dummy, thread, "Serial", divisor, \
          tmpVgpr, tmpSgpr)
      #kStr += dump(vgpr(thread))
      #kStr += dump(vgpr(thread))
    else:
      # thread = (serial / SG0) % SG1
      sd0 = self.vgprPool.checkOut(1,"sd0")
      divisor = kernel["SubGroup0"]
      kStr += vectorStaticDivide(sd0, "Serial", divisor, \
          tmpVgpr, tmpSgpr) # thread = serial / SG0
      divisor = kernel["SubGroup1"]
      thread = self.vgprPool.checkOut(1,"thread")
      kStr += vectorStaticRemainder(dummy, thread, sd0, divisor, \
          tmpVgpr, tmpSgpr) # thread = (serial / SG0) % SG1
      self.vgprPool.checkIn(sd0)

    # which glvw vector of thread to shift? wgMT / (SG0*VW) -> (wgMT%VW) / glvw
    # (wgMT/(WG0*VW))*(VW/glvw) + (wgMT%VW) / glvw
    if True:#tP["tensorIdx"] > kernel["VectorWidth"]:
      mvReg = self.vgprPool.checkOut(1,"mvReg")
      divisor = kernel[tP["sg"]]*kernel["VectorWidth"]
      kStr += vectorStaticDivide(mvReg, wgMT, divisor, \
          tmpVgpr, tmpSgpr)
      if vw < kernel["VectorWidth"]:
        kStr += inst("v_lshlrev_b32", vgpr(mvReg), hex(log2(kernel["VectorWidth"]//vw)), vgpr(mvReg), "vId *= VW/glvw")
    #kStr += dump(vgpr(mvReg))

    vReg = self.vgprPool.checkOut(1,"vReg")
    divisor = kernel["VectorWidth"]
    kStr += vectorStaticRemainder(dummy, vReg, wgMT, divisor, \
        tmpVgpr, tmpSgpr)
    vRegD = self.vgprPool.checkOut(1,"vRegD")
    kStr += inst("v_mov_b32", vgpr(vRegD), vgpr(vReg), "duplicate")
    divisor = vw
    kStr += vectorStaticDivide(vReg, vRegD, divisor, \
        tmpVgpr, tmpSgpr)
    #kStr += dump(vgpr(vReg))

    if True:#tP["tensorIdx"] > kernel["VectorWidth"]:
      kStr += inst("_v_add_co_u32", vgpr(vReg), "vcc", vgpr(mvReg), vgpr(vReg), "vId = 2 components")
      self.vgprPool.checkIn(mvReg)
      self.vgprPool.checkIn(vRegD)

    kStr += inst("v_cmp_eq_u32", sgpr(tmpSgpr,2), vgpr(thread), \
        vgpr(eReg), "mask" )
    kStr += inst("v_mov_b32", vgpr(tmpVgpr+0), sgpr(tmpSgpr+0), "")
    kStr += inst("v_mov_b32", vgpr(tmpVgpr+1), sgpr(tmpSgpr+1), "")

    # for each remainder, jump
    for r in range(1, vw):
      kStr += inst("v_cmp_eq_u32", "vcc", vgpr(rReg), \
          hex(r), "wgMT%%VW == %u"%r )
      kStr += inst("s_cbranch_vccnz label_%04u"\
          % svrLabels[(r-1)%vw], \
          "shift d%u r=%u"%(tP["idx"], r))
      #kStr += inst("s_mov_b32", sgpr(sgprLoc), hex(location), \
      #    "location=%u"%location) location *= 2
      #kStr += inst("v_or_b32", vgpr(vgprPath), sgpr(sgprLoc), \
      #    vgpr(vgprPath), "path+=location")
    kStr += inst("s_branch label_%04u"%svrLabels[vw-1], \
        "no shifting" )

    # code blocks for shifting
    for r in range(1, vw):
      kStr += self.comment3("shift d%u r=%u"%(tP["idx"], r))
      kStr += "label_%04u:%s" % (svrLabels[r-1], self.endLine)
      #if r==3 and tP["isA"]:
      #  kStr += dump(vgpr("Serial"))

      # for each vector index, jump
      for vectorIdx in range(0, numVectors):
        kStr += inst("v_cmp_eq_u32", "vcc", vgpr(vReg), \
            hex(vectorIdx), "wgMT/(SG*VW) == %u"%vectorIdx )
        kStr += inst("s_cbranch_vccnz label_%04u"\
            % sviLabels[(r-1)%vw][vectorIdx], \
            "shift d%u, r=%u, v=%u"%(tP["idx"], r, vectorIdx))

      # code blocks for shifting vector
      for vectorIdx in range(0, numVectors):
        kStr += self.comment("shift d%u r=%u v=%u"%(tP["idx"], r, vectorIdx))
        kStr += "label_%04u:%s" % (sviLabels[r-1][vectorIdx], self.endLine)
        # mask if last thread in thread#-tile column
        kStr += inst("_v_cmpx_eq_u32", sgpr(tmpSgpr,2), vgpr(thread), \
          vgpr(eReg), "serial % SG == (wgMT/VECTOR_WIDTH)%SG" )
        tto = kernel["ThreadTile%u"%((tP["idx"]+1)%2)] # thread tile orthogonal
        for tt in range(0, tto):
          for s in range(0, r):
            if tP["isA"]: # shift d0
              dst = (s) \
                  + vectorIdx * vw + tt * kernel["ThreadTile0"]
              src = (s+vw-r) \
                  + vectorIdx * vw + tt * kernel["ThreadTile0"]
              comment = "rC[%u+%u*VW+%u*TT%s] = rC[%u+%u*VW+%u*TT%s]" \
                  % (s, vectorIdx, tt, self.tileChar0, \
                  s+vw-r, vectorIdx, tt, self.tileChar0 )
            else: # shift d1
              dst = (tt) \
                  + vectorIdx*vw*kernel["ThreadTile0"] + s * kernel["ThreadTile0"]
              src = (tt) \
                  + vectorIdx * vw*kernel["ThreadTile0"] + (s+vw-r) * kernel["ThreadTile0"]
              comment = "rC[%u+%u*TT%s*VW+%u*TT%s] = rC[%u+%u*TT%s*VW+%u*TT%s]" \
                % (tt, vectorIdx, self.tileChar0, s, self.tileChar0, \
                tt, vectorIdx, self.tileChar0, \
                s+vw-r, self.tileChar0)

            kStr += "// src=%u, dst=%u\n" % (src,dst)

            # f16
#jgolds I think this should be bpeCinternal
            if self.bpeCinternal == 2:
              srcVgpr = self.startVgprValuC+src*self.bpeCinternal//self.bpr
              dstVgpr = self.startVgprValuC+dst*self.bpeCinternal//self.bpr
              kStr += "// %u, %u, %u, %u, %u, %u\n" % (r, vectorIdx, tt, s, dst, src)
              if tP["isA"]: # f16 d0
                if r % 2 == 0: # even shift can use mov_b32
                  if s % 2 == 0:
                    kStr += inst("v_mov_b32", vgpr(dstVgpr), \
                        vgpr(srcVgpr), comment)
                  else:
                    pass # above command performs two moves
                else: # odd shift
                  srcLo = src % 2 == 0 # even
                  dstLo = dst % 2 == 0 # even
                  kStr += "// srcLo=%u, dstLo=%u\n" % (srcLo,dstLo)
                  if dstLo: # hi src to lo dst; can clobber hi bits
                    kStr += inst("v_lshrrev_b32", vgpr(dstVgpr), \
                        hex(16), vgpr(srcVgpr), "hi16 -> lo16")
                  else: # dstHi; cannot clobber lo bits
                    tmpSrcVgpr = self.vgprPool.checkOut(1,"tmpSrcVgpr")
                    # zero out dst hi bits -> dst
                    kStr += inst("v_and_b32", vgpr(dstVgpr), \
                        "0x0000FFFF", vgpr(dstVgpr), "zero out dst hi16")
                    if srcLo: # lo src to hi dst
                      # left shift src 16 bits -> tmpSrc
                      kStr += inst("v_lshlrev_b32", vgpr(tmpSrcVgpr), \
                          hex(16), vgpr(srcVgpr), "left shift src 16 bits")
                    else: # hi src to hi dst
                      # zero out src lo bits -> tmpSrc
                      kStr += inst("v_and_b32", vgpr(srcVgpr), \
                          "0xFFFF0000", vgpr(tmpSrcVgpr), "zero out src lo16")
                    # dst = tmpSrc | dst
                    kStr += inst("v_or_b32", vgpr(dstVgpr), \
                        vgpr(tmpSrcVgpr), vgpr(dstVgpr), "dst = tmpSrc | dst")
                    self.vgprPool.checkIn(tmpSrcVgpr)

              else: # f16 d1
                if tt%2==0:
                  kStr += inst("v_mov_b32", vgpr(dstVgpr), \
                      vgpr(srcVgpr), comment)
                else:
                  pass # above shift moves two f16

            # f32 or larger
            else:
              for i in range(0, self.bpeCinternal//self.bpr):
                kStr += inst("v_mov_b32", vgpr(self.startVgprValuC+dst*self.bpeCinternal//self.bpr+i), \
                    vgpr(self.startVgprValuC+src*self.bpeCinternal//self.bpr+i), comment)

        # end shift reset mask and jump out
        kStr += inst("s_mov_b64", sgpr(tmpSgpr,2), \
            "0xFFFFFFFFFFFFFFFF", "to restore all threads active")
        kStr += inst("s_or_saveexec_b64", "vcc", sgpr(tmpSgpr,2), \
            "all threads active")
        kStr += inst("s_branch label_%04u"%svrLabels[vw-1], \
            "done shifting" )
    #kStr += inst("s_mov_b32", sgpr(sgprLoc), hex(location), "location=%u"%location) location *= 2
    #kStr += inst("v_or_b32", vgpr(vgprPath), sgpr(sgprLoc), vgpr(vgprPath), "path+=location")
    kStr += "label_%04u: // end shift0%s" % (svrLabels[vw-1], self.endLine)
    #kStr += inst("s_mov_b64", "exec","0xFFFFFFFFFFFFFFFF","")
    #kStr += dump(vgpr(vgprPath))

    # checkin scratch vgprs
    self.vgprPool.checkIn(wgMT)
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(qReg)
    self.vgprPool.checkIn(rReg)
    self.vgprPool.checkIn(sReg)
    self.vgprPool.checkIn(eReg)
    self.vgprPool.checkIn(thread)
    self.vgprPool.checkIn(dummy)
    self.vgprPool.checkIn(vReg)
    return kStr


  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponentsMFMA(self, kernel, tP):
    """ when we enable shift ptr with vectorwidth(2), we shift global read on edge block when size % vectorwidth != 0.
        For example if M size == 3 vector width == 2, we want to do global read for [0-1] and [2-3].
        But 3 is not in memory object, so we shift to do global read [0-1] and [1-2].
        So buffer become [0, 1, 1, 2], assume result in register is same as input [0, 1, 1, 2]
        We need to shift it back to [0, 1, 2].

        In MFMA outputs, We have numContinuousOutput(4) for each thread.
        We have numThreadInWave(64) threads.
        number of thread in N is sames as kernel["MatrixInstN"] (32)
        number of thread in M is numThreadInWave/numOutputThreads1 = 2
        stride of continous output for each thread (numSubOutputPerWave0) is numOutputThreads0 * numContinuousOutput, (8).
        we have numSubOutputGroupsPerWave0 which is 4 (kernel[tP["mt"]](64) // numSubOutputPerWave0(8))

        So we do shift back by below alorithm.
        1. check if M_size % vectorwidth != 0, return if == 0
        2. decide which subgroup we need to shift, M_size(3) means 3/8 = group 0
        3. decide which thread we need to shift, we have different groups of thread, (0-31) for first group, (32-63) for second group.
        4. decide which shift block (subTile1) we want to shift. for ex [0-1], [1-2], we want to shift second subtile
    """

    kStr = ""

    glvw                       = tP["glvw"]
    numThreadInWave            = globalParameters["WavefrontWidth"]
    MIBShape0                  = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
    numContinuousOutput        = kernel["MIOutputVectorWidth"]
    numOutputThreads1          = kernel["MatrixInstN"]
    numOutputThreads0          = kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else (numThreadInWave // numOutputThreads1)
    numSubOutputPerWave0       = numOutputThreads0 * numContinuousOutput
    numSubOutputGroupsPerWave0 = MIBShape0 // numSubOutputPerWave0
    numShiftBlock              = numContinuousOutput // glvw
    numOutputElements          = numSubOutputGroupsPerWave0 * numContinuousOutput * kernel["MIWaveTile"][0]
    subTile1                   = kernel["MIWaveTile"][1] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBN"] * kernel["MIWaveTile"][1]

    # labels for reminder of vectorwidth
    svrLabels = []
    # label for reminder of subgroup
    sviLabels = []
    # label for reminder of shift block(subtile)
    svoLabels = []

    for i in range(0, glvw):
      r = (i+1) % glvw
      label = self.getLabelNum("ShiftVectorComponents%u_R%u" % (tP["idx"], r) )
      svrLabels.append(label)
      tmpLabels = []
      tmp2Labels = []
      for wt in range(0, kernel["MIWaveTile"][0]):
        for v in range(0, numSubOutputGroupsPerWave0):
          label = self.getLabelNum("ShiftVectorComponents%u_R%u_WT%u_V%u" % (tP["idx"], r, wt, v) )
          tmpLabels.append(label)
          tmp2Labels2 = []
          for o in range(0, numShiftBlock):
            label = self.getLabelNum("ShiftVectorComponents%u_R%u_Wt%u_V%u_O%u" % (tP["idx"], r, wt, v, o) )
            tmp2Labels2.append(label)
          tmp2Labels.append(tmp2Labels2)
      sviLabels.append(tmpLabels)
      svoLabels.append(tmp2Labels)

    # wgMT value
    tmpSgpr = self.getTmpSgpr(2).idx()
    tmpVgpr = self.vgprPool.checkOut(2)
    dummy   = self.vgprPool.checkOut(1)
    wgMT    = self.vgprPool.checkOut(1)

    # get M size of edge block
    mtReg = self.vgprPool.checkOut(1)
    kStr += inst("v_mov_b32"    , vgpr(wgMT), sgpr(tP["wg"]), "")
    kStr += inst("v_mul_i32_i24", vgpr(wgMT), hex(-kernel[tP["mt"]]), vgpr(wgMT), "wg*MT")
    kStr += inst("_v_add_co_u32", vgpr(wgMT), "vcc", sgpr("SizesFree+%u"%tP["idx"]), vgpr(wgMT), "wgMT = Size - wg*MT")
    kStr += inst("v_mov_b32"    , vgpr(mtReg), hex(kernel[tP["mt"]]), "MT")
    kStr += inst("v_cmp_lt_u32" , sgpr(tmpSgpr,2), vgpr(wgMT), vgpr(mtReg), "wgMT < MT" )
    kStr += inst("v_cndmask_b32", vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), sgpr(tmpSgpr,2), "wgMT = (wgMT < MT) ? wgMT : MT" )

    wReg = self.vgprPool.checkOut(1)
    kStr += vectorStaticDivide(wReg, "Serial", globalParameters["WavefrontWidth"], tmpVgpr, tmpSgpr)
    kStr += vectorStaticRemainder(dummy, wReg, wReg, kernel["MIWaveGroup"][0], tmpVgpr, tmpSgpr)
    sReg = self.vgprPool.checkOut(1)
    kStr += vectorStaticDivide(sReg, wgMT, MIBShape0, tmpVgpr, tmpSgpr)
    kStr += vectorStaticRemainder(dummy, sReg, sReg, kernel["MIWaveGroup"][0], tmpVgpr, tmpSgpr)
    kStr += inst("v_cmp_eq_u32" , sgpr(tmpSgpr,2), vgpr(sReg), vgpr(wReg), "wave_id0 == block_belong_to_wave0?" )
    kStr += inst("v_cndmask_b32", vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), sgpr(tmpSgpr,2), "wgMT = (wgMT < MT) ? wgMT : MT" )
    self.vgprPool.checkIn(mtReg)
    self.vgprPool.checkIn(sReg)

    # gReg : group id of numSubOutputGroupsPerWave0
    gReg = self.vgprPool.checkOut(1)
    kStr += staticMultiply(vgpr(wReg), vgpr(wReg), MIBShape0 // numSubOutputPerWave0, sgpr(tmpSgpr))
    kStr += vectorStaticDivide(gReg, wgMT, numSubOutputPerWave0, tmpVgpr, tmpSgpr)
    kStr += inst("v_sub_u32", vgpr(gReg), vgpr(gReg), vgpr(wReg), "")
    self.vgprPool.checkIn(wReg)

    # eReg : use to disguish which shift block (sub-tile) we need to deal with
    eReg = self.vgprPool.checkOut(1)
    kStr += vectorStaticRemainder(dummy, eReg, wgMT, numContinuousOutput, tmpVgpr, tmpSgpr)

    # mRge : decide which thread have to deal with this M-size
    mReg = self.vgprPool.checkOut(1)
    kStr += vectorStaticDivide(mReg, wgMT, numContinuousOutput, tmpVgpr, tmpSgpr)
    kStr += vectorStaticRemainder(dummy, mReg, mReg, numOutputThreads0, tmpVgpr, tmpSgpr)

    # mRge : thread group id [0-31] or [32-63] for mfma 32x32x2
    tReg = self.vgprPool.checkOut(1)
    kStr += vectorStaticDivide(tReg, "Serial", kernel["MatrixInstN"], tmpVgpr, tmpSgpr)
    kStr += vectorStaticRemainder(dummy, tReg, tReg, numOutputThreads0, tmpVgpr, tmpSgpr)

    # rReg : reminder of M_size % vectorwidth
    # decide to jump to block which handle this case, M_szie % vector width
    rReg = self.vgprPool.checkOut(1)
    kStr += vectorStaticRemainder(dummy, rReg, wgMT, glvw, tmpVgpr, tmpSgpr)
    for r in range(1, glvw):
      kStr += inst("v_cmp_eq_u32", "vcc", vgpr(rReg), hex(r), "wgMT%%VW == %u"%r )
      kStr += inst("s_cbranch_vccnz label_%04u" % svrLabels[(r-1)], "shift d%u r=%u"%(tP["idx"], r))
    kStr += inst("s_branch label_%04u"%svrLabels[glvw-1], "no shifting" )
    self.vgprPool.checkIn(rReg)

    # blocks for handle M_szie % vector width
    for r in range(1, glvw):
      kStr += self.comment3("shift d%u r=%u"%(tP["idx"], r))
      kStr += "label_%04u:%s" % (svrLabels[r-1], self.endLine)

      for wt in range(0, kernel["MIWaveTile"][0]):
        # decide to jump to block wich handle sub group id for numSubOutputGroupsPerWave0
        # we have 8 blocks for MT-M 64 with mfma 32x32x2. 64/2(thread group)/4(continous output)
        for ot in range(0, numSubOutputGroupsPerWave0):
          packIdx = wt * numSubOutputGroupsPerWave0 + ot
          grpVal  = wt * numSubOutputGroupsPerWave0 * kernel["MIWaveGroup"][0] + ot
          kStr += inst("v_cmp_eq_u32", "vcc", vgpr(gReg), hex(grpVal), "wgMT/8 == %u" % packIdx )
          kStr += inst("s_cbranch_vccnz label_%04u" % sviLabels[(r-1)][packIdx], "shift d%u, r=%u, v=%u" % (tP["idx"], r, packIdx))

      for wt in range(0, kernel["MIWaveTile"][0]):
        # blocks for handle sub group id for numSubOutputGroupsPerWave0
        for ot in range(0, numSubOutputGroupsPerWave0):
          packIdx = wt * numSubOutputGroupsPerWave0 + ot
          kStr += self.comment("shift d%u r=%u v=%u" % (tP["idx"], r, packIdx))
          kStr += "label_%04u:%s" % (sviLabels[r-1][packIdx], self.endLine)

          # mask if last thread in thread#-tile column
          kStr += inst("v_cmpx_eq_u32", sgpr(tmpSgpr,2), vgpr(tReg), vgpr(mReg), "(serial % 64) / 32 == (wgMT/4)%2" )

          # decide to jump to block wich handle element of shfit block (subtile)
          # for vector widht 2 with continuous 4, we have 1, 3 case to handle
          for outIdx in range(0, numShiftBlock):
            kStr += inst("v_cmp_eq_u32", "vcc", vgpr(eReg), hex(outIdx*glvw+r), "wgMT %% 4 == %u" % (outIdx*2+1) )
            kStr += inst("s_cbranch_vccnz label_%04u" % svoLabels[(r-1)][packIdx][outIdx], "shift d%u, r=%u, v=%u, o=%u" % (tP["idx"], r, packIdx, outIdx))

          # blocks to handle shfiting
          for outIdx in range(0, numShiftBlock):
            kStr += "label_%04u:%s" % (svoLabels[(r-1)][packIdx][outIdx], self.endLine)
            for subTile1Idx in range(0, subTile1):
              for shiftIdx in range(0, r):
                dstVgpr = subTile1Idx * numOutputElements + packIdx * numContinuousOutput + outIdx * glvw + shiftIdx
                srcVgpr = subTile1Idx * numOutputElements + packIdx * numContinuousOutput + outIdx * glvw + shiftIdx + (glvw - r)
                kStr += inst("v_mov_b32", vgpr(dstVgpr), vgpr(srcVgpr), "")

          # end shift reset mask and jump out
          kStr += inst("s_mov_b64", sgpr(tmpSgpr,2), "0xFFFFFFFFFFFFFFFF", "to restore all threads active")
          kStr += inst("s_or_saveexec_b64", "vcc", sgpr(tmpSgpr,2), "all threads active")
          kStr += inst("s_branch label_%04u" % svrLabels[glvw-1], "done shifting" )

    kStr += "label_%04u: // end shift0%s" % (svrLabels[glvw-1], self.endLine)

    # checkin scratch vgprs
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(wgMT)
    self.vgprPool.checkIn(dummy)
    self.vgprPool.checkIn(gReg)
    self.vgprPool.checkIn(eReg)
    self.vgprPool.checkIn(mReg)
    self.vgprPool.checkIn(tReg)

    return kStr


  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponents(self, kernel, tP):
    if kernel["EnableMatrixInstruction"]:
      return self.shiftVectorComponentsMFMA(kernel, tP)
    else:
      return self.shiftVectorComponentsVALU(kernel, tP)


  ##############################################################################
  # Complex Declare Tmp Registers - SKIP
  ##############################################################################
  def complexDeclareTmpRegisters(self, kernel):
    kStr = ""
    return kStr
    #if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
    #  kStr += "  float type_mac_tmp" + self.endLine
    #if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
    #  kStr += "  double type_mac_tmp" + self.endLine
    #return kStr

  ##############################################################################
  # isLds = true if querying about LDS operations (which can use dword operations)
  #     isLds=False if we want element step for the VALU add operations
  ##############################################################################
  def getLocalSplitUElementStep(self, kernel, isLds):

    if isLds and \
       kernel["VectorWidth"]*self.bpeCinternal >= 8 and \
       kernel["GlobalWriteVectorWidth"]*self.bpeCinternal >= 8:
      useDwordX2 = 1
    else:
      useDwordX2 = 0

    #useDwordX2 = 0

    if kernel["ProblemType"]["DataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
      assert(kernel["VectorWidth"]%2 == 0)
      elementStep = 2*(useDwordX2+1)
    elif kernel["ProblemType"]["DataType"].isInt8x4() or \
         kernel["ProblemType"]["DataType"].isBFloat16() or \
         kernel["ProblemType"]["DataType"].isHalf() or \
         kernel["ProblemType"]["DataType"].isSingle():
      elementStep = 1*(useDwordX2+1)
    elif kernel["ProblemType"]["DataType"].isDouble() or \
         kernel["ProblemType"]["DataType"].isSingleComplex():
      if isLds:
        assert (useDwordX2==1)
      elementStep = 1
    elif kernel["ProblemType"]["DataType"].isDoubleComplex():
      if isLds:
        assert (useDwordX2==1)
      elementStep = 1

    return (elementStep, useDwordX2)

  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  def localSplitULocalWrite(self, kernel):
    kStr = ""
    # wait for summation to be done with lds before writing reduction values
    kStr += self.syncThreads(kernel, "pre-lsu local write")

    tmpVgpr = self.vgprPool.checkOut(2,"tmpVgpr")
    lr0 = self.vgprPool.checkOut(1,"lr0")
    lr1 = self.vgprPool.checkOut(1,"lr1")
    sg = self.vgprPool.checkOut(1,"sg")
    copy = self.vgprPool.checkOut(1,"copy")
    tmpSgpr = self.getTmpSgpr(1).idx()

    # lr0 = serial % SG0
    kStr += vectorStaticDivideAndRemainder(lr1, lr0, "Serial", \
        kernel["SubGroup0"], tmpVgpr, tmpSgpr)

    # lr1 = (serial / SG0) % SG1
    # sg  = (serial / SG0) / SG1
    kStr += inst("v_mov_b32", vgpr(copy), vgpr(lr1), "copy for divide")
    kStr += vectorStaticDivideAndRemainder(sg, lr1, copy, \
        kernel["SubGroup1"], tmpVgpr, tmpSgpr)

    # lr0 *= VW
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["VectorWidth"]*self.bpeCinternal), "VW")
    kStr += inst("v_mul_lo_u32", vgpr(lr0), sgpr(tmpSgpr), vgpr(lr0), \
        "lr0 *= VW")
    # lr1 *= VW*MT0
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), \
        hex(kernel["VectorWidth"]*kernel["MacroTile0"]*self.bpeCinternal), "VW*MT0")
    kStr += inst("v_mul_lo_u32", vgpr(lr1), sgpr(tmpSgpr), vgpr(lr1), \
        "lr1 *= VW*MT0")
    # sg  *= MT0*MT1
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), \
        hex(kernel["MacroTile0"]*kernel["MacroTile1"]*self.bpeCinternal), "MT0*MT1")
    kStr += inst("v_mul_lo_u32", vgpr(sg), sgpr(tmpSgpr), vgpr(sg), \
        "sg *= MT0*MT1")

    # thread offset
    addr = lr0
    kStr += inst("_v_add_co_u32", vgpr(addr), "vcc", vgpr(lr1), vgpr(addr),  "")
    kStr += inst("_v_add_co_u32", vgpr(addr), "vcc", vgpr(sg), vgpr(addr),  "threadOffset")
    self.vgprPool.checkIn(lr0)
    self.vgprPool.checkIn(lr1)
    self.vgprPool.checkIn(sg)
    self.vgprPool.checkIn(copy)
    self.vgprPool.checkIn(tmpVgpr)

    # dump addr
    #kStr += dump(vgpr(addr))

    # do writes
    # LDS Layout example (for Sgemm, LSU=4, TT=8x8, WG=[8,4,4]), 128 WI/WG
    # VectorWidth = GlobalWriteVectorWidth = 4
    # SubGroup0 (WI:00-32)  : LDS 0x0000-
    # SubGroup1 (WI:33-64)  : LDS 0x2000-
    # SubGroup2 (WI:65-95)  : LDS 0x4000-
    # SubGroup3 (WI:96-127) : LDS 0x6000-

    # Interleave within a subgroup is interesting...
    #       Start LDS Addr
    # WI00 - 0x000
    # WI01 - 0x010
    # ...
    # WI07 - 0x070
    # WI08 - 0x400
    # WI09 - 0x410
    # ...
    # WI0F - 0x470
    # WI10 - 0x800
    # ...
    # ...
    # WI1f - 0xc70
    # WI20 - 0x1000  (start SubGroup1)

    # so a zoom-in on the pattern at beginning of LDS, for the case above:
    #   WI (hex) |x00-|x01-|...   |x07-|0x0-|0x1-|...|0x7-|0x0-| ... ... ||0x8-|
    # ValuC      |0123|0123|...   |0123|4567|4567|...|4567|89AB| ... ... ||0123
    #            |                     |                  |               |
    # LDS Addr  0x0                  0x80               0x100           0x400

    # Perhaps could optimize this into something simpler with fewer bank conflicts
    (elementStep, useDwordX2) = self.getLocalSplitUElementStep(kernel, True)
    for j in range(0, kernel["ThreadTile1"]//kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]//kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          for vc in range(0, kernel["VectorWidth"], elementStep):
            # for half, write 2 elements (4 bytes)
            # for single, write 1 element (4 bytes)
            # double doesn't work yet
            writeOffset = vc \
                + i*kernel["SubGroup0"]*kernel["VectorWidth"] \
                + s*kernel["MacroTile0"] \
                + j*kernel["MacroTile0"]*kernel["SubGroup1"]*kernel["VectorWidth"]
            regIdx = vc \
                + i*kernel["VectorWidth"] \
                + s*kernel["ThreadTile0"] \
                + j*kernel["ThreadTile0"]*kernel["VectorWidth"]
            writeOffset /= elementStep
            if useDwordX2:
              regIdx = regIdx*self.bpeCinternal // 4
              kStr += inst("ds_write_b64", vgpr(addr), vgpr("ValuC+%u"%regIdx,2), \
                           "offset:%u"%(elementStep*writeOffset*self.bpeCinternal), 
                           "j=%u i=%u s=%u vc=%u"%(j,i,s,vc))
            else:
              regIdx //= elementStep
              kStr += inst("ds_write_b32", vgpr(addr), vgpr("ValuC+%u"%regIdx), \
                           "offset:%u"%(elementStep*writeOffset*self.bpeCinternal), 
                           "j=%u i=%u s=%u vc=%u"%(j,i,s,vc))
            # ds_write value
            #kStr += dump(vgpr(regIdx))
    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for all writes")
    if self.archCaps["SeparateVscnt"]:
      kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
    kStr += self.syncThreads(kernel, "post-lsu local write")
    #kStr += self.dumpLds(kernel, 0, 16)
    #kStr += self.bomb(5)
    return kStr

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    kStr = ""
    tmpSgpr = self.getTmpSgpr(1).idx()
    baseAddr = self.vgprPool.checkOut(1,"baseAddr")
    kStr += staticMultiply(vgpr(baseAddr), vgpr("Serial"), kernel["GlobalWriteVectorWidth"]*self.bpeAB, sgpr(tmpSgpr))
    (elementStep, useDwordX2) = self.getLocalSplitUElementStep(kernel, True)
    # Load values for each subgroup
    for r in range(0, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"], elementStep):
          offset = s + i*kernel["NumThreads"]*kernel["GlobalWriteVectorWidth"] + r * kernel["MacroTile0"]*kernel["MacroTile1"]
          regIdx = s + i*kernel["GlobalWriteVectorWidth"] + r*kernel["GlobalWriteVectorWidth"]*kernel["NumGlobalWriteVectorsPerThread"]
          if useDwordX2:
            regIdx = regIdx * self.bpeCinternal // 4
            kStr += inst("ds_read_b64", vgpr("ValuC+%u"%regIdx,2), \
                vgpr(baseAddr), "offset:%u"%(offset*self.bpeCinternal), "r=%u i=%u s=%u"%(r,i,s))
          else:
            regIdx //= elementStep
            kStr += inst("ds_read_b32", vgpr("ValuC+%u"%regIdx), \
                vgpr(baseAddr), "offset:%u"%(offset*self.bpeCinternal), "r=%u i=%u s=%u"%(r,i,s))
    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for all reads")
    if self.archCaps["SeparateVscnt"]:
      kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
    self.vgprPool.checkIn(baseAddr)
    return kStr

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  def localSplitUReduction(self, kernel):
    kStr = ""
    (elementStep, useDwordX2) = self.getLocalSplitUElementStep(kernel, False)
    for r in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"],elementStep):
          cIdx = s + i*kernel["GlobalWriteVectorWidth"]
          regIdx = s + i*kernel["GlobalWriteVectorWidth"] \
              + r*kernel["GlobalWriteVectorWidth"]*kernel["NumGlobalWriteVectorsPerThread"]

          if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
            cIdx //= elementStep
            regIdx //= elementStep
            kStr += inst("v_pk_add_f16", vgpr("ValuC+%u"%cIdx), \
                vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), "c[%u] += c[%u]"%(cIdx, regIdx) )
          elif kernel["ProblemType"]["DataType"].isInt8x4():
            cIdx //= elementStep
            regIdx //= elementStep
            # assume v_add_i32 can be used in place of v_add_f32
            # may need to add saturation directive to v_add_i32 instruction to clamp integer arithmetic
            kStr += inst("v_add_i32", vgpr("ValuC+%u"%cIdx), \
                vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), "c[%u] += c[%u]"%(cIdx, regIdx) )
          elif kernel["ProblemType"]["DataType"].isSingle():
            cIdx //= elementStep
            regIdx //= elementStep
            kStr += inst("v_add_f32", vgpr("ValuC+%u"%cIdx), \
                vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), "c[%u] += c[%u]"%(cIdx, regIdx) )
          elif kernel["ProblemType"]["DataType"].isDouble():
            cIdx *= 2
            regIdx *= 2 # for doubles, each element takes two regs
            kStr += inst("v_add_f64", vgpr("ValuC+%u"%cIdx,2), \
                vgpr("ValuC+%u" % regIdx,2), vgpr("ValuC+%u"%cIdx,2), "c[%u] += c[%u]"%(cIdx, regIdx) )
          elif kernel["ProblemType"]["DataType"].isSingleComplex():
            cIdx *= 2
            regIdx *= 2
            kStr += inst("v_add_f32", vgpr("ValuC+%u"%cIdx), \
                vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), "c[%u] += c[%u], real part"%(cIdx, regIdx) )
            kStr += inst("v_add_f32", vgpr("ValuC+%u"%(cIdx+1)), \
                vgpr("ValuC+%u" % (regIdx+1)), vgpr("ValuC+%u"%(cIdx+1)), "c[%u] += c[%u], imaginary part"%(cIdx+1, regIdx+1) )
          elif kernel["ProblemType"]["DataType"].isDoubleComplex():
            cIdx *= 4
            regIdx *= 4
            kStr += inst("v_add_f64", vgpr("ValuC+%u"%cIdx), \
                vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), "c[%u] += c[%u], real part"%(cIdx, regIdx) )
            kStr += inst("v_add_f64", vgpr("ValuC+%u"%(cIdx+2)), \
                vgpr("ValuC+%u" % (regIdx+2)), vgpr("ValuC+%u"%(cIdx+2)), "c[%u] += c[%u], imaginary part"%(cIdx+2, regIdx+2) )
          else:
            assert(0) # unsupported data type, need to modify here and LSU write/read code
    return kStr

  ##############################################################################
  # computeStoreSrd
  # Add tile assignment fields to store srd
  # This is based on WG not the WI/TT assignment
  ##############################################################################
  def computeStoreSrdStart(self, kernel):
    kStr = ""

    tmpS0 = self.getTmpSgpr(3).idx()
    tmpS1 = tmpS0+1
    wgMT1 = tmpS0+2

    # Compute and save wg1*MT1 - the element offset that is top of the macro-tile in output space
    assert kernel["BufferStore"]
    kStr += "\n"
    kStr += inst("s_mul_i32", \
        sgpr(wgMT1), \
        "MT1", \
        sgpr("WorkGroup1"), \
        "<- wg1*MT1")

    # Overall strategy is to set the SRD to the top-left of the macro-tile.
    # TT offsets are from this base (and include the column)

    # In non-packed mode:
    # higher-order tensor dims are static since this kernel operates within
    # the 2D Tensor formed by Index0 and Indexa.
    # Index0 and Index1 vary for each work-item (aka 'dynamic') so roll these into the VGPR

    # In packed mode:
    # Higher-order dimensions may be packed into coord0 / coord1 - see rowstart calculation below

    # Walk through addressing components (each tensor index) in C
    # For static dims add to SrdC / SrdD to compute a new base.
    # For dynamic (based on TT assignment) - save in coutRowPtr in computeStoreVgprs,
    # which saves the TT assignment for each WI scaled by StrideC0
    # TODO - future opportunities for store vgpr and other optimization
    #  - coutRowPtr and tid1 are strongly related - can we merge or remove one of these?
    # Packed follows same philosophy but may have more vector components
    indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
    numDim = len(indices)
    for i in range(1, numDim):
      if i == kernel["ProblemType"]["Index0"]:
        # Used if the output is transposed?
        addToSrd = False
      elif i == kernel["ProblemType"]["Index1"]:
        coord = sgpr(wgMT1)
        addToSrd = True
      elif not isPackedIndex(kernel, i):
        # group index, this is higher-order Tensor dimension, just add to SRD base:
        coord = sgpr("WorkGroup2")
        addToSrd = True
      else:
        # could be packed higher-order index, just ignore
        coord = None
        addToSrd = False

      if addToSrd:
        # These are constant across all workitems, just add to the SRD:
        strideC = "StrideC%s"%self.indexChars[i]
        kStr += self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(strideC), "CScale %s by Stride"%coord)
        #kStr += assert_no_shift_of(tmpS1, log2(self.bpeCexternal), "Need temp")
        kStr += inst("s_lshl_b64", sgpr(tmpS0,2), sgpr(tmpS0,2), log2(self.bpeCexternal), "scale by bpe")

        kStr += inst("s_add_u32",  sgpr("SrdC+0"), sgpr("SrdC+0"), sgpr(tmpS0), "add lo to SRD")
        kStr += inst("s_addc_u32", sgpr("SrdC+1"), sgpr("SrdC+1"), sgpr(tmpS1), "add hi to SRD")

        if not kernel["LdcEqualsLdd"]:
          # These are constant across all workitems, just add to the SRD:
          stride = "StrideD%s" % (self.indexChars[i])
          kStr += self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(stride), "Scale %s by Stride"%coord)
          #kStr += assert_no_shift_of(tmpS1, log2(self.bpeCexternal), "Need temp")
          kStr += inst("s_lshl_b64", sgpr(tmpS0,2), sgpr(tmpS0,2), log2(self.bpeCexternal), "scale by bpe")

        kStr += inst("s_add_u32",  sgpr("SrdD+0"), sgpr("SrdD+0"), sgpr(tmpS0), "add lo to SRD")
        kStr += inst("s_addc_u32", sgpr("SrdD+1"), sgpr("SrdD+1"), sgpr(tmpS1), "add hi to SRD")

        kStr += "\n"

    for cdir in (0,1):
      indices = kernel["PackedC%uIndicesX"%cdir]
      packedSizes = "PackedSize%u"%cdir
      if len(indices) > 1:
        for i,idx in enumerate(indices[1:]):
          if i==0:
            kStr += inst("s_mul_i32", sgpr(packedSizes), self.sizeRef(indices[0]), \
                      self.sizeRef(idx), "first packed size")
          else:
            kStr += inst("s_mul_i32", sgpr(packedSizes), sgpr(packedSizes), \
                      self.sizeRef (idx), "first packed size")


    return kStr

  ##############################################################################
  # computeStoreVgprs
  # Compute workitem/TT offsets in VGPRS
  # and coord0/coord1
  # tid0Scale specifies the number of output elements in 0/coalesced dim
  # that should be written by each work-item in each batch element.
  ##############################################################################
  def computeStoreVgprsVALU(self, kernel, divisor, tid0Scale, tid1Scale):

    kStr = ""

    tmpS0 = self.getTmpSgpr(3).idx()
    tmpS1 = tmpS0+1
    wgMT1 = tmpS0+2

    if self.prefetchAcrossPersistent:
      wg0="PrevWorkGroup0"
      wg1="PrevWorkGroup1"
    else:
      wg0="WorkGroup0"
      wg1="WorkGroup1"

    # tid0, tid1: element offsets from the start of macroTile in 0 and 1 direction
    # These will live for entire GlobalWrite loop - allocate before tmps
    # to avoid fragmentation
    tid0 = self.vgprPool.checkOut(1, "tid0")
    tid1 = self.vgprPool.checkOut(1, "tid1")

    if kernel["BufferStore"]:
      self.cinRowPtr  = self.vgprPool.checkOut(1, "cinRowPtr")
      if not kernel["LdcEqualsLdd"]:
        self.coutRowPtr = self.vgprPool.checkOut(1, "coutRowPtr")

    tmpV0 = self.vgprPool.checkOut(2)
    kStr += vectorStaticDivideAndRemainder(tid1, tid0, "Serial", divisor, \
        tmpV0, tmpS0)
    kStr += staticMultiply(vgpr(tid0), vgpr(tid0), tid0Scale, sgpr(tmpS1))
    if tid1Scale != 1:
      kStr += staticMultiply(vgpr(tid1), vgpr(tid1), tid1Scale, sgpr(tmpS1))
    self.vgprPool.checkIn(tmpV0)

    if kernel["BufferStore"]:
      # compute rowStart- this is just tid1 scaled by appropriate stride.
      # rowPtr is offset from the beginning of the tile/SRD not the tensor base
      # when incremented, it moves in units of (col) Stride to get to a new row
      # it is used for address computation, not element range detection.
      # rowPtr is in the element space and must be scaled by bpe if bytes are required.
      # Do this before code below which overwries the tid1:
      # TODO-packed
      # Eventually need to modify if supporting packed coord1, to start just assert if that case is detected
      #--
      packedC1 = kernel["PackedC1IndicesX"]
      assert (len(packedC1) == 1) # would need to extract/scale indices from coord1
      strideC1 = "StrideC%s" % (self.indexChars[packedC1[0]])
      kStr += inst("v_mul_lo_u32", vgpr(self.cinRowPtr),
                  vgpr(tid1), sgpr(strideC1), \
                  "rowStart vgpr")
      if not kernel["LdcEqualsLdd"]:
        strideD1 = "StrideD%s" % (self.indexChars[packedC1[0]])
        kStr += inst("v_mul_lo_u32", vgpr(self.coutRowPtr),
                    vgpr(tid1), sgpr(strideD1), \
                    "rowStart vgpr")
      kStr += "\n"

      #kStr += self.assert_ne(sgpr("WorkGroup1"),1)

    # Compute coord0 and coord1
    # These are element offsets from the beginning of the tensor.
    # These are 'flattened' meaning they span packed tensor dims.
    # They need to be preserved so can use in comparisons against
    # product-of-packed sizes to determine OOB cases. (for Edge tiles only)
    kStr += inst("s_mul_i32", \
        sgpr(tmpS0), \
        hex(kernel["MacroTile0"]), \
        sgpr(wg0), \
        "%s = wg0*MT0"%sgpr(tmpS0))

    # coord = tid*VW + workgroup offset
    kStr += inst("_v_add_co_u32", \
        vgpr(tid0), \
        "vcc", \
        sgpr(tmpS0), \
        vgpr(tid0), \
        "coord0 = tid0*VW + wg0*MT0")
    kStr += inst("s_mul_i32", \
        sgpr(wgMT1), \
        hex(kernel["MacroTile1"]), \
        sgpr(wg1), \
        "<- wg1*MT1")
    kStr += inst("_v_add_co_u32", \
        vgpr(tid1), \
        "vcc", \
        sgpr(wgMT1), \
        vgpr(tid1), \
        "coord1 = tid1*VW + wg1*MT1")

    self.coord0 = tid0
    self.coord1 = tid1

    return kStr


  ##############################################################################
  # computeStoreVgprs
  # Compute workitem/TT offsets in VGPRS
  # and coord0/coord1
  # tid0Scale specifies the number of output elements in 0/coalesced dim
  # that should be written by each work-item in each batch element.
  ##############################################################################
  def computeStoreVgprsMFMA(self, kernel, divisor, tid0Scale, tid1Scale):

    # self.coord0
    # self.coord1
    # self.cinRowPtr  : C buffer coulmn offset
    # self.coutRowPtr : D buffer coulmn offset

    # alloc resources
    tid0 = self.vgprPool.checkOut(1)
    tid1 = self.vgprPool.checkOut(1)
    if kernel["BufferStore"]:
      self.cinRowPtr  = self.vgprPool.checkOut(1, "cinRowPtr")
      if not kernel["LdcEqualsLdd"]:
        self.coutRowPtr = self.vgprPool.checkOut(1, "coutRowPtr")

    wave_id = self.vgprPool.checkOut(1)

    tmpVgpr0 = self.vgprPool.checkOut(1,"tmpVgpr0")
    tmpVgpr1 = self.vgprPool.checkOut(2,"tmpVgpr1")
    dummy    = self.vgprPool.checkOut(1,"dummy")
    tmpSgpr  = self.getTmpSgpr(1).idx()

    # constant
    MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
    MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

    kStr = ""

    # coord 1 : wave part
    kStr += vectorStaticDivide(wave_id, "Serial", globalParameters["WavefrontWidth"], tmpVgpr1, tmpSgpr)
    kStr += vectorStaticDivide(tid1, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr)
    kStr += inst("v_mul_lo_u32", vgpr(tid1), hex(MIBShape1), vgpr(tid1), "wave coordination offset 1")

    # coord 1 : thread part
    kStr += vectorStaticRemainder(dummy, tmpVgpr0, "Serial", kernel["MatrixInstN"], tmpVgpr1, tmpSgpr)
    kStr += inst("v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "coordination 1 = wave_id1 + tid1")


    if kernel["MatrixInstM"] == 4:
      divisor =  kernel["MatrixInstN"] * kernel["MatrixInstBM"]
      kStr   += vectorStaticRemainder(dummy, tmpVgpr0, "Serial", globalParameters["WavefrontWidth"], tmpVgpr1, tmpSgpr)
      kStr   += vectorStaticDivide(tmpVgpr0, tmpVgpr0, divisor, tmpVgpr1, tmpSgpr)
      kStr   += staticMultiply(vgpr(tmpVgpr0), vgpr(tmpVgpr0), kernel["MatrixInstN"], sgpr(tmpSgpr))
      kStr   += inst("v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "coordination 1 = wave_id1 + tid1")

    # coord 1 : offset part
    kStr += inst("v_mul_lo_u32", vgpr(self.cinRowPtr), vgpr(tid1), sgpr("StridesC"), " offset 1")

    # coord 0 : wave part
    kStr += vectorStaticRemainder(dummy, tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr)
    kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape0), vgpr(tmpVgpr0), "wave coordination offset 0")

    # coord 0 : thread part
    kStr += vectorStaticRemainder(dummy, tid0, "Serial", globalParameters["WavefrontWidth"], tmpVgpr1, tmpSgpr)
    kStr += vectorStaticDivide(tid0, tid0, kernel["MatrixInstM"], tmpVgpr1, tmpSgpr)
    if kernel["MatrixInstM"] == 4:
      kStr += vectorStaticRemainder(dummy, tid0, tid0, kernel["MatrixInstBM"], tmpVgpr1, tmpSgpr)
    kStr += inst("v_lshlrev_b32", vgpr(tid0), hex(2), vgpr(tid0), "thread0 * 4 : mfma output 4 continuous outputs")
    kStr += inst("v_add_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), "coordination 0 = wave_id0 + tid0")

    # macro tile 0 part
    kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile0"], sgpr("WorkGroup0"), "wgp0 * MT0")
    kStr += inst("v_add_u32", vgpr(tid0), sgpr(tmpSgpr), vgpr(tid0), "coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0")

    # macro tile 1 part
    kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile1"], sgpr("WorkGroup1"), "wgp1 * MT1")
    kStr += inst("v_add_u32", vgpr(tid1), sgpr(tmpSgpr), vgpr(tid1), "coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1")

    # release resource
    self.vgprPool.checkIn(dummy)
    self.vgprPool.checkIn(tmpVgpr1)
    self.vgprPool.checkIn(tmpVgpr0)
    self.vgprPool.checkIn(wave_id)

    self.coord0 = tid0
    self.coord1 = tid1

    return kStr


  ##############################################################################
  # computeStoreVgprs
  # Compute workitem/TT offsets in VGPRS
  # and coord0/coord1
  # tid0Scale specifies the number of output elements in 0/coalesced dim
  # that should be written by each work-item in each batch element.
  ##############################################################################
  def computeStoreVgprs(self, kernel, divisor, tid0Scale, tid1Scale):

    kStr = ""
    kStr += self.comment1("computeStoreVgprs")

    if kernel["EnableMatrixInstruction"]:
      kStr += self.computeStoreVgprsMFMA(kernel, divisor, tid0Scale, tid1Scale)
    else:
      kStr += self.computeStoreVgprsVALU(kernel, divisor, tid0Scale, tid1Scale)

    return kStr


  ##############################################################################
  # globalWriteWorkGroupInit:
  ##############################################################################
  def globalWriteWorkGroupInit(self, kernel):
    kStr = ""
    if kernel["BufferStore"]:
      kStr += self.allocPostLoopSrd(kernel, "D")
      kStr += self.allocPostLoopSrd(kernel, "C")
      kStr += self.computeStoreSrdStart(kernel)
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    kStr = ""

    # lr0 = serial % SG0
    kStr += self.computeStoreVgprs(kernel, \
              divisor = kernel["MacroTile0"] // kernel["GlobalWriteVectorWidth"], \
              tid0Scale=kernel["GlobalWriteVectorWidth"], \
              tid1Scale=1)

    if kernel["BufferStore"]:
      #print "----AddressC-LocalSplitU"
      #print self.vgprPool.state()
      self.addrD = -1
      self.addrC = -1
    else:
      self.addrD = self.vgprPool.checkOut(2)
      kStr += inst("v_mov_b32", \
          vgpr(self.addrD+0), \
          sgpr("AddressD+0"), \
          "sgpr -> vgpr")
      kStr += inst("v_mov_b32", \
          vgpr(self.addrD+1), \
          sgpr("AddressD+1"), \
          "sgpr -> vgpr")
      self.addrC = self.vgprPool.checkOut(2)
      kStr += inst("v_mov_b32", \
          vgpr(self.addrC+0), \
          sgpr("AddressC+0"), \
          "sgpr -> vgpr")
      kStr += inst("v_mov_b32", \
          vgpr(self.addrC+1), \
          sgpr("AddressC+1"), \
          "sgpr -> vgpr")

    return kStr

  ##############################################################################
  ##############################################################################
  def allocPostLoopSrd(self, kernel, ch):
    kStr = ""
    # Buffer-load uses one base read pointer stored in the SRD - set it here:
    kStr += inst("s_mov_b32", sgpr("Srd%s+0"%ch), sgpr("Address%s+0"%ch), "init SRD base address (lower)" )
    kStr += inst("s_mov_b32", sgpr("Srd%s+1"%ch), sgpr("Address%s+1"%ch), "init SRD base address (upper) + other fields" )
    kStr += inst("s_mov_b32", sgpr("Srd%s+2"%ch), hex(0x80000000), "")
    kStr += inst("s_mov_b32", sgpr("Srd%s+3"%ch), "Srd127_96", "Set bits 127_96 in post-loop SRD")
    kStr += "\n"
    return kStr


  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    #print "GlobalWriteIndices"
    if not self.do["PostLoop"]: return ""
    kStr = ""

    kStr += self.computeStoreVgprs(kernel,
              divisor = kernel["SubGroup0"],\
              tid0Scale=kernel["VectorWidth"], \
              tid1Scale=kernel["VectorWidth"])

    if kernel["BufferStore"]:
      #print "----AddressC-nonLSU-----"
      #print self.vgprPool.state()
      self.addrD = -1
      self.addrC = -1
    else:
      self.addrD = self.vgprPool.checkOut(2, 'addrD')
      kStr += inst("v_mov_b32", \
          vgpr(self.addrD+0), \
          sgpr("AddressD+0"), \
          "sgpr -> vgpr")
      kStr += inst("v_mov_b32", \
          vgpr(self.addrD+1), \
          sgpr("AddressD+1"), \
          "sgpr -> vgpr")
      self.addrC = self.vgprPool.checkOut(2, 'addrC')
      kStr += inst("v_mov_b32", \
          vgpr(self.addrC+0), \
          sgpr("AddressC+0"), \
          "sgpr -> vgpr")
      kStr += inst("v_mov_b32", \
          vgpr(self.addrC+1), \
          sgpr("AddressC+1"), \
          "sgpr -> vgpr")
    return kStr

  ##############################################################################
  # Release any resources used by the global write
  def cleanupGlobalWrite(self, kernel):
    self.vgprPool.checkIn(self.coord0)
    self.vgprPool.checkIn(self.coord1)

    if kernel["BufferStore"]:
      self.vgprPool.checkIn(self.cinRowPtr)
      if not kernel["LdcEqualsLdd"]:
        self.vgprPool.checkIn(self.coutRowPtr)
    if not kernel["BufferStore"]:
      self.vgprPool.checkIn(self.addrD)
      self.vgprPool.checkIn(self.addrC)

    if self.betaVgpr != None:
      self.vgprPool.checkIn(self.betaVgpr)

  ##############################################################################
  # Return max global write vector width, in elements
  def maxGwvw(self, kernel):
    atomic = kernel["GlobalSplitU"] > 1

    if kernel["BufferStore"]:
      if atomic:
        return kernel["VectorAtomicWidth"]
      else:
        return 1000  # no limit
    else:
      if atomic:
        return 1  # flat vector atomic is not tested
      else:
        return 1000  # no limit

  ##############################################################################
  # Partition thread-tile into writeElements for store code
  # This function creates the writeElement mapping for full tiles
  # (ie non-edge cases)
  ##############################################################################
  def notLocalFullTileElementsVALU(self, kernel, edge):
    elements    = []
    vectorwidth = 0

    if edge:
      vectorwidth = kernel["VectorWidth"] if kernel["_VectorStore"] else 1
      vectorwidth = min(vectorwidth, self.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
      assert(kernel["VectorWidth"] % vectorwidth == 0)
    else:
      vectorwidth = kernel["VectorWidth"] if kernel["_VectorStore"] else 1
      vectorwidth = min(vectorwidth, self.maxGwvw(kernel))

    # Full tile loop:
    for tt1 in range(0, kernel["ThreadTile1"]//kernel["VectorWidth"]):
      for vc1 in range(0, kernel["VectorWidth"]):
        for tt0 in range(0, kernel["ThreadTile0"]//kernel["VectorWidth"]):
          for vc0 in range(0, kernel["VectorWidth"], vectorwidth): # note step by fullVw
            element = (tt1, tt0, vc1, vc0)
            elements.append(element)

    return (vectorwidth, elements)


  ##############################################################################
  # Partition thread-tile into writeElements for store code
  # This function creates the writeElement mapping for full tiles
  # (ie non-edge cases)
  ##############################################################################
  def notLocalFullTileElementsMFMA(self, kernel, edge):
    elements    = []
    vectorwidth = 0

    if edge:
      vectorwidth = kernel["StoreVectorWidth"] if kernel["_VectorStore"] else 1
      vectorwidth = min(vectorwidth, self.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
    else:
      vectorwidth = kernel["StoreVectorWidth"] if kernel["_VectorStore"] else 1
      vectorwidth = min(vectorwidth, self.maxGwvw(kernel))

    MFMAcontinoutsOuptut = kernel["MIOutputVectorWidth"]

    if kernel["MatrixInstM"] == 4:
      totalTT0             = kernel["MIWaveTile"][0] * MFMAcontinoutsOuptut
      totalTT1             = kernel["MIWaveTile"][1]
    else:
      outputsPerThread     = kernel["MatrixInstM"] * kernel["MatrixInstN"] // globalParameters["WavefrontWidth"]
      totalTT0             = kernel["MatrixInstBM"] * kernel["MIWaveTile"][0] * outputsPerThread
      totalTT1             = kernel["MatrixInstBN"] * kernel["MIWaveTile"][1]

    for tt1 in range(0, totalTT1):
      for vc1 in range(0, 1):
        for tt0 in range(0, totalTT0 // MFMAcontinoutsOuptut):
          for vc0 in range(0, MFMAcontinoutsOuptut, vectorwidth): # note step by vectorwidth
            element = (tt1, tt0, vc1, vc0)
            elements.append(element)

    return (vectorwidth, elements)


  ##############################################################################
  # Partition thread-tile into writeElements for store code
  # This function creates the writeElement mapping for full tiles
  # (ie non-edge cases)
  ##############################################################################
  def notLocalFullTileElements(self, kernel, edge):
    if kernel["EnableMatrixInstruction"]:
        return self.notLocalFullTileElementsMFMA(kernel, edge)
    else:
        return self.notLocalFullTileElementsVALU(kernel, edge)


  ##############################################################################
  # Not LocalSplitU: Global Write
  # Determine write batching pattern
  # element() specifies TT 'coordinate' to write
  # vectorWidths specifies width of vector to store
  # TODO - why does this use VectorWidth to control store width?  Could be GlobalWriteVectorWidth?
  #
  # Function creates one mapping for full tiles and one for edge tiles,
  # then calls globalWriteElements to generate the code for the new tiles.
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""
    elements = [[] for y in range(2)] # 2D array for Full, Edge

    (fullVw, elements[False]) = self.notLocalFullTileElements(kernel, False)
    (edgeVw, elements[True])  = self.notLocalFullTileElements(kernel, True)

    vectorWidths = [fullVw, edgeVw]

    kStr = self.globalWriteElements(kernel, vectorWidths, elements)

    self.cleanupGlobalWrite(kernel)

    return kStr


  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""

    fullVw = kernel["GlobalWriteVectorWidth"] if kernel["_VectorStore"] else 1
    fullVw = min(fullVw, self.maxGwvw(kernel))
    elements = [[] for y in range(2)] # 2D array for Full, Edge
    # Full tile loop:
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for vc1 in range(0, 1):
        for tt0 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], fullVw): # note step by fullVw
            element = (tt1, tt0, vc1, vc0)
            elements[False].append(element)

    # Edge tile loop - note if we know AF0EM we can can use a larger vector
    # and reduce the boundary checks accordingly.  But if no AF0EM guarantee
    # then use a conservative 1
    edgeVw = kernel["GlobalWriteVectorWidth"] if kernel["_VectorStore"] else 1
    edgeVw = min(edgeVw, self.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
    assert(kernel["GlobalWriteVectorWidth"]%edgeVw == 0)
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for vc1 in range(0, 1):
        for tt0 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], edgeVw):
            element = (tt1, tt0, vc1, vc0)
            elements[True].append(element)

    vectorWidths = [fullVw, edgeVw]
    kStr =  self.globalWriteElements(kernel, vectorWidths, elements)
    self.cleanupGlobalWrite(kernel)
    return kStr




  ##############################################################################
  # StoreState
  # tracks state that is preserved across globalWriteBatch calls:
  # init is called before globalWriteBatch
  # the KernelWriter object
  ##############################################################################
  class StoreState:

    ##############################################################################
    # Setup store config for number of sgpr and vgpr needed
    # These are set based on edge, atomic, etc - do not change during
    # the generation of the store code.
    ##############################################################################
    class StoreConstConfig:
      def __init__(self, kernelWriter, kernel, ss, gwvw, edge, beta, atomic):

        self.gwvw = gwvw


        if ss.optSingleColVgpr:
          # use one vgpr (allocated in ss.sharedColVgprs) for all addressing
          # - need 0 additional vgpr per element.
          self.numVgprsPerAddr = 0
        else:
          self.numVgprsPerAddr = kernelWriter.rpgo if kernel["BufferStore"] else kernelWriter.rpga

        if ss.optSharedMask:
          self.numSgprsPerElement = 0
          self.fixedSgprsPerBatch = 2
        else:
          self.numSgprsPerElement = 2
          self.fixedSgprsPerBatch = 6

        if self.numSgprsPerElement:
          numSgprAvailable = kernelWriter.maxSgprs - kernelWriter.sgprPool.size() + kernelWriter.sgprPool.availableBlockAtEnd()
          numSgprAvailable = numSgprAvailable & ~0x1 # make sure it's aligned
          #print("numSgprAvailable=", numSgprAvailable)
          self.numElementsPerBatchLimitedBySgprs = (numSgprAvailable - self.fixedSgprsPerBatch) // self.numSgprsPerElement
        else:
          self.numElementsPerBatchLimitedBySgprs = 9999 # no limit

        if self.numElementsPerBatchLimitedBySgprs<=0:
          kernelWriter.overflowedResources = 2
          self.numElementsPerBatchLimitedBySgprs = 1 # dummy value
            #assert self.numElementsPerBatchLimitedBySgprs > 0, "numElementsPerBatchLimitedBySgprs=0 for %s"%self.kernelName


        if atomic:
          # flat atomics have another VGPR to allow different data for return#
          regsPerElement = 2 if kernel["BufferStore"] else 3
          # The atomic loop processes multiple elements in single instruction
          # so will use VGPR from consec elements? TODO
          self.numVgprsPerDataPerVI = (1.0*regsPerElement*kernelWriter.bpeCexternal)/kernelWriter.bpr
        elif beta:
          self.numVgprsPerDataPerVI = (1.0*kernelWriter.bpeCexternal)/kernelWriter.bpr
        else:
          self.numVgprsPerDataPerVI = 0.0

        # indicates each vector element is actually half -
        # changes vgpr allocation so two elements share a data vgpr
        # Really only used if gwvw=1 - edge cases
        self.halfDataRegPerVI = True if gwvw*self.numVgprsPerDataPerVI < 1.0 else False

    # StoreState constructor:
    def __init__(self, kernelWriter, kernel, gwvw, edge, beta, atomic, elements):
      self.kernelWriter = kernelWriter
      self.kernel = kernel

      #--
      # Optimizations for coord0/column address calculations:
      #
      # optSingleColVgpr:
      #  - works in cases where the data is written row by row to memory.
      # In this case we can use a single vgpr for addressing:
      #  - Use the load/store instruction offset (fixed at compile-time)
      #  - the horizontal addresses are fixed offsets from the base
      #  - as we move to a new row, increment the appropriate SRDs

      # optSharedColVgpr:
      #  - Each col gets it's own address, but elements in later rows with the same col will share VGPR.
      #  - allows cols to be non-adjacent
      #  - this is mutually exclusive with optSingleColVgpr - not as optimal but provides
      #    more flexibility.

      # optSrdIncForRow: optimize coord1/row address calculations:
      #  - Move the SRD bewtween memory operations to get to new row
      #    atomic needs to reset the SRD to handle retry loop.  Then might work.

      self.optSingleColVgpr = 0
      self.optSharedColVgpr = 0
      self.optSrdIncForRow  = 0

      # opt*ColVgpr doesn't work for edge since each element requires own addr VGPR so
      #    we can perform bounds check and set to -1 for OOB accesses.
      # if optSingleColVgpr = optSharedColVgpr = 0, then each element gets
      #  1-2 VGPRs to track address.  Address calcs are performed independently
      #  for each element.

      # atomic contains multiple memory operations which need to preserve
      # the address for each load.  Memops in same row can use offsets
      # and share a base register but Memops in different rows need
      # different registers or need to inteliigently reset the SRD.
      if kernel["BufferStore"] and not edge and not atomic:
        if len(kernel["PackedC0IndicesX"]) > 1:
          # packed mode needs a unique VGPR address calc for each column.
          self.optSharedColVgpr = 1
        else:
          self.optSingleColVgpr = 1

        if not atomic and len(kernel["PackedC1IndicesX"]) == 1:
          self.optSrdIncForRow = 1

      if kernel["ProblemType"]["UseInitialStridesCD"]:
        self.optSingleColVgpr = 0 # BOZO, hack to disable this
        self.optSharedColVgpr = 0# BOZO, hack to disable this

      self.optSharedMask  = kernel["BufferStore"] and not edge and not atomic


      # can't have both of these enabled:
      assert (not (self.optSingleColVgpr and self.optSharedColVgpr))

      # need another set of col VGPR to preserve scaled LDD
      assert (not (self.optSharedColVgpr and not kernel["LdcEqualsLdd"]))

      # packed1 not yet supported.  Would need to:
      # - extract packed dimensions from coord1 into 
      assert( len(kernel["PackedC1IndicesX"]) == 1)

      self.cfg = self.StoreConstConfig(kernelWriter, kernel, self, gwvw, edge, beta, atomic)

      # Use to detect new rows:
      self.lastCoordOffset1 = 0

      # vgpr holding current coord, setup initial state
      self.coord1Vgpr = kernelWriter.coord1

      if self.optSharedColVgpr:
        numCols = len([e for e in elements if e[0] == 0 and e[2] == 0]) # count #elements with row d1=v1==0
        self.numAddrVgpr = numCols
        self.sharedColVgprs = kernelWriter.vgprPool.checkOut(self.numAddrVgpr, "sharedColVgprs for packed elements")
      elif self.optSingleColVgpr:
        self.numAddrVgpr = 1
        self.sharedColVgprs = kernelWriter.vgprPool.checkOut(1, "sharedColVgprs")
      else:
        self.numAddrVgpr = 0
        self.sharedColVgprs = None

      # For detecting when we are running first batch
      self.firstBatch = True


    ##############################################################################
    # Setup data structures to feed store loops:
    #   self.elementAddr, self.elementData, self.elementMask, self.elementSumIdx
    # batchElements is a list of (d0,d1,v0,v1) for which stores to perform
    # batchElementSgprs is SGPRs to use for mask.  If None, elementMask is
    #  not initialized.
    #
    # Also create an AddrCalc for each memory operation.
    ##############################################################################
    def setupStoreElementsForBatch(self, kernel, gwvw, batchElements, batchElementSgprs, isOptNLL):

      self.elementAddr = []
      self.elementData = []  # VGPR to use for element data, needed for atomic or beta
      self.elementMask = []  # SGPR to use for element mask
      self.elementSumIdx = []

      kw = self.kernelWriter

      lastData = 0
      for elementIdx in range(0, len(batchElements)):
        # Create the AddrCalc for each memory load/store
        # This is the control code that sets up the dest, source, offsets, etc and
        # identifies cases where the AddrCalc is a new row and therefore needs some
        # additional math.  Each AddrCalc contains isolated state sufficient to
        # perform any needed range checks and address calculations for the element.
        #
        # The AddrCalc creation code here maintains state across elements (including
        # across write batches) to remove replicated calculations.
        #
        # Later the AddrCalc::emitAddressSetupCode will emit the necessary code
        # Also allocate VGPR resources here, if needed.

        element = batchElements[elementIdx]
        (d1,d0,vc1,vc0) = element

        coordOffset1 = 0
        if kernel["EnableMatrixInstruction"]:
          if kernel["MatrixInstM"] == 4:
            coordOffset1 = d1 * kernel["MatrixInstN"] *  kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] + vc1
          else:
            bIdx1  = d1 % kernel["MatrixInstBN"]
            wtIdex = (d1 // kernel["MatrixInstBN"]) % kernel["MIWaveTile"][1]

            coordOffset1  = bIdx1 * kernel["MatrixInstN"]
            coordOffset1 += wtIdex * kernel["MatrixInstN"] *  kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1]
            coordOffset1 += vc1
        else:
          if kernel["LocalSplitU"] > 1:
            strideD1 = (kernel["NumThreads"]*kernel["VectorWidth"]//kernel["MacroTile0"])
          else:
            strideD1 = (kernel["SubGroup1"] * kernel["VectorWidth"])
          coordOffset1 = d1 * strideD1 + vc1

        newCoord1 = (self.firstBatch and elementIdx==0) or (coordOffset1 != self.lastCoordOffset1)

        # gpr and offset assignments for element
        coordOffset0 = 0
        if kernel["EnableMatrixInstruction"]:
          if kernel["MatrixInstM"] == 4:
            coordOffset0 = d0 * kernel["MatrixInstM"] *  kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0] + vc0
          else:
            MFMAContinuousOutputs = kernel["MIOutputVectorWidth"]
            OutputsPerMIMN        = kernel["MatrixInstM"] * kernel["MatrixInstN"] // globalParameters["WavefrontWidth"]

            eIdx0        = d0 % (OutputsPerMIMN // MFMAContinuousOutputs)
            remain_d0    = d0 // (OutputsPerMIMN // MFMAContinuousOutputs)
            bIdx0        = remain_d0 % kernel["MatrixInstBM"]
            remain_d0    = remain_d0 // kernel["MatrixInstBM"]
            wtIdex       = remain_d0 % kernel["MIWaveTile"][0]

            coordOffset0  = eIdx0  * (globalParameters["WavefrontWidth"] // kernel["MatrixInstN"]) * MFMAContinuousOutputs
            coordOffset0 += bIdx0  * kernel["MatrixInstM"]
            coordOffset0 += wtIdex * kernel["MatrixInstM"] *  kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0]
            coordOffset0 += vc0
        else:
          coordOffset0 = d0 * kernel["SubGroup0"]*kernel["VectorWidth"] + vc0

        if self.optSingleColVgpr:
          # use same address vgpr for all
          addr = self.sharedColVgprs
        elif self.optSharedColVgpr:
          if kernel["EnableMatrixInstruction"]:
            elementCol = (d0 * gwvw + vc0) / gwvw
          else:
            elementCol = (d0 * kernel["VectorWidth"] + vc0) / gwvw
          assert (modf(elementCol)[0] < 0.001)
          elementCol = trunc(elementCol)
          addr = self.sharedColVgprs+elementCol
          #print ("d0=", d0, "vc0=", vc0, "elementCol=", elementCol)
        else:
          # allocate new VGPR for each element:
          addr = kw.vgprPool.checkOut(self.cfg.numVgprsPerAddr, \
              "writeBatch-addr for ei=%u"%(elementIdx), preventOverflow=not isOptNLL)

        self.elementAddr.append(kw.AddrCalc(kw, self, addr, element, coordOffset0, \
          self.kernelWriter.coord1, coordOffset1, coordOffset1 - self.lastCoordOffset1, newCoord1))
        # if numVgprsPerDataPerVI == 0.5, then two consecutive elements
        # should have same data pointer, next should move.

        if self.cfg.numVgprsPerDataPerVI > 0:
          if self.cfg.halfDataRegPerVI:
            if kernel["ProblemType"]["HighPrecisionAccumulate"] and \
               (kernel["ProblemType"]["DataType"].isBFloat16() or kernel["ProblemType"]["DataType"].isHalf()):
              data = kw.vgprPool.checkOut(int(2*self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw), \
                    "writeBatch-data for ei=%u and ei=%u"%(elementIdx,elementIdx+1), preventOverflow=not isOptNLL)
            else:
              if elementIdx%2 == 0:
                # allocate for two elements:
                data = kw.vgprPool.checkOut(int(2*self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw), \
                        "writeBatch-data for ei=%u and ei=%u"%(elementIdx,elementIdx+1), preventOverflow=not isOptNLL)
                lastData = data
              else:
                data = lastData
                del lastData
          else:
            data = kw.vgprPool.checkOut(int(self.cfg.numVgprsPerDataPerVI*self.cfg.gwvw), \
                  "writeBatch-data for ei=%u"%elementIdx, preventOverflow=False)
        else:
          data = 0

        self.elementData.append(data)
        if batchElementSgprs != None:
          mask = batchElementSgprs + elementIdx * self.cfg.numSgprsPerElement # elementSgprs+0
          self.elementMask.append(mask)

        #print "Edge=", edge, element
        sumIdx = 0
        if kernel["LocalSplitU"] > 1:
          sumIdx = kw.startVgprValuC + vc0 + d1*kernel["VectorWidth"]
        else:
          bestVw = kernel["VectorWidth"]
          elementsLoadedPerVw = kernel["NumThreads"]*bestVw
          elementsLoadedPerbestVw = kernel["NumThreads"]*kernel["StoreVectorWidth"]
          if elementsLoadedPerVw < elementsLoadedPerbestVw:
            bestVw = kernel["StoreVectorWidth"]
          if kernel["EnableMatrixInstruction"]:
            if kernel["MatrixInstM"] == 4:
              sumIdx    = kw.startVgprValuC + vc0 + (d0 * kernel["MIOutputVectorWidth"]) + d1 * (kernel["MIOutputVectorWidth"] * kernel["MIWaveTile"][0])
            else:
              d1_stride = ((kernel["MatrixInstM"] * kernel["MatrixInstN"]) // globalParameters["WavefrontWidth"]) * kernel["MatrixInstBM"] * kernel["MIWaveTile"][0]
              sumIdx    = kw.startVgprValuC + vc0 + (d0 * kernel["MIOutputVectorWidth"]) + (d1 * d1_stride)
          else:
            sumIdx = kw.startVgprValuC + vc0 + d0*kernel["VectorWidth"] + vc1*kernel["ThreadTile0"] + d1*kernel["VectorWidth"]*kernel["ThreadTile0"]
        self.elementSumIdx.append(sumIdx) # sumIdx is an element idx, need to div/2 for half
        self.lastCoordOffset1 = coordOffset1

    def __del__(self):
      if (self.sharedColVgprs != None):
        self.kernelWriter.vgprPool.checkIn(self.sharedColVgprs)


  ##############################################################################
  # Fields associated with computing address
  ##############################################################################
  class AddrCalc:
    # rowInc is number of rows to add to the base address
    # coord0Vgpr : This is VGPR that holds coord0.  Coord0 is element-space
    #    packed index for the 0 coordinate of the C/D matrix.
    # coord1Vgpr : VGPR which tracks the last coord1 calculation.
    #          If this is new coord1, just overwrite it with latest calc.
    def __init__(self, kernelWriter, ss, addrVgpr, element, \
        coordOffset0, coord1Vgpr, coordOffset1, rowInc, newCoord1):
      self.kernelWriter = kernelWriter

      # vgprs for address, could be more than one (for flat)
      # If LdcEqualsLdd and if the store batch needs a load (beta or atomic),
      # then the value in the addrVgpr will be recomputed with LDD
      # after all loads have completed.
      self.addrVgpr = addrVgpr
      self.coord1Vgpr = coord1Vgpr # vpgpr that stores coord1Vgpr

      self.element = element
      self.coordOffset0 = coordOffset0
      self.coordOffset1 = coordOffset1
      self.rowInc = rowInc
      self.rowIncDirtyRowPtr = 0 # rowInc was used to modify rowPtr, need to recompute addr
      self.newCoord1 = newCoord1 # vgpr that stores newCoord1

      if ss.optSingleColVgpr:
        # optimized stores use the load offset for coordOffset0 calculations.
        self.globalOffset = coordOffset0 * kernelWriter.bpeCexternal
      else:
        # else non-opt stores include the coord0 offset into VGPR address calcs
        self.globalOffset = 0

    """
    Use minimally efficient instructions to add stride*scale
    """
    def addScaled(self, destV, src0, src1, scale1, tmpS01, comment=""):
      kStr = ""
      if scale1 == 1:
        kStr += inst("_v_add_u32", destV, src0, \
                  src1, comment)
      else:
        kStr += inst("s_mul_i32", sgpr(tmpS01), src1, scale1, "scale stride")
        kStr += inst("_v_add_u32", destV, src0,  \
                        sgpr(tmpS01), comment)
      return kStr


    """
    Emit code that computes the coord0 and coord1 for this element
    sets self.coord0Vgpr with the address that holds the coord0 value for this element.
    Input:
      - tmpVgpr is a 1 temporary VGPR used for coord0 calculation on edges
        If LdcEqualsLdd we could re-use addrVgpr here and perhaps save the temp.

    """
    def emitAddressCoordIncrement(self, kernel, ss, tmpVgpr, tmpS01, edge):
      kStr = ""
      kw = self.kernelWriter
      (d1,d0,vc1,vc0) = self.element
      self.coord0Vgpr = None # will set below

      #kStr += self.kernelWriter.comment1("store addr=v%u coordOffset0=%u"% \
      #    (self.addr, self.coordOffset0))
      kStr += self.kernelWriter.comment1("(d1,vc1,d0,vc0)=(%u,%u,%u,%u)"\
          % (d1,vc1,d0,vc0))
      if ss.optSingleColVgpr:
        self.coord0Vgpr = kw.coord0
      elif not ss.optSharedColVgpr or (d1 == vc1 == 0):
        # not share mode or first row always does the address calc math:

        if self.coordOffset0 == 0:
          self.coord0Vgpr = kw.coord0
        elif self.coordOffset0 <= 64:
          self.coord0Vgpr = tmpVgpr
          kStr += inst("_v_add_co_u32", vgpr(self.coord0Vgpr), "vcc", vgpr(kw.coord0), self.coordOffset0, \
                    "coord0.1: coord0 += d0*sg0*VW + vc0")
        else:
          self.coord0Vgpr = tmpVgpr
          kStr += inst("s_mov_b32", sgpr(tmpS01), self.coordOffset0, "coordOffset0 d0=%u vc0=%u"%(d0, vc0))
          kStr += inst("_v_add_co_u32", vgpr(self.coord0Vgpr), "vcc", vgpr(kw.coord0), sgpr(tmpS01), \
                    "coord0.2: coord0 += d0*sg0*VW + vc0")

        if self.newCoord1:
          if not kernel["BufferStore"] or edge: #TODO, do we need edge?
            if self.rowInc== 0:
              None
            elif self.rowInc <= 64:
              # rowInc fits in instruction:
              kStr += inst("_v_add_co_u32", vgpr(self.coord1Vgpr), "vcc", \
                        vgpr(self.kernelWriter.coord1), self.rowInc, \
                        "coord1.1: coord1Vgpr += d1*sg1*VW + vc1")
            else:
              kStr += inst("s_mov_b32", sgpr(tmpS01), self.rowInc, "rowInc d1=%u vc1=%u"%(d0, vc0))
              kStr += inst("_v_add_co_u32", vgpr(self.coord1Vgpr), "vcc", \
                        vgpr(self.kernelWriter.coord1), sgpr(tmpS01), \
                        "coord1.2: coord1 += d1*sg1*VW + vc1")
      return kStr

    # storeChar is 'C' or 'D'
    # elementVgpr is coord0Vgpr*strideCD0, or optimized to just coord0Vgpr if strideCD0 is unit const
    def emitExtractAndScalePackedDims(self, kernel, ss, beta, atomic, tmpVgpr, storeChar):
      kStr = ""
      kw = self.kernelWriter
      packedIndices = kernel["PackedC0IndicesX"]
      packedBits = self.coord0Vgpr # start with coord0, will move to temp below
      for i,idx in enumerate(packedIndices[:-1]):
        # vgprTmp assignments:
        #   - tmp+0 may be the incoming packed coordinate 0, used on replay too
        #   - tmp+1 is DIV output
        #   - tmp+2 is scratch
        idxChar= globalParameters["IndexChars"][idx]
        kStr += kw.comment1("extract %s"%kw.sizeRef(idx))
        assert(tmpVgpr+1 != packedBits) # bad since we still need packedBits below for remainder (can't overwrite here)
        kStr += "V_MAGIC_DIV %s, %s, %s, %s, %s\n" % \
                 (tmpVgpr+1, vgpr(packedBits), sgpr("MagicNumberSize%s"%idxChar), \
                  sgpr("MagicShiftSize%s"%idxChar), sgpr("MagicAbitSize%s"%idxChar) if kernel["MagicDivAlg"]==2 else "0")
        # tmpVgpr+1 returns the quotient, tmpVgpr+2 is overwritten

        # compute remainder, packedBits % sizeIdx - this is the 'extracted' index that must be scaled
        # remainder is mul and sub
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+2), vgpr(tmpVgpr+1), kw.sizeRef(idx), \
                     "remainder part 1")
        kStr += inst("_v_sub_u32", vgpr(tmpVgpr+2), vgpr(packedBits), vgpr(tmpVgpr+2),
                      "remainder part 2")

        if i==0:
          kStr += inst("v_mul_lo_u32", vgpr(self.addrVgpr), vgpr(tmpVgpr+2), \
                    kw.strideRef(storeChar, idx), "addrCalc <- scaled extracted dim")
        else:
          kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+2), vgpr(tmpVgpr+2), \
                    kw.strideRef(storeChar, idx), "scale extracted dim")
          kStr += inst("_v_add_u32", vgpr(self.addrVgpr), vgpr(self.addrVgpr), \
                    vgpr(tmpVgpr+2), "addrCalc += scaled extracted dim ")

        if i < len(packedIndices)-2:
          # TODO - might be able to eliminate this
          kStr += inst("v_mov_b32", vgpr(tmpVgpr+0), vgpr(tmpVgpr+1), \
                    "Copy remaining bits for next divide")
          packedBits = tmpVgpr+0

      if len(packedIndices)>1:
        # if we unpacked something, then scale it to BPE
        kStr += kw.comment1("extract final %s"%kw.sizeRef(packedIndices[-1]))
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+2), vgpr(tmpVgpr+1), \
                  kw.strideRef(storeChar, packedIndices[-1]), "scale final extracted dim")
        kStr += inst("_v_add_u32", vgpr(self.addrVgpr), vgpr(self.addrVgpr), \
                  vgpr(tmpVgpr+2), "addrCalc += scaled extracted dim ")

        kStr += inst("_v_add_lshl_u32", vgpr(self.addrVgpr), \
                  vgpr(kw.cinRowPtr), \
                  vgpr(self.addrVgpr), \
                  hex(log2(kw.bpeCexternal)), \
                  "packed: add rowPtr and scaleToBpe")

      return kStr

    """
    Needs 3 temporary VGPRs
    """
    def emitScaleToBpe(self, kernel, ss, beta, atomic, tmpVgpr, firstInBatch):
      kStr = ""
      kw = self.kernelWriter
      (d1,d0,vc1,vc0) = self.element

      # set when we generate code that updates the address
      # optSingleColVgpr and optSharedColVgpr attempt to minimize these updates
      updatedAddr = False

      # scale and set final address:
      if kernel["LdcEqualsLdd"] or beta or atomic:
        strideC0 = kw.strideRef('C', 0)
        if kw.isConstUnitStride(strideC0):
          elementVgpr = self.coord0Vgpr
        else:
          kStr += inst("v_mul_lo_u32", \
              vgpr(self.addrVgpr), \
              vgpr(self.coord0Vgpr), \
              strideC0, \
              "scale element by non-unit stride")
          elementVgpr = self.addrVgpr

        if ss.optSingleColVgpr:
          # This is first element in the first batch, create a byte address that will
          # be re-used by subsequent elements:
          # if this element is firstInBatch - may need to set up a bpe-scaled row pointer for the batch:
          #  - for not LdcEqualsLdd - need row-ptr start of each batch
          #  - or always init rowptr at the start of the first batch (so can be re-used in each batch)
          assert (kw.coord0 == self.coord0Vgpr) # elementAddr assignment above assumes these are the same

          if (not kernel["LdcEqualsLdd"] or ss.firstBatch) and firstInBatch:
            updatedAddr = True
            kStr += inst("_v_add_lshl_u32", \
              vgpr(self.addrVgpr), \
              vgpr(kw.cinRowPtr), \
              vgpr(elementVgpr), \
              hex(log2(kw.bpeCexternal)), \
              "optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=%d, coord0Vgpr=%d"%(kw.coord0, self.coord0Vgpr))
        elif ss.optSharedColVgpr:
          # Need an address calculation for the first address in each row:
          if d1==0 and vc1==0:
            packedIndices = kernel["PackedC0IndicesX"]
            if len(packedIndices) > 1:
              updatedAddr = True
              kStr += self.emitExtractAndScalePackedDims(kernel, ss, beta, atomic, tmpVgpr, 'C')
            else:
              updatedAddr = True
              kStr += inst("_v_add_lshl_u32", \
                vgpr(self.addrVgpr), \
                vgpr(kw.cinRowPtr), \
                vgpr(elementVgpr), \
                hex(log2(kw.bpeCexternal)), \
                "optSharedColVgpr scaleToBpe for first row: col addr <- cinRowPtr + coord0, scaled by BPE")
        else:
          # Generate final address calculation (to bytes) for each element
          # The unpacking takes 8-10 instructions so could be worth optimizing someday :
          # each col has same offset so could create a class to hold column-specific state including
          # the byte address offset for that col and the mask in/out.
          packedIndices = kernel["PackedC0IndicesX"]
          if len(packedIndices) > 1:
            updatedAddr = True
            kStr += self.emitExtractAndScalePackedDims(kernel, ss, beta, atomic, tmpVgpr, 'C')
          else:
            updatedAddr = True
            kStr += inst("_v_add_lshl_u32", \
                vgpr(self.addrVgpr), \
                vgpr(kw.cinRowPtr), \
                vgpr(elementVgpr), \
                hex(log2(kw.bpeCexternal)), \
                "scaleToBpe: accumulate d0 lower and *= bpe into Cin addr")

        # if not optSrdIncForRow then we may have moved the row pointer
        # and depending on paths above may not have refreshed addrVgpr already.
        # if so - do it here:
        if self.rowIncDirtyRowPtr and not updatedAddr:
          kStr += inst("_v_add_lshl_u32", \
            vgpr(self.addrVgpr), \
            vgpr(kw.cinRowPtr), \
            vgpr(kw.coord0), \
            hex(log2(kw.bpeCexternal)), \
            "scaleToBpe: Update address with new rowPtr")

      return kStr

    """
    Generate code to set up the address vgpr
    Input:
      tmpVgpr : two temp vgprs
    Output:
      Returns kStr with appropriate setup code
      Sets self.coord0Vgpr with vgpr that contains the coord0 for this element.  This enables
        optimization - if no setup code is required the coord0 can be the input.
    """
    # TODO - mask should be part of AddrCalc state not passed as parm
    def emitAddressSetupCode(self, kernel, ss, tmpVgpr01, tmpS01, edge, beta, atomic, mask, elementIdx):
      kStr = ""
      kw = self.kernelWriter

      kStr += self.emitAddressCoordIncrement(kernel, ss, tmpVgpr01, tmpS01, edge)

      # Move the row ptr VGPR
      # optSrdIncForRow moves the SRD so don't move here
      if not ss.optSrdIncForRow and kernel["BufferStore"]:
        if self.rowInc > 0:
          self.rowIncDirtyRowPtr = 1
          #assert (not kernel["ProblemType"]["UseInitialStridesCD"])
          kStr += kw.comment("Fix for UseInitialStridesCD, emitAddressSetupCode")
          assert (len(kernel["PackedC1IndicesX"])==1)

          strideChar = self.kernelWriter.indexChars[kernel["PackedC1IndicesX"][0]]
          kStr += self.addScaled(vgpr(kw.cinRowPtr),  vgpr(kw.cinRowPtr),  \
                    sgpr("StrideC%s"%strideChar), self.rowInc, tmpS01, "ROWINC- Move cinRowPtr to next row")
          if not kernel["LdcEqualsLdd"]:
            kStr += self.addScaled(vgpr(kw.coutRowPtr), vgpr(kw.coutRowPtr), \
                      sgpr("StrideD%s"%strideChar), self.rowInc, tmpS01, "Move coutRowPtr to next row")

      # Shift Pointer for MFMA:
      #   For MFMA shift pointer, correct data is stored in another thread.
      #   Therefore, MFMA cannot use v_mov to amend store data
      #   It needs to modify the coord1 of thread directly.
      if not kernel["GuaranteeNoPartialB"] and kw.readTileDimVectorB and kernel["EnableMatrixInstruction"] and edge:
        (d1,d0,vc1,vc0) = self.element
        if (d1 == vc1 == d0 == vc0 == 0) or self.newCoord1:
          packedC1 = kernel["PackedC1IndicesX"]
          strideC1 = "StrideC%s" % (kw.indexChars[packedC1[0]])
          kStr += kw.comment("shift vector components d1")
          vw = kernel["GlobalLoadVectorWidthB"]
          vTmp1 = tmpVgpr01
          vTmp2 = tmpVgpr01+1
          sTmp1 = tmpS01
          sTmp2 = tmpS01+2
          # check conditions
          kStr += inst("v_bfi_b32", vgpr(vTmp1), vw-1, 0, vgpr(self.coord1Vgpr), "coord1 & ~(vw-1)")
          kStr += inst("v_bfi_b32", vgpr(vTmp2), vw-1, 0, sgpr("SizesFree+%u"%kw.tPB["idx"]), "sizeFree & ~(vw-1)")
          kStr += inst("v_cmp_eq_u32", sgpr(sTmp1,2), vgpr(vTmp1), vgpr(vTmp2), "if coord1 is in edge glvw")
          kStr += inst("v_and_b32", vgpr(vTmp2), sgpr("SizesFree+%u"%kw.tPB["idx"]), vw-1, "sizeFree mod VW")
          kStr += inst("v_cmp_gt_u32", sgpr(sTmp2,2), vgpr(vTmp2), 0, "this problem is not multiple size of glvw")
          kStr += inst("s_and_b64", sgpr(sTmp1,2), sgpr(sTmp1,2), sgpr(sTmp2,2), "AND both conditions")
          # calculate new coord
          kStr += inst("v_add_u32", vgpr(vTmp1), vgpr(self.coord1Vgpr), vgpr(vTmp2), "shift coord1")
          kStr += inst("v_bfi_b32", vgpr(vTmp1), vw-1, vgpr(vTmp1), sgpr("SizesFree+%u"%kw.tPB["idx"]), "new coord1 = (shift coord1 & (vw-1)) |  (sizeFree & ~(vw-1))")
          kStr += inst("v_sub_i32", vgpr(vTmp2), vgpr(vTmp1), vgpr(self.coord1Vgpr), "shift how many column")
          kStr += inst("v_cndmask_b32", vgpr(self.coord1Vgpr), vgpr(self.coord1Vgpr), vgpr(vTmp1), \
                        sgpr(sTmp1,2), "set new coord1 if meet conditions" )
          kStr += inst("v_mul_i32_i24", vgpr(vTmp2), vgpr(vTmp2), sgpr(strideC1), "shift column *  StridesC")
          kStr += inst("v_add_i32", vgpr(vTmp1), vgpr(kw.cinRowPtr), vgpr(vTmp2), "new rowStart address")
          kStr += inst("v_cndmask_b32", vgpr(kw.cinRowPtr), vgpr(kw.cinRowPtr), vgpr(vTmp1), \
                        sgpr(sTmp1,2), "set new rowStart if meet conditions" )
          kStr += "\n"


      # Now do the edge check and compute the address in bytes:
      if kernel["BufferStore"]:
        if edge:
          # Set address to -1 if OOB on either dimension
          # and only check the x/coord0 index here, save a couple inst
          sizeBoundary = [0,0]
          sizeBoundary[0] = \
              sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
              else kw.sizeRef(kernel["ProblemType"]["Index0"])
          sizeBoundary[1] = \
              sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
              else kw.sizeRef(kernel["ProblemType"]["Index1"])

          kStr += inst("v_cmp_lt_u32",  sgpr(tmpS01,2), vgpr(self.coord0Vgpr), \
                    sizeBoundary[0], "coord0 < size0" )
          kStr += inst("v_cmp_lt_u32",  sgpr(mask,2), vgpr(self.coord1Vgpr), \
                    sizeBoundary[1], "coord1 < size1" )
          kStr += inst("s_and_b64",  sgpr(mask,2), sgpr(tmpS01,2), sgpr(mask,2), "in0 && in1" )

          kStr += self.emitScaleToBpe(kernel, ss, beta, atomic, tmpVgpr01, elementIdx==0)

          if kernel["LdcEqualsLdd"] or beta or atomic:
              kStr += inst("v_cndmask_b32", vgpr(self.addrVgpr), -1, vgpr(self.addrVgpr), \
                        sgpr(mask,2), "clip if OOB. offset" )
        else:
          kStr += self.emitScaleToBpe(kernel, ss, beta, atomic, tmpVgpr01, elementIdx==0)

      return kStr


    def emitLddChange(self, kernel, ss, edge, mask, isLastElement):
      kStr = ""
      kw = self.kernelWriter

      # Set write address for D - this overwrites self.addrVgpr
      if kernel["BufferStore"]:
        strideD0 = kw.strideRef('D', 0)
        if kw.isConstUnitStride(strideD0):
          elementVgpr = self.coord0Vgpr
        else:
          kStr += inst("v_mul_lo_u32", \
              vgpr(self.addrVgpr), \
              vgpr(self.coord0Vgpr), \
              strideD0, \
              "scale element by non-unit stride")
          elementVgpr = self.addrVgpr

        if ss.optSingleColVgpr:
          assert (kw.coord0 == self.coord0Vgpr)
          if isLastElement:
            kStr += inst("_v_add_lshl_u32", \
              vgpr(self.addrVgpr), \
              vgpr(kw.coutRowPtr), \
              vgpr(elementVgpr), \
              hex(log2(kw.bpeCexternal)), \
              "emitLddChange: init cb addr <-  coutRowPtr + coord0, scaled by BPE")
        elif ss.optSharedColVgpr:
          (d1,d0,vc1,vc0) = self.element
          assert(0) # need a new VGPR for LDD != LDC here.
          # since we are re-using the column VGPRs in the next batch
        else: # not ss.optSingleColVgpr or optSharedColVgpr

          kStr += inst("_v_add_lshl_u32", \
              vgpr(self.addrVgpr), \
              vgpr(kw.coutRowPtr), \
              vgpr(elementVgpr), \
              hex(log2(kw.bpeCexternal)), \
              "emitLddChange: accumulate d0 lower and *= bpe into addr")

        if edge:
          kStr += inst("v_cndmask_b32", vgpr(self.addrVgpr), -1, \
                    vgpr(self.addrVgpr), sgpr(mask,2), \
                    "LDD clip if OOB. offset")

      return kStr

    """
    Generate code to move to the next row(s)
    If optSrdIncForRow, this will move the SRD forward
    If not, this could generate some other instructions
    """
    def incrementToNextRow(self, kernel, tc, ss, stmp):
      kStr = ""
      numRows = self.rowInc
      if ss.optSrdIncForRow:
        if numRows:
          packedC1 = kernel["PackedC1IndicesX"]
          assert(len(packedC1) == 1)  # would need to extract each dim and scale
          strideCD1 = "Stride%s%s"%(tc,self.kernelWriter.indexChars[packedC1[0]])
          if numRows > 1:
            kStr += inst("s_mul_i32", sgpr(stmp), \
                         sgpr(strideCD1), \
                         numRows*self.kernelWriter.bpeCexternal, \
                         "scale Stride%s *= numRows(%u) * bpe"%(tc,numRows))
          else:
            kStr += inst("s_lshl_b32 ", \
                  sgpr(stmp), \
                  sgpr(strideCD1), \
                  log2(self.kernelWriter.bpeCexternal), \
                  "incToNextRow: Scale by BPE")

          kStr += inst("s_add_u32 ", \
               sgpr("Srd%s+0"%(tc)), \
               sgpr("Srd%s+0"%(tc)), \
               sgpr(stmp), \
               "incToNextRow: gra SRD += inc(lower)" )
          kStr += inst("s_addc_u32 ", \
               sgpr("Srd%s+1"%(tc)), \
               sgpr("Srd%s+1"%(tc)), \
               0, \
               "incToNextRow: gra SRD += inc(upper)" )

        None

      return kStr

  ##############################################################################
  # checkIsBetaZero
  # tmpSgpr is one temp sgpr
  # betaLabel is label to branch to if beta != 0
  ##############################################################################
  def checkIsBetaZero(self, kernel, tmpSgpr, betaLabel):
    kStr = ""
    if kernel["ProblemType"]["UseBeta"]:
      if self.bpeCinternal <= self.bpr: # 1 register to check for Beta==0
        kStr += inst("s_cmpk_eq_u32", sgpr("Beta"), hex(0), "Beta == 0")
      else: # multiple registers to check for Beta==0
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), sgpr("Beta+0"), "tmp = Beta[0]")
        for i in range(1, self.bpeCinternal//self.bpr):
          kStr += inst("s_or_b32", sgpr(tmpSgpr), sgpr("Beta+%u"%i), sgpr(tmpSgpr), "tmp |= Beta[%u] " % i)
        kStr += inst("s_cmpk_eq_u32", sgpr(tmpSgpr), hex(0), "Beta == 0")
      kStr += inst("s_cbranch_scc0 %s" % betaLabel, \
          "Branch if Beta is not zero")
      kStr += "\n"
    return kStr

  ##############################################################################
  # checkIsEdge
  # tmpSgpr must have at least 6 free SGPR
  # isEdgeTarget is the branch target if edges are required
  ##############################################################################
  def checkIsEdge(self, kernel, tmpSgpr, isEdgeTarget):
    kStr = ""
    tmpS01 = tmpSgpr
    tmpS23 = tmpS01 + 2
    tmpS45 = tmpS23 + 2

    if self.prefetchAcrossPersistent:
      wg0="PrevWorkGroup0"
      wg1="PrevWorkGroup1"
    else:
      wg0="WorkGroup0"
      wg1="WorkGroup1"

    # check edge0 ###
    # s23 = rMT0 = Size0 % MT0
    #--
    sizeBoundary = [0,0]
    sizeBoundary[0] = \
        sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
        else self.sizeRef(kernel["ProblemType"]["Index0"])
    sizeBoundary[1] = \
        sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
        else self.sizeRef(kernel["ProblemType"]["Index1"])

    kStr += scalarStaticDivideAndRemainder(tmpS23, tmpS01, sizeBoundary[0], \
        kernel["MacroTile0"], tmpS45, 2)
    # s23 = nwg0-1
    kStr += inst("s_add_u32", sgpr(tmpS23), hex(-1), sgpr("NumWorkGroups0"), "" )
    kStr += inst("s_cmp_ge_u32", sgpr(wg0), sgpr(tmpS23), "wg0 >= nwg0-1 ?")
    kStr += inst("s_cselect_b32", sgpr(tmpS01), sgpr(tmpS01), 0, "set rMT0")
    # s01 now = myMT0 = wg0 < nwg0-1 ? MT0 : rMT0

    # if rMT0 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      kStr += inst("s_cmpk_gt_u32", sgpr(tmpS01), hex(0), "rMT0 > 0")
      if self.db["ForceEdgeStores"]:
        kStr += inst("s_cmp_eq_u32", sgpr(tmpS01), sgpr(tmpS01), "ForceEdgeStores!")
      kStr += inst("s_cbranch_scc1 %s" % isEdgeTarget, "jump if edges required")

    # check edge1 ###
    # TODO-packed - this only needs to change to handle packing into C1 index
    # change would be similar to above - multiply by product of packed sizes in C1
    # --

    # s23 = rMT1 = Size1 % MT1
    kStr += scalarStaticDivideAndRemainder(tmpS23, tmpS01, sizeBoundary[1], \
        kernel["MacroTile1"], tmpS45, 2)
    # s01 now = myMT1 = wg1 < nwg1-1 ? MT1 : rMT1

    # s23 = nwg1-1
    kStr += inst("s_add_u32", sgpr(tmpS23), hex(-1), sgpr("NumWorkGroups1"), "" )
    kStr += inst("s_cmp_ge_u32", sgpr(wg1), sgpr(tmpS23), "wg1 >= nwg1-1")
    kStr += inst("s_cselect_b32", sgpr(tmpS01), sgpr(tmpS01), 0, "set rMT1")

    # if rMT1 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      kStr += inst("s_cmpk_gt_u32", sgpr(tmpS01), hex(0), "rMT1 > 0")
      kStr += inst("s_cbranch_scc1 %s" % isEdgeTarget, "jump if edges required")

    return kStr


  ##############################################################################
  # Global Write Elements
  ##############################################################################
  def globalWriteElements(self, kernel, vectorWidths, elements):
    if not self.do["PostLoop"]: return ""
    kStr = ""
    atomic = kernel["GlobalSplitU"] > 1


    # write possibilities and labels
    betas = [False, True] if kernel["ProblemType"]["UseBeta"] else [False]
    edges = [False, True] if self.do["EdgeWrite"] else [False]
    writeLabels = {}
    for beta in betas:
      writeLabels[beta] = {}
      for edge in edges:
        writeLabels[beta]["EdgeCheck0"] = self.getNamedLabel("GW_B%u_E%u_EdgeCheck0" % ( 1 if beta else 0, 1 if edge else 0) )
        writeLabels[beta]["EdgeCheck1"] = self.getNamedLabel("GW_B%u_E%u_EdgeCheck1" % ( 1 if beta else 0, 1 if edge else 0) )
        writeLabels[beta][edge] = self.getNamedLabel("GW_B%u_E%u" % ( 1 if beta else 0, 1 if edge else 0) )
      if not beta:
        betaLabel = self.getNamedLabel("GW_Beta")
    endLabel = self.getLabelNum("GW_End")

    # Layout
    """
    if B1 goto label_B1
    if E1 goto label_B0_E1
    label_B0_E0:
    writes
    goto label_End
    label_B0_E1:
    writes
    goto label_End
    label_B1:
    if E1 goto label_B1_E1
    label_B1_E0:
    writes
    goto label_End
    label_B1_E1:
    writes
    goto label_End
    label_End
    """
    self.betaVgpr = None
    # Also can push alpha/beta recalc back to host for HPA mode?
    if kernel["ProblemType"]["DataType"].isHalf():
      if kernel["ProblemType"]["HighPrecisionAccumulate"]:
        alphaVgprTmp = self.vgprPool.checkOut(1, "alpha")
        # alpha, beta are packed halfs in half mode (f16.hi == f16.lo) - setup on host
        kStr += inst("v_mov_b32", vgpr(alphaVgprTmp), sgpr("Alpha"), "sgpr -> vgpr b/c op_sel")
        kStr += inst("v_cvt_f32_f16", vgpr(alphaVgprTmp), vgpr(alphaVgprTmp), "convert alpha to fp32")
        kStr += inst("v_readfirstlane_b32", sgpr("Alpha"), vgpr(alphaVgprTmp), "restore alpha sgpr")
        self.vgprPool.checkIn(alphaVgprTmp)

      if beta:
#jgolds look at moving these converted values back to scalar regs and free up the VGPRs
# TODO - for hpa the host should pass in an F32 alpha so we don't have to do it here
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          self.betaVgpr = self.vgprPool.checkOut(1, "beta")
          kStr += inst("v_mov_b32", vgpr(self.betaVgpr), sgpr("Beta"), "sgpr -> vgpr b/c op_sel")
          kStr += inst("v_cvt_f32_f16", vgpr(self.betaVgpr), vgpr(self.betaVgpr), "convert beta to fp32")
          if self.betaInSgpr:
            kStr += inst("v_readfirstlane_b32", sgpr("Beta"), vgpr(self.betaVgpr), "restore beta sgpr")
            self.vgprPool.checkIn(self.betaVgpr)
            self.betaVgpr = None

    ########################################
    # Vgprs
    if kernel["BufferStore"]:
      numTmpVgpr = 2
      if len(kernel["PackedC0IndicesX"]) > 1:
        numTmpVgpr += 1
    else:
      numTmpVgpr = 2 + 3 # GLOBAL_OFFSET_C needs 3, plus 2 tmps?
    tmpVgpr = self.vgprPool.checkOut(numTmpVgpr,"store tmps")

    ########################################
    # Sgprs

    # allocate tmps for the store header (before the batch implementations)
    tmpSgpr = self.getTmpSgpr(6).idx()

    # branch B1 or B0
    betaLabel = self.getNamedLabel("GW_Beta")
    kStr += self.checkIsBetaZero(kernel, tmpSgpr, betaLabel)

    for beta in betas:
      # start B1
      if beta:
        kStr += "%s:\n"%(betaLabel)

      ########################################
      # branch if Edge0 or Edge1
      kStr += self.checkIsEdge(kernel, tmpSgpr, "%s" % writeLabels[beta][True])

      # by now we either jumped to E1 or stayed at E0
      for edge in edges:
        kStr += "%s:%s"%(writeLabels[beta][edge], self.endLine)

        edgeI = edge
        #edgeI = True  # set to True to disable vector stores
        gwvw = vectorWidths[edgeI]
        #print "globalWriteElements: edge=", edge, "beta=", beta, "atomic=", atomic

        ########################################
        # Calculate Vgprs for Write Batching
        ########################################

        self.ss = self.StoreState(self, kernel, gwvw, edge, beta, atomic, elements[edgeI])

        # how many vgprs are needed for zero elements
        # 2 for addressC in vgpr for addition - already checked out
        # 2 for coord0,1 of thread - already checked out
        # 2 for tmp - already checked out

        # 5 = how many vgprs are needed per element (flat)
        #  - 2 for addr
        #  - 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
        #  - if beta gwvw*rpe for new value
        #  - if atomic 2*rpe for old and cmp values

        numVgprsPerElement = self.ss.cfg.numVgprsPerAddr + int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw))

        #print self.vgprPool.state()
        # Use VGPR up to next occupancy threshold:
        #numVgprAvailable = self.getMaxRegsForOccupancy(self.vgprPool.available())
        numVgprAvailable = self.vgprPool.availableBlock(numVgprsPerElement)

        # Grow the register pool if needed - we need enough regs for at least one element
        # Unfortunate since this means the write logic is setting the VGPR requirement
        # for the entire kernel but at least we have a functional kernel.
        # Before growing the pool, see if we can shrink the write vector width instead?
        # TODO : the vgprSerial is needed for-ever and if we grow here will split the
        # range of the tmps.  Maybe want to move vgprSerial to first vgpr?
        minElements = 2 if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) else 1
        minNeeded = minElements*numVgprsPerElement
        shrinkDb = 0
        if shrinkDb:
          print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)
        if numVgprAvailable < minNeeded:
          gwvwOrig = gwvw
          currentOccupancy = self.getOccupancy(kernel, self.vgprPool.size())
          futureOccupancy = self.getOccupancy(kernel, \
              self.vgprPool.size() - numVgprAvailable + minNeeded)

          if shrinkDb:
            print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
                % (currentOccupancy, futureOccupancy, self.vgprPool.size(), \
                   numVgprAvailable, minElements*numVgprsPerElement))
          if futureOccupancy > currentOccupancy:
            if shrinkDb:
              print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                    (self.kernelName))
              print("   numVgprAvailable=", numVgprAvailable, \
                    "numVgprsPerElement=", numVgprsPerElement, "atomic=", atomic, \
                    "beta=", beta, "gwvw=", gwvw)
          elif gwvw != gwvwOrig:
            self.ss.gwvw = gwvw # make both representations consistent
            if shrinkDb:
              print("info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                  % (self.kernelName, gwvwOrig, gwvw, currentOccupancy))

          if numVgprAvailable < minElements*numVgprsPerElement:
            print("info: growing pool += %d * %d for GlobalWrite\n" \
                % (minElements,numVgprsPerElement))
            print(self.vgprPool.state())
            tl = []
            for i in range(0,minElements):
              tl.append(self.vgprPool.checkOut(numVgprsPerElement, "grow-pool for GlobalWrite"))
            for t in tl:
              self.vgprPool.checkIn(t)
            numVgprAvailable = self.vgprPool.available()
            print(self.vgprPool.state())

        # set atomicW after we potentially resize GWVW
        atomicW = min(gwvw, kernel["VectorAtomicWidth"])

        #print "NumVgprAvailable", numVgprAvailable
        if numVgprsPerElement:
          numElementsPerBatch = numVgprAvailable // numVgprsPerElement
        else:
          numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

        if shrinkDb:
          print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", self.ss.cfg.numElementsPerBatchLimitedBySgprs, \
              "WARNING" if self.ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
        if self.ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
          numElementsPerBatch = self.ss.cfg.numElementsPerBatchLimitedBySgprs

        if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
          # only do an even number of halves - since these share hi/lo pieces of some registers?
          if numElementsPerBatch > 1:
            numElementsPerBatch = int(numElementsPerBatch/2)*2
          elif not kernel["EnableMatrixInstruction"]:
            # The globalWriteBatch routine below can't handle odd elements per batch
            # and 0 elements per batch is illegal.
            # so if we don't have *GPR resources to handle a larger batch then need
            # to mark overflowedResources rather than generate a kernel that won't work.
            # It might be possible to fix globalWriteBatch to handle this case but these
            # are likely to be low-performing so likely not worth optimizing.
            if shrinkDb:
              print("WARNING: half requires at least two elements per batch")
            self.overflowedResources = 3

        assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%self.kernelName

        #numElementsPerBatch=min(2,numElementsPerBatch) # hack to control number of batches
        if atomic and (self.ss.optSingleColVgpr or self.ss.optSharedColVgpr):
          # hack to avoid re-using address vgpr across rows
          # atomics need to perform several memory operations
          # if the batch spans multiple rows, need multiple address vgpr
          # which is not currently supported in the two opt*ColVgpr modes
          firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0]
          numElementsPerBatch=min(len(firstRow),numElementsPerBatch)

        # if no atomics and no edge, then write whole vectors
        #if not atomic and not edge:
        #  numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
        #  #print "  NumVectorsPerBatch", numVectorsPerBatch
        #  numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
        numBatches = max(1, ceil_divide(len(elements[edgeI]),numElementsPerBatch))

        numSgprs = self.ss.cfg.fixedSgprsPerBatch + self.ss.cfg.numSgprsPerElement*numElementsPerBatch
        if self.db["PrintStoreRegisterDb"]:
          print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
          print ("numSgprs=", numSgprs, "sgprPool.size()=", self.sgprPool.size(), \
                  "fixedSgprsPerBatch=", self.ss.cfg.fixedSgprsPerBatch, "numSgprsPerElement=", self.ss.cfg.numSgprsPerElement)
          print(self.sgprPool.state())
        kStr += self.comment("edge=%d, allocate %u sgpr. perBatch=%u perElement=%u elementsPerBatch=%u"%\
            (edgeI, numSgprs, self.ss.cfg.fixedSgprsPerBatch, self.ss.cfg.numSgprsPerElement, numElementsPerBatch))
        #kStr += "// storeStats, %d, %d, %d\n"% (edgeI, numSgprs, numElementsPerBatch)
        # so if we don't have *GPR resources to handle a larger batch then need
        # to mark overflowedResources rather than generate a kernel that won't work.
        tmpSgpr = self.getTmpSgpr(numSgprs, 2).idx()

        elementSgprs = tmpSgpr + self.ss.cfg.fixedSgprsPerBatch


        for batchIdx in range(0, numBatches):
          elementStartIdx = batchIdx * numElementsPerBatch
          elementStopIdx = min( elementStartIdx + numElementsPerBatch, len(elements[edgeI]) )
          elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
          #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,numVgprsPerElement ))
          # elementVgprs can be large and should be perfectly tuned to the number of available
          # VGPRS.  We do not want to accidentally overflow and grow the pool here:

          kStr += self.globalWriteBatch(kernel, self.ss, batchIdx, beta, edge, atomic, gwvw, atomicW, \
              elementsThisBatch, self.coord0, self.coord1, self.addrD, self.addrC, \
              tmpVgpr, \
              elementSgprs, tmpSgpr)
        # TODO - if this is the last tile, don't need to jump to next instruction
        kStr += inst("s_branch", "label_%04u"%endLabel, "jump to end")

    # End label
    kStr += "label_%04u:%s"%(endLabel, self.endLine)
    self.vgprPool.checkIn(tmpVgpr)
    return kStr


  ##############################################################################
  # chooseGlobalRead :
  # create the load instruction for requested vector width and other parms
  # return an Inst class
  #
  # bpl = bytes per load op
  ##############################################################################
  def chooseGlobalRead(self, useBuffer, bpl, destVgpr, \
                       addr0, addr1, soffset, offset, extraFields, hi16=0, comment="load C"):

  # rpv = regs per vector
    rpv = bpl/4.0

    if self.version == (10,1,0):
      extraFields += " glc, slc, dlc"

    if useBuffer:
      tailFields = "offen offset:%u"%offset
      if extraFields != "":
        tailFields += ", %s"% extraFields
      if bpl==2 and hi16:
        return Code.GlobalReadInst("buffer_load_short_d16_hi", vgpr(destVgpr, rpv*2), addr0, \
                  addr1, soffset, tailFields, comment)
      elif bpl==2 and not hi16:
        return Code.GlobalReadInst("buffer_load_short_d16", vgpr(destVgpr, rpv*2), addr0, \
                  addr1, soffset, tailFields, comment)
      elif bpl==4:
        return Code.GlobalReadInst("buffer_load_dword", vgpr(destVgpr, rpv), addr0, \
                  addr1, soffset, tailFields, comment)
      elif bpl==8:
        return Code.GlobalReadInst("buffer_load_dwordx2", vgpr(destVgpr, rpv), addr0, \
                  addr1, soffset, tailFields, comment)
      elif bpl==16:
        return Code.GlobalReadInst("buffer_load_dwordx4", vgpr(destVgpr, rpv), addr0, \
                  addr1, soffset, tailFields, comment)
      else:
        assert ("chooseGlobalRead: bad bpl")

    else:
      if bpl==2 and hi16:
        return Code.GlobalReadInst("flat_load_short_d16_hi", vgpr(destVgpr, rpv*2), addr0, extraFields, comment )
      elif bpl==2 and not hi16:
        return Code.GlobalReadInst("flat_load_short_d16", vgpr(destVgpr, rpv*2), addr0, extraFields, comment )
      elif bpl==4:
        return Code.GlobalReadInst("flat_load_dword", vgpr(destVgpr, rpv), addr0, extraFields, comment )
      elif bpl==8:
        return Code.GlobalReadInst("flat_load_dwordx2", vgpr(destVgpr, rpv), addr0, extraFields, comment )
      elif bpl==16:
        return Code.GlobalReadInst("flat_load_dwordx4", vgpr(destVgpr, rpv), addr0, extraFields, comment )
      else:
        assert ("chooseGlobalRead: bad bpl")

  ##############################################################################
  def chooseGlobalWrite(self, useBuffer, bps, srcVgpr, rpv, \
                        addr0, addr1, offset, extraFields, hi16=0):
    """
    create the store instruction for requested vector width and other parms
   
    rpv = regs per vector
    """

    kStr = ""

    if useBuffer:
      if bps==2 and hi16:
        kStr += inst("buffer_store_short_d16_hi", vgpr(srcVgpr, rpv*2), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==2 and not hi16:
        kStr += inst("buffer_store_short", vgpr(srcVgpr, rpv*2), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==4:
        kStr += inst("buffer_store_dword", vgpr(srcVgpr, rpv), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==8:
        kStr += inst("buffer_store_dwordx2", vgpr(srcVgpr, rpv), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store D")
      elif bps==16:
        kStr += inst("buffer_store_dwordx4", vgpr(srcVgpr, rpv), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store D")
      else:
        assert ("bad bps")
    else:
      if bps==2 and hi16:
        kStr += inst("flat_store_short_d16_hi", addr0, vgpr(srcVgpr*2), extraFields, "store D" )
      elif bps==2 and not hi16:
        kStr += inst("flat_store_short", addr0, vgpr(srcVgpr, rpv*2), extraFields, "store D" )
      elif bps==4:
        kStr += inst("flat_store_dword", addr0, vgpr(srcVgpr, rpv), extraFields, "store D" )
      elif bps==8:
        kStr += inst("flat_store_dwordx2", addr0, vgpr(srcVgpr, rpv), extraFields, "store D" )
      elif bps==16:
        kStr += inst("flat_store_dwordx4", addr0, vgpr(srcVgpr, rpv), extraFields, "store D" )
      else:
         assert ("bad bps")

    return kStr

  ##############################################################################
  ##############################################################################
  def addStore(self, kernel, ss, addrCalc, sumIdx, tmpS01, edge):
    """
    Add stores for the element with addrCalc and sumIdx.
    tmpS01 is a single :temp sGPR
    """
    kStr = ""
    if self.do["GlobalWrite"]:
      # perform vector stores here, so no VI indexing.
      # if GWVW > Vw, might need to support loops to
      # implement wider stores
      ntStr = ""
      if kernel["NonTemporalC"]%2==1:
        ntStr += " glc"
      if kernel["NonTemporalC"]//2==1:
        ntStr += " slc"

      bps = kernel["ProblemType"]["DataType"].numBytes() * ss.cfg.gwvw
      rpv = kernel["ProblemType"]["DataType"].numRegisters() * ss.cfg.gwvw
      if kernel["BufferStore"]:
        addr0 = vgpr(addrCalc.addrVgpr)
        addr1 = sgpr("SrdD", 4)
      else:
        addr0 = vgpr(addrCalc.addrVgpr,2)
        addr1 = ""

      useBuffer = kernel["BufferStore"]
      if ss.optSrdIncForRow and addrCalc.rowInc:
        kStr += addrCalc.incrementToNextRow(kernel, "D", ss, tmpS01)
      if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx//2, rpv, \
                    addr0, addr1, addrCalc.globalOffset, ntStr, hi16=sumIdx%2)
        else:
          kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                    addr0, addr1, addrCalc.globalOffset, ntStr, hi16=0)
      elif kernel["ProblemType"]["DataType"].isInt8x4() or kernel["ProblemType"]["DataType"].isSingle():
        kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                  addr0, addr1, addrCalc.globalOffset, ntStr)
      elif kernel["ProblemType"]["DataType"].isDouble() or kernel["ProblemType"]["DataType"].isSingleComplex():
        kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx*2, rpv, \
                  addr0, addr1, addrCalc.globalOffset, ntStr)
      elif kernel["ProblemType"]["DataType"].isDoubleComplex():
        rps = kernel["ProblemType"]["DataType"].numRegisters()
        kStr += self.chooseGlobalWrite(useBuffer, bps, sumIdx*rps, rpv, \
                  addr0, addr1, addrCalc.globalOffset, ntStr)

    return kStr


  ##############################################################################
  # choose the ADD instruction for combining external C with internal C
  # used in atomic=1 case to compute expected external data
  ##############################################################################
  def chooseAddForAtomic(self, kernel, dst, src0, src1, comment):
    kStr = ""
    if kernel["ProblemType"]["DataType"].isHalf():
      if kernel["ProblemType"]["HighPrecisionAccumulate"]:
        kStr += inst("v_mad_mix need madmix bozo", \
                  dst, src0, src1, \
                  comment)
      else:
        kStr += inst("v_pk_add_f16", \
                  dst, src0, src1, \
                  comment)
    elif kernel["ProblemType"]["DataType"].isInt8x4():
      # assume v_add_i32 can be used in place of v_add_f32
      # need to add saturation directive to v_add_i32 instruction to clamp integer arithmetic
      kStr += inst("v_add_i32", \
                dst, src0, src1, \
                comment)
    elif kernel["ProblemType"]["DataType"].isSingle():
      kStr += inst("v_add_f32", \
                dst, src0, src1, \
                comment)
    else:
      assert(0) # no double GSU yet

    return kStr

  ##############################################################################
  ##############################################################################
  def applyAlpha(self, kernel, gwvw, elementSumIdx, elementIdx, tmpS01):
    kStr = ""

    if self.do["ApplyAlpha"]:
      for vi in range(0, gwvw):
        sumIdxV = elementSumIdx[elementIdx] + vi
        if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
          if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
            if sumIdxV%2:
              kStr += inst("v_pk_mul_f16", vgpr("ValuC+%u"%(sumIdxV//2)), sgpr("Alpha"), vgpr("ValuC+%u"%(sumIdxV//2)), "*= alpha sumIdx=%u vi=%u"%(elementSumIdx[elementIdx], vi))
          else: # HPA
            kStr += inst("v_mul_f32", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha")

        elif kernel["ProblemType"]["DataType"].isInt8x4():
          # below assume we use v_mul_lo_u32. Could also use v_mul_i32_i24.
#           kStr += inst("v_mul_i32_i24", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha" )
          kStr += inst("v_mul_lo_u32", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha" )

        elif kernel["ProblemType"]["DataType"].isSingle():
          kStr += inst("v_mul_f32", vgpr("ValuC+%u"%sumIdxV), sgpr("Alpha"), vgpr("ValuC+%u"%sumIdxV), "*= alpha" )
          if self.db["ForceExpectedValue"]:
            kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), self.db["ValueCExpectedValue"], "force expected value" )
          if self.db["ForceVSerial"]:
            kStr += inst("v_mov_b32", vgpr("ValuC+%u"%sumIdxV), vgpr("Serial"), "force expected value to serial" )
          if self.db["CheckValueC"]:
            kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["ValueCExpectedValue"], "Move expected value")
            kStr += self.assert_eq(vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01))

        elif kernel["ProblemType"]["DataType"].isDouble():
          kStr += inst("v_mul_f64", vgpr("ValuC+%u"%(sumIdxV*2),2), sgpr("Alpha",2), vgpr("ValuC+%u"%(sumIdxV*2),2), "*= alpha")

        # single precision complex
        elif kernel["ProblemType"]["DataType"].isSingleComplex():
          tmpVgpr = self.vgprPool.checkOut(1)
          kStr += inst("v_mov_b32", vgpr(tmpVgpr), vgpr("ValuC+%u"%(sumIdxV*2)), "store Cr")
          kStr += inst("v_mul_f32", vgpr("ValuC+%u"%(sumIdxV*2)), sgpr("Alpha"), vgpr("ValuC+%u"%(sumIdxV*2)), "*= alpha ( Cr = Ar * Cr)")
          kStr += inst("v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2)), "-" + sgpr("Alpha+1"), vgpr("ValuC+%u"%(sumIdxV*2+1)), "*= alpha ( Cr += -Ai * Ci )")
          kStr += inst("v_mul_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), sgpr("Alpha"), vgpr("ValuC+%u"%(sumIdxV*2+1)), "*= alpha ( Ci = Ar * Ci)")
          kStr += inst("v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), sgpr("Alpha+1"), vgpr(tmpVgpr), "*= alpha ( Ci += Ai * Cr_backup )")
          self.vgprPool.checkIn(tmpVgpr)
          
        # double precision complex
        elif kernel["ProblemType"]["DataType"].isDoubleComplex():
          vtmp1 = self.vgprPool.checkOut(2)
          vtmp2 = self.vgprPool.checkOut(2)
          # tmp1 = a.real * b.real
          kStr += inst("v_mul_f64", vgpr(vtmp1,2), sgpr("Alpha+0",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), "")
          # tmp2 = a.imag * b.real
          kStr += inst("v_mul_f64", vgpr(vtmp2,2), sgpr("Alpha+2",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), "")
          # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
          kStr += "v_fma_f64 %s, %s, -%s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+0),2), sgpr("Alpha+2",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(vtmp1,2), self.endLine)
          # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
          kStr += "v_fma_f64 %s, %s, %s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+2),2), sgpr("Alpha+0",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(vtmp2,2), self.endLine)
          self.vgprPool.checkIn(vtmp1)
          self.vgprPool.checkIn(vtmp2)
          
    return kStr

  ##############################################################################
  # Global Write Batch
  ##############################################################################
  def globalWriteBatch(self, kernel, ss, batchIdx, beta, edge, atomic, gwvw, atomicW, \
      batchElements, coord0, coord1, addrD, addrC,  \
      tmpVgpr, batchElementSgprs, tmpSgpr):
    kStr = ""

    kStr += self.comment1("optSingleColVgpr=%u optSharedColVgpr=%u optSharedMask=%u optSrdIncForRow=%u" % \
              (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSharedMask, ss.optSrdIncForRow))
    if atomic:
      # all kinds of code relies on this assumption:
      assert(atomicW <= gwvw)
      if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
        assert(atomicW >= 2)

    # comment tt1, tt0, vc1, vc0
    # tt = trhead tile, vc=vector component
    commentStr = "Global Write%s%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
        % (" Beta" if beta else "", " Edge" if edge else "", batchIdx)
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      commentStr += "(%u,%u,%u,%u:vw%u%s)" % \
        (element[0], element[1], element[2], element[3], gwvw,
         ":vaw:%u"%atomicW if atomic else "")
      if elementIdx < len(batchElements)-1:
        commentStr += "; "
    kStr += self.comment3(commentStr)

    ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=False)

    loadsIssued = 0
    storesIssued = 0
    tmpS01 = tmpSgpr # scratch sgprs
    tmpS23 = tmpS01+2

    ########################################
    # calculate addr and masks
    kStr += self.comment("calc coords, apply mask, and issue loads (if necessary)")
    # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
    # on the thread and tid number.  These are ELEMENT offsets from start of tensor C
    # for the top-left corner this thread will write.  These are not changed
    # across all the store loop iters.
    if self.db["ConservativeWaitCnt"] & 0x10:
      kStr += "s_barrier // debug\n"
      kStr += inst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
      kStr += "s_barrier // debug\n"
    if not edge and self.db["ForceEdgeStores"]>=2:
      kStr += self.bomb() # should not get here
    if edge and self.db["AssertNoEdge"]:
      kStr += self.bomb() # should not get here
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      addr = ss.elementAddr[elementIdx].addrVgpr
      addrCalc = ss.elementAddr[elementIdx]
      data = ss.elementData[elementIdx]
      mask = ss.elementMask[elementIdx]
      sumIdx = ss.elementSumIdx[elementIdx]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]

      kStr += addrCalc.emitAddressSetupCode(kernel, ss, tmpVgpr, tmpS01, edge, beta, atomic, mask, elementIdx)
      if not kernel["BufferStore"]:
        # flat: in-bounds exec mask
        if edge:
          kStr += inst("v_cmp_lt_u32",  sgpr(tmpS01,2), vgpr(addrCalc.coord0Vgpr), sgpr("SizesFree+0"), "coord0 < size0" )
          kStr += inst("v_cmp_lt_u32",  sgpr(tmpS23,2), vgpr(addrCalc.coord1Vgpr), sgpr("SizesFree+1"), "coord1 < size1" )
          kStr += inst("s_and_b64",  sgpr(mask,2), sgpr(tmpS01,2), sgpr(tmpS23,2), "in0 && in1" )

          if (beta or atomic):
            kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )

        # global offset macro (requires 3 tmpVgpr)
        # final address = C + index*bytes
        kStr += "GLOBAL_OFFSET_C %u" % addr
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          if i == kernel["ProblemType"]["Index0"]:
            kStr += ", %s" % (addrCalc.coord0Vgpr)
          elif i == kernel["ProblemType"]["Index1"]:
            kStr += ", %s" % (addrCalc.coord1Vgpr)
          else: # just a group index
            kStr += ", sgprWorkGroup%u"%i
        kStr += ", %s%s" % ((tmpVgpr+2), self.endLine)

        # store a copy of the offset in 2 of the tmpVgpr for D
        if beta:
          kStr += inst("v_mov_b32", vgpr(tmpVgpr+2), vgpr(addr+0), "temp store offset 0")
          kStr += inst("v_mov_b32", vgpr(tmpVgpr+3), vgpr(addr+1), "temp store offset 1")

          kStr += inst("_v_add_co_u32",  vgpr(addr+0), "vcc", vgpr(addrC+0), \
              vgpr(addr+0), "addr = C + index*bytes (lo)" )
          kStr += inst("_v_addc_co_u32", vgpr(addr+1), "vcc", vgpr(addrC+1), \
              vgpr(addr+1), "vcc", "addr = C + index*bytes (hi)")

      if atomic:
        # load c into data+1 because of CAS structure
        # TODO - Fix for double here, would need bigger load
        # FIME
        bps = kernel["ProblemType"]["DataType"].numBytes()
        # gwvw is the number of elements in the batch
        # iterate over number of atomic operations to perform, each of width atomicW
        for avi in range(0, gwvw//atomicW):
          dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
          bpm = self.bpeCexternal * atomicW
          useBuffer = kernel["BufferStore"]
          if kernel["BufferStore"]: # yes, BufferStore here - use same addressing regs for this load
            addr0 = vgpr(addr)
            addr1 = sgpr("SrdC", 4)
          else:
            addr0 = vgpr(addr,2)
            addr1 = ""
          kStr += self.chooseGlobalRead(useBuffer, bpm, dataV+1, \
                    addr0, addr1, soffset=0, offset=addrCalc.globalOffset, extraFields="",
                    comment="load C (atomic) bpm=%u vaw=%u"%(bpm,atomicW)).toStr()
      elif beta:
        bps = kernel["ProblemType"]["DataType"].numBytes() * gwvw
        useBuffer = kernel["BufferStore"]
        if kernel["BufferStore"]:
          addr0 = vgpr(addr)
          addr1 = sgpr("SrdC", 4)
        else:
          addr0 = vgpr(addr,2)
          addr1 = ""
        extraFields = ""
        loadsIssued += 1

        if ss.optSrdIncForRow and addrCalc.rowInc:
          kStr += addrCalc.incrementToNextRow(kernel, "C", ss, tmpS01)

          #if not kernel["LdcEqualsLdd"]:
          #  kStr += addrCalc.incrementToNextRow(kernel, "D", ss.optSrdIncForRow, tmpS01)
        if kernel["ProblemType"]["DataType"].isHalf():
          kStr += self.chooseGlobalRead(useBuffer, bps, data, \
                    addr0, addr1, soffset=0, offset=addrCalc.globalOffset, \
                    extraFields=extraFields, hi16=sumIdx%2,
                    comment="load C for beta calc").toStr()
        elif kernel["ProblemType"]["DataType"].isBFloat16() or \
             kernel["ProblemType"]["DataType"].isInt8x4() or \
             kernel["ProblemType"]["DataType"].isSingle() or \
             kernel["ProblemType"]["DataType"].isDouble() or \
             kernel["ProblemType"]["DataType"].isSingleComplex() or \
             kernel["ProblemType"]["DataType"].isDoubleComplex():
          kStr += self.chooseGlobalRead(useBuffer, bps, data, \
                    addr0, addr1, soffset=0, offset=addrCalc.globalOffset, \
                    extraFields=extraFields, \
                    comment="load C for beta calc").toStr()

      if kernel["InterleaveAlpha"]:
        kStr += self.applyAlpha(kernel, gwvw, ss.elementSumIdx, elementIdx, tmpS01)

      if not kernel["LdcEqualsLdd"]:
        kStr += addrCalc.emitLddChange(kernel, ss, edge, mask, elementIdx == len(batchElements)-1)

      if not kernel["BufferStore"]:
        offsetSrc = (tmpVgpr+2) if beta else addr

        kStr += inst("_v_add_co_u32",  vgpr(addr+0), "vcc", vgpr(addrD+0), \
            vgpr(offsetSrc+0), "addr = D + index*bytes (lo)" )
        kStr += inst("_v_addc_co_u32", vgpr(addr+1), "vcc", vgpr(addrD+1), \
            vgpr(offsetSrc+1), "vcc", "addr = D + index*bytes (hi)")

        # restore full exec mask for calculating addr of next element
        if edge and (beta or atomic):
          kStr += inst("s_mov_b64", "exec", -1, "full mask -1 -> exec" )

    ########################################
    # rC *= alpha
    if not kernel["InterleaveAlpha"]:
      kStr += self.comment("rC *= alpha batchEements=%s"%batchElements)
      for elementIdx in range(0, len(batchElements)):
        kStr += self.applyAlpha(kernel, gwvw, ss.elementSumIdx, elementIdx, tmpS01)

    ########################################
    # Atomic
    ########################################
    # flat_atomic_cmpswap tmp addr data:
    #   tmp = mem[addr]
    #   src = data[vi*numVgprsPerDataPerVI][0] new C
    #   cmp = data[vi*numVgprsPerDataPerVI][1] original C
    #   mem[addr] = (tmp==cmp) ? src : tmp
    #   addr = vgpr(addr,2)
    #   data = vgpr(tmpVgpr,2)
    #   tmp = vgpr(tmpVgpr+4)

    # buffer_atomic_cmpswap:
    #   dest is 64 bits, two consec VGPR:
    #     - lower is desired swap value (computed new value) "src"
    #       src = data[vi*numVgprsPerDataPerVI][0] new C
    #     - upper is expected value in memory (from prev load).  "cmp".
    #       cmp = data[vi*numVgprsPerDataPerVI][1] original C
    #   src0 is address offset from SRD
    #
    # After buffer_atomic_cmpswap:
    #   dest =
    #       - data[vi*numVgprsPerDataPerVI][0] C loaded from memory, overwrites src
    if atomic:
      del tmpVgpr # catch bugs
      # TODO for atomic GWVW:
      #  - Use vi to compute addresses, sumIdx.
      #  - Need a solution for the mask.  Can move to all buffer or can fix?

      # atomic loop label
      element = batchElements[0]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]
      labelString = "Global_Write%s%s_vc=%u,%u_d=%u,%u" \
        % (" Beta" if beta else "", " Edge" if edge else "", vc0, vc1, d0, d1 )
      label = self.getLabelNum(labelString)
      labelString += "EarlyExit"
      labelAfterAtomicLoop = self.getLabelNum(labelString)

      ########################################
      # wait for batched load
      # TODO - we are always atomic here?
      kStr += inst("s_waitcnt", "vmcnt(0)", "wait C (atomic)" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

      ########################################
      # first attempt write
      kStr += self.comment("issue first atomic writes")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addrCalc = ss.elementAddr[elementIdx]
        mask = ss.elementMask[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]

        # apply in-bounds exec mask
        if edge:
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec (before atomic)" )

        for avi in range(0, gwvw//atomicW):
          dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
          sumIdxV = ss.elementSumIdx[elementIdx] + avi
          if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():  sumIdxV //= 2
          # for atomic, data[1] = original c, data[0] = new c
          kStr += self.chooseAddForAtomic(kernel, \
                    vgpr(dataV+0), vgpr(dataV+1), vgpr("ValuC+%u"%sumIdxV), \
                    "desired value avi=%u"%avi)

          # attempt write
          atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
          if self.do["GlobalWrite"]:
            bps = kernel["ProblemType"]["DataType"].numBytes()
            if kernel["BufferStore"]:
              kStr += "buffer_atomic_cmpswap %s, %s, %s %s    // %s%s" % \
                  (vgpr(dataV,2), \
                   vgpr(addrCalc.addrVgpr,1), \
                   sgpr("SrdD", 4),  \
                   "0 offen offset:%u glc" % addrCalc.globalOffset, \
                   "attempt write avi=%u"%(avi), self.endLine )
            else:
              kStr += "flat_atomic_cmpswap %s, %s, %s %s    // %s%s" % \
                  (vgpr(atomicDestVgpr), vgpr(addrCalc.addrVgpr,2), \
                  vgpr(dataV,2), "glc", "attempt write", self.endLine )
          else:
             kStr += inst("v_mov_b32", vgpr(atomicDestVgpr), vgpr(dataV+1), "Fake successful CAS" )
             # Fake successful CAS swap:

      ########################################
      # wait for first attempt write
      kStr += inst("s_waitcnt vmcnt(0)", "wait for atomic writes" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

      ########################################
      # check first attempt
      kStr += self.comment("check success of writes, update masks")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        mask = ss.elementMask[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]

        # calculate new masks
        if edge:
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )
          for avi in range(0, gwvw//atomicW):
            dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
            atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
            # need to apply element mask before comparison
            # so that all valid lanes are doing the cmp
            if avi == 0:
              kStr += inst("v_cmp_ne_u32", sgpr(tmpS01,2), vgpr(atomicDestVgpr), \
                  vgpr(dataV+1), "c read during atomic == c read during prior load (avi=%u, first)"%avi )
            else:
              kStr += inst("v_cmp_ne_u32", sgpr(tmpS23,2), vgpr(atomicDestVgpr), \
                  vgpr(dataV+1), "c read during atomic == c read during prior load (avi=%u)"%avi )
              kStr += inst("s_or_b64", sgpr(tmpS01,2), sgpr(tmpS01,2), sgpr(tmpS23,2), "combine with tmp mask")

          if kernel["DisableAtomicFail"]:
            kStr += inst("s_mov_b64",  sgpr(mask,2), 0, "DisableAtomicFail, force 0" )
          else:
            kStr += inst("s_and_b64",  sgpr(mask,2), sgpr(tmpS01,2), sgpr(mask,2), "inBounds & must try again" )

        else:
          for avi in range(0, gwvw//atomicW):
            dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
            atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
            if kernel["DisableAtomicFail"]:
              kStr += inst("s_mov_b64",  sgpr(mask,2), 0, "DisableAtomicFail, force 0" )
            else:
              kStr += inst("v_cmp_ne_u32", sgpr(mask,2), vgpr(atomicDestVgpr), \
                  vgpr(dataV+1), "c read during atomic != c read during prior load" )

      # or masks together to check early exit
      kStr += self.comment("or masks to check for exit")
      kStr += inst("s_mov_b64", sgpr(tmpS01,2), hex(0), "empty mask" )
      for elementIdx in range(0, len(batchElements)):
        mask = ss.elementMask[elementIdx]
        kStr += inst("s_or_b64", sgpr(tmpS01,2), sgpr(mask,2), sgpr(tmpS01,2), "or to add threads" )
      kStr += inst("s_or_saveexec_b64", sgpr(tmpS23,2), sgpr(tmpS01,2), "apply combined mask" )
      kStr += inst("s_cbranch_execz", "label_%04u" % labelAfterAtomicLoop, "if exec is zero skip loop" )

      # begin atomic loop
      kStr += self.comment("atomic CAS loop")
      kStr += "label_%04u:%s" % (label, self.endLine)

      kStr += self.comment("apply updated masks and issue writes again")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addrCalc = ss.elementAddr[elementIdx]
        addr = ss.elementAddr[elementIdx].addrVgpr
        mask = ss.elementMask[elementIdx]
        bps = kernel["ProblemType"]["DataType"].numBytes()

        for avi in range(0, gwvw//atomicW):
          dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
          atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
          sumIdxV = ss.elementSumIdx[elementIdx] + avi
          if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():  sumIdxV //= 2

          # apply mask for element
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "must try again" )
          kStr += inst("v_mov_b32", vgpr(dataV+1), vgpr(atomicDestVgpr), "dataV+1 = tmp (new original C)" )
          kStr += self.chooseAddForAtomic(kernel, \
                    vgpr(dataV+0), vgpr(dataV+1), vgpr("ValuC+%u"%sumIdxV), \
                    "newC = rC + originalC")
          if self.do["GlobalWrite"]:
            if kernel["BufferStore"]:
              # Using no-ret version here?
              kStr += "buffer_atomic_cmpswap %s, %s, %s %s    // %s%s" % \
                  (vgpr(dataV,2), \
                   vgpr(addr,1), \
                   sgpr("SrdD", 4), \
                   "0 offen offset:%u glc" % (addrCalc.globalOffset), \
                   "try again", self.endLine )
            else:
              kStr += "flat_atomic_cmpswap %s, %s, %s %s    // %s%s" % ( vgpr(atomicDestVgpr), \
                  vgpr(addr,2), vgpr(dataV,2), "glc", "try again", self.endLine)

      # wait for batched write
      kStr += inst("s_waitcnt vmcnt(0)", "wait for atomic writes" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

      # check batched write success
      kStr += self.comment("apply masks and check for success")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        data = ss.elementData[elementIdx]
        mask = ss.elementMask[elementIdx]
        for avi in range(0, gwvw//atomicW):
          dataV = ss.elementData[elementIdx] + int(avi*ss.cfg.numVgprsPerDataPerVI)
          atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2

          # apply mask for element
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "must try again" )

          # compare success
          kStr += inst("v_cmp_ne_u32", sgpr(tmpS01,2), vgpr(data+1), vgpr(atomicDestVgpr), \
              "c read during atomic == c read during prior load" )
          # update element mask
          kStr += inst("s_and_b64",  sgpr(mask,2), sgpr(tmpS01,2), sgpr(mask,2), "inBounds & must try again" )

      # or masks together
      kStr += self.comment("or masks to check for exit")
      kStr += inst("s_mov_b64", sgpr(tmpS01,2), hex(0), "empty mask" )
      for elementIdx in range(0, len(batchElements)):
        mask = ss.elementMask[elementIdx]
        kStr += inst("s_or_b64", sgpr(tmpS01,2), sgpr(mask,2), sgpr(tmpS01,2), "or to add threads" )

      # apply combined masks and exit
      kStr += inst("s_or_saveexec_b64", sgpr(tmpS23,2), sgpr(tmpS01,2), "apply combined mask" )
      kStr += inst("s_cbranch_execnz", "label_%04u" % label, "try again if not complete" )
      kStr += "label_%04u:%s" % (labelAfterAtomicLoop, self.endLine)
      kStr += inst("s_mov_b64", "exec", -1, "full mask -> exec" )

      #  kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "apply new mask" )
      #  #kStr += inst("s_and_saveexec_b64", sgpr(tmpS45,2), "vcc", "apply new mask" )
      #  kStr += inst("s_cbranch_execnz", "label_%04u" % labelIdx, "try again if not complete" )
      #  kStr += inst("s_mov_b64", "exec", sgpr(fullExecMaskSgpr,2), "full mask -> exec" )


    ########################################
    # Not Atomic
    ########################################
    else:
      # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
      interleaveStoreVmcnt = self.interleaveStoreVmcnt and not edge

      ########################################
      # wait for batched load
      if beta and not interleaveStoreVmcnt:
        kStr += inst("s_waitcnt", "vmcnt(0)", "wait C")
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

      kStr += self.comment("apply mask, calc new C and issue writes")
      #kStr += self.bomb() # can see store addresses just before the store inst

      if kernel["ProblemType"]["DataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        vgprBf16Temp = self.vgprPool.checkOut(4)
        vgprBf16Mask = vgprBf16Temp + 1
        vgprFp32Nan = vgprBf16Temp + 2
        vgprBf16Inc = vgprBf16Temp + 3
        kStr += inst("v_mov_b32", vgpr(vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" )
        kStr += inst("v_mov_b32", vgpr(vgprFp32Nan), "0x7fff0000", "fp32 Nan" )
        kStr += inst("v_mov_b32", vgpr(vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" )

      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = ss.elementAddr[elementIdx].addrVgpr
        mask = ss.elementMask[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]
        sumIdx = ss.elementSumIdx[elementIdx]

        # apply in-bounds exec mask
        if edge and not kernel["BufferStore"]:
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )

        if beta:
          # if GWVW=1 the half path still assumes we have
          # at least two stores so does some combining across VI -
          # for example assuming we can have two elements and can use pk_mul
          # here:
          if beta and interleaveStoreVmcnt:
            if self.archCaps["SeparateVscnt"]:
              vmcnt = loadsIssued - elementIdx - 1
              vmComment = "{} = {} - {} - 1".format(vmcnt, loadsIssued, elementIdx)
            else:
              vmcnt = loadsIssued - elementIdx + storesIssued - 1
              vmComment = "{} = {} - {} + {} - 1".format(vmcnt, loadsIssued, elementIdx, storesIssued)

            maxVmcnt = globalParameters["AsmCaps"][self.version]["MaxVmcnt"]
            vmcnt = min(vmcnt, maxVmcnt)
            #print "wmvcnt=", vmcnt
            kStr += "\n"
            kStr += inst("s_waitcnt", "vmcnt(%u)"%vmcnt, "wait C (interleaved) " + vmComment)
          for vi in range(0, gwvw):
            dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
            sumIdxV = ss.elementSumIdx[elementIdx] + vi
            if kernel["ProblemType"]["DataType"].isHalf():
              if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
                if sumIdxV%2==0:
                  # dataV+0 = new c = old c*beta
                  kStr += inst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
                      "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
                  # dataV+0 = new c = old c*beta + rC
                  kStr += inst("v_pk_add_f16", vgpr("ValuC+%u"%(sumIdxV//2)), vgpr(dataV), vgpr("ValuC+%u"%(sumIdxV//2)), \
                      "sum*alpha + C*beta")
                else:
                  pass # add will have been done previously
              else: # HPA
                # dataV+0 = new c = old c*beta + rC
                # src0 = beta = f32 = opsel 00
                # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
                # src2 = sumIdxV = f32 = opsel 00
                dataCExternal = ss.elementData[elementIdx] + vi//2
                hi16 = sumIdxV%2
                kStr += inst(self.mixinst, vgpr("ValuC+%u"%sumIdxV), sgpr("Beta"), \
                    vgpr(dataCExternal), vgpr("ValuC+%u"%sumIdxV), \
                    "op_sel:[0,%u,0] op_sel_hi:[0,1,0]" % (hi16), \
                    "//C*=beta")

            elif kernel["ProblemType"]["DataType"].isBFloat16():
              if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                # dataV+0 = new c = old c*beta + rC
                # src0 = beta = f32 = opsel 00
                # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
                # src2 = sumIdxV = f32 = opsel 00
                tmpVgpr = self.vgprPool.checkOut(1)
                dataCExternal = ss.elementData[elementIdx] + vi//2
                if (vi%2) == 1:
                  kStr += inst("v_and_b32", vgpr(tmpVgpr), vgpr(dataCExternal), vgpr(vgprBf16Mask), "convert bf16 to fp32")
                else:
                  kStr += inst("v_lshlrev_b32", vgpr(tmpVgpr), "16", vgpr(dataCExternal), "convert bf16 to fp32" )
                kStr += inst("v_mac_f32", vgpr("ValuC+%u"%sumIdxV), vgpr(tmpVgpr), sgpr("Beta"), \
                    "finalSum = sum*alpha + C*beta")
                self.vgprPool.checkIn(tmpVgpr)

            elif kernel["ProblemType"]["DataType"].isSingle():
              kStr += inst("v_mac_f32", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), sgpr("Beta"), \
                  "finalSum = sum*alpha + C*beta")

            elif kernel["ProblemType"]["DataType"].isInt8x4():
              # assume we will need to replace v_mac_f32 with v_add_u32 and s_mul_lo_i32
              # v_mad_i32_i24
              #kStr += inst("v_mad_i32_i24", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), sgpr("Beta"), vgpr("ValuC+%u"%sumIdxV), \
              #    "finalSum = sum*alpha + C*beta")
              kStr += inst("v_mul_lo_i32", vgpr(dataV+0), sgpr("Beta"), vgpr(dataV+0), \
                  "C = C*beta")
              kStr += inst("v_add_u32", vgpr("ValuC+%u"%sumIdxV), vgpr(dataV+0), vgpr("ValuC+%u"%sumIdxV), \
                  "finalSum = sum*alpha + C*beta")
              kStr += " "

            elif kernel["ProblemType"]["DataType"].isDouble():
              # dataV+0 = new c = old c*beta
              kStr += inst("v_fma_f64", vgpr("ValuC+%u"%(sumIdxV*2),2), vgpr(dataV+0,2), sgpr("Beta",2), vgpr("ValuC+%u"%(sumIdxV*2),2), \
                  "finalSum = sum*alpha + C*beta")

            # single precision complex
            elif kernel["ProblemType"]["DataType"].isSingleComplex():
              kStr += inst("v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2)), vgpr(dataV+0), sgpr("Beta"), "finalSum Cr += old Cr * Br")
              kStr += inst("v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2)), vgpr(dataV+1), "-"+sgpr("Beta+1"), "finalSum Cr += old Ci * -Bi")
              kStr += inst("v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), vgpr(dataV+1), sgpr("Beta"), "finalSum Ci += old Ci * Br")
              kStr += inst("v_mac_f32", vgpr("ValuC+%u"%(sumIdxV*2+1)), vgpr(dataV+0), sgpr("Beta+1"), "finalSum Ci += old Cr * Bi")

            # double precision complex
            elif kernel["ProblemType"]["DataType"].isDoubleComplex():
              # c.real += a.real * b.real 
              kStr += "v_fma_f64 %s, %s, %s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+0),2), vgpr(dataV+0,2), sgpr("Beta+0",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), self.endLine)
              # c.real -= a.imag * b.imag
              kStr += "v_fma_f64 %s, %s, -%s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+0),2), vgpr(dataV+2,2), sgpr("Beta+2",2), vgpr("ValuC+%u"%(sumIdxV*4+0),2), self.endLine)
              # c.imag += a.real * b.imag
              kStr += "v_fma_f64 %s, %s, %s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(dataV+0,2), sgpr("Beta+2",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), self.endLine)
              # c.imag += a.imag * b.real
              kStr += "v_fma_f64 %s, %s, %s, %s%s" % (vgpr("ValuC+%u"%(sumIdxV*4+2),2), vgpr(dataV+2,2), sgpr("Beta+0",2), vgpr("ValuC+%u"%(sumIdxV*4+2),2), self.endLine)

        # pack stores, beta and non-beta reach here:
        for vi in range(0, gwvw):
          sumIdxV = ss.elementSumIdx[elementIdx] + vi
          if kernel["ProblemType"]["DataType"].isHalf():
            if kernel["ProblemType"]["HighPrecisionAccumulate"]:
              kStr += inst("v_cvt_f16_f32", vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), "convert C to fp16" )
              if vi%2 == 1:
                assert (gwvw % 2 == 0)
                d = ss.elementSumIdx[elementIdx] + vi//2
                kStr += inst("v_pack_b32_f16", vgpr(d), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr("ValuC+%u"%sumIdxV), "Pack with neighbor" )

          elif kernel["ProblemType"]["DataType"].isBFloat16():
            if kernel["ProblemType"]["HighPrecisionAccumulate"]:
              kStr += inst("v_cmp_u_f32", sgpr(tmpS01,2), vgpr("ValuC+%u"%sumIdxV), vgpr("ValuC+%u"%sumIdxV), "check Nan" )
              kStr += inst("v_bfe_u32", vgpr(vgprBf16Temp), vgpr("ValuC+%u"%sumIdxV), "16", "1", "Non-Nan case: store lsb of bf16" )
              kStr += inst("v_add3_u32", vgpr(vgprBf16Temp), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprBf16Temp), vgpr(vgprBf16Inc), "Non-Nan case: add lsb and the increment for rounding" )
              kStr += inst("v_cndmask_b32", vgpr("ValuC+%u"%sumIdxV), vgpr(vgprBf16Temp), vgpr(vgprFp32Nan), sgpr(tmpS01,2), "" )
              if vi%2 == 0:
                kStr += inst("v_lshrrev_b32", vgpr("ValuC+%u"%sumIdxV), "16", vgpr("ValuC+%u"%sumIdxV), "convert C to bf16" )
              elif vi%2 == 1:
                d = ss.elementSumIdx[elementIdx] + vi//2
                kStr += inst("v_and_or_b32", vgpr(d), vgpr("ValuC+%u"%sumIdxV), vgpr(vgprBf16Mask), vgpr("ValuC+%u"%(sumIdxV-1)), "pack two bf16 to dword")

        addrCalc = ss.elementAddr[elementIdx]
        kStr += self.addStore(kernel, ss, addrCalc, sumIdx, tmpS01, edge)
        storesIssued += 1

      if kernel["ProblemType"]["DataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        self.vgprPool.checkIn(vgprBf16Temp)

          #kStr += self.bomb(5)
      if self.db["CheckStoreC"]>=0:
        # Note - CheckStoreC won't work for EDGE store cases since they load 0 for OOB, would need more sophisticated check
        kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
        for elementIdx in range(0, len(batchElements)):
          addr = ss.elementAddr[elementIdx].addrVgpr
          sumIdx = ss.elementSumIdx[elementIdx]

          bps = kernel["ProblemType"]["DataType"].numBytes() * gwvw
          if kernel["BufferStore"]:
            addr0 = vgpr(addr)
            addr1 = sgpr("SrdC", 4)
          else:
            addr0 = vgpr(addr,2)
            addr1 = ""

          if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
            if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
              kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx//2, \
                        addr0, addr1, soffset=0, offset=0, extraFields="", hi16=sumIdx%2).toStr()
            else:
              kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
                        addr0, addr1, soffset=0, offset=0, extraFields="", hi16=0).toStr()
          elif kernel["ProblemType"]["DataType"].isInt8x4() or kernel["ProblemType"]["DataType"].isSingle():
            kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
                      addr0, addr1, soffset=0, offset=0, extraFields="").toStr()
          elif kernel["ProblemType"]["DataType"].isDouble() or kernel["ProblemType"]["DataType"].isSingleComplex() :
            kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*2, \
                      addr0, addr1, soffset=0, offset=0, extraFields="").toStr()
          elif kernel["ProblemType"]["DataType"].isDoubleComplex():
            kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*4, \
                      addr0, addr1, soffset=0, offset=0, extraFields="").toStr()
        kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

        # Add checks for expected values:
        kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["CheckStoreC"], "expected value")
        for elementIdx in range(0, len(batchElements)):
          sumIdx = ss.elementSumIdx[elementIdx]
          # Need to fix for other types:
          assert (kernel["ProblemType"]["DataType"].isSingle())
          kStr += self.assert_eq(vgpr(sumIdx), sgpr(tmpS01))


      if edge and (atomic or not kernel["BufferStore"]):
        # subsequent batch must start with full exec mask
        # BufferStore doesn't need exec since it used buffer range checking when
        # possible
        kStr += inst("s_mov_b64", "exec", -1, "full mask -> exec" )

      if self.db["ConservativeWaitCnt"] & 0x40:
        kStr += "s_barrier // debug\n"
        kStr += inst("s_waitcnt", "vmcnt(0)", "ConservativeWaitCnt" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
        kStr += "s_barrier // debug\n"

    # return registers to pool:
    lastData = -1
    for elementIdx in range(0, len(batchElements)):
      if not ss.sharedColVgprs:
        addr = ss.elementAddr[elementIdx].addrVgpr
        self.vgprPool.checkIn(addr)

      data = ss.elementData[elementIdx]
      if data != 0:
        if data != lastData:
          self.vgprPool.checkIn(data)
        lastData = data

    self.ss.firstBatch = False

    return kStr

  ##############################################################################
  ##############################################################################
  def openPrefetchAcrossPersistent(self, kernel):
    imod = Code.Module()
    stmp = self.getTmpSgpr(1).idx()
    imod.addInst("s_mul_i32", sgpr(stmp), sgpr("NumWorkGroups0"), sgpr("NumWorkGroups1"), "Total WG")
    imod.addInst("s_cmp_ge_u32", sgpr("SerialWorkGroupIter"), sgpr(stmp), "outside legal WG?")
    imod.addInst("s_cbranch_scc1", self.getLabelTarget("SkipPrefetchAcrossPersistent"), "skip pf if OOB")
    #imod.addInst("s_branch", self.getLabelTarget("SkipPrefetchAcrossPersistent"), "skip pf if OOB")
    return imod

  ##############################################################################
  ##############################################################################
  def closePrefetchAcrossPersistent(self, kernel):
    imod = Code.Module()
    imod.addCode(Code.WaitCnt(self.version, 0,0, "bozo, conservative wait"))
    imod.addCode(Code.Label(self.getLabelNum("SkipPrefetchAcrossPersistent"), \
        "SkipPrefetchAcrossPersistent"))
    #imod.addText(self.bomb())
    return imod

  ##############################################################################
  # Function End
  ##############################################################################
  def functionEnd(self, kernel, addLabel=True):
    imod = Code.Module()
    if kernel["PersistentKernel"]:
      # Persistent may generate a SerialWorkGroupIter which is OOB, only loop back if we are in a valid WG:
      stmp = self.getTmpSgpr(1).idx()
      imod.addInst("s_mul_i32", sgpr(stmp), sgpr("NumWorkGroups0"), sgpr("NumWorkGroups1"), "Total WG")
      imod.addInst("s_cmp_ge_u32", sgpr("SerialWorkGroupIter"), sgpr(stmp), "outside legal WG?")
      imod.addInst("s_cbranch_scc0", self.getLabelTarget("PersistentLoopStart"), "persistent loop back")
    if addLabel:
      imod.addCode(Code.Label(self.getLabelNum("KernelEnd"), "KernelEnd"))
    imod.addInst("s_endpgm", "Kernel End")
    return imod

  ##############################################################################
  # Function Suffix
  ##############################################################################
  def functionSuffix(self, kernel):
    kStr = ""
    if self.vgprPool.size() > self.maxVgprs:
      self.overflowedResources = 1
    elif self.sgprPool.size() > self.maxSgprs:
      self.overflowedResources = 2

    vgprPerCU = 65536
    vgprPerThreadPerOccupancy = vgprPerCU // kernel["NumThreads"]
    numWorkGroupsPerCU = vgprPerThreadPerOccupancy // self.vgprPool.size()
    if numWorkGroupsPerCU < 1:
      self.overflowedResources = 4

    if self.overflowedResources:
      kStr += ".endif // overflowed resources \n"

    self.vgprPool.checkFinalState()
    return kStr

  ##############################################################################
  # Kernel Body Prefix
  ##############################################################################
  def kernelBodyPrefix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # Kernel Body Suffix
  ##############################################################################
  def kernelBodySuffix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # Open String
  ##############################################################################
  def openString(self, kernel):
    return ""

  ##############################################################################
  # Close String
  ##############################################################################
  def closeString(self, kernel):
    return ""

  ##############################################################################
  # WaitCnt- DONE
  # 3 components can contribute to the waitcnt:
  #   - Pending global reads.  (skipGlobalRead)
  #   - Pending local write.  (skipLocalWrite)
  #   - Pending local reads (skipLocalRead)
  # If a skip* arg is -1, the associated component does not contribute to
  # the expected lgkmcnt or vmcnt
  ##############################################################################
  def wait(self, kernel, tPA, tPB, skipGlobalRead, skipLocalWrite, \
      skipLocalRead, comment):
    if not self.do["Wait"]: return ""
    # skip = -1 -> ignore
    # skip =  n -> waitcnt(n*num)

    lgkmcnt = 0 if skipLocalWrite > -1 or skipLocalRead > -1 else -1

    if skipLocalWrite > -1 or skipLocalRead > -1:
      if skipLocalWrite > -1:
        numA = 0 if kernel["DirectToLdsA"] \
               else tPA["nrp"]*tPA["nrc"]*max(tPA["nwcv"],tPA["nwpv"])//tPA["nwcvpi"]
        numB = 0 if kernel["DirectToLdsB"] \
               else tPB["nrp"]*tPB["nrc"]*max(tPB["nwcv"],tPB["nwpv"])//tPB["nwcvpi"]
        lgkmcnt += skipLocalWrite * (numA + numB)
      if skipLocalRead > -1:
        lrvw = kernel["VectorWidth"]
        if kernel["EnableMatrixInstruction"]:
          #FIXME  lrvw = 1 for all precision cases 
          # check fp16  requries 2 src(S) might use lrvw=2
          # Also explore using VW>1 for cases ThreadTile0>1 && ThreadTile1
          lrvw = 1
          numA = kernel["InnerUnroll"] * (kernel["MIWaveTile"][0] // lrvw) \
              // self.localReadInstructionA.numOffsets
          numB = kernel["InnerUnroll"] * (kernel["MIWaveTile"][1] // lrvw) \
              // self.localReadInstructionB.numOffsets
        else:
          numB = kernel["InnerUnroll"]*(kernel["ThreadTile1"] // lrvw) \
              // self.localReadInstructionB.numOffsets
          numA = kernel["InnerUnroll"]*(kernel["ThreadTile0"] // lrvw) \
              // self.localReadInstructionA.numOffsets
        lgkmcnt += skipLocalRead * (numA + numB)

    vmcnt = 0 if skipGlobalRead > -1 else -1
    if skipGlobalRead > -1:
      numA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"] \
          * self.numReadVectorComponentsA
      numB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"] \
          * self.numReadVectorComponentsB
      vmcnt += skipGlobalRead * (numA + numB)

      # Unlike flat loads, BufferLoad do not increment the outstanding
      # lgkmcnt
      if lgkmcnt > -1 and not kernel["BufferLoad"]:
        lgkmcnt += skipGlobalRead * (numA + numB)

    if (self.db["ConservativeWaitCnt"] & 0x2) and skipGlobalRead != -1 or \
       (self.db["ConservativeWaitCnt"] & 0x4) and skipLocalWrite != -1 or \
       (self.db["ConservativeWaitCnt"] & 0x8) and skipLocalRead  != -1:
        imod = Code.Module("ConservativeWaitCnt")
        imod.addInst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "debug %s"%comment )
        if self.archCaps["SeparateVscnt"]:
          imod.addInst("s_waitcnt_vscnt", "null", "0", "writes")
        imod.addInst("s_barrier", "debug" )
        return imod

    lgkmcnt = min(lgkmcnt, 15)
    if lgkmcnt >= 0 and vmcnt >= 0:
      vmcnt = -1 # preserve prior behavior of removing vmcnt here?
    maxVmcnt = globalParameters["AsmCaps"][self.version]["MaxVmcnt"]
    vmcnt = min(vmcnt, maxVmcnt)

    waitcnt = Code.WaitCnt(self.version, lgkmcnt,vmcnt,comment)
    if 0 and lgkmcnt == 0:
      imod = Code.Module("DebugWait")
      imod.addCode(waitcnt)
      imod.addText(self.bomb())
      return imod
    return waitcnt

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def syncThreads(self, kernel, comment=""):
    if kernel["NumThreads"] > 64 and self.do["Sync"]:
      kStr = ""
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_lgkmcnt", "null", "0", "extra navi wait")
      elif self.archCaps["Waitcnt0Disabled"]:
        kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "force waitcnt0" )

      kStr += self.indent + self.syncStr + " //" + comment + self.endLine
      return kStr
    else:
      return "// Skip barrier: NumThreads=%s"%(kernel["NumThreads"]) + \
              comment + self.endLine

  ########################################
  # dump lds state
  ########################################
  def dumpLds(self, kernel, startU, numU):
    kStr = ""
    if globalParameters["DebugKernel"]:
      kStr += self.comment("dump lds state")
      kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
      if self.archCaps["SeparateVscnt"]:
        kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
      kStr += inst("s_barrier", "dump LDS" )
      tmp = self.vgprPool.checkOut(1)
      tmpAddr = self.vgprPool.checkOut(1)
      kStr += inst("v_lshlrev_b32", \
          vgpr(tmpAddr), \
          hex(log2(self.bpeAB)), \
          vgpr("Serial"), \
          "dump lds")
      for i in range(startU, startU+numU):
        kStr += inst("ds_read_b32", vgpr(tmp), \
            vgpr(tmpAddr) + " offset:%u"%(i*kernel["NumThreads"]*4), "dump lds")
        kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "dump" )
        if self.archCaps["SeparateVscnt"]:
          kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
        kStr += self.dump(vgpr(tmp))
      self.vgprPool.checkIn(tmp)
      self.vgprPool.checkIn(tmpAddr)
    return kStr


  ########################################
  # init lds state
  ########################################
  def initLds(self, kernel, value):
    kStr = ""
    kStr += self.comment("init lds state")
    kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
    if self.archCaps["SeparateVscnt"]:
      kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
    kStr += inst("s_barrier", "init LDS" )
    tmp = self.vgprPool.checkOut(1)
    tmpAddr = self.vgprPool.checkOut(1)
    kStr += inst("v_mov_b32", vgpr(tmp), hex(value), "Init value")
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    writesPerThread = ((kernel["LdsNumElements"]*numBytesPerElement-1)//kernel["NumThreads"]//4) + 1
    kStr += inst("v_lshlrev_b32", \
        vgpr(tmpAddr), \
        2,
        vgpr("Serial"), \
        "set per-thread address to init LDS")
    for i in range(0, writesPerThread):
      kStr += "ds_write_b32 %s, %s offset:%u %s" \
          %( vgpr(tmpAddr), vgpr(tmp), (i*kernel["NumThreads"]*4), \
          "//init lds" + self.endLine)

    kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "wait for LDS init to complete" )
    if self.archCaps["SeparateVscnt"]:
      kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
    kStr += inst("s_barrier", "init LDS exit" )
    self.vgprPool.checkIn(tmp)
    self.vgprPool.checkIn(tmpAddr)
    return kStr


  ##############################################################################
  # MapAcctoArch
  # function to map MFMA Acc  Registers to Arch VGPR regsiter
  # option :
  #         0 - one-to-one mapping of ACC -> VGPR  using VW
  #         1 - using ds swizzle map strided lanes output of MFMA to  coalscing
  #             lanes of v_mac
  ##############################################################################
  def MapAcctoArchRegs(self, kernel, option):
    kStr = ""
    kStr += self.comment("Mapping of Acc register -> C Vgpr register")

    # remove the C regs from the pool since we are about to write them here:
    kStr += self.comment("remove C-tile %u-%u from pool"%(self.startVgprValuC, self.startVgprValuC+self.numVgprValuC))
    self.vgprPool.remove(self.startVgprValuC, self.numVgprValuC, "ValuC")

    if kernel["MatrixInstM"] == 4:
      for i in range(0, kernel["MIOutputVectorWidth"] * kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1]):
          kStr += inst("v_accvgpr_read_b32", vgpr("ValuC+%u"%i), "acc%u"%i, "copy areg to vreg")
    else:
      OutputsPerMFMA1B = kernel["MatrixInstM"] * kernel["MatrixInstN"] // globalParameters["WavefrontWidth"]

      for wgIdx1 in range(0, kernel["MIWaveTile"][1]):
        for wgIdx0 in range(0, kernel["MIWaveTile"][0]):
          for bIdx1 in range(0, kernel["MatrixInstBN"]):
            for bIdx0 in range(0, kernel["MatrixInstBM"]):
              for tIdx in range(0, OutputsPerMFMA1B):
                src = tIdx + OutputsPerMFMA1B * (bIdx0 + kernel["MatrixInstBM"] * (bIdx1 + kernel["MatrixInstBN"] * (wgIdx0 + kernel["MIWaveTile"][0] * wgIdx1)))
                dst = tIdx + OutputsPerMFMA1B * (bIdx0 + kernel["MatrixInstBM"] * (wgIdx0 + kernel["MIWaveTile"][0] * (bIdx1 + kernel["MatrixInstBN"] * wgIdx1)))
                kStr += inst("v_accvgpr_read_b32", vgpr("ValuC+%u"%dst), "acc%u"%src, "copy areg to vreg")

    return kStr


  ##############################################################################
  #
  #   Beta-Only Kernel
  #
  ##############################################################################

  ##############################################################################
  # Function Signature
  ##############################################################################
  def functionSignatureBetaOnly(self, kernel):
    kStr = ""
    return kStr

  ##############################################################################
  # Kernel Body Beta-Only
  ##############################################################################
  def kernelBodyBetaOnly(self, kernel):
    kStr = ""
    return kStr


  # Perform 32-bit scalar mul and save u64 result in two SGPR
  # src0 and src1 are 32-bit unsigned ints in scalar sgpr or small int constants (<64?))
  # return retuns in dst0:dest (lower 32-bit in dst0, high 64-bit in dst1))
  def s_mul_u64_u32 (self, dst0, dst1,  src0, src1, comment):
    kStr = ""
    assert(dst1 != src0) # no worky since dst1 overwritten by first mul operations
    assert(dst1 != src1) # no worky since dst1 overwritten by first mul operations
    # the else path below has less restrictions but prefer consistency
    if globalParameters["AsmCaps"][self.version]["HasSMulHi"]:
      kStr += inst("s_mul_hi_u32", dst1, src0, src1, comment)
      kStr += inst("s_mul_i32", dst0, src0, src1, comment)
    else:
      if type(src1) != 'str' or not src1.startswith("s"):
        # Swap operands, need a scalar sgpr in src1 (not a constant)
        t = src0
        src0 = src1
        src1 = t
      vtmp0 = self.vgprPool.checkOut(2)
      vtmp1 = vtmp0+1
      kStr += inst("v_mov_b32", vgpr(vtmp0), src0, comment)
      kStr += inst("v_mul_hi_u32", vgpr(vtmp1), vgpr(vtmp0), src1, comment)
      kStr += inst("v_readfirstlane_b32", dst1, vgpr(vtmp1), comment)
      kStr += inst("v_mul_lo_u32", vgpr(vtmp1), vgpr(vtmp0), src1, comment)
      kStr += inst("v_readfirstlane_b32", dst0, vgpr(vtmp1), comment)
      self.vgprPool.checkIn(vtmp0)
    return kStr

  # dividend is a symbol (constant or sgpr).  Used directly not inside automatic sgpr(..)
  # dst is 2 consecutive SGPR
  #   result returned in dst0. dst1 is used as a temp,
  # dst[1] cannot be same as divident, dst[0] can be same as dividend and this can be useful
  def scalarMagicDivExplicit(self, dst, dividend, magicNumber, magicAbit, magicShift):
    kStr = ""
    kStr = self.comment("dst1:0 = dividend(%s) / magicTag(%s)" % (dividend, magicNumber))
    kStr += inst("s_mul_hi_u32", sgpr(dst+1), dividend, sgpr(magicNumber), "scalar magic div (magicnum)")
    kStr += inst("s_mul_i32", sgpr(dst+0), dividend, sgpr(magicAbit), "scalar magic div (abit)")
    kStr += inst("s_add_u32", sgpr(dst+0), sgpr(dst+0), sgpr(dst+1), "scalar magic div (combine)")
    kStr += inst("s_lshr_b32", sgpr(dst+0), sgpr(dst+0), sgpr(magicShift), \
                "scalar magic div (shift), quotient in s%s"%dst)
    return kStr

  def scalarMagicDiv(self, dst, dividend, magicTag):
    return self.scalarMagicDivExplicit(dst, dividend,
                                        magicNumber="MagicNumberSize"+magicTag,
                                        magicAbit="MagicAbitSize"+magicTag,
                                        magicShift="MagicShiftSize"+magicTag)

  # Perform 32-bit scalar mul and save u64 result in two SGPR
  # src0 and src1 are 32-bit unsigned ints in scalar sgpr or small int constants (<64?))
  # return retuns in dst0:dest (lower 32-bit in dst0, high 64-bit in dst1))
  def s_mul_i64_i32 (self, dst0, dst1,  src0, src1, comment):
    kStr = ""
    assert(dst1 != src0) # no worky since dst1 overwritten by first mul operations
    assert(dst1 != src1) # no worky since dst1 overwritten by first mul operations
    # the else path below has less restrictions but prefer consistency
    if globalParameters["AsmCaps"][self.version]["HasSMulHi"]:
      kStr += inst("s_mul_hi_i32", dst1, src0, src1, comment)
      kStr += inst("s_mul_i32", dst0, src0, src1, comment)
    else:
      if type(src1) != 'str' or not src1.startswith("s"):
        # Swap operands, need a scalar sgpr in src1 (not a constant)
        t = src0
        src0 = src1
        src1 = t
      vtmp0 = self.vgprPool.checkOut(2)
      vtmp1 = vtmp0+1
      kStr += inst("v_mov_b32", vgpr(vtmp0), src0, comment)
      kStr += inst("v_mul_hi_i32", vgpr(vtmp1), vgpr(vtmp0), src1, comment)
      kStr += inst("v_readfirstlane_b32", dst1, vgpr(vtmp1), comment)
      kStr += inst("v_mul_lo_u32", vgpr(vtmp1), vgpr(vtmp0), src1, comment)
      kStr += inst("v_readfirstlane_b32", dst0, vgpr(vtmp1), comment)
      self.vgprPool.checkIn(vtmp0)
    return kStr



  def bomb(self,cookie=None,scratchVgpr=-1):
      """
      Cause a GPUVM fault.
      Instruction after the bomb will write the cookie to SGPR0, so you can see the cookie in the 
      backtrace. Useful for locating which spot in code generated the bomb
      vgprAddr controls which vgpr to overwrite with the null pointer address
      """

      kStr =""
      if scratchVgpr==-1:
        vgprAddr = self.vgprPool.checkOut(2)
      else:
        vgprAddr = scratchVgpr
      if cookie != None:
        if cookie < 0:
          kStr += "bomb_neg%u:\n" % abs(cookie)
        else:
          kStr += "bomb_%u:\n" % abs(cookie)
      kStr += inst("v_mov_b32", vgpr(vgprAddr+0), 0, "")
      kStr += inst("v_mov_b32", vgpr(vgprAddr+1), 0, "")
      #kStr += inst("s_trap",1,  "")
      kStr += inst("flat_load_dword", vgpr(vgprAddr), vgpr(vgprAddr,2), "bomb - force fault" )

      # This move does not execute but appears in the instruction stream immediately following
      # the faulting load:
      if cookie != None:
        kStr += inst("s_mov_b32", sgpr(0), cookie, "bomb cookie=%d(0x%x)"%(cookie,cookie&0xffffffff))

      if scratchVgpr == -1:
        self.vgprPool.checkIn(vgprAddr)
      return kStr


  ##############################################################################
  # assertCommon : Common routine for all assert functions.
  # On entry, we have already set the exec-mask so any enabled lanes should bomb
  ##############################################################################
  def assertCommon(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      self.printedAssertCnt += 1

      # Default cookie for asserts is negative of printed #asserts
      # Can be used to roughly identify which assert in the code is firing
      kStr += self.bomb(cookie if cookie != -1 else -self.printedAssertCnt)

    return kStr

  ##############################################################################
  # assertCmpCommon : Common routine for all assert comparison functions
  ##############################################################################
  def assertCmpCommon(self, cond, val0, val1, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_or_saveexec_b64", sgpr("SaveExecMask",2), 0, \
          "assert: saved execmask")

      kStr += inst("v_cmpx_%s"%cond, "vcc", val0, val1, "v_cmp" )

      kStr += self.assertCommon(cookie)

      kStr += inst("s_or_saveexec_b64", "vcc", sgpr("SaveExecMask",2), \
          "assert: restore execmask")

    return kStr

  ##############################################################################
  # Handle different conditions for the asserts:
  # These support uin32 compare, float could be added later
  # Asserts currently modify vcc
  ##############################################################################
  def assert_eq(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("ne_u32", val0, val1, cookie)

  def assert_ne(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("eq_u32", val0, val1, cookie)

  def assert_lt_u32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("ge_u32", val0, val1, cookie)

  def assert_gt_u32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("le_u32", val0, val1, cookie)

  def assert_le_u32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("gt_u32", val0, val1, cookie)

  def assert_ge_u32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("lt_u32", val0, val1, cookie)

  def assert_ge_i32(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("lt_i32", val0, val1, cookie)

  # can left shift w/o losing non-zero bits:
  def assert_no_shift_of(self, val0, shift, stmp, cookie=-1):
    kStr = ""
    # TODO - use BFE here:
    kStr += inst ("s_mov_b32", stmp, hex((shift-1) << (32-log2(shift))), "assert_no_shift_of - compute mask")
    kStr += inst ("s_and_b32", stmp, stmp, val0, "assert_no_shift_of")
    kStr += self.assert_eq(stmp, 0, cookie)
    return kStr


  def bomb_at_wg3d(self, wg0, wg1, wg2, cookie=-1):
    kStr = ""
    tmp0 = sgpr("SaveExecMask")
    tmp1 = sgpr("SaveExecMask"+1)
    kStr += inst("s_cmp_u32", tmp0, sgpr("WorkGroup0"), wg0)
    kStr += inst("s_cmp_u32", tmp1, sgpr("WorkGroup1"), wg1)
    kStr += inst("s_or_b32", tmp0, tmp0, tmp1, "")
    kStr += inst("s_cmp_u32", tmp1, sgpr("WorkGroup2"), wg2)
    kStr += inst("s_or_b32", tmp0, tmp0, tmp1, "")
    kStr += "WIP"



  # asserts if val0 is not an integer multiple of multiple2
  # multiple2 must be a constant and power of 2
  # for example assert_multiple(A, 8) will assert if A is not multiple of 8
  def assert_multiple_b32(self, sval, multiple2, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:

      stmp = sgpr("SaveExecMask") # repurpose to get a tmp sgpr

      kStr += inst("s_and_b32", stmp, sval, multiple2-1, "mask" )
      kStr += inst("s_cmp_eq_u32", stmp, 0, "if maskedBits==0 then SCC=1 == no fault" )
      kStr += inst("s_mov_b64", sgpr("SaveExecMask",2), -1, "")
      kStr += inst("s_cmov_b64", sgpr("SaveExecMask", 2),  0, "Clear exec mask")

      kStr += inst("s_and_saveexec_b64", sgpr("SaveExecMask",2), sgpr("SaveExecMask",2), \
          "assert: saved execmask")

      kStr += self.assertCommon(cookie)

      kStr += inst("s_or_saveexec_b64", "vcc", sgpr("SaveExecMask",2), \
          "assert: restore execmask")

    return kStr

  def assert_s_eq(self, sval0, sval1, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_and_saveexec_b64", sgpr("SaveExecMask",2), sgpr("SaveExecMask",2), \
          "assert: saved execmask")

      kStr += inst("s_mov_b64", sgpr("SaveExecMask",2), -1, "")
      kStr += inst("s_cmp_eq_u32", sval0, sval1, "cmp")
      kStr += inst("s_cmov_b64", sgpr("SaveExecMask", 2),  0, "No assert if SCC=1")

      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b64", "vcc", sgpr("SaveExecMask",2), \
          "assert: restore execmask")

      return kStr


  def assert_scc_is_1(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_and_saveexec_b64", sgpr("SaveExecMask",2), sgpr("SaveExecMask",2), \
          "assert: saved execmask")

      kStr += inst("s_mov_b64", sgpr("SaveExecMask",2), -1, "")
      kStr += inst("s_cmov_b64", sgpr("SaveExecMask", 2),  0, "No assert if SCC=1")

      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b64", "vcc", sgpr("SaveExecMask",2), \
          "assert: restore execmask")

      return kStr

  def assert_scc_is_0(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_and_saveexec_b64", sgpr("SaveExecMask",2), sgpr("SaveExecMask",2), \
          "assert: saved execmask")

      kStr += inst("s_mov_b64", sgpr("SaveExecMask",2), -1, "")
      kStr += inst("s_cmov_b64", sgpr("SaveExecMask", 2),  0, "")
      kStr += inst("s_not_b64", sgpr("SaveExecMask",2), sgpr("SaveExecMask", 2), "Assert if SCC==1")

      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b64", "vcc", sgpr("SaveExecMask",2), \
          "assert: restore execmask")

      return kStr

  # Assert that all bits in vcc are true, or assert/bomb otherwise
  def assert_vcc_all_true(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_or_saveexec_b64", sgpr("SaveExecMask",2), 0, \
          "assert: saved execmask")
      kStr += inst("s_mov_b64", "exec", "vcc", "Predicate based on VCC")
      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b64", "vcc", sgpr("SaveExecMask",2), \
          "assert: restore execmask")
    return kStr

  # Assert that all bits in vcc are false, or assert/bomb otherwise
  def assert_vcc_all_false(self, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_or_saveexec_b64", sgpr("SaveExecMask",2), 0, \
          "assert: saved execmask")
      kStr += inst("s_not_b64", "exec", "vcc", "Predicate based on !VCC")
      kStr += self.assertCommon(cookie)
      kStr += inst("s_or_saveexec_b64", "vcc", sgpr("SaveExecMask",2), \
          "assert: restore execmask")
    return kStr

  # assert v0 + expectedScalarDiff == v1
  # Verify that each element in v1 is scalar offset from v0
  def assert_vector_diff(self, v0, v1, expectedScalarDiff, cookie=-1):
    kStr = ""
    cmpVgpr = self.vgprPool.checkOut(1)
    kStr += inst("_v_add_co_u32", \
                 vgpr(cmpVgpr), "vcc", \
                 expectedScalarDiff, \
                 v0, \
                 "assert_vector_diff add expectedScalarDiff")
    kStr += self.assert_eq(vgpr(cmpVgpr), v1, cookie)
    self.vgprPool.checkIn(cmpVgpr)
    return kStr


  ########################################
  # Store to Debug Buffer
  ########################################
  def dump(self, vgprStore):
    kStr = ""
    if globalParameters["DebugKernel"]:
      afterDump = -1
      if self.db["DebugKernelMaxItems"] != -1:
        afterDump = self.getUniqLabel()
        kStr += inst("s_cmp_lt_u32", sgpr("DebugKernelItems"), 16,  "")
        kStr += inst("s_cbranch_scc0", "label_%04u"%afterDump, \
                     "skip if already wrote enough work-items" )
        kStr += inst("s_add_u32", sgpr("DebugKernelItems"), \
                     sgpr("DebugKernelItems"), \
                     hex(1), "inc items written" )

      kStr += inst("flat_store_dword", vgpr("AddressDbg", 2), \
          vgprStore, "debug dump store" )
      kStr += inst("_v_add_co_u32", vgpr("AddressDbg"), "vcc", vgpr("AddressDbg"), \
          hex(4), "debug dump inc" )

      if self.db["DebugKernelMaxItems"] != -1:
        kStr += "label_%04u:%s  %s" % (afterDump, "// skip debug target", self.endLine)

    return kStr

################################################################################
# Helper Functions
################################################################################

########################################
# Format Instruction
#######################################
def inst(*args):
  # exclude the last parameter (before comment)
  # if it is empty (needed for clang++ assembler)
  if len(args) > 2 and args[len(args)-2] == "":
    params = args[0:len(args)-2]
  else:
    params = args[0:len(args)-1]
  comment = args[len(args)-1]
  formatting = "%s"
  if len(params) > 1:
    formatting += " %s"
  for _ in range(0, len(params)-2):
    formatting += ", %s"
  instStr = formatting % (params)
  line = "%-50s // %s\n" % (instStr, comment)
  return line

########################################
# Format GPRs
########################################
def gpr(*args):
  gprType = args[0]
  args = args[1]
  if isinstance(args[0], int):
    if len(args) == 1:
      return "%s%u"%(gprType, args[0])
    elif len(args) == 2:
      if args[1] == 1:
        return "%s%u"%(gprType, args[0])
      else:
        return "%s[%u:%u]"%(gprType, args[0], args[0]+args[1]-1)
  if isinstance(args[0], str):
    if len(args) == 1:
      return "%s[%sgpr%s]"%(gprType, gprType, args[0])
    elif len(args) == 2:
      if args[1] == 1:
        return "%s[%sgpr%s]"%(gprType, gprType, args[0])
      else:
        return "%s[%sgpr%s:%sgpr%s+%u]"%(gprType, gprType, args[0], \
            gprType, args[0], args[1]-1)
def vgpr(*args):
  return gpr("v", args)
def sgpr(*args):
  return gpr("s", args)

########################################
# Log 2
########################################
def log2(x):
  return int(log(x, 2) + 0.5)

########################################
# Divide & Remainder
# quotient register, remainder register, dividend register, divisor, tmpVgprx2, tmpSgpr
########################################
def vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, \
    doRemainder=True):
  kStr = ""
  if ((divisor & (divisor - 1)) == 0): # pow of 2
    divisor_log2 = log2(divisor)
    kStr += inst("v_lshrrev_b32", vgpr(qReg), divisor_log2, vgpr(dReg), \
        "vectorStaticDiv: %s = %s / %u"%(vgpr(qReg), vgpr(dReg), divisor) )
    if doRemainder:
      kStr += inst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), \
          "vectorStaticDiv: %s = %s %% %u"%(vgpr(rReg), vgpr(dReg), divisor) )
  else:
    """
    if divisor == 30:
      shift = 32+2
    elif divisor >= 14:
      shift = 32+4
    elif divisor >= 7:
      shift = 32+3
    elif divisor >= 6:
      shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
    elif divisor >= 5:
      shift = 32+2
    elif divisor >= 3:
      shift = 32+1
    """
    shift = 32+1
    magic = ((2**shift) // divisor) + 1
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(magic), "")
    kStr += inst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("v_lshrrev_b64", vgpr(tmpVgpr,2), hex(shift), vgpr(tmpVgpr,2), "")
    kStr += inst("v_mov_b32", vgpr(qReg), vgpr(tmpVgpr), "vectorStaticDiv: quotient")
    if doRemainder:
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), "divisor")
      kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(qReg), sgpr(tmpSgpr), "vectorStaticDiv: product = quotient * divisor")
      kStr += inst("_v_sub_co_u32", vgpr(rReg), "vcc", vgpr(dReg), vgpr(tmpVgpr), "vectorStaticDiv: remainder = dividend - product")
  return kStr

def vectorStaticDivide(qReg, dReg, divisor, tmpVgpr, tmpSgpr):
  rReg = -1 # unused
  kStr = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, False)
  return kStr

def vectorStaticRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr):
  kStr = ""
  if ((divisor & (divisor - 1)) == 0): # pow of 2
    kStr += inst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), \
        "vectorStaticDiv: %s = %s %% %u"%(vgpr(rReg), vgpr(dReg), divisor) )
  else:
    """
    if divisor == 30:
      shift = 32+2
    elif divisor >= 14:
      shift = 32+4
    elif divisor >= 7:
      shift = 32+3
    elif divisor >= 6:
      shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
    elif divisor >= 5:
      shift = 32+2
    elif divisor >= 3:
      shift = 32+1
    """
    shift = 32+1
    magic = ((2**shift) // divisor) + 1
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(magic), "")
    kStr += inst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(shift), "")
    kStr += inst("v_lshrrev_b64", vgpr(tmpVgpr,2), sgpr(tmpSgpr), vgpr(tmpVgpr,2), "")
    kStr += inst("v_mov_b32", vgpr(qReg), vgpr(tmpVgpr), "vectorStaticDiv: quotient")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), "divisor")
    kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(qReg), sgpr(tmpSgpr), "vectorStaticDiv: product = quotient * divisor")
    kStr += inst("_v_sub_co_u32", vgpr(rReg), "vcc", vgpr(dReg), vgpr(tmpVgpr), "vectorStaticDiv: remainder = dividend - product")
  return kStr

# only used for loop unroll and GlobalSplitU
# doRemainder==0 : compute quotient only
# doRemainder==1 : compute quotient and remainder
# doRemainder==2 : only compute remainder (not quotient unless required for remainder)
# dreg == dividend
# tmpSgpr must be 2 SPGRs
# qReg and dReg can be "sgpr[..]" or names of sgpr (will call sgpr)
def scalarStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr, \
    doRemainder=1):

  assert (qReg != tmpSgpr)


  qRegSgpr = qReg if type(qReg) == str and qReg.startswith("s[") else sgpr(qReg)

  dRegSgpr = dReg if type(dReg) == str and dReg.startswith("s[") else sgpr(dReg)

  kStr = ""
  if ((divisor & (divisor - 1)) == 0): # pow of 2
    divisor_log2 = log2(divisor)
    if doRemainder != 2:
      kStr += inst("s_lshr_b32", qRegSgpr, dRegSgpr, divisor_log2, \
          "%s = %s / %u"%(qRegSgpr, dRegSgpr, divisor) )
    if doRemainder:
      kStr += inst("s_and_b32", sgpr(rReg), (divisor-1), dRegSgpr, \
          "%s = %s %% %u"%(sgpr(rReg), dRegSgpr, divisor) )
  else:
    """
    if divisor == 30:
      shift = 32+2
    elif divisor >= 14:
      shift = 32+4
    elif divisor >= 6:
      shift = 32+3
    elif divisor >= 5:
      shift = 32+2
    elif divisor >= 3:
      shift = 32+1
    """
    shift = 32+1
    magic = ((2**shift) // divisor) + 1
    magicHi = magic // (2**16)
    magicLo = magic & (2**16-1)

    kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), hex(0), "STATIC_DIV: divisior=%s"%divisor)
    kStr += inst("s_mul_i32", sgpr(tmpSgpr+0), hex(magicHi), dRegSgpr, "tmp1 = dividend * magic hi")
    kStr += inst("s_lshl_b64", sgpr(tmpSgpr,2), sgpr(tmpSgpr,2), hex(16), "left shift 16 bits")
    kStr += inst("s_mul_i32", qRegSgpr, dRegSgpr, hex(magicLo), "tmp0 = dividend * magic lo")
    kStr += inst("s_add_u32", sgpr(tmpSgpr+0), qRegSgpr, sgpr(tmpSgpr+0), "add lo")
    kStr += inst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), hex(0), "add hi")
    kStr += inst("s_lshr_b64", sgpr(tmpSgpr,2), sgpr(tmpSgpr,2), hex(shift), "tmp1 = (dividend * magic) << shift")
    kStr += inst("s_mov_b32", qRegSgpr, sgpr(tmpSgpr), "quotient")
    if doRemainder:
      kStr += inst("s_mul_i32", sgpr(tmpSgpr), qRegSgpr, hex(divisor), "quotient*divisor")
      kStr += inst("s_sub_u32", sgpr(rReg), dRegSgpr, sgpr(tmpSgpr), "rReg = dividend - quotient*divisor")
  return kStr

########################################
# Multiply
# product register, operand register, multiplier
########################################
def staticMultiply(product, operand, multiplier, tmpSgpr=None):
  if multiplier == 0:
      return inst("v_mov_b32", product, hex(multiplier), "staticMultiply: %s = %u" % (product, multiplier) )
  elif ((multiplier & (multiplier - 1)) == 0): # pow of 2
    multiplier_log2 = log2(multiplier)
    if multiplier_log2==0 and product == operand:
      return ""
    else:
      return inst("v_lshlrev_b32", product, multiplier_log2, operand, \
          "staticMultiply: %s = %s * %u"%(product, operand, multiplier) )
  else:
    kStr = ""
    if product == operand:
      kStr += inst("s_mov_b32", tmpSgpr, hex(multiplier), \
          "staticMultiply: %s = %u"%(tmpSgpr, multiplier) )
      kStr += inst("v_mul_lo_u32", product, tmpSgpr, operand, \
          "staticMultiply: %s *= %s"%(product, operand) )
    else:
      kStr += inst("v_mov_b32", product, hex(multiplier), \
          "staticMultiply: %s = %u"%(product, multiplier) )
      kStr += inst("v_mul_lo_u32", product, product, operand, \
          "staticMultiply: %s *= %s"%(product, operand) )
    return kStr
