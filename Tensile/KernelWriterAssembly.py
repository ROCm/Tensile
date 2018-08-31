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

from SolutionStructs import DataType
from Common import globalParameters, printExit, printWarning
from KernelWriter import KernelWriter
from math import log, ceil
import collections
import traceback

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
    if nonTemporal/2==1:
      instStr += " slc"
    line = "%-50s // %s%s" % (instStr, comment, self.endLine)
    return line
  def __str__(self):
    return self.name

################################################################################
# RegisterPool
################################################################################
class RegisterPool:
  statusUnAvailable = 0
  statusAvailable = 1
  statusInUse = 2

  class Register:
    def __init__(self, status, tag=""):
      self.status = status
      self.tag = tag


  ########################################
  # Init
  def __init__(self, size, printRP=0):
    self.printRP=printRP
    self.pool = [self.Register(self.statusUnAvailable, "init") for i in range(0,size)]
    self.checkOutSize = {}

  ########################################
  # Adds registers to the pool so they can be used as temps
  # Add
  def add(self, start, size, tag=""):
    # reserve space
    if self.printRP:
      print "RP::add(%u, %u for '%s')"%(start,size,tag)
    newSize = start + size
    oldSize = len(self.pool)
    if newSize > oldSize:
      for i in range(0, newSize-oldSize):
        self.pool.append(self.Register(self.statusUnAvailable,tag))
    # mark as available
    for i in range(start, start+size):
      if self.pool[i].status == self.statusUnAvailable:
        self.pool[i].status = self.statusAvailable
      elif self.pool[i].status == self.statusAvailable:
        printWarning("RegisterPool::add(%u,%u) pool[%u] already available" % (start, size, i))
      elif self.pool[i].status == self.statusInUse:
        printWarning("RegisterPool::add(%u,%u) pool[%u] already in use" % (start, size, i))
      else:
        printExit("RegisterPool::add(%u,%u) pool[%u] = %s" % (start, size, i, self.pool[i].status))

  ########################################
  # Remove
  # Removes registers from the pool so they cannot be subsequently allocated for tmps
  def remove(self, start, size):
    if self.printRP:
      print "RP::remove(%u,%u)"%(start,size)
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
        printWarning("RegisterPool::remove(%u,%u) pool[%u] still in use" % (start, size, i))
      else:
        printExit("RegisterPool::remove(%u,%u) pool[%u] = %s" % (start, size, i, self.pool[i].status))

  ########################################
  # Check Out
  def checkOut(self, size, tag="", preventOverflow=False):
    return self.checkOutAligned(size, 1, tag, preventOverflow)
  def checkOutAligned(self, size, alignment, tag="", preventOverflow=False):
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
      self.checkOutSize[found] = size
      if self.printRP:
        print "RP::checkOut '%s' (%u,%u) @ %u avail=%u"%(tag, size,alignment, found, self.available())
      return found
    # need overflow
    else:
      #print "RegisterPool::checkOutAligned(%u,%u) overflowing past %u" % (size, alignment, len(self.pool))
      # where does tail sequence of available registers begin
      assert (not preventOverflow)
      start = len(self.pool)
      for i in range(len(self.pool)-1, 0, -1):
        if self.pool[i].status == self.statusAvailable:
          start = i
          continue
        else:
          break
      #print "Start: ", start
      # move forward for alignment
      start = ((start + alignment - 1) / alignment) * alignment
      #print "Aligned Start: ", start
      # new checkout can begin at start
      newSize = start + size
      oldSize = len(self.pool)
      overflow = newSize - oldSize
      #print "Overflow: ", overflow
      for i in range(start, len(self.pool)):
        self.pool[i].status = self.statusInUse
      for i in range(0, overflow):
        self.pool.append(self.Register(self.statusInUse,tag))
      self.checkOutSize[start] = size
      if self.printRP:
        print self.state()
        print "RP::checkOut' %s' (%u,%u) @ %u (overflow)"%(tag, size, alignment, start)
      return start

  ########################################
  # Check In
  def checkIn(self, start):
    if self.printRP:
      print "RP::checkIn() @ %u"%(start)
    if start in self.checkOutSize:
      size = self.checkOutSize[start]
      for i in range(start, start+size):
        self.pool[i].status = self.statusAvailable
      self.checkOutSize.pop(start)
      if self.printRP:
        print "RP::checkIn() @ %u +%u"%(start,size)
    else:
      if 0:
        traceback.print_stack(None)
      printWarning("RegisterPool::checkIn(%s) but it was never checked out"%start)

  ########################################
  # Size
  def size(self):
    return len(self.pool)


  ########################################
  # Number of available registers
  def available(self):
    numAvailable = 0
    for s in self.pool:
      if s.status == self.statusAvailable:
        numAvailable += 1
    return numAvailable

  ########################################
  # Size of largest consecutive block
  def availableBlock(self):
    maxAvailable = 0
    numAvailable = 0
    for s in self.pool:
      if s.status == self.statusAvailable:
        numAvailable += 1
      else:
        if numAvailable > maxAvailable:
          maxAvailable = numAvailable
        numAvailable = 0
    if numAvailable > maxAvailable:
      maxAvailable = numAvailable
    #print self.state()
    #print "available()=", self.available(), "availableBlock()=",maxAvailable
    return maxAvailable

  ########################################
  def checkFinalState(self):
    for si in range(0,len(self.pool)):
      if self.pool[si].status == self.statusInUse:
        printWarning("RegisterPool::checkFinalState: temp (%s, '%s') was never checked in." \
            %(si, self.pool[si].tag))
        if self.printRP:
          print self.state()

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
            pvs += "%u"%((i%priorPlaceValue)/placeValue)
          else:
            pvs += " "
        stateStr += pvs + "\n"
    for i in range(0, len(self.pool)):
      if self.pool[i].status == self.statusUnAvailable:
        stateStr += "."
      elif self.pool[i].status == self.statusAvailable:
        stateStr += "|"
      elif self.pool[i].status == self.statusInUse:
        stateStr += "#"
    return stateStr


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

    # Various debug flags and modes
    self.db = {}
    self.db["EnableAsserts"]     = True  # Enable assertion codegen
    self.db["DebugKernelMaxItems"] = 16  # Capture first N(=16) print values, ignore subsequent.  If -1, debug writing is faster but writing more than 16 values is undefined.

    # Chicken bit to add conservative synchronization at strategic points:
    # 0x1 = waitcnt + barrier after vector load
    # 0x2 = waitcnt at self.wait() for globalRead
    # 0x4 = waitcnt at self.wait() for localWrite
    # 0x8 = waitcnt at self.wait() for localRead
    self.db["ConservativeWaitCnt"] = 0x0

    self.db["InitLds"]     = True  # Initialize LDS at start of kernel
    self.printedAssertCnt  = 0
    self.initLdsValue     = 0xFFFFFFFF  # Value to use for LDS Init, if enabled

    # Check A and B values loaded from memory to ensure they are 1
    # Requires DataInitTypeAB=1.
    # Mismatches will assert (generate GPUVM fault)
    self.db["CheckValue1A"] = False
    self.db["CheckValue1B"] = False

    # print register pool checkins and checkouts
    self.db["PrintRP"] = False

    # Number of times localReadDo(localWriteDo) has been called by the code-generator.
    # Used to control debug enablement.
    # Note this increments as the assembly code is generated not as it executes
    # so it can be used to determine which iteration of the unroll is being generated
    self.localReadDoCnt   = 0
    self.localWriteDoCnt  = 0

    self.maxVgprs = 256
    self.maxSgprs = 99

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


  ########################################
  def getOccupancy(self, kernel, vgprs):
    multiplier = int(ceil(max(kernel["NumThreads"], 256) / 256.0))
    # example: wg=512 multiplier=2, 1024=4

    maxLds = 65536
    ldsSize = kernel["LdsNumElements"] * kernel["ProblemType"]["DataType"].numBytes()
    ldsSize = (ldsSize + 255) & 0xff00 # 256-byte granularity
    ldsLimitedOccupancy = int(ceil(maxLds / float(ldsSize)))

    vgprs *= multiplier
    vgprLimitedOccupancy =  self.vgprOccupancy[vgprs] if vgprs <= 256 else 0

    return min(ldsLimitedOccupancy, vgprLimitedOccupancy)

  ########################################
  # Get Label
  def getLabel(self, name):
    if name not in self.labels:
      self.labels[name] = len(self.labels)
    return self.labels[name]

  def getUniqLabel(self):
    name = "uniq_label_" + str(len(self.labels))
    return self.getLabel(name)

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

  def setStartTmpPool(self, newStartTmpPool):
    #print "set tmpSgprPool to ", newStartTmpPool
    self.startSgprTmpPool = newStartTmpPool

  def getTmpSgpr(self, num):
    pad = 0 if num ==1 else self.startSgprTmpPool & 0x1
    if self.startSgprTmpPool+num+pad > self.totalSgprs:
      self.totalSgprs = self.startSgprTmpPool + num + pad
      #print "warning: growing SGPR pool to ", self.totalSgprs

    return self.startSgprTmpPool + pad

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

      tmp = self.vgprPool.checkOut(1)
      kStr += inst("v_mov_b32", vgpr(tmp), sgprStore, "Debug")
      kStr += inst("flat_store_dword", vgpr("AddressDbg", 2), \
          vgpr(tmp), "debug dump sgpr store" )
      kStr += inst("_v_add_co_u32", vgpr("AddressDbg"), "vcc", vgpr("AddressDbg"), \
          hex(4), "debug dump inc" )
      self.vgprPool.checkIn(tmp)

      if self.db["DebugKernelMaxItems"] != -1:
        kStr += "label_%04u:%s  %s" % (afterDump, "// skip debug target", self.endLine)

    return kStr


  ##############################################################################
  #
  #   Functions to Write Kernel Segments
  #
  ##############################################################################

  def defineSgpr(self, name, numSgprs, align=1):
    if numSgprs == 0: return

    # round up to next alignment boundary:
    self.sgprIdx = ((self.sgprIdx+align-1) / align) * align

    self.sgprs[name] = self.sgprIdx
    self.sgprIdx += numSgprs
    return

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def initKernel(self, kernel, tPA, tPB ):
    super(KernelWriterAssembly, self).initKernel(kernel, tPA, tPB)

    dkp = kernel["DisableKernelPieces"]
    self.do["NullKernel"]  = dkp >= 9 or dkp == -9

    self.sgprs=collections.OrderedDict()
    self.sgprIdx = 0

    self.LdsOOB = 0xF00000

    #---
    # Internal optimization and debug controls.
    # These have a default which is almost always faster so don't make a full-blown YAML parm
    # But have a control here so we can disable for debugging and also easily tell
    # which parts of the code were changed to support the new mode.
    self.globalReadIncsUseVgpr = False if kernel["BufferLoad"] else True

    self.checkGRO = False
    # checkGRO requires useSgprForGRO=0 so that code allocates and uses
    # the VGPRs that are used for the GRO offset checking
    assert not (kernel["UseSgprForGRO"] and self.checkGRO)

    # Debug mode to explore combining VGPRs.
    # Saves VGPRs but doesn't generate correct answer
    self.combineLocalAddresses = 0

    # ISA version, such as 803
    self.version = globalParameters["CurrentISA"]
    if "ISA" in kernel:
      self.version = kernel["ISA"]

    self.AsmBugs = {}
    self.AsmBugs["ExplicitCO"] = globalParameters["AsmCaps"][self.version]["HasExplicitCO"]

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
        "%s, %s" )
    flat_load_dwordx2 = MemoryInstruction("flat_load_dwordx2",  1, 0, 0, 2, \
        "%s, %s" )
    flat_load_dword = MemoryInstruction("flat_load_dword",      1, 0, 0, 1, \
        "%s, %s" )

    buffer_load_dwordx4 = MemoryInstruction("buffer_load_dwordx4", 1, 0, 0, 4, \
        "%s, %s, %s, %s offen offset:0 %s" )
    buffer_load_dwordx2 = MemoryInstruction("buffer_load_dwordx2", 1, 0, 0, 2, \
        "%s, %s, %s, %s offen offset:0 %s" )
    buffer_load_dword = MemoryInstruction("buffer_load_dword", 1, 0, 0, 1, \
        "%s, %s, %s, %s offen offset:0 %s" )

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
    else:
      chosen_load_dwordx4 = flat_load_dwordx4
      chosen_load_dwordx2 = flat_load_dwordx2
      chosen_load_dword   = flat_load_dword

    chosen_store_dwordx4 = flat_store_dwordx4
    chosen_store_dwordx2 = flat_store_dwordx2
    chosen_store_dword   = flat_store_dword

    self.memoryInstructions = {
        (8,0,3): {
          "GlobalRead": [ chosen_load_dwordx4, chosen_load_dwordx2,
            chosen_load_dword ],
          "GlobalWrite": [ chosen_store_dwordx4, chosen_store_dwordx2,
            chosen_store_dword ],
          "LocalRead": [ ds_read_b128, ds_read2_b64,
            ds_read_b64, ds_read2_b32, ds_read_b32 ],
          "LocalWrite": [ ds_write_b128, ds_write2_b64,
            ds_write_b64, ds_write2_b32, ds_write_b32, ds_write_b16 ]
          }, # 803
        (9,0,0): {
          "GlobalRead": [ chosen_load_dwordx4, chosen_load_dwordx2,
            chosen_load_dword ],
          "GlobalWrite": [ chosen_store_dwordx4, chosen_store_dwordx2,
            chosen_store_dword ],
          "LocalRead": [ ds_read_b128, ds_read2_b64,
            ds_read_b64, ds_read2_b32, ds_read_b32 ],
          "LocalWrite": [ ds_write_b128, ds_write2_b64,
            ds_write_b64, ds_write2_b32, ds_write_b32, ds_write_b16 ]
          } # 900
        }


    self.overflowedResources = False # if true, comment out whole kernel

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
        else:
            print "HighPrecisionAccumulate only valid when DataType is half."
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
    self.globalReadWidthA = (tPA["nrcv"]*tPA["bpe"])/self.bpr
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
    self.globalReadWidthB = (tPB["nrcv"]*tPB["bpe"])/self.bpr
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
    self.localWriteWidthA = (tPA["nwcv"]*tPA["bpe"])/self.bpr
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
    self.localWriteStrideTileA = (self.localWriteStrideTileA*tPA["bpe"])/self.bpr
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
        (self.localWriteStrideUnrollA*tPA["bpe"])/self.bpr
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
    self.localWriteWidthB = (tPB["nwcv"]*tPB["bpe"])/self.bpr
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
    self.localWriteStrideTileB = (self.localWriteStrideTileB*tPB["bpe"])/self.bpr
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
        (self.localWriteStrideUnrollB*tPB["bpe"])/self.bpr
    self.localWriteInstructionIdxB = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthB, \
        kernel["LocalWrite2B"], \
        self.localWrite2CoalescedB, self.localWrite2PerpendicularB,
        [self.localWriteStrideTileB, self.localWriteStrideUnrollB] )

    ########################################
    # localRead A
    localReadWidth = (kernel["VectorWidth"] * tPA["bpe"])/self.bpr
    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedA = \
        (kernel["ThreadTile0"] * tPA["bpe"])/self.bpr
    self.localRead2CoalescedA = kernel["ThreadTile0"]/kernel["VectorWidth"] > 1
    self.localReadInstructionIdxA = \
        self.selectMemoryInstruction("LocalRead", localReadWidth, \
        kernel["LocalRead2A"], \
        self.localRead2CoalescedA, localRead2Perpendicular,
        [self.localReadStrideCoalescedA] )

    ########################################
    # localRead B
    localReadWidth = (kernel["VectorWidth"] * tPB["bpe"])/self.bpr
    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedB = \
    (kernel["ThreadTile1"] * tPB["bpe"])/self.bpr
    self.localRead2CoalescedB = kernel["ThreadTile1"]/kernel["VectorWidth"] > 1
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
    tPA["nrcvpi"] = int((self.globalReadInstructionA.totalWidth*self.bpr) / tPA["bpe"])
    tPB["nrcvpi"] = int((self.globalReadInstructionB.totalWidth*self.bpr) / tPB["bpe"])
    tPA["nwcvpi"] = int((self.localWriteInstructionA.totalWidth*self.bpr) / tPA["bpe"])
    tPB["nwcvpi"] = int((self.localWriteInstructionB.totalWidth*self.bpr) / tPB["bpe"])

    ####################################
    # VGPR Allocation
    ####################################

    ####################################
    # num vgprs: valu
#jgolds bpeCinternal because we are allocating accumulation registers here
    self.numVgprValuC = (kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.bpeCinternal)/self.bpr

    valuBlocks = (1+kernel["PrefetchLocalRead"]) * kernel["InnerUnroll"]
    self.numVgprValuAPerBlock = (kernel["ThreadTileA"]*tPA["bpe"])/self.bpr
    self.numVgprValuBPerBlock = (kernel["ThreadTileB"]*tPB["bpe"])/self.bpr
    numVgprValuA = self.numVgprValuAPerBlock * valuBlocks
    numVgprValuB = self.numVgprValuBPerBlock * valuBlocks

    ####################################
    # num vgprs: global -> local elements
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
      numVgprG2LA = (kernel["NumLoadsCoalescedA"] \
          * kernel["NumLoadsPerpendicularA"] * kernel["GlobalLoadVectorWidthA"] * tPA["bpe"])/self.bpr
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      numVgprG2LB = (kernel["NumLoadsCoalescedB"] \
          * kernel["NumLoadsPerpendicularB"] * kernel["GlobalLoadVectorWidthB"] * tPB["bpe"])/self.bpr

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
    numVgprLocalWriteAddressesA = 0 if kernel["LocalWriteUseSgprA"] else 1 * self.rpla

    #numLocalWritesB = kernel["NumLoadsCoalescedB"] \
    #    * nlp * self.numWriteVectorComponentsB
    #numLocalWriteInstructionsB = numLocalWritesB \
    #    / self.localWriteInstructionB[self.instructionIdxNumOffsets]
    numVgprLocalWriteAddressesB = 0 if kernel["LocalWriteUseSgprB"] else 1 * self.rpla

    ####################################
    # num vgprs: global read addresses
    numGlobalReadsA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["GlobalLoadVectorWidthA"] \
        * self.numReadVectorComponentsA
    numGlobalReadInstructionsA = (numGlobalReadsA * tPA["bpe"])\
        / (self.globalReadInstructionA.blockWidth * 4)

    if kernel["BufferLoad"]:
      numGlobalReadOffsetsA = numGlobalReadInstructionsA * self.rpgo
    else:
      numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.rpga

    numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["GlobalLoadVectorWidthB"] \
        * self.numReadVectorComponentsB
    numGlobalReadInstructionsB = (numGlobalReadsB * tPB["bpe"]) \
        / (self.globalReadInstructionB.blockWidth * 4)
    if kernel["BufferLoad"]:
      numGlobalReadOffsetsB = numGlobalReadInstructionsB * self.rpgo 
    else:
      numVgprGlobalReadAddressesB = numGlobalReadInstructionsB * self.rpga
    numVgprSerial = 1
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
        self.startVgprG2LA = vgprIdx; vgprIdx += numVgprG2LA
      else: # g2l can overlap valu
        self.startVgprG2LA = self.startVgprValuA
        vgprIdx = self.startVgprValuA \
            + max(self.numVgprValuAPerBlock*valuBlocks, numVgprG2LA)

    self.startVgprValuB = vgprIdx; vgprIdx += numVgprValuB
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
      if kernel["PrefetchGlobalRead"]:
        self.startVgprG2LB = vgprIdx; vgprIdx += numVgprG2LB
      else: # g2l can overlap valu
        self.startVgprG2LB = self.startVgprValuB
        vgprIdx = self.startVgprValuB \
            + max(self.numVgprValuBPerBlock*valuBlocks, numVgprG2LB)

    # Registers allocated above this point can be used as temps during setup
    # Registers above here are reserved in initC, near the end of the setup
    # code
    self.lastPreLoopTempVgpr = vgprIdx+1
    #----------------------------------

    self.startVgprLocalReadAddressesA = vgprIdx
    vgprIdx += numVgprLocalReadAddressesA
    if self.combineLocalAddresses:
      self.startVgprLocalReadAddressesB = self.startVgprLocalReadAddressesA
    else:
      self.startVgprLocalReadAddressesB = vgprIdx
      vgprIdx += numVgprLocalReadAddressesB
    if not kernel["LocalWriteUseSgprA"]:
      if self.combineLocalAddresses:
        self.startVgprLocalWriteAddressesA = self.startVgprLocalReadAddressesA
      else:
        self.startVgprLocalWriteAddressesA = vgprIdx
        vgprIdx += numVgprLocalWriteAddressesA

    if not kernel["LocalWriteUseSgprB"]:
      if self.combineLocalAddresses:
        self.startVgprLocalWriteAddressesB = self.startVgprLocalReadAddressesA
      else:
        self.startVgprLocalWriteAddressesB = vgprIdx
        vgprIdx += numVgprLocalWriteAddressesB

    # BufferLoad:
    # Uses a resource descriptor (SRD) which is stored in 4 SGPRs and thus shared by all work-items.
    # Each work-item also uses  a unique 32-bit offset into vgprGlobalReadOffset.  These offsets are set when 
    # the tile is initialized and stay constant through the execution of the kernel.
    # The base address in the SRD is updated when the algoritm moves to a new tile
    # BufferLoad disables the gptGlobalReadAddr used in flat addressing.
    if kernel["BufferLoad"]:
       self.startVgprGlobalReadOffsetA = vgprIdx
       vgprIdx += 1 if kernel["UseSgprForGRO"] else numGlobalReadOffsetsA
       self.startVgprGlobalReadOffsetB = vgprIdx
       vgprIdx += 1 if kernel["UseSgprForGRO"] else numGlobalReadOffsetsB
    else:
      self.startVgprGlobalReadAddressesA = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesA
      self.startVgprGlobalReadAddressesB = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesB
    self.startVgprGlobalReadIncsA = vgprIdx
    vgprIdx += numVgprGlobalReadIncsA
    self.startVgprGlobalReadIncsB = vgprIdx
    vgprIdx += numVgprGlobalReadIncsB
    self.startVgprAddressDbg = vgprIdx
    vgprIdx += numVgprAddressDbg
    self.startVgprSerial = vgprIdx
    vgprIdx += numVgprSerial

    # Point at last VGPR that can be reclaimed for use in the summation loop 
    # This should be just BEFORE the vgprSerial, which may still be used.
    # If more VGPRs are added here be aware of the register reclaim code in
    # endSummation - registers that should be preserved should be allocated
    # with numbers higher than vgprReclaimAfterSummation
    self.vgprReclaimAfterSummation = self.startVgprSerial-1

    # tmp vgprs
    #minVgprTmp = 1
    #if kernel["LoopTail"]:
    #  minVgprTmp += 4
    #if globalParameters["DebugKernel"]:
    #  minVgprTmp += 2
    #vgprIdx += minVgprTmp
    #print2("%3u vgprs <- %s" % (vgprIdx, self.kernelName) )
    vgprPerCU = 65536
    vgprPerThreadPerOccupancy = vgprPerCU / kernel["NumThreads"]
    numWorkGroupsPerCU = vgprPerThreadPerOccupancy / vgprIdx
    if numWorkGroupsPerCU < 1:
      self.overflowedResources = True
      numWorkGroupsPerCU  = 1 # dummy value

    self.totalVgprs = vgprIdx

    ########################################
    # SGPR Allocation
    ########################################

    ####################################
    # num sgprs: initial kernel state
    numSgprAddressC = self.rpga # til end
    numSgprAddressA = self.rpga # til read offsets
    numSgprAddressB = self.rpga # til read offsets
    numSgprOffsetC = 1
    numSgprOffsetA = 1
    numSgprOffsetB = 1
    numSgprAlpha = max(1,int(tPA["bpe"]/4))
    numSgprBeta  = max(1,int(self.bpeCexternal/4)) if kernel["ProblemType"]["UseBeta"] else 0
    self.numSgprStridesC = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprStridesA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    self.numSgprStridesB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    if not kernel["ProblemType"]["UseInitialStrides"]:
      self.numSgprStridesC -= 1
      self.numSgprStridesA -= 1
      self.numSgprStridesB -= 1
    self.numSgprSizesSum = kernel["ProblemType"]["NumIndicesSummation"]

    self.numSgprSizesFree = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprAddressDbg = self.rpga if globalParameters["DebugKernel"] else 0

    ####################################
    # num sgprs: global read increments
    if self.globalReadIncsUseVgpr:
      numSgprGlobalReadIncsA = 0
      numSgprGlobalReadIncsB = 0
    else:
      numSgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga
      numSgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga

    numSgprLoopCounters = 1 * kernel["ProblemType"]["NumIndicesSummation"]


    ########################################
    # SGPR Assignment according to AMDGPU-ABI
    ########################################

    self.defineSgpr("KernArgAddress", self.rpga)
    assert(self.sgprs["KernArgAddress"] ==  0) # kernarg is passed to kernel as SGPR0

    if kernel["WorkGroupMapping"]>0 :
      self.defineSgpr("WorkGroup0", 1)
      self.defineSgpr("WorkGroup1", 1)
    else:
      self.defineSgpr("WorkGroup1", 1)
      self.defineSgpr("WorkGroup0", 1)

    assert (kernel["ProblemType"]["NumIndicesC"] <= 3) # else seems registers below would collide??
    for i in range(2, kernel["ProblemType"]["NumIndicesC"]):
      self.defineSgpr("WorkGroup%u"%i, 1)

    self.defineSgpr("NumWorkGroups0", 1)
    self.defineSgpr("NumWorkGroups1", 1)

    if kernel["BufferLoad"]:
       # resource descriptor (SRD) A and B, must be aligned on 4-SGPR boundary
      self.defineSgpr("SrdA", 4, 4)
      self.defineSgpr("SrdB", 4, 4)
    if kernel["BufferStore"]:
      self.defineSgpr("SrdC", 4, 4)

    # To avoid corrupting tmp sgprs that may be used around the assert,
    # reserve some sgprs to save/restore the execmask
    if self.db["EnableAsserts"]:
      self.defineSgpr("SaveExecMask", 2)

    self.defineSgpr("GSUSumIdx", 2 if kernel["GlobalSplitU"] > 1 else 0)
    self.defineSgpr("AddressC", numSgprAddressC)
    self.defineSgpr("StridesC", self.numSgprStridesC)

    # doubles need to be aligned to even
    #if tPA["bpe"] > 4 and self.sgprIdx%2==1:
    #  self.sgprIdx += 1
    self.defineSgpr("Alpha", numSgprAlpha, numSgprAlpha)
    if kernel["ProblemType"]["UseBeta"]:
      self.defineSgpr("Beta", numSgprBeta, numSgprBeta)

    self.defineSgpr("SizesFree", self.numSgprSizesFree)
    self.defineSgpr("SizesSum", self.numSgprSizesSum)
    self.defineSgpr("LoopCounters", numSgprLoopCounters)
    self.defineSgpr("StridesA", self.numSgprStridesA)
    self.defineSgpr("StridesB", self.numSgprStridesB)
    self.defineSgpr("AddressA", numSgprAddressA)
    self.defineSgpr("AddressB", numSgprAddressB)
    if kernel["FractionalLoad"]:
      if kernel["fractionalPerpOverhangA"]:
        self.defineSgpr("PerpOverhangVccA", 2, 2)
      if kernel["fractionalPerpOverhangB"]:
        self.defineSgpr("PerpOverhangVccB", 2, 2)
    if globalParameters["DebugKernel"]:
      self.defineSgpr("AddressDbg", self.numSgprAddressDbg)
      self.defineSgpr("DebugKernelItems", 1)

    #------------------------
    # Registers defined below this point are not available in the post-loop
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    self.lastPostLoopSgpr = self.sgprIdx

    self.defineSgpr("OffsetC", numSgprOffsetC)
    self.defineSgpr("OffsetA", numSgprOffsetA)
    self.defineSgpr("OffsetB", numSgprOffsetB)

    self.defineSgpr("GlobalReadIncsA", numSgprGlobalReadIncsA)
    self.defineSgpr("GlobalReadIncsB", numSgprGlobalReadIncsB)

    if kernel["LocalWriteUseSgprA"]:
        self.defineSgpr("LocalWriteAddrA", 1)
    if kernel["LocalWriteUseSgprB"]:
        self.defineSgpr("LocalWriteAddrB", 1)

    if kernel["UseSgprForGRO"]:
      self.defineSgpr("ScalarGlobalReadOffsetA", numGlobalReadOffsetsA-1)
      self.defineSgpr("ScalarGlobalReadOffsetB", numGlobalReadOffsetsB-1)

    self.totalSgprs = self.sgprIdx
    self.setStartTmpPool(self.totalSgprs)

    ########################################
    # Register Pools
    ########################################
    #print "TotalVgprs", self.totalVgprs
    self.vgprPool = RegisterPool(self.totalVgprs, self.db["PrintRP"])
    #print self.vgprPool.state()

    self.vgprPool.add(self.startVgprValuC, \
        self.lastPreLoopTempVgpr - self.startVgprValuC, "CoreRegs") # Add as available
    #print self.vgprPool.state()

    self.sgprPool = RegisterPool(self.totalSgprs)

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
    #self.getLabel("PrefetchGlobalBegin")
    self.getLabel("PrefetchGlobalEnd")
    self.getLabel("LoopBegin%s"%(unrollChar))
    self.getLabel("LoopEnd%s"%(unrollChar))
    self.getLabel("PrefetchGlobalLastIterEnd")
    self.getLabel("TailLoopBegin%s"%(unrollChar))
    self.getLabel("TailLoopEnd%s"%(unrollChar))
    # shift vectors determined later

    if self.db["InitLds"] : print ("\n***WARNING: InitLds enabled, may impact performance\n")
    if self.db["ConservativeWaitCnt"] : print ("\n***WARNING: ConservativeWaitCnt enabled, may impact performance\n")
    if self.do["KeepDirectToLdsAlloc"] : print ("\n***WARNING: KeepDirectToLdsAlloc enabled, may impact performance\n")
    if not kernel["LoopTail"] : print ("\n***WARNING: LoopTail disabled, kernel may not function correctly for all inputs\n")
    if self.db["CheckValue1A"] : print ("\n***WARNING: CheckValue1A enabled, may impact performance\n")
    if self.db["CheckValue1B"] : print ("\n***WARNING: CheckValue1B enabled, may impact performance\n")
    if self.db["PrintRP"] : print ("\n***WARNING: PrintRP enabled, may generate verbose output\n")
    if kernel["CheckTensorDimAsserts"] : print ("\n***WARNING: CheckTensorDimAsserts enabled, may impact performance\n")


  ##############################################################################
  # format macro
  def macroRegister(self, name, value):
    return ".set %s, %s%s" % (name, value, self.endLine)


  ##############################################################################
  # Function Prefix
  ##############################################################################
  def functionPrefix(self, kernel):
    kStr = ""


    return kStr

  def defineMACMacro(self, kernel, innerUnroll):

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
      kStr += ".macro MAC_%ux%u_X%u%s" \
          % (kernel["ThreadTile0"], kernel["ThreadTile1"], m, ext)
      kStr += self.endLine
      macIdx = 0
      # half precision
      if kernel["ProblemType"]["DataType"].isHalf():
        for blockB in range(0, kernel["ThreadTile1"]/2):
          for blockA in range(0, kernel["ThreadTile0"]/2):
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
            elif self.version == (9,0,0):
              if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                # we treat HighPrecisionAccumulate as expanded packed math
                b = blockB*2
                a = blockA*2
                if kernel["LocalDotLayout"] > 1:    # Only supports LocalDotLayout == 2 for now
                  lcldot = kernel["LocalDotLayout"]
                  iua = blockA / ((kernel["ThreadTileA"]/2) / lcldot)
                  iub = blockB / ((kernel["ThreadTileB"]/2) / lcldot)
                  rema = blockA % ((kernel["ThreadTileA"]/2) / lcldot)
                  remb = blockB % ((kernel["ThreadTileB"]/2) / lcldot)
                  #print "lcldot %u, blockA %u, blockB %u, rema %u, remb %u, ThreadTileA %u%s" % (lcldot, blockA, blockB, rema, remb, kernel["ThreadTileA"], self.endLine)
                  cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 0
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iua), rema*lcldot)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iub), remb*lcldot)
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] iua=%u iub=%u%s" % (cStr, aStr, bStr, cStr, cidx, iua, iub, self.endLine)
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u] iua=%u iub=%u%s" % (cStr, aStr, bStr, cStr, cidx, iua, iub, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iua), rema*lcldot+1)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iub), remb*lcldot)
                  cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iua), rema*lcldot)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iub), remb*lcldot+1)
                  cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iua), rema*lcldot+1)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iub), remb*lcldot+1)
                  cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
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
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                    cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                    cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                    cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
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

                  cStr = "v[%s+%u+%u*%u+%u]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
                  kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
                  """
                  D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                  D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                  C[0] = A[0]*B[0]+D[0]
                  C[1] = A[1]*B[1]+D[1]
                  """
            elif self.version == (9,0,6):
              if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                # we treat HighPrecisionAccumulate as expanded packed math
                b = blockB*2
                a = blockA*2
                if kernel["LocalDotLayout"] > 1:    # Only supports LocalDotLayout == 2 for now
                  lcldot = kernel["LocalDotLayout"]
                  iua = blockA / ((kernel["ThreadTileA"]/2) / lcldot)
                  iub = blockB / ((kernel["ThreadTileB"]/2) / lcldot)
                  rema = blockA % ((kernel["ThreadTileA"]/2) / lcldot)
                  remb = blockB % ((kernel["ThreadTileB"]/2) / lcldot)
                  #print "lcldot %u, blockA %u, blockB %u, rema %u, remb %u, ThreadTileA %u%s" % (lcldot, blockA, blockB, rema, remb, kernel["ThreadTileA"], self.endLine)
                  cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 0
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iua), rema*lcldot)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iub), remb*lcldot)
                  kStr += "v_dot2_f32_f16 %s, %s, %s, %s //ValuC[%u] iua=%u iub=%u%s" % (cStr, aStr, bStr, cStr, cidx, iua, iub, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iua), rema*lcldot+1)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iub), remb*lcldot)
                  cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                  kStr += "v_dot2_f32_f16 %s, %s, %s, %s //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iua), rema*lcldot)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iub), remb*lcldot+1)
                  cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
                  kStr += "v_dot2_f32_f16 %s, %s, %s, %s //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                  cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA_X%u_I%u"%(m,iua), rema*lcldot+1)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB_X%u_I%u"%(m,iub), remb*lcldot+1)
                  cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
                  kStr += "v_dot2_f32_f16 %s, %s, %s, %s //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
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
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                    cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                    kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[1,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                    cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
                    kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                    cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
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

                  cStr = "v[%s+%u+%u*%u+%u]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]/2)
                  kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
                  """
                  D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                  D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                  C[0] = A[0]*B[0]+D[0]
                  C[1] = A[1]*B[1]+D[1]
                  """
            else:
              printExit("Half-precision not supported for arch=%u" % self.version )


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
              if macIdx == kernel["PerformanceWaitLocation"]:
                  kStr += "s_waitcnt lgkmcnt(%u) // extra wait for performance%s" \
                      % (kernel["PerformanceWaitCount"], self.endLine)
              if macIdx == kernel["PerformanceSyncLocation"]:
                  kStr += "s_barrier // extra barrier for performance%s" \
                      % (self.endLine)
              macIdx += 1

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

      # other precision
      else:
        printExit("Assembly doesn't support %s" % kernel["ProblemType"]["DataType"])

      kStr += ".endm%s" % self.endLine


    return kStr

  ##############################################################################
  # Function Signature
  ##############################################################################
  def functionSignature(self, kernel ):
    kStr = ""

    # begin kernel descriptor
    kStr += ".hsa_code_object_version 2,0%s" % self.endLine
    kStr += ".hsa_code_object_isa %u, %u, %u, \"AMD\", \"AMDGPU\" %s" \
        % (self.version[0], self.version[1], self.version[2], self.endLine)
    kStr += ".text%s" % self.endLine
    kStr += ".p2align 8%s" % self.endLine
    kStr += ".amdgpu_hsa_kernel %s%s" % (self.kernelName, self.endLine)
    kStr += "%s:%s" % (self.kernelName, self.endLine)
    kStr += ".amd_kernel_code_t%s" % self.endLine
    kStr += "  is_ptr64 = 1%s" % self.endLine
    kStr += "  enable_sgpr_kernarg_segment_ptr = 1%s" % self.endLine

    # kern arg size
    kernArgReg = 0
    kernArgReg += 3*self.rpga
    kernArgReg += max(1,int(self.bpeAB/4)) # alpha
    if kernel["ProblemType"]["UseBeta"]:
      kernArgReg += max(1,int(self.bpeCexternal/4)) # beta
    kernArgReg += 3 # offsets
    kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
    kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsA"]) # strides
    kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsB"]) # strides
    if not kernel["ProblemType"]["UseInitialStrides"]:
      kernArgReg -= 3 # strides
    kernArgReg += kernel["ProblemType"]["NumIndicesSummation"]
    kernArgReg += kernel["ProblemType"]["NumIndicesC"]
    if globalParameters["DebugKernel"]:
      kernArgReg += self.rpga # debug buffer
    kernArgBytes = kernArgReg * 4 # bytes/reg
    kStr += "  kernarg_segment_byte_size = %u // bytes of kern args%s" \
        % (kernArgBytes, self.endLine)

    # register allocation
    totalVgprs = self.vgprPool.size()
    if self.vgprPool.size() > self.maxVgprs:
      self.overflowedResources = True
    if self.overflowedResources:
      totalVgprs = 1
    #totalSgprs = self.sgprPool.size()
    kStr += "  workitem_vgpr_count = %u // vgprs%s" \
        % (totalVgprs, self.endLine)
    kStr += "  wavefront_sgpr_count = %u // sgprs%s" \
        % (self.totalSgprs, self.endLine)
    kStr += "  compute_pgm_rsrc1_vgprs = %u // floor((%u-1)/4)%s" \
        % ( (totalVgprs-1)/4, totalVgprs, self.endLine)
    kStr += "  compute_pgm_rsrc1_sgprs = %u // floor((%u-1)/8)%s" \
        % ( 1+(self.totalSgprs-1)/8, self.totalSgprs, self.endLine)

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
#jgolds which bpe should we use? assuming A
    kStr += "  workgroup_group_segment_byte_size = %u // lds bytes%s" \
        % ( kernel["LdsNumElements"] * self.bpeAB, self.endLine )

    # other
    kStr += "  compute_pgm_rsrc2_user_sgpr = 2 // vcc%s" % self.endLine
    kStr += "  kernarg_segment_alignment = 4%s" % self.endLine
    kStr += "  group_segment_alignment = 4%s" % self.endLine
    kStr += "  private_segment_alignment = 4%s" % self.endLine
    kStr += ".end_amd_kernel_code_t%s" % self.endLine

    kStr += self.comment3("Optimizations and Config:")
    kStr += self.comment1("ThreadTile=%u x %u" % (kernel["ThreadTile0"], kernel["ThreadTile1"]))
    kStr += self.comment1("VectorWidth=%u" % (kernel["VectorWidth"]))
    kStr += self.comment1("GlobalLoadVectorWidthA=%u, GlobalLoadVectorWidthB=%u" % (kernel["GlobalLoadVectorWidthA"], kernel["GlobalLoadVectorWidthB"]))
    kStr += self.comment1("DirectToLdsA=%s" % kernel["DirectToLdsA"])
    kStr += self.comment1("DirectToLdsB=%s" % kernel["DirectToLdsB"])
    kStr += self.comment1("PreciseBoundsCheck=%s" % kernel["PreciseBoundsCheck"])
    kStr += self.comment1("UseSgprForGRO=%s" % kernel["UseSgprForGRO"])


    kStr += self.comment3("ASM syntax bug workarounds")
    kStr += ".macro _v_add_co_u32 dst, cc, src0, src1, dpp=" + self.endLine
    if self.AsmBugs["ExplicitCO"]:
        kStr += "   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += "   v_add_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += ".macro _v_sub_co_u32 dst, cc, src0, src1, dpp=" + self.endLine
    if self.AsmBugs["ExplicitCO"]:
        kStr += "   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    else:
        kStr += "   v_sub_u32 \dst, \cc, \src0, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    kStr += ".macro _v_addc_co_u32 dst, ccOut, src0, ccIn, src1, dpp=" + self.endLine
    if self.AsmBugs["ExplicitCO"]:
        kStr += "   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp" + self.endLine
    else:
        kStr += "   v_addc_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp" + self.endLine
    kStr += ".endm" + self.endLine

    # Use combined add+shift, where available:
    kStr += ".macro _v_add_lshl_u32 dst, src0, src1, shiftCnt" + self.endLine
    if globalParameters["AsmCaps"][self.version]["HasAddLshl"]:
      kStr += "    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt" + self.endLine
    else:
      if self.AsmBugs["ExplicitCO"]:
        kStr += "    v_add_co_u32 \dst, vcc, \src0, \src1" + self.endLine
      else:
        kStr += "    v_add_u32 \dst, vcc, \src0, \src1" + self.endLine
      kStr += "    v_lshlrev_b32 \dst, \shiftCnt, \dst" + self.endLine
    kStr += ".endm" + self.endLine



    ########################################
    # VGPR Macros
    ########################################
    kStr += self.comment3("VGPR Assignments")
    kStr += self.macroRegister("vgprValuC", self.startVgprValuC)

    kStr += self.comment1("ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx")
    ri = 0
    for bi in range(0,kernel["PrefetchLocalRead"]+1): # buffer indicies
      for iui in range(0, kernel["InnerUnroll"]):
        kStr += self.macroRegister("vgprValuA_X%u_I%u"%(bi,iui), self.startVgprValuA+ri)
        ri += self.numVgprValuAPerBlock
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
        kStr += self.macroRegister("vgprG2LA", self.startVgprG2LA)

    ri = 0
    for bi in range(0,kernel["PrefetchLocalRead"]+1): # buffer indicies
      for iui in range(0, kernel["InnerUnroll"]):
        kStr += self.macroRegister("vgprValuB_X%u_I%u"%(bi,iui), self.startVgprValuB+ri)
        ri += self.numVgprValuBPerBlock
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
        kStr += self.macroRegister("vgprG2LB", self.startVgprG2LB)
    kStr += self.macroRegister("vgprLocalReadAddrA", \
        self.startVgprLocalReadAddressesA)
    kStr += self.macroRegister("vgprLocalReadAddrB", \
        self.startVgprLocalReadAddressesB)
    if not kernel["LocalWriteUseSgprA"]:
      kStr += self.macroRegister("vgprLocalWriteAddrA", \
          self.startVgprLocalWriteAddressesA)
    if not kernel["LocalWriteUseSgprB"]:
      kStr += self.macroRegister("vgprLocalWriteAddrB", \
          self.startVgprLocalWriteAddressesB)
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
    if self.globalReadIncsUseVgpr:
      kStr += self.macroRegister("vgprGlobalReadIncsA", \
          self.startVgprGlobalReadIncsA)
      kStr += self.macroRegister("vgprGlobalReadIncsB", \
          self.startVgprGlobalReadIncsB)

    if globalParameters["DebugKernel"]:
      kStr += self.macroRegister("vgprAddressDbg", \
          self.startVgprAddressDbg)
    kStr += self.macroRegister("vgprSerial", \
        self.startVgprSerial)
    #kStr += self.comment1("Occu: %u waves/simd" % self.numWavesPerSimd )
    kStr += self.comment1("max VGPR=%u"%self.vgprPool.size())


    ########################################
    # SGPR Macros
    ########################################
    kStr += self.comment3("SGPR Assignments")


    # Emit declarations for all sgprs allocated with defineSgpr
    # in the order they were declared
    for skey in self.sgprs:
      kStr += self.macroRegister("sgpr"+skey, self.sgprs[skey])
    kStr += self.comment1("max SGPR=%u"%self.totalSgprs)


    if kernel["BufferLoad"] or kernel["BufferStore"]:
      if not kernel["PreciseBoundsCheck"]:
        kStr += self.comment3("2GB limit - set offsets to -1 to exceed this and clamp")
        kStr += self.macroRegister("BufferLimit", "0x80000000")
      kStr += self.comment3("Bits 127:96 of SRD.  Set DataFormat = 32 bit")
      kStr += self.macroRegister("Srd127_96",   "0x0020000")
      #TODO-64 : This is max 32-bit negative value, the tail loop
      # does incrementally step through the GRO and increment GRO
      # which are initialized with this value
      kStr += self.macroRegister("BufferOOB", "0x80000000")

    ########################################
    # Global Offsets
    ########################################
    # justOffset32 means we should only write the 32-bit offset
    # This is used in Buffer addressing modes.
    # Flat addressing modes expect the GLOBAL_OFFSET to initialize a full 64-bit address
    for (tensorChar, indices, justOffset32) in [ \
        ("C", range(0, kernel["ProblemType"]["NumIndicesC"]), kernel["BufferStore"]), \
        ("A", kernel["ProblemType"]["IndexAssignmentsA"], kernel["BufferLoad"]), \
        ("B", kernel["ProblemType"]["IndexAssignmentsB"], kernel["BufferLoad"]) ]:
      kStr += self.comment("Global Offset %s"%tensorChar)
      numDim = len(indices)
      idxChars = []
      for i in indices:
        idxChars.append(self.indexChars[i])

      # macro declaration
      kStr += ".macro GLOBAL_OFFSET_%s vgprAddr"%tensorChar
      for i in range(0, numDim):
        # tile index or unroll vgpr
        if indices[i] == kernel["ProblemType"]["Index0"] \
            or indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          kStr += " vgprOffset%s" % idxChars[i]
        # other c index sgpr
        elif indices[i] < kernel["ProblemType"]["NumIndicesC"]:
          kStr += " sgprOffset%s" % idxChars[i]
        # other sum index
        else:
          pass # these offsets are zero
      kStr += " vgprTmp%s" % self.endLine

      ########################################
      # index 0
      # tile index or unroll vgpr
      if indices[0] == kernel["ProblemType"]["Index0"] \
          or indices[0] == kernel["ProblemType"]["Index1"] \
          or indices[0] == kernel["ProblemType"]["IndexUnroll"]:
        offset = "v[\\vgprOffset%s]" % idxChars[0]
      # other c index sgpr
      elif indices[0] < kernel["ProblemType"]["NumIndicesC"]:
        offset = "s[\\sgprOffset%s]" % idxChars[0]
      # other sum index
      else:
        offset = "hex(0)"

      # d1+
      for i in range(1, numDim):

        # tile index or unroll vgpr
        if indices[i] == kernel["ProblemType"]["Index0"] \
            or indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          # offset * stride
          kStr += inst("v_mul_lo_u32", \
              "v[\\vgprTmp+0]", \
              sgpr("Strides%s+%u"%(tensorChar,i-1)), \
              "v[\\vgprOffset%s]" % idxChars[i],  \
              "mul d%u lower"%i)
          if not justOffset32:
            kStr += inst("v_mul_hi_u32", \
                "v[\\vgprTmp+1]", \
                sgpr("Strides%s+%u"%(tensorChar,i-1)), \
                "v[\\vgprOffset%s]" % idxChars[i],  \
                "mul d%u upper"%i)
          needAdd = 1
        # other c index sgpr
        elif indices[i] < kernel["ProblemType"]["NumIndicesC"]:
          kStr += inst("v_mov_b32", \
              "v[\\vgprTmp+2]", \
              "s[\\sgprOffset%s]"%idxChars[i], \
              "sgprOffset -> vgprTmp+2")
          # offset * stride
          kStr += inst("v_mul_lo_u32", \
              "v[\\vgprTmp+0]", \
              sgpr("Strides%s+%u"%(tensorChar,i-1)), \
              "v[\\vgprTmp+2]",  \
              "mul d%u lower"%i)
          if not justOffset32:
            kStr += inst("v_mul_hi_u32", \
                "v[\\vgprTmp+1]", \
                sgpr("Strides%s+%u"%(tensorChar,i-1)), \
                "v[\\vgprTmp+2]",  \
                "mul d%u upper"%i)
          needAdd = 1
        # other sum index
        else:
          # don't even need to add b/c offset=zero
          needAdd = 0

        if needAdd:
          # addr += offset * stride (lo)
          kStr += inst("_v_add_co_u32", \
              "v[\\vgprAddr+0]", \
              "vcc", \
              "v[\\vgprTmp+0]", \
              offset, \
              "accumulate d%u lower"%i)
          # addr += offset * stride (hi)
          if not justOffset32:
            kStr += inst("_v_addc_co_u32", \
                "v[\\vgprAddr+1]", \
                "vcc", \
                "v[\\vgprTmp+1]",  \
                0, \
                "vcc", \
                "accumulate d%u upper"%i)
        else:
          kStr += inst("v_mov_b32", "v[\\vgprAddr+0]", offset, "d0 lower")
          if not justOffset32:
            kStr += inst("v_mov_b32", "v[\\vgprAddr+1]", hex(0), "d0 upper")

        # Change offset for subsequent dims (if needed)
        offset = "v[\\vgprAddr+0]"

      # addr *= bytes/element
#jgolds which bpe should we use? assuming A
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

    kStr += self.defineMACMacro(kernel, kernel["InnerUnroll"])
    if kernel["InnerUnroll"] > 1:
      kStr += self.defineMACMacro(kernel, 1) # define OneIter case

    # if overflowed vgpr pool, comment out the whole kernel body and let it fail gracefully
    if self.vgprPool.size() > self.maxVgprs or self.sgprPool.size() > self.maxSgprs:
      self.overflowedResources = True
    if self.overflowedResources:
      print ""
      printWarning("%s invalid: too many vgprs(%u) or sgprs(%u)" \
          % (self.kernelName, self.vgprPool.size(), self.sgprPool.size()) )
      kStr += "s_endpgm // too many vgprs\n"
      kStr += ".if 0\n"


    return kStr


  ##############################################################################
  # Function Beginning
  ##############################################################################
  def functionSignaturePrefix(self, kernel): return ""
  def functionSignatureSuffix(self, kernel): return ""
  def functionBegin(self, kernel): return ""
  def allocateResources(self, kernel):
    kStr = ""
    if self.do["NullKernel"]:
      kStr += inst("s_endpgm", "Skip the whole kernel")

    if self.do["PreLoop"]: 
      # set m0
#jgolds which bpe here? Using A for now
      kStr += inst("s_mov_b32", "m0", hex(kernel["LdsNumElements"] \
          * self.bpeAB), "LDS clamp at %u bytes" \
          %(kernel["LdsNumElements"] * self.bpeAB) )

      kStr += inst("v_mov_b32", vgpr("Serial"), vgpr(0), "thread serial id")


      ########################################
      # load kernel args
      kStr += self.comment("Load Kernel Args")
      kernArgOffset = 0
      if globalParameters["DebugKernel"]:
        kStr += inst("s_load_dword", sgpr("AddressDbg"), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug" )
        kernArgOffset += 1*4
        kStr += inst("s_load_dword", sgpr("AddressDbg+1"), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug + 1" )
        kernArgOffset += 1*4
      kStr += inst("s_load_dword", sgpr("AddressC"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr c" )
      kernArgOffset += 1*4
      kStr += inst("s_load_dword", sgpr("AddressC+1"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr c" )
      kernArgOffset += 1*4
      kStr += inst("s_load_dword", sgpr("AddressA"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr a" )
      kernArgOffset += 1*4
      kStr += inst("s_load_dword", sgpr("AddressA+1"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr a" )
      kernArgOffset += 1*4
      kStr += inst("s_load_dword", sgpr("AddressB"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr b" )
      kernArgOffset += 1*4
      kStr += inst("s_load_dword", sgpr("AddressB+1"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr b" )
      kernArgOffset += 1*4
      # for half precision or smaller, data is padded to fill up 32-bits
      if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isSingle():
        kStr += inst("s_load_dword", sgpr("Alpha"), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load alpha" )
      elif kernel["ProblemType"]["DataType"].isDouble():
        kStr += inst("s_load_dword", sgpr("Alpha+0"), \
            sgpr("KernArgAddress",2), hex(kernArgOffset+0), "load alpha" )
        kStr += inst("s_load_dword", sgpr("Alpha+1"), \
            sgpr("KernArgAddress",2), hex(kernArgOffset+4), "load alpha" )
      kernArgOffset += 1*max(4,self.bpeAB)
      if kernel["ProblemType"]["UseBeta"]:
        if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isSingle():
          kStr += inst("s_load_dword", sgpr("Beta"), \
              sgpr("KernArgAddress",2), hex(kernArgOffset), "load beta" )
        elif kernel["ProblemType"]["DataType"].isDouble():
          kStr += inst("s_load_dword", sgpr("Beta+0"), \
              sgpr("KernArgAddress",2), hex(kernArgOffset+0), "load beta" )
          kStr += inst("s_load_dword", sgpr("Beta+1"), \
              sgpr("KernArgAddress",2), hex(kernArgOffset+4), "load beta" )
        kernArgOffset += 1*max(4,self.bpeCexternal)
      kStr += inst("s_load_dword", sgpr("OffsetC"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset c" )
      kernArgOffset += 1*4
      kStr += inst("s_load_dword", sgpr("OffsetA"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset a" )
      kernArgOffset += 1*4
      kStr += inst("s_load_dword", sgpr("OffsetB"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset b" )
      kernArgOffset += 1*4
      for i in range(0, self.numSgprStridesC):
        kStr += inst("s_load_dword", sgpr("StridesC+%u"%i), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride c %u"%i )
        kernArgOffset += 1*4
      for i in range(0, self.numSgprStridesA):
        kStr += inst("s_load_dword", sgpr("StridesA+%u"%i), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride a %u"%i )
        kernArgOffset += 1*4
      for i in range(0, self.numSgprStridesB):
        kStr += inst("s_load_dword", sgpr("StridesB+%u"%i), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride b %u"%i )
        kernArgOffset += 1*4
      for i in range(0, self.numSgprSizesFree):
        kStr += inst("s_load_dword", sgpr("SizesFree+%u"%i), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load size free %u"%i )
        kernArgOffset += 1*4
      for i in range(0, self.numSgprSizesSum):
        kStr += inst("s_load_dword", sgpr("SizesSum+%u"%i), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load size free %u"%i )
        kernArgOffset += 1*4
      kStr += inst("s_waitcnt", "lgkmcnt(0)", \
          "wait for %u bytes of kern args" % kernArgOffset )
    else:
      kStr += ".if 0\n"

    ########################################
    # Apply User Offsets
    kStr += self.comment("User Offsets")
    kStr += inst("s_add_u32", sgpr("AddressC"), sgpr("OffsetC"), \
        sgpr("AddressC"), "addrC += offsetC" )
    kStr += inst("s_mov_b32", sgpr("OffsetC"), 0, "")
    kStr += inst("s_addc_u32", sgpr("AddressC"), sgpr("OffsetC"),\
        sgpr("AddressC"), "addrC += offsetC carry" )
    kStr += inst("s_add_u32", sgpr("AddressA"), sgpr("OffsetA"), \
        sgpr("AddressA"), "addrA += offsetA" )
    kStr += inst("s_mov_b32", sgpr("OffsetA"), 0, "")
    kStr += inst("s_addc_u32", sgpr("AddressA"), sgpr("OffsetA"),\
        sgpr("AddressA"), "addrA += offsetA carry" )
    kStr += inst("s_add_u32", sgpr("AddressB"), sgpr("OffsetB"), \
        sgpr("AddressB"), "addrB += offsetB" )
    kStr += inst("s_mov_b32", sgpr("OffsetB"), 0, "")
    kStr += inst("s_addc_u32", sgpr("AddressB"), sgpr("OffsetB"),\
        sgpr("AddressB"), "addrB += offsetB carry" )
    # now sgpr OffsetC,A,B are freed up for arithmetic
    #self.setStartTmpPool(self.sgprs["OffsetC"])

    ########################################
    # NumWorkGroups
    # nwg0
    size0 = self.vgprPool.checkOut(1)
    tmpVgpr = self.vgprPool.checkOut(2)
    nwg0 = self.vgprPool.checkOut(1)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += "// size0 = (size%s + MT%s - 1) / MT%s;%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
    kStr += inst("v_mov_b32", vgpr(size0), sgpr("SizesFree+0"), "")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile0"]-1), "")
    kStr += inst("_v_add_co_u32", vgpr(size0), "vcc", sgpr(tmpSgpr), vgpr(size0), \
        "%s = size0+MT0-1"%vgpr(size0))
    kStr += vectorStaticDivide(nwg0, size0, kernel["MacroTile0"], tmpVgpr, tmpSgpr)
    self.vgprPool.checkIn(size0)
    self.vgprPool.checkIn(tmpVgpr)
    kStr += inst("v_readfirstlane_b32", sgpr("NumWorkGroups0"), \
        vgpr(nwg0), "")
    self.vgprPool.checkIn(nwg0)

    # nwg1
    size1 = self.vgprPool.checkOut(1)
    tmpVgpr = self.vgprPool.checkOut(2)
    nwg1 = self.vgprPool.checkOut(1)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += "// size1 = (size%s + MT%s - 1) / MT%s;%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
    kStr += inst("v_mov_b32", vgpr(size1), sgpr("SizesFree+1"), "")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile1"]-1), "")
    kStr += inst("_v_add_co_u32", vgpr(size1), "vcc", sgpr(tmpSgpr), vgpr(size1), \
        "%s = size1+MT1-1"%vgpr(size1))
    kStr += vectorStaticDivide(nwg1, size1, kernel["MacroTile1"], tmpVgpr, tmpSgpr)
    self.vgprPool.checkIn(size1)
    self.vgprPool.checkIn(tmpVgpr)
    kStr += inst("v_readfirstlane_b32", sgpr("NumWorkGroups1"), \
        vgpr(nwg1), "")
    self.vgprPool.checkIn(nwg1)


    ########################################
    # Debug Buffer
    if globalParameters["DebugKernel"]:
      kStr += self.comment("Debug Buffer")

      # nwg0 FIXME use nwg0 from above
      nwg0 = self.vgprPool.checkOut(1)
      tmpVgpr = self.vgprPool.checkOut(2)
      tmpSgpr = self.getTmpSgpr(1)
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

    return kStr


  ##############################################################################
  # Global Read Addresses: WorkGroup
  ##############################################################################
  def graWorkGroup(self, kernel):
    kStr = ""

    if kernel["GlobalSplitU"] > 1:
      if kernel["GlobalSplitUWorkGroupMappingRoundRobin"]:
        # gsuSumIdx = wg1 / nwg1
        # wg1       = wg1 % nwg1

        # nwg1
        nwg1 = self.vgprPool.checkOut(1)
        tmpVgpr = self.vgprPool.checkOut(2)
        quotient = self.vgprPool.checkOut(1)
        tmpSgpr = self.getTmpSgpr(1)
        kStr += "// nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
            % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
        kStr += inst("v_mov_b32", vgpr(nwg1), sgpr("SizesFree+1"), "")
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile1"]-1), "")
        kStr += inst("_v_add_co_u32", vgpr(nwg1), "vcc", sgpr(tmpSgpr), vgpr(nwg1), \
            "%s = size1+MT1-1"%vgpr(nwg1))
        kStr += vectorStaticDivide(quotient, nwg1, kernel["MacroTile1"], tmpVgpr, tmpSgpr)
        self.vgprPool.checkIn(nwg1)
        nwg1 = quotient

        # wg1
        wg1 = self.vgprPool.checkOut(1)
        kStr += inst("v_mov_b32", vgpr(wg1), sgpr("WorkGroup1"), "wg1")

        # gsuSumIdx = wg1 / nwg1
        # wg1       = wg1 % nwg1
        quotient = self.vgprPool.checkOut(1)
        remainder = self.vgprPool.checkOut(1)
        tmpVgpr1 = self.vgprPool.checkOut(1)
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
        if False:
          quotient = self.vgprPool.checkOut(1)
          remainder = self.vgprPool.checkOut(1)
          dividend = "Serial"
          tmpVgpr = self.vgprPool.checkOut(3)
          tmpSgpr = self.getTmpSgpr(1)
          divisor = 3
          kStr += vectorStaticDivideAndRemainder(quotient, remainder, dividend, divisor, tmpVgpr, tmpSgpr, True )
          kStr += "s_endpgm\n"


        # gsuSumIdx = wg1 % GSU
        # wg1       = wg1 / GSU
        tmpSgpr = self.getTmpSgpr(3) # needs 3
        divisor = tmpSgpr+2
        kStr += inst("s_mov_b32", sgpr(divisor), sgpr("WorkGroup1"), \
            "copying for divisor")

        #tmp = self.vgprPool.checkOut(1)

        #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("WorkGroup1"), "wg1")
        #kStr += dump(vgpr(tmp)) # numerator

        kStr += scalarStaticDivideAndRemainder("WorkGroup1", "GSUSumIdx", \
            divisor, kernel["GlobalSplitU"], tmpSgpr, True)

        #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("WorkGroup1"), "wg1")
        #kStr += dump(vgpr(tmp)) # quotient
        #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("GSUSumIdx"), "gsusumidx")
        #kStr += dump(vgpr(tmp)) # remainder
        #self.vgprPool.checkIn(tmp)
        #kStr += "s_endpgm\n"

    ########################################
    # Blocked rows or columns
    if kernel["WorkGroupMappingType"] == "B" and abs(kernel["WorkGroupMapping"]) > 1:
      tmpVgpr = self.vgprPool.checkOut(2)
      # nwg0
      nwg0 = self.vgprPool.checkOut(1)
      tmpSgpr = self.getTmpSgpr(2)
      kStr += "// nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
      kStr += inst("v_mov_b32", vgpr(nwg0), sgpr("SizesFree+0"), "")
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile0"]-1), "")
      kStr += inst("_v_add_co_u32", vgpr(nwg0), "vcc", sgpr(tmpSgpr), vgpr(nwg0), \
          "%s = size0+MT0-1"%vgpr(nwg0))
      kStr += vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpVgpr, tmpSgpr)
      #kStr += dump(vgpr(nwg0))

      # nwg1
      nwg1 = self.vgprPool.checkOut(1)
      kStr += "// nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
      kStr += inst("v_mov_b32", vgpr(nwg1), sgpr("SizesFree+1"), "")
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile1"]-1), "")
      kStr += inst("_v_add_co_u32", vgpr(nwg1), "vcc", sgpr(tmpSgpr), vgpr(nwg1), \
          "%s = size1+MT1-1"%vgpr(nwg1))
      kStr += vectorStaticDivide(nwg1, nwg1, kernel["MacroTile1"], tmpVgpr, tmpSgpr)
      #kStr += dump(vgpr(nwg1))

      # blockId and serial within block
      blockId = self.vgprPool.checkOut(1)
      wgSerial = self.vgprPool.checkOut(1)
      wg1 = self.vgprPool.checkOut(1)
      kStr += inst("v_mov_b32", vgpr(wg1), sgpr("WorkGroup1"), "wg1")

      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr("WorkGroup0"), "wg0")
      #kStr += dump(vgpr(tmpVgpr))
      #kStr += dump(vgpr(wg1))

      kStr += vectorStaticDivideAndRemainder(blockId, wgSerial, wg1, \
          abs(kernel["WorkGroupMapping"]), tmpVgpr, tmpSgpr)
      kStr += inst("v_mul_lo_u32", vgpr(wgSerial), vgpr(wgSerial), \
          vgpr(nwg0), "(wg1 % WGM)*nwg0")
      self.vgprPool.checkIn(nwg0)
      kStr += inst("_v_add_co_u32", vgpr(wgSerial), "vcc", sgpr("WorkGroup0"), vgpr(wgSerial), \
          "wgSerial = wg0 + (wg1 % WGM)*nwg0")
      #kStr += "s_endpgm\n"
      #return kStr

      # num full blocks
      numFullBlocks = self.vgprPool.checkOut(1)
      kStr += "// numFullBlocks = (nwg1) / WGM%s" % (self.endLine)
      blockRemainder = self.vgprPool.checkOut(1)
      kStr += vectorStaticDivideAndRemainder(numFullBlocks, blockRemainder, \
          nwg1, abs(kernel["WorkGroupMapping"]), tmpVgpr, tmpSgpr)
      self.vgprPool.checkIn(nwg1)

      #kStr += dump(vgpr(blockId))
      #kStr += dump(vgpr(numFullBlocks))
      #kStr += dump(vgpr(blockRemainder))
      # lastBlockWidth = blockRemainder

      # my block's width
      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), hex(111), "")
      #kStr += dump(vgpr(tmpVgpr))
      kStr += inst("v_cmp_lt_u32", sgpr(tmpSgpr,2), vgpr(blockId), vgpr(numFullBlocks), "blockId < numFullBlocks" )
      self.vgprPool.checkIn(numFullBlocks)
      blockWidth = self.vgprPool.checkOut(1)
      kStr += inst("v_cndmask_b32", vgpr(blockWidth), vgpr(blockRemainder), hex(abs(kernel["WorkGroupMapping"])), sgpr(tmpSgpr,2), "blockWidth = (blockId < numFullBlocks) ? WGM : remainder" )
      self.vgprPool.checkIn(blockRemainder)
      #kStr += dump(vgpr(blockWidth))

      # dynamic divide and remainder
      # wg0 = wgSerialInBlock / myBlockWidth
      # wg1 = wgSerialInBlock % myBlockWidth + blockId*WGM
      wg0 = self.vgprPool.checkOut(1)
      kStr += "DYNAMIC_VECTOR_DIVIDE %s %s %s %s %s %s %s%s" % ( wg0, wg1, wgSerial, blockWidth, tmpVgpr, tmpVgpr+1, tmpSgpr, self.endLine )
      kStr += inst("v_mul_lo_u32", vgpr(blockId), vgpr(blockId), \
          abs(kernel["WorkGroupMapping"]), "blockId * WGM")
      kStr += inst("_v_add_co_u32", vgpr(wg1), "vcc", vgpr(wg1), \
          vgpr(blockId), "wg1 += blockId * WGM")

      # move wg0,1 in vgprs into sgprs
      kStr += inst("v_readfirstlane_b32", sgpr("WorkGroup0"), vgpr(wg0), "")
      kStr += inst("v_readfirstlane_b32", sgpr("WorkGroup1"), vgpr(wg1), "")

      # checkin scratch registers
      self.vgprPool.checkIn(wg0)
      self.vgprPool.checkIn(wg1)
      self.vgprPool.checkIn(blockWidth)
      self.vgprPool.checkIn(wgSerial)
      self.vgprPool.checkIn(blockId)

      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr("WorkGroup0"), "")
      #kStr += dump(vgpr(tmpVgpr))
      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr("WorkGroup1"), "")
      #kStr += dump(vgpr(tmpVgpr))
      self.vgprPool.checkIn(tmpVgpr)
      #kStr += "s_endpgm\n"

    return kStr

  ##############################################################################
  # Global Read Addresses: Subgroup
  ##############################################################################
  def graSubgroup(self, kernel):
    return self.comment1("  not needed until local read addresses")


  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  # stores to v1,2
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
      rReg = self.vgprPool.checkOut(1) # gro-tile = serial%divisor
      qReg = self.vgprPool.checkOut(1) # gro-unroll = serial/divisor
      tReg = rReg
      uReg = qReg
      tOpStr = "%"
      uOpStr = "/"
    else:
      qReg = self.vgprPool.checkOut(1) # gro-tile = serial/divisor
      rReg = self.vgprPool.checkOut(1) # gro-unroll = serial%divisor
      tReg = qReg
      uReg = rReg
      tOpStr = "/"
      uOpStr = "%"
    tReg2 = self.vgprPool.checkOut(1)
    kStr += self.comment1("%s = gro%s-tile = serial%s%s + (wg%s*MT%s)" \
        % (vgpr(tReg2), tP["tensorChar"], tOpStr, divisorName, tP["tensorChar"], tP["tensorChar"]) )
    kStr += self.comment1("%s = gro%s-unroll = serial%s%s" \
        % (vgpr(uReg), tP["tensorChar"], uOpStr, divisorName) )
    dividendReg = "Serial" # local serial
    tmpVgpr = self.vgprPool.checkOut(2)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)

    if tP["glvw"] > 1:
      if tP["grcv"] == tP["tlu"]:
        kStr += self.comment1("gro-tile *= glvw")
        kStr += staticMultiply(vgpr(tReg), vgpr(tReg), tP["glvw"], sgpr(tmpSgpr))
      else:
        kStr += self.comment1("gro-unroll *= glvw")
        kStr += staticMultiply(vgpr(uReg), vgpr(uReg), tP["glvw"], sgpr(tmpSgpr))
    kStr += staticMultiply(vgpr(tmpVgpr), sgpr(tP["wg"]), kernel[tP["mt"]])
    kStr += inst("_v_add_co_u32", vgpr(tReg2), "vcc", vgpr(tmpVgpr), \
        vgpr(tReg), "gro%s-tile = serial%s%s*VW + (wg%s*MT%s)" \
        % (tP["tensorChar"], tOpStr, divisorName, tP["tensorChar"], tP["tensorChar"]) )

    if kernel["GlobalSplitU"] > 1:
      uReg2 = self.vgprPool.checkOut(1)
      kStr += inst("v_mov_b32", vgpr(uReg2), vgpr(uReg), "copy for GlobalSplitU")
      tP["gpr"]["uReg2"] = uReg2
    tP["gpr"]["lwoT"] = tReg
    tP["gpr"]["tReg"] = tReg2
    tP["gpr"]["uReg"] = uReg
    self.vgprPool.checkIn(tmpVgpr)
    #kStr += dump(vgpr(tReg2))
    #kStr += dump(vgpr(uReg))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Assignment
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    kStr = ""
    if kernel["GlobalSplitU"] > 1:
      gsuOffset = self.vgprPool.checkOut(1)
      kStr += inst("v_mov_b32", vgpr(gsuOffset), sgpr("GSUSumIdx"), "=gsuSumIdx")
      if kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
        # graUnrollAssignment += gsuSumIdx*DepthU
        tmpSgpr = self.getTmpSgpr(1)
        kStr += staticMultiply(vgpr(gsuOffset), vgpr(gsuOffset), kernel["DepthU"], sgpr(tmpSgpr))
      else:
        # graUnrollAssignment += gsuSumIdx*(SizeU/GSU)
        sizeU = self.vgprPool.checkOut(1)
        kStr += inst("v_mov_b32", vgpr(sizeU), sgpr("SizesSum+0"), \
            "=Size%s"%self.unrollChar)
        quotient = self.vgprPool.checkOut(1)
        dummy = self.vgprPool.checkOut(1)
        tmpVgpr = self.vgprPool.checkOut(2)
        tmpSgpr = self.getTmpSgpr(1)
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
      kStr += ".set globalReadOffsetA%s 0%s" \
          % (self.indexChars[index], self.endLine)
      kStr += ".set globalReadOffsetB%s 0%s" \
          % (self.indexChars[index], self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  def graTileOffsets(self, kernel, tP):
    kStr = ""
    numTileOffsets = tP["nrt"]
    if tP["rtc"]:
      numTileOffsets *= tP["glvw"]
    tP["vgprTileOffsets"] = self.vgprPool.checkOut(numTileOffsets)
    v = tP["vgprTileOffsets"]
    strideIdx = tP["lsc"] if tP["tlu"] else tP["lsp"]
    stride = kernel[strideIdx]
    if tP["rtc"]:
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
    self.vgprPool.checkIn(tP["gpr"]["tReg"])
    return kStr


  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  def graUnrollOffsets(self, kernel, tP):
    kStr = ""
    numUnrollOffsets = tP["nru"]
    if tP["ruc"]:
      numUnrollOffsets *= tP["glvw"]
    tP["gpr"]["unrollOffsets"] = self.vgprPool.checkOut(numUnrollOffsets)
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
  ##############################################################################
  def graShift(self, kernel, tP):
    if kernel["PreciseBoundsCheck"]: return ""

    kStr = ""
    # edge value
    margin = tP["glvw"] if tP["rtv"] else 1
    edge = self.vgprPool.checkOut(1)


    if kernel["BufferLoad"] and kernel["PreciseBoundsCheck"]:
      # Go to the edge. we can rely on preciseboundscheck to keep things inline
      # Results in more loads of 0 which is better for power and debug
      kStr += inst("v_mov_b32", vgpr(edge), sgpr("SizesFree+%u"%tP["idx"]), \
                "edge = Size%s"%(tP["tileChar"]) )
    else:
      tmpSgpr = self.getTmpSgpr(1)
      kStr += inst("s_add_u32", sgpr(tmpSgpr), hex(-margin), sgpr("SizesFree+%u"%tP["idx"]), \
          "edge = Size%s-%u"%(tP["tileChar"], margin) )
      kStr += inst("v_mov_b32", vgpr(edge), sgpr(tmpSgpr), \
          "edge = Size%s-%u"%(tP["tileChar"], margin) )

    # shift offsets
    v = tP["vgprTileOffsets"]
    tmpSgpr = self.getTmpSgpr(2)
    for l in range(0, tP["nrt"]):
      # compare
      #kStr += dump(vgpr(v+l))
      kStr += inst("v_cmp_lt_u32", sgpr(tmpSgpr,2), vgpr(v+l), vgpr(edge), "offset < edge" )
      # shift
      kStr += inst("v_cndmask_b32", vgpr(v+l), vgpr(edge), vgpr(v+l), sgpr(tmpSgpr,2), "offset = (offset < edge) ? offset : edge" )
      #kStr += dump(vgpr(v+l))
    self.vgprPool.checkIn(edge)
    #if tP["isB"]:
    #  kStr += "s_endpgm\n"


    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  def graFinalOffsets(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
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
    tileOffsets = tP["vgprTileOffsets"]
    unrollOffsets = tP["gpr"]["unrollOffsets"]
    tmp = self.vgprPool.checkOut(3)
    graIdx = 0
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):
            # vgpr assignments
            if tP["tlu"]:
              vgprTile   = tileOffsets   + para*tVW + sPara*tVS
              vgprUnroll = unrollOffsets + perp*uVW + sPerp*uVS
            else:
              vgprTile   = tileOffsets   + perp*tVW + sPara*tVS
              vgprUnroll = unrollOffsets + para*uVW + sPerp*uVS

            if graIdx==0 or not kernel["UseSgprForGRO"]:
              # emit global offset macro
              if kernel["BufferLoad"]:
                  kStr += "GLOBAL_OFFSET_%s vgprGlobalReadOffset%s+%u"%(tP["tensorChar"], tP["tensorChar"], graIdx)
              else:
                kStr += "GLOBAL_OFFSET_%s vgprGlobalReadAddr%s+%u"%(tP["tensorChar"], tP["tensorChar"], graIdx)
              for i in tP["ia"]:
                if i < kernel["ProblemType"]["NumIndicesC"]:
                  if i == tP["tileIdx"]:
                    kStr += ", %2u" % vgprTile
                  else: # just a group index
                    kStr += ", sgprWorkGroup%u"%i
                else: # summation index
                  if i == kernel["ProblemType"]["IndexUnroll"]:
                    kStr += ", %2u" % vgprUnroll
                  else:
                    kStr += "globalReadOffset%s%s" % (tP["tensorChar"], self.indexChars[i] )
              kStr += ", %u // gRO%s_%u_%u_%u_%u%s" % (tmp, tP["tensorChar"], \
                  para, sPara, perp, sPerp, self.endLine)

              if kernel["BufferLoad"] and kernel["FractionalLoad"]:
                tmpSgpr = self.getTmpSgpr(2)
                lastValidThread = kernel[tP["lsc"]]*kernel[tP["lsp"]]/tP["glvw"]
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

            if graIdx >0 and (kernel["UseSgprForGRO"] or self.checkGRO):
              if kernel["UseSgprForGRO"]:
                scalarGro = "ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1)
              else:
                scalarGro = self.getTmpSgpr(1)

              if tP["tlu"]:
                tileStride   = kernel[tP["lsc"]] * (para*tVW + sPara*tVS)
                unrollStride = kernel[tP["lsp"]] * (perp*uVW + sPerp*uVS)
                kStr += inst("s_mul_i32", sgpr(scalarGro), sgpr("Strides%s"%tc), unrollStride, \
                             "compute offset diff (scaled unrollDim)")
                if tileStride:
                  kStr += inst("s_add_u32", sgpr(scalarGro), sgpr(scalarGro), tileStride, \
                             "compute offset diff (tileDim)")
              else:
                tileStride   = kernel[tP["lsp"]] * (perp*tVW + sPara*tVS)
                unrollStride = kernel[tP["lsc"]] * (para*uVW + sPerp*uVS)
                kStr += inst("s_mul_i32", sgpr(scalarGro), sgpr("Strides%s"%tc), tileStride, \
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

                print tc, "tileStride=", tileStride, "unrollStride=", unrollStride, \
                      "Strides%s="%tc

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
    self.vgprPool.checkIn(tileOffsets)
    self.vgprPool.checkIn(unrollOffsets)
    self.vgprPool.checkIn(tmp)

    if kernel["FractionalLoad"] and kernel["fractionalPerpOverhang%s"%tc]:
      overhang = kernel["fractionalPerpOverhang%s"%tc]
      validWI = overhang*kernel[tP["lsc"]]/tP["glvw"]
      kStr += inst("s_mov_b32", sgpr("PerpOverhangVcc%s"%tc), validWI, \
          "overhang=%u, validWI=%u" % (overhang, validWI))
      kStr += inst("v_cmp_lt_u32", \
          sgpr("PerpOverhangVcc%s"%tc,2),
          vgpr("Serial"), \
          sgpr("PerpOverhangVcc%s"%tc), \
          "fractional-overhang: some wi write to harmless LDS location")


    return kStr

  ##############################################################################
  # Global Read Addresses: Apply User Offsets
  ##############################################################################
  def graApplyUserOffsets(self, kernel):
    kStr = ""
    kStr += self.comment1("moved earlier")
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
      dim = len(tP["ia"])-1 # dim
      strideIdx = dim-1 # largest stride
      sizeIdx = tP["ia"][dim]

      sizeIdxIsSum = sizeIdx in kernel["ProblemType"]["IndicesSummation"]
      if sizeIdxIsSum:
        sizeIdx -= kernel["ProblemType"]["NumIndicesC"]

      # Buffer-load uses one base read pointer stored in the SRD - set it here:
      kStr += inst("s_mov_b32", sgpr("Srd%s+0"%tc), sgpr("Address%s+0"%tc), "init SRD base address (lower)" )
      kStr += inst("s_mov_b32", sgpr("Srd%s+1"%tc), sgpr("Address%s+1"%tc), "init SRD base address (upper) + other fields" )
      if kernel["PreciseBoundsCheck"]:
        kStr += inst("s_mul_i32", \
            sgpr("Srd%s+2"%tc), \
            sgpr("Sizes%s+%u"%("Sum" if sizeIdxIsSum else "Free", sizeIdx)),  \
            sgpr("Strides%s+%u"%(tc,strideIdx)), \
            "set limit to bottom-right corner of array")
        kStr += inst("s_lshl_b32",
            sgpr("Srd%s+2"%tc), \
            sgpr("Srd%s+2"%tc), \
            hex(log2(tP["bpe"])), \
            "Size in bytes") #TODO-64B
      else:
        kStr += inst("s_mov_b32", sgpr("Srd%s+2"%tc), "BufferLimit", "")
      kStr += inst("s_mov_b32", sgpr("Srd%s+3"%tc), "Srd127_96", "Set bits 127_96 in SRD")
    else:
      tmp = self.vgprPool.checkOut(2)
      kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("Address%s+0"%tP["tensorChar"]), "" )
      kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("Address%s+1"%tP["tensorChar"]), "" )
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):

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
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    kStr = ""
    # depthU
    depthU = kernel["DepthU"]
    if kernel["GlobalSplitU"] > 1 \
        and kernel["GlobalSplitUSummationAssignmentRoundRobin"]:
      depthU *= kernel["GlobalSplitU"]

    if loopIdx==kernel["ProblemType"]["NumIndicesSummation"]-1:
      if tP["tlu"]:
        if self.globalReadIncsUseVgpr:
          tmpSgpr = self.getTmpSgpr(1)
#jgolds which bpe here? assuming tP
          kStr += inst("s_mul_i32", sgpr(tmpSgpr+0), \
              hex(depthU*tP["bpe"]), sgpr("Strides%s"%tP["tensorChar"]), \
              "incr = stride*%u*bytes"%depthU )
          """
          kStr += inst("s_addc_u32", \
              sgpr(tmpSgpr+1), \
              hex(0), \
              hex(0), \
              "(carry)")
          """
          kStr += inst("s_mov_b32", \
              sgpr(tmpSgpr+1), \
              hex(0), \
              "(carry)")
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              sgpr(tmpSgpr+0), \
              "" )
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              sgpr(tmpSgpr+1), \
              "" )
        else: # not globalReadIncsUseVgpr, ie use SGPR
#jgolds which bpe here? assuming tP
          kStr += inst("s_mul_i32", sgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(depthU*tP["bpe"]), sgpr("Strides%s"%tP["tensorChar"]), \
              "incr = stride*%u*bytes"%depthU )
          """
          kStr += inst("s_addc_u32", \
              sgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), \
              hex(0), \
              "(carry)")
          """
          kStr += inst("s_mov_b32", \
              sgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), \
              "(carry)")

      else: # transposed
#jgolds which bpe here? assuming tP
        if self.globalReadIncsUseVgpr:
          kStr += inst("v_mov_b32", vgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(depthU*tP["bpe"]), \
              "incr = %u*bytes"%depthU )
          kStr += inst("v_mov_b32", vgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), "incr = %u*bytes (upper)"%depthU )
        else:
          kStr += inst("s_mov_b32", sgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(depthU*tP["bpe"]), \
              "incr = %u*bytes"%depthU )
          kStr += inst("s_mov_b32", sgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), "incr = %u*bytes (upper)"%depthU )
    else:
      printExit("NumIndicesSummation=%u not yet supported in assembly" \
          % kernel["ProblemType"]["NumIndicesSummation"] )

    #kStr += dump(vgpr("GlobalReadIncs%s"%tP["tensorChar"]))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  def lwaTileAssignment(self, kernel, tP):
    return self.comment1("lwaTile%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["lwoT"])))

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    return self.comment1("lwaUnroll%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"])))

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    #"lwFOA = lwA%s + lwA%s*MT%s" \
    #    % (tP["tileChar"], self.unrollChar, tP["tileChar"])
    uReg = tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"]
    if kernel["LocalWriteUseSgpr%s"%tc]:
      destVgpr = self.vgprPool.checkOut(1)
    else:
      destVgpr = "LocalWriteAddr%s"%tc

    dotInterleave = kernel["LocalDotLayout"]

    if dotInterleave == 1:
      kStr += inst("v_mul_u32_u24", \
          vgpr(destVgpr), \
          hex(kernel["MacroTile%s"%tP["tensorChar"]] + kernel["LdsPad%s"%tc]), \
          vgpr(destVgpr), \
          "lw%s%s**(MT%s + PAD)"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
    if dotInterleave:
      ldlOffsetVgpr = self.vgprPool.checkOut(1)
      kStr += inst("v_and_b32", \
          vgpr(destVgpr), \
          kernel["LocalDotLayout"]-1, \
          vgpr(uReg), \
          "uReg & LDL")
      kStr += inst("v_and_b32", \
          vgpr(uReg), \
          ~(kernel["LocalDotLayout"]-1), \
          vgpr(uReg), \
          "uReg & LDL")
      kStr += inst("v_mul_u32_u24", \
          vgpr(uReg), \
          hex(kernel["MacroTile%s"%tP["tensorChar"]] + kernel["LdsPad%s"%tc]), \
          vgpr(uReg), \
          "lw%s%s**(MT%s + PAD)"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
      kStr += inst("_v_add_co_u32", \
          vgpr(destVgpr), \
          "vcc", \
          vgpr(destVgpr), \
          vgpr(uReg), \
          "add scraps from LDL masking")
      kStr += inst("v_lshl_add_u32", \
          vgpr(destVgpr), \
          vgpr(tP["gpr"]["lwoT"]), \
          hex(log2(kernel["LocalDotLayout"])), \
          # 0, \
          vgpr(destVgpr), \
          "+= lw%s * LDL" % (tc))
      kStr += inst("v_lshlrev_b32", \
          vgpr(destVgpr), \
          hex(log2(tP["bpe"])), \
          vgpr(destVgpr), \
          " *= bpe")
      self.vgprPool.checkIn(ldlOffsetVgpr)
      #kStr += self.bomb(-40)
    else:
      kStr += inst("_v_add_lshl_u32", \
          vgpr(destVgpr), \
          vgpr(tP["gpr"]["lwoT"]), \
          vgpr(destVgpr), \
          hex(log2(tP["bpe"])), \
          "lwFO%s = (lw%s%s + lw%s%s*(MT%s+PAD))*bpe" \
          % (tc, tc, tc, tc, self.unrollChar, tP["tileChar"]) )

    if tP["isB"]:
      kStr += inst("_v_add_co_u32", \
          vgpr(destVgpr), \
          "vcc", \
          hex(kernel["LdsOffsetB"]*tP["bpe"]), \
          vgpr(destVgpr), \
          "lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u*%u" % (tP["tileChar"], \
          self.unrollChar, tP["tileChar"], kernel["LdsOffsetB"], self.bpeAB) )
    self.vgprPool.checkIn(tP["gpr"]["lwoT"])
    self.vgprPool.checkIn(tP["gpr"]["uReg"])
    if kernel["GlobalSplitU"] > 1:
      self.vgprPool.checkIn(tP["gpr"]["uReg2"])

    #LSC_ * LSP_
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    validWIPerLoad = kernel[tP["lsc"]] * kernel[tP["lsp"]] / tP["glvw"]
    validBytesPerLoad = kernel[tP["lsc"]] * kernel[tP["lsp"]] * numBytesPerElement
    maxBytesPerLoad = kernel["NumThreads"] * tP["glvw"] * numBytesPerElement

    assert (validBytesPerLoad <= maxBytesPerLoad)
    assert (kernel[tP["lsc"]] * kernel[tP["lsp"]] % tP["glvw"] == 0)

    if validBytesPerLoad != maxBytesPerLoad:
      tmpSgpr = self.getTmpSgpr(1)
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), validWIPerLoad, \
          "lsc*lsp=%u*%u"%(kernel[tP["lsc"]],kernel[tP["lsp"]] ))
      kStr += inst("v_cmp_lt_u32", \
          "vcc", \
          vgpr("Serial"), \
          sgpr(tmpSgpr), \
          "fractional: ensure tid < global read tile elements")
      tmpVgpr = self.vgprPool.checkOut(1)
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

    # dump lds write offsets
    #if tP["isA"]:
      #kStr += self.dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
      #kStr += self.bomb(-40)
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A/B
  ##############################################################################
  def lwaFinalOffsets(self, kernel, tP):
    return self.comment("N/A")

  ##############################################################################
  # Local Write Addresses: Declare Addresses A/B
  ##############################################################################
  def lwaDeclareAddresses(self, kernel, tP):
    return self.comment1("N/A")

  ##############################################################################
  # Local Read Addresses: Tile Assignment A
  ##############################################################################
  def lraTileAssignmentA(self, kernel, tP):
    kStr = ""
    kStr += "%slr%s = serial %% SG%s%s%s" \
        % (self.commentPrefix, tP["tileChar"], tP["tileChar"], \
        self.commentSuffix, self.endLine)

    divisor = kernel["SubGroup0"]
    qReg = self.vgprPool.checkOut(1) # quotient
    rReg = self.vgprPool.checkOut(1) # remainder
    dividendReg = "Serial" # local serial
    tmpVgpr = self.vgprPool.checkOut(2)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    tP["gpr"]["lro"] = rReg
    self.tmplroB = qReg
    self.vgprPool.checkIn(tmpVgpr)
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment B
  ##############################################################################
  def lraTileAssignmentB(self, kernel, tP):
    kStr = ""
    kStr += "%slr%s = (serial / SG%s) %% SG%s%s%s" \
        % (self.commentPrefix, tP["tileChar"], tP["tileChar"], \
        tP["tileChar"], self.commentSuffix, self.endLine)
    divisor = kernel["SubGroup1"]
    qReg = self.vgprPool.checkOut(1) # quotient
    rReg = self.vgprPool.checkOut(1) # remainder
    dividendReg = self.tmplroB
    tmpVgpr = self.vgprPool.checkOut(2)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    self.vgprPool.checkIn(self.tmplroB) # old
    tP["gpr"]["lro"] = rReg
    self.vgprPool.checkIn(qReg)
    self.vgprPool.checkIn(tmpVgpr)
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    kStr = ""
    tc = tP["tensorChar"]
    divisor = kernel["SubGroup0"]*kernel["SubGroup1"]
    qReg = self.vgprPool.checkOut(1) # quotient
    rReg = self.vgprPool.checkOut(1) # remainder, unused here
    dividendReg = "Serial"
    tmpVgpr = self.vgprPool.checkOut(2)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    sgid = qReg

    kStr += inst("s_mov_b32", \
        sgpr(tmpSgpr), \
        hex(kernel["MacroTile%u"%tP["tensorIdx"]] + kernel["LdsPad%s"%tc]), \
        "MT%u+PAD"%tP["tensorIdx"] )
    kStr += inst("v_mul_lo_u32", \
        vgpr(sgid), \
        sgpr(tmpSgpr), \
        vgpr(sgid), \
        "sgid=sgid*(MT%u+PAD)"%tP["tensorIdx"] )
    if kernel["VectorWidth"] > 1:
      kStr += staticMultiply(vgpr(tP["gpr"]["lro"]), vgpr(tP["gpr"]["lro"]), \
          kernel["VectorWidth"], sgpr(tmpSgpr))
    kStr += inst("_v_add_lshl_u32", \
        vgpr("LocalReadAddr%s"%tc), \
        vgpr(sgid), \
        vgpr(tP["gpr"]["lro"]), \
        hex(log2(tP["bpe"])), \
        "o = (lro%s*VW+sgid*MT%u)*bpe"%(tc, tP["tensorIdx"]) )
    
    if kernel["LocalDotLayout"] > 1:
      argStr = vgpr("LocalReadAddr%s"%tc)
      kStr += "v_mul_lo_u32 %s, %s, %u //o *= LocalDotLayout %s"%(argStr,argStr, kernel["LocalDotLayout"], self.endLine)


    #if tP["isA"]:
    #  kStr += self.bomb(113)


    # dump lra final offset
    #if tP["isA"]:
    #  kStr += dump(vgpr("LocalReadAddr%s"%tP["tensorChar"]))
    #  kStr += dump(vgpr("ElementIndex%s"%tP["tensorChar"])) 


    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(qReg)
    self.vgprPool.checkIn(rReg)
    self.vgprPool.checkIn(tP["gpr"]["lro"])
    return kStr

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    if tP["isA"]:
      return self.comment1("N/A")
    else:
#jgolds which bpe here? Looks like tP, which is B
      return inst("_v_add_co_u32", \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          "vcc", \
          hex(kernel["LdsOffset%s"%tP["tensorChar"]]*tP["bpe"]), \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          " += LdsOffset%s (lower)"%tP["tensorChar"])


  ##############################################################################
  # Initialize C
  ##############################################################################
  def initC(self, kernel):
    self.vgprPool.remove(self.startVgprValuC, \
        self.lastPreLoopTempVgpr - self.startVgprValuC)

    kStr = ""
    for i in range(0, self.numVgprValuC):
      kStr += inst("v_mov_b32", vgpr("ValuC+%u"%i), hex(0), "")
    return kStr


  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  def declareLoopNumIter(self, kernel):
    return ""


  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx):
    kStr = ""

    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]

    ########################################
    # Tail Loop
    if tailLoop:
      kStr += "%s//numIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)%s" \
          % (self.indent, self.unrollChar, self.unrollChar, self.endLine)
      tmpSgpr = self.getTmpSgpr(4)
      # size % DepthU
      kStr += scalarStaticDivideAndRemainder(tmpSgpr, "LoopCounters+%u"%loopIdx, "SizesSum+%u"%loopIdx, kernel["DepthU"], tmpSgpr+2, True)

      if kernel["LocalSplitU"] > 1:
        # (size % DepthU) + LSU - 1
        kStr += inst("s_add_u32", sgpr("LoopCounters+%u"%loopIdx), hex(kernel["LocalSplitU"]-1), sgpr("LoopCounters+%u"%loopIdx), "(size % DepthU) + LSU - 1" )
        dividend = tmpSgpr+2
        kStr += inst("s_mov_b32", sgpr(dividend), sgpr("LoopCounters+%u"%loopIdx), "copy for divide" )
        kStr += scalarStaticDivideAndRemainder( "LoopCounters+%u"%loopIdx, None, dividend, kernel["LocalSplitU"], tmpSgpr, False)

      # if GSU numIter=0 if gsuSumIdx != remainder
      if kernel["GlobalSplitU"] > 1:
        kStr += inst("s_cmp_eq_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
            "gsuSumIdx == numIterPerWgRemainder" )
        afterZero = self.getLabel("AfterNumIterZero")
        kStr += inst("s_cbranch_scc1", "label_%04u"%afterZero, "skip" )
        kStr += inst("s_mov_b32", sgpr("LoopCounters+%u"%loopIdx), hex(0), "numIter=0" )
        kStr += "label_%04u:%s" % (afterZero, self.endLine)

      # if tail numIter == 0 skip altogether
      tailLoopLabelEnd = self.getLabel("TailLoopEnd%s"%(loopChar) )
      kStr += inst("s_cmp_eq_u32", sgpr("LoopCounters+%u"%loopIdx), \
          hex(0), "numIter%s == 0"%loopChar )
      kStr += inst("s_cbranch_scc1 label_%04u"\
          % tailLoopLabelEnd, \
          "skip to end of tail loop b/c numIter==0")

    ########################################
    # Unrolled Loop
    elif loopIdx == self.unrollIdx:
      if not self.do["PreLoop"]: kStr += ".endif\n"
      tmpSgpr = self.getTmpSgpr(2)
      quotient = "LoopCounters+%u"%loopIdx
      dividend = "SizesSum+%u"%loopIdx
      divisor = kernel["DepthU"]
      kStr += scalarStaticDivideAndRemainder(quotient, None, dividend, divisor, tmpSgpr, False)

      # if GSU numIter++ if gsuSumIdx < remainder
      if kernel["GlobalSplitU"] > 1:
        tmpSgpr = self.getTmpSgpr(3)
        quotient = "LoopCounters+%u"%loopIdx
        remainder = "GSUSumIdx+1" # numIterPerWgRemainder
        dividend = tmpSgpr+2 # numIterMyWg
        divisor = kernel["GlobalSplitU"]
        kStr += inst("s_mov_b32", sgpr(dividend), sgpr("LoopCounters+%u"%loopIdx), "copy for divide" )
        kStr += scalarStaticDivideAndRemainder(quotient, remainder, dividend, divisor, tmpSgpr, True)

        # if gsuSumIdx < numIterPerWgRemainder
        kStr += inst("s_cmp_lt_u32", sgpr("GSUSumIdx"), sgpr("GSUSumIdx+1"), \
            "gsuSumIdx < numIterPerWgRemainder" )
        afterInc = self.getLabel("AfterNumIterInc")
        kStr += inst("s_cbranch_scc0", "label_%04u"%afterInc, "skip" )
        kStr += inst("s_add_u32", sgpr("LoopCounters+%u"%loopIdx), hex(1), sgpr("LoopCounters+%u"%loopIdx), "numIterMyWg++" )
        kStr += "label_%04u:%s" % (afterInc, self.endLine)

    ########################################
    # Multi-dimensional summation
    else:
      printExit("no assembly support for 2+ dimensional summation")
      kStr += "%snumIter%s = size%s" \
          % (self.indent, loopChar, loopChar)

    # counter = -counter
    kStr += inst("s_sub_u32", \
        sgpr("LoopCounters+%u"%loopIdx), \
        hex(0), \
        sgpr("LoopCounters+%u"%loopIdx), \
        "counter%s = -size%s"%(loopChar, loopChar) )
    return kStr

  ##############################################################################
  # Open Loop
  ##############################################################################
  def openLoop(self, kernel, loopIdx):
    kStr = ""
    #kStr += "s_endpgm\n"
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
      self.inTailLoop = True
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    loopLabelBegin = self.getLabel("%sLoopBegin%s"%("Tail" if tailLoop else "", loopChar) )
    loopLabelEnd = self.getLabel("%sLoopEnd%s"%("Tail" if tailLoop else "", loopChar) )

    # is numIter at least 1? otherwise skip to end
    endCounter = -1 if kernel["PrefetchGlobalRead"] and not tailLoop else 0
    kStr += inst("s_cmp_ge_i32", \
        sgpr("LoopCounters+%u"%loopIdx), \
        hex(endCounter), \
        "LoopCounter%s < EndCounter"%(loopChar) )
    kStr += inst("s_cbranch_scc1 label_%04u"%loopLabelEnd, \
        "don't enter Loop%s"%loopChar )

    # LSU not all threads will do summation
    if tailLoop and kernel["LocalSplitU"] > 1:
      tmpSgpr = self.getTmpSgpr(2)
      kStr += self.comment("apply exec mask for LSU")
      tmpVgpr = self.vgprPool.checkOut(2)
      dummy = self.vgprPool.checkOut(1)
      sgId = self.vgprPool.checkOut(1)
      divisor = kernel["SubGroup0"]*kernel["SubGroup1"]
      kStr += vectorStaticDivide(sgId, "Serial", divisor, tmpVgpr, tmpSgpr)
      numIter = self.vgprPool.checkOut(1)
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
    if tailLoop and kernel["LocalSplitU"] > 1:
      kStr += inst("v_cmpx_lt_u32", "vcc", \
          vgpr(sgId), vgpr(numIter), "sgId < numIter")
      kStr += inst("_v_add_co_u32", vgpr(sgId), "vcc", hex(kernel["LocalSplitU"]), \
          vgpr(sgId), "sgId+=LSU")
      self.vgprPool.checkIn(sgId)
      self.vgprPool.checkIn(numIter)
      #kStr += dump(vgpr(sgId))

    return kStr


  ##############################################################################
  # Close Loop
  ##############################################################################
  def closeLoop(self, kernel, loopIdx):
    kStr = ""
    #kStr += self.indent + self.syncStr + self.endLine
    #kStr += "s_endpgm\n"
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    loopLabelBegin = self.getLabel("%sLoopBegin%s"%("Tail" if tailLoop else "", loopChar) )
    loopLabelEnd = self.getLabel("%sLoopEnd%s"%("Tail" if tailLoop else "", loopChar) )
    endCounter = -1 if kernel["PrefetchGlobalRead"] and not tailLoop else 0

    if tailLoop and kernel["AssertSummationElementMultiple"]%kernel["InnerUnroll"]==0:
      unrollInc = kernel["InnerUnroll"]
    else:
      unrollInc = 1

    kStr += inst("s_add_u32", \
        sgpr("LoopCounters+%u"%loopIdx), \
        sgpr("LoopCounters+%u"%loopIdx), \
        hex(unrollInc), \
        "inc counter%s"%(loopChar) )
    kStr += inst("s_cmp_eq_i32", \
        sgpr("LoopCounters+%u"%loopIdx), \
        hex(endCounter), \
        "counter%s==0"%(loopChar) )
    kStr += inst("s_cbranch_scc1 label_%04u"%loopLabelEnd, \
        "exit Loop%s"%loopChar )
    kStr += inst("s_branch label_%04u"%loopLabelBegin, \
        "restart %s Loop%s"%("tailLoop" if tailLoop else "unrolled loop", loopChar ))
    kStr += "label_%04u:%s" % (loopLabelEnd, self.endLine)

    # restore all threads
    if tailLoop and kernel["LocalSplitU"] > 1:
      kStr += self.comment("restore full exec mask")
      fullExec = self.getTmpSgpr(2)
      kStr += inst("s_mov_b64", sgpr(fullExec,2), \
          "0xFFFFFFFFFFFFFFFF", "restore all threads active")
      kStr += inst("s_or_saveexec_b64",  sgpr(fullExec,2), sgpr(fullExec,2), "full mask -> exec" )
    return kStr

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self):
    self.vgprPool.add(self.startVgprValuA, \
        self.vgprReclaimAfterSummation - self.startVgprValuA + 1)
    self.setStartTmpPool(self.lastPostLoopSgpr)
    return ""

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  def macIter(self, kernel, bufferIdx, iuiCount):
    if not self.do["MAC"]: return ""
    kStr = ""

    if kernel["ProblemType"]["DataType"].isHalf():
      kStr += ".align32 8, 0xbf800001\n"   # Align v_pk_fma instructions used in MAC_ blocks

    if kernel["InnerUnroll"] > 1 and iuiCount==1:
      # This it tail-loop case where we just want one IUI, 
      kStr += "MAC_%ux%u_X%u_OneIUI" % (kernel["ThreadTile0"],kernel["ThreadTile1"], bufferIdx)
    else:
      kStr += "MAC_%ux%u_X%u" % (kernel["ThreadTile0"],kernel["ThreadTile1"], bufferIdx)

    kStr += self.endLine
    return kStr

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel, prefetch):
    kStr = ""
    if prefetch:
      lastIterEnd = self.getLabel("PrefetchGlobalLastIterEnd")
      kStr += inst("s_cmp_eq_u32", sgpr("LoopCounters+%u"%self.unrollIdx), \
          hex(0), "numIter%s == 0"%self.indexChars[self.unrollIdx])
      kStr += inst("s_cbranch_scc1 label_%04u"\
          % lastIterEnd, \
          "skip to end of prefetch last iter b/c numIter==0")
    return kStr

  def closeSumAtLeastUnroll(self, kernel, prefetch):
    kStr = ""
    if not prefetch:
      label = self.getLabel("PrefetchGlobalLastIterEnd")
      kStr += "label_%04u:%s" % (label, self.endLine)
    return kStr


  ##############################################################################
  ##############################################################################
  def incrementSrd(self, kernel, tP, incLower, incUpper):

    tc = tP["tensorChar"]
    kStr = ""

    kStr += inst("s_add_u32 ", \
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
    # TODO-64
    if kernel["PreciseBoundsCheck"]:
      kStr += inst("s_sub_u32 ", \
           sgpr("Srd%s+2"%(tc)), \
           sgpr("Srd%s+2"%(tc)), \
           incLower, \
            "limit -= inc)" )

    return kStr


  ##############################################################################
  # Global Read: Increment A/B
  ##############################################################################
  def globalReadIncrement(self, kernel, loopIdx, tP):
    if not self.do["GlobalInc"]: return ""
    kStr = ""
    tc = tP["tensorChar"]

    if kernel["BufferLoad"]:
      return self.incrementSrd(kernel, tP, \
              sgpr("GlobalReadIncs%s+0"%tc), \
              sgpr("GlobalReadIncs%s+1"%tc))
    else:
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      graIdx = 0
      tmp = self.vgprPool.checkOut(1, "groInc")
      #for perp in range(0, tP["nrp"]):
      #  for para in range(0, tP["nrc"]):
      #    for s in range(0, tP["nrcv"]):
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):
              if self.globalReadIncsUseVgpr:
                kStr += inst("_v_add_co_u32 ", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    vgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], loopIdx)), \
                    "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
                kStr += inst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    vgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], loopIdx)), \
                    "vcc", \
                    "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
              else:
                kStr += inst("_v_add_co_u32 ", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    sgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], loopIdx)), \
                    "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
                kStr += inst("v_mov_b32 ", \
                    vgpr(tmp), \
                    sgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], loopIdx)), \
                    "vgpr GlobalReadIncs%s"%tP["tensorChar"] )
                kStr += inst("_v_addc_co_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    vgpr(tmp), \
                    "vcc", \
                    "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
              graIdx += self.rpga
      self.vgprPool.checkIn(tmp)
      #kStr += dump(vgpr("GlobalReadAddrA+0"))
      #kStr += dump(vgpr("GlobalReadAddrA+1"))
      #kStr += "s_endpgm\n"

    return kStr

  ##############################################################################
  # Global Read: Do It A/B
  ##############################################################################
  def globalReadDo(self, kernel, guardK, tP):
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return ""
    kStr = ""
    tc = tP["tensorChar"]
    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth
    ldsOffset = 0

    if tP["isA"] and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
      kStr += self.comment1("before DirectToLds load, ensure prior ds_reads have finished")
      kStr += self.syncThreads(kernel)

    if kernel["DirectToLds%s"%tP["tensorChar"]]:
      # DirectToLds only enabled for TLU=1 cases, where the registers are directly copied into LDS
      if kernel["LocalWriteUseSgpr%s"%tc]:
        kStr += inst("s_mov_b32", "m0", sgpr("LocalWriteAddr%s"%tc), "m0 <- LDS write address")
      else:
        # TODO - remove this code? No reason not to use LocalWriteUseSgpr?
        lwaSgpr = self.getTmpSgpr(1)
        kStr += inst("v_readfirstlane_b32", sgpr(lwaSgpr), \
            vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            "Set lds write address to SGPR")
        kStr += inst("s_mov_b32", "m0", sgpr(lwaSgpr), "m0 <- LDS write address")

    # sizeK % LOCAL_DEPTHU
    if guardK:
      incrementSrd = False   # move the srd + base vs move the GRO

      ########################################
      # Calculate Max Addr
      ########################################
      maxAddrSgpr = self.getTmpSgpr(2) # 3+6 = 9 sgprs available
      tmpSgpr = maxAddrSgpr + 2 # 7 sgprs available
      #dumpVgpr = self.vgprPool.checkOut(1)

      # TODO-64B:
      # Assumes the product of the two sizes is <4GB here.
      # We would need to slide the SRD if this is not the case.
      kStr += self.comment1("max read address = size[n] * stride[n-1]")
      dim = len(tP["ia"])-1 # dim
      strideIdx = dim-1 # largest stride
      sizeIdx = tP["ia"][dim]
      sizeIdxIsSum = sizeIdx in kernel["ProblemType"]["IndicesSummation"]
      if sizeIdxIsSum:
	sizeIdx -= kernel["ProblemType"]["NumIndicesC"]
      if kernel["BufferLoad"] and not kernel["PreciseBoundsCheck"]:
          # Set maxAddrSgpr to max allowed byte offset
          # maxAddrSgpr = size[n] * stride[n-1] * bpe
          # SRD has moved ahead for each tile so subtract original A to see if we are OOB:

          kStr += inst("s_sub_u32", \
              sgpr(tmpSgpr), \
              sgpr("Srd%s+0"%tc), \
              sgpr("Address%s+0"%tc), \
              "Compute distance of SRD from original array in bytes")

          kStr += inst("s_mul_i32", \
              sgpr(maxAddrSgpr+0), \
              sgpr("Sizes%s+%u"%("Sum" if sizeIdxIsSum else "Free", sizeIdx)),  \
              sgpr("Strides%s+%u"%(tP["tensorChar"],strideIdx)), \
              "Array size")

          kStr += inst("s_lshl_b32",
              sgpr(maxAddrSgpr+0), \
              sgpr(maxAddrSgpr+0), \
              hex(log2(tP["bpe"])), \
              "Array size in bytes")

          kStr += inst("s_sub_u32", \
              sgpr(maxAddrSgpr), \
              sgpr(maxAddrSgpr), \
              sgpr(tmpSgpr), \
              "Max byte offset =  MaxSize - SRD_Distance")

      if not kernel["BufferLoad"]:
	kStr += inst("s_mul_i32", \
	    sgpr(maxAddrSgpr+0), \
	    sgpr("Sizes%s+%u"%("Sum" if sizeIdxIsSum else "Free", sizeIdx)),  \
	    sgpr("Strides%s+%u"%(tP["tensorChar"],strideIdx)), \
	    "mul d%u lower"%dim)

        kStr += inst("s_mov_b32", sgpr(maxAddrSgpr+1), hex(0), "zero (upper)")
        # maxAddrSgpr *= bytes/element

        kStr += inst("s_lshl_b64", \
            sgpr(maxAddrSgpr,2), \
            sgpr(maxAddrSgpr,2), \
            hex(log2(tP["bpe"])), "offset *= bytes/element")
            # maxAddrSgpr += initial address
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
        zeroVgpr = self.vgprPool.checkOut(1)
        kStr += inst("v_mov_b32", vgpr(zeroVgpr), hex(0), "zero")

      # End if guardK

    directToLdsLoads = 0

    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):
            i = sPara + (tP["nrcv"]/tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            graIdx = i * self.rpgo if kernel["BufferLoad"] else i * self.rpga
            g2lIdx = i * loadWidth
            if guardK:
              # for each component in vector
              r = 0
              while r < loadWidth*self.bpr/tP["bpe"]:
                kStr += self.comment1("g2l=%u, load component %u"%(g2lIdx, r))
                # load single element from address (except packed half case below)
                numElementsPerLoad = 1
                offset = 0

                if kernel["BufferLoad"]:
                  # mask if current address if in bounds
                  if kernel["PreciseBoundsCheck"]:
                    if kernel["UseSgprForGRO"]:
                      offsetVgpr = "GlobalReadOffset%s+0"%(tc)
                      if graIdx==0:
                        soffset = "0"
                      else:
                        soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))
                    else:
                      offsetVgpr = "GlobalReadOffset%s+%u"%(tc, graIdx)
                      soffset = "0"

                    offset = r * numElementsPerLoad * tP["bpe"]
                  else:
                    offsetVgpr = self.vgprPool.checkOut(1)
                    soffset = "0"
                    kStr += inst("v_cmp_lt_u32", "vcc", \
                          vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx)), \
                          sgpr(maxAddrSgpr), \
                          "addr < maxAddr")

                    kStr += inst("v_cndmask_b32", \
                                 vgpr(offsetVgpr), \
                                  -1,
                                  vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx),1), \
                                  "vcc",
                                  "Select offset or clip if OOB. offset")
                  if kernel["DirectToLds%s"%tP["tensorChar"]]:
                    ldsInc = kernel["NumThreads"]*4
                    if directToLdsLoads != 0:
                      kStr += inst("s_add_u32", "m0", "m0", ldsInc, \
                          "Move LDS write address to next line" )
                    directToLdsLoads+=1

                  if kernel["ProblemType"]["DataType"].isHalf():
                    if kernel["AssertSummationElementMultiple"] % 2 == 0:
                      if kernel["DirectToLds%s"%tP["tensorChar"]]:
                        # Assembler expects a destination VGPR even though not written
                        kStr += tP["globalReadInstruction"].toString( \
                          (\
                          vgpr(0), \
                          vgpr(offsetVgpr), \
                          sgpr("Srd%s"%(tP["tensorChar"]), 4), \
                          soffset,"lds"), \
                          "load packed 2xhalf  G -> LDS(%s)", tP["NonTemporal"], 0)
                      else:
                        kStr += inst("buffer_load_dword", \
                          vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r/2)),
                          vgpr(offsetVgpr), \
                          sgpr("Srd%s+%u"%(tP["tensorChar"], 0), 4), \
                          soffset, \
                          " offen offset:%u"%offset,\
                          "load packed 2xhalf")
                      numElementsPerLoad = 2
                      r += 1 # skip next element since we loaded 2X here
                    else:
                      kStr += inst("buffer_load_short_d16%s"%("_hi" if r%2==1 else ""), \
                          vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r/2)),
                          vgpr(offsetVgpr), \
                          sgpr("Srd%s+%u"%(tP["tensorChar"], 0), 4), \
                          soffset, \
                          " offen offset:%u"%offset,\
                          "load single f16")
                  elif kernel["ProblemType"]["DataType"].isSingle():
                    if kernel["DirectToLds%s"%tP["tensorChar"]]:
                      # Assembler expects a destination VGPR even though not written
                      kStr += tP["globalReadInstruction"].toString( \
                          (\
                          vgpr(0), \
                          vgpr(offsetVgpr), \
                          sgpr("Srd%s"%(tP["tensorChar"]), 4), \
                          soffset,"lds"), \
                          "load single float G -> LDS(%s)", tP["NonTemporal"], 0)
                    else:
                      kStr += inst("buffer_load_dword", \
                        vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r)),
                        vgpr(offsetVgpr), \
                        sgpr("Srd%s+%u"%(tP["tensorChar"], 0), 4), \
                        soffset, \
                        " offen offset:%u"%offset,\
                        "load single float")
                  elif kernel["ProblemType"]["DataType"].isDouble():
                    kStr += inst("buffer_load_dwordx2", \
                        vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r*2),2),
                        vgpr(offsetVgpr), \
                        sgpr("Srd%s+%u"%(tP["tensorChar"], 0), 4), \
                        soffset, \
                        " offen offset:%u"%offset,\
                        "load single double")
                  else:
                    printWarning("DataType unsupported")
                  if not kernel["PreciseBoundsCheck"]:
                    self.vgprPool.checkIn(offsetVgpr)

                  # increment offset by 1 element
                  if not kernel["UseSgprForGRO"]:
                    if incrementSrd:
                      kStr += self.incrementSrd(kernel, tP, numElementsPerLoad * tP["bpe"], 0)
                      kStr += inst("s_sub_u32", \
                          sgpr(maxAddrSgpr), \
                          sgpr(maxAddrSgpr), \
                          tP["bpe"], \
                          "Not USFGROAdjust max addr to account for SRD move")
                    else:
                      kStr += inst("_v_add_co_u32", \
                          vgpr("GlobalReadOffset%s+%u"%(tc, graIdx)), \
                          "vcc", \
                          vgpr("GlobalReadOffset%s+%u"%(tc, graIdx)), \
                          numElementsPerLoad * tP["bpe"], "graOffset += %u * bpe" % (numElementsPerLoad))
                else: # Not buffer load
                  # mask if current address if in bounds
                  kStr += inst("v_cmpx_lt_u64", "vcc", \
                      vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), \
                      vgpr(maxAddrVgpr,2), \
                      "addr < maxAddr")

                  # load single element from address
                  if kernel["ProblemType"]["DataType"].isHalf():
                    kStr += inst("flat_load_short_d16%s"%("_hi" if r%2==1 else ""), \
                        vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r/2)),
                        vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), "load single f16")
                  elif kernel["ProblemType"]["DataType"].isSingle():
                    kStr += inst("flat_load_dword", \
                        vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r)),
                        vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), "load single float")
                  elif kernel["ProblemType"]["DataType"].isDouble():
                    kStr += inst("flat_load_dwordx2", \
                        vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r*2),2),
                        vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), "load single double")
                  else:
                    printWarning("DataType unsupported")

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
                r += 1 # next component (for half)
            else: # not guardK
              if kernel["BufferLoad"]:
                if graIdx==0 or not kernel["UseSgprForGRO"]:
                  offsetVgpr= "GlobalReadOffset%s+%u"%(tc, graIdx)
                  soffset = "0"
                else:
                  offsetVgpr= "GlobalReadOffset%s+0"%(tc)
                  soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))

                if kernel["DirectToLds%s"%tP["tensorChar"]]:

                  # Get offset (for checking, see comment below) and comment:
                  (checkOffset, iDummy, comment) = \
                      self.calculateLdsWriteOffset(perp, para, sPerp, sPara, kernel, tP)
                  # Direct to LDS always writes consecutive LDS locations at m0 + 4 * TidInWave
                  # Therefore we double-check here to ensure the desired LDS write offset
                  # is moving at NumThreads*4.  This should already be guaranteed since
                  # we only use direct-to-lds for non-transpose cases but double-check here.
                  ldsInc = kernel["NumThreads"]*4
                  #print ("checkOffset=", checkOffset, "ldsOffset=", ldsOffset, "ldsInc=", ldsInc)

                  if directToLdsLoads != 0:
                    kStr += inst("s_add_u32", "m0", "m0", ldsInc, \
                        "Move LDS write address to next line" )
                  directToLdsLoads+=1
                  ldsOffset += ldsInc

                  # Assembler expects a destination VGPR even though not written
                  kStr += tP["globalReadInstruction"].toString( \
                      (\
                      vgpr(0), \
                      vgpr(offsetVgpr), \
                      sgpr("Srd%s"%(tP["tensorChar"]), 4), \
                      soffset,"lds"), \
                      "G -> LDS(%s)"%(comment), \
                      tP["NonTemporal"], 0)

                else:
                  kStr += tP["globalReadInstruction"].toString( \
                      (vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), loadWidth), \
                      vgpr(offsetVgpr), \
                      sgpr("Srd%s"%(tP["tensorChar"]), 4), \
                      soffset,""), \
                      "G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp ),\
                      tP["NonTemporal"], 0)
              else:
                kStr += tP["globalReadInstruction"].toString( \
                    (vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), loadWidth), \
                    vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2)), \
                    "G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp ), tP["NonTemporal"], 0 )

              #kStr += "s_waitcnt vmcnt(0)\n"
              #kStr += self.bomb()
              #kStr += dump(vgpr("G2L%s+%u"%(tP["tensorChar"], graIdx)))

    if self.db["ConservativeWaitCnt"] & 0x1:
        kStr += "s_barrier // debug\n"
        kStr += "s_waitcnt lgkmcnt(0) & vmcnt(0)\n"
        kStr += "s_barrier // debug\n"
        #kStr += self.assert_lt(vgpr("Serial"), 64) # examine second wavefront

    if guardK and kernel["UseSgprForGRO"]:
      # increment offset 0 by 1 element
      # have to do this after all the component loads since they all use 0
      if incrementSrd:
        kStr += self.incrementSrd(kernel, tP, tP["bpe"], 0)
        kStr += inst("s_sub_u32", \
            sgpr(maxAddrSgpr), \
            sgpr(maxAddrSgpr), \
            numElementsPerLoad * tP["bpe"], \
            "Adjust max addr to account for SRD move")
      else:
        kStr += inst("_v_add_co_u32", \
            vgpr("GlobalReadOffset%s+0"%(tc)), \
            "vcc", \
            vgpr("GlobalReadOffset%s+0"%(tc)), \
            tP["bpe"], "graOffset += bpe")

    # TODO - can remove one of these m0 restores if A and B both TLU
    if kernel["DirectToLds%s"%tP["tensorChar"]]:
      kStr += inst("s_mov_b32", "m0", \
          hex(kernel["LdsNumElements"] * tP["bpe"]), \
          "Restore LDS clamp at %u bytes"%(kernel["LdsNumElements"] * tP["bpe"]))

    if guardK:
      if not kernel["BufferLoad"]:
        self.vgprPool.checkIn(maxAddrVgpr)
        self.vgprPool.checkIn(bpeVgpr)
        self.vgprPool.checkIn(zeroVgpr)
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, tP):
    if not self.do["LocalWrite"]: return ""
    kStr = ""
    tc = tP["tensorChar"]
#jgolds which bpe here? assuming tP
#fixme-iui  need to use wrapping increment for double or triple buffering:
    if kernel["LocalWriteUseSgpr%s"%tc]:
      kStr += inst("s_xor_b32", \
          sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
          sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
          "swap Red Blk SGPR")
    else:
      kStr += inst("v_xor_b32", \
          vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
          vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
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
#jgolds which bpe here? assuming tP
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
    return self.comment1("N/A")


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
  def calculateLdsWriteOffset(self, perp, para, sPerp, sPara, kernel, tP):
    tc = tP["tensorChar"]
    lscaOffset = para * kernel[tP["lsc"]]
    lspaOffset = perp * kernel[tP["lsp"]]

    # Add component offset to interleave from different regs
    # and compute mysterious "i"
    assert(sPerp==0 or sPara==0)
    if tP["tlu"]:
      lspaOffset += sPerp
      lscaOffset += sPara
      i = sPara + (tP["nrcv"]/tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
    else:
      lscaOffset += sPara
      lspaOffset += sPerp
      i = sPara + (tP["nrcv"]/tP["nrcvpi"]) * (para * tP["glvw"] + tP["nrc"] * (sPerp + tP["glvw"] * tP["nrpv"] * perp ))


    if kernel["LocalDotLayout"] > 1:
      # apply interleave for LocalDot:
      # Else they complement the address calculation to place adjacent-in-u data
      # so adjacent-in-lds.
      print "  ", tc, ": perp", perp, "para", para , "sPerp=", sPerp, "sPara=", sPara, \
            "wtc=", tP["wtc"], "wuc=", tP["wuc"], "grcv=", tP["grcv"], \
            "lscaOffset=", lscaOffset, "lspaOffset=", lspaOffset
      spacing = tP["glvw"]
      lscaOffset += (lspaOffset % spacing) * kernel["LocalDotLayout"]
      lspaOffset /= spacing
      print "    After LDL: lscaOffset=", lscaOffset, "lspaOffset=", lspaOffset

    #if not tP["tlu"]:
    #  tmp = sPara
    #  sPara = sPerp
    #  sPerp = tmp
    #print "0lspaOffset", lspaOffset
    #print "0lscaOffset", lscaOffset

    if tP["tlu"]:
      lspaOffset *= (kernel[tP["mt"]] + kernel["LdsPad%s"%tc])
    else:
      lscaOffset *= (kernel[tP["mt"]] + kernel["LdsPad%s"%tc])
    #print "1lspaOffset", lspaOffset
    #print "1lscaOffset", lscaOffset
    #if tP["tlu"] == tP["grcv"]:
    #  lspaOffset *= tP["glvw"]
    #  lscaOffset *= tP["glvw"]

    #print "2lspaOffset", lspaOffset
    #print "2lscaOffset", lscaOffset
    offsetElements = (lspaOffset + lscaOffset)
    #print "offsetElements", offsetElements
    offsetBytes = offsetElements*tP["bpe"]
    #print "offsetBytes", offsetBytes
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

    self.localWriteDoCnt += 1
    kStr = ""
    tc = tP["tensorChar"]
    if not kernel["DirectToLds%s"%tc]:
      instruction = tP["localWriteInstruction"]
      numBlocks = instruction.numBlocks
      numOffsets = instruction.numOffsets
      blockWidth = instruction.blockWidth
      #offsetMultiplier = instruction.offsetMultiplier
      g2lIdx = 0
      #kStr += dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
      if 0:
        print "\nLocalWrite", tP["tensorChar"]
        print "tlu", tP["tlu"]
        print "lsc", kernel[tP["lsc"]]
        print "lsp", kernel[tP["lsp"]]
        print "grcv", tP["grcv"]
        print "wtc", tP["wtc"]
        print "wuc", tP["wuc"]
        print "nrc", tP["nrc"]
        print "nrp", tP["nrp"]
        print "nwcv", tP["nwcv"]
        print "nwpv", tP["nwpv"]
        print "nrcvpi", tP["nrcvpi"]
        print "nwcvpi", tP["nwcvpi"]

      tmpLocalWriteAddr = -1

      # if transposing, positions of sPerp and sPara are transposed
      for perp in range(0, tP["nrp"]):
        lwa = "LocalWriteAddr%s"%tc  # default
        if kernel["FractionalLoad"] and perp==tP["nrp"]-1:
          overhang = kernel["fractionalPerpOverhang%s"%tc]
          if overhang:
            if tmpLocalWriteAddr == -1:
              tmpLocalWriteAddr = self.vgprPool.checkOut(1)

            validWI = overhang*kernel[tP["lsc"]]/tP["glvw"]
            #print "%s: overhang=%u element validWI=%u" % (tc, overhang, validWI)
            kStr += self.comment1("LastPerp.  overhang=%u, mask WI>%u" % (overhang, validWI))
            kStr += inst("v_cndmask_b32", \
                        vgpr(tmpLocalWriteAddr), \
                        1.0, \
                        vgpr("LocalWriteAddr%s"%tc), \
                        sgpr("PerpOverhangVcc%s"%tc,2), \
                        "Mask load so out-of-gr-tile bounds returns 0. Note 1.0f=0x3f80000 which is large non-neg int")
            lwa = tmpLocalWriteAddr

        for para in range(0, tP["nrc"]):
          for s in range(0, max(tP["nwcv"],tP["nwpv"])/tP["nwcvpi"]):

            sPerp = 0
            sPara = 0
            if tP["tlu"]:
              if tP["wtc"] == tP["grcv"]:
                sPerp = s
              elif tP["wuc"] == tP["grcv"]:
                sPara = s
            else:
              if tP["wtc"] == tP["grcv"]:
                sPara = s
              elif tP["wuc"] == tP["grcv"]:
                sPerp = s

            (offset, i, comment) = self.calculateLdsWriteOffset(perp, para, sPerp, sPara, kernel, tP)
            g2lIdx = i*blockWidth


            paramList = []
            paramList.append(vgpr(lwa))
            for blockIdx in range(0, numBlocks):
              if blockWidth == 1:
                paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx)))
              else:
                paramList.append( vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), \
                    blockWidth))
            #kStr += dump(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx)))
            for oIdx in range(0, numOffsets):
              paramList.append(offset)

            #print "offset", offset

            paramTuple = tuple(paramList)
            #comment = "Reg -> L %u_%u_%u_%u"%(para, sPara, perp, sPerp)
            nonTemporal = 0
            highBits = False
            if kernel["ProblemType"]["DataType"].isHalf():
              if s%2==1:
                highBits = True
            kStr += tP["localWriteInstruction"].toString(paramTuple, comment, \
                nonTemporal, highBits)
        #kStr += "s_endpgm\n"

      if tmpLocalWriteAddr != -1:
        self.vgprPool.checkIn(tmpLocalWriteAddr)

    #kStr += self.assert_lt(vgpr("Serial"), 64)
    if 0:
      kStr += inst("s_barrier", "temp debug wait to check sync issue" )

    if 0 and tP["isA"]:
    #if 0 and self.localWriteDoCnt >= 0:
      kStr += "s_waitcnt lgkmcnt(0) & vmcnt(0)\n"
      kStr += inst("s_barrier", "dump LDS" )
      kStr += self.bomb(105)

    return kStr

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  def localReadSwapOffsets(self, kernel, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return ""
    kStr = ""
#jgolds which bpe here? assuming tP
    kStr += inst("v_xor_b32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A/B
  # x % n == n & (n-1) for n power of 2
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return ""
    kStr = ""
    if tP["localReadInstruction"].numOffsets == 1:
      tP["localReadOffset"] = 0
      tP["localReadElementOffset"] = 0
      kStr += self.comment1("handled internally")
#jgolds which bpe here? assuming tP
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
      tP["localReadOffset"] = 0
      tP["localReadElementOffset"] = 0
      kStr += self.comment1("N/A")
    else:
#jgolds which bpe here? assuming tP
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffset%s_Blk"%tP["tensorChar"]]*tP["bpe"]-1), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "reset Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  def localReadInc(self, kernel, iui, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return ""
    kStr = ""
    tc = tP["tensorChar"]
    if self.inTailLoop:
#jgolds which bpe here? assuming tP
      inc = kernel["LocalSplitU"]*(kernel["MacroTile%u"%tP["tensorIdx"]]+kernel["LdsPad%s"%tc])*tP["bpe"]
      tmpSgpr = self.getTmpSgpr(1)
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(inc), "inc")
      kStr += inst("_v_add_co_u32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "vcc", \
          sgpr(tmpSgpr), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "lr%s += %u (LSU*(MT+PAD)*bpe)"%(tP["tensorChar"], inc) )
    else:
      if tP["localReadInstruction"].numOffsets == 1:
        ldl = kernel["LocalDotLayout"]
        if ldl > 1:
          #jgolds
          #HACK just hard coding to verify it works for the case I am testing
          partialInc = 8    # in elements
          if iui < (kernel["InnerUnroll"] - 1):
            tP["localReadOffset"] += partialInc
          else:
            tP["localReadOffset"] += ldl * kernel["LocalSplitU"]*(kernel["MacroTile%u"%tP["tensorIdx"]] + kernel["LdsPad%s"%tc]) - partialInc * (ldl - 1)
        else:
          tP["localReadOffset"] += kernel["LocalSplitU"]*(kernel["MacroTile%u"%tP["tensorIdx"]] + kernel["LdsPad%s"%tc])
        kStr += self.comment1("N/A, lro->%d"%tP["localReadOffset"])
      else:
        inc = kernel["LocalSplitU"]*(kernel["MacroTile%u"%tP["tensorIdx"]]+kernel["LdsPad%s"%tc])
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
  ##############################################################################
  def localReadDo(self, kernel, bufferIdx, iui, tP):

    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return ""
    kStr = ""
    self.localReadDoCnt += 1
    instruction = tP["localReadInstruction"]
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    offsetMultiplier = 1 # instruction.offsetMultiplier
    #totalReads = (kernel["ThreadTile%u"%tP["tensorIdx"]]/blockWidth) / numOffsets
    valuIdx = 0
    numVectorsPerTile = (kernel["ThreadTile%u"%tP["tensorIdx"]]/kernel["VectorWidth"])
    #print "numVectorsPerTile", numVectorsPerTile
    numReadsPerVector = (kernel["VectorWidth"] * tP["bpe"] ) / (blockWidth*4) # bytes/register
    #print "numReadsPerVector", numReadsPerVector
    for vIdx in range(0, numVectorsPerTile):
      for rIdx in range(0, numReadsPerVector):
        paramList = []
        destVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuIdx), blockWidth)
        paramList.append(destVgpr)
        paramList.append(vgpr("LocalReadAddr%s"%tc))
        for oIdx in range(0, numOffsets):
          paramList.append((rIdx*blockWidth + kernel["SubGroup%u"%tP["tensorIdx"]]*(vIdx*numOffsets+oIdx)*kernel["VectorWidth"] \
              + tP["localReadOffset"])*tP["bpe"]/offsetMultiplier)
        paramTuple = tuple(paramList)
        comment = "L -> Reg lro=%d ti=%u vIdx=%u rIdx=%u oIdx=%u buffer=%u iui=%u"\
            %(tP["localReadOffset"],kernel["SubGroup%u"%tP["tensorIdx"]], vIdx, rIdx, oIdx, bufferIdx, iui)
        kStr += instruction.toString(paramTuple, comment)
        valuIdx += blockWidth

        # TODO - handle vector-load
        if self.db["CheckValue1%s"%tc]:
            kStr += "s_waitcnt lgkmcnt(0) // CheckValue1 wait for LDS read\n"
            if kernel["ProblemType"]["DataType"].isHalf():
              kStr += self.assert_eq(destVgpr, hex(0x3c003c00)) # packed 1s
            elif kernel["ProblemType"]["DataType"].isSingle():
              kStr += self.assert_eq(destVgpr, 1.0)

    #if tP["isB"]:
    #  kStr += self.dumpLds(kernel, 0, 16)
    #  kStr += "s_endpgm\n"
    #if tP["isA"]:
    #kStr += "s_waitcnt lgkmcnt(0)\n"
    #if tP["isA"]:
    #  kStr += dump(vgpr("Valu%s%s+%u"%("Blk" if bufferColor else "", tP["tensorChar"], 0)))
    #if tP["isB"]:
    #  kStr += dump(vgpr("Valu%s%s+%u"%("Blk" if bufferColor else "", tP["tensorChar"], 0)))

    #if tP["isA"] and self.localReadDoCnt >=3: # TODO - disable
    #  # skip over tmp used above, so it doesn't get trashed
    #  tmpVgpr = self.vgprPool.checkOut(3) 
    #  kStr += self.bomb(self.localReadDoCnt + 10, tmpVgpr+1)
    #  self.vgprPool.checkIn(tmpVgpr)
    return kStr

  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponents(self, kernel, tP):
    kStr = ""

    # glvw
    vw = tP["glvw"]
    numVectors = kernel[tP["tt"]]/vw

    # labels
    svrLabels = []
    sviLabels = []
    for i in range(0, vw):
      r = (i+1) % vw
      label = self.getLabel("ShiftVectorComponents%u_R%u"%(tP["idx"], r) )
      svrLabels.append(label)
      tmpLabels = []
      for v in range(0, numVectors):
        label = self.getLabel("ShiftVectorComponents%u_R%u_V%u"%(tP["idx"], r, v) )
        tmpLabels.append(label)
      sviLabels.append(tmpLabels)

    # wgMT value
    tmpSgpr = self.getTmpSgpr(2)
    tmpVgpr = self.vgprPool.checkOut(2)
    wgMT = self.vgprPool.checkOut(1)
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
    dummy = self.vgprPool.checkOut(1)

    # qReg
    qReg = self.vgprPool.checkOut(1)
    divisor = kernel["VectorWidth"] # vw
    kStr += vectorStaticDivide(qReg, wgMT, divisor, \
        tmpVgpr, tmpSgpr)

    # rReg
    rReg = self.vgprPool.checkOut(1)
    divisor = vw
    kStr += vectorStaticRemainder(dummy, rReg, wgMT, divisor, \
        tmpVgpr, tmpSgpr)

    # qReg %/ SG
    sReg = self.vgprPool.checkOut(1)
    eReg = self.vgprPool.checkOut(1)
    divisor = kernel[tP["sg"]]
    kStr += vectorStaticDivideAndRemainder(sReg, eReg, qReg, divisor, \
        tmpVgpr, tmpSgpr)

    if tP["isA"]:
      # thread = serial % SG0
      thread = self.vgprPool.checkOut(1)
      divisor = kernel["SubGroup0"]
      kStr += vectorStaticRemainder(dummy, thread, "Serial", divisor, \
          tmpVgpr, tmpSgpr)
      #kStr += dump(vgpr(thread))
      #kStr += dump(vgpr(thread))
    else:
      # thread = (serial / SG0) % SG1
      sd0 = self.vgprPool.checkOut(1)
      divisor = kernel["SubGroup0"]
      kStr += vectorStaticDivide(sd0, "Serial", divisor, \
          tmpVgpr, tmpSgpr) # thread = serial / SG0
      divisor = kernel["SubGroup1"]
      thread = self.vgprPool.checkOut(1)
      kStr += vectorStaticRemainder(dummy, thread, sd0, divisor, \
          tmpVgpr, tmpSgpr) # thread = (serial / SG0) % SG1
      self.vgprPool.checkIn(sd0)

    # which glvw vector of thread to shift? wgMT / (SG0*VW) -> (wgMT%VW) / glvw
    # (wgMT/(WG0*VW))*(VW/glvw) + (wgMT%VW) / glvw
    if tP["tt"] > kernel["VectorWidth"]:
      mvReg = self.vgprPool.checkOut(1)
      divisor = kernel[tP["sg"]]*kernel["VectorWidth"]
      kStr += vectorStaticDivide(mvReg, wgMT, divisor, \
          tmpVgpr, tmpSgpr)
      if vw < kernel["VectorWidth"]:
        kStr += inst("v_lshlrev_b32", vgpr(mvReg), hex(log2(kernel["VectorWidth"]/vw)), vgpr(mvReg), "vId *= VW/glvw")
    #kStr += dump(vgpr(mvReg))

    vReg = self.vgprPool.checkOut(1)
    divisor = kernel["VectorWidth"]
    kStr += vectorStaticRemainder(dummy, vReg, wgMT, divisor, \
        tmpVgpr, tmpSgpr)
    vRegD = self.vgprPool.checkOut(1)
    kStr += inst("v_mov_b32", vgpr(vRegD), vgpr(vReg), "duplicate")
    divisor = vw
    kStr += vectorStaticDivide(vReg, vRegD, divisor, \
        tmpVgpr, tmpSgpr)
    #kStr += dump(vgpr(vReg))

    if tP["tt"] > kernel["VectorWidth"]:
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
        # mask if last thread in thread-tile column
        kStr += inst("v_cmpx_eq_u32", sgpr(tmpSgpr,2), vgpr(thread), \
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
              srcVgpr = self.startVgprValuC+src*self.bpeCinternal/self.bpr
              dstVgpr = self.startVgprValuC+dst*self.bpeCinternal/self.bpr
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
                    tmpSrcVgpr = self.vgprPool.checkOut(1)
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
              for i in range(0, self.bpeCinternal/self.bpr):
                kStr += inst("v_mov_b32", vgpr(self.startVgprValuC+dst*self.bpeCinternal/self.bpr+i), \
                    vgpr(self.startVgprValuC+src*self.bpeCinternal/self.bpr+i), comment)

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
  # Complex Declare Tmp Registers - SKIP
  ##############################################################################
  def complexDeclareTmpRegisters(self, kernel):
    kStr = ""
    return kStr
    if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
      kStr += "  float type_mac_tmp" + self.endLine
    if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
      kStr += "  double type_mac_tmp" + self.endLine
    return kStr

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
    elif kernel["ProblemType"]["DataType"].isSingle():
      elementStep = 1*(useDwordX2+1)
    elif kernel["ProblemType"]["DataType"].isDouble():
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

    tmpVgpr = self.vgprPool.checkOut(2)
    lr0 = self.vgprPool.checkOut(1)
    lr1 = self.vgprPool.checkOut(1)
    sg = self.vgprPool.checkOut(1)
    copy = self.vgprPool.checkOut(1)
    tmpSgpr = self.getTmpSgpr(1)

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
    for j in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
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
              regIdx = regIdx * self.bpeCinternal / 4
              kStr += inst("ds_write_b64", vgpr(addr), vgpr("ValuC+%u"%regIdx,2), \
                           "offset:%u"%(elementStep*writeOffset*self.bpeCinternal), 
                           "j=%u i=%u s=%u vc=%u"%(j,i,s,vc))
            else:
              regIdx /= elementStep
              kStr += inst("ds_write_b32", vgpr(addr), vgpr("ValuC+%u"%regIdx), \
                           "offset:%u"%(elementStep*writeOffset*self.bpeCinternal), 
                           "j=%u i=%u s=%u vc=%u"%(j,i,s,vc))
            # ds_write value
            #kStr += dump(vgpr(regIdx))
    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for all writes")
    kStr += self.syncThreads(kernel, "post-lsu local write")
    #kStr += self.dumpLds(kernel, 0, 16)
    #kStr += self.bomb(5)
    return kStr

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    kStr = ""
    tmpSgpr = self.getTmpSgpr(1)
    baseAddr = self.vgprPool.checkOut(1)
#jgolds which bpe should we use?
    kStr += staticMultiply(vgpr(baseAddr), vgpr("Serial"), kernel["GlobalWriteVectorWidth"]*self.bpeAB, sgpr(tmpSgpr))
    (elementStep, useDwordX2) = self.getLocalSplitUElementStep(kernel, True)
    # Load values for each subgroup
    for r in range(0, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"], elementStep):
          offset = s + i*kernel["NumThreads"]*kernel["GlobalWriteVectorWidth"] + r * kernel["MacroTile0"]*kernel["MacroTile1"]
          regIdx = s + i*kernel["GlobalWriteVectorWidth"] + r*kernel["GlobalWriteVectorWidth"]*kernel["NumGlobalWriteVectorsPerThread"]
          if useDwordX2:
            regIdx = regIdx * self.bpeCinternal / 4
            kStr += inst("ds_read_b64", vgpr("ValuC+%u"%regIdx,2), \
                vgpr(baseAddr), "offset:%u"%(offset*self.bpeCinternal), "r=%u i=%u s=%u"%(r,i,s))
          else:
            regIdx /= elementStep
            kStr += inst("ds_read_b32", vgpr("ValuC+%u"%regIdx), \
                vgpr(baseAddr), "offset:%u"%(offset*self.bpeCinternal), "r=%u i=%u s=%u"%(r,i,s))
    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for all reads")
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

          if kernel["ProblemType"]["DataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
            cIdx /= elementStep
            regIdx /= elementStep
            kStr += inst("v_pk_add_f16", vgpr("ValuC+%u"%cIdx), \
                vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), "c[%u] += c[%u]"%(cIdx, regIdx) )
          elif kernel["ProblemType"]["DataType"].isSingle():
            cIdx /= elementStep
            regIdx /= elementStep
            kStr += inst("v_add_f32", vgpr("ValuC+%u"%cIdx), \
                vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), "c[%u] += c[%u]"%(cIdx, regIdx) )
          elif kernel["ProblemType"]["DataType"].isDouble():
            cIdx *= 2
            regIdx *= 2 # for doubles, each element takes two regs
            kStr += inst("v_add_f64", vgpr("ValuC+%u"%cIdx,2), \
                vgpr("ValuC+%u" % regIdx,2), vgpr("ValuC+%u"%cIdx,2), "c[%u] += c[%u]"%(cIdx, regIdx) )
          else:
            assert(0) # unsupported data type, need to modify here and LSU write/read code
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    kStr = ""

    if kernel["BufferStore"]:
      kStr += self.allocPostLoopSrd(kernel, "C")

    # tmp gprs
    tid0 = self.vgprPool.checkOut(1)
    tid1 = self.vgprPool.checkOut(1)
    tmpVgpr = self.vgprPool.checkOut(2)
    tmpSgpr = self.getTmpSgpr(2)
    tmpS0 = tmpSgpr
    tmpS1 = tmpS0+1

    # lr0 = serial % SG0
    divisor = kernel["MacroTile0"] / kernel["GlobalWriteVectorWidth"]
    kStr += vectorStaticDivideAndRemainder(tid1, tid0, "Serial", \
        divisor, tmpVgpr, tmpSgpr)
    kStr += staticMultiply(vgpr(tid0), vgpr(tid0), kernel["GlobalWriteVectorWidth"], sgpr(tmpSgpr))

    # workgroup offset
    kStr += inst("s_mul_i32", \
        sgpr(tmpS0), \
        hex(kernel["MacroTile0"]), \
        sgpr("WorkGroup0"), \
        "%s = wg0*MT0"%sgpr(tmpS0))
    kStr += inst("s_mul_i32", \
        sgpr(tmpS1), \
        hex(kernel["MacroTile1"]), \
        sgpr("WorkGroup1"), \
        "%s = wg1*MT1"%sgpr(tmpS1))
    #kStr += dump(vgpr(tid0))

    # coord = tid*VW + workgroup offset
    kStr += inst("_v_add_co_u32", \
        vgpr(tid0), \
        "vcc", \
        sgpr(tmpS0), \
        vgpr(tid0), \
        "coord0 = tid0*VW + wg0*MT0")
    kStr += inst("_v_add_co_u32", \
        vgpr(tid1), \
        "vcc", \
        sgpr(tmpS1), \
        vgpr(tid1), \
        "coord1 = tid1*VW + wg1*MT1")

    self.vgprPool.checkIn(tmpVgpr)
    self.coord0 = tid0
    self.coord1 = tid1
    if kernel["BufferStore"]:
      #print "----AddressC-LocalSplitU"
      #print self.vgprPool.state()
      self.addrC = -1
    else:
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
    kStr += inst("s_mov_b32", sgpr("Srd%s+3"%ch), "Srd127_96", "Set bits 127_96 in SRD")
    return kStr


  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    #print "GlobalWriteIndices"
    if not self.do["PostLoop"]: return ""
    kStr = ""

    if kernel["BufferStore"]:
      kStr += self.allocPostLoopSrd(kernel, "C")

    # thread id 0,1
    # These will live for entire GlobalWrite loop - allocate before tmps
    # to avoid fragmentation
    tid0 = self.vgprPool.checkOut(1)
    tid1 = self.vgprPool.checkOut(1)

    self.scratchSgprs = self.getTmpSgpr(1)

    tmpS0 = self.scratchSgprs
    tmpS1 = tmpS0+1
    tmpV0 = self.vgprPool.checkOut(2)

    divisor = kernel["SubGroup0"]
    kStr += vectorStaticDivideAndRemainder(tid1, tid0, "Serial", divisor, \
        tmpV0, tmpS0)
    kStr += staticMultiply(vgpr(tid0), vgpr(tid0), kernel["VectorWidth"], sgpr(tmpS1))
    kStr += staticMultiply(vgpr(tid1), vgpr(tid1), kernel["VectorWidth"], sgpr(tmpS1))

    # workgroup offset
    kStr += inst("s_mul_i32", \
        sgpr(tmpS0), \
        hex(kernel["MacroTile0"]), \
        sgpr("WorkGroup0"), \
        "%s = wg0*MT0"%sgpr(tmpS0))
    kStr += inst("s_mul_i32", \
        sgpr(tmpS1), \
        hex(kernel["MacroTile1"]), \
        sgpr("WorkGroup1"), \
        "%s = wg1*MT1"%sgpr(tmpS1))
    #kStr += dump(vgpr(tid0))
    #kStr += dump(vgpr(tid1))

    # coord = tid*VW + workgroup offset
    kStr += inst("_v_add_co_u32", \
        vgpr(tid0), \
        "vcc", \
        sgpr(tmpS0), \
        vgpr(tid0), \
        "coord0 = tid0*VW + wg0*MT0")
    kStr += inst("_v_add_co_u32", \
        vgpr(tid1), \
        "vcc", \
        sgpr(tmpS1), \
        vgpr(tid1), \
        "coord1 = tid1*VW + wg1*MT1")
    self.vgprPool.checkIn(tmpV0)
    self.coord0 = tid0
    self.coord1 = tid1
    if kernel["BufferStore"]:
      #print "----AddressC-nonLSU-----"
      #print self.vgprPool.state()
      self.addrC = -1
    else:
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
  # Release any resources used by the global write
  def cleanupGlobalWrite(self, kernel):
    self.vgprPool.checkIn(self.coord0)
    self.vgprPool.checkIn(self.coord1)
    if not kernel["BufferStore"]:
      self.vgprPool.checkIn(self.addrC)

    if self.alphaVgpr != None:
      self.vgprPool.checkIn(self.alphaVgpr)
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
  # Not LocalSplitU: Global Write
  # Determine write batching pattern
  # element() specifies TT 'coordinate' to write
  # vectorWidths specifies width of vector to store
  # TODO - why does this use VectorWidth to control store width?  Could be GlobalWriteVectorWidth?
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""
    lsu = False


    fullVw = kernel["VectorWidth"] if kernel["VectorStore"] else 1
    fullVw = min(fullVw, self.maxGwvw(kernel))
    elements = [[] for y in range(2)] # 2D array for Full, Edge

    # Full tile loop:
    for tt1 in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for tt0 in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for vc1 in range(0, kernel["VectorWidth"]):
          for vc0 in range(0, kernel["VectorWidth"], fullVw): # note step by fullVw
            element = (tt1, tt0, vc1, vc0)
            elements[False].append(element)

    # Edge tile loop - note if we know AF0EM we can can use a larger vector
    # and reduce the boundary checks accordingly.  But if no AF0EM guarantee 
    # then use a conservative 1
    edgeVw = kernel["VectorWidth"] if kernel["VectorStore"] else 1
    edgeVw = min(edgeVw, self.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
    assert(kernel["VectorWidth"]%edgeVw == 0)
    for tt1 in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for tt0 in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for vc1 in range(0, kernel["VectorWidth"]):
          for vc0 in range(0, kernel["VectorWidth"], edgeVw):
            element = (tt1, tt0, vc1, vc0)
            elements[True].append(element)

    vectorWidths = [fullVw, edgeVw]
    kStr =  self.globalWriteElements(kernel, lsu, vectorWidths, elements)
    self.cleanupGlobalWrite(kernel)
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""
    lsu = True

    fullVw = kernel["GlobalWriteVectorWidth"] if kernel["VectorStore"] else 1
    fullVw = min(fullVw, self.maxGwvw(kernel))
    elements = [[] for y in range(2)] # 2D array for Full, Edge
    # Full tile loop:
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for tt0 in range(0, 1):
        for vc1 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], fullVw): # note step by fullVw
            element = (tt1, tt0, vc1, vc0)
            elements[False].append(element)

    # Edge tile loop - note if we know AF0EM we can can use a larger vector
    # and reduce the boundary checks accordingly.  But if no AF0EM guarantee 
    # then use a conservative 1
    edgeVw = kernel["GlobalWriteVectorWidth"] if kernel["VectorStore"] else 1
    edgeVw = min(edgeVw, self.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
    assert(kernel["GlobalWriteVectorWidth"]%edgeVw == 0)
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for tt0 in range(0, 1):
        for vc1 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], edgeVw):
            element = (tt1, tt0, vc1, vc0)
            elements[True].append(element)

    vectorWidths = [fullVw, edgeVw]
    kStr =  self.globalWriteElements(kernel, lsu, vectorWidths, elements)
    self.cleanupGlobalWrite(kernel)
    return kStr

  ##############################################################################
  # Global Write Elements
  ##############################################################################
  def globalWriteElements(self, kernel, lsu, vectorWidths, elements):
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
        writeLabels[beta]["EdgeCheck0"] = self.getLabel("GW_B%u_E%u_EdgeCheck0" % ( 1 if beta else 0, 1 if edge else 0) )
        writeLabels[beta]["EdgeCheck1"] = self.getLabel("GW_B%u_E%u_EdgeCheck1" % ( 1 if beta else 0, 1 if edge else 0) )
        writeLabels[beta][edge] = self.getLabel("GW_B%u_E%u" % ( 1 if beta else 0, 1 if edge else 0) )
      if not beta:
        betaLabel = self.getLabel("GW_Beta")
    endLabel = self.getLabel("GW_End")

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
    self.alphaVgpr = None
    self.betaVgpr = None
    if kernel["ProblemType"]["DataType"].isHalf():
      self.alphaVgpr = self.vgprPool.checkOut(1, "alpha")
      # alpha, beta are packed halfs in half mode (f16.hi == f16.lo) - setup on host
      kStr += inst("v_mov_b32", vgpr(self.alphaVgpr), sgpr("Alpha"), "sgpr -> vgpr b/c op_sel")
      if beta:
        self.betaVgpr = self.vgprPool.checkOut(1, "beta")
        kStr += inst("v_mov_b32", vgpr(self.betaVgpr), sgpr("Beta"), "sgpr -> vgpr b/c op_sel")
#jgolds look at moving these converted values back to scalar regs and free up the VGPRs
# bozo - should be able to keep alpha in float32 form?  How wide is alpha in HPA mode?
      if kernel["ProblemType"]["HighPrecisionAccumulate"]:
        kStr += inst("v_cvt_f32_f16", vgpr(self.alphaVgpr), vgpr(self.alphaVgpr), "convert alpha to fp32")
        if beta:
          kStr += inst("v_cvt_f32_f16", vgpr(self.betaVgpr), vgpr(self.betaVgpr), "convert beta to fp32")

    ########################################
    # Vgprs
    goc = 2 if kernel["BufferStore"] else 3 # GLOBAL_OFFSET_C
    tmpVgpr = self.vgprPool.checkOut(2+goc,"tmp-GlobalWrite") # 2 for coord + GLOBAL_OFFSET_C

    ########################################
    # Sgprs
    numSgprsPerElement = 2
    maxElementsPerBatch = 4 if not beta else 8
    numSgprsForPostLoop = 2+2+6+ maxElementsPerBatch*numSgprsPerElement
    globalWriteSgprs = self.getTmpSgpr(numSgprsForPostLoop)
    # create full exec mask
    fullExecMaskSgpr = globalWriteSgprs
    globalWriteSgprs += 2
    kStr += inst("s_mov_b64", \
        sgpr(fullExecMaskSgpr,2), \
        "0xFFFFFFFFFFFFFFFF", \
        "full exec mask")
    tmpSgpr = globalWriteSgprs
    globalWriteSgprs += 6
    elementSgprs = globalWriteSgprs

    # branch B1 or B0
    if kernel["ProblemType"]["UseBeta"]:
      betaLabel = self.getLabel("GW_Beta")
      if self.bpeCinternal <= self.bpr: # 1 register to check for Beta==0
        kStr += inst("s_cmpk_eq_u32", sgpr("Beta"), hex(0), "Beta == 0")
      else: # multiple registers to check for Beta==0
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), sgpr("Beta+0"), "tmp = Beta[0]")
        for i in range(1, self.bpeCinternal/self.bpr):
          kStr += inst("s_or_b32", sgpr(tmpSgpr), sgpr("Beta+%u"%i), sgpr(tmpSgpr), "tmp |= Beta[%u] " % i)
        kStr += inst("s_cmpk_eq_u32", sgpr(tmpSgpr), hex(0), "Beta == 0")
      kStr += inst("s_cbranch_scc0 label_%04u" % betaLabel, \
          "Beta not not zero; so jump to B nonzero")

    for beta in betas:
      # start B1
      if beta:
        kStr += "label_%04u:%s"%(betaLabel, self.endLine)
      tmpS01 = tmpSgpr
      tmpS23 = tmpS01 + 2
      tmpS45 = tmpS23 + 2

      ########################################
      # branch E1 or E0

      # check edge0 ###

      # s01 = rMT0
      kStr += inst("s_mov_b32", sgpr(tmpS01), hex(0), "rMT0=0" ) 

      # s23 = nwg0-1
      kStr += inst("s_add_u32", sgpr(tmpS23), hex(-1), sgpr("NumWorkGroups0"), "" ) 
      kStr += inst("s_cmp_lt_u32", sgpr("WorkGroup0"), sgpr(tmpS23), "wg0 < nwg0-1")
      kStr += inst("s_cbranch_scc1 label_%04u" % writeLabels[beta]["EdgeCheck0"], \
          "wg0 < nwg0-1 so skip rMT0 = Size0 % MT0")

      # s23 = rMT0 = Size0 % MT0
      kStr += scalarStaticDivideAndRemainder(tmpS23, tmpS01, "SizesFree+0", \
          kernel["MacroTile0"], tmpS45, True)
      kStr += "label_%04u:%s"%(writeLabels[beta]["EdgeCheck0"], self.endLine)
      # s01 now = myMT0 = wg0 < nwg0-1 ? MT0 : rMT0

      # if rMT0 > 0 goto label_B?_E1
      if self.do["EdgeWrite"]:
        kStr += inst("s_cmpk_gt_u32", sgpr(tmpS01), hex(0), "rMT0 > 0")
        kStr += inst("s_cbranch_scc1 label_%04u" % writeLabels[beta][True], \
            "edges required so jump to E1")

      # check edge1 ###

      # s01 = rMT1
      kStr += inst("s_mov_b32", sgpr(tmpS01), hex(0), "rMT1=0" ) 

      # s23 = nwg1-1
      kStr += inst("s_add_u32", sgpr(tmpS23), hex(-1), sgpr("NumWorkGroups1"), "" ) 
      kStr += inst("s_cmp_lt_u32", sgpr("WorkGroup1"), sgpr(tmpS23), "wg1 < nwg1-1")
      kStr += inst("s_cbranch_scc1 label_%04u" % writeLabels[beta]["EdgeCheck1"], \
          "wg1 < nwg1-1 so skip rMT1 = Size1 % MT1")

      # s23 = rMT1 = Size1 % MT1
      kStr += scalarStaticDivideAndRemainder(tmpS23, tmpS01, "SizesFree+1", \
          kernel["MacroTile1"], tmpS45, True)
      kStr += "label_%04u:%s"%(writeLabels[beta]["EdgeCheck1"], self.endLine)
      # s01 now = myMT1 = wg1 < nwg1-1 ? MT1 : rMT1

      # if rMT1 > 0 goto label_B?_E1
      if self.do["EdgeWrite"]:
        kStr += inst("s_cmpk_gt_u32", sgpr(tmpS01), hex(0), "rMT1 > 0")
        kStr += inst("s_cbranch_scc1 label_%04u" % writeLabels[beta][True], \
            "edges required so jump to E1")

      # by now we either jumped to E1 or stayed at E0
      for edge in edges:
        kStr += "label_%04u:%s"%(writeLabels[beta][edge], self.endLine)

        edgeI = edge  # set to True to disable vector stores
        #edgeI = True  # set to True to disable vector stores

        gwvw = vectorWidths[edgeI]


        ########################################
        # Calculate Vgprs for Write Batching
        ########################################

        numElementSgprs = self.totalSgprs - elementSgprs
        numElementsPerBatchLimitedBySgprs = numElementSgprs / numSgprsPerElement
        # how many vgprs are needed for zero elements
        # 2 for addressC in vgpr for addition - already checked out
        # 2 for coord0,1 of thread - already checked out
        # 2 for tmp - already checked out

        # 5 = how many vgprs are needed per element
        # 2 for addr
        # 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
        # if beta gwvw*rpe for new value
        # if atomic 2*rpe for old and cmp values

        # Use bpeCexternal for all external values

        numVgprsPerAddr = self.rpgo if kernel["BufferStore"] else self.rpga
#jgolds which bpe should we use?
        numVgprsPerDataPerVI = 0
        if atomic:
          # flat atomics have another VGPR to allow different data for return#
          regsPerElement = 2 if kernel["BufferStore"] else 3
          # The atomic loop processes multiple elements in single instruction
          # so will use VGPR from consec elements? TODO
          numVgprsPerDataPerVI = (regsPerElement*self.bpeCexternal)/self.bpr
        elif beta:
          numVgprsPerDataPerVI = (1.0*self.bpeCexternal)/self.bpr
        numVgprsPerElement = numVgprsPerAddr + int(ceil(numVgprsPerDataPerVI * gwvw))

        #print self.vgprPool.state()
        numVgprAvailable = self.vgprPool.availableBlock()
        # Grow the register pool if needed - we need enough regs for at least one element
        # Unfortunate since this means the write logic is setting the VGPR requirement
        # for the entire kernel but at least we have a functional kernel
        # TODO : the vgprSerial is needed for-ever and if we grow here will split the
        # range of the tmps.  Maybe want to move vgprSerial to first vgpr?
        minElements = 2 if kernel["ProblemType"]["DataType"].isHalf() else 1
        shrinkDb = 0
        if numVgprAvailable < minElements*numVgprsPerElement:
          gwvwOrig = gwvw
          currentOccupancy = self.getOccupancy(kernel, self.vgprPool.size())
          futureOccupancy = self.getOccupancy(kernel, \
              self.vgprPool.size() - numVgprAvailable + minElements*numVgprsPerElement)
          while gwvw > kernel["MinGlobalWriteVectorWidth"]:
            futureOccupancy = self.getOccupancy(kernel, \
                self.vgprPool.size() - numVgprAvailable + minElements*numVgprsPerElement)
            if futureOccupancy < currentOccupancy:
              if shrinkDb:
                print "shrink-gwvw-before: gwvw=%u  numVgprsPerElement=%u %s" % (gwvw, numVgprsPerElement, self.kernelName)
              gwvw = gwvw/2
              numVgprsPerElement = numVgprsPerAddr + int(numVgprsPerDataPerVI * gwvw)
              if shrinkDb:
                print "shrink-gwvw-after: gwvw=%u  numVgprsPerElement=%u" % (gwvw, numVgprsPerElement)
            else:
              break  # good enough

          if shrinkDb:
            print "currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
                % (currentOccupancy, futureOccupancy, self.vgprPool.size(), \
                   numVgprAvailable, minElements*numVgprsPerElement)
          if futureOccupancy > currentOccupancy:
            print "warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                  (self.kernelName)
            print "   numVgprAvailable=", numVgprAvailable, \
                  "numVgprsPerElement=", numVgprsPerElement, "atomic=", atomic, \
                  "beta=", beta, "gwvw=", gwvw
          elif gwvw != gwvwOrig:
            if shrinkDb:
              print "info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                  % (self.kernelName, gwvwOrig, gwvw, currentOccupancy)


          if numVgprAvailable < minElements*numVgprsPerElement:
            newVgprs = int(ceil(minElements*numVgprsPerElement))
            if shrinkDb:
              print "info: growing pool += %u\n" % (newVgprs)
              print self.vgprPool.state()
            t = self.vgprPool.checkOut(newVgprs, "grow-pool for GlobalWrite")
            self.vgprPool.checkIn(t)
            numVgprAvailable = self.vgprPool.availableBlock()

        # set atomicW after we potentially resize GWVW
        atomicW = min(gwvw, kernel["VectorAtomicWidth"])

        #print "NumVgprAvailable", numVgprAvailable
        numElementsPerBatch = min(numVgprAvailable / numVgprsPerElement, \
                                  maxElementsPerBatch)
        #print "NumElementsPerBatch", numElementsPerBatch, "LimitedBySgprs", numElementsPerBatchLimitedBySgprs, "WARNING" if numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay"
        if numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
          numElementsPerBatch = numElementsPerBatchLimitedBySgprs

        if kernel["ProblemType"]["DataType"].isHalf():
          # only do an even number of halves
          numElementsPerBatch = int(numElementsPerBatch/2)*2
          assert(numElementsPerBatch > 0)

        # if no atomics and no edge, then write whole vectors
        #if not atomic and not edge:
        #  numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
        #  #print "  NumVectorsPerBatch", numVectorsPerBatch
        #  numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
        numBatches = max(1, (len(elements[edgeI])+numElementsPerBatch-1) / numElementsPerBatch)
        #print "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", numVgprsPerElement
        self.lastCoordOffset1 = -1
        self.coordVgpr1 = -1
        for batchIdx in range(0, numBatches):
          elementStartIdx = batchIdx * numElementsPerBatch
          elementStopIdx = min( elementStartIdx + numElementsPerBatch, len(elements[edgeI]) )
          elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
          numElementsThisBatch = len(elementsThisBatch)
          numElementVgprs = int(numElementsThisBatch * ceil(numVgprsPerElement))
          #print "BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx, numElementVgprs)
          # elementVgprs can be large and should be perfectly tuned to the number of available
          # VGPRS.  We do not want to accidentally overflow and grow the pool here:
          elementVgprs = self.vgprPool.checkOut(numElementVgprs, "elementVgprs", preventOverflow=True)
          kStr += self.globalWriteBatch(kernel, beta, edge, lsu, atomic, gwvw, atomicW, \
              elementsThisBatch, self.coord0, self.coord1, self.addrC, \
              elementVgprs, numVgprsPerElement, numVgprsPerAddr, numVgprsPerDataPerVI, tmpVgpr, \
              fullExecMaskSgpr, elementSgprs, numSgprsPerElement, tmpSgpr)
          self.vgprPool.checkIn(elementVgprs)

        kStr += inst("s_branch", "label_%04u"%endLabel, "jump to end")

    # End label
    kStr += "label_%04u:%s"%(endLabel, self.endLine)
    self.vgprPool.checkIn(tmpVgpr)
    return kStr


  ##############################################################################
  # chooseGlobalLoad :
  # create the store instruction for requested vector width and other parms
  #
  # bpl = bytes per load op
  # rpv = regs per vector
  ##############################################################################
  def chooseGlobalLoad(self, useBuffer, bpl, destVgpr, rpv, \
                       addr0, addr1, offset, extraFields, hi16=0, comment="load C"):
    kStr = ""

    if useBuffer:
      if bpl==2 and hi16:
        kStr += inst("buffer_load_short_d16_hi", vgpr(destVgpr, rpv*2), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, comment)
      elif bpl==2 and not hi16:
        kStr += inst("buffer_load_short_d16", vgpr(destVgpr, rpv*2), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, comment)
      elif bpl==4:
        kStr += inst("buffer_load_dword", vgpr(destVgpr, rpv), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, comment)
      elif bpl==8:
        kStr += inst("buffer_load_dwordx2", vgpr(destVgpr, rpv), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, comment)
      elif bpl==16:
        kStr += inst("buffer_load_dwordx4", vgpr(destVgpr, rpv), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, comment)
      else:
        assert ("chooseGlobalLoad: bad bpl")
    else:
      if bpl==2 and hi16:
        kStr += inst("flat_load_short_d16_hi", vgpr(destVgpr, rpv*2), addr0, extraFields, comment )
      elif bpl==2 and not hi16:
        kStr += inst("flat_load_short_d16", vgpr(destVgpr, rpv*2), addr0, extraFields, comment )
      elif bpl==4:
        kStr += inst("flat_load_dword", vgpr(destVgpr, rpv), addr0, extraFields, comment )
      elif bpl==8:
        kStr += inst("flat_load_dwordx2", vgpr(destVgpr, rpv), addr0, extraFields, comment )
      elif bpl==16:
        kStr += inst("flat_load_dwordx4", vgpr(destVgpr, rpv), addr0, extraFields, comment )
      else:
        assert ("chooseGlobalLoad: bad bpl")

    return kStr


  ##############################################################################
  # chooseGlobalStore
  # create the store instruction for requested vector width and other parms
  #
  # rpv = regs per vector
  ##############################################################################
  def chooseGlobalStore(self, useBuffer, bps, srcVgpr, rpv, \
                        addr0, addr1, offset, extraFields, hi16=0):
    kStr = ""

    if useBuffer:
      if bps==2 and hi16:
        kStr += inst("buffer_store_short_d16_hi", vgpr(srcVgpr, rpv*2), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store C")
      elif bps==2 and not hi16:
        kStr += inst("buffer_store_short", vgpr(srcVgpr, rpv*2), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store C")
      elif bps==4:
        kStr += inst("buffer_store_dword", vgpr(srcVgpr, rpv), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store C")
      elif bps==8:
        kStr += inst("buffer_store_dwordx2", vgpr(srcVgpr, rpv), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store C")
      elif bps==16:
        kStr += inst("buffer_store_dwordx4", vgpr(srcVgpr, rpv), addr0, \
                  addr1, 0, "offen", "offset:%u"%offset, extraFields, "store C")
      else:
        assert ("bad bps")
    else:
      if bps==2 and hi16:
        kStr += inst("flat_store_short_d16_hi", addr0, vgpr(srcVgpr*2), extraFields, "store C" )
      elif bps==2 and not hi16:
        kStr += inst("flat_store_short", addr0, vgpr(srcVgpr, rpv*2), extraFields, "store C" )
      elif bps==4:
        kStr += inst("flat_store_dword", addr0, vgpr(srcVgpr, rpv), extraFields, "store C" )
      elif bps==8:
        kStr += inst("flat_store_dwordx2", addr0, vgpr(srcVgpr, rpv), extraFields, "store C" )
      elif bps==16:
        kStr += inst("flat_store_dwordx4", addr0, vgpr(srcVgpr, rpv), extraFields, "store C" )
      else:
         assert ("bad bps")

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
    elif kernel["ProblemType"]["DataType"].isSingle():
      kStr += inst("v_add_f32", \
                dst, src0, src1, \
                comment)
    else:
      assert(0) # no double GSU yet

    return kStr


##############################################################################
  # Global Write Batch
  # numVgprsPerDataPerVI : Uses bpeCinternal
  ##############################################################################
  def globalWriteBatch(self, kernel, beta, edge, lsu, atomic, gwvw, atomicW, \
      batchElements, coord0, coord1, addrC,  \
      batchElementVgprs, numVgprsPerElement, numVgprsPerAddr, numVgprsPerDataPerVI, tmpVgpr, \
      fullExecMaskSgpr, batchElementSgprs, numSgprsPerElement, tmpSgpr):
    kStr = ""

    if atomic:
      # all kinds of code relies on this assumption:
      assert(atomicW <= gwvw)
      if kernel["ProblemType"]["DataType"].isHalf():
        assert(atomicW >= 2)

    # comment
    commentStr = "Global Write%s%s Batch:" \
        % (" Beta" if beta else "", " Edge" if edge else "")
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      commentStr += "(%u,%u,%u,%u:vw%u%s)" % \
        (element[0], element[1], element[2], element[3], gwvw, 
         ":vaw:%u"%atomicW if atomic else "")
      if elementIdx < len(batchElements)-1:
        commentStr += "; "
    kStr += self.comment3(commentStr)

    ########################################
    # allocate per-element resources
    #numVgprsPerData = numVgprsPerElement - numVgprsPerAddr # might be decimal for half
    addrVgprOffset = 0
    dataVgprOffset = addrVgprOffset + numVgprsPerAddr*len(batchElements)
    elementAddr = []
    elementData = []
    elementMask = []
    elementSumIdx = []
    for elementIdx in range(0, len(batchElements)):
      # gpr assignments for element
      addr = batchElementVgprs + addrVgprOffset + elementIdx*numVgprsPerAddr # elementVgprs+0
      elementAddr.append(addr)
      # if numVgprsPerDataPerVI == 0.5, then two consecutive elements
      # should have same data pointer, next should move.

      data = batchElementVgprs + dataVgprOffset + int(elementIdx*numVgprsPerDataPerVI*gwvw) # elementVgprs+self.rpga
      elementData.append(data)
      mask = batchElementSgprs + elementIdx * numSgprsPerElement # elementSgprs+0
      elementMask.append(mask)

      element = batchElements[elementIdx]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]
      #print "Edge=", edge, element
      if lsu:
        sumIdx = self.startVgprValuC + vc0 + d1*kernel["VectorWidth"]
      else:
        sumIdx = self.startVgprValuC + vc0 + d0*kernel["VectorWidth"] + vc1*kernel["ThreadTile0"] + d1*kernel["VectorWidth"]*kernel["ThreadTile0"]
      elementSumIdx.append(sumIdx) # sumIdx is an element idx, need to div/2 for half

    tmpS01 = tmpSgpr # scratch sgprs
    tmpS23 = tmpS01+2

    ########################################
    # calculate addr and masks
    kStr += self.comment("calc coords, apply mask, and issue loads (if necessary)")
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      addr = elementAddr[elementIdx]
      data = elementData[elementIdx]
      mask = elementMask[elementIdx]
      sumIdx = elementSumIdx[elementIdx]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]

      #d0 always equals 0 for lsu
      #strideD0 = 0 # never used for lsu
      strideD1 = (kernel["NumThreads"]*kernel["VectorWidth"]/kernel["MacroTile0"]) if lsu else (kernel["SubGroup1"]*kernel["VectorWidth"])
      #fullExecMaskSgpr = ((self.startSgprSizesSum+1)/2)*2 # even sgpr

      # Compute scaled offset requires 2 SGPR

      scaledCoordVgpr1 = tmpVgpr+2
      coordOffset0 = d0 * kernel["SubGroup0"]*kernel["VectorWidth"] + vc0
      if coordOffset0 == 0:
        # just use coord0 directly
        coordVgpr0 = coord0
        kStr += self.comment1("coordOffset=0, use coordVgpr0=v%u directly"%coordVgpr0)
      elif coordOffset0 <= 64:
        # coordOffset0 fits in instruction:
        kStr += inst("_v_add_co_u32", vgpr(tmpVgpr+0), "vcc", vgpr(coord0), coordOffset0, \
            "coord0 += d0*sg0*VW + vc0")
        coordVgpr0 = tmpVgpr+0
      else:
        kStr += inst("s_mov_b32", sgpr(tmpS01), coordOffset0, "coord0Offset d0=%u vc0=%u"%(d0, vc0))
        kStr += inst("_v_add_co_u32", vgpr(tmpVgpr+0), "vcc", vgpr(coord0), sgpr(tmpS01), \
            "coord0 += d0*sg0*VW + vc0")
        coordVgpr0 = tmpVgpr+0

      # coord1
      # coord0
      coordOffset1 = d1*strideD1 + vc1
      if coordOffset1 != self.lastCoordOffset1:
        kStr += self.comment1("new offset1=%u: d1=%u vc1=%u" % (coordOffset1, d1, vc1))
        self.lastCoordOffset1 = coordOffset1

        if coordOffset1 == 0:
          # just use coord0 directly
          self.coordVgpr1 = coord1
          kStr += self.comment1("coordOffset1=0, use coordVgpr1=v%u directly"%self.coordVgpr1)
        elif coordOffset1 <= 64:
          # coordOffset1 fits in instruction:
          kStr += inst("_v_add_co_u32", vgpr(tmpVgpr+1), "vcc", vgpr(coord1), coordOffset1, \
              "coord1 += d1*sg1*VW + vc1")
          self.coordVgpr1 = tmpVgpr+1
        else:
          kStr += inst("s_mov_b32", sgpr(tmpS01), coordOffset1, "coordOffset1 d1=%u vc1=%u"%(d0, vc0))
          kStr += inst("_v_add_co_u32", vgpr(tmpVgpr+1), "vcc", vgpr(coord1), sgpr(tmpS01), \
              "coord1 += d1*sg1*VW + vc1")
          self.coordVgpr1 = tmpVgpr+1

        # in tmpVgpr+2, save the scaled address:
        indices = range(0, kernel["ProblemType"]["NumIndicesC"])
        numDim = len(indices)
        for i in range(1, numDim):
          assert( indices[i] < kernel["ProblemType"]["NumIndicesC"])
          coord = -1
          if i == kernel["ProblemType"]["Index0"]:
            assert(0) # Should never get here?
            coord = vgpr(coordVgpr0)
            useSgpr = 0
          elif i == kernel["ProblemType"]["Index1"]:
            coord = vgpr(self.coordVgpr1)
            useSgpr = 0
          else: # just a group index
            coord = sgpr("WorkGroup%u"%i)
            useSgpr = 1

          if useSgpr:
            kStr += inst("v_mov_b32", \
                vgpr(tmpVgpr+3), \
                coord, "vgpr <- sgpr")
            kStr += inst("v_mul_lo_u32", \
                vgpr(tmpVgpr+3), \
                sgpr("StridesC+%u"%(i-1)), \
                vgpr(tmpVgpr+3), \
                "Coffset %u "%i)
          else:
            kStr += inst("v_mul_lo_u32", \
                vgpr(scaledCoordVgpr1), \
                sgpr("StridesC+%u"%(i-1)), \
                coord, \
                "Coffset %u "%i)

          if i>1:
            kStr += inst("_v_add_co_u32", \
                vgpr(scaledCoordVgpr1), \
                "vcc", \
                vgpr(scaledCoordVgpr1), \
                vgpr(tmpVgpr+3), \
                "accumulate d%u into addr"%i)

      # in-bounds exec mask
      if kernel["BufferStore"]:
        kStr += inst("_v_add_lshl_u32", \
            vgpr(addr), \
            vgpr(scaledCoordVgpr1), \
            vgpr(coordVgpr0), \
            hex(log2(self.bpeCexternal)), \
            "accumulate d0 lower and *= bpe into addr")
        if edge:
          kStr += inst("v_cmp_lt_u32",  sgpr(tmpS01,2), vgpr(coordVgpr0), sgpr("SizesFree+0"), "coord0 < size0" )
          kStr += inst("v_cmp_lt_u32",  sgpr(tmpS23,2), vgpr(self.coordVgpr1), sgpr("SizesFree+1"), "coord1 < size1" )
          kStr += inst("s_and_b64",  sgpr(mask,2), sgpr(tmpS01,2), sgpr(tmpS23,2), "in0 && in1" )
          kStr += inst("v_cndmask_b32", \
                       vgpr(addr), \
                        -1,
                       vgpr(addr), \
                       sgpr(mask,2), \
                        "clip if OOB. offset")
      else:
        if edge:
          kStr += inst("v_cmp_lt_u32",  sgpr(tmpS01,2), vgpr(coordVgpr0), sgpr("SizesFree+0"), "coord0 < size0" )
          kStr += inst("v_cmp_lt_u32",  sgpr(tmpS23,2), vgpr(self.coordVgpr1), sgpr("SizesFree+1"), "coord1 < size1" )
          kStr += inst("s_and_b64",  sgpr(mask,2), sgpr(tmpS01,2), sgpr(tmpS23,2), "in0 && in1" )

          if (beta or atomic):
            kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )  # fixme, do we need this for atomic? BOZO

        # global offset macro (requires 3 tmpVgpr)
        # final address = C + index*bytes
        kStr += "GLOBAL_OFFSET_C %u" % addr
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          if i == kernel["ProblemType"]["Index0"]:
            kStr += ", %s" % (coordVgpr0)
          elif i == kernel["ProblemType"]["Index1"]:
            kStr += ", %s" % (self.coordVgpr1)
          else: # just a group index
            kStr += ", sgprWorkGroup%u"%i
        kStr += ", %s%s" % ((tmpVgpr+2), self.endLine)

      if not kernel["BufferStore"]:
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
        for avi in range(0, gwvw/atomicW):
          # TODO: use chooseGlobalLoad, could use vector loads here too perhaps:
          dataV = elementData[elementIdx] + int(avi*numVgprsPerDataPerVI)
          bpm = self.bpeCexternal * atomicW
          rpv = float(bpm)/4
          useBuffer = kernel["BufferStore"]
          if kernel["BufferStore"]: # yes, BufferStore here - use same addressing regs for this load
            addr0 = vgpr(addr)
            addr1 = sgpr("SrdC", 4)
          else:
            addr0 = vgpr(addr,2)
            addr1 = ""
          kStr += self.chooseGlobalLoad(useBuffer, bpm, dataV+1, rpv, \
                    addr0, addr1, offset=avi*bpm, extraFields="",
                    comment="load C (atomic) bpm=%u vaw=%u"%(bpm,atomicW))
          #  kStr += inst("buffer_load_dword", vgpr(dataV+1), vgpr(addr), \
          #            sgpr("SrdC", 4), 0, "offen", "offset:%u"%(vi*bps), "load C (atomic) vi=%u"%vi)
      elif beta:
        bps = kernel["ProblemType"]["DataType"].numBytes() * gwvw
        rpv = kernel["ProblemType"]["DataType"].numRegisters() * gwvw
        useBuffer = kernel["BufferStore"]
        if kernel["BufferStore"]:
          addr0 = vgpr(addr)
          addr1 = sgpr("SrdC", 4)
        else:
          addr0 = vgpr(addr,2)
          addr1 = ""
        extraFields = ""
        kStr += self.comment("beta loads")
        if kernel["ProblemType"]["DataType"].isHalf():
          kStr += self.chooseGlobalLoad(useBuffer, bps, data, rpv, \
                    addr0, addr1, 0, extraFields, hi16=sumIdx%2)
        elif kernel["ProblemType"]["DataType"].isSingle() or \
             kernel["ProblemType"]["DataType"].isDouble():
          kStr += self.chooseGlobalLoad(useBuffer, bps, data, rpv, \
                    addr0, addr1, 0, extraFields)

      # restore full exec mask for calculating addr of next element
      if not kernel["BufferStore"] and edge and (beta or atomic):
        #kStr += inst("s_or_saveexec_b64",  sgpr(tmpS45,2), sgpr(fullExecMaskSgpr,2), "full mask -> exec" )
        kStr += inst("s_mov_b64", "exec", sgpr(fullExecMaskSgpr,2), "full mask -> exec" )

    ########################################
    # rC *= alpha
    if self.do["ApplyAlpha"]:
      kStr += self.comment("rC *= alpha batchEements=%s"%batchElements)
      for elementIdx in range(0, len(batchElements)):
        for vi in range(0, gwvw):
          sumIdxV = elementSumIdx[elementIdx] + vi
          if kernel["ProblemType"]["DataType"].isHalf():
            if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
              if sumIdxV%2:
                kStr += inst("v_pk_mul_f16", vgpr(sumIdxV/2), vgpr(self.alphaVgpr), vgpr(sumIdxV/2), "*= alpha sumIdx=%u vi=%u"%(elementSumIdx[elementIdx], vi))
            else: # HPA
              kStr += inst("v_mul_f32", vgpr(sumIdxV), vgpr(self.alphaVgpr), vgpr(sumIdxV), "*= alpha")

          elif kernel["ProblemType"]["DataType"].isSingle():
            kStr += inst("v_mul_f32", vgpr(sumIdxV), sgpr("Alpha"), vgpr(sumIdxV), "*= alpha" )
          elif kernel["ProblemType"]["DataType"].isDouble():
            kStr += inst("v_mul_f64", vgpr(sumIdxV*2,2), sgpr("Alpha",2), vgpr(sumIdxV*2,2), "*= alpha")

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
      label = self.getLabel(labelString)
      labelString += "EarlyExit"
      labelAfterAtomicLoop = self.getLabel(labelString)

      ########################################
      # wait for batched load
      # TODO - we are always atomic here?
      assert(beta or atomic) # bozo, remove this assert
      kStr += inst("s_waitcnt", "vmcnt(0)", "wait C" )

      ########################################
      # first attempt write
      kStr += self.comment("issue first atomic writes")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = elementAddr[elementIdx]
        mask = elementMask[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]

        # apply in-bounds exec mask
        if edge:
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec (before atomic)" )

        for avi in range(0, gwvw/atomicW):
          dataV = elementData[elementIdx] + int(avi*numVgprsPerDataPerVI)
          sumIdxV = elementSumIdx[elementIdx] + avi
          if kernel["ProblemType"]["DataType"].isHalf():  sumIdxV /= 2
          # for atomic, data[1] = original c, data[0] = new c
          kStr += self.chooseAddForAtomic(kernel, \
                    vgpr(dataV+0), vgpr(dataV+1), vgpr(sumIdxV), \
                    "desired value avi=%u"%avi)

          # attempt write
          atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
          if self.do["GlobalWrite"]:
            bps = kernel["ProblemType"]["DataType"].numBytes()
            if kernel["BufferStore"]:
              kStr += "buffer_atomic_cmpswap %s, %s, %s %s    // %s%s" % \
                  (vgpr(dataV,2), \
                   vgpr(addr,1), \
                   sgpr("SrdC", 4),  \
                   "0 offen offset:%u glc" % (avi*bps), \
                   "attempt write avi=%u"%(avi), self.endLine )
            else:
              kStr += "flat_atomic_cmpswap %s, %s, %s %s    // %s%s" % \
                  (vgpr(atomicDestVgpr), vgpr(addr,2), \
                  vgpr(dataV,2), "glc", "attempt write", self.endLine )
          else:
             kStr += inst("v_mov_b32", vgpr(atomicDestVgpr), vgpr(dataV+1), "Fake successful CAS" )
             # Fake successful CAS swap:

      ########################################
      # wait for first attempt write
      kStr += inst("s_waitcnt vmcnt(0)", "wait for atomic writes" )

      ########################################
      # check first attempt
      kStr += self.comment("check success of writes, update masks")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = elementAddr[elementIdx]
        mask = elementMask[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]

        # calculate new masks
        if edge:
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )
          for avi in range(0, gwvw/atomicW):
            dataV = elementData[elementIdx] + int(avi*numVgprsPerDataPerVI)
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

          kStr += inst("s_and_b64",  sgpr(mask,2), sgpr(tmpS01,2), sgpr(mask,2), "inBounds & must try again" )
        else:
          for avi in range(0, gwvw/atomicW):
            dataV = elementData[elementIdx] + int(avi*numVgprsPerDataPerVI)
            atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
            #kStr += inst("s_mov_b64", sgpr(mask,2), sgpr(fullExecMaskSgpr,2), "mask = full" )
            kStr += inst("v_cmp_ne_u32", sgpr(mask,2), vgpr(atomicDestVgpr), \
                vgpr(dataV+1), "c read during atomic != c read during prior load" )

      # or masks together to check early exit
      kStr += self.comment("or masks to check for exit")
      kStr += inst("s_mov_b64", sgpr(tmpS01,2), hex(0), "empty mask" )
      for elementIdx in range(0, len(batchElements)):
        mask = elementMask[elementIdx]
        kStr += inst("s_or_b64", sgpr(tmpS01,2), sgpr(mask,2), sgpr(tmpS01,2), "or to add threads" )
      kStr += inst("s_or_saveexec_b64", sgpr(tmpS23,2), sgpr(tmpS01,2), "apply combined mask" )
      kStr += inst("s_cbranch_execz", "label_%04u" % labelAfterAtomicLoop, "if exec is zero skip loop" )

      # begin atomic loop
      kStr += self.comment("atomic CAS loop")
      kStr += "label_%04u:%s" % (label, self.endLine)

      kStr += self.comment("apply updated masks and issue writes again")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = elementAddr[elementIdx]
        mask = elementMask[elementIdx]
        bps = kernel["ProblemType"]["DataType"].numBytes()

        for avi in range(0, gwvw/atomicW):
          dataV = elementData[elementIdx] + int(avi*numVgprsPerDataPerVI)
          atomicDestVgpr = dataV if kernel["BufferStore"] else dataV+2
          sumIdxV = elementSumIdx[elementIdx] + avi
          if kernel["ProblemType"]["DataType"].isHalf():  sumIdxV /= 2

          # apply mask for element
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "must try again" )
          kStr += inst("v_mov_b32", vgpr(dataV+1), vgpr(atomicDestVgpr), "dataV+1 = tmp (new original C)" )
          kStr += self.chooseAddForAtomic(kernel, \
                    vgpr(dataV+0), vgpr(dataV+1), vgpr(sumIdxV), \
                    "newC = rC + originalC")
          if self.do["GlobalWrite"]:
            if kernel["BufferStore"]:
              # Using no-ret version here?
              kStr += "buffer_atomic_cmpswap %s, %s, %s %s    // %s%s" % \
                  (vgpr(dataV,2), \
                   vgpr(addr,1), \
                   sgpr("SrdC", 4), \
                   "0 offen offset:%u glc" % (avi*bps), \
                   "try again", self.endLine )
            else:
              kStr += "flat_atomic_cmpswap %s, %s, %s %s    // %s%s" % ( vgpr(atomicDestVgpr), \
                  vgpr(addr,2), vgpr(dataV,2), "glc", "try again", self.endLine)

      # wait for batched write
      kStr += inst("s_waitcnt vmcnt(0)", "wait for atomic writes" )

      # check batched write success
      kStr += self.comment("apply masks and check for success")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = elementAddr[elementIdx]
        data = elementData[elementIdx]
        mask = elementMask[elementIdx]
        for avi in range(0, gwvw/atomicW):
          dataV = elementData[elementIdx] + int(avi*numVgprsPerDataPerVI)
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
        mask = elementMask[elementIdx]
        kStr += inst("s_or_b64", sgpr(tmpS01,2), sgpr(mask,2), sgpr(tmpS01,2), "or to add threads" )

      # apply combined masks and exit
      kStr += inst("s_or_saveexec_b64", sgpr(tmpS23,2), sgpr(tmpS01,2), "apply combined mask" )
      kStr += inst("s_cbranch_execnz", "label_%04u" % label, "try again if not complete" )
      kStr += "label_%04u:%s" % (labelAfterAtomicLoop, self.endLine)
      kStr += inst("s_mov_b64", "exec", sgpr(fullExecMaskSgpr,2), "full mask -> exec" )

      #  kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "apply new mask" )
      #  #kStr += inst("s_and_saveexec_b64", sgpr(tmpS45,2), "vcc", "apply new mask" )
      #  kStr += inst("s_cbranch_execnz", "label_%04u" % labelIdx, "try again if not complete" )
      #  kStr += inst("s_mov_b64", "exec", sgpr(fullExecMaskSgpr,2), "full mask -> exec" )


    ########################################
    # Not Atomic
    ########################################
    else:

      ########################################
      # wait for batched load
      if beta: # FIXME can this be moved to below or do flat instructions return out of order
        kStr += inst("s_waitcnt", "vmcnt(0)", "wait C" )

      kStr += self.comment("apply mask, calc new C and issue write")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = elementAddr[elementIdx]
        mask = elementMask[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]
        sumIdx = elementSumIdx[elementIdx]

        #if beta: # FIXME kept above since flat instruction may return out of order
        #  kStr += inst("s_waitcnt", "vmcnt(%u)"%(len(batchElements)-1), "wait C")

        # apply in-bounds exec mask
        if edge and not kernel["BufferStore"]:
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )

        if beta:
          # if GWVW=1 the half path still assumes we have
          # at least two stores so does some combining across VI -
          # for example assuming we can have two elements and can use pk_mul
          # here:
          for vi in range(0, gwvw):
            dataV = elementData[elementIdx] + int(vi*numVgprsPerDataPerVI)
            sumIdxV = elementSumIdx[elementIdx] + vi
            if kernel["ProblemType"]["DataType"].isHalf():
              if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
                if sumIdxV%2==0:
                  # dataV+0 = new c = old c*beta
                  kStr += inst("v_pk_mul_f16", vgpr(dataV), vgpr(self.betaVgpr), vgpr(dataV+0), \
                      "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
                  # dataV+0 = new c = old c*beta + rC
                  kStr += inst("v_pk_add_f16", vgpr(sumIdxV/2), vgpr(dataV), vgpr(sumIdxV/2), \
                      "sum*alpha + C*beta")
                else:
                  pass # add will have been done previously
              else: # HPA
                # dataV+0 = new c = old c*beta + rC
                # src0 = beta = f32 = opsel 00
                # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
                # src2 = sumIdxV = f32 = opsel 00
                dataCExternal = elementData[elementIdx] + vi/2
                hi16 = sumIdxV%2
                kStr += inst("v_mad_mix_f32", vgpr(sumIdxV), vgpr(self.betaVgpr), \
                    vgpr(dataCExternal), vgpr(sumIdxV), \
                    "op_sel:[0,%u,0] op_sel_hi:[0,1,0]" % (hi16), \
                    "//C*=beta")
            elif kernel["ProblemType"]["DataType"].isSingle():
              kStr += inst("v_mac_f32", vgpr(sumIdxV), vgpr(dataV+0), sgpr("Beta"), \
                  "finalSum = sum*alpha + C*beta")

            elif kernel["ProblemType"]["DataType"].isDouble():
              # dataV+0 = new c = old c*beta
              kStr += inst("v_fma_f64", vgpr(sumIdxV*2,2), vgpr(dataV+0,2), sgpr("Beta",2), vgpr(sumIdxV*2,2), \
                  "finalSum = sum*alpha + C*beta")


       # pack stores:
        for vi in range(0, gwvw):
          dataV = elementData[elementIdx] + int(vi*numVgprsPerDataPerVI)
          sumIdxV = elementSumIdx[elementIdx] + vi
          if kernel["ProblemType"]["DataType"].isHalf():
            if kernel["ProblemType"]["HighPrecisionAccumulate"]:
              kStr += inst("v_cvt_f16_f32", vgpr(sumIdxV), vgpr(sumIdxV), "convert C to fp16" )
              if vi%2 == 1:
                assert (gwvw % 2 == 0)
                d = elementSumIdx[elementIdx] + vi/2
                kStr += inst("v_pack_b32_f16", vgpr(d), vgpr(sumIdxV-1), vgpr(sumIdxV), "Pack with neighbor" )

        if self.do["GlobalWrite"]:
          # perform vector stores here, so no VI indexing.
          # if GWVW > Vw, might need to support loops to 
          # implement wider stores
          ntStr = ""
          if kernel["NonTemporalC"]%2==1:
            ntStr += " glc"
          if kernel["NonTemporalC"]/2==1:
            ntStr += " slc"

          bps = kernel["ProblemType"]["DataType"].numBytes() * gwvw
          rpv = kernel["ProblemType"]["DataType"].numRegisters() * gwvw
          if kernel["BufferStore"]:
            addr0 = vgpr(addr)
            addr1 = sgpr("SrdC", 4)
          else:
            addr0 = vgpr(addr,2)
            addr1 = ""

          useBuffer = kernel["BufferStore"]

          if kernel["ProblemType"]["DataType"].isHalf():
            if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
              kStr += self.chooseGlobalStore(useBuffer, bps, sumIdx/2, rpv, \
                        addr0, addr1, 0, ntStr, hi16=sumIdx%2)
            else:
              kStr += self.chooseGlobalStore(useBuffer, bps, sumIdx, rpv, \
                        addr0, addr1, 0, ntStr, hi16=0)
          elif kernel["ProblemType"]["DataType"].isSingle():
            kStr += self.chooseGlobalStore(useBuffer, bps, sumIdx, rpv, \
                      addr0, addr1, 0, ntStr)
          elif kernel["ProblemType"]["DataType"].isDouble():
            kStr += self.chooseGlobalStore(useBuffer, bps, sumIdx*2, rpv, \
                      addr0, addr1, 0, ntStr)

          #kStr += self.bomb(5)

      if edge: # subsequent batch must start with full exec mask
        kStr += inst("s_mov_b64", "exec", sgpr(fullExecMaskSgpr,2), "full mask -> exec" )

    return kStr

  ##############################################################################
  # Function End
  ##############################################################################
  def functionEnd(self, kernel):
    return inst("s_endpgm", "End Kernel")

  ##############################################################################
  # Function Suffix
  ##############################################################################
  def functionSuffix(self, kernel):
    kStr = ""
    if self.vgprPool.size() > self.maxVgprs or \
       self.sgprPool.size() > self.maxSgprs:
      self.overflowedResources = True
    if self.overflowedResources:
      kStr += ".endif // too many gprs\n"

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
               else tPA["nrp"]*tPA["nrc"]*max(tPA["nwcv"],tPA["nwpv"])/tPA["nwcvpi"]
        numB = 0 if kernel["DirectToLdsB"] \
               else tPB["nrp"]*tPB["nrc"]*max(tPB["nwcv"],tPB["nwpv"])/tPB["nwcvpi"]
        lgkmcnt += skipLocalWrite * (numA + numB)
      if skipLocalRead > -1:
        numA = kernel["InnerUnroll"]*(kernel["ThreadTile0"] / kernel["VectorWidth"]) \
            / self.localReadInstructionA.numOffsets
        numB = kernel["InnerUnroll"]*(kernel["ThreadTile1"] / kernel["VectorWidth"]) \
            / self.localReadInstructionB.numOffsets
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
        dbKStr = ""
        dbKStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "debug %s"%comment )
        dbKStr += inst("s_barrier", "debug" )
        return dbKStr

    lgkmcnt = min(lgkmcnt, 15)
    vmcnt = min(vmcnt, 15)

    kStr = ""
    kStr += "s_waitcnt "
    if lgkmcnt >= 0:
      kStr += "lgkmcnt(%u)"%lgkmcnt
    #if lgkmcnt >= 0 and vmcnt >= 0:
    #  kStr += " & "
    elif vmcnt >= 0:
      kStr += "vmcnt(%u)"%vmcnt
    kStr += " // %s" % comment
    kStr += self.endLine
    return kStr

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def syncThreads(self, kernel, comment=""):
    if kernel["NumThreads"] > 64 and self.do["Sync"]:
      return self.indent + self.syncStr + " //" + comment + self.endLine
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
      kStr += inst("s_barrier", "dump LDS" )
      tmp = self.vgprPool.checkOut(1)
      tmpAddr = self.vgprPool.checkOut(1)
#jgolds which bpe should we use?
      kStr += inst("v_lshlrev_b32", \
          vgpr(tmpAddr), \
          hex(log2(self.bpeAB)), \
          vgpr("Serial"), \
          "dump lds")
      for i in range(startU, startU+numU):
        kStr += inst("ds_read_b32", vgpr(tmp), \
            vgpr(tmpAddr) + " offset:%u"%(i*kernel["NumThreads"]*4), "dump lds")
        kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "dump" )
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
    kStr += inst("s_barrier", "init LDS" )
    tmp = self.vgprPool.checkOut(1)
    tmpAddr = self.vgprPool.checkOut(1)
    kStr += inst("v_mov_b32", vgpr(tmp), hex(value), "Init value")
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    writesPerThread = ((kernel["LdsNumElements"]*numBytesPerElement-1)/kernel["NumThreads"]/4) + 1
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
    self.vgprPool.checkIn(tmp)
    self.vgprPool.checkIn(tmpAddr)
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


  ##############################################################################
  # Cause a GPUVM fault.  
  # Instruction after the bomb will write the cookie to SGPR0, so you can see the cookie in the 
  # backtrace. Useful for locating which spot in code generated the bomb
  # vgprAddr controls which vgpr to overwrite with the null pointer address
  ##############################################################################
  def bomb(self,cookie=None,scratchVgpr=-1):
      kStr =""
      if scratchVgpr==-1:
        vgprAddr = self.vgprPool.checkOut(2)
      else:
        vgprAddr = scratchVgpr
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
  def assertCmpCommon(self, c, val0, val1, cookie=-1):
    kStr = ""
    if self.db["EnableAsserts"]:
      kStr += inst("s_or_saveexec_b64", sgpr("SaveExecMask",2), 0, \
          "assert: saved execmask")
      kStr += inst("v_cmpx_%s_u32"%c, "vcc", val0, val1, "v_cmp" )

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
    return self.assertCmpCommon("ne", val0, val1, cookie)

  def assert_ne(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("eq", val0, val1, cookie)

  def assert_lt(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("ge", val0, val1, cookie)

  def assert_gt(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("le", val0, val1, cookie)

  def assert_le(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("gt", val0, val1, cookie)

  def assert_ge(self, val0, val1, cookie=-1):
    return self.assertCmpCommon("lt", val0, val1, cookie)

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

  # Assert that all bits in vcc are true, or assert/bomb otherwise
  def assert_vcc_true(self, cookie=-1):
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
  def assert_vcc_false(self, cookie=-1):
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
  params = args[0:len(args)-1]
  comment = args[len(args)-1]
  formatting = "%s"
  if len(params) > 1:
    formatting += " %s"
  for i in range(0, len(params)-2):
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
    magic = ((2**shift) / divisor) + 1
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(magic), "")
    kStr += inst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(shift), "")
    kStr += inst("v_lshrrev_b64", vgpr(tmpVgpr,2), sgpr(tmpSgpr), vgpr(tmpVgpr,2), "")
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
    magic = ((2**shift) / divisor) + 1
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
def scalarStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr, \
    doRemainder=True):

  assert (qReg != tmpSgpr)

  kStr = ""
  if ((divisor & (divisor - 1)) == 0): # pow of 2
    divisor_log2 = log2(divisor)
    kStr += inst("s_lshr_b32", sgpr(qReg), sgpr(dReg), divisor_log2, \
        "%s = %s / %u"%(sgpr(qReg), sgpr(dReg), divisor) )
    if doRemainder:
      kStr += inst("s_and_b32", sgpr(rReg), (divisor-1), sgpr(dReg), \
          "%s = %s %% %u"%(sgpr(rReg), sgpr(dReg), divisor) )
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
    magic = ((2**shift) / divisor) + 1
    magicHi = magic / (2**16)
    magicLo = magic & (2**16-1)

    kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), hex(0), "STATIC_DIV: divisior=%s"%divisor)
    kStr += inst("s_mul_i32", sgpr(tmpSgpr+0), hex(magicHi), sgpr(dReg), "tmp1 = dividend * magic hi")
    kStr += inst("s_lshl_b64", sgpr(tmpSgpr,2), sgpr(tmpSgpr,2), hex(16), "left shift 16 bits")
    kStr += inst("s_mul_i32", sgpr(qReg), sgpr(dReg), hex(magicLo), "tmp0 = dividend * magic lo")
    kStr += inst("s_add_u32", sgpr(tmpSgpr+0), sgpr(qReg), sgpr(tmpSgpr+0), "add lo")
    kStr += inst("s_addc_u32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), hex(0), "add hi")
    kStr += inst("s_lshr_b64", sgpr(tmpSgpr,2), sgpr(tmpSgpr,2), hex(shift), "tmp1 = (dividend * magic) << shift")
    kStr += inst("s_mov_b32", sgpr(qReg), sgpr(tmpSgpr), "quotient")
    if doRemainder:
      kStr += inst("s_mul_i32", sgpr(tmpSgpr), sgpr(qReg), hex(divisor), "quotient*divisor")
      kStr += inst("s_sub_u32", sgpr(rReg), sgpr(dReg), sgpr(tmpSgpr), "rReg = dividend - quotient*divisor")
  return kStr

########################################
# Multiply
# product register, operand register, multiplier
########################################
def staticMultiply(product, operand, multiplier, tmpSgpr=None):
  if ((multiplier & (multiplier - 1)) == 0): # pow of 2
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


