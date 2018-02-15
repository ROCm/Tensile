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

  ########################################
  # Init
  def __init__(self, size):
    self.pool = [self.statusUnAvailable]*size
    self.checkOutSize = {}

  ########################################
  # Add
  def add(self, start, size):
    # reserve space
    newSize = start + size
    oldSize = len(self.pool)
    if newSize > oldSize:
      for i in range(0, newSize-oldSize):
        self.pool.append(self.statusUnAvailable)
    # mark as available
    for i in range(start, start+size):
      if self.pool[i] == self.statusUnAvailable:
        self.pool[i] = self.statusAvailable
      elif self.pool[i] == self.statusAvailable:
        printWarning("RegisterPool::add(%u,%u) pool[%u] already available" % (start, size, i))
      elif self.pool[i] == self.statusInUse:
        printWarning("RegisterPool::add(%u,%u) pool[%u] already in use" % (start, size, i))
      else:
        printExit("RegisterPool::add(%u,%u) pool[%u] = %s" % (start, size, self.pool[i]))

  ########################################
  # Remove
  def remove(self, start, size):
    #print "RP::remove(%u,%u)"%(start,size)
    # reserve space
    newSize = start + size
    oldSize = len(self.pool)
    if newSize > oldSize:
      printWarning("RegisterPool::remove(%u,%u) but poolSize=%u" % (start, size, oldSize))
    # mark as unavailable
    for i in range(start, start+size):
      if  self.pool[i] == self.statusAvailable:
        self.pool[i] = self.statusUnAvailable
      elif self.pool[i] == self.statusUnAvailable:
        printWarning("RegisterPool::remove(%u,%u) pool[%u] already unavailable" % (start, size, i))
      elif  self.pool[i] == self.statusInUse:
        printWarning("RegisterPool::remove(%u,%u) pool[%u] still in use" % (start, size, i))
      else:
        printExit("RegisterPool::remove(%u,%u) pool[%u] = %s" % (start, size, self.pool[i]))

  ########################################
  # Check Out
  def checkOut(self, size):
    return self.checkOutAligned(size, 1)
  def checkOutAligned(self, size, alignment):
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
        if self.pool[i+j] != self.statusAvailable:
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
        self.pool[i] = self.statusInUse
      self.checkOutSize[found] = size
      #print "RP::checkOut(%u,%u) @ %u"%(size,alignment, found)
      return found
    # need overflow
    else:
      #print "RegisterPool::checkOutAligned(%u,%u) overflowing past %u" % (size, alignment, len(self.pool))
      # where does tail sequence of available registers begin
      start = len(self.pool)
      for i in range(len(self.pool)-1, 0, -1):
        if self.pool[i] == self.statusAvailable:
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
        self.pool[i] = self.statusInUse
      for i in range(0, overflow):
        self.pool.append(self.statusInUse)
      self.checkOutSize[start] = size
      #print "RP::checkOut(%u,%u) @ %u (overflow)"%(size, alignment, start)
      return start

  ########################################
  # Check In
  def checkIn(self, start):
    #print "RP::checkIn() @ %u"%(start)
    if start in self.checkOutSize:
      size = self.checkOutSize[start]
      for i in range(start, start+size):
        self.pool[i] = self.statusAvailable
      self.checkOutSize.pop(start)
    else:
      printWarning("RegisterPool::checkIn(%u) but it was never checked out"%start)

  ########################################
  # Size
  def size(self):
    return len(self.pool)

  ########################################
  # Size
  def available(self):
    numAvailable = 0
    for s in self.pool:
      if s == self.statusAvailable:
        numAvailable += 1
    return numAvailable

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
      if self.pool[i] == self.statusUnAvailable:
        stateStr += "."
      elif self.pool[i] == self.statusAvailable:
        stateStr += "|"
      elif self.pool[i] == self.statusInUse:
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

    self.do["PreLoop"]    = True
    self.do["GlobalRead"] = True
    self.do["GlobalInc"]  = True
    self.do["LocalWrite"] = True
    self.do["LocalRead"]  = True
    self.do["Wait"]       = True
    self.do["Sync"]       = True
    self.do["MAC"]        = True
    self.do["PostLoop"]   = True

    self.AsmBugs = {}
    self.AsmBugs["ExplicitCO"] = True # New assembler require explicit reference to CO (carry-out)

    # ISA version, such as 803
    self.version = globalParameters["CurrentISA"]
    self.maxVgprs = 256

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


  ########################################
  # Get Label
  def getLabel(self, name):
    if name not in self.labels:
      self.labels[name] = len(self.labels)
    return self.labels[name]

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

  def getTmpSgpr(self, num):
    if num==1:
      return self.startSgprTmpPool
    else:
      return ((self.startSgprTmpPool+1)/2)*2

  ##############################################################################
  #
  #   Functions to Write Kernel Segments
  #
  ##############################################################################

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def initKernel(self, kernel, tPA, tPB ):
    super(KernelWriterAssembly, self).initKernel(kernel, tPA, tPB)

    # True=slightly fewer [v_mov] instructions but extra registers
    self.globalReadIncsUseVgpr = False if kernel["BufferLoad"] else True

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
        "%s, %s, %s, %s offen offset:0" )
    buffer_load_dwordx2 = MemoryInstruction("buffer_load_dwordx2", 1, 0, 0, 2, \
        "%s, %s, %s, %s offen offset:0" )
    buffer_load_dword = MemoryInstruction("buffer_load_dword", 1, 0, 0, 1, \
        "%s, %s, %s, %s offen offset:0" )

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
      chosen_load_dwordx4 = buffer_load_dwordx4;
      chosen_load_dwordx2 = buffer_load_dwordx2;
      chosen_load_dword   = buffer_load_dword;
    else:
      chosen_load_dwordx4 = flat_load_dwordx4;
      chosen_load_dwordx2 = flat_load_dwordx2;
      chosen_load_dword   = flat_load_dword;

    self.memoryInstructions = {
        (8,0,3): {
          "GlobalRead": [ chosen_load_dwordx4, chosen_load_dwordx2,
            chosen_load_dword ],
          "GlobalWrite": [ flat_store_dwordx4, flat_store_dwordx2,
            flat_store_dword ],
          "LocalRead": [ ds_read_b128, ds_read2_b64,
            ds_read_b64, ds_read2_b32, ds_read_b32 ],
          "LocalWrite": [ ds_write_b128, ds_write2_b64,
            ds_write_b64, ds_write2_b32, ds_write_b32, ds_write_b16 ]
          }, # 803
        (9,0,0): {
          "GlobalRead": [ chosen_load_dwordx4, chosen_load_dwordx2,
            chosen_load_dword ],
          "GlobalWrite": [ flat_store_dwordx4, flat_store_dwordx2,
            flat_store_dword ],
          "LocalRead": [ ds_read_b128, ds_read2_b64,
            ds_read_b64, ds_read2_b32, ds_read_b32 ],
          "LocalWrite": [ ds_write_b128, ds_write2_b64,
            ds_write_b64, ds_write2_b32, ds_write_b32, ds_write_b16 ]
          } # 900
        }

    if "ISA" in kernel:
      self.version = kernel["ISA"]
    self.overflowedResources = False # if true, comment out whole kernel

    self.kernelName = self.getKernelName(kernel)
    self.inTailLoop = False

    # registers per element
    self.bpr = 4 # all registers are 32bit
    self.bpe = int(self.bpr*kernel["ProblemType"]["DataType"].numRegisters())
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
    self.globalReadWidthA = (tPA["nrcv"]*self.bpe)/self.bpr
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
    self.globalReadWidthB = (tPB["nrcv"]*self.bpe)/self.bpr
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
    self.localWriteWidthA = (tPA["nwcv"]*self.bpe)/self.bpr
    if self.localWriteWidthA < 1:
      self.localWriteWidthA = (1.0*tPA["nwcv"]*self.bpe)/self.bpr
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
    self.localWriteStrideTileA = (self.localWriteStrideTileA*self.bpe)/self.bpr
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
    self.localWriteStrideUnrollA = (self.localWriteStrideUnrollA*self.bpe)/self.bpr
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
    self.localWriteWidthB = (tPB["nwcv"]*self.bpe)/self.bpr
    if self.localWriteWidthB < 1:
      self.localWriteWidthB = (1.0*tPB["nwcv"]*self.bpe)/self.bpr
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
    self.localWriteStrideTileB = (self.localWriteStrideTileB*self.bpe)/self.bpr
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
    self.localWriteStrideUnrollB = (self.localWriteStrideUnrollB*self.bpe)/self.bpr
    self.localWriteInstructionIdxB = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthB, \
        kernel["LocalWrite2B"], \
        self.localWrite2CoalescedB, self.localWrite2PerpendicularB,
        [self.localWriteStrideTileB, self.localWriteStrideUnrollB] )

    ########################################
    # localRead A
    self.localReadWidth = (kernel["VectorWidth"] * self.bpe)/self.bpr
    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedA = (kernel["ThreadTile0"] * self.bpe)/self.bpr
    self.localRead2CoalescedA = kernel["ThreadTile0"]/kernel["VectorWidth"] > 1
    self.localReadInstructionIdxA = \
        self.selectMemoryInstruction("LocalRead", self.localReadWidth, \
        kernel["LocalRead2A"], \
        self.localRead2CoalescedA, localRead2Perpendicular,
        [self.localReadStrideCoalescedA] )

    ########################################
    # localRead B
    self.localReadWidth = (kernel["VectorWidth"] * self.bpe)/self.bpr
    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedB = (kernel["ThreadTile1"] * self.bpe)/self.bpr
    self.localRead2CoalescedB = kernel["ThreadTile1"]/kernel["VectorWidth"] > 1
    self.localReadInstructionIdxB = \
        self.selectMemoryInstruction("LocalRead", self.localReadWidth, \
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
    tPA["nrcvpi"] = int((self.globalReadInstructionA.totalWidth*self.bpr) / self.bpe)
    tPB["nrcvpi"] = int((self.globalReadInstructionB.totalWidth*self.bpr) / self.bpe)
    tPA["nwcvpi"] = int((self.localWriteInstructionA.totalWidth*self.bpr) / self.bpe)
    tPB["nwcvpi"] = int((self.localWriteInstructionB.totalWidth*self.bpr) / self.bpe)

    ####################################
    # VGPR Allocation
    ####################################

    ####################################
    # num vgprs: valu
    self.numVgprValuC = (kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.bpe)/self.bpr
    numVgprValuA = (kernel["ThreadTileA"]*self.bpe)/self.bpr
    numVgprValuB = (kernel["ThreadTileB"]*self.bpe)/self.bpr
    numVgprValuBlkA = numVgprValuA if kernel["PrefetchLocalRead"] else 0
    numVgprValuBlkB = numVgprValuB if kernel["PrefetchLocalRead"] else 0

    ####################################
    # num vgprs: global -> local elements
    numVgprG2LA = (kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["GlobalLoadVectorWidthA"] * self.bpe)/self.bpr
    numVgprG2LB = (kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["GlobalLoadVectorWidthB"] * self.bpe)/self.bpr

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
    numVgprLocalWriteAddressesA = 1 * self.rpla

    #numLocalWritesB = kernel["NumLoadsCoalescedB"] \
    #    * nlp * self.numWriteVectorComponentsB
    #numLocalWriteInstructionsB = numLocalWritesB \
    #    / self.localWriteInstructionB[self.instructionIdxNumOffsets]
    numVgprLocalWriteAddressesB = 1 * self.rpla

    ####################################
    # num vgprs: global read addresses
    numGlobalReadsA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["GlobalLoadVectorWidthA"] \
        * self.numReadVectorComponentsA
    numGlobalReadInstructionsA = (numGlobalReadsA * self.bpe)\
        / (self.globalReadInstructionA.blockWidth * 4)

    if kernel["BufferLoad"]:
      numVgprGlobalReadOffsetsA = numGlobalReadInstructionsA * self.rpgo 
    else:
      numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.rpga 

    numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["GlobalLoadVectorWidthB"] \
        * self.numReadVectorComponentsB
    numGlobalReadInstructionsB = (numGlobalReadsB * self.bpe) \
        / (self.globalReadInstructionB.blockWidth * 4)
    if kernel["BufferLoad"]:
      numVgprGlobalReadOffsetsB = numGlobalReadInstructionsB * self.rpgo 
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

    numVgprAddressD = self.rpga if globalParameters["DebugKernel"] else 0

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
    self.startVgprValuBlkA = vgprIdx; vgprIdx += numVgprValuBlkA
    if kernel["PrefetchGlobalRead"]:
      self.startVgprG2LA = vgprIdx; vgprIdx += numVgprG2LA
    else: # g2l can overlap valu
      self.startVgprG2LA = self.startVgprValuA
      vgprIdx = self.startVgprValuA \
          + max(numVgprValuA+numVgprValuBlkA, numVgprG2LA)

    self.startVgprValuB = vgprIdx; vgprIdx += numVgprValuB
    self.startVgprValuBlkB = vgprIdx; vgprIdx += numVgprValuBlkB
    if kernel["PrefetchGlobalRead"]:
      self.startVgprG2LB = vgprIdx; vgprIdx += numVgprG2LB
    else: # g2l can overlap valu
      self.startVgprG2LB = self.startVgprValuB
      vgprIdx = self.startVgprValuB \
          + max(numVgprValuB+numVgprValuBlkB, numVgprG2LB)

    self.startVgprLocalReadAddressesA = vgprIdx
    vgprIdx += numVgprLocalReadAddressesA
    self.startVgprLocalReadAddressesB = vgprIdx
    vgprIdx += numVgprLocalReadAddressesB
    self.startVgprLocalWriteAddressesA = vgprIdx
    vgprIdx += numVgprLocalWriteAddressesA
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
      vgprIdx += numVgprGlobalReadOffsetsA
      self.startVgprGlobalReadOffsetB = vgprIdx
      vgprIdx += numVgprGlobalReadOffsetsB
    else:
      self.startVgprGlobalReadAddressesA = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesA
      self.startVgprGlobalReadAddressesB = vgprIdx
      vgprIdx += numVgprGlobalReadAddressesB
    self.startVgprGlobalReadIncsA = vgprIdx
    vgprIdx += numVgprGlobalReadIncsA
    self.startVgprGlobalReadIncsB = vgprIdx
    vgprIdx += numVgprGlobalReadIncsB
    self.startVgprAddressD = vgprIdx
    vgprIdx += numVgprAddressD
    self.startVgprSerial = vgprIdx
    vgprIdx += numVgprSerial
    # tmp vgprs
    #minVgprTmp = 1
    #if kernel["LoopTail"]:
    #  minVgprTmp += 4
    #if globalParameters["DebugKernel"]:
    #  minVgprTmp += 2
    self.startVgprTmp = vgprIdx
    #vgprIdx += minVgprTmp
    #print2("%3u vgprs <- %s" % (vgprIdx, self.kernelName) )
    vgprPerCU = 65536
    vgprPerThreadPerOccupancy = vgprPerCU / kernel["NumThreads"]
    numWorkGroupsPerCU = vgprPerThreadPerOccupancy / vgprIdx
    if numWorkGroupsPerCU < 1:
      self.overflowedResources = True
      numWorkGroupsPerCU  = 1 # dummy value
    numWavesPerWorkGroup = kernel["NumThreads"] / 64
    numWavesPerCU = numWorkGroupsPerCU * numWavesPerWorkGroup
    self.numWavesPerSimd = numWavesPerCU / 4
    maxVgprSameOccupancy = vgprPerThreadPerOccupancy / numWorkGroupsPerCU
    self.numVgprTmp = maxVgprSameOccupancy - self.startVgprTmp
    self.totalVgprs = maxVgprSameOccupancy

    # move serial to last vgpr and shift tmp forward
    self.startVgprSerial = self.totalVgprs-1
    self.startVgprTmp -= 1

    ########################################
    # SGPR Allocation
    ########################################

    ####################################
    # num sgprs: initial kernel state
    numSgprKernArgAddress = self.rpga
    numSgprWorkGroup0 = 1
    numSgprWorkGroup1 = 1
    numSgprWorkGroup2 = 1
    numSgprSrdA = 4  # resource descriptor (SRD) A, must be aligned on 4-SGPR boundary
    numSgprSrdB = 4  # resource descriptor (SRD) B, must be aligned on 4-SGPR boundary
    #numSgprSrdC = 4  # resource descriptor (SRD) C, must be aligned on 4-SGPR boundary
    numSgprNumWorkGroups0 = 1 # num macro tiles, not multiplied by GSU
    numSgprNumWorkGroups1 = 1 # num macro tiles, not multiplied by GSU
    numSgprGSUSumIdx = 2 if kernel["GlobalSplitU"] > 1 else 0
    numSgprAddressC = self.rpga # til end
    numSgprAddressA = self.rpga # til read offsets
    numSgprAddressB = self.rpga # til read offsets
    numSgprOffsetC = 1
    numSgprOffsetA = 1
    numSgprOffsetB = 1
    numSgprAlpha = max(1,int(self.bpe/4))
    numSgprBeta  = max(1,int(self.bpe/4)) if kernel["ProblemType"]["UseBeta"] else 0
    self.numSgprStridesC = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprStridesA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    self.numSgprStridesB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    if not kernel["ProblemType"]["UseInitialStrides"]:
      self.numSgprStridesC -= 1
      self.numSgprStridesA -= 1
      self.numSgprStridesB -= 1
    self.numSgprSizesSum = kernel["ProblemType"]["NumIndicesSummation"]
    self.numSgprSizesFree = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprAddressD = self.rpga if globalParameters["DebugKernel"] else 0

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

    numSgprLoopCountersAndIncrements = numSgprGlobalReadIncsA \
        + numSgprGlobalReadIncsB + numSgprLoopCounters
    numSgprFreedBeforeLoops = self.numSgprStridesA + self.numSgprStridesB \
        + self.numSgprSizesFree + numSgprAddressA + numSgprAddressB \
        + numSgprOffsetC + numSgprOffsetA + numSgprOffsetB
    numSgprLoopPadding = max(0, numSgprFreedBeforeLoops  \
        - numSgprLoopCountersAndIncrements)
    if kernel["LoopTail"]:
      numSgprLoopTail = 6
    else:
      numSgprLoopTail = 0

    ########################################
    # SGPR Assignment according to AMDGPU-ABI
    ########################################
    sgprIdx = 0
    self.startSgprKernArgAddress = sgprIdx; sgprIdx += numSgprKernArgAddress
    assert(self.startSgprKernArgAddress == 0) # kernarg is passed to kernel as SGPR0

    self.startSgprWorkGroup0 = sgprIdx;     sgprIdx += numSgprWorkGroup0
    self.startSgprWorkGroup1 = sgprIdx;     sgprIdx += numSgprWorkGroup1
    self.startSgprWorkGroup2 = sgprIdx;     sgprIdx += numSgprWorkGroup2

    self.startSgprNumWorkGroups0 = sgprIdx; sgprIdx += numSgprNumWorkGroups0
    self.startSgprNumWorkGroups1 = sgprIdx; sgprIdx += numSgprNumWorkGroups1

    if kernel["BufferLoad"]:
      sgprIdx = ((sgprIdx+3) / 4) * 4  # Round Up to next 4-byte aligned.  
      self.startSgprSrdA = sgprIdx; sgprIdx += numSgprSrdA;
      assert (self.startSgprSrdA % 4 == 0) # must be aligned to 4 SGPRs
      self.startSgprSrdB = sgprIdx; sgprIdx += numSgprSrdB;
      assert (self.startSgprSrdB % 4 == 0) # must be aligned to 4 SGPRs

    self.startSgprGSUSumIdx = sgprIdx;      sgprIdx += numSgprGSUSumIdx
    self.startSgprAddressC = sgprIdx;       sgprIdx += numSgprAddressC
    self.startSgprStridesC = sgprIdx;       sgprIdx += self.numSgprStridesC
    # doubles need to be aligned to even
    if self.bpe > 4 and sgprIdx%2==1:
      sgprIdx += 1
    self.startSgprAlpha = sgprIdx;          sgprIdx += numSgprAlpha
    self.startSgprBeta = sgprIdx;           sgprIdx += numSgprBeta
    self.startSgprSizesFree = sgprIdx;      sgprIdx += self.numSgprSizesFree
    self.startSgprSizesSum = sgprIdx;       sgprIdx += self.numSgprSizesSum
    self.startSgprLoopPadding = sgprIdx;    sgprIdx += numSgprLoopPadding # overlap
    self.startSgprStridesA = sgprIdx;       sgprIdx += self.numSgprStridesA
    self.startSgprStridesB = sgprIdx;       sgprIdx += self.numSgprStridesB
    self.startSgprAddressA = sgprIdx;       sgprIdx += numSgprAddressA
    self.startSgprAddressB = sgprIdx;       sgprIdx += numSgprAddressB
    self.startSgprOffsetC = sgprIdx;        sgprIdx += numSgprOffsetC
    self.startSgprOffsetA = sgprIdx;        sgprIdx += numSgprOffsetA
    self.startSgprOffsetB = sgprIdx;        sgprIdx += numSgprOffsetB
    self.startSgprLoopTail = sgprIdx;       sgprIdx += numSgprLoopTail
    self.startSgprAddressD = sgprIdx;       sgprIdx += self.numSgprAddressD
    self.totalSgprs = sgprIdx

    self.startSgprTmpPool = self.totalSgprs

    # assign loop sgprs which overlap above assignments
    sgprIdx = self.startSgprLoopPadding
    self.startSgprGlobalReadIncsA = sgprIdx; sgprIdx += numSgprGlobalReadIncsA
    self.startSgprGlobalReadIncsB = sgprIdx; sgprIdx += numSgprGlobalReadIncsB
    self.startSgprLoopCounters = sgprIdx

    ########################################
    # Register Pools
    ########################################
    #print "TotalVgprs", self.totalVgprs
    self.vgprPool = RegisterPool(self.totalVgprs-1) # don't initially reserve Serial
    #print self.vgprPool.state()

    self.vgprPool.add(self.startVgprValuC, \
        self.startVgprLocalReadAddressesA - self.startVgprValuC)
    #print self.vgprPool.state()

    self.vgprPool.add( self.startVgprTmp, self.numVgprTmp)
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
    kernArgReg += max(1,int(self.bpe/4)) # alpha
    if kernel["ProblemType"]["UseBeta"]:
      kernArgReg += max(1,int(self.bpe/4)) # beta
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
    totalVgprs = self.vgprPool.size()+1 # + Serial
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
    kStr += "  workgroup_group_segment_byte_size = %u // lds bytes%s" \
        % ( kernel["LdsNumElements"] * self.bpe, self.endLine )

    # other
    kStr += "  compute_pgm_rsrc2_user_sgpr = 2 // vcc%s" % self.endLine
    kStr += "  kernarg_segment_alignment = 4%s" % self.endLine
    kStr += "  group_segment_alignment = 4%s" % self.endLine
    kStr += "  private_segment_alignment = 4%s" % self.endLine
    kStr += ".end_amd_kernel_code_t%s" % self.endLine


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




    ########################################
    # VGPR Macros
    ########################################
    kStr += self.comment3("VGPR Assignments")
    kStr += self.macroRegister("vgprValuC", self.startVgprValuC)
    kStr += self.macroRegister("vgprValuA", self.startVgprValuA)
    kStr += self.macroRegister("vgprValuBlkA", self.startVgprValuBlkA)
    kStr += self.macroRegister("vgprG2LA", self.startVgprG2LA)
    kStr += self.macroRegister("vgprValuB", self.startVgprValuB)
    kStr += self.macroRegister("vgprValuBlkB", self.startVgprValuBlkB)
    kStr += self.macroRegister("vgprG2LB", self.startVgprG2LB)
    kStr += self.macroRegister("vgprLocalReadAddrA", \
        self.startVgprLocalReadAddressesA)
    kStr += self.macroRegister("vgprLocalReadAddrB", \
        self.startVgprLocalReadAddressesB)
    kStr += self.macroRegister("vgprLocalWriteAddrA", \
        self.startVgprLocalWriteAddressesA)
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
      kStr += self.macroRegister("vgprAddressD", \
          self.startVgprAddressD)
    self.startVgprSerial = totalVgprs - 1
    kStr += self.macroRegister("vgprSerial", \
        self.startVgprSerial)
    #kStr += self.comment1("VGPRs: %u + %u = %u" \
    #    % (self.startVgprTmp, self.numVgprTmp, self.totalVgprs) )
    #kStr += self.comment1("Occu: %u waves/simd" % self.numWavesPerSimd )


    ########################################
    # SGPR Macros
    ########################################
    kStr += self.comment3("SGPR Assignments")

    kStr += self.macroRegister("sgprKernArgAddress", \
        self.startSgprKernArgAddress)
    kStr += self.macroRegister("sgprWorkGroup%u"%(0 if kernel["WorkGroupMapping"]>0 else 1), self.startSgprWorkGroup0)
    kStr += self.macroRegister("sgprWorkGroup%u"%(1 if kernel["WorkGroupMapping"]>0 else 0), self.startSgprWorkGroup1)
    kStr += self.macroRegister("sgprNumWorkGroups0", self.startSgprNumWorkGroups0)
    kStr += self.macroRegister("sgprNumWorkGroups1", self.startSgprNumWorkGroups1)
    for i in range(2, kernel["ProblemType"]["NumIndicesC"]):
      kStr += self.macroRegister("sgprWorkGroup%u"%i, self.startSgprWorkGroup0+i)
    if kernel["BufferLoad"]:
      kStr += self.macroRegister("sgprSrdA", self.startSgprSrdA)
      kStr += self.macroRegister("sgprSrdB", self.startSgprSrdB)
    if kernel["GlobalSplitU"] > 1:
      kStr += self.macroRegister("sgprGSUSumIdx",self.startSgprGSUSumIdx)
    kStr += self.macroRegister("sgprAddressC", self.startSgprAddressC)
    kStr += self.macroRegister("sgprStridesC", self.startSgprStridesC)
    kStr += self.macroRegister("sgprAlpha", self.startSgprAlpha)
    if kernel["ProblemType"]["UseBeta"]:
      kStr += self.macroRegister("sgprBeta", self.startSgprBeta)
    kStr += self.macroRegister("sgprSizesFree", self.startSgprSizesFree)
    kStr += self.macroRegister("sgprSizesSum", self.startSgprSizesSum)
    kStr += self.macroRegister("sgprLoopPadding", self.startSgprLoopPadding)
    kStr += self.macroRegister("sgprStridesA", self.startSgprStridesA)
    kStr += self.macroRegister("sgprStridesB", self.startSgprStridesB)
    kStr += self.macroRegister("sgprAddressA", self.startSgprAddressA)
    kStr += self.macroRegister("sgprAddressB", self.startSgprAddressB)
    kStr += self.macroRegister("sgprOffsetC", self.startSgprOffsetC)
    kStr += self.macroRegister("sgprOffsetA", self.startSgprOffsetA)
    kStr += self.macroRegister("sgprOffsetB", self.startSgprOffsetB)
    if globalParameters["DebugKernel"]:
      kStr += self.macroRegister("sgprAddressD", self.startSgprAddressD)
    if not self.globalReadIncsUseVgpr:
      kStr += self.macroRegister("sgprGlobalReadIncsA", \
          self.startSgprGlobalReadIncsA)
      kStr += self.macroRegister("sgprGlobalReadIncsB", \
          self.startSgprGlobalReadIncsB)
    kStr += self.macroRegister("sgprLoopCounters", self.startSgprLoopCounters)
    #kStr += self.comment1("SGPR: %u" % self.totalSgprs)

    if kernel["BufferLoad"]:
        kStr += self.comment3("2GB limit - set offsets to -1 to exceed this and clamp")
        kStr += self.macroRegister("BufferLimit", "0x8000000")
        kStr += self.comment3("Bits 127:96 of SRD.  Set DataFormat = 32 bit")
        kStr += self.macroRegister("Srd127_96",   "0x0020000")

    ########################################
    # Global Offsets
    ########################################
    for (tensorChar, indices) in [ \
        ("C", range(0, kernel["ProblemType"]["NumIndicesC"])), \
        ("A", kernel["ProblemType"]["IndexAssignmentsA"]), \
        ("B", kernel["ProblemType"]["IndexAssignmentsB"]) ]:
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
        kStr += inst("v_mov_b32", "v[\\vgprAddr+0]", "v[\\vgprOffset%s]" \
            % idxChars[0], "d0 lower")
        kStr += inst("v_mov_b32", "v[\\vgprAddr+1]", hex(0), "d0 upper")
      # other c index sgpr
      elif indices[0] < kernel["ProblemType"]["NumIndicesC"]:
        kStr += inst("v_mov_b32", "v[\\vgprAddr+0]", "s[\\sgprOffset%s]" \
            % idxChars[0], "d0 lower")
        kStr += inst("v_mov_b32", "v[\\vgprAddr+1]", hex(0), "d0 upper")
      # other sum index
      else:
        kStr += inst("v_mov_b32", "v[\\vgprAddr+0]", hex(0), "d0 lower")
        kStr += inst("v_mov_b32", "v[\\vgprAddr+1]", hex(0), "d0 upper")

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
          kStr += inst("v_mul_hi_u32", \
              "v[\\vgprTmp+1]", \
              sgpr("Strides%s+%u"%(tensorChar,i-1)), \
              "v[\\vgprOffset%s]" % idxChars[i],  \
              "mul d%u upper"%i)
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
          kStr += inst("v_mul_hi_u32", \
              "v[\\vgprTmp+1]", \
              sgpr("Strides%s+%u"%(tensorChar,i-1)), \
              "v[\\vgprTmp+2]",  \
              "mul d%u upper"%i)
        # other sum index
        else:
          # don't even need to add b/c offset=zero
          continue

        # addr += offset * stride (lo)
        kStr += inst("_v_add_co_u32", \
            "v[\\vgprAddr+0]", \
            "vcc", \
            "v[\\vgprTmp+0]", \
            "v[\\vgprAddr+0]",  \
            "accumulate d%u lower"%i)
        # addr += offset * stride (hi)
        kStr += inst("_v_addc_co_u32", \
            "v[\\vgprAddr+1]", \
            "vcc", \
            "v[\\vgprTmp+1]",  \
            "v[\\vgprAddr+1]",  \
            "vcc", \
            "accumulate d%u upper"%i)

      # addr *= bytes/element
      kStr += inst("v_lshlrev_b64", \
          "v[\\vgprAddr+0:\\vgprAddr+1]", \
          hex(log2(self.bpe)), \
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

    ########################################
    # MACs
    kStr += self.comment3("%dx%d thread-tile" \
        % (kernel["ThreadTile0"], kernel["ThreadTile1"]) )
    numMacs = 2 if kernel["PrefetchLocalRead"] else 1
    for m in range(0, numMacs):
      kStr += ".macro MAC_%ux%u" \
          % (kernel["ThreadTile0"], kernel["ThreadTile1"])
      if kernel["PrefetchLocalRead"]:
        kStr += ("" if m==0 else "_BLK")
      kStr += self.endLine
      macIdx = 0
      # half precision
      if kernel["ProblemType"]["DataType"].isHalf():
        for blockB in range(0, kernel["ThreadTile1"]/2):
          for blockA in range(0, kernel["ThreadTile0"]/2):
            if self.version == (8,0,3):
              for b in range(blockB*2, (blockB+1)*2):
                for a in range(blockA*2, (blockA+1)*2):
                  # v_mac_f16 or v_fma_f16
                  cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"])
                  aStr = "v[%s+%u]" \
                      % ("vgprValuA" if m==0 else "vgprValuBlkA", blockA)
                  bStr = "v[%s+%u]" \
                      % ("vgprValuB" if m==0 else "vgprValuBlkB", blockB)
                  kStr += "v_mac_f16 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine) # FIXME op_sel
            elif self.version == (9,0,0):
              b = blockB*2
              a = blockA*2
              cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # /2 b/c of 2 f16's per 32-bit vgpr
              aStr = "v[%s+%u]" \
                  % ("vgprValuA" if m==0 else "vgprValuBlkA", blockA)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB" if m==0 else "vgprValuBlkB", blockB)
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
            cStr = "v[%s+%u+%u*%u]" % ("vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u]" \
                % ("vgprValuA" if m==0 else "vgprValuBlkA", a)
            bStr = "v[%s+%u]" \
                % ("vgprValuB" if m==0 else "vgprValuBlkB", b)
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
            cStr = "v[%s+(%u+%u*%u)*2:(%s+%u+%u*%u)*2+1]" % ("vgprValuC", a, b, kernel["ThreadTile0"], "vgprValuC", a, b, kernel["ThreadTile0"])
            aStr = "v[%s+%u*2:%s+%u*2+1]" \
                % ("vgprValuA" if m==0 else "vgprValuBlkA", a, "vgprValuA" if m==0 else "vgprValuBlkA", a)
            bStr = "v[%s+%u*2:%s+%u*2+1]" \
                % ("vgprValuB" if m==0 else "vgprValuBlkB", b, "vgprValuB" if m==0 else "vgprValuBlkB", b)
            kStr += "v_fma_f64 %s, %s, %s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)


      # other precision
      else:
        printExit("Assembly doesn't support %s" % kernel["ProblemType"]["DataType"])
      kStr += ".endm%s" % self.endLine


    # if overflowed vgpr pool, comment out the whole kernel body and let it fail gracefully
    if self.vgprPool.size() > self.maxVgprs:
      self.overflowedResources = True
    if self.overflowedResources:
      print ""
      printWarning("%s invalid: too many vgprs" % (self.kernelName) )
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

    if self.do["PreLoop"]: 
      # set m0
      kStr += inst("s_mov_b32", "m0", hex(kernel["LdsNumElements"] \
          * self.bpe), "LDS clamp at %u bytes" \
          %(kernel["LdsNumElements"] * self.bpe) )

      kStr += inst("v_mov_b32", vgpr("Serial"), vgpr(0), "thread serial id")

      ########################################
      # load kernel args
      kStr += self.comment("Load Kernel Args")
      kernArgOffset = 0
      if globalParameters["DebugKernel"]:
        kStr += inst("s_load_dword", sgpr("AddressD"), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug" )
        kernArgOffset += 1*4
        kStr += inst("s_load_dword", sgpr("AddressD+1"), \
            sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug" )
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
      kernArgOffset += 1*max(4,self.bpe)
      if kernel["ProblemType"]["UseBeta"]:
        if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isSingle():
          kStr += inst("s_load_dword", sgpr("Beta"), \
              sgpr("KernArgAddress",2), hex(kernArgOffset), "load beta" )
        elif kernel["ProblemType"]["DataType"].isDouble():
          kStr += inst("s_load_dword", sgpr("Beta+0"), \
              sgpr("KernArgAddress",2), hex(kernArgOffset+0), "load beta" )
          kStr += inst("s_load_dword", sgpr("Beta+1"), \
              sgpr("KernArgAddress",2), hex(kernArgOffset+4), "load beta" )
        kernArgOffset += 1*max(4,self.bpe)
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
    self.startSgprTmpPool = self.startSgprOffsetC

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
      kStr += inst("_v_add_co_u32", vgpr("AddressD"), "vcc", sgpr("AddressD"), \
          vgpr(v), "%s=AddrD* + serial*nipt*4"%vgpr("AddressD") )
      kStr += inst("v_mov_b32", vgpr(v+2), sgpr("AddressD+1"), "%s=AddressD1"%vgpr(v+2) )
      kStr += inst("_v_addc_co_u32", vgpr("AddressD+1"), "vcc", vgpr(v+2), \
          vgpr(v+1), "vcc", "%s=AddrD* + serial*nipt*4"%vgpr("AddressD") )
      self.vgprPool.checkIn(v)
      self.vgprPool.checkIn(nwg0)

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
        tmpSgpr = self.getTmpSgpr(2) # needs 3
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
        kStr += staticMultiply(vgpr(tReg), vgpr(tReg), tP["glvw"], sgpr(tmpSgpr))
      else:
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
    stride = tP["lsc"] if tP["tlu"] else tP["lsp"]
    stride = kernel[stride]
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
            "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], l, 0) )
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
            vgpr(v+l-1), "gro%s%s_%u"%(tP["tensorChar"], tP["tileChar"], l) )
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
    stride = (tP["lsp"] if tP["tlu"] else tP["lsc"])
    stride = kernel[stride]
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
            "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, l, 0) )
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
            vgpr(v+l-1), "gro%s%s_%u"%(tP["tensorChar"], self.unrollChar, l) )
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
    kStr = ""
    # edge value
    margin = tP["glvw"] if tP["rtv"] else 1
    edge = self.vgprPool.checkOut(1)

    tmpSgpr = self.getTmpSgpr(1)
    kStr += inst("s_add_u32", sgpr(tmpSgpr), hex(-margin), sgpr("SizesFree+%u"%tP["idx"]), \
        "edge = Size%s-%u"%(tP["tileChar"], margin) )
    kStr += inst("v_mov_b32", vgpr(edge), sgpr(tmpSgpr), \
        "edge = Size%s-%u"%(tP["tileChar"], margin) )
    # correct but invalid instruction
    #kStr += inst("_v_add_co_u32", vgpr(edge), "vcc", -margin, sgpr("SizesFree+%u"%tP["idx"]), \
    #    "edge = Size%s-%u"%(tP["tileChar"], margin) )

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
            # global offset macro
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

            # dump final offsets
            #kStr += dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)))
            #kStr += dump(vgpr("GlobalReadAddr%s+%u+1"%tP["tensorChar"], graIdx))
            graIdx += self.rpgo if kernel["BufferLoad"] else self.rpga
    self.vgprPool.checkIn(tileOffsets)
    self.vgprPool.checkIn(unrollOffsets)
    self.vgprPool.checkIn(tmp)
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
    graIdx = 0

    if kernel["BufferLoad"]:
      # Buffer-load uses one base read pointer stored in the SRD - set it here:
      kStr += inst("s_mov_b32", sgpr("Srd%s+0"%tP["tensorChar"]), sgpr("Address%s+0"%tP["tensorChar"]), "init SRD base address (lower)" )
      kStr += inst("s_mov_b32", sgpr("Srd%s+1"%tP["tensorChar"]), sgpr("Address%s+1"%tP["tensorChar"]), "init SRD base address (upper) + other fields" )
      kStr += inst("s_mov_b32", sgpr("Srd%s+2"%tP["tensorChar"]), "BufferLimit", "")
      kStr += inst("s_mov_b32", sgpr("Srd%s+3"%tP["tensorChar"]), "Srd127_96", "Set bits 127_96 in SRD" )

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
          kStr += inst("s_mul_i32", sgpr(tmpSgpr+0), \
              hex(depthU*self.bpe), sgpr("Strides%s"%tP["tensorChar"]), \
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
        else:
          kStr += inst("s_mul_i32", sgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(depthU*self.bpe), sgpr("Strides%s"%tP["tensorChar"]), \
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
        if self.globalReadIncsUseVgpr:
          kStr += inst("v_mov_b32", vgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(depthU*self.bpe), \
              "incr = %u*bytes"%depthU )
          kStr += inst("v_mov_b32", vgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), "incr = %u*bytes (upper)"%depthU )
        else:
          kStr += inst("s_mov_b32", sgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(depthU*self.bpe), \
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
    #"lwFOA = lwA%s + lwA%s*MT%s" \
    #    % (tP["tileChar"], self.unrollChar, tP["tileChar"])
    uReg = tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"]
    kStr += inst("v_mul_u32_u24", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        hex(kernel["MacroTile%s"%tP["tensorChar"]]), \
        vgpr(uReg), \
        "lw%s%s*MT%s"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
    kStr += inst("_v_add_co_u32", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        "vcc", \
        vgpr(tP["gpr"]["lwoT"]), \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        "lwFO%s = lw%s%s + lw%s%s*MT%s" \
        % (tP["tensorChar"], tP["tensorChar"], tP["tileChar"], \
        tP["tensorChar"], self.unrollChar, tP["tileChar"]) )
    kStr += inst("v_lshlrev_b32", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        hex(log2(self.bpe)), \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        " *= bytes/element" )
    if tP["isB"]:
      kStr += inst("_v_add_co_u32", \
          vgpr("LocalWriteAddrB"), \
          "vcc", \
          hex(kernel["LdsOffsetB"]*self.bpe), \
          vgpr("LocalWriteAddrB"), \
          "lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u*%u" % (tP["tileChar"], \
          self.unrollChar, tP["tileChar"], kernel["LdsOffsetB"], self.bpe) )
    self.vgprPool.checkIn(tP["gpr"]["lwoT"])
    self.vgprPool.checkIn(tP["gpr"]["uReg"])
    if kernel["GlobalSplitU"] > 1:
      self.vgprPool.checkIn(tP["gpr"]["uReg2"])
    # dump lds write offsets
    #kStr += dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A/B
  # initially assume write offsets fit into 8-bits
  ##############################################################################
  def lwaFinalOffsets(self, kernel, tP):
    return self.comment("N/A")
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for para in range(0, tP["nrc"]):
        for s in range(0, max(tP["nwcv"],tP["nwpv"])/tP["nwcvpi"]):
          lscaOffset = para * kernel[tP["lsc"]]
          lspaOffset = perp * kernel[tP["lsp"]]
          sPara = 0
          sPerp = 0
          if tP["wtc"]:
            sPerp = s
            lscaOffset += s
          elif tP["wuc"]:
            sPara = s
            lspaOffset += s # * VW could go here, check transpose options
          if tP["tlu"]:
            lspaOffset *= kernel[tP["mt"]]
            #lspa *= tP["glvw"]
          else:
            lscaOffset *= kernel[tP["mt"]]
          if tP["tlu"] == tP["grcv"]:
            lspaOffset *= tP["glvw"]
          offset = lspaOffset + lscaOffset
          offset *= self.bpe
          offset /= tP["localWriteInstruction"].offsetMultiplier
          kStr += "%slwo%s_%u_%u_%u_%u = (%s%d*%s)" \
              % (self.commentPrefix, tP["tensorChar"], \
              para, sPara, perp, sPerp, \
              (("%u + "%sPara) if tP["wtc"] else ""), \
              para, tP["lsc"] )
          if not tP["tlu"]:
            kStr += "*MT%s" % (tP["tileChar"])
          kStr += " + (%s%d*%s)" % (
              (("%u + "%sPerp) if tP["wuc"] else ""), perp, \
              tP["lsp"])
          if tP["tlu"]:
            kStr += "*MT%s" % (tP["tileChar"])
          kStr += " = %u%s%s" % (offset, self.commentSuffix, self.endLine)
    return kStr

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
    divisor = kernel["SubGroup0"]*kernel["SubGroup1"]
    qReg = self.vgprPool.checkOut(1) # quotient
    rReg = self.vgprPool.checkOut(1) # remainder
    dividendReg = "Serial"
    tmpVgpr = self.vgprPool.checkOut(2)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    sgid = qReg
    kStr += inst("s_mov_b32", \
        sgpr(tmpSgpr), \
        hex(kernel["MacroTile%u"%tP["tensorIdx"]]), \
        "MT%u"%tP["tensorIdx"] )
    kStr += inst("v_mul_lo_u32", \
        vgpr(sgid), \
        sgpr(tmpSgpr), \
        vgpr(sgid), \
        "sgid*sgid*MT%u"%tP["tensorIdx"] )
    if kernel["VectorWidth"] > 1:
      kStr += staticMultiply(vgpr(tP["gpr"]["lro"]), vgpr(tP["gpr"]["lro"]), \
          kernel["VectorWidth"], sgpr(tmpSgpr))
    kStr += inst("_v_add_co_u32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "vcc", \
        vgpr(sgid), \
        vgpr(tP["gpr"]["lro"]), \
        "o = lro%s*VW+sgid*MT%u"%(tP["tensorChar"], tP["tensorIdx"]) )
    kStr += inst("v_lshlrev_b32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        hex(log2(self.bpe)), \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "*= bytes/element" )

    # dump lra final offset
    #kStr += dump(vgpr("LocalReadAddr%s"%tP["tensorChar"])) # all zeros for B

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
      return inst("_v_add_co_u32", \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          "vcc", \
          hex(kernel["LdsOffset%s"%tP["tensorChar"]]*self.bpe), \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          " += LdsOffset%s (lower)"%tP["tensorChar"])

  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  def declareLoopNumIter(self, kernel):
    self.vgprPool.remove(self.startVgprValuC, \
        self.startVgprLocalReadAddressesA - self.startVgprValuC)
    #print "vgpr pool adding tmp registers"
    #self.vgprPool.add(self.startVgprTmp, self.numVgprTmp)
    kStr = ""
    for i in range(0, self.numVgprValuC):
      kStr += inst("v_mov_b32", vgpr("ValuC+%u"%i), hex(0), "")
    return kStr


  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx):
    kStr = ""

    tmp = self.vgprPool.checkOut(1)
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
      tmpSgpr = self.getTmpSgpr(2)
      # size % DepthU
      kStr += scalarStaticDivideAndRemainder(tmpSgpr, "LoopCounters+%u"%loopIdx, "SizesSum+%u"%loopIdx, kernel["DepthU"], tmpSgpr, True)

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
        tmpSgpr = self.getTmpSgpr(2)
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
    self.vgprPool.checkIn(tmp)

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
      kStr += self.comment("apply exec mask")
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

    kStr += inst("s_add_u32", \
        sgpr("LoopCounters+%u"%loopIdx), \
        sgpr("LoopCounters+%u"%loopIdx), \
        hex(1), \
        "counter%s++"%(loopChar) )
    kStr += inst("s_cmp_eq_i32", \
        sgpr("LoopCounters+%u"%loopIdx), \
        hex(endCounter), \
        "counter%s==0"%(loopChar) )
    kStr += inst("s_cbranch_scc1 label_%04u"%loopLabelEnd, \
        "exit Loop%s"%loopChar )
    kStr += inst("s_branch label_%04u"%loopLabelBegin, \
        "restart Loop%s"%loopChar )
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
        self.startVgprTmp - self.startVgprValuA)
    self.startSgprTmpPool = self.startSgprSizesSum
    return ""

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  def macIter(self, kernel, black):
    if not self.do["MAC"]: return ""
    kStr = ""
    kStr += "MAC_%ux%u" % (kernel["ThreadTile0"],kernel["ThreadTile1"])
    if black:
      kStr += "_BLK"
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
  # Global Read: Increment A/B
  ##############################################################################
  def globalReadIncrement(self, kernel, loopIdx, tP):
    if not self.do["GlobalInc"]: return ""
    kStr = ""

    if kernel["BufferLoad"]:
      kStr += inst("s_add_u32 ", \
           sgpr("Srd%s+0"%(tP["tensorChar"])), \
           sgpr("Srd%s+0"%(tP["tensorChar"])), \
           sgpr("GlobalReadIncs%s+0"%(tP["tensorChar"])), \
          "gra SRD += inc(lower)" )
      kStr += inst("s_addc_u32 ", \
           sgpr("Srd%s+1"%(tP["tensorChar"])), \
           sgpr("Srd%s+1"%(tP["tensorChar"])),\
           sgpr("GlobalReadIncs%s+1"%(tP["tensorChar"])), \
          "gra SRD += inc(upper)" )
    else:
      loopChar = self.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      graIdx = 0
      tmp = self.vgprPool.checkOut(1)
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
    if not self.do["GlobalRead"]: return ""
    kStr = ""
    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth

    # sizeK % LOCAL_DEPTHU
    if guardK:
      ########################################
      # Calculate Max Addr
      ########################################
      maxAddr = self.getTmpSgpr(2) # 3+6 = 9 sgprs available
      tmpSgpr = maxAddr + 2 # 7 sgprs available
      #dumpVgpr = self.vgprPool.checkOut(1)

      # maxAddr = size[n] * stride[n-1]
      kStr += self.comment1("max read address = size[n] * stride[n-1]")
      dim = len(tP["ia"])-1 # dim
      strideIdx = dim-1 # largest stride
      sizeIdx = tP["ia"][dim]

      sizeIdxIsSum = sizeIdx in kernel["ProblemType"]["IndicesSummation"]
      if sizeIdxIsSum:
        sizeIdx -= kernel["ProblemType"]["NumIndicesC"]

      kStr += inst("s_mul_i32", \
          sgpr(maxAddr+0), \
          sgpr("Sizes%s+%u"%("Sum" if sizeIdxIsSum else "Free", sizeIdx)),  \
          sgpr("Strides%s+%u"%(tP["tensorChar"],strideIdx)), \
          "mul d%u lower"%dim)
      kStr += inst("s_mov_b32", sgpr(maxAddr+1), hex(0), "zero (upper)")
      # maxAddr *= bytes/element
      kStr += inst("s_lshl_b64", \
          sgpr(maxAddr,2), \
          sgpr(maxAddr,2), \
          hex(log2(self.bpe)), "offset *= bytes/element")
      # maxAddr += initial address
      kStr += inst("s_add_u32", \
          sgpr(maxAddr+0), \
          sgpr(self.startSgprAddressA if tP["isA"] else self.startSgprAddressB), \
          sgpr(maxAddr+0), \
          "prepend address lower")
      kStr += inst("s_addc_u32", \
          sgpr(maxAddr+1), \
          sgpr((self.startSgprAddressA if tP["isA"] else self.startSgprAddressB)+1), \
          sgpr(maxAddr+1), \
          "prepend address upper")
      # sgpr->vgpr
      tmpVgpr = self.vgprPool.checkOut(2)
      kStr += inst("v_mov_b32", vgpr(tmpVgpr+0), sgpr(maxAddr+0), "sgpr->vgpr")
      kStr += inst("v_mov_b32", vgpr(tmpVgpr+1), sgpr(maxAddr+1), "sgpr->vgpr")
      maxAddr = tmpVgpr
      #kStr += dump(vgpr(maxAddr+0))
      #kStr += dump(vgpr(maxAddr+1))

      if kernel["BufferLoad"]:
        fullAddrVgpr = self.vgprPool.checkOut(2)
      else:
        # full exec mask
        fullExec = tmpSgpr
        kStr += inst("s_mov_b64", sgpr(fullExec,2), \
            "0xFFFFFFFFFFFFFFFF", "to restore all threads active")

      # inc
      bpeVgpr = self.vgprPool.checkOut(1)
      kStr += inst("v_mov_b32", vgpr(bpeVgpr), \
          hex(self.bpe), "bytes per element")
      zeroVgpr = self.vgprPool.checkOut(1)
      kStr += inst("v_mov_b32", vgpr(zeroVgpr), \
          hex(0), "zero")
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):
            i = sPara + (tP["nrcv"]/tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            graIdx = i * self.rpgo if kernel["BufferLoad"] else i * self.rpga
            g2lIdx = i * loadWidth
            if guardK:
              # for each component in vector
              for r in range(0, loadWidth*self.bpr/self.bpe):
                kStr += self.comment1("load component %u"%r)
                #kStr += dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)))
                #kStr += dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)))
                #kStr += "s_endpgm\n"
                # zero out data regardless of load or not
                for i in range(0, self.bpe/self.bpr):
                  kStr += inst("v_mov_b32", vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r*(self.bpe/self.bpr)+i)), hex(0), "zero")


                if kernel["BufferLoad"]:

                  # Compute the full address so we can 
                  # TODO - replace this with a direct compare against the offset 
                  kStr += inst("_v_add_co_u32",  
                      vgpr(fullAddrVgpr+0), \
                      "vcc", \
                      sgpr("Srd%s+%u"%(tP["tensorChar"], 0), 1), \
                      vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx),1), \
                      "Recompute full addr (lo)")
                  kStr += inst("v_mov_b32",  
                      vgpr(fullAddrVgpr+1), \
                      sgpr("Srd%s+%u"%(tP["tensorChar"], 1), 1), \
                      "full addr (upper)")
                  kStr += inst("_v_addc_co_u32",  
                      vgpr(fullAddrVgpr+1), \
                      "vcc", \
                      vgpr(fullAddrVgpr+1), \
                      0, \
                      "vcc", \
                      "Recompute full addr (upper)")
                      

                  # mask if current address if in bounds
                  kStr += inst("v_cmp_lt_u64", "vcc", \
                      vgpr(fullAddrVgpr,2), \
                      vgpr(maxAddr,2), \
                      "addr < maxAddr")

                  kStr += inst("v_cndmask_b32", \
                               vgpr(fullAddrVgpr), \
                               -1,
                               vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx),1), \
                               "vcc",
                               "Select offset or clip if OOB. Repurposing fullAddrVgpr+0 for offset")

                  # load single element from address
                  if kernel["ProblemType"]["DataType"].isHalf():
                    kStr += inst("buffer_load_short_d16%s"%("_hi" if r%2==1 else ""), \
                        vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r/2)),
                        vgpr(fullAddrVgpr), \
                        sgpr("Srd%s+%u"%(tP["tensorChar"], 0), 4), \
                        "0 offen offset:0",\
                        "load single f16")
                  elif kernel["ProblemType"]["DataType"].isSingle():
                    kStr += inst("buffer_load_dword", \
                        vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r)),
                        vgpr(fullAddrVgpr), \
                        sgpr("Srd%s+%u"%(tP["tensorChar"], 0), 4), \
                        "0 offen offset:0",\
                        "load single float")
                  elif kernel["ProblemType"]["DataType"].isDouble():
                    kStr += inst("buffer_load_dwordx2", \
                        vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r*2),2),
                        vgpr(fullAddrVgpr), \
                        sgpr("Srd%s+%u"%(tP["tensorChar"], 0), 4), \
                        "0 offen offset:0",\
                        "load single double")
                  else:
                    printWarning("DataType unsupported")

                  # increment offset by 1 element
                  kStr += inst("_v_add_co_u32", \
                      vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx),1), \
                      "vcc", \
                      vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx),1), \
                      vgpr(bpeVgpr), "graOffset += 1 (lower)")
                else:
                  # mask if current address if in bounds
                  kStr += inst("v_cmpx_lt_u64", "vcc", \
                      vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), \
                      vgpr(maxAddr,2), \
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

                  # increment address by 1 element
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
            else: # not guardK
              if kernel["BufferLoad"]:
                kStr += tP["globalReadInstruction"].toString( \
                    (vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), loadWidth), \
                    vgpr("GlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx),1), \
                    sgpr("Srd%s"%(tP["tensorChar"]), 4), 0), \
                    "G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp ), tP["NonTemporal"] )
              else:
                kStr += tP["globalReadInstruction"].toString( \
                    (vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), loadWidth), \
                    vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2)), \
                    "G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp ), tP["NonTemporal"] )

              #kStr += "s_waitcnt vmcnt(0)\n"
              #kStr += dump(vgpr("G2L%s+%u"%(tP["tensorChar"], graIdx)))
    if guardK:
      self.vgprPool.checkIn(bpeVgpr)
      self.vgprPool.checkIn(zeroVgpr)
      self.vgprPool.checkIn(tmpVgpr)
      if kernel["BufferLoad"]: # TODO - remove me, this is temp workaround
          self.vgprPool.checkIn(fullAddrVgpr)
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, tP):
    if not self.do["LocalWrite"]: return ""
    kStr = ""
    kStr += inst("v_xor_b32", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe), \
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
    kStr += inst("v_and_b32", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe-1), \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        "reset to Red")
    return kStr

  ##############################################################################
  # Local Write: Init Pointers A/B
  ##############################################################################
  def localWriteInitPointers(self, kernel, tP):
    return self.comment1("N/A")

  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  def localWriteDo(self, kernel, tP):
    if not self.do["LocalWrite"]: return ""
    kStr = ""
    instruction = tP["localWriteInstruction"]
    numBlocks = instruction.numBlocks
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    #offsetMultiplier = instruction.offsetMultiplier
    g2lIdx = 0
    #kStr += dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
    #print "\nLocalWrite", tP["tensorChar"]
    #print "tlu", tP["tlu"]
    #print "lsc", kernel[tP["lsc"]]
    #print "lsp", kernel[tP["lsp"]]
    #print "grcv", tP["grcv"]
    #print "wtc", tP["wtc"]
    #print "wuc", tP["wuc"]
    #print "nrp", tP["nrp"]
    #print "nrc", tP["nrc"]
    #print "nwcv", tP["nwcv"]
    #print "nwpv", tP["nwpv"]
    #print "nrcvpi", tP["nrcvpi"]
    #print "nwcvpi", tP["nwcvpi"]
    # if transposing, positions of sPerp and sPara are transposed
    for perp in range(0, tP["nrp"]):
      for para in range(0, tP["nrc"]):
        for s in range(0, max(tP["nwcv"],tP["nwpv"])/tP["nwcvpi"]):
          #print "  ", "perp", perp, "para", para, "s", s
          sPara = 0
          sPerp = 0
          lscaOffset = para * kernel[tP["lsc"]]
          lspaOffset = perp * kernel[tP["lsp"]]
          if tP["tlu"]:
            if tP["wtc"] == tP["grcv"]:
              sPerp = s
              lspaOffset += s
            elif tP["wuc"] == tP["grcv"]:
              sPara = s
              lscaOffset += s
            i = sPara + (tP["nrcv"]/tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
          else:
            if tP["wtc"] == tP["grcv"]:
              sPara = s
              lscaOffset += s
            elif tP["wuc"] == tP["grcv"]:
              sPerp = s
              lspaOffset += s
            i = sPara + (tP["nrcv"]/tP["nrcvpi"]) * (para * tP["glvw"] + tP["nrc"] * (sPerp + tP["glvw"] * tP["nrpv"] * perp ))
          #if not tP["tlu"]:
          #  tmp = sPara
          #  sPara = sPerp
          #  sPerp = tmp
          g2lIdx = i*blockWidth
          #print "0lspaOffset", lspaOffset
          #print "0lscaOffset", lscaOffset

          if tP["tlu"]:
            lspaOffset *= kernel[tP["mt"]]
          else:
            lscaOffset *= kernel[tP["mt"]]
          #print "1lspaOffset", lspaOffset
          #print "1lscaOffset", lscaOffset
          #if tP["tlu"] == tP["grcv"]:
          #  lspaOffset *= tP["glvw"]
          #  lscaOffset *= tP["glvw"]

          #print "2lspaOffset", lspaOffset
          #print "2lscaOffset", lscaOffset
          offsetElements = (lspaOffset + lscaOffset)
          #print "offsetElements", offsetElements
          offsetBytes = offsetElements*self.bpe
          #print "offsetBytes", offsetBytes
          #offset = offsetBytes*offsetMultiplier
          offset = offsetBytes*1
          #print "offset", offset

          comment = "lwo%s_%u_%u_%u_%u = (%s%d*%s)" \
              % (tP["tensorChar"], \
              para, sPara, perp, sPerp, \
              (("%u + "%sPara) if tP["wtc"] else ""), \
              para, tP["lsc"] )
          if not tP["tlu"]:
            comment += "*MT%s" % (tP["tileChar"])
          comment += " + (%s%d*%s)" % (
              (("%u + "%sPerp) if tP["wuc"] else ""), perp, \
              tP["lsp"])
          if tP["tlu"]:
            comment += "*MT%s" % (tP["tileChar"])
          comment += " = %u" % (offset)

          paramList = []
          paramList.append(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
          for blockIdx in range(0, numBlocks):
            if blockWidth == 1:
              paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx)))
            else:
              paramList.append( vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), \
                  blockWidth))
          for oIdx in range(0, numOffsets):
            paramList.append(offset)

          paramTuple = tuple(paramList)
          #comment = "Reg -> L %u_%u_%u_%u"%(para, sPara, perp, sPerp)
          nonTemporal = 0
          highBits = False
          if kernel["ProblemType"]["DataType"].isHalf():
            if s%2==1:
              highBits = True
          kStr += tP["localWriteInstruction"].toString(paramTuple, comment, \
              nonTemporal, highBits)
    #if tP["isB"]:
    #  kStr += self.dumpLds(kernel, 0, 8)
    #  kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  def localReadSwapOffsets(self, kernel, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    kStr += inst("v_xor_b32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe), \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A/B
  # x % n == n & (n-1) for n power of 2
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    if tP["localReadInstruction"].numOffsets == 1:
      tP["localReadOffset"] = 0
      kStr += self.comment1("handled internally")
    kStr += inst("v_and_b32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe-1), \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "reset Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  def localReadInitPointers(self, kernel, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    if self.localReadInstructionA.numOffsets == 1:
      tP["localReadOffset"] = 0
      kStr += self.comment1("N/A")
    else:
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffset%s_Blk"%tP["tensorChar"]]*self.bpe-1), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "reset Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  def localReadInc(self, kernel, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    if self.inTailLoop:
      inc = kernel["LocalSplitU"]*kernel["MacroTile%u"%tP["tensorIdx"]]*self.bpe
      tmpSgpr = self.getTmpSgpr(1)
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(inc), "inc")
      kStr += inst("_v_add_co_u32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "vcc", \
          sgpr(tmpSgpr), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "lr%s += %u"%(tP["tensorChar"], inc) )
    else:
      if tP["localReadInstruction"].numOffsets == 1:
        tP["localReadOffset"] += kernel["LocalSplitU"]*kernel["MacroTile%u"%tP["tensorIdx"]]
        kStr += self.comment1("N/A")
      else:
        inc = kernel["LocalSplitU"]*kernel["MacroTile%u"%tP["tensorIdx"]]
        kStr += inst("_v_add_co_u32", \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            "vcc", \
            hex(inc), \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            "lr%s += %u"%(tP["tensorChar"], inc) )
    return kStr

  ##############################################################################
  # Local Read: Do It A/B
  ##############################################################################
  def localReadDo(self, kernel, black, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    #kStr += dump(vgpr("Valu%s%s+%u"%("Blk" if black else "", tP["tensorChar"], 0)))
    instruction = tP["localReadInstruction"]
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    offsetMultiplier = 1 # instruction.offsetMultiplier
    #totalReads = (kernel["ThreadTile%u"%tP["tensorIdx"]]/blockWidth) / numOffsets
    valuIdx = 0
    numVectorsPerTile = (kernel["ThreadTile%u"%tP["tensorIdx"]]/kernel["VectorWidth"])
    #print "numVectorsPerTile", numVectorsPerTile
    numReadsPerVector = (kernel["VectorWidth"] * self.bpe ) / (blockWidth*4) # bytes/register
    #print "numReadsPerVector", numReadsPerVector
    for vIdx in range(0, numVectorsPerTile):
      for rIdx in range(0, numReadsPerVector):
        paramList = []
        if blockWidth == 1:
          paramList.append(vgpr("Valu%s%s+%u"%( \
              "Blk" if black else "", tP["tensorChar"], valuIdx)))
        else:
          paramList.append( vgpr("Valu%s%s+%u"%( \
              "Blk" if black else "", tP["tensorChar"],valuIdx), \
              blockWidth))
        paramList.append(vgpr("LocalReadAddr%s"%tP["tensorChar"]))
        for oIdx in range(0, numOffsets):
          paramList.append((rIdx*blockWidth + kernel["SubGroup%u"%tP["tensorIdx"]]*(vIdx*numOffsets+oIdx)*kernel["VectorWidth"] \
              + tP["localReadOffset"])*self.bpe/offsetMultiplier)
        paramTuple = tuple(paramList)
        comment = "L -> Reg %u"%rIdx
        kStr += instruction.toString(paramTuple, comment)
        valuIdx += blockWidth
    #if tP["isB"]:
    #  kStr += self.dumpLds(kernel, 0, 16)
    #  kStr += "s_endpgm\n"
    #if tP["isA"]:
    #kStr += "s_waitcnt lgkmcnt(0)\n"
    #if tP["isA"]:
    #  kStr += dump(vgpr("Valu%s%s+%u"%("Blk" if black else "", tP["tensorChar"], 0)))
    #if tP["isB"]:
    #  kStr += dump(vgpr("Valu%s%s+%u"%("Blk" if black else "", tP["tensorChar"], 0)))

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
    kStr += vectorStaticDivideAndRemainder(qReg, dummy, wgMT, divisor, \
        tmpVgpr, tmpSgpr)

    # rReg 
    rReg = self.vgprPool.checkOut(1)
    divisor = vw
    kStr += vectorStaticDivideAndRemainder(dummy, rReg, wgMT, divisor, \
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
      kStr += vectorStaticDivideAndRemainder(dummy, thread, "Serial", divisor, \
          tmpVgpr, tmpSgpr)
      #kStr += dump(vgpr(thread))
      #kStr += dump(vgpr(thread))
    else:
      # thread = (serial / SG0) % SG1
      sd0 = self.vgprPool.checkOut(1)
      divisor = kernel["SubGroup0"]
      kStr += vectorStaticDivideAndRemainder(sd0, dummy, "Serial", divisor, \
          tmpVgpr, tmpSgpr) # thread = serial / SG0
      divisor = kernel["SubGroup1"]
      thread = self.vgprPool.checkOut(1)
      kStr += vectorStaticDivideAndRemainder(dummy, thread, sd0, divisor, \
          tmpVgpr, tmpSgpr) # thread = (serial / SG0) % SG1
      self.vgprPool.checkIn(sd0)

    # which glvw vector of thread to shift? wgMT / (SG0*VW) -> (wgMT%VW) / glvw
    # (wgMT/(WG0*VW))*(VW/glvw) + (wgMT%VW) / glvw
    if tP["tt"] > kernel["VectorWidth"]:
      mvReg = self.vgprPool.checkOut(1)
      divisor = kernel[tP["sg"]]*kernel["VectorWidth"]
      kStr += vectorStaticDivideAndRemainder(mvReg, dummy, wgMT, divisor, \
          tmpVgpr, tmpSgpr)
      if vw < kernel["VectorWidth"]:
        kStr += inst("v_lshlrev_b32", vgpr(mvReg), hex(log2(kernel["VectorWidth"]/vw)), vgpr(mvReg), "vId *= VW/glvw")
    #kStr += dump(vgpr(mvReg))

    vReg = self.vgprPool.checkOut(1)
    divisor = kernel["VectorWidth"]
    kStr += vectorStaticDivideAndRemainder(dummy, vReg, wgMT, divisor, \
        tmpVgpr, tmpSgpr)
    vRegD = self.vgprPool.checkOut(1)
    kStr += inst("v_mov_b32", vgpr(vRegD), vgpr(vReg), "duplicate")
    divisor = vw
    kStr += vectorStaticDivideAndRemainder(vReg, dummy, vRegD, divisor, \
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
            if self.bpe == 2:
              srcVgpr = self.startVgprValuC+src*self.bpe/self.bpr
              dstVgpr = self.startVgprValuC+dst*self.bpe/self.bpr
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
              for i in range(0, self.bpe/self.bpr):
                kStr += inst("v_mov_b32", vgpr(self.startVgprValuC+dst*self.bpe/self.bpr+i), \
                    vgpr(self.startVgprValuC+src*self.bpe/self.bpr+i), comment)

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
    #kStr += inst("s_nop", "5", "wait for exec update")
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
  # LocalSplitU: Local Write
  ##############################################################################
  def localSplitULocalWrite(self, kernel):
    kStr = ""
    # wait for summation to be done with lds before writing reduction values
    kStr += self.syncThreads(kernel)

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
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["VectorWidth"]*self.bpe), "VW")
    kStr += inst("v_mul_lo_u32", vgpr(lr0), sgpr(tmpSgpr), vgpr(lr0), \
        "lr0 *= VW")
    # lr1 *= VW*MT0
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), \
        hex(kernel["VectorWidth"]*kernel["MacroTile0"]*self.bpe), "VW*MT0")
    kStr += inst("v_mul_lo_u32", vgpr(lr1), sgpr(tmpSgpr), vgpr(lr1), \
        "lr1 *= VW*MT0")
    # sg  *= MT0*MT1
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), \
        hex(kernel["MacroTile0"]*kernel["MacroTile1"]*self.bpe), "MT0*MT1")
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

    # dump ds_write addr
    #kStr += dump(vgpr(addr))

    # do writes
    for j in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          for vc in range(0, kernel["VectorWidth"]):
            writeOffset = vc \
                + i*kernel["SubGroup0"]*kernel["VectorWidth"] \
                + s*kernel["MacroTile0"] \
                + j*kernel["MacroTile0"]*kernel["SubGroup1"]*kernel["VectorWidth"]
            regIdx = vc \
                + i*kernel["VectorWidth"] \
                + s*kernel["ThreadTile0"] \
                + j*kernel["ThreadTile0"]*kernel["VectorWidth"]
            kStr += "ds_write_b32 %s, %s offset:%u%s" \
                % (vgpr(addr), vgpr(regIdx), writeOffset*self.bpe, self.endLine)
            # ds_write value
            #kStr += dump(vgpr(regIdx))
    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for all writes")
    kStr += self.syncThreads(kernel)
    #kStr += self.dumpLds(kernel, 0, 16)
    return kStr

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    kStr = ""
    tmpSgpr = self.getTmpSgpr(1)
    baseAddr = self.vgprPool.checkOut(1)
    kStr += staticMultiply(vgpr(baseAddr), vgpr("Serial"), kernel["GlobalWriteVectorWidth"]*self.bpe, sgpr(tmpSgpr))
    for r in range(0, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"]):
          offset = s + i*kernel["NumThreads"]*kernel["GlobalWriteVectorWidth"] + r * kernel["MacroTile0"]*kernel["MacroTile1"]
          regIdx = s + i*kernel["GlobalWriteVectorWidth"] + r*kernel["GlobalWriteVectorWidth"]*kernel["NumGlobalWriteVectorsPerThread"]
          kStr += "ds_read_b32 %s, %s offset:%u%s" % (vgpr("ValuC+%u"%regIdx), \
              vgpr(baseAddr), offset*self.bpe, self.endLine)
    kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for all reads")
    self.vgprPool.checkIn(baseAddr)
    return kStr

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  def localSplitUReduction(self, kernel):
    kStr = ""
    for r in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"]):
          cIdx = s + i*kernel["GlobalWriteVectorWidth"]
          regIdx = s + i*kernel["GlobalWriteVectorWidth"] \
              + r*kernel["GlobalWriteVectorWidth"]*kernel["NumGlobalWriteVectorsPerThread"]
          kStr += inst("v_add_f32", vgpr("ValuC+%u"%cIdx), \
              vgpr("ValuC+%u" % regIdx), vgpr("ValuC+%u"%cIdx), "c[%u] += c[%u]"%(cIdx, regIdx) )
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    kStr = ""

    # tmp gprs
    tmpVgpr = self.vgprPool.checkOut(2)
    tid0 = self.vgprPool.checkOut(1)
    tid1 = self.vgprPool.checkOut(1)
    tmpSgpr = self.getTmpSgpr(1)
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
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    #print "GlobalWriteIndices"
    if not self.do["PostLoop"]: return ""
    kStr = ""

    self.scratchSgprs = self.getTmpSgpr(1)

    tmpS0 = self.scratchSgprs
    tmpS1 = tmpS0+1
    tmpV0 = self.vgprPool.checkOut(2)

    # thread id 0,1
    tid0 = self.vgprPool.checkOut(1)
    tid1 = self.vgprPool.checkOut(1)
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
  # Not LocalSplitU: Global Write
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""
    lsu = False
    elements = []
    for tt1 in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for tt0 in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for vc1 in range(0, kernel["VectorWidth"]):
          for vc0 in range(0, kernel["VectorWidth"]):
              element = (tt1, tt0, vc1, vc0)
              elements.append(element)
    return self.globalWriteElements(kernel, lsu, elements)

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""
    lsu = True
    elements = []
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for tt0 in range(0, 1):
        for vc1 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"]):
            element = (tt1, tt0, vc1, vc0)
            elements.append(element)
    return self.globalWriteElements(kernel, lsu, elements)

  ##############################################################################
  # Global Write Elements
  ##############################################################################
  def globalWriteElements(self, kernel, lsu, elements ):
    if not self.do["PostLoop"]: return ""
    kStr = ""
    atomic = kernel["GlobalSplitU"] > 1


    # write possibilities and labels
    betas = [False, True] if kernel["ProblemType"]["UseBeta"] else [False]
    edges = [False, True]
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
    if kernel["ProblemType"]["DataType"].isHalf():
      self.alphaVgpr = self.vgprPool.checkOut(1)
      self.betaVgpr = self.vgprPool.checkOut(1)
      kStr += inst("v_mov_b32", vgpr(self.alphaVgpr), sgpr("Alpha"), "sgpr -> vgpr b/c op_sel")
      kStr += inst("v_mov_b32", vgpr(self.betaVgpr), sgpr("Beta"), "sgpr -> vgpr b/c op_sel")

    ########################################
    # Vgprs
    tmpVgpr = self.vgprPool.checkOut(2+3) # 2 for coord and 3 for GLOBAL_OFFSET_C

    ########################################
    # Sgprs
    globalWriteSgprs = self.getTmpSgpr(2)
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
      if self.bpe <= self.bpr: # 1 register to check for Beta==0
        kStr += inst("s_cmpk_eq_u32", sgpr("Beta"), hex(0), "Beta == 0")
      else: # multiple registers to check for Beta==0
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), sgpr("Beta+0"), "tmp = Beta[0]")
        for i in range(1, self.bpe/self.bpr):
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
      kStr += inst("s_cmpk_gt_u32", sgpr(tmpS01), hex(0), "rMT1 > 0")
      kStr += inst("s_cbranch_scc1 label_%04u" % writeLabels[beta][True], \
          "edges required so jump to E1")
      # by now we either jumped to E1 or stayed at E0
      for edge in edges:
        kStr += "label_%04u:%s"%(writeLabels[beta][edge], self.endLine)

        if edge:
          # store free sizes in vgprs for comparison
          sizesFreeVgprs = self.vgprPool.checkOut(2)
          kStr += inst("v_mov_b32", \
              vgpr(sizesFreeVgprs+0), \
              sgpr("SizesFree+0"), \
              "free sizes sgpr -> vgpr")
          kStr += inst("v_mov_b32", \
              vgpr(sizesFreeVgprs+1), \
              sgpr("SizesFree+1"), \
              "free sizes sgpr -> vgpr")
        else:
          sizesFreeVgprs = None

        ########################################
        # Calculate Vgprs for Write Batching
        ########################################

        numElementSgprs = self.totalSgprs - elementSgprs
        numSgprsPerElement = 2
        numElementsPerBatchLimitedBySgprs = numElementSgprs / numSgprsPerElement
        # how many vgprs are needed for zero elements
        # 2 for addressC in vgpr for addition - already checked out
        # 2 for coord0,1 of thread - already checked out
        # 2 for tmp - already checked out

        # 5 = how many vgprs are needed per element
        # 2 for addr
        # 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
        # if beta 1*rpe for new value
        # if atomic 2*rpe for old and cmp values
        numVgprsPerElement = 2
        if atomic:
          numVgprsPerElement += (3*self.bpe)/self.bpr
        elif beta:
          if self.bpe >= self.bpr:
            numVgprsPerElement += (1*self.bpe)/self.bpr
          else:
            numVgprsPerElement += (1.0*self.bpe)/self.bpr

        #print self.vgprPool.state()
        numVgprAvailable = self.vgprPool.available()
        #print "NumVgprAvailable", numVgprAvailable
        numElementsPerBatch = numVgprAvailable / numVgprsPerElement
        #print "NumElementsPerBatch", numElementsPerBatch, "LimitedBySgprs", numElementsPerBatchLimitedBySgprs, "WARNING" if numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay"
        if numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
          numElementsPerBatch = numElementsPerBatchLimitedBySgprs 

        if kernel["ProblemType"]["DataType"].isHalf():
          # only do an even number of halves
          numElementsPerBatch = int(numElementsPerBatch/2)*2

        # if no atomics and no edge, then write whole vectors
        #if not atomic and not edge:
        #  numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
        #  #print "  NumVectorsPerBatch", numVectorsPerBatch
        #  numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
        numBatches = max(1, (len(elements)+numElementsPerBatch-1) / numElementsPerBatch)
        #print "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch
        for batchIdx in range(0, numBatches):
          elementStartIdx = batchIdx * numElementsPerBatch
          elementStopIdx = min( elementStartIdx + numElementsPerBatch, len(elements) )
          elementsThisBatch = elements[elementStartIdx:elementStopIdx]
          #print "BATCH[%u/%u]: elements[%u:%u]" % (batchIdx, numBatches, elementStartIdx, elementStopIdx)
          numElementsThisBatch = len(elementsThisBatch)
          numElementVgprs = int(numElementsThisBatch * ceil(numVgprsPerElement))
          elementVgprs = self.vgprPool.checkOut(numElementVgprs)
          kStr += self.globalWriteBatch(kernel, beta, edge, lsu, atomic, \
              elementsThisBatch, self.coord0, self.coord1, self.addrC, \
              sizesFreeVgprs, elementVgprs, numVgprsPerElement, tmpVgpr, \
              fullExecMaskSgpr, elementSgprs, numSgprsPerElement, tmpSgpr)
          self.vgprPool.checkIn(elementVgprs)

        kStr += inst("s_branch", "label_%04u"%endLabel, "jump to end")
        if edge:
          self.vgprPool.checkIn(sizesFreeVgprs)

    # End label
    kStr += "label_%04u:%s"%(endLabel, self.endLine)
    self.vgprPool.checkIn(tmpVgpr)
    return kStr

  ##############################################################################
  # Global Write Batch
  ##############################################################################
  def globalWriteBatch(self, kernel, beta, edge, lsu, atomic, \
      batchElements, coord0, coord1, addrC, sizes, \
      batchElementVgprs, numVgprsPerElement, tmpVgpr, \
      fullExecMaskSgpr, batchElementSgprs, numSgprsPerElement, tmpSgpr):
    kStr = ""

    # comment
    commentStr = "Global Write%s%s Batch:" \
        % (" Beta" if beta else "", " Edge" if edge else "")
    for elementIdx in range(0, len(batchElements)):
      element = batchElements[elementIdx]
      commentStr += "(%u,%u,%u,%u)" % element
      if elementIdx < len(batchElements)-1:
        commentStr += "; "
    kStr += self.comment3(commentStr)

    ########################################
    # allocate per-element resources
    numVgprsPerAddr = self.rpga
    numVgprsPerData = numVgprsPerElement - numVgprsPerAddr # might be decimal for half
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
      data = batchElementVgprs + dataVgprOffset + int(elementIdx*numVgprsPerData) # elementVgprs+self.rpga
      elementData.append(data)
      mask = batchElementSgprs + elementIdx * numSgprsPerElement # elementSgprs+0
      elementMask.append(mask)

      element = batchElements[elementIdx]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]
      if lsu:
        sumIdx = self.startVgprValuC + vc0 + d1*kernel["VectorWidth"]
      else:
        sumIdx = self.startVgprValuC + vc0 + d0*kernel["VectorWidth"] + vc1*kernel["ThreadTile0"] + d1*kernel["VectorWidth"]*kernel["ThreadTile0"]
      elementSumIdx.append(sumIdx)

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

      # coord0
      kStr += staticMultiply(vgpr(tmpVgpr+0), d0, (kernel["SubGroup0"]*kernel["VectorWidth"]))
      kStr += inst("_v_add_co_u32", vgpr(tmpVgpr+0), "vcc", vc0, vgpr(tmpVgpr+0), \
          "tmp0 = d0*sg0*VW + vc0")
      kStr += inst("_v_add_co_u32", vgpr(tmpVgpr+0), "vcc", vgpr(coord0), vgpr(tmpVgpr+0), \
          "coord0 += d0*sg0*VW + vc0")

      # coord1
      kStr += staticMultiply(vgpr(tmpVgpr+1), d1, strideD1)
      kStr += inst("_v_add_co_u32", vgpr(tmpVgpr+1), "vcc", hex(vc1), vgpr(tmpVgpr+1), \
          "tmp1 = d1*sg1*VW + vc1")
      kStr += inst("_v_add_co_u32", vgpr(tmpVgpr+1), "vcc", vgpr(coord1), vgpr(tmpVgpr+1), \
          "coord1 += d1*sg1*VW + vc1")
      #kStr += dump(vgpr(tmp+1))

      # in-bounds exec mask
      if edge:
        kStr += inst("v_cmp_lt_u32",  sgpr(tmpS01,2), vgpr(tmpVgpr+0), vgpr(sizes+0), "coord0 < size0" )
        kStr += inst("v_cmp_lt_u32",  sgpr(tmpS23,2), vgpr(tmpVgpr+1), vgpr(sizes+1), "coord1 < size1" )
        kStr += inst("s_and_b64",  sgpr(mask,2), sgpr(tmpS01,2), sgpr(tmpS23,2), "in0 && in1" )

      if edge and (beta or atomic):
        # apply in-bounds exec mask for read
        kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )


      # global offset macro (requires 3 tmpVgpr)
      kStr += "GLOBAL_OFFSET_C %u" % addr
      for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
        if i == kernel["ProblemType"]["Index0"]:
          kStr += ", %s" % (tmpVgpr+0)
        elif i == kernel["ProblemType"]["Index1"]:
          kStr += ", %s" % (tmpVgpr+1)
        else: # just a group index
          kStr += ", sgprWorkGroup%u"%i
      kStr += ", %s%s" % ((tmpVgpr+2), self.endLine)

      # final address = C + index*bytes
      kStr += inst("_v_add_co_u32",  vgpr(addr+0), "vcc", vgpr(addrC+0), \
          vgpr(addr+0), "addr = C + index*bytes (lo)" )
      kStr += inst("_v_addc_co_u32", vgpr(addr+1), "vcc", vgpr(addrC+1), \
          vgpr(addr+1), "vcc", "addr = C + index*bytes (hi)")

      if atomic:
        # load c into data+1 becaue of CAS structure
        kStr += inst("flat_load_dword", vgpr(data+1), vgpr(addr,2), \
            "load C" )
      elif beta:
        # load c into data+0
        if kernel["ProblemType"]["DataType"].isHalf():
          if sumIdx%2:
            kStr += inst("flat_load_short_d16_hi", vgpr(data+0), vgpr(addr,2), "load C" )
          else:
            kStr += inst("flat_load_short_d16", vgpr(data+0), vgpr(addr,2), "load C" )
        elif kernel["ProblemType"]["DataType"].isSingle():
          kStr += inst("flat_load_dword", vgpr(data+0), vgpr(addr,2), "load C" )
        elif kernel["ProblemType"]["DataType"].isDouble():
          kStr += inst("flat_load_dwordx2", vgpr(data+0,2), vgpr(addr,2), "load C" )

      # restore full exec mask for calculating addr of next element
      if edge and (beta or atomic):
        #kStr += inst("s_or_saveexec_b64",  sgpr(tmpS45,2), sgpr(fullExecMaskSgpr,2), "full mask -> exec" )
        kStr += inst("s_mov_b64", "exec", sgpr(fullExecMaskSgpr,2), "full mask -> exec" )

    ########################################
    # rC *= alpha
    kStr += self.comment("rC *= alpha")
    for elementIdx in range(0, len(batchElements)):
      sumIdx = elementSumIdx[elementIdx]
      if kernel["ProblemType"]["DataType"].isHalf():
        if sumIdx%2:
          kStr += inst("v_pk_mul_f16", vgpr(sumIdx/2), vgpr(self.alphaVgpr), vgpr(sumIdx/2), "*= alpha")
      elif kernel["ProblemType"]["DataType"].isSingle():
        kStr += inst("v_mul_f32", vgpr(sumIdx), sgpr("Alpha"), vgpr(sumIdx), "*= alpha" )
      elif kernel["ProblemType"]["DataType"].isDouble():
        kStr += inst("v_mul_f64", vgpr(sumIdx*2,2), sgpr("Alpha",2), vgpr(sumIdx*2,2), "*= alpha")

    ########################################
    # Atomic
    ########################################
    # flat_atomic_cmpswap tmp addr data
    # tmp = mem[addr]
    # src = data[0] new C
    # cmp = data[1] original C
    # mem[addr] = (tmp==cmp) ? src : tmp
    # addr = vgpr(addr,2)
    # data = vgpr(tmpVgpr,2)
    # tmp = vgpr(tmpVgpr+4)
    if atomic:

      # atomic loop label
      element = batchElements[0]
      d1 = element[0]
      d0 = element[1]
      vc1 = element[2]
      vc0 = element[3]
      labelString = "Global_Write%s%s_vc=%u,%u_d=%u,%u" \
        % (" Beta" if beta else "", " Edge" if edge else "", vc0, vc1, d0, d1 )
      label = self.getLabel(labelString)
      labelString += "EarlyExid"
      labelAfterAtomicLoop = self.getLabel(labelString)

      ########################################
      # wait for batched load
      if beta or atomic:
        kStr += inst("s_waitcnt", "vmcnt(0)", "wait C" )

      ########################################
      # first attempt write
      kStr += self.comment("issue first atomic writes")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = elementAddr[elementIdx]
        data = elementData[elementIdx]
        tmpVgpr = data+2
        mask = elementMask[elementIdx]
        sumIdx = elementSumIdx[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]

        # apply in-bounds exec mask
        if edge:
          #kStr += inst("s_and_saveexec_b64",  sgpr(tmpS45,2), sgpr(mask,2), "sgprs -> exec" )
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )

        # for atomic, data[1] = original c, data[0] = new c
        kStr += inst("v_add_f32", vgpr(data+0), vgpr(data+1), vgpr(sumIdx), \
            "sum*alpha + C*beta")

        # attempt write
        kStr += "flat_atomic_cmpswap %s, %s, %s %s    // %s%s" % ( vgpr(tmpVgpr), vgpr(addr,2), \
            vgpr(data,2), "glc", "attempt write", self.endLine )

      ########################################
      # wait for first attempt write
      kStr += inst("s_waitcnt vmcnt(0)", "wait for atomic writes" )

      ########################################
      # check first attempt
      kStr += self.comment("check success of writes, update masks")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = elementAddr[elementIdx]
        data = elementData[elementIdx]
        tmpVgpr = data+2
        mask = elementMask[elementIdx]
        sumIdx = elementSumIdx[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]

        # calculate new masks
        if edge:
          # need to apply element mask before comparison
          # so that all valid lanes are doing the cmp
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )
          kStr += inst("v_cmp_ne_u32", sgpr(tmpS01,2), vgpr(tmpVgpr), \
              vgpr(data+1), "c read during atomic == c read during prior load" )
          kStr += inst("s_and_b64",  sgpr(mask,2), sgpr(tmpS01,2), sgpr(mask,2), "inBounds & must try again" )
        else:
          #kStr += inst("s_mov_b64", sgpr(mask,2), sgpr(fullExecMaskSgpr,2), "mask = full" )
          kStr += inst("v_cmp_ne_u32", sgpr(mask,2), vgpr(tmpVgpr), \
              vgpr(data+1), "c read during atomic != c read during prior load" )

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
        data = elementData[elementIdx]
        tmpVgpr = data+2
        mask = elementMask[elementIdx]
        sumIdx = elementSumIdx[elementIdx]

        # apply mask for element
        kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "must try again" )
        kStr += inst("v_mov_b32", vgpr(data+1), vgpr(tmpVgpr), "data+1 = tmp (new original C)" )
        kStr += inst("v_add_f32", vgpr(data+0), vgpr(sumIdx), vgpr(data+1), \
            "newC = rC + originalC" )
        kStr += "flat_atomic_cmpswap %s, %s, %s %s    // %s%s" % ( vgpr(tmpVgpr), \
            vgpr(addr,2), vgpr(data,2), "glc", "try again", self.endLine)

      # wait for batched write
      kStr += inst("s_waitcnt vmcnt(0)", "wait for atomic writes" )

      # check batched write success
      kStr += self.comment("apply masks and check for success")
      for elementIdx in range(0, len(batchElements)):
        element = batchElements[elementIdx]
        addr = elementAddr[elementIdx]
        data = elementData[elementIdx]
        tmpVgpr = data+2
        mask = elementMask[elementIdx]
        sumIdx = elementSumIdx[elementIdx]

        # apply mask for element
        kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "must try again" )

        # compare success
        kStr += inst("v_cmp_ne_u32", sgpr(tmpS01,2), vgpr(data+1), vgpr(tmpVgpr), \
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
        data = elementData[elementIdx]
        mask = elementMask[elementIdx]
        sumIdx = elementSumIdx[elementIdx]
        d1 = element[0]
        d0 = element[1]
        vc1 = element[2]
        vc0 = element[3]

        #if beta: # FIXME kept above since flat instruction may return out of order
        #  kStr += inst("s_waitcnt", "vmcnt(%u)"%(len(batchElements)-1), "wait C")

        # apply in-bounds exec mask
        if edge:
          kStr += inst("s_mov_b64", "exec", sgpr(mask,2), "sgprs -> exec" )

        if beta:
          if kernel["ProblemType"]["DataType"].isHalf():
            if sumIdx%2==0:
              # data+0 = new c = old c*beta
              kStr += inst("v_pk_mul_f16", vgpr(data+0), vgpr(self.betaVgpr), vgpr(data+0), \
                  "%s = C*beta"%vgpr(data+0))
              # data+0 = new c = old c*beta + rC
              kStr += inst("v_pk_add_f16", vgpr(sumIdx/2), vgpr(data+0), vgpr(sumIdx/2), \
                  "sum*alpha + C*beta")
            else:
              pass # add will have been done previously
          elif kernel["ProblemType"]["DataType"].isSingle():
            # data+0 = new c = old c*beta
            kStr += inst("v_mul_f32", vgpr(data+0), sgpr("Beta"), vgpr(data+0), \
                "%s = C*beta"%vgpr(data+0) )
            # data+0 = new c = old c*beta + rC
            kStr += inst("v_add_f32", vgpr(sumIdx), vgpr(data+0), vgpr(sumIdx), \
                "sum*alpha + C*beta")
          elif kernel["ProblemType"]["DataType"].isDouble():
            # data+0 = new c = old c*beta
            kStr += inst("v_mul_f64", vgpr(data+0,2), sgpr("Beta",2), vgpr(data+0,2), \
                "%s = C*beta"%vgpr(data+0,2) )
            # data+0 = new c = old c*beta + rC
            kStr += inst("v_add_f64", vgpr(sumIdx*2,2), vgpr(data+0,2), vgpr(sumIdx*2,2), \
                "sum*alpha + C*beta")

        nonTemporalStr = ""
        if kernel["NonTemporalC"]%2==1:
          nonTemporalStr += " glc"
        if kernel["NonTemporalC"]/2==1:
          nonTemporalStr += " slc"
        if kernel["ProblemType"]["DataType"].isHalf():
          if sumIdx%2:
            kStr += inst("flat_store_short_d16_hi", vgpr(addr,2), vgpr(sumIdx/2), "store C" )
          else:
            kStr += inst("flat_store_short", vgpr(addr,2), vgpr(sumIdx/2), "store C" )
        elif kernel["ProblemType"]["DataType"].isSingle():
          kStr += "flat_store_dword %s, %s%s // store C\n" % ( vgpr(addr,2), vgpr(sumIdx), nonTemporalStr )
        elif kernel["ProblemType"]["DataType"].isDouble():
          kStr += "flat_store_dwordx2 %s, %s%s  // store C\n" % ( vgpr(addr,2), vgpr(sumIdx*2,2), nonTemporalStr )

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
    if self.vgprPool.size() > self.maxVgprs:
      self.overflowedResources = True
    if self.overflowedResources:
      kStr += ".endif // too many vgprs\n"
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
  ##############################################################################
  def wait(self, kernel, tPA, tPB, skipGlobalRead, skipLocalWrite, \
      skipLocalRead, comment):
    if not self.do["Wait"]: return ""
    # skip = -1 -> ignore
    # skip =  n -> waitcnt(n*num)

    lgkmcnt = 0 if skipLocalWrite > -1 or skipLocalRead > -1 else -1

    if skipLocalWrite > -1 or skipLocalRead > -1:
      if skipLocalWrite > -1:
        numA = tPA["nrp"]*tPA["nrc"]*max(tPA["nwcv"],tPA["nwpv"])/tPA["nwcvpi"]
        numB = tPB["nrp"]*tPB["nrc"]*max(tPB["nwcv"],tPB["nwpv"])/tPB["nwcvpi"]
        lgkmcnt += skipLocalWrite * (numA + numB)
      if skipLocalRead > -1:
        numA = (kernel["ThreadTile0"] / kernel["VectorWidth"]) \
            / self.localReadInstructionA.numOffsets
        numB = (kernel["ThreadTile1"] / kernel["VectorWidth"]) \
            / self.localReadInstructionB.numOffsets
        lgkmcnt += skipLocalRead * (numA + numB)

    vmcnt = 0 if skipGlobalRead > -1 else -1
    if skipGlobalRead > -1:
      numA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"] \
          * self.numReadVectorComponentsA
      numB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"] \
          * self.numReadVectorComponentsB
      vmcnt += skipGlobalRead * (numA + numB)
      if lgkmcnt > -1:
        lgkmcnt += skipGlobalRead * (numA + numB)

    if False:
      return "s_waitcnt lgkmcnt(0) & vmcnt(0) // debug%s" % self.endLine

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
  def syncThreads(self, kernel):
    if kernel["NumThreads"] > 64 and self.do["Sync"]:
      return self.indent + self.syncStr + self.endLine
    else:
      return ""


  ########################################
  # dump lds state
  ########################################
  def dumpLds(self, kernel, startU, numU):
    kStr = ""
    if globalParameters["DebugKernel"]:
      kStr += self.comment("dump lds state")
      kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
      kStr += inst("s_barrier", "" )
      tmp = self.vgprPool.checkOut(1)
      tmpAddr = self.vgprPool.checkOut(1)
      kStr += inst("v_lshlrev_b32", \
          vgpr(tmpAddr), \
          hex(log2(self.bpe)), \
          vgpr("Serial"), \
          "dump lds")
      for i in range(startU, startU+numU):
        kStr += inst("ds_read_b32", vgpr(tmp), \
            vgpr(tmpAddr) + " offset:%u"%(i*kernel["NumThreads"]*4), "dump lds")
        kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "dump" )
        kStr += dump(vgpr(tmp))
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



################################################################################
# Helper Functions
################################################################################

########################################
# Store to Debug Buffer
########################################
def dump(vgprStore):
  kStr = ""
  if globalParameters["DebugKernel"]:
    kStr += inst("flat_store_dword", vgpr("AddressD", 2), \
        vgprStore, "debug dump store" )
    kStr += inst("_v_add_co_u32", vgpr("AddressD"), "vcc", vgpr("AddressD"), \
        hex(4), "debug dump inc" )
  return kStr


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
        "%s = %s / %u"%(vgpr(qReg), vgpr(dReg), divisor) )
    if doRemainder:
      kStr += inst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), \
          "%s = %s %% %u"%(vgpr(rReg), vgpr(dReg), divisor) )
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
    kStr += inst("v_mov_b32", vgpr(qReg), vgpr(tmpVgpr), "quotient")
    if doRemainder:
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), "divisor")
      kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(qReg), sgpr(tmpSgpr), "product = quotient * divisor")
      kStr += inst("_v_sub_co_u32", vgpr(rReg), "vcc", vgpr(dReg), vgpr(tmpVgpr), "remainder = dividend - product")
  return kStr

def vectorStaticDivide(qReg, dReg, divisor, tmpVgpr, tmpSgpr):
  rReg = -1 # unused
  kStr = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, False)
  return kStr

# only used for loop unroll and GlobalSplitU
def scalarStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr, \
    doRemainder=True):
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

    kStr += inst("s_mov_b32", sgpr(tmpSgpr+1), hex(0), "hi = 0")
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
    return inst("v_lshlrev_b32", product, multiplier_log2, operand, \
        "%s = %s * %u"%(product, operand, multiplier) )
  else:
    kStr = ""
    if product == operand:
      kStr += inst("s_mov_b32", tmpSgpr, hex(multiplier), \
        "%s = %u"%(tmpSgpr, multiplier) )
      kStr += inst("v_mul_lo_u32", product, tmpSgpr, operand, \
        "%s *= %s"%(product, operand) )
    else:
      kStr += inst("v_mov_b32", product, hex(multiplier), \
        "%s = %u"%(product, multiplier) )
      kStr += inst("v_mul_lo_u32", product, product, operand, \
        "%s *= %s"%(product, operand) )
    return kStr
