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

# TODO
# (1) explicit and implicit incrementing of global/local offsets to handle
# to handle when implicit offset spills, attach offsets to a register, when
# we increment that offset and it spills, then it automatically triggers an
# explicit increment and the implicit increment resets
# Address class, request explicit or implicit increment, request reset
#  32,64-bit, which registers, pass into MemoryInstruction class
# (2) MemoryPool to handle all allocations
# (3) Dictionary of gpr names?
# (4) Will I need to use fewer sgprs?
# (6) Add and multiply functions, compile time known, handles carrying, only need to debug in one place


from SolutionStructs import DataType
from Common import globalParameters, print1, print2, printExit, printWarning
from KernelWriter import KernelWriter
from math import log
import abc

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
  def toString(self, params, comment):
    instStr = "%s %s" % (self.name, (self.formatting % params) )
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
    #print "RegisterPool::checkOutAligned(%u,%u)"%(size,alignment)
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
      return start

  ########################################
  # Check In
  def checkIn(self, start):
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
    self.globalReadIncsUseVgpr = True # slightly fewer [v_mov] instructions but extra registers

    # ISA version, such as 803
    self.version = globalParameters["CurrentISA"]
    self.maxVgprs = 256

    ########################################
    # Available Memory Instructions
    ########################################

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
    ########################################
    # Global Read
    flat_load_dwordx4 = MemoryInstruction("flat_load_dwordx4",  1, 0, 0, 4, \
        "%s, %s" )
    flat_load_dwordx2 = MemoryInstruction("flat_load_dwordx2",  1, 0, 0, 2, \
        "%s, %s" )
    flat_load_dword = MemoryInstruction("flat_load_dword",      1, 0, 0, 1, \
        "%s, %s" )
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
    self.memoryInstructions = {
        (8,0,3): {
          "GlobalRead": [ flat_load_dwordx4, flat_load_dwordx2,
            flat_load_dword ],
          "GlobalWrite": [ flat_store_dwordx4, flat_store_dwordx2,
            flat_store_dword ],
          "LocalRead": [ ds_read_b128, ds_read2_b64,
            ds_read_b64, ds_read2_b32, ds_read_b32 ],
          "LocalWrite": [ ds_write_b128, ds_write2_b64,
            ds_write_b64, ds_write2_b32, ds_write_b32 ]
          }, # 803
        (9,0,0): {
          "GlobalRead": [ flat_load_dwordx4, flat_load_dwordx2,
            flat_load_dword ],
          "GlobalWrite": [ flat_store_dwordx4, flat_store_dwordx2,
            flat_store_dword ],
          "LocalRead": [ ds_read_b128, ds_read2_b64,
            ds_read_b64, ds_read2_b32, ds_read_b32 ],
          "LocalWrite": [ ds_write_b128, ds_write2_b64,
            ds_write_b64, ds_write2_b32, ds_write_b32 ]
          } # 900
        }

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
      name = instruction.name
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
      return self.startSgprOffsetC
    else:
      return ((self.startSgprOffsetC+1)/2)*2

# TODO: option: when offset bits aren't sufficient, do we use VALU to
# increment address or do we use extra registers to store addresses?
# (1) write1 and aways have sufficient offset bits
# (2) write2 and if insufficient offset bits then IncrementAndReset
# (3) write2 and if insufficient offset bits then AllocateAdditionalAddresses

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

    if "ISA" in kernel:
      self.version = kernel["ISA"]

    self.kernelName = self.getKernelName(kernel)
    self.inTailLoop = False

    # registers per element
    self.rpe = kernel["ProblemType"]["DataType"].numRegisters()
    self.bpe = self.rpe * 4
    # registers per global address
    self.rpga = 2 # 64-bit
    # registers per local address
    self.rpla = 1 # 32-bit

    ####################################
    # choose memory instructions
    ####################################

    ########################################
    # globalReadA instruction; no flat_load2_*
    self.globalReadWidthA = tPA["nrcv"]
    #self.globalReadWidthA = kernel["VectorWidth"] if (self.readTileDimVectorA \
    #    or self.readUnrollDimVectorA) else 1
    self.globalReadWidthA *= self.rpe
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
    self.globalReadWidthB = tPB["nrcv"]
    #self.globalReadWidthB = kernel["VectorWidth"] if (self.readTileDimVectorB \
    #    or self.readUnrollDimVectorB) else 1
    self.globalReadWidthB *= self.rpe
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
    self.localWriteWidthA = tPA["nwcv"]
    self.localWriteWidthA *= self.rpe
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
    self.localWriteStrideTileA *= self.rpe
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
    self.localWriteStrideUnrollA *= self.rpe
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
    self.localWriteWidthB = tPB["nwcv"]
    self.localWriteWidthB *= self.rpe
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
    self.localWriteStrideTileB *= self.rpe
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
    self.localWriteStrideUnrollB *= self.rpe
    self.localWriteInstructionIdxB = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthB, \
        kernel["LocalWrite2B"], \
        self.localWrite2CoalescedB, self.localWrite2PerpendicularB,
        [self.localWriteStrideTileB, self.localWriteStrideUnrollB] )

    ########################################
    # localRead A
    self.localReadWidth = kernel["VectorWidth"] * self.rpe
    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedA = kernel["ThreadTile0"] * self.rpe
    self.localRead2CoalescedA = kernel["ThreadTile0"]/kernel["VectorWidth"] > 1
    self.localReadInstructionIdxA = \
        self.selectMemoryInstruction("LocalRead", self.localReadWidth, \
        kernel["LocalRead2A"], \
        self.localRead2CoalescedA, localRead2Perpendicular,
        [self.localReadStrideCoalescedA] )

    ########################################
    # localRead B
    self.localReadWidth = kernel["VectorWidth"] * self.rpe
    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedB = kernel["ThreadTile1"] * self.rpe
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
    tPA["nrcvpi"] = self.globalReadInstructionA.totalWidth / self.rpe
    tPB["nrcvpi"] = self.globalReadInstructionB.totalWidth / self.rpe
    tPA["nwcvpi"] = self.localWriteInstructionA.totalWidth / self.rpe
    tPB["nwcvpi"] = self.localWriteInstructionB.totalWidth / self.rpe

    ####################################
    # VGPR Allocation
    ####################################

    ####################################
    # num vgprs: valu
    self.numVgprValuC = kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.rpe
    numVgprValuA = kernel["ThreadTileA"]*self.rpe
    numVgprValuB = kernel["ThreadTileB"]*self.rpe
    numVgprValuBlkA = numVgprValuA if kernel["PrefetchLocalRead"] else 0
    numVgprValuBlkB = numVgprValuB if kernel["PrefetchLocalRead"] else 0

    ####################################
    # num vgprs: global -> local elements
    numVgprG2LA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["GlobalLoadVectorWidthA"] * self.rpe
    numVgprG2LB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["GlobalLoadVectorWidthB"] * self.rpe

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
    numGlobalReadInstructionsA = numGlobalReadsA \
        / self.globalReadInstructionA.blockWidth
    numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.rpga

    numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["GlobalLoadVectorWidthB"] \
        * self.numReadVectorComponentsB
    numGlobalReadInstructionsB = numGlobalReadsB \
        / self.globalReadInstructionB.blockWidth
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
    numWavesPerWorkGroup = kernel["NumThreads"] / 64
    numWavesPerCU = numWorkGroupsPerCU * numWavesPerWorkGroup
    self.numWavesPerSimd = numWavesPerCU / 4
    maxVgprSameOccupancy = vgprPerThreadPerOccupancy / numWorkGroupsPerCU
    self.numVgprTmp = maxVgprSameOccupancy - self.startVgprTmp
    self.totalVgprs = maxVgprSameOccupancy

    # move serial to last vgpr and shift tmp forward
    #self.startVgprSerial = self.totalVgprs-1
    #self.startVgprTmp -= 1

    #self.globalWriteAddrC = self.totalVgprs-4 # match macro
    self.globalWriteAddrC = self.startVgprSerial-4 # match macro

    ########################################
    # SGPR Allocation
    ########################################

    ####################################
    # num sgprs: initial kernel state
    numSgprKernArgAddress = self.rpga
    numSgprWorkGroup0 = 1
    numSgprWorkGroup1 = 1
    numSgprWorkGroup2 = 1 # assume batched gemm at least
    numSgprAddressC = self.rpga # til end
    numSgprAddressA = self.rpga # til read offsets
    numSgprAddressB = self.rpga # til read offsets
    numSgprOffsetC = 1
    numSgprOffsetA = 1
    numSgprOffsetB = 1
    numSgprAlpha = 1
    numSgprBeta = 1 if kernel["ProblemType"]["UseBeta"] else 0
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
    self.startSgprWorkGroup0 = sgprIdx;     sgprIdx += numSgprWorkGroup0
    self.startSgprWorkGroup1 = sgprIdx;     sgprIdx += numSgprWorkGroup1
    self.startSgprWorkGroup2 = sgprIdx;     sgprIdx += numSgprWorkGroup2
    self.startSgprAddressC = sgprIdx;       sgprIdx += numSgprAddressC
    self.startSgprStridesC = sgprIdx;       sgprIdx += self.numSgprStridesC
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

    # assign loop sgprs which overlap above assignments
    sgprIdx = self.startSgprLoopPadding
    self.startSgprGlobalReadIncsA = sgprIdx; sgprIdx += numSgprGlobalReadIncsA
    self.startSgprGlobalReadIncsB = sgprIdx; sgprIdx += numSgprGlobalReadIncsB
    self.startSgprLoopCounters = sgprIdx

    ########################################
    # Register Pools
    ########################################
    self.vgprPool = RegisterPool(self.startVgprTmp)
    #print self.totalVgprs
    #print self.vgprPool.state()
    self.vgprPool.add(self.startVgprValuC, \
        self.startVgprLocalReadAddressesA - self.startVgprValuC)
    #print self.vgprPool.state()
    self.sgprPool = RegisterPool(self.totalSgprs)

    # place any of these gpr inst values into tPA, tPB for later reference
    tPA["localWriteOffsets"] = []
    tPA["globalReadInstruction"] = self.globalReadInstructionA
    tPA["localWriteInstruction"] = self.localWriteInstructionA
    tPA["localReadInstruction"] = self.localReadInstructionA
    tPA["gpr"] = {}

    tPB["localWriteOffsets"] = []
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


  ####################################
  # Global Write Macros
  ####################################
  def globalWriteMacro(self, kernel, beta, edge):
    kStr = ""
    kStr += self.comment3("Global Write%s%s"%(" Beta" if beta else "", " Edge" if edge else ""))
    kStr += ".macro GLOBAL_WRITE%s%s vc0 vc1 d0 d1 coord0 coord1 addrC sizes tmpVgpr%s"%("_Beta" if beta else "", "_Edge" if edge else "", self.endLine)
    fullExecMaskSgpr = ((self.startSgprSizesSum+1)/2)*2 # even sgpr
    tmpS01 = fullExecMaskSgpr+2 # scratch sgprs
    tmpS23 = tmpS01+2
    tmpS45 = tmpS23+2
    tmpS67 = tmpS45+2

    kStr += ".set idx, %u + \\vc0 + \\d0*%u + \\vc1*%u + \\d1*%u*%u %s" \
        % (self.startVgprValuC, \
        kernel["VectorWidth"], \
        kernel["ThreadTile0"], \
        kernel["VectorWidth"], kernel["ThreadTile0"], \
        self.endLine )
    kStr += ".set addr, \\tmpVgpr+2%s" % self.endLine

    # coord0
    kStr += staticMultiply("v[\\tmpVgpr+0]", "\\d0", (kernel["SubGroup0"]*kernel["VectorWidth"]))
    kStr += inst("v_add_u32", "v[\\tmpVgpr+0]", "vcc", "\\vc0", "v[\\tmpVgpr+0]", \
        "tmp0 = d0*sg0*VW + vc0")
    kStr += inst("v_add_u32", "v[\\tmpVgpr+0]", "vcc", "v[\\coord0]", "v[\\tmpVgpr+0]", \
        "coord0 += d0*sg0*VW + vc0")
    #kStr += dump("v[\\tmp+0]")
    #kStr += dump("v[\\sizes+0]")

    # coord1
    kStr += staticMultiply("v[\\tmpVgpr+1]", "\\d1", (kernel["SubGroup1"]*kernel["VectorWidth"]))
    kStr += inst("v_add_u32", "v[\\tmpVgpr+1]", "vcc", "\\vc1", "v[\\tmpVgpr+1]", \
        "tmp1 = d1*sg1*VW + vc1")
    kStr += inst("v_add_u32", "v[\\tmpVgpr+1]", "vcc", "v[\\coord1]", "v[\\tmpVgpr+1]", \
        "coord1 += d1*sg1*VW + vc1")
    #kStr += dump("v[\\tmp+1]")

    if False:
      kStr += inst("s_mov_b32",  sgpr(tmpS67), hex(0), "zero" )
      kStr += inst("v_mov_b32",  "v[\\tmpVgpr+2]", sgpr(tmpS67), "zero" )

    # in-bounds exec mask
    if edge:
      kStr += inst("v_cmp_lt_u32",  sgpr(tmpS01,2), "v[\\tmpVgpr+0]", "v[\\sizes+0]", "coord0 < size0" )
      kStr += inst("v_cmp_lt_u32",  sgpr(tmpS23,2), "v[\\tmpVgpr+1]", "v[\\sizes+1]", "coord1 < size1" )
      #kStr += inst("v_mov_b32", "v[\\tmp+2]", sgpr(tmpS01), "to dump")
      #kStr += dump("v[\\tmp+2]")
      kStr += inst("s_and_b64",  sgpr(tmpS45,2), sgpr(tmpS01,2), sgpr(tmpS23,2), "in0 && in1" )
      kStr += inst("s_and_saveexec_b64",  sgpr(tmpS67,2), sgpr(tmpS45,2), "sgprs -> exec" )

    # global offset macro
    kStr += "GLOBAL_OFFSET_C addr"
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      if i == kernel["ProblemType"]["Index0"]:
        kStr += ", \\tmpVgpr+0"
      elif i == kernel["ProblemType"]["Index1"]:
        kStr += ", \\tmpVgpr+1"
      else: # just a group index
        kStr += ", sgprWorkGroup%u"%i
    kStr += ", \\tmpVgpr+4%s" % self.endLine

    # final address = C + index*4bytes
    kStr += inst("v_add_u32",  "v[addr+0]", "vcc", "v[\\addrC+0]", \
        "v[addr+0]", "addr = C + index*4bytes (lo)" )
    kStr += inst("v_addc_u32", "v[addr+1]", "vcc", "v[\\addrC+1]", \
        "v[addr+1]", "vcc", "addr = C + index*4bytes (hi)")

    # RESUME
    kStr += inst("v_mul_f32", "v[idx]", sgpr("Alpha"), "v[idx]", "*= alpha" )
    #kStr += dump("v[idx]")
    if beta:
      kStr += inst("flat_load_dword", "v[\\tmpVgpr]", "v[addr:addr+1]", \
          "load C" )
      kStr += inst("s_waitcnt", "vmcnt(0) & lgkmcnt(0)", "wait C" )
      #kStr += dump("v[\\tmp]")
      kStr += inst("v_mul_f32", "v[\\tmpVgpr]", sgpr("Beta"), "v[\\tmpVgpr]", \
          "%s = C*beta"%"v[\\tmpVgpr]" )
      #kStr += dump(vgpr("v[\\tmp]")
      kStr += inst("v_add_f32", "v[idx]", "v[\\tmpVgpr]", "v[idx]", \
          "v[idx] = sum*alpha + C*beta" )
    #kStr += dump("v[addr+0]")
    #kStr += dump("v[addr+1]")
    #kStr += dump("v[idx]")
    #kStr += "s_endpgm\n"

    kStr += inst("flat_store_dword", "v[addr:addr+1]", "v[idx]", "store C" )

    # restore full exec mask
    if edge:
      kStr += inst("s_or_saveexec_b64",  sgpr(tmpS67,2), sgpr(fullExecMaskSgpr,2), "full mask -> exec" )

    kStr += ".endm%s"%self.endLine
    return kStr

  ##############################################################################
  # Function Prefix - DONE
  ##############################################################################
  def functionPrefix(self, kernel):
    kStr = ""

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
    for i in range(2, kernel["ProblemType"]["NumIndicesC"]):
      kStr += self.macroRegister("sgprWorkGroup%u"%i, self.startSgprWorkGroup0+i)
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
        kStr += inst("v_add_u32", \
            "v[\\vgprAddr+0]", \
            "vcc", \
            "v[\\vgprTmp+0]", \
            "v[\\vgprAddr+0]",  \
            "accumulate d%u lower"%i)
        # addr += offset * stride (hi)
        kStr += inst("v_addc_u32", \
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
    # Global Write Macros
    ########################################
    kStr += self.globalWriteMacro(kernel, False, False)
    kStr += self.globalWriteMacro(kernel, False, True)
    if kernel["ProblemType"]["UseBeta"]:
      kStr += self.globalWriteMacro(kernel, True,  False)
      kStr += self.globalWriteMacro(kernel, True,  True)

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
    kStr += inst("v_sub_u32",     "v[\\vTmp1]",      "vcc", hex(0),    "v[\\vRemainder]", "" )
    kStr += inst("v_cmp_ne_i32",  "s[\\sTmp:\\sTmp+1]", hex(0),        "v[\\vTmp0]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vRemainder]", "v[\\vTmp1]",     "v[\\vRemainder]", "s[\\sTmp:\\sTmp+1]", "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vRemainder]", "v[\\vRemainder]", "v[\\vQuotient]", "" )
    kStr += inst("v_sub_u32",     "v[\\vTmp0]",      "vcc",            "v[\\vQuotient]", "v[\\vRemainder]", "" )
    kStr += inst("v_add_u32",     "v[\\vQuotient]",  "vcc",            "v[\\vQuotient]", "v[\\vRemainder]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vTmp0]", "s[\\sTmp:\\sTmp+1]", "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vDividend]", "" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
    kStr += inst("v_sub_u32",     "v[\\vTmp0]",      "vcc",            "v[\\vDividend]", "v[\\vRemainder]", "" )
    kStr += inst("v_cmp_ge_u32",  "s[\\sTmp:\\sTmp+1]", "v[\\vDividend]", "v[\\vRemainder]", "" )
    kStr += inst("v_add_u32",     "v[\\vRemainder]", "vcc",            hex(1), "v[\\vQuotient]", "" )
    kStr += inst("v_add_u32",     "v[\\vTmp1]",      "vcc", -1,        "v[\\vQuotient]", "" )
    kStr += inst("v_cmp_le_u32",  "vcc",             "v[\\vDivisor]", "v[\\vTmp0]", "" )
    kStr += inst("s_and_b64",     "vcc",             "s[\\sTmp:\\sTmp+1]", "vcc", "" ) # FIXME
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vRemainder]", "vcc", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vTmp1]",     "v[\\vQuotient]", "s[\\sTmp:\\sTmp+1]", "" )
    kStr += inst("v_cmp_ne_i32",  "vcc", hex(0),     "v[\\vDivisor]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  -1, "v[\\vQuotient]", "vcc", "final result" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
    kStr += inst("v_sub_u32",     "v[\\vRemainder]", "vcc",            "v[\\vDividend]", "v[\\vRemainder]", "final result" )
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
      kStr += ".endm%s" % self.endLine

    return kStr


  ##############################################################################
  # Function Signature - DONE
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
    kernArgReg += 1 # alpha
    if kernel["ProblemType"]["UseBeta"]:
      kernArgReg += 1 # beta
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
    totalSgprs = self.sgprPool.size()
    kStr += "  workitem_vgpr_count = %u // vgprs%s" \
        % (totalVgprs, self.endLine)
    kStr += "  wavefront_sgpr_count = %u // sgprs%s" \
        % (totalSgprs, self.endLine)
    kStr += "  compute_pgm_rsrc1_vgprs = %u // floor((%u-1)/4)%s" \
        % ( (totalVgprs-1)/4, totalVgprs, self.endLine)
    kStr += "  compute_pgm_rsrc1_sgprs = %u // floor((%u-1)/8)%s" \
        % ( 1+(totalSgprs-1)/8, totalSgprs, self.endLine)

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
    kStr += "  compute_pgm_rsrc2_lds_size = 1 // ?%s" % self.endLine
    kStr += "  workgroup_group_segment_byte_size = %u // lds bytes%s" \
        % ( kernel["LdsNumElements"] \
        * self.bpe, self.endLine )

    # other
    kStr += "  compute_pgm_rsrc2_user_sgpr = 2 // vcc%s" % self.endLine
    kStr += "  kernarg_segment_alignment = 4%s" % self.endLine
    kStr += "  group_segment_alignment = 4%s" % self.endLine
    kStr += "  private_segment_alignment = 4%s" % self.endLine
    kStr += ".end_amd_kernel_code_t%s" % self.endLine

    # if overflowed vgpr pool, comment out the whole kernel body and let it fail gracefully
    if self.vgprPool.size() > self.maxVgprs:
      print ""
      printWarning("%s invalid @ %u > %u max vgprs" % (self.kernelName, self.vgprPool.size(), self.maxVgprs) )
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

    # set m0
    if self.do["PreLoop"]: kStr += inst("s_mov_b32", "m0", hex(kernel["LdsNumElements"] \
        * self.bpe), "LDS clamp at %u bytes" \
        %(kernel["LdsNumElements"] * self.bpe) )

    if self.do["PreLoop"]: kStr += inst("v_mov_b32", vgpr("Serial"), vgpr(0), "thread serial id")

    ########################################
    # load kernel args
    # TODO revert to s_load_dwordx2
    kStr += self.comment("Load Kernel Args")
    kernArgOffset = 0
    if globalParameters["DebugKernel"]:
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressD"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug" )
      kernArgOffset += 1*4
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressD+1"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug" )
      kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressC"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr c" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressC+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr c" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressA"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr a" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressA+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr a" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressB"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr b" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressB+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr b" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("Alpha"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load alpha" )
    kernArgOffset += 1*4
    if kernel["ProblemType"]["UseBeta"]:
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("Beta"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load beta" )
      kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("OffsetC"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset c" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("OffsetA"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset a" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("OffsetB"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset b" )
    kernArgOffset += 1*4
    for i in range(0, self.numSgprStridesC):
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("StridesC+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride c %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprStridesA):
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("StridesA+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride a %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprStridesB):
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("StridesB+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride b %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprSizesFree):
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("SizesFree+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load size free %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprSizesSum):
      kStr += inst("s_load_dword", sgpr("SizesSum+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load size free %u"%i )
      kernArgOffset += 1*4
    kStr += inst("s_waitcnt", "lgkmcnt(0)", \
        "wait for %u bytes of kern args" % kernArgOffset )
    if not self.do["PreLoop"]:
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

    # test debug buffer
    #v = self.vgprPool.checkOut(3)
    #kStr += inst("v_mov_b32", vgpr(v), sgpr("AddressC"), "" )
    #kStr += inst("v_mov_b32", vgpr(v+1), sgpr("AddressC+1"), "" )
    #kStr += inst("v_mov_b32", vgpr(v+2), hex(3), "" )
    #kStr += inst("flat_store_dword", vgpr(v, 2), vgpr(v+2), "debug serial" )
    #kStr += "s_endpgm\n"

    ########################################
    # Debug Buffer
    if globalParameters["DebugKernel"]:
      kStr += self.comment("Debug Buffer")
      nt_log2 = log2(kernel["NumThreads"])

      # nwg0
      nwg0 = self.vgprPool.checkOut(1)
      tmpVgpr = self.vgprPool.checkOut(1)
      tmpSgpr = self.getTmpSgpr(1)
      kStr += "// nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile0"]-1), "MT0-1")
      kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr(tmpSgpr), "MT0-1")
      kStr += inst("v_add_u32", vgpr(nwg0), "vcc", sgpr("SizesFree+0"), \
          vgpr(tmpVgpr), "%s = size0+MT0-1"%vgpr(nwg0))
      kStr += vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpVgpr, tmpSgpr)
      tmpVgpr = self.vgprPool.checkIn(tmpVgpr)
      self.nipt = 16 # num integers per thread
      v = self.vgprPool.checkOut(3)
      kStr += inst("v_mov_b32", vgpr(v), sgpr("WorkGroup0"), "%s=wg0"%vgpr(v) )
      kStr += inst("v_mov_b32", vgpr(v+1), sgpr("WorkGroup1"), "%s=wg1"%vgpr(v+1) )
      kStr += inst("v_mul_lo_u32", vgpr(v+1), vgpr(v+1), vgpr(nwg0), \
          "%s=wg1*nwg0"%vgpr(v+1) )
      kStr += inst("v_add_u32", vgpr(v), "vcc", vgpr(v), vgpr(v+1), \
          "%s=wg1*nwg0+wg0"%vgpr(v) )
      kStr += inst("v_lshlrev_b32", vgpr(v), nt_log2, vgpr(v), \
          "%s=NT*(wg1*nwg0+wg0)"%vgpr(v) )
      kStr += inst("v_add_u32", vgpr(v), "vcc", vgpr(v), vgpr("Serial"), \
          "%s=tid+NT*(wg1*nwg0+wg0)=serial"%vgpr(v) )
      kStr += inst("v_mul_lo_u32", vgpr(v), hex(self.nipt*4), vgpr(v), \
          "%s=serial*nipt*4"%vgpr(v) )
      kStr += inst("v_mov_b32", vgpr(v+1), 0, "")
      kStr += inst("v_add_u32", vgpr("AddressD"), "vcc", sgpr("AddressD"), \
          vgpr(v), "%s=AddrD* + serial*nipt*4"%vgpr("AddressD") )
      kStr += inst("v_mov_b32", vgpr(v+2), sgpr("AddressD+1"), "%s=AddressD1"%vgpr(v+2) )
      kStr += inst("v_addc_u32", vgpr("AddressD+1"), "vcc", vgpr(v+2), \
          vgpr(v+1), "vcc", "%s=AddrD* + serial*nipt*4"%vgpr("AddressD") )
      self.vgprPool.checkIn(v)

    return kStr

  ##############################################################################
  # Global Read Addresses: Work-Group - LATER
  ##############################################################################
  def graWorkGroup(self, kernel):
    kStr = ""

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
      kStr += inst("v_add_u32", vgpr(nwg0), "vcc", sgpr(tmpSgpr), vgpr(nwg0), \
          "%s = size0+MT0-1"%vgpr(nwg0))
      kStr += vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpVgpr, tmpSgpr)
      #kStr += dump(vgpr(nwg0))

      # nwg1
      nwg1 = self.vgprPool.checkOut(1)
      kStr += "// nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
      kStr += inst("v_mov_b32", vgpr(nwg1), sgpr("SizesFree+1"), "")
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile1"]-1), "")
      kStr += inst("v_add_u32", vgpr(nwg1), "vcc", sgpr(tmpSgpr), vgpr(nwg1), \
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
      kStr += inst("v_add_u32", vgpr(wgSerial), "vcc", sgpr("WorkGroup0"), vgpr(wgSerial), \
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
      kStr += inst("v_add_u32", vgpr(wg1), "vcc", vgpr(wg1), \
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
  # Global Read Addresses: Subgroup - DONE
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
    tmpVgpr = self.vgprPool.checkOut(1)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)

    if tP["glvw"] > 1:
      if tP["grcv"] == tP["tlu"]:
        kStr += staticMultiply(vgpr(tReg), vgpr(tReg), tP["glvw"])
      else:
        kStr += staticMultiply(vgpr(uReg), vgpr(uReg), tP["glvw"])
    kStr += staticMultiply(vgpr(tmpVgpr), sgpr(tP["wg"]), kernel[tP["mt"]])
    kStr += inst("v_add_u32", vgpr(tReg2), "vcc", vgpr(tmpVgpr), \
        vgpr(tReg), "gro%s-tile = serial%s%s*VW + (wg%s*MT%s)" \
        % (tP["tensorChar"], tOpStr, divisorName, tP["tensorChar"], tP["tensorChar"]) )
    tP["gpr"]["lwoT"] = tReg
    tP["gpr"]["tReg"] = tReg2
    tP["gpr"]["uReg"] = uReg
    self.vgprPool.checkIn(tmpVgpr)
    #kStr += dump(vgpr(tReg2))
    #kStr += dump(vgpr(uReg))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A/B
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    return self.comment1(vgpr(tP["gpr"]["uReg"]))

  ##############################################################################
  # Global Read Addresses: Other Free Assignments - LATER
  ##############################################################################
  def graOtherFreeAssignments(self, kernel):
    # LATER: support more dimensions that just batched
    return self.comment1(sgpr("WorkGroup2"))

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments - DONE
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
        kStr += inst("v_add_u32", vgpr(v+s), "vcc", 1, \
            vgpr(v+s-1), "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], 0, s) )
      for l in range(1, tP["nrt"]):
        # l>0, s=0
        kStr += inst("v_add_u32", vgpr(v+l*tP["glvw"]), "vcc", stride, \
            vgpr(v+(l-1)*tP["glvw"]), \
            "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], l, 0) )
        # l>0, s>0
        for s in range(1, tP["glvw"]):
          kStr += inst("v_add_u32", vgpr(v+l*tP["glvw"]+s), "vcc", \
              1, vgpr(v+l*tP["glvw"]+(s-1)), \
              "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], l, s) )
    else:
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(tP["gpr"]["tReg"]), "gro%s%s_%u"%(tP["tensorChar"], tP["tileChar"], 0) )
      for l in range(1, tP["nrt"]):
        kStr += inst("v_add_u32", vgpr(v+l), "vcc", stride, \
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
        kStr += inst("v_add_u32", vgpr(v+s), "vcc", 1, \
            vgpr(v+s-1), "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, s) )
      for l in range(1, tP["nru"]):
        # l>0, s=0
        kStr += inst("v_add_u32", vgpr(v+l*tP["glvw"]), "vcc", stride, \
            vgpr(v+(l-1)*tP["glvw"]), \
            "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, l, 0) )
        # l>0, s>0
        for s in range(1, tP["glvw"]):
          kStr += inst("v_add_u32", vgpr(v+l*tP["glvw"]+s), "vcc", \
              1, vgpr(v+l*tP["glvw"]+(s-1)), \
              "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, s) )
    else:
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(tP["gpr"]["uReg"]), "gro%s%s_%u"%(tP["tensorChar"], self.unrollChar, 0) )
      for l in range(1, tP["nru"]):
        kStr += inst("v_add_u32", vgpr(v+l), "vcc", stride, \
            vgpr(v+l-1), "gro%s%s_%u"%(tP["tensorChar"], self.unrollChar, l) )
    #self.vgprPool.checkIn(tP["gpr"]["uReg"])
    return kStr


  ##############################################################################
  # Global Read Addresses: Branch A/B - SKIP
  ##############################################################################
  def graBranch(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A/B - SKIP
  ##############################################################################
  def graShift(self, kernel, tP):
    kStr = ""
    # edge value
    margin = tP["glvw"] if tP["rtv"] else 1
    edge = self.vgprPool.checkOut(1)
    kStr += inst("v_add_i32", vgpr(edge), "vcc", sgpr("SizesFree+%u"%tP["idx"]), \
        hex(-margin), "edge = Size%s-%u"%(tP["tileChar"], margin) )
    #kStr += dump(vgpr(edge))

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
# resume here, a uVS is staying zero
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
            graIdx += self.rpga
    #if False:
      #kStr += dump(vgpr("GlobalReadAddrA+0"))
      #kStr += dump(vgpr("GlobalReadAddrA+2"))
      #kStr += dump(vgpr("GlobalReadAddrA+4"))
      #kStr += dump(vgpr("GlobalReadAddrA+6"))
      #kStr += dump(vgpr("GlobalReadAddrA+8"))
      #kStr += dump(vgpr("GlobalReadAddrA+10"))
      #kStr += dump(vgpr("GlobalReadAddrA+12"))
      #kStr += dump(vgpr("GlobalReadAddrA+14"))

    self.vgprPool.checkIn(tileOffsets)
    self.vgprPool.checkIn(unrollOffsets)
    self.vgprPool.checkIn(tmp)
    return kStr

  ##############################################################################
  # Global Read Addresses: Apply User Offsets - DONE
  ##############################################################################
  def graApplyUserOffsets(self, kernel):
    kStr = ""
    kStr += self.comment1("moved earlier")
    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses A/B - DONE
  ##############################################################################
  def graAddresses(self, kernel, tP):
    kStr = ""
    graIdx = 0
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
            #kStr += dump(vgpr("GlobalReadAddrA+0"))
            #kStr += dump(vgpr("GlobalReadAddrA+1"))
            #kStr += dump(vgpr(tmp+0))
            #kStr += dump(vgpr(tmp+1))

            kStr += inst("v_add_u32", \
                vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                "vcc", \
                vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                vgpr(tmp+0), \
                comment+" (lower)")
            kStr += inst("v_addc_u32", \
                vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                "vcc", \
                vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                vgpr(tmp+1), \
                "vcc", \
                comment+" (upper)")
            #kStr += dump(vgpr("GlobalReadAddrA+%u+0"%graIdx))
            #kStr += dump(vgpr("GlobalReadAddrA+%u+1"%graIdx))
            graIdx += self.rpga
    #kStr += "s_endpgm\n"
    self.vgprPool.checkIn(tmp)
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments A/B - DONE
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    kStr = ""
    if loopIdx==kernel["ProblemType"]["NumIndicesSummation"]-1:
      if tP["tlu"]:
        if self.globalReadIncsUseVgpr:
          tmpSgpr = self.getTmpSgpr(1)
          kStr += inst("s_mul_i32", sgpr(tmpSgpr+0), \
              hex(kernel["DepthU"]*4), sgpr("Strides%s"%tP["tensorChar"]), \
              "incr = stride*%u*4bytes"%kernel["DepthU"] )
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
              hex(kernel["DepthU"]*4), sgpr("Strides%s"%tP["tensorChar"]), \
              "incr = stride*%u*4bytes"%kernel["DepthU"] )
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
        #tmp = self.vgprPool.checkOut(2)
        #kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("GlobalReadIncsA+0"), "" )
        #kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("GlobalReadIncsA+1"), "" )
        #kStr += dump(vgpr(tmp+0))
        #kStr += dump(vgpr(tmp+1))
        #self.vgprPool.checkIn(tmp)
      else: # transposed
        if self.globalReadIncsUseVgpr:
          kStr += inst("v_mov_b32", vgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(kernel["DepthU"]*4), \
              "incr = %u*4bytes"%kernel["DepthU"] )
          kStr += inst("v_mov_b32", vgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), "incr = %u*4bytes (upper)"%kernel["DepthU"] )
        else:
          kStr += inst("s_mov_b32", sgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(kernel["DepthU"]*4), \
              "incr = %u*4bytes"%kernel["DepthU"] )
          kStr += inst("s_mov_b32", sgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), "incr = %u*4bytes (upper)"%kernel["DepthU"] )
    else:
      printExit("NumIndicesSummation=%u not yet supported in assembly" \
          % kernel["ProblemType"]["NumIndicesSummation"] )
    #kStr += dump(vgpr("GlobalReadIncsA+0"))
    #kStr += dump(vgpr("GlobalReadIncsA+1"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B - DONE
  ##############################################################################
  def lwaTileAssignment(self, kernel, tP):
    return self.comment1("lwaTile%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["lwoT"])))

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B - DONE
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    return self.comment1("lwaUnroll%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["uReg"])))

  ##############################################################################
  # Local Write Addresses: First Offset A/B - DONE
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP):
    kStr = ""
    #"lwFOA = lwA%s + lwA%s*MT%s" \
    #    % (tP["tileChar"], self.unrollChar, tP["tileChar"])

    kStr += inst("v_mul_u32_u24", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        hex(kernel["MacroTile%s"%tP["tensorChar"]]), \
        vgpr(tP["gpr"]["uReg"]), \
        "lw%s%s*MT%s"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
    #kStr += dump(vgpr("LocalWriteAddrA"))
    kStr += inst("v_add_u32", \
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
      kStr += inst("v_add_u32", \
          vgpr("LocalWriteAddrB"), \
          "vcc", \
          hex(kernel["LdsOffsetB"]*self.bpe), \
          vgpr("LocalWriteAddrB"), \
          "lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u*%u" % (tP["tileChar"], \
          self.unrollChar, tP["tileChar"], kernel["LdsOffsetB"], self.bpe) )
    self.vgprPool.checkIn(tP["gpr"]["lwoT"])
    self.vgprPool.checkIn(tP["gpr"]["uReg"])
    #kStr += dump(vgpr("LocalWriteAddrA"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A - DONE
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
          tP["localWriteOffsets"].append(offset)
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
  # Local Write Addresses: Declare Addresses A/B - DONE
  ##############################################################################
  def lwaDeclareAddresses(self, kernel, tP):
    return self.comment1("N/A")

  ##############################################################################
  # Local Read Addresses: Tile Assignment A - DONE
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
    tmpVgpr = self.vgprPool.checkOut(1)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    #kStr += dump(vgpr(rReg))
    #kStr += dump(vgpr(qReg))
    tP["gpr"]["lro"] = rReg
    #kStr += dump(vgpr(tP["gpr"]["lro"]))
    self.tmplroB = qReg
    self.vgprPool.checkIn(tmpVgpr)
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment B - DONE
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
    tmpVgpr = self.vgprPool.checkOut(1)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    #kStr += dump(vgpr(rReg))
    #kStr += dump(vgpr(qReg))
    #kStr += "s_endpgm\n"
    self.vgprPool.checkIn(self.tmplroB) # old
    tP["gpr"]["lro"] = rReg
    #kStr += dump(vgpr(tP["gpr"]["lro"]))
    self.vgprPool.checkIn(qReg)
    self.vgprPool.checkIn(tmpVgpr)
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset A/B - DONE
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    kStr = ""
    divisor = kernel["NumThreads"]
    qReg = self.vgprPool.checkOut(1) # quotient
    rReg = self.vgprPool.checkOut(1) # remainder
    dividendReg = 0
    tmpVgpr = self.vgprPool.checkOut(1)
    tmpSgpr = self.getTmpSgpr(1)
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    sgid = qReg
    #kStr += dump(vgpr(tP["gpr"]["lro"]))
    kStr += inst("s_mov_b32", \
        sgpr(tmpSgpr), \
        hex(kernel["MacroTile%u"%tP["tensorIdx"]]), \
        "MT%u"%tP["tensorIdx"] )
    kStr += inst("v_mul_lo_u32", \
        vgpr(sgid), \
        vgpr(sgid), \
        sgpr(tmpSgpr), \
        "sgid*sgid*MT%u"%tP["tensorIdx"] )
    #kStr += dump(vgpr(sgid))
    if kernel["VectorWidth"] > 1:
      #kStr += inst("v_lshlrev_b32", \
      #    vgpr(tP["gpr"]["lro"]), \
      #    log2(kernel["VectorWidth"]), \
      #    vgpr(tP["gpr"]["lro"]), \
      #    "lroA *= VW" )
      kStr += staticMultiply(vgpr(tP["gpr"]["lro"]), vgpr(tP["gpr"]["lro"]), kernel["VectorWidth"])
      #kStr += dump(vgpr(tP["gpr"]["lro"]))
    kStr += inst("v_add_u32", \
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
    """
    if tP["isB"]:
      kStr += inst("v_lshlrev_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(log2(self.bpe)), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "*= bytes/element" )
    """
    #kStr += dump(vgpr("LocalReadAddrA"))
    #kStr += "s_endpgm\n"

    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(qReg)
    self.vgprPool.checkIn(rReg)
    self.vgprPool.checkIn(tP["gpr"]["lro"])
    #kStr += "  unsigned int localReadOffsetA = lr%s*VECTOR_WIDTH + sgId*MT%s%s" \
    #    % ( tP["tileChar"], tP["tileChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B - DONE
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    if tP["isA"]:
      return self.comment1("N/A")
    else:
      return inst("v_add_u32", \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          "vcc", \
          hex(kernel["LdsOffset%s"%tP["tensorChar"]]*self.bpe), \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          " += LdsOffset%s (lower)"%tP["tensorChar"])

  ##############################################################################
  # Declare Loop Num Iterations - DONE
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
  # Calculate Loop Num Iter - DONE
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx):
    kStr = ""

    tmp = self.vgprPool.checkOut(1)
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    # Tail Loop
    if tailLoop:
      kStr += "%s//numIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)%s" \
          % (self.indent, self.unrollChar, self.unrollChar, self.endLine)
      if kernel["GlobalSplitU"] > 1:
        # SKIP
        printExit("Asm GSU>1 not yet supported")
        kStr += "%sif (gsuSumIdx != numIterPerWgRemainder) {%s" \
            % (self.indent, self.endLine)
        kStr += "%s  numIter%s = 0%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%s}%s" % (self.indent, self.endLine)
      else:
        tmpSgpr = self.getTmpSgpr(1)
        kStr += scalarStaticDivideAndRemainder(tmpSgpr, "LoopCounters+%u"%loopIdx, "SizesSum+%u"%loopIdx, kernel["DepthU"], tmpSgpr+1, True)
        kStr += inst("s_sub_u32", \
            sgpr("LoopCounters+%u"%loopIdx), \
            hex(0), \
            sgpr("LoopCounters+%u"%loopIdx), \
            "counter%s = -size%s"%(loopChar, loopChar) )

      # tail loop count == 0?
      tailLoopLabelBegin = self.getLabel("TailLoopBegin%s"%(loopChar) )
      tailLoopLabelEnd = self.getLabel("TailLoopEnd%s"%(loopChar) )
      kStr += inst("s_cmp_eq_u32", sgpr("LoopCounters+%u"%loopIdx), \
          hex(0), "numIter%s == 0"%loopChar )
      kStr += inst("s_cbranch_scc1 label_%04u"\
          % tailLoopLabelEnd, \
          "skip to end of tail loop b/c numIter==0")
    else: # Unrolled Loop
      if not self.do["PreLoop"]: kStr += ".endif\n"
      if loopIdx == self.unrollIdx:
        #kStr += inst("v_mov_b32", vgpr(tmp), \
        #    sgpr("SizesSum+0"), "" )
        #kStr += dump(vgpr(tmp))

        # TODO doesn't support DU non-pow2
        kStr += inst("s_lshr_b32", \
            sgpr("LoopCounters+%u"%loopIdx), \
            sgpr("SizesSum+%u"%loopIdx), \
            log2(kernel["DepthU"]), \
            "numIter%s = size%s / DU"%(loopChar, loopChar) )


        #kStr += inst("v_mov_b32", vgpr(tmp), \
        #    sgpr("LoopCounters+0"), "" )
        #kStr += dump(vgpr(tmp))

        kStr += inst("s_sub_u32", \
            sgpr("LoopCounters+%u"%loopIdx), \
            hex(0), \
            sgpr("LoopCounters+%u"%loopIdx), \
            "counter%s = -size%s"%(loopChar, loopChar) )

        # unrolled loop count == 0?
        #loopLabelBegin = self.getLabel("LoopBegin%s"%(loopChar) )
        #loopLabelEnd = self.getLabel("LoopEnd%s"%(loopChar) )
        #kStr += inst("s_cmp_eq_u32", sgpr("LoopCounters+%u"%loopIdx), \
        #    hex(0), "numIter%s == 0"%loopChar )
        #kStr += inst("s_cbranch_scc1 label_%04u"\
        #  % loopLabelEnd, \
        #  "skip to end of unrolled loop b/c numIter==0")
      else:
        # SKIP
        printExit("Asm GSU>1 not yet supported")
        kStr += "%snumIter%s = size%s" \
            % (self.indent, loopChar, loopChar)

      if loopIdx == self.unrollIdx and kernel["GlobalSplitU"] > 1:
        # SKIP
        printExit("Asm GSU>1 not yet supported")
        kStr += "%sunsigned int numIterMyWg = numIter%s / GLOBAL_SPLITU%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%sunsigned int numIterPerWgRemainder = numIter%s %% GLOBAL_SPLITU%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%sif (gsuSumIdx < numIterPerWgRemainder) {%s" \
            % (self.indent, self.endLine)
        kStr += "%s  numIterMyWg++%s" % (self.indent, self.endLine)
        kStr += "%s}%s" % (self.indent, self.endLine)
        kStr += "%snumIter%s = numIterMyWg%s" \
            % (self.indent, self.unrollChar, self.endLine)

    #kStr += inst("v_mov_b32", vgpr(tmp), \
    #    sgpr("LoopCounters+0"), "" )
    #kStr += dump(vgpr(tmp))

    #kStr += "s_endpgm\n"
    self.vgprPool.checkIn(tmp)
    return kStr

  ##############################################################################
  # Open Loop - DONE
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
    
    # begin loop
    kStr += "label_%04u:%s" % (loopLabelBegin, self.endLine)
    return kStr


  ##############################################################################
  # Close Loop - DONE
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
    #kStr += "s_endpgm\n"
    #kStr += self.dumpLds(8, 8)
    return kStr

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self):
    #self.vgprPool.remove(self.startVgprTmp, self.numVgprTmp)
    self.vgprPool.add(self.startVgprValuA, \
        self.startVgprSerial - self.startVgprValuA)
    return ""

  ##############################################################################
  # MAC Iteration - DONE
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
  # At Least 1 Unroll - SKIP
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
  # Global Read: Increment A/B - DONE
  ##############################################################################
  def globalReadIncrement(self, kernel, loopIdx, tP):
    if not self.do["GlobalInc"]: return ""
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    graIdx = 0
    tmp = self.startVgprSerial - 1
    #for perp in range(0, tP["nrp"]):
    #  for para in range(0, tP["nrc"]):
    #    for s in range(0, tP["nrcv"]):
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):
            if self.globalReadIncsUseVgpr:
              kStr += inst("v_add_u32 ", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                  vgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], loopIdx)), \
                  "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
              kStr += inst("v_addc_u32", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  vgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], loopIdx)), \
                  "vcc", \
                  "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
            else:
              kStr += inst("v_add_u32 ", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                  sgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], loopIdx)), \
                  "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
              kStr += inst("v_mov_b32 ", \
                  vgpr(tmp), \
                  sgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], loopIdx)), \
                  "vgpr GlobalReadIncs%s"%tP["tensorChar"] )
              kStr += inst("v_addc_u32", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  vgpr(tmp), \
                  "vcc", \
                  "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
            graIdx += self.rpga
    #kStr += dump(vgpr("GlobalReadAddrA+0"))
    #kStr += dump(vgpr("GlobalReadAddrA+1"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Global Read: Do It A/B - DONE
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
            graIdx = i * self.rpga
            g2lIdx = i * loadWidth
            if guardK:
              # for each component in vector
              for r in range(0, loadWidth):
                kStr += self.comment1("load component %u"%r)
                #kStr += dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)))
                #kStr += dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)))
                #kStr += "s_endpgm\n"
                # zero out data regardless of load or not
                kStr += inst("v_mov_b32", vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r)), hex(0), "zero")

                # mask if current address if in bounds
                kStr += inst("v_cmpx_lt_u64", "vcc", \
                    vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), \
                    vgpr(maxAddr,2), \
                    "addr < maxAddr")

                # load single element from address
                kStr += inst("flat_load_dword", \
                    vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r)),
                    vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), "load single")

                # restore full exec mask
                kStr += inst("s_or_saveexec_b64", "vcc", sgpr(fullExec,2), \
                    "all threads active")
                #kStr += "s_waitcnt vmcnt(0)\n"
                #kStr += dump(vgpr("G2L%s+%u+%u"%(tP["tensorChar"], g2lIdx, r)))

                # increment address by 1 element
                kStr += inst("v_add_u32", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    vgpr(bpeVgpr), "gra += 1 (lower)")
                kStr += inst("v_addc_u32", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    "vcc", \
                    vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    vgpr(zeroVgpr), \
                    "vcc", \
                    "gra += 1 (upper)")
            else:
              kStr += tP["globalReadInstruction"].toString( \
                  (vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), loadWidth), \
                  vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2)), \
                  "G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp ) )
    if guardK:
      self.vgprPool.checkIn(bpeVgpr)
      self.vgprPool.checkIn(zeroVgpr)
      self.vgprPool.checkIn(tmpVgpr)
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets A/B - DONE
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
  # Local Write: Reset Offsets A/B - SKIP
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
  # Local Write: Init Pointers A/B - DONE
  ##############################################################################
  def localWriteInitPointers(self, kernel, tP):
    return self.comment1("N/A")

  ##############################################################################
  # Local Write: Do It A/B - DONE
  ##############################################################################
  def localWriteDo(self, kernel, tP):
    if not self.do["LocalWrite"]: return ""
    kStr = ""
    instruction = tP["localWriteInstruction"]
    numBlocks = instruction.numBlocks
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    offsetMultiplier = instruction.offsetMultiplier
    totalWrites = len(tP["localWriteOffsets"])/numOffsets
    g2lIdx = 0
    graIdx = 0
    #kStr += dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))

    # if transposing, positions of sPerp and sPara are transposed
    for perp in range(0, tP["nrp"]):
      for para in range(0, tP["nrc"]):
        for s in range(0, max(tP["nwcv"],tP["nwpv"])/tP["nwcvpi"]):
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

          if tP["tlu"]:
            lspaOffset *= kernel[tP["mt"]]
          else:
            lscaOffset *= kernel[tP["mt"]]
          if tP["tlu"] == tP["grcv"]:
            lspaOffset *= tP["glvw"]
            lscaOffset *= tP["glvw"]
          offset = lspaOffset + lscaOffset
          offset *= self.bpe
          offset /= offsetMultiplier

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
          kStr += tP["localWriteInstruction"].toString(paramTuple, comment)
    #if tP["isB"]:
    #  kStr += self.dumpLds(kernel, 0, 16)
    return kStr

  ##############################################################################
  # Local Read: Swap Offsets A/B - DONE
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
  # Local Read: Reset Offsets A/B - DONE
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
  # Local Read: Init Pointers A/B - DONE
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
  # Local Read: Increment A/B - DONE
  ##############################################################################
  def localReadInc(self, kernel, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    if self.inTailLoop:
      inc = kernel["LocalSplitU"]*kernel["MacroTile%u"%tP["tensorIdx"]]*self.bpe
      tmpSgpr = self.getTmpSgpr(1)
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(inc), "inc")
      kStr += inst("v_add_u32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "vcc", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          sgpr(tmpSgpr), \
          "lr%s += %u"%(tP["tensorChar"], inc) )
    else:
      if tP["localReadInstruction"].numOffsets == 1:
        tP["localReadOffset"] += kernel["LocalSplitU"]*kernel["MacroTile%u"%tP["tensorIdx"]]
        kStr += self.comment1("N/A")
      else:
        inc = kernel["LocalSplitU"]*kernel["MacroTile%u"%tP["tensorIdx"]]
        kStr += inst("v_add_u32", \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            "vcc", \
            vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            hex(inc), \
            "lr%s += %u"%(tP["tensorChar"], inc) )
    return kStr

  ##############################################################################
  # Local Read: Do It A/B - DONE
  ##############################################################################
  def localReadDo(self, kernel, black, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    instruction = tP["localReadInstruction"]
    numBlocks = instruction.numBlocks
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    offsetMultiplier = 1 # instruction.offsetMultiplier
    #totalReads = (kernel["ThreadTile%u"%tP["tensorIdx"]]/blockWidth) / numOffsets
    valuIdx = 0
    numVectorsPerTile = (kernel["ThreadTile%u"%tP["tensorIdx"]]/kernel["VectorWidth"])
    numReadsPerVector = kernel["VectorWidth"] / blockWidth
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
    return kStr

  ##############################################################################
  # Shift Vector Components d0,1 - DONE
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
    tmpVgpr = self.vgprPool.checkOut(1)
    wgMT = self.vgprPool.checkOut(1)
    kStr += inst("v_mov_b32", vgpr(wgMT), sgpr(tP["wg"]), "")
    kStr += inst("v_mul_i32_i24", vgpr(wgMT), hex(-kernel[tP["mt"]]), vgpr(wgMT), \
        "wg*MT")
    kStr += inst("v_add_u32", vgpr(wgMT), "vcc", sgpr("SizesFree+%u"%tP["idx"]), \
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
      dummy = self.vgprPool.checkOut(1)
      divisor = kernel["SubGroup0"]
      kStr += vectorStaticDivideAndRemainder(sd0, dummy, "Serial", divisor, \
          tmpVgpr, tmpSgpr) # thread = serial / SG0
      divisor = kernel["SubGroup1"]
      thread = self.vgprPool.checkOut(1)
      kStr += vectorStaticDivideAndRemainder(dummy, thread, sd0, divisor, \
          tmpVgpr, tmpSgpr) # thread = (serial / SG0) % SG1

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
      kStr += inst("v_add_u32", vgpr(vReg), "vcc", vgpr(mvReg), vgpr(vReg), "vId = 2 components")
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
        if vectorIdx==2 and tP["isA"]:
          kStr += dump(vgpr("Serial"))
        # mask if last thread in thread-tile column
        kStr += inst("v_cmpx_eq_u32", sgpr(tmpSgpr,2), vgpr(thread), \
          vgpr(eReg), "serial % SG == (wgMT/VECTOR_WIDTH)%SG" )
        tto = kernel["ThreadTile%u"%((tP["idx"]+1)%2)] # thread tile orthogonal
        for tt in range(0, tto):
          for s in range(0, r):
            if tP["isA"]:
              dst = (s) \
                  + vectorIdx * vw + tt * kernel["ThreadTile0"]
              src = (s+vw-r) \
                  + vectorIdx * vw + tt * kernel["ThreadTile0"]
              comment = "rC[%u+%u*VW+%u*TT%s] = rC[%u+%u*VW+%u*TT%s]" \
                  % (s, vectorIdx, tt, self.tileChar0, \
                  s+vw-r, vectorIdx, tt, self.tileChar0 )
            else:
              dst = (tt) \
                  + vectorIdx*vw*kernel["ThreadTile0"] + s * kernel["ThreadTile0"]
              src = (tt) \
                  + vectorIdx * vw*kernel["ThreadTile0"] + (s+vw-r) * kernel["ThreadTile0"]
              comment = "rC[%u+%u*TT%s*VW+%u*TT%s] = rC[%u+%u*TT%s*VW+%u*TT%s]" \
                % (tt, vectorIdx, self.tileChar0, s, self.tileChar0, \
                tt, vectorIdx, self.tileChar0, \
                s+vw-r, self.tileChar0)

            kStr += inst("v_mov_b32", vgpr(self.startVgprValuC+dst), \
                vgpr(self.startVgprValuC+src), comment)
            #kStr += inst("s_nop", "4", "")

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
  # LocalSplitU: Local Write - SKIP
  ##############################################################################
  def localSplitULocalWrite(self, kernel):
    kStr = ""
    kStr += "  %sVECTOR_TYPE *localLocalSplitU = (%sVECTOR_TYPE *)(localMemory)%s" \
        % (self.sharedPtrStr, self.sharedPtrStr, self.endLine)
    for j in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          kStr += "%slocalLocalSplitU[lr%s + %u*SG%s + (MT%s/VECTOR_WIDTH)*(lr%s*VECTOR_WIDTH + %u + SG%s*VECTOR_WIDTH*%u) + (MT%s*MT%s/VECTOR_WIDTH)*sgId] = rC[%u+%u*(TT%s/VECTOR_WIDTH)+%u*TT%s]%s" \
              % (self.indent, self.tileChar0, i, self.tileChar0, \
              self.tileChar0, self.tileChar1, \
              s, self.tileChar1, j, self.tileChar0, self.tileChar1, i, s, \
              self.tileChar0, j, self.tileChar0, self.endLine)
    kStr += self.indent + self.syncStr + self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Local Read - SKIP
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    kStr = ""
    for i in range(0, kernel["NumVectorsPerThread"]):
      kStr += "  rC[%3u] = localLocalSplitU[serial+%u*NUM_THREADS];%s" \
          % (i, i, self.endLine)
    kStr += self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Reduction - SKIP
  ##############################################################################
  def localSplitUReduction(self, kernel):
    kStr = ""
    for s in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumVectorsPerThread"]):
        kStr += "  rC[%3u] += localLocalSplitU[serial+%u*NUM_THREADS + %u*(MT%s*MT%s/VECTOR_WIDTH)];%s" \
            % (i, i, s, self.tileChar0, self.tileChar1, self.endLine)
      kStr += self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write Indices - SKIP
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    kStr = ""
    kStr += "  unsigned int localC%s = (serial %% (MT%s/VECTOR_WIDTH))*VECTOR_WIDTH;%s" \
        % (self.tileChar0, self.tileChar0, self.endLine)
    kStr += "  unsigned int localC%s = serial / (MT%s/VECTOR_WIDTH);%s" \
        % (self.tileChar1, self.tileChar0, self.endLine)
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int globalC%s = wg%s" \
          % (self.indexChars[i], self.indexChars[i])
      if i == kernel["ProblemType"]["Index0"]:
        kStr += "*MT%s + localC%s" \
            % (self.tileChar0, self.tileChar0)
      if i == kernel["ProblemType"]["Index1"]:
        kStr += "*MT%s + localC%s" \
            % (self.tileChar1, self.tileChar1)
      kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write - SKIP
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    kStr = ""
    if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
      kStr += "  float type_mac_tmp;" + self.endLine
    if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
      kStr += "  double type_mac_tmp;" + self.endLine

    for b in range(0, kernel["NumVectorsPerThread"]):
      for s in range(0, kernel["VectorWidth"]):
        if kernel["EdgeType"] != "None":
          kStr += "  if (globalC%s%s < size%s) {" \
              % (self.tileChar0, \
              ((" + %u" %s) if kernel["VectorWidth"]>1 else ""), \
              self.tileChar0)
          kStr += "  if (globalC%s + %u*CPSV < size%s) {" \
              % (self.tileChar1, b, self.tileChar1)

        kStr += "  TYPE_MAC_WRITE( C[ GLOBAL_C( (%s)" % self.uint64Str
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          kStr += " globalC%s" % self.indexChars[i]
          if i == kernel["ProblemType"]["Index0"] and kernel["VectorWidth"]>1:
            kStr += " + %u" %s
          if i == kernel["ProblemType"]["Index1"]:
            kStr += " + %u*CPSV" %b
          if i < kernel["ProblemType"]["NumIndicesC"]-1:
            kStr += ", (%s)" % self.uint64Str
        kStr += ") ]"
        kStr += ", alpha"
        kStr += ", rC[%d]%s" % (b, \
            ((".%s"%self.vectorComponents[s]) if kernel["VectorWidth"]>1 \
            else "") )

        if kernel["ProblemType"]["UseBeta"]:
          kStr += ", beta"
        kStr += ")"

        if kernel["EdgeType"] != "None":
          kStr += "} }"
        kStr += self.endLine
    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write Indices - DONE
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    #print "GlobalWriteIndices"
    if not self.do["PostLoop"]: return ""
    kStr = ""
    self.scratchSgprs = self.startSgprSizesSum

    tmpS0 = self.scratchSgprs
    tmpS1 = tmpS0+1
    tmpS2 = tmpS1+1
    tmpS3 = tmpS2+1
    tmpV0 = self.vgprPool.checkOut(1)

    # thread id 0,1
    tid0 = self.vgprPool.checkOut(1)
    tid1 = self.vgprPool.checkOut(1)
    divisor = kernel["SubGroup0"]
    kStr += vectorStaticDivideAndRemainder(tid1, tid0, "Serial", divisor, \
        tmpV0, tmpS0)
    kStr += staticMultiply(vgpr(tid0), vgpr(tid0), kernel["VectorWidth"])
    kStr += staticMultiply(vgpr(tid1), vgpr(tid1), kernel["VectorWidth"])

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
    kStr += inst("v_add_u32", \
        vgpr(tid0), \
        "vcc", \
        sgpr(tmpS0), \
        vgpr(tid0), \
        "coord0 = tid0*VW + wg0*MT0")
    kStr += inst("v_add_u32", \
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

    # create full exec mask
    if kernel["EdgeType"] != "None":
      fullExecMaskSgpr = ((self.startSgprSizesSum+1)/2)*2 # even sgpr
      kStr += inst("s_mov_b64", \
          sgpr(fullExecMaskSgpr,2), \
          "0xFFFFFFFFFFFFFFFF", \
          "full exec mask")

    # store free sizes in vgprs for comparison
    self.sizesFreeVgprs = self.vgprPool.checkOut(2)
    kStr += inst("v_mov_b32", \
        vgpr(self.sizesFreeVgprs+0), \
        sgpr("SizesFree+0"), \
        "free sizes sgpr -> vgpr")
    kStr += inst("v_mov_b32", \
        vgpr(self.sizesFreeVgprs+1), \
        sgpr("SizesFree+1"), \
        "free sizes sgpr -> vgpr")

    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write - DONE
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    if not self.do["PostLoop"]: return ""
    kStr = ""
    kStr += self.comment1("GLOBAL_WRITE vc0 vc1 tt0 tt1 coord0 coord1%s" \
        % (self.endLine) )
    globalWriteTmp = self.vgprPool.checkOut(7)

    if kernel["ProblemType"]["UseBeta"]:
      betaLabel = self.getLabel("GW_Beta")
      endLabel = self.getLabel("WG_End")
      kStr += inst("s_cmpk_eq_u32", sgpr("Beta"), hex(0), "Beta == 0")
      kStr += inst("s_cbranch_scc0 label_%04u" % betaLabel, \
          "Beta not not zero; so jump to B nonzero")
      betas = [False, True]
    else:
      betas = [False]

    for beta in betas:
      for tt1 in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
        for tt0 in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
          for vc1 in range(0, kernel["VectorWidth"]):
            for vc0 in range(0, kernel["VectorWidth"]):
              kStr += "GLOBAL_WRITE%s_Edge %u %u %u %u %u %u %u %u %u%s" \
                  % ("_Beta" if beta else "", \
                  vc0, vc1, tt0, tt1, self.coord0, self.coord1, self.addrC, \
                  self.sizesFreeVgprs, globalWriteTmp, self.endLine)
      if not beta and kernel["ProblemType"]["UseBeta"]:
        kStr += inst("s_branch", "label_%04u"%endLabel, "jump to end")
        kStr += "label_%04u:%s"%(betaLabel, self.endLine)
    if kernel["ProblemType"]["UseBeta"]:
      kStr += "label_%04u:%s"%(endLabel, self.endLine)
    self.vgprPool.checkIn(globalWriteTmp)
    return kStr

  ##############################################################################
  # Function End - DONE
  ##############################################################################
  def functionEnd(self, kernel):
    return inst("s_endpgm", "End Kernel")

  ##############################################################################
  # Function Suffix - DONE
  ##############################################################################
  def functionSuffix(self, kernel):
    kStr = ""
    if self.vgprPool.size() > self.maxVgprs:
      kStr += ".endif // too many vgprs\n"
    return kStr

  ##############################################################################
  # Kernel Body Prefix - DONE
  ##############################################################################
  def kernelBodyPrefix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # Kernel Body Suffix - DONE
  ##############################################################################
  def kernelBodySuffix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # Open String - DONE
  ##############################################################################
  def openString(self, kernel):
    return ""

  ##############################################################################
  # Close String - DONE
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
        numA = len(tPA["localWriteOffsets"]) \
            / tPA["localWriteInstruction"].numOffsets
        numB = len(tPB["localWriteOffsets"]) \
            / tPB["localWriteInstruction"].numOffsets
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
      kStr += inst("s_barrier", "" )
      kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
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
  # Function Signature - SKIP
  ##############################################################################
  def functionSignatureBetaOnly(self, kernel):
    kStr = ""
    return kStr

  ##############################################################################
  # Kernel Body Beta-Only - SKIP
  ##############################################################################
  def kernelBodyBetaOnly(self, kernel):
    kStr = ""
    return kStr



################################################################################
# Helper Functions
################################################################################

########################################
# Store to Debug Buffer - DONE
########################################
def dump(vgprStore):
  kStr = ""
  if globalParameters["DebugKernel"]:
    kStr += inst("flat_store_dword", vgpr("AddressD", 2), \
        vgprStore, "debug dump store" )
    kStr += inst("v_add_u32", vgpr("AddressD"), "vcc", vgpr("AddressD"), \
        hex(4), "debug dump inc" )
  return kStr


########################################
# Format Instruction - DONE
########################################
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
# Format GPRs - DONE
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
# Log 2 - DONE
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

  elif (((divisor/3) & ((divisor/3) - 1)) == 0): # 3 * pow of 2
    shift = 33 + log2(divisor/3)
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), "0xaaaaaaab", "")
    kStr += inst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(shift), "")
    kStr += inst("v_lshrrev_b64", vgpr(tmpVgpr,2), sgpr(tmpSgpr), vgpr(tmpVgpr,2), "")
    kStr += inst("v_mov_b32", vgpr(qReg), vgpr(tmpVgpr), "quotient")
    #kStr += dump(vgpr(qReg))
    if doRemainder:
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), "divisor")
      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr(tmpSgpr), "")
      #kStr += dump(vgpr(tmpVgpr))
      kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(qReg), sgpr(tmpSgpr), "product = quotient * divisor")
      #kStr += dump(vgpr(tmpVgpr))
      kStr += inst("v_sub_u32", vgpr(rReg), "vcc", vgpr(dReg), vgpr(tmpVgpr), "remainder = dividend - product")
      #kStr += dump(vgpr(rReg))

  else:
    printExit("KernelWriterAssembly::divmod doesn't support %u" % divisor)
  return kStr

def vectorStaticDivide(qReg, dReg, divisor, tmpVgpr, tmpSgpr):
  rReg = -1 # unused
  kStr = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, False)
  return kStr

# only used for loop unroll
def scalarStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr, \
    doRemainder=True):
  kStr = ""
  if ((divisor & (divisor - 1)) == 0): # pow of 2
    divisor_log2 = log2(divisor)
    kStr += inst("s_lshr_b32", sgpr(qReg), sgpr(dReg), divisor_log2, \
        "%s = %s / %u"%(sgpr(qReg), sgpr(dReg), divisor) )
    #kStr += dump(sgpr(qReg))
    if doRemainder:
      kStr += inst("s_and_b32", sgpr(rReg), (divisor-1), sgpr(dReg), \
          "%s = %s %% %u"%(sgpr(rReg), sgpr(dReg), divisor) )
      #kStr += dump(sgpr(rReg))
  elif (((divisor/3) & ((divisor/3) - 1)) == 0): # 3 * pow of 2 TODO FIXME
    printExit("KernelWriterAssembly::scalarStaticDivide doesn't support %u" % divisor)
    shift = 32 + log2(divisor/3)
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), "0xaaaaaaab", "tmp = magic")
    kStr += inst("s_mul_i32", sgpr(tmpSgpr+1), sgpr(dReg), sgpr(tmpSgpr), "tmp1 = dividend * magic")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(shift), "tmp = shift")
    kStr += inst("s_lshr_b32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), tmpSgpr, "tmp1 = (dividend * magic) << shift")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), "tmp = divisor")
    kStr += inst("v_mul_i32", sgpr(qReg), sgpr(tmpSgpr+1), sgpr(tmpSgpr), "qReg = ( (dividend*magic)<<shift )*divisor")
    if doRemainder:
      kStr += inst("v_sub_u32", sgpr(rReg), "vcc", sgpr(dReg), sgpr(tmpSgpr), "rReg = dividend - divisor")
  else:
    printExit("KernelWriterAssembly::divmod doesn't support %u" % divisor)
  return kStr

#def staticRemainder(rReg, dReg, divisor, tmpVgpr, tmpSgpr):
#  qReg = self.vgprPool.checkOut(1)
#  kStr = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr)
#  self.vgprPool.checkIn(qReg)
#  return kStr

########################################
# Multiply
# product register, operand register, multiplier
# TODO clarify support of sgpr and constant operand/multipliers
########################################
def staticMultiply(product, operand, multiplier):
  if ((multiplier & (multiplier - 1)) == 0): # pow of 2
    multiplier_log2 = log2(multiplier)
    return inst("v_lshlrev_b32", product, multiplier_log2, operand, \
        "%s = %s * %u"%(product, operand, multiplier) )
  else:
    if True: # operand.startswith("s["):
      kStr = ""
      kStr += inst("v_mov_b32", product, multiplier, \
        "%s = %u"%(product, multiplier) )
      kStr += inst("v_mul_lo_u32", product, product, operand, \
        "%s *= %s"%(product, operand) )
      return kStr
    else:
      return inst("v_mul_lo_u32", product, operand, hex(multiplier), \
          "%s = %s * %u"%(product, operand, multiplier) )
    """
    VOP3 (cannot use literal constant)
    v_mul_lo_u32
    v_mul_hi_u32
    v_mul_hi_i32
    VOP2
    v_mul_i32_i24
    v_mul_u32_u24
    v_mul_hi_i32_i24
    v_mul_hi_u32_u24

    """

  """
# mod 3 v0 -> v0
s_mov_b32  s1 0xaaaaaaab
v_mul_hi_u32  v2 v0 s1
v_mul_lo_u32  v1 v0 s1
v_lshrrev_b64  v[1:2] 33 v[1:2]
v_mul_lo_u32  v1 v1 3
v_sub_u32  v0 vcc v0 v1
# mod 6
s_mov_b32  s1 0xaaaaaaab
v_mul_hi_u32  v2 v0 s1
v_mul_lo_u32  v1 v0 s1
v_lshrrev_b64  v[1:2] 34 v[1:2]
v_mul_lo_u32  v1 v1 6
v_sub_u32  v0 vcc v0 v1
# mod 12
s_mov_b32  s1 0xaaaaaaab
v_mul_hi_u32  v2 v0 s1
v_mul_lo_u32  v1 v0 s1
v_lshrrev_b64  v[1:2] 35 v[1:2]
v_mul_lo_u32  v1 v1 12
v_sub_u32  v0 vcc v0 v1
# mod 2
V_AND_B32  v0 1 v0
# mod 4
V_AND_B32  v0 3 v0
# mod 8
V_AND_B32  v0 7 v0
# mod 16
V_AND_B32  v0 15 v0

    else:
      kStr += "/"
# div 2
V_LSHLREV_B32  v0 1 v0
# div 2
V_LSHLREV_B32  v0 2 v0
# div 8
V_LSHLREV_B32  v0 3 v0
# div 16
V_LSHLREV_B32  v0 4 v0
# div 3
s_mov_b32  s0 0xaaaaaaab
v_mul_hi_u32  v3 v0 s0
v_mul_lo_u32  v2 v0 s0
v_lshrrev_b64  v[2:3] 33 v[2:3]
# div 6
s_mov_b32  s0 0xaaaaaaab
v_mul_hi_u32  v3 v0 s0
v_mul_lo_u32  v2 v0 s0
v_lshrrev_b64  v[2:3] 34 v[2:3]
# div 12
s_mov_b32  s0 0xaaaaaaab
v_mul_hi_u32  v3 v0 s0
v_mul_lo_u32  v2 v0 s0
v_lshrrev_b64  v[2:3] 35 v[2:3]
  """
