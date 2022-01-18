################################################################################
# Copyright 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import print2, printExit, printWarning
from .Utils import roundUpToNearestMultiple

import traceback
from enum import Enum
from math import log

########################################
# Format Instruction
########################################

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
# Format Trailing Comment Only
########################################

def instCommentOnly(comment=""):
    # Aligned with inst (50 chars)
    return "%-50s // %s\n" % ("", comment)

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

def accvgpr(*args):
    return gpr("acc", args)

########################################
# Log 2
########################################

def log2(x):
    return int(log(x, 2) + 0.5)

########################################
# Divide & Remainder
# quotient register, remainder register, dividend register, divisor, tmpVgprx2, tmpSgpr
########################################

def vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, doRemainder=True, comment=""):

    dComment = "%s = %s / %s"    % (vgpr(qReg), vgpr(dReg), divisor) if (comment=="") else comment
    rComment = "%s = %s %% %s" % (vgpr(rReg), vgpr(dReg), divisor) if (comment=="") else comment

    kStr = ""
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        divisor_log2 = log2(divisor)
        kStr += inst("v_lshrrev_b32", vgpr(qReg), divisor_log2, vgpr(dReg), dComment)
        if doRemainder:
            kStr += inst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), rComment)
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
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(magic), dComment)
        kStr += inst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), dComment)
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), dComment)
        kStr += inst("v_lshrrev_b64", vgpr(tmpVgpr,2), hex(shift), vgpr(tmpVgpr,2), dComment)
        kStr += inst("v_mov_b32", vgpr(qReg), vgpr(tmpVgpr), dComment)
        if doRemainder:
            kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), rComment)
            kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(qReg), sgpr(tmpSgpr), rComment)
            kStr += inst("_v_sub_u32", vgpr(rReg), vgpr(dReg), vgpr(tmpVgpr), rComment)
    return kStr

def vectorStaticDivide(qReg, dReg, divisor, tmpVgpr, tmpSgpr, comment=""):
    rReg = -1 # unused
    kStr = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, False, comment)
    return kStr

def vectorStaticRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, comment=""):
    if comment == "":
        comment = "%s = %s %% %s" % (vgpr(rReg), vgpr(dReg), divisor)

    kStr = ""
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        kStr += inst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), comment)
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
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(magic), comment)
        kStr += inst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), comment)
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), comment)
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(shift), comment)
        kStr += inst("v_lshrrev_b64", vgpr(tmpVgpr,2), sgpr(tmpSgpr), vgpr(tmpVgpr,2), comment)
        kStr += inst("v_mov_b32", vgpr(qReg), vgpr(tmpVgpr), comment)
        kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), comment)
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(qReg), sgpr(tmpSgpr), comment)
        kStr += inst("_v_sub_u32", vgpr(rReg), vgpr(dReg), vgpr(tmpVgpr), comment)
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

def staticMultiply(product, operand, multiplier, tmpSgpr=None, comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    if multiplier == 0:
            return inst("v_mov_b32", product, hex(multiplier), comment)
    elif ((multiplier & (multiplier - 1)) == 0): # pow of 2
        multiplier_log2 = log2(multiplier)
        if multiplier_log2==0 and product == operand:
            return instCommentOnly(comment + " (multiplier is 1, do nothing)")
        else:
            return inst("v_lshlrev_b32", product, hex(multiplier_log2), operand, comment)
    else:
        kStr = ""
        if product == operand:
            kStr += inst("s_mov_b32", tmpSgpr, hex(multiplier), comment)
            kStr += inst("v_mul_lo_u32", product, tmpSgpr, operand, comment)
        else:
            kStr += inst("v_mov_b32", product, hex(multiplier), comment)
            kStr += inst("v_mul_lo_u32", product, product, operand, comment)
        return kStr


########################################
# Multiply scalar for 64bit
# product register, operand register, multiplier
########################################

def scalarStaticMultiply(product, operand, multiplier, tmpSgpr=None, comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    if multiplier == 0:
            return inst("s_mov_b64", product, hex(multiplier), comment)

    # TODO- to support non-pow2, need to use mul_32 and mul_hi_32 ?
    assert ((multiplier & (multiplier - 1)) == 0) # assert pow of 2

    multiplier_log2 = log2(multiplier)
    if multiplier_log2==0 and product == operand:
        return instCommentOnly(comment + " (multiplier is 1, do nothing)")
    else:
        # notice that the src-order of s_lshl_b64 is different from v_lshlrev_b32.
        return inst("s_lshl_b64", product, operand, hex(multiplier_log2), comment)

################################################################################
# RegisterPool
# Debugging register performance problems:
# - Enable self.db["PrintRP"] to see messages as vgprPool state changes.
# - Search for 'overlow' to see when pool grows dynamically - typically this
#   indicates growth for temps or other cases.
# - checkIn, checkout take optional tag but this is not widely used in tensile.
# - checkout returns vgpr index that was returned - can search disasm to see where
#   this vgpr is used.
################################################################################
class RegisterPool:
  class Status(Enum):
    Unavailable = 0
    Available = 1
    InUse = 2

  class Register:
    def __init__(self, status, tag):
      self.status = status
      self.tag = tag

  ########################################
  # Init
  # defaultPreventOverflow: control behavior of checkout and checkoutAligned when preventOverflow is not explicitly specificed.
  def __init__(self, size, type, defaultPreventOverflow, printRP=0):
    self.printRP=printRP
    self.type = type
    self.defaultPreventOverflow = defaultPreventOverflow
    self.pool = [self.Register(RegisterPool.Status.Unavailable, "init") for i in range(0,size)]
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
        self.pool.append(self.Register(RegisterPool.Status.Unavailable,tag))
    # mark as available
    for i in range(start, start+size):
      if self.pool[i].status == RegisterPool.Status.Unavailable:
        self.pool[i].status = RegisterPool.Status.Available
        self.pool[i].tag = tag
      elif self.pool[i].status == RegisterPool.Status.Available:
        printWarning("RegisterPool::add(%u,%u) pool[%u](%s) already available" % (start, size, i, self.pool[i].tag))
      elif self.pool[i].status == RegisterPool.Status.InUse:
        printWarning("RegisterPool::add(%u,%u) pool[%u](%s) already in use" % (start, size, i, self.pool[i].tag))
      else:
        raise RuntimeError("RegisterPool::add(%u,%u) pool[%u](%s) = %s" % (start, size, i, self.pool[i].tag, self.pool[i].status))
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
      if  self.pool[i].status == RegisterPool.Status.Available:
        self.pool[i].status = RegisterPool.Status.Unavailable
      elif self.pool[i].status == RegisterPool.Status.Unavailable:
        printWarning("RegisterPool::remove(%u,%u) pool[%u](%s) already unavailable" % (start, size, i, self.pool[i].tag))
      elif  self.pool[i].status == RegisterPool.Status.InUse:
        printWarning("RegisterPool::remove(%u,%u) pool[%u](%s) still in use" % (start, size, i, self.pool[i].tag))
      else:
        printExit("RegisterPool::remove(%u,%u) pool[%u](%s) = %s" % (start, size, i, self.pool[i].tag, self.pool[i].status))

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
        if self.pool[i+j].status != RegisterPool.Status.Available:
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
        self.pool[i].status = RegisterPool.Status.InUse
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
        if self.pool[i].status == RegisterPool.Status.Available:
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
        self.pool[i].status = RegisterPool.Status.InUse
        self.pool[i].tag = tag
      for i in range(0, overflow):
        if len(self.pool) < start:
          # this is padding to meet alignment requirements
          self.pool.append(self.Register(RegisterPool.Status.Available,tag))
        else:
          self.pool.append(self.Register(RegisterPool.Status.InUse,tag))
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
      if self.pool[i].status==RegisterPool.Status.Available:
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
        self.pool[i].status = RegisterPool.Status.Available
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
    return len(self.pool)


  ########################################
  # Number of available registers
  def available(self):
    numAvailable = 0
    for s in self.pool:
      if s.status == RegisterPool.Status.Available:
        numAvailable += 1
    return numAvailable

  ########################################
  # Size of registers of at least specified blockSize
  def availableBlock(self, blockSize, align):
    if blockSize ==0:
      blockSize = 1
    blocksAvail = 0
    consecAvailable = 0
    #for s in self.pool:
    for i in range(0, len(self.pool)):
      s = self.pool[i]
      if s.status == RegisterPool.Status.Available:
        if not (consecAvailable == 0 and i % align != 0):
          # do not increment if the first item is not aligned
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
      if s.status == RegisterPool.Status.Available:
        availCnt += 1
      else:
        break

    return availCnt


  ########################################
  def checkFinalState(self):
    for si in range(0,len(self.pool)):
      if self.pool[si].status == RegisterPool.Status.InUse:
        if self.printRP:
          print(self.state())
        raise RuntimeError("RegisterPool::checkFinalState: temp (%s, '%s') was never checked in." \
            %(si, self.pool[si].tag))
    print2("total vgpr count: %u\n"%self.size())

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
      if self.pool[i].status == RegisterPool.Status.Unavailable:
        stateStr += "." # 'removed', this indicates a fixed assignment from "remove", ie a non-tmp allocation
      elif self.pool[i].status == RegisterPool.Status.Available:
        stateStr += "|" # Can be allocated
      elif self.pool[i].status == RegisterPool.Status.InUse:
        stateStr += "#" # Checked out
    return stateStr

  def stateDetailed(self):
    for index, register in enumerate(self.vgprPool.pool):
        print("%u: %s"%(index, register.tag))
