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

import ctypes
import math
import struct


from .ActivationType import ActivationType
from .AsmUtils import inst, vgpr, sgpr, RegisterPool
from .Common import printExit


################################################################################
# How to add an activation
# 1. Add a new type in ActivationType.py
# 2. Create a new getXXXAssembly function in class Activation
# 3. Add if-else condition in generateAssembly in class Activation
# 4. Add if-else condition in generateInlineAssemblyBody in class
#    ActivationInline
#
# Helper function(s)
# 1. getRegAndInitAssembly
#    ```
#    getRegAndInitAssembly(<v for vgpr/ s for sgpr>,
#                          <False for reg pool/ True for tmp reg pool>,
#                          <size of reg>,
#                          <init value>,
#                          <key>,
#                          <comment>)
#    ```
#    Returns
#    1. sgprinf: The original checkOut return value
#    2. regInitStr: The init instruction string
#
#    Example,
#    ```
#    sgprinf, regInitStr = self.getRegAndInitAssembly('s', False, 1, \
#        "0x3f4c422a", "FloatGeluK0", "float gelu k0")
#    ```
#    this will generate ``regInitStr`` as
#    ```
#    s_mov_b32 sXX, "0x3f4c422a" // float16 max
#    ```
#    if the key "FloatGeluK0" is not found in sgprDict
# 2. class ActivationRegisterPool
#    A wrapper of RegisterPool. All the checkOut-ed registers will be checkIn-ed
#    at the end of the numBatches for loop.
#    When ActivationType is set to 'all', the registers will be checkIn-ed after
#    activation's gwvw for loop.
################################################################################

ActivationMagicNumbers = {"FloatGeluK0": 0x3f4c422a, \
                          "FloatGeluK1": 0x3d372713, \
                          "Float16GeluK1": 0x29b9 }

# float32 union
class floatUnion(ctypes.Union):
  _fields_ = [('u', ctypes.c_uint), ('f', ctypes.c_float)]

def removeStringAfter(inputStr, symbolStr):
  pos = inputStr.find(symbolStr)
  if pos != -1:
    inputStr = inputStr[:pos]
  return inputStr

# Only compatible with funsion inst in AsmUtils.py
def extractLastInst(kStr):
  lastInstEndPos = kStr.rfind("\n")
  lastInstStr = kStr[:lastInstEndPos]
  lastInstStartPos = lastInstStr.rfind("\n")
  if lastInstStartPos != -1:
    lastInstStartPos += 1
  else:
    lastInstStartPos = 0
  lastInstStr = lastInstStr[lastInstStartPos:]
  commentStartPos = lastInstStr.find("//")
  if commentStartPos != -1:
    commentStr = lastInstStr[commentStartPos+2:].strip().rstrip()
  else:
    commentStr = ""
  lastInstStr = lastInstStr.replace("\"", "")
  lastInstStr = removeStringAfter(lastInstStr, "//")
  lastInstStr = removeStringAfter(lastInstStr, "\\")
  instList = lastInstStr.strip().split(" ")
  for idx, value in enumerate(instList):
    if (value.find(",") == 0):
      instList[idx] = value[1:]
    if (value.rfind(",") == len(value) - 1):
      instList[idx] = value[:len(value) - 1]
  instList.append(commentStr)
  return instList, lastInstStartPos

# Internal use
class ActivationRegisterPool:
  def __init__(self, checkOut, checkIn) -> None:
    self.checkOut = checkOut
    self.checkIn = checkIn
    self.dict = dict()
    self.tempDict = dict()
  # Internal use
  def register(self, num, name, comment = ""):
    if name in self.dict:
      return self.dict[name], False
    reg = self.checkOut(num, comment)
    self.dict[name] = reg
    return self.dict[name], True
  # Internal use
  def registerTemp(self, num, name, comment = ""):
    if name in self.tempDict:
      return self.tempDict[name], False
    reg = self.checkOut(num, comment)
    self.tempDict[name] = reg
    return self.tempDict[name], True
  # Internal use
  def unregisterTemp(self, reg):
    for key, val in list(self.tempDict.items()):
      if (reg == val):
        self.checkIn(val)
        del self.tempDict[key]
  # Internal use
  def unregisterAllTemp(self):
    for key, val in list(self.tempDict.items()):
      self.checkIn(val)
      del self.tempDict[key]
  # Internal use
  def unregisterAll(self):
    for key, val in list(self.dict.items()):
      self.checkIn(val)
      del self.dict[key]
    self.unregisterAllTemp()
class Activation:
  # Public function
  def __init__(self, checkOutSgpr, checkInSgpr, checkOutVgpr, checkInVgpr, vcc) -> None:
    self.inst = inst
    self.addGprPrefix = False
    self.gprInlineAsmMode = False
    self.gprIsTempGpr = False

    self.sgprActivationPool = ActivationRegisterPool(checkOutSgpr, checkInSgpr)
    self.vgprActivationPool = ActivationRegisterPool(checkOutVgpr, checkInVgpr)
    self.vcc = vcc
    self.usePK = True
    self.tanhInitAlpha = False
  # Public function
  def deinit(self):
    self.sgprActivationPool.unregisterAll()
    self.vgprActivationPool.unregisterAll()
    self.tanhInitAlpha = False
  # Public function. If true, generated code will add "ValuC+" before the given working v(s)gpr index.
  def setAddGprPrefix(self, value):
    self.addGprPrefix = value
  # Public function
  def generateAssembly(self, cDataType, activationType, vgprIdx):
    kStr = ""

    if (activationType == 'abs'):
      kStr += self.getAbsAssembly(cDataType, vgprIdx)
    elif (activationType == 'clippedrelu'):
      kStr += self.getClippedReluAssembly(cDataType, vgprIdx, "activationAlpha", "activationBeta")
    elif (activationType == 'exp'):
      kStr += self.getExpAssembly(cDataType, 1, vgprIdx)
    elif (activationType == 'leakyrelu'):
      kStr += self.getLeakyreluAssembly(cDataType, vgprIdx, "activationAlpha")
    elif (activationType == 'relu'):
      kStr += self.getReluAssembly(cDataType, vgprIdx)
    elif (activationType == 'sigmoid'):
      kStr += self.getSigmoidAssembly(cDataType, vgprIdx)
    elif (activationType == 'tanh'):
      kStr += self.getTanhAssembly(cDataType, 1, vgprIdx, "activationAlpha", "activationBeta")
    elif (activationType == 'gelu'):
      kStr += self.getGeluAssembly(cDataType, vgprIdx)
    return kStr
  # Internal use, for generating inline assembly format
  def overWriteInst(self, newInst):
    self.inst = newInst
  # Internal use, for generating inline assembly format
  def enableInlineAsmMode(self, enable):
    self.gprInlineAsmMode = enable
  # Internal use, for generating inline assembly format
  def getGprStr(self, gprType, *args):
    if self.gprInlineAsmMode and (not self.gprIsTempGpr):
      return "%%%u"%args[0]
    funcGpr = vgpr if gprType == "v" else sgpr
    if len(args) == 1:
      args = args[0]

    if isinstance(args[0], int) and self.addGprPrefix and (not self.gprIsTempGpr):
      vgprStr = "ValuC+%u"%args[0]
    else:
      vgprStr = args[0]

    if len(args) == 1:
      return funcGpr(vgprStr)
    else:
      args = args[1]
      return funcGpr(vgprStr, args)
  # Internal use, for generating inline assembly format
  def getVgprStr(self, *args):
    return self.getGprStr("v", args)
  # Internal use, for generating inline assembly format
  def getSgprStr(self, *args):
    return self.getGprStr("s", args)
  # Internal use
  def getRegAndInitAssembly(self, regType, isTemp, size, constantValue, tag, comment):
    kStr = ""
    if (regType == 's'):
      reg, needInit = self.sgprActivationPool.registerTemp(size, tag, comment) \
                      if isTemp else self.sgprActivationPool.register(size, tag, comment)
      if needInit:
        kStr += self.inst("s_mov_b32", sgpr(reg), constantValue, comment )
      return reg, kStr
    elif (regType == 'v'):
      reg, needInit = self.vgprActivationPool.registerTemp(size, tag, comment) \
                      if isTemp else self.vgprActivationPool.register(size, tag, comment)
      if needInit:
        kStr += self.inst("v_mov_b32", vgpr(reg), constantValue, comment)
      return reg, kStr
    raise RuntimeError("Only sgpr or vgpr is supported.")
  # Internal use
  def setUsePK(self, usePK):
    self.usePK = usePK
  # Internal use
  def magicNumToStr(self, cDataType, *args):
    if len(args) == 1:
      magicNum = args[0]
      uint32 = ctypes.c_uint(magicNum).value
      if self.usePK and cDataType.isHalf():
        uint32 = ((uint32 << 16) | uint32)
      magicNumStr = str(hex(uint32))
    else:
      raise RuntimeError("Currently does not support multiple args.")
    return magicNumStr
  # Internal use
  def getAbsAssembly(self, cDataType, vgprIdx):
    kStr = ""
    if cDataType.isHalf() or cDataType.isBFloat16():
      absMagic = "0x7fff7fff" if self.usePK else "0x7fff"
      kStr += self.inst("v_and_b32", self.getVgprStr(vgprIdx), absMagic, self.getVgprStr(vgprIdx), "Remove sign bit")
    elif cDataType.isSingle():
      kStr += self.inst("v_and_b32", self.getVgprStr(vgprIdx), "0x7fffffff", self.getVgprStr(vgprIdx), "Remove sign bit")
    elif cDataType.isDouble():
      kStr += self.inst("v_and_b32", self.getVgprStr(vgprIdx+1), "0x7fffffff", self.getVgprStr(vgprIdx+1), "Remove sign bit")
    elif cDataType.isInt32():
      vgprtemp, vgprPInit = self.vgprActivationPool.registerTemp(1, "vgprtmp_absi32")
      kStr += self.inst("v_sub_i32", vgpr(vgprtemp), 0, self.getVgprStr(vgprIdx), "x2 = -x")
      kStr += self.inst("v_max_i32", self.getVgprStr(vgprIdx), vgpr(vgprtemp), self.getVgprStr(vgprIdx), "y = max(x, x2)")
    else:
      raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
    return kStr
  # Internal Use
  def getClippedReluAssembly(self, cDataType, vgprIdx, activationAlpha, activationBeta):
    kStr = ""
    if cDataType.isHalf():
      for i in range(0, 2):
        instStr = "%s src0_sel:WORD_%d src1_sel:WORD_0"%(self.getSgprStr(activationAlpha), i)
        kStr += self.inst("v_cmp_ge_f16", self.vcc, self.getVgprStr(vgprIdx), instStr, "x > alpha ?")
        vcc = self.vcc + " dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d src1_sel:WORD_%d"%(i, i, i)
        kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx), 0.0, self.getVgprStr(vgprIdx), vcc, "set x to 0 if < alpha")
      kStr += self.inst("v_pk_min_f16", self.getVgprStr(vgprIdx), self.getSgprStr(activationBeta), self.getVgprStr(vgprIdx), "min(x, beta)")
    elif cDataType.isSingle():
      kStr += self.inst("v_cmp_ge_f32", self.vcc, self.getVgprStr(vgprIdx), self.getSgprStr(activationAlpha), "x >= alpha ?")
      kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx), 0.0, self.getVgprStr(vgprIdx), self.vcc, "set x to 0 if < alpha")
      kStr += self.inst("v_min_f32", self.getVgprStr(vgprIdx), self.getSgprStr(activationBeta), self.getVgprStr(vgprIdx), "min(x, beta)")
    elif cDataType.isDouble():
      kStr += self.inst("v_cmp_ge_f64", self.vcc, self.getVgprStr(vgprIdx, 2), self.getSgprStr(activationAlpha, 2), "x >= alpha ?")
      kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx), 0, self.getVgprStr(vgprIdx), self.vcc, "set x to 0 if < 0")
      kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx+1), 0, self.getVgprStr(vgprIdx+1), self.vcc, "set x to 0 if < 0")
      kStr += self.inst("v_min_f64", self.getVgprStr(vgprIdx, 2), self.getSgprStr(activationBeta, 2), self.getVgprStr(vgprIdx, 2), "min(x, beta)")
    elif cDataType.isInt32():
      kStr += self.inst("v_cmp_ge_i32", self.vcc, self.getVgprStr(vgprIdx), self.getSgprStr(activationAlpha), "x >= alpha ?")
      kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx), 0.0, self.getVgprStr(vgprIdx), self.vcc, "set x to 0 if < alpha")
      kStr += self.inst("v_min_i32", self.getVgprStr(vgprIdx), self.getSgprStr(activationBeta), self.getVgprStr(vgprIdx), "min(x, beta)")
    return kStr
  # Internal Use
  def getExpMagicStrAndComment(self, cDataType, coef):
    # Double = [0x652b82fe, 0x3ff71547]
    # Float  = 0x3fb8aa3b
    if cDataType.isDouble():
      printExit("Currently Exp Magic to string is not supported for double.")
    invLn2 = floatUnion(f=math.log(math.e,2))
    value = coef * invLn2.f
    commentStr = str(hex(floatUnion(f=coef).u))
    coefUnion = floatUnion(f=value)
    if cDataType.isHalf():
      magicNum = struct.unpack('<H', struct.pack('<e', coefUnion.f))[0]
      if self.usePK:
        magicNum = ctypes.c_uint(magicNum).value
        magicNum = ((magicNum << 16) | magicNum)
    elif cDataType.isSingle():
      magicNum = coefUnion.u
    magicNumStr = str(hex(magicNum))
    commentStr = "%s = (%s * invln2(%s))"%(magicNumStr, commentStr, str(hex(invLn2.u)))
    return magicNumStr, commentStr
  # Internal use
  def getExpAssembly(self, cDataType, coef, vgprIdx):
    kStr = ""
    if cDataType.isHalf():
      magicNumStr, commentStr = self.getExpMagicStrAndComment(cDataType, coef)
      sgprmagic, regInitStr = self.getRegAndInitAssembly('s', False, 1, magicNumStr, magicNumStr, "exp(x) %s"%commentStr)
      kStr += regInitStr
      if self.usePK:
        kStr += self.inst("v_pk_mul_f16", self.getVgprStr(vgprIdx), sgpr(sgprmagic), self.getVgprStr(vgprIdx), "exp step 1" )
        for i in range(0, 2):
          vgprsrc = self.getVgprStr(vgprIdx)
          vgprsrc += " dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d"%(i, i)
          kStr += self.inst("v_exp_f16", self.getVgprStr(vgprIdx), vgprsrc, "exp step %d"%(i + 2) )
      else:
        kStr += self.inst("v_mul_f16", self.getVgprStr(vgprIdx), sgpr(sgprmagic), self.getVgprStr(vgprIdx), "exp step 1" )
        kStr += self.inst("v_exp_f16", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "exp step 2" )
    elif cDataType.isSingle():
      magicNumStr, commentStr = self.getExpMagicStrAndComment(cDataType, coef)
      kStr += self.inst("v_mul_f32", self.getVgprStr(vgprIdx), magicNumStr, self.getVgprStr(vgprIdx), "exp step 1: %s"%commentStr )
      kStr += self.inst("v_exp_f32", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "exp step 2" )
    else:
      raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
    return kStr
  def getGeluAssembly(self, cDataType, vgprIdx):
    kStr = ""
    # Gelu(x) = 0.5 * x * (1 + tanh(k0 * x * (1 + k1 * x * x)))
    if cDataType.isHalf():
      pkStr = "_pk" if self.usePK else ""
      flt16GeluK0Str = self.magicNumToStr(cDataType, ActivationMagicNumbers["Float16GeluK1"])
      sgprk1, regK1InitStr = self.getRegAndInitAssembly('s', False, 1, flt16GeluK0Str, "Float16GeluK1", "gelu k1")
      kStr += regK1InitStr
      vgprtmp, needInit = self.vgprActivationPool.registerTemp(1, 'gelu vgpr temp', "activation exp")
      kStr += self.inst("v%s_mul_f16"%pkStr, vgpr(vgprtmp), self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "x * x" )
      if self.usePK:
        vgprOneStr = "1.0 op_sel_hi:[1,1,0,1]"
      else:
        vgprOneStr = 1.0
      kStr += self.inst("v%s_fma_f16"%pkStr, vgpr(vgprtmp), vgpr(vgprtmp), sgpr(sgprk1), vgprOneStr, "x^2 * k1 + 1" )
      kStr += self.inst("v%s_mul_f16"%pkStr, vgpr(vgprtmp), self.getVgprStr(vgprIdx), vgpr(vgprtmp), "x * (x^2 * k1 + 1)" )
      coef = floatUnion(u=ActivationMagicNumbers["FloatGeluK0"])
      self.gprIsTempGpr = True
      kStr += self.getTanhAssembly(cDataType, coef.f, vgprtmp, "", "")
      self.gprIsTempGpr = False
      instList, lastInstStartPos = extractLastInst(kStr)
      if (instList[0] == "v%s_fma_f16"%pkStr) and (instList[1] == vgpr(vgprtmp)) and (instList[4] == "1.0"):
        commentStr = "(" + instList[len(instList) - 1] + ") + 1 (fused)"
        kStr = kStr[:lastInstStartPos]
        instList[4] = "2.0 %s"%instList[5] if self.usePK else "2.0"
        kStr += self.inst(instList[0], instList[1], instList[2], instList[3], instList[4], commentStr)
      else:
        lastStr = vgpr(vgprtmp)
        lastStr += " op_sel_hi:[0,1,1]" if self.usePK else ""
        kStr += self.inst("v%s_add_f16"%pkStr, vgpr(vgprtmp), 1.0, lastStr, "1 + tanh(...)" )
      kStr += self.inst("v%s_mul_f16"%pkStr, vgpr(vgprtmp), self.getVgprStr(vgprIdx), vgpr(vgprtmp), "x * (1 + tanh(...))" )
      lastStr = vgpr(vgprtmp)
      lastStr += " op_sel_hi:[0,1,1]" if self.usePK else ""
      kStr += self.inst("v%s_mul_f16"%pkStr, self.getVgprStr(vgprIdx), 0.5, lastStr, "0.5 * x * (1 + tanh(...))" )
      self.vgprActivationPool.unregisterTemp(vgprtmp)
    elif cDataType.isSingle():
      vgprtmp, needInit = self.vgprActivationPool.registerTemp(1, 'gelu vgpr temp', "activation exp")
      fltGeluK0Str = self.magicNumToStr(cDataType, ActivationMagicNumbers["FloatGeluK1"])
      kStr += self.inst("v_mul_f32", vgpr(vgprtmp), fltGeluK0Str, self.getVgprStr(vgprIdx), "k1 * x" )
      kStr += self.inst("v_fma_f32", vgpr(vgprtmp), self.getVgprStr(vgprIdx), vgpr(vgprtmp), 1.0, "1 + (k1 * x * x)" )
      kStr += self.inst("v_mul_f32", vgpr(vgprtmp), self.getVgprStr(vgprIdx), vgpr(vgprtmp), "x * (1 + k1 * x * x)" )
      coef = floatUnion(u=ActivationMagicNumbers["FloatGeluK0"])
      self.gprIsTempGpr = True
      kStr += self.getTanhAssembly(cDataType, coef.f, vgprtmp, "", "")
      self.gprIsTempGpr = False
      instList, lastInstStartPos = extractLastInst(kStr)
      if (instList[0] == "v_fma_f32") and (instList[1] == vgpr(vgprtmp)) and (instList[4] == "1.0"):
        commentStr = "(" + instList[len(instList) - 1] + ") + 1 (fused)"
        kStr = kStr[:lastInstStartPos]
        instList[4] = 2.0
        kStr += self.inst(instList[0], instList[1], instList[2], instList[3], instList[4], commentStr)
      else:
        kStr += self.inst("v_add_f32", vgpr(vgprtmp), 1.0, vgpr(vgprtmp), "1 + tanh(...)" )
      kStr += self.inst("v_mul_f32", vgpr(vgprtmp), self.getVgprStr(vgprIdx), vgpr(vgprtmp), "x * (1 + tanh(...))" )
      kStr += self.inst("v_mul_f32", self.getVgprStr(vgprIdx), 0.5, vgpr(vgprtmp), "0.5 * x * (1 + tanh(...))" )
      self.vgprActivationPool.unregisterTemp(vgprtmp)
    else:
      raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
    return kStr
  def getLeakyreluAssembly(self, cDataType, vgprIdx, activationAlpha):
    kStr = ""
    if cDataType.isHalf():
      # We don't need s_pack_ll_b32_b16 cause the input is already duplicated
      vgprtmp, needInit = self.vgprActivationPool.registerTemp(1, "tmp for leaky relu")
      kStr += self.inst("v_pk_mul_f16", vgpr(vgprtmp), self.getSgprStr(activationAlpha), self.getVgprStr(vgprIdx), "tmp = x * alpha")
      for i in range(0, 2):
        kStr += self.inst("v_cmp_ge_f16", self.vcc, self.getVgprStr(vgprIdx), ("0.0 src0_sel:WORD_%d src1_sel:WORD_0"%i), "x > 0 ?")
        vcc = self.vcc + " dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d src1_sel:WORD_%d"%(i, i, i)
        kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx), vgpr(vgprtmp), self.getVgprStr(vgprIdx), vcc, "set x to tmp if < 0")
      self.vgprActivationPool.unregisterTemp(vgprtmp)
    elif cDataType.isSingle():
      vgprtmp, needInit = self.vgprActivationPool.registerTemp(1, "tmp for leaky relu")
      kStr += self.inst("v_mul_f32", vgpr(vgprtmp), self.getSgprStr(activationAlpha), self.getVgprStr(vgprIdx), "tmp = x * alpha")
      kStr += self.inst("v_cmp_ge_f32", self.vcc, self.getVgprStr(vgprIdx), 0.0, "x >= 0 ?")
      kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx), vgpr(vgprtmp), self.getVgprStr(vgprIdx), self.vcc, "set x to tmp if < 0")
      self.vgprActivationPool.unregisterTemp(vgprtmp)
    elif cDataType.isDouble():
      vgprtmp, needInit = self.vgprActivationPool.registerTemp(2, "tmp for leaky relu")
      kStr += self.inst("v_mul_f64", vgpr(vgprtmp, 2), self.getSgprStr(activationAlpha, 2), self.getVgprStr(vgprIdx, 2), "tmp = x * alpha")
      kStr += self.inst("v_cmp_ge_f64", self.vcc, self.getVgprStr(vgprIdx, 2), 0.0, "x >= 0 ?")
      # No v_cndmask_b64
      kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx), vgpr(vgprtmp), self.getVgprStr(vgprIdx), self.vcc, "set x to tmp if < 0")
      kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx+1), vgpr(vgprtmp + 1), self.getVgprStr(vgprIdx+1), self.vcc, "set x to tmp if < 0")
      self.vgprActivationPool.unregisterTemp(vgprtmp)
    elif cDataType.isInt32():
      vgprtmp, needInit = self.vgprActivationPool.registerTemp(1, "tmp for leaky relu")
      kStr += self.inst("v_mul_lo_u32", vgpr(vgprtmp), self.getSgprStr(activationAlpha), self.getVgprStr(vgprIdx), "tmp = x * alpha")
      kStr += self.inst("v_cmp_ge_i32", self.vcc, self.getVgprStr(vgprIdx), 0, "x >= 0 ?")
      kStr += self.inst("v_cndmask_b32", self.getVgprStr(vgprIdx), vgpr(vgprtmp), self.getVgprStr(vgprIdx), self.vcc, "set x to tmp if < 0")
      self.vgprActivationPool.unregisterTemp(vgprtmp)
    else:
      raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
    return kStr
  def getReluAssembly(self, cDataType, vgprIdx):
    kStr = ""
    if cDataType.isHalf():
      kStr += self.inst("v_pk_max_f16", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), 0, "x = max(0, x)" )
    elif cDataType.isSingle():
      kStr += self.inst("v_max_f32", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), 0, "x = max(0, x)" )
    elif cDataType.isDouble():
      kStr += self.inst("v_max_f64", self.getVgprStr(vgprIdx, 2), self.getVgprStr(vgprIdx, 2), 0, "x = max(0, x)" )
    elif cDataType.isInt32():
      kStr += self.inst("v_max_i32", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), 0, "x = max(0, x)" )
    else:
      raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
    return kStr
  def getSigmoidAssembly(self, cDataType, vgprIdx):
    kStr = ""
    kStr += self.getExpAssembly(cDataType, -1, vgprIdx)
    if cDataType.isHalf():
      if self.usePK:
        lastStr = self.getVgprStr(vgprIdx) + " op_sel_hi:[0,1,1]"
        kStr += self.inst("v_pk_add_f16", self.getVgprStr(vgprIdx), 1.0, lastStr, "1 + exp(-x)" )
        for i in range(0, 2):
          vgprsrc = self.getVgprStr(vgprIdx)
          vgprsrc += " dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d"%(i, i)
          kStr += self.inst("v_rcp_f16", self.getVgprStr(vgprIdx), vgprsrc, "1 / (1 + exp(-x))" )
      else:
        kStr += self.inst("v_add_f16", self.getVgprStr(vgprIdx), 1.0, self.getVgprStr(vgprIdx), "1 + exp(-x)" )
        kStr += self.inst("v_rcp_f16", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "1 / (1 + exp(-x))" )
    elif cDataType.isSingle():
      kStr += self.inst("v_add_f32", self.getVgprStr(vgprIdx), 1.0, self.getVgprStr(vgprIdx), "1 + exp(-x)" )
      kStr += self.inst("v_rcp_f32", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "1 / (1 + exp(-x))" )
    else:
      raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
    return kStr
  def getTanhAssembly(self, cDataType, coef, vgprIdx, activationAlpha, activationBeta):
    kStr = ""
    if cDataType.isHalf():
      pkStr = "_pk" if self.usePK else ""
      # We don't need s_pack_ll_b32_b16 cause the input is already duplicated
      if activationAlpha:
        kStr += self.inst("v%s_mul_f16"%pkStr, self.getVgprStr(vgprIdx), self.getSgprStr(activationAlpha), self.getVgprStr(vgprIdx), "x * alpha")
      coef = coef * 2
      kStr += self.getExpAssembly(cDataType, coef, vgprIdx)
      lastStr = self.getVgprStr(vgprIdx)
      lastStr += " op_sel_hi:[0,1,1]" if self.usePK else ""
      kStr += self.inst("v%s_add_f16"%pkStr, self.getVgprStr(vgprIdx), 1.0, lastStr, "e^2x + 1")
      if self.usePK:
        for i in range(0, 2):
          vgprsrc = self.getVgprStr(vgprIdx)
          vgprsrc += " dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d"%(i, i)
          kStr += self.inst("v_rcp_f16", self.getVgprStr(vgprIdx), vgprsrc, "1 / (1 + exp(-x))" )
      else:
        kStr += self.inst("v_rcp_f16", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "1 / (1 + exp(-x))" )
      lastStr = "1.0"
      lastStr += " op_sel_hi:[0,1,0,1]" if self.usePK else ""
      kStr += self.inst("v%s_fma_f16"%pkStr, self.getVgprStr(vgprIdx), -2.0, self.getVgprStr(vgprIdx), lastStr, "tanh(x) = (1 / (e^2x + 1)) * (-2) + 1")
      if activationBeta:
        kStr += self.inst("v%s_mul_f16"%pkStr, self.getVgprStr(vgprIdx), self.getSgprStr(activationBeta), self.getVgprStr(vgprIdx), "beta * tanh(x)")
    elif cDataType.isSingle():
      if activationAlpha:
        kStr += self.inst("v_mul_f32", self.getVgprStr(vgprIdx), self.getSgprStr(activationAlpha), self.getVgprStr(vgprIdx), "x * alpha")
      coef = coef * 2
      kStr += self.getExpAssembly(cDataType, coef, vgprIdx)
      kStr += self.inst("v_add_f32", self.getVgprStr(vgprIdx), 1.0, self.getVgprStr(vgprIdx), "e^2x + 1")
      kStr += self.inst("v_rcp_f32", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "1 / (e^2x + 1)")
      kStr += self.inst("v_fma_f32", self.getVgprStr(vgprIdx), -2.0, self.getVgprStr(vgprIdx), 1.0, "(-2) * (1 / (e^2x + 1)) + 1")
      if activationBeta:
        kStr += self.inst("v_mul_f32", self.getVgprStr(vgprIdx), self.getSgprStr(activationBeta), self.getVgprStr(vgprIdx), "beta * tanh(x)")
    else:
      raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
    return kStr

def inlineInst(str):
  return "\"" + str + "\\n\\t\""

def addSpace(alignStr, str):
  totalLength = len(alignStr) + len(str)
  return '{message: >{width}}'.format(message=str, width=totalLength)

class ActivationInline:
  def __init__(self, wavefrontSize, dataType) -> None:
    self.wavefrontSize = wavefrontSize
    self.dataType = dataType
    self.asmStr = "asm("

  # Public Function
  def generateInlineAssemblyFunction(self, activationType):
    kStr = ""
    if activationType == 'none':
      return kStr

    ptrStr = self.dataType.toDevice("HIP")
    names = ""
    if activationType == 'all':
      names += ",\n"
      names += "  Tensile::ActivationType const activationType"
    for name in activationType.getAdditionalArgStringList(False):
      names += ",\n"
      names += "  %s const %s"%(ptrStr, name)
    changeLine = "\n  " if names else ""
    kStr += "__device__ inline %s activation(%s%s value%s)\n{\n"%(ptrStr, changeLine, ptrStr, names)
    # function body
    if activationType == 'all':
      for index, enumStr in enumerate(ActivationType.getEnumStrList(self.dataType, includeNone=False)):
        if index == 0:
          kStr += "  if (activationType == Tensile::ActivationType::%s) {\n"%(ActivationType(enumStr).toEnum())
        else:
          kStr += " else if (activationType == Tensile::ActivationType::%s) {\n"%(ActivationType(enumStr).toEnum())
        kStr += self.generateInlineAssemblyBody(4, enumStr)
        kStr += "  }"
      kStr += "\n"
    else:
      kStr += self.generateInlineAssemblyBody(2, activationType)
    # function body end
    kStr += "  return value;\n"
    kStr += "}\n"
    return kStr
  # Internal Function
  def generateInlineAssemblyBody(self, spaces, activationType):
    ptrStr = self.dataType.toDevice("HIP")
    sgprPool = RegisterPool(0, 's', defaultPreventOverflow=False, printRP=0)
    vgprPool = RegisterPool(0, 'v', defaultPreventOverflow=False, printRP=0)
    activation = Activation(sgprPool.checkOut, sgprPool.checkIn, vgprPool.checkOut, vgprPool.checkIn, \
                            self.vcc)
    activation.overWriteInst(self.instInlineAsm)
    activation.enableInlineAsmMode(True)
    activation.setUsePK(False)
    kStr = ""
    padSpacesStr = ' ' * spaces
    asm = padSpacesStr + self.asmStr
    if (activationType == 'abs'):
      if self.dataType.isHalf() or self.dataType.isBFloat16():
        unionDataTypeStr = "_Float16" if self.dataType.isHalf() else "BFloat16"
        unionName = "f16_union" if self.dataType.isHalf() else "bf16_union"
        kStr += (padSpacesStr + "union {\n")
        kStr += (padSpacesStr + "  %s f;\n"%unionDataTypeStr)
        kStr += (padSpacesStr + "  short s;\n")
        kStr += (padSpacesStr + "} %s;\n"%unionName)
        kStr += (padSpacesStr + "%s.f = value;\n"%unionName)
        kStr += (padSpacesStr + "%s.s = %s.s & 0x7fff;\n"%(unionName, unionName))
        kStr += (padSpacesStr + "value = %s.f;\n"%unionName)
      elif (self.dataType.isSingle() or self.dataType.isDouble() or self.dataType.isInt32()):
        kStr += (padSpacesStr + "value = abs(value);\n")
      else:
        raise RuntimeError("Unrecognized data type %s."%self.dataType)
    elif (activationType == 'clippedrelu'):
      if (self.dataType.isSingle() or self.dataType.isHalf() or self.dataType.isDouble()):
        kStr += (padSpacesStr + "value = (value >= alpha) ? min(value, beta) : 0.0;\n")
      elif self.dataType.isInt32():
        kStr += (padSpacesStr + "value = (value >= alpha) ? min(value, beta) : 0;\n")
    elif (activationType == 'exp'):
      kStr += (asm + "// Exp\n")
      kStr += activation.getExpAssembly(self.dataType, 1, 0)
      kStr += addSpace(asm, ": \"+v\"(value) : \n")
      kStr += self.getRequiredRegStr(asm, vgprPool.size(), sgprPool.size())
    elif (activationType == 'gelu'):
      kStr += (asm + " // gelu\n")
      kStr += activation.getGeluAssembly(self.dataType, 0)
      kStr += addSpace(asm, ": \"+v\"(value) : \n")
      kStr += self.getRequiredRegStr(asm, vgprPool.size(), sgprPool.size())
    elif (activationType == 'leakyrelu'):
      if (self.dataType.isSingle() or self.dataType.isHalf() or self.dataType.isDouble()):
        kStr += (padSpacesStr + "value = (value >= 0.0) ? value : (value * alpha);\n")
      elif self.dataType.isInt32():
        kStr += (padSpacesStr + "value = (value >= 0) ? value : (value * alpha);\n")
      else:
        raise RuntimeError("Unsupported data type %s."%ptrStr)
    elif (activationType == 'relu'):
      if (self.dataType.isSingle() or self.dataType.isHalf() or self.dataType.isDouble()):
        kStr += (padSpacesStr + "value = max(0.0, value);\n")
      elif self.dataType.isInt32():
        kStr += (padSpacesStr + "value = max(0, value);\n")
      else:
        raise RuntimeError("Unsupported data type %s."%ptrStr)
    elif (activationType == 'sigmoid'):
      kStr += (asm + " // Sigmoid\n")
      kStr += activation.getSigmoidAssembly(self.dataType, 0)
      kStr += addSpace(asm, ": \"+v\"(value) : \n")
      kStr += self.getRequiredRegStr(asm, vgprPool.size(), sgprPool.size())
    elif (activationType == 'tanh'):
      kStr += (asm + " // tanh\n")
      kStr += activation.getTanhAssembly(self.dataType, 1, 0, 1, 2)
      kStr += addSpace(asm, ": \"+v\"(value) : \"s\"(alpha), \"s\"(beta)\n")
      kStr += self.getRequiredRegStr(asm, vgprPool.size(), sgprPool.size())
    else:
      if (activationType != 'none'):
        raise RuntimeError("Unrecognized type %s."%activationType)
    return kStr
  # Internal use. Automatically gets the required vgprs and sgprs for inline assembly
  def getRequiredRegStr(self, spaceAlignStr, numOfVgpr, numOfSgpr):
    requiredReg = []
    for i in range(0, numOfVgpr):
      requiredReg.append("\"v%d\""%i)
    for i in range(0, numOfSgpr):
      requiredReg.append("\"s%d\""%i)
    requiredStr = ""
    if (len(requiredReg) > 0):
      requiredStr = requiredReg[0]
      for i in range(1, len(requiredReg)):
        requiredStr += ", %s"%requiredReg[i]
    kStr = ""
    kStr += addSpace(spaceAlignStr,":%s);\n"%requiredStr)
    return kStr
  # Internal use. Overwrite inst in class Activation
  def instInlineAsm(self, *args):
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
    instStr = addSpace(self.asmStr, inlineInst(instStr))
    line = "%-50s // %s\n" % (instStr, comment)
    return line
  # FIXME: Copy from KernelWriterAssembly
  # Internal use.
  @property
  def vcc(self) -> str:
    if self.wavefrontSize == 64:
      return "vcc"
    else:
      return "vcc_lo"
