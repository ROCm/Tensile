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

from .ActivationType import ActivationType
from .AsmUtils import inst, vgpr, sgpr, RegisterPool

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
#        ActivationMagicNumbers["Float16Max"], "Float16Max", "float16 max")
#    ```
#    this will generate ``regInitStr`` as
#    ```
#    s_mov_b32 sXX, "0x7bff" // float16 max
#    ```
#    if the key "Float16Max" is not found in sgprDict
# 2. class ActivationRegisterPool
#    A wrapper of RegisterPool. All the checkOut-ed registers will be checkIn-ed
#    at the end of the numBatches for loop.
#    When ActivationType is set to 'all', the registers will be checkIn-ed after
#    activation's gwvw for loop.
################################################################################

ActivationMagicNumbers = {"FloatPosInvln2": "0x3fb8aa3b", \
                      "FloatNegInvln2": "0xbfb8aa3b", \
                      "Float16PKPosInvln2": "0x3dc53dc5", \
                      "Float16PKNegInvln2": "0xbdc5bdc5", \
                      "DoublePosInvln2": ["0x652b82fe", "0x3ff71547"], \
                      "FloatMax": "0x7f7fffff", \
                      "Float16Max": "0x7bff", \
                      "FloatGeluK0": "0x3f4c422a", \
                      "Float16PKGeluK0": "0x3a623a62", \
                      "FloatGeluK1": "0x3d372713", \
                      "Float16PKGeluK1": "0x29b929b9", \
                      "Float16PKOne": "0x3c003c00" }

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
  # Public function
  def generateAssembly(self, cDataType, activationType, vgprIdx):
    kStr = ""

    if (activationType == 'abs'):
      kStr += self.getAbsAssembly(cDataType, vgprIdx)
    elif (activationType == 'exp'):
      kStr += self.getExpAssembly(cDataType, 1, vgprIdx)
    elif (activationType == 'leakyrelu'):
      kStr += self.getLeakyreluAssembly(cDataType, vgprIdx, "activationAlpha")
    elif (activationType == 'relu'):
      kStr += self.getReluAssembly(cDataType, vgprIdx)
    elif (activationType == 'sigmoid'):
      kStr += self.getSigmoidAssembly(cDataType, vgprIdx)
    elif (activationType == 'tanh'):
      kStr += self.getTanhAssembly(cDataType, vgprIdx, "activationAlpha", "activationBeta")
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

    if isinstance(args[0], int) and (not self.gprIsTempGpr):
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
  # Internal use
  def getExpAssembly(self, cDataType, sign, vgprIdx):
    kStr = ""
    if cDataType.isHalf():
      tag = "Float16PKPosInvln2" if (sign > 0) else "Float16PKNegInvln2"
      sgprmagic, regInitStr = self.getRegAndInitAssembly('s', False, 1, ActivationMagicNumbers[tag], tag, "exp(x) magic number")
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
      tag = "FloatPosInvln2" if (sign > 0) else "FloatNegInvln2"
      kStr += self.inst("v_mul_f32", self.getVgprStr(vgprIdx), ActivationMagicNumbers[tag], self.getVgprStr(vgprIdx), "exp step 1" )
      kStr += self.inst("v_exp_f32", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "exp step 2" )
    else:
      raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
    return kStr
  def getGeluAssembly(self, cDataType, vgprIdx):
    kStr = ""
    # Gelu(x) = 0.5 * x * (1 + tanh(k0 * x * (1 + k1 * x * x)))
    if cDataType.isHalf():
      pkStr = "_pk" if self.usePK else ""
      sgprk0, regK0InitStr = self.getRegAndInitAssembly('s', False, 1, ActivationMagicNumbers["Float16PKGeluK0"], "Float16PKGeluK0", "gelu k0")
      kStr += regK0InitStr
      sgprk1, regK1InitStr = self.getRegAndInitAssembly('s', False, 1, ActivationMagicNumbers["Float16PKGeluK1"], "Float16PKGeluK1", "gelu k1")
      kStr += regK1InitStr
      vgprtmp, needInit = self.vgprActivationPool.registerTemp(1, 'gelu vgpr temp', "activation exp")
      kStr += self.inst("v%s_mul_f16"%pkStr, vgpr(vgprtmp), self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "x * x" )
      vgprOne, regOneInitStr = self.getRegAndInitAssembly('v', True, 1, ActivationMagicNumbers["Float16PKOne"], "Float16PKOne", "1 in float16 pk format")
      kStr += regOneInitStr
      kStr += self.inst("v%s_fma_f16"%pkStr, vgpr(vgprtmp), vgpr(vgprtmp), sgpr(sgprk1), vgpr(vgprOne), "x^2 * k1 + 1" )
      kStr += self.inst("v%s_mul_f16"%pkStr, vgpr(vgprtmp), self.getVgprStr(vgprIdx), vgpr(vgprtmp), "x * (x^2 * k1 + 1)" )
      kStr += self.inst("v%s_mul_f16"%pkStr, vgpr(vgprtmp), sgpr(sgprk0), vgpr(vgprtmp), "k0 * x * (x^2 * k1 + 1)" )
      self.gprIsTempGpr = True
      kStr += self.getTanhAssembly(cDataType, vgprtmp, "", "")
      self.gprIsTempGpr = False
      kStr += self.inst("v%s_add_f16"%pkStr, vgpr(vgprtmp), vgpr(vgprOne), vgpr(vgprtmp), "1 + tanh(...)" )
      if regOneInitStr:
        self.vgprActivationPool.unregisterTemp(vgprOne)
      kStr += self.inst("v%s_mul_f16"%pkStr, vgpr(vgprtmp), self.getVgprStr(vgprIdx), vgpr(vgprtmp), "x * (1 + tanh(...))" )
      vgpr0Five, reg0FiveInitStr = self.getRegAndInitAssembly('v', True, 1, "0x38003800", "Float16PK0Five", "0.5 in float16 pk format")
      kStr += reg0FiveInitStr
      kStr += self.inst("v%s_mul_f16"%pkStr, self.getVgprStr(vgprIdx), vgpr(vgpr0Five), vgpr(vgprtmp), "0.5 * x * (1 + tanh(...))" )
      if reg0FiveInitStr:
        self.vgprActivationPool.unregisterTemp(vgpr0Five)
      self.vgprActivationPool.unregisterTemp(vgprtmp)
    elif cDataType.isSingle():
      vgprtmp, needInit = self.vgprActivationPool.registerTemp(1, 'gelu vgpr temp', "activation exp")
      kStr += self.inst("v_mul_f32", vgpr(vgprtmp), self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "x * x" )
      kStr += self.inst("v_mul_f32", vgpr(vgprtmp), ActivationMagicNumbers["FloatGeluK1"], vgpr(vgprtmp), "k1 * x * x" )
      kStr += self.inst("v_add_f32", vgpr(vgprtmp), 1.0, vgpr(vgprtmp), "1 + k1 * x * x" )
      kStr += self.inst("v_mul_f32", vgpr(vgprtmp), self.getVgprStr(vgprIdx), vgpr(vgprtmp), "x * (1 + k1 * x * x)" )
      kStr += self.inst("v_mul_f32", vgpr(vgprtmp), ActivationMagicNumbers["FloatGeluK0"], vgpr(vgprtmp), "k0 * x * (1 + k1 * x * x)" )
      self.gprIsTempGpr = True
      kStr += self.getTanhAssembly(cDataType, vgprtmp, "", "")
      self.gprIsTempGpr = False
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
      sgpr1, regInitStr = self.getRegAndInitAssembly('s', False, 1, ActivationMagicNumbers["Float16PKOne"], "Float16PKOne", "1 in pk float16 format")
      kStr += regInitStr
      if self.usePK:
        kStr += self.inst("v_pk_add_f16", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), sgpr(sgpr1), "1 + exp(-x)" )
        for i in range(0, 2):
          vgprsrc = self.getVgprStr(vgprIdx)
          vgprsrc += " dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d"%(i, i)
          kStr += self.inst("v_rcp_f16", self.getVgprStr(vgprIdx), vgprsrc, "1 / (1 + exp(-x))" )
      else:
        kStr += self.inst("v_add_f16", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), sgpr(sgpr1), "1 + exp(-x)" )
        kStr += self.inst("v_rcp_f16", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "1 / (1 + exp(-x))" )
    elif cDataType.isSingle():
      kStr += self.inst("v_add_f32", self.getVgprStr(vgprIdx), 1.0, self.getVgprStr(vgprIdx), "1 + exp(-x)" )
      kStr += self.inst("v_rcp_f32", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "1 / (1 + exp(-x))" )
    else:
      raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
    return kStr
  def getTanhAssembly(self, cDataType, vgprIdx, activationAlpha, activationBeta):
    kStr = ""
    if cDataType.isHalf():
      pkStr = "_pk" if self.usePK else ""
      # We don't need s_pack_ll_b32_b16 cause the input is already duplicated
      if activationAlpha:
        newVgprAlpha, regInitStr = self.getRegAndInitAssembly('v', False, 1, "0x40004000", "newVgprAlpha", "new vgpr alpha")
        if (not self.tanhInitAlpha):
          vgprTwo, regTwoInitStr = self.getRegAndInitAssembly('v', True, 1, "0x40004000", "Float16PK(2)", "2 in float16")
          kStr += regTwoInitStr
          kStr += self.inst("v%s_mul_f16"%pkStr, vgpr(newVgprAlpha), vgpr(vgprTwo), self.getSgprStr(activationAlpha), "new alpha = 2 * alpha")
          if regTwoInitStr:
            self.vgprActivationPool.unregisterTemp(vgprTwo)
          self.tanhInitAlpha = True
        kStr += self.inst("v%s_mul_f16"%pkStr, self.getVgprStr(vgprIdx), vgpr(newVgprAlpha), self.getVgprStr(vgprIdx), "x * alpha")
      else:
        vgprTwo, regTwoInitStr = self.getRegAndInitAssembly('v', True, 1, "0x40004000", "Float16PK(2)", "2 in float16")
        kStr += regTwoInitStr
        kStr += self.inst("v%s_mul_f16"%pkStr, self.getVgprStr(vgprIdx), vgpr(vgprTwo), self.getVgprStr(vgprIdx), "2 * x")
        if regTwoInitStr:
          self.vgprActivationPool.unregisterTemp(vgprTwo)
      kStr += self.getExpAssembly(cDataType, 1, vgprIdx)
      vgprOne, regOneInitStr = self.getRegAndInitAssembly('v', True, 1, ActivationMagicNumbers["Float16PKOne"], "Float16PKOne", "1 in pk float16 format")
      kStr += regOneInitStr
      kStr += self.inst("v%s_add_f16"%pkStr, self.getVgprStr(vgprIdx), vgpr(vgprOne), self.getVgprStr(vgprIdx), "e^2x + 1")
      if self.usePK:
        for i in range(0, 2):
          vgprsrc = self.getVgprStr(vgprIdx)
          vgprsrc += " dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d"%(i, i)
          kStr += self.inst("v_rcp_f16", self.getVgprStr(vgprIdx), vgprsrc, "1 / (1 + exp(-x))" )
      else:
        kStr += self.inst("v_rcp_f16", self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), "1 / (1 + exp(-x))" )
      vgprTwo, regNegTwoInitStr = self.getRegAndInitAssembly('v', True, 1, "0xc000c000", "Float16PK(-2)", "-2 in float16")
      kStr += regNegTwoInitStr
      kStr += self.inst("v%s_fma_f16"%pkStr, self.getVgprStr(vgprIdx), self.getVgprStr(vgprIdx), vgpr(vgprTwo), vgpr(vgprOne), "tanh(x) = (1 / (e^2x + 1)) * (-2) + 1")
      if regNegTwoInitStr:
        self.vgprActivationPool.unregisterTemp(vgprTwo)
      if regOneInitStr:
        self.vgprActivationPool.unregisterTemp(vgprOne)
      if activationBeta:
        kStr += self.inst("v%s_mul_f16"%pkStr, self.getVgprStr(vgprIdx), self.getSgprStr(activationBeta), self.getVgprStr(vgprIdx), "beta * tanh(x)")
    elif cDataType.isSingle():
      if activationAlpha:
        kStr += self.inst("v_mul_f32", self.getVgprStr(vgprIdx), self.getSgprStr(activationAlpha), self.getVgprStr(vgprIdx), "x * alpha")
      kStr += self.inst("v_mul_f32", self.getVgprStr(vgprIdx), 2.0, self.getVgprStr(vgprIdx), "2 * x")
      kStr += self.getExpAssembly(cDataType, 1, vgprIdx)
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
      kStr += activation.getTanhAssembly(self.dataType, 0, 1, 2)
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
