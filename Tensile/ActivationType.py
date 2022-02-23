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

from collections import OrderedDict
from .Common import printWarning

################################################################################
# This is the ActivationType class
# stringList:
#   This list stores the names of extra arguments, e.g.
#   y = (x > 0) ? x : x * alpha
# lookup:
#   This dict stores the supported activation types as keys and number of
#   arguments as values. Insert any new type before 'none' and 'all'. The
#   sequence of the table should match the enum in Activation.hpp.
#
# To add an activation type, see the instruction in Activation.py.
################################################################################

class ActivationAvailable:
  def __init__(self, canHalf=False, canSingle=False, canDouble=False, canBFloat16=False, canInt8=False, canInt16=False, canInt32=False):
    self.half = canHalf
    self.single = canSingle
    self.double = canDouble
    self.bfloat16 = canBFloat16
    self.int8 = canInt8
    self.int16 = canInt16
    self.int32 = canInt32

class ActivationTypeRegister:
  def __init__(self, name, extraArgs, canHalf=False, canSingle=False, canDouble=False, canBFloat16=False, canInt8=False, canInt16=False, canInt32=False):
    self.name = name
    self.extraArgs = extraArgs
    self.can = ActivationAvailable(canHalf, canSingle, canDouble, canBFloat16, canInt8, canInt16, canInt32)
  def typeAvailable(self, dataType):
    if dataType.isHalf() and self.can.half:
      return True
    elif dataType.isSingle() and self.can.single:
      return True
    elif dataType.isDouble() and self.can.double:
      return True
    elif dataType.isBFloat16() and self.can.bfloat16:
      return True
    elif dataType.isInt8() and self.can.int8:
      return True
    elif dataType.isInt32() and self.can.int32:
      return True
    return False


class ActivationType:
  stringList = ['alpha', 'beta', 'gamma', 'delta' ]
  # Exp is only for verification. So we will not return exp in the supported list.
                                                                           # Half,Single,Double,BFloat16,  Int8, Int16, Int32
  lookupVeri = OrderedDict([('exp',       ActivationTypeRegister('exp', 0,       True,  True, False,   False, False, False, False)) ])

  # Note: The BFloat16 gemm uses Single type activations. The int8 gemm uses int32 type activations.
                                                                               # Half,Single,Double,BFloat16,  Int8, Int16, Int32
  lookup = OrderedDict([('abs',         ActivationTypeRegister('abs', 0,         True,  True,  True,    True, False, False,  True)), \
                        ('clippedrelu', ActivationTypeRegister('clippedrelu', 2, True,  True,  True,   False, False, False,  True)), \
                        ('gelu',        ActivationTypeRegister('gelu', 0,        True,  True, False,   False, False, False, False)), \
                        ('leakyrelu',   ActivationTypeRegister('leakyrelu', 1,   True,  True,  True,   False, False, False,  True)), \
                        ('relu',        ActivationTypeRegister('relu', 0,        True,  True,  True,   False, False, False,  True)), \
                        ('sigmoid',     ActivationTypeRegister('sigmoid', 0,     True,  True, False,   False, False, False, False)), \
                        ('tanh',        ActivationTypeRegister('tanh', 2,        True,  True, False,   False, False, False, False)), \
                        ('none',        ActivationTypeRegister('none', 0)), \
                        ('all',         ActivationTypeRegister('all', 0)) ])
  def __init__(self, value):
    if isinstance(value, str):
      strValue = value.lower()
      if strValue in self.lookup:
        self.value = strValue
      elif strValue in self.lookupVeri:
        self.value = strValue
      else:
        raise RuntimeError("Unrecognized activation type %s"%value)
    elif isinstance(value, ActivationType):
      self.value = value.value
    else:
      raise RuntimeError("Unrecognized input type %s, should be string or ActivationType"%str(value))
  def getAdditionalArgNum(self):
    if self.value == 'all':
      maxArgNum = 0
      for key, activationInst in self.lookup.items():
        maxArgNum = max(maxArgNum, activationInst.extraArgs)
      return maxArgNum
    elif self.value in self.lookup:
      return self.lookup[self.value].extraArgs
    return 0
  def getAdditionalArgStringList(self, addPrefix=True):
    list = []
    for i in range(0, self.getAdditionalArgNum()):
      if addPrefix:
        list.append("activation" + self.stringList[i].capitalize())
      else:
        list.append(self.stringList[i])
    return list
  @classmethod
  def getEnumIndex(cls, enumStr):
    return list(cls.lookup.keys()).index(enumStr)
  @classmethod
  def getEnumStrList(cls, dataType, includeNone = True):
    enumList = []
    for key, activationInst in cls.lookup.items():
      if (((key != 'none') or includeNone) and (key != 'all')):
        if activationInst.typeAvailable(dataType):
          enumList.append(key)
    if not enumList:
      printWarning("No available activation for this data type %s.\n"%str(dataType))
    return enumList
  def state(self): return self.value.capitalize()
  def __repr__(self):
    return self.__str__()
  def __str__(self):
    return self.value.capitalize()
  def __eq__(self, other):
    if isinstance(other, str):
      return self.value == other.lower()
    elif isinstance(other, ActivationType):
      return self.value == other.value
    else:
      raise RuntimeError("Unrecognized type in rhs, should be string or ActivationType")
  def toEnum(self):
    return self.value.capitalize()
