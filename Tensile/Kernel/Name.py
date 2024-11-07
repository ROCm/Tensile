################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from ..CustomKernels import isCustomKernelConfig
from ..Common import parameterNameAbbreviations, printExit, validParameters
from ..SolutionStructs import ProblemType, Solution

def paramValueAbbr(key, value):
    if key == 'ISA':
      return str(value[0]) + str(value[1]) + ('%x' % value[2])
    elif isinstance(value, str):
      return ''.join([c for c in value if c.isupper()])
    elif isinstance(value, bool):
      return "1" if value else "0"
    elif isinstance(value, int):
      if value >= 0:
        return "%u" % value
      else: # -1 -> n1
        return "n%01u" % abs(value)
    elif isinstance(value, ProblemType):
      return str(value)
    elif isinstance(value, tuple):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += str(value[i])
      return abbrev
    elif isinstance(value, list):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += paramValueAbbr(key, value[i])
        if i < len(value)-1:
          abbrev += "_"
      return abbrev
    elif isinstance(value, dict):
      s =  "_".join(["%d%d"%(pos,k) for pos,k in value.items()])
      return s
    elif isinstance(value, float):
      val1 = int(value)
      val2 = int(round(value*100)) - int(value)*100
      if val2 > 0:
        s =  "%dp%s" % (val1,str(val2).zfill(2))
      else:
        s = "%d" % (val1)
      return s
    else:
      printExit('Parameter {key}={value} is new object type ({t})'.format(key=key, value=value, t=type(value)))
      return str(value)

def getNameMin(state, requiredParameters):
    if isCustomKernelConfig(state):
      return state["CustomKernelName"]

    name = ""
    first = True
    # put problem first
    if "ProblemType" in state:
      name += str(state["ProblemType"]) + "_"
    if "MacroTile0" in state \
        and "MacroTile1" in state \
        and "DepthU" in state:
      name += "%s%ux%ux%u_" \
          % ( parameterNameAbbreviations["MacroTile"], \
          state["MacroTile0"], state["MacroTile1"], state["DepthU"] )
    if "MatrixInstM" in state:
      name += "%s%ux%ux%ux%u_" \
          % ( parameterNameAbbreviations["MatrixInstruction"], \
          state["MatrixInstM"], state["MatrixInstN"], state["MatrixInstK"], state["MatrixInstB"])
    if "LdcEqualsLdd" in state:
      if state["LdcEqualsLdd"]:
        name += "SE_"
      else:
        name += "SN_"
    for key in sorted(state.keys()):
      if key in requiredParameters and key[0] != '_':
        if requiredParameters[key] and key != "CustomKernelName":
          if not first:
            name += "_"
          else:
            first = False
          name += "%s%s" % ( parameterNameAbbreviations[key], \
              paramValueAbbr(key, state[key]) )
    return name

def shortenFileBase(kernel, kernelMinNaming, maxFileName):
    base = getNameMin(kernel, kernelMinNaming)
    if len(base) <= maxFileName:
      return base

    import hashlib
    import base64  

    pivot = maxFileName * 3 // 4
    firstPart = base[:pivot]
    secondPart = base[pivot:]
    secondHash = hashlib.sha256(secondPart.encode()).digest()
    secondPart = base64.b64encode(secondHash, b'_-').decode()
    return firstPart + secondPart

def getNameSerial(state, serialNaming):
    data = serialNaming[0]
    numDigits = serialNaming[1]

    serial = 0
    multiplier = 1
    for paramName in sorted(state.keys()):
      if paramName in list(validParameters.keys()):
        paramValue = state[paramName]
        paramData = data[paramName]
        paramNameMultiplier = len(paramData)
        if paramValue in paramData:
          paramValueIdx = paramData.index(paramValue)
        serial += paramValueIdx * multiplier
        multiplier *= paramNameMultiplier
    name = "%s%0*u" % ("S" if isinstance(state, Solution) else "K", \
        numDigits, serial)
    return name

def getKernelFileBase(kernel, kernelMinNaming):
    if isCustomKernelConfig(kernel):
      fileBase = kernel["CustomKernelName"]
    #elif useShortNames:
    #  fileBase = getNameSerial(kernel, self.kernelSerialNaming)
    else:
      fileBase = shortenFileBase(kernel, kernelMinNaming, 64)
    return fileBase
