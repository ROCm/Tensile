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

from __future__ import print_function
from .Common import globalParameters, printExit
# Global to print module names around strings
printModuleNames = 0

"""
Base class for Modules, Instructions, etc
Item is a atomic collection of or more instructions and commentsA
"""
class Item:
  pass

  def toStr(self):
    return str(self)

  def countType(self,ttype):
    return int(isinstance(self, ttype))


"""
Modules contain lists of text instructions, Inst objects, or additional modules
They can be easily converted to string that represents all items in the list
and can be mixed with standard text.
The intent is to allow the kernel writer to express the structure of the
code (ie which instructions are a related module) so the scheduler can later
make intelligent and legal transformations.
"""
class Module(Item):
  def __init__(self, name=""):
    self.name = name
    self.itemList = []

  def __str__(self):
    s = ""
    if printModuleNames:
      s += "// %s { \n" % self.name
    s += "".join([str(x) for x in self.itemList])
    if printModuleNames:
      s += "// } %s\n" % self.name
    return s

  """
  Add specified item to the list of items in the module.
  Item MUST be a Item (not a string) - can use
  addText(...)) to add a string.
  All additions to itemList should use this function.

  Returns item to facilitate one-line create/add patterns
  """
  def addCode(self, item):
    #assert (isinstance(item, Item)) # for debug
    if isinstance(item,Item):
      self.itemList.append(item)
    elif isinstance(item,str):
      self.addCode(TextBlock(item))
    else:
      assert 0, "unknown item type (%s) for Module.addCode. item=%s"%(type(item), item)
    return item

  """
  Convenience function to format arg as a comment and add TextBlock item
  This comment is a single line /* MYCOMMENT  */
  """
  def addComment0(self, comment):
    self.addCode(TextBlock("/* %s */\n"%comment))

  """
  Convenience function to format arg as a comment and add TextBlock item
  This comment is a blank line followed by /* MYCOMMENT  */
  """
  def addComment1(self, comment):
    self.addCode(TextBlock("\n/* %s */\n"%comment))

  """
  Convenience function to construct a single Inst and add to items
  """
  def addInst(self, *args):
    self.addCode(Inst(*args))

  """
  Convenience function to construct a TextBlock and add to items
  """
  def addText(self,text):
    self.addCode(TextBlock(text))

  def prettyPrint(self,indent=""):
    print("%sModule %s:"% (indent,self.name))
    for i in self.itemList:
      if isinstance(i, Module):
        i.prettyPrint(indent+"  ")
      elif isinstance(i, str):
        print(indent, '"', str(i).strip('\n'), '"')
      else: # Inst
          print(indent, "%s: [ %s ]" % \
              (i.__class__.__name__, str(i).strip('\n')))

  """
  Count number of items with specified type in this Module
  Will recursively count occurrences in submodules
  (Overrides Item.countType)
  """
  def countType(self,ttype):
    count=0
    for i in self.itemList:
      if isinstance(i, Module):
        count += i.countType(ttype)
      else:
        count += int(isinstance(i, ttype))
    return count

  def count(self):
    count=0
    for i in self.itemList:
      if isinstance(i, Module):
        count += i.count()
      else:
        count += 1
    return count

  """
  Return list of items in the Module
  Items may be other Modules, TexBlock, or Inst
  """
  def items(self):
    return self.itemList

  """
  Return flattened list of items in the Module
  Items in sub-modules will be flattened into single list
  Items may be TexBlock or Inst
  """
  def flatitems(self):
    flatitems = []
    for i in self.itemList:
      if isinstance(i, Module):
        flatitems += i.flatitems()
      else:
        flatitems.append(i)
    return flatitems


class StructuredModule(Module):
  def __init__(self, name=None):
    Module.__init__(self,name)
    self.header = Module("header")
    self.middle = Module("middle")
    self.footer =  Module("footer")

    self.addCode(self.header)
    self.addCode(self.middle)
    self.addCode(self.footer)


"""
Label that can be the target of a jump.
"""
class Label (Item):
  def __init__(self, labelNum, comment):
    self.labelNum = labelNum
    self.comment = comment

  def __str__(self):
    t = "label_%04u:" % (self.labelNum)
    if self.comment:
      t += "  /// %s" % self.comment
    t += "\n"
    return t


"""
An unstructured block of text that can contain comments and instructions
"""
class TextBlock(Item):
  def __init__(self,text):
    assert(isinstance(text, str))
    self.text = text

  def __str__(self):
    return self.text


"""
Inst is a single instruction and is base class for other instructions.
Currently just stores text+comment but over time may grow
"""
class Inst(Item):
  def __init__(self, *args):
    params = args[0:len(args)-1]
    comment = args[len(args)-1]
    assert(isinstance(comment, str))
    formatting = "%s"
    if len(params) > 1:
      formatting += " %s"
    for i in range(0, len(params)-2):
      formatting += ", %s"
    instStr = formatting % (params)
    self.text = self.formatWithComment(instStr, comment)

  def formatWithComment(self, instStr, comment):
    return "%-50s // %s\n" % (instStr, comment)

  def __str__(self):
    return self.text
class WaitCnt (Inst):
  """
  Construct a waitcnt from specified lgkmcnt and vmcnt:
  lgkmcnt, vmcnt:
    if -1 then will not be added to the wait term.
  
  If lgkmcnt=vmcnt= -1 then the waitcnt is a nop and 
  an instruction with a comment is returned.
  """
  def __init__(self,lgkmcnt=-1,vmcnt=-1,comment=""):
    self.lgkmcnt = lgkmcnt
    self.vmcnt   = vmcnt
    self.comment = comment

  def __str__(self):
    waitStr = ""
    if self.lgkmcnt != -1 or self.vmcnt != -1:
      waitStr = "s_waitcnt"
      if self.lgkmcnt != -1:
        waitStr += " lgkmcnt(%u)" % self.lgkmcnt
      if self.vmcnt != -1:
        if self.lgkmcnt != -1:
          waitStr += " &"
        waitStr += " vmcnt(%u)" % self.vmcnt
    else:
      waitStr = "// disabled s_waitcnt"

    return self.formatWithComment(waitStr, self.comment)


# uniq type that can be used in Module.countType
class GlobalReadInst (Inst):
  def __init__(self,*args):
    Inst.__init__(self,*args)

# uniq type that can be used in Module.countType
class LocalWriteInst (Inst):
  def __init__(self,*args):
    Inst.__init__(self,*args)

# uniq type that can be used in Module.countType
class LocalReadInst (Inst):
  def __init__(self,*args):
    Inst.__init__(self,*args)

################################################################################
# Mac Instruction
# can be generic as VALU instruction
# implement later generic
################################################################################
class  MacInst (Inst):
  """
  Construct a mac instruction from specified dataType, aIndex, bIndex, PLR, innerUnroll:

  dataType:
  aIndex:  index value from range (0, kernel["ThreadTile0"])
  bIndex:  index value from range (0, kernel["ThreadTile1"])

  PLR:     valida values 0,1

  usage Module.addCode(Code.MacInst())

  """
  def  __init__(self,kernel,aIdx,bIdx,PLRval,innerUnroll):
       self.endLine = ""
       self.version = globalParameters["CurrentISA"]
       self.kernel  = kernel
       self.aIdx    = aIdx
       self.bIdx    = bIdx
       self.PLR     = PLRval
       self.innerUnroll = innerUnroll

  #def toCodeInst(self,kernel,aIdx,bIdx,PLRval,innerUnroll):
  def __str__(self):
      # half precision
      kStr = ""
      if self.kernel["ProblemType"]["DataType"].isHalf():
        if self.version == (8,0,3):
          for b in range(self.bIdx*2, (self.bIdx+1)*2):
            for a in range(self.aIdx*2, (self.aIdx+1)*2):
              for iui in range(0, self.innerUnroll):
                    # v_mac_f16 or v_fma_f16
                    cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"])
                    aStr = "v[%s+%u]" \
                        % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
                    bStr = "v[%s+%u]" \
                        % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
                    kStr += "v_mac_f16 %s, %s, %s\n" % (cStr, aStr, bStr) # FIXME op_sel
        elif self.version == (9,0,0):
          if self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
            # we treat HighPrecisionAccumulate as expanded packed math
            #b = self.bIdx*2
            #a = self.aIdx*2
            if self.kernel["LocalDotLayout"] > 1 and self.innerUnroll == 2:    # Only supports LocalDotLayout == 2 for now
              cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 0
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I0"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I0"%self.PLR, self.bIdx)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I1"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I0"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 0
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I0"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I1"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 1
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I1"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I1"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
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
              for iui in range(0, self.innerUnroll):
                 cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
                 cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 0
                 aStr = "v[%s+%u]" \
                     % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
                 bStr = "v[%s+%u]" \
                     % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
                 kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] iui=%u%s" % (cStr, aStr, bStr, cStr, cidx, iui, self.endLine)
                 cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
                 cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
                 kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                 cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 0
                 cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
                 kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                 cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 1
                 cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
                 kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                 """
                    ignore this, not quite correct for mixed precision
                    D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                    D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                    C[0] = A[0]*B[0]+D[0]
                    C[1] = A[1]*B[1]+D[1]
                 """
          else:
            #b = self.bIdx*2
            #a = self.aIdx*2
            for iui in range(0, self.innerUnroll):
              cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # /2 b/c of 2 f16's per 32-bit vgpr
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
              kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,0,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
              cStr = "v[%s+%u+%u*%u+%u]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
              """
               D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
               D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
               C[0] = A[0]*B[0]+D[0]
               C[1] = A[1]*B[1]+D[1]
              """
        elif self.version == (9,0,6):
          if self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
             # we treat HighPrecisionAccumulate as expanded packed math
            #b = self.bIdx*2
            #a = self.aIdx*2
            if self.kernel["LocalDotLayout"] > 1 and self.innerUnroll == 2:    # Only supports LocalDotLayout == 2 for now
              cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 0
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I0"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I0"%self.PLR, self.bIdx)
              kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I1"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I0"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
              kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 0
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I0"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I1"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_dot2_f32_f16 %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
              cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 1
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I1"%self.PLR, self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I1"%self.PLR, self.bIdx)
              cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
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
              for iui in range(0, self.innerUnroll):
                cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 0
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
                kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] iui=%u%s" % (cStr, aStr, bStr, cStr, cidx, iui, self.endLine)
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
                aStr = "v[%s+%u]" \
                    % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
                bStr = "v[%s+%u]" \
                    % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
                kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] iui=%u%s" % (cStr, aStr, bStr, cStr, cidx, iui, self.endLine)
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + 1
                cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # *2 b/c of fp32
                kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[1,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 0
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
                kStr += "v_fma_mix_f32 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
                cidx = self.aIdx*2 + self.bIdx*self.kernel["ThreadTile0"]*2 + self.kernel["ThreadTile0"] + 1
                cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
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
            #b = self.bIdx*2
            #a = self.aIdx*2
            for iui in range(0, self.innerUnroll):
              cStr = "v[%s+%u+%u*%u+0]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"]) # /2 b/c of 2 f16's per 32-bit vgpr
              aStr = "v[%s+%u]" \
                  % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
              bStr = "v[%s+%u]" \
                  % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
              kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,0,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)

              cStr = "v[%s+%u+%u*%u+%u]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], self.kernel["ThreadTile0"]/2)
              kStr += "v_pk_fma_f16 %s, %s, %s, %s op_sel:[0,1,0] op_sel_hi:[1,1,1]%s" % (cStr, aStr, bStr, cStr, self.endLine)
              """
              D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
              D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
              C[0] = A[0]*B[0]+D[0]
              C[1] = A[1]*B[1]+D[1]
              """
        else:
          printExit("Half-precision not supported for arch=%u" % self.version )

      # integer i8
      elif self.kernel["ProblemType"]["DataType"].isInt8x4():
        if self.version == (8,0,3):
          kStr += "// int8 not implemented yet for gfx803:"
        elif self.version == (9,0,0):
          kStr += "// int8 not implemented yet for gfx900:"
        elif self.version == (9,0,6):
          for iui in range(0, self.innerUnroll):
            cidx = self.aIdx + self.bIdx*self.kernel["ThreadTile0"] + 0
            cStr = "v[%s+%u+%u*%u]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"])
            aStr = "v[%s+%u]"       % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
            bStr = "v[%s+%u]"       % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
            kStr += "v_dot4_i32_i8  %s, %s, %s, %s op_sel:[0,0] op_sel_hi:[1,1] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)
      # single precision
      elif self.kernel["ProblemType"]["DataType"].isSingle():
        for iui in range(0, self.innerUnroll):
          cStr = "v[%s+%u+%u*%u]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"])
          aStr = "v[%s+%u]" \
              % ("vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
          bStr = "v[%s+%u]" \
              % ("vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
          #if a==0 and b==0:
          #  kStr += dump(aStr)
          kStr += "v_mac_f32 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine)
          ##if macIdx == self.kernel["PerformanceWaitLocation"]:
          ##    kStr += "s_waitcnt lgkmcnt(%u) // extra wait for performance%s" \
          ##        % (self.kernel["PerformanceWaitCount"], self.endLine)
          ##if macIdx == self.kernel["PerformanceSyncLocation"]:
          ##    kStr += "s_barrier // extra barrier for performance%s" \
          ##        % (self.endLine)
          ##macIdx += 1

      # double precision
      elif self.kernel["ProblemType"]["DataType"].isDouble():
        for iui in range(0, self.innerUnroll):
           cStr = "v[%s+(%u+%u*%u)*2:(%s+%u+%u*%u)*2+1]" % ("vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"], "vgprValuC", self.aIdx, self.bIdx, self.kernel["ThreadTile0"])
           aStr = "v[%s+%u*2:%s+%u*2+1]" \
               % ("vgprValuA_X%u_I%u"%(self.PLR,iui) , self.aIdx, "vgprValuA_X%u_I%u"%(self.PLR,iui), self.aIdx)
           bStr = "v[%s+%u*2:%s+%u*2+1]" \
               % ("vgprValuB_X%u_I%u"%(self.PLR,iui) , self.bIdx, "vgprValuB_X%u_I%u"%(self.PLR,iui), self.bIdx)
           kStr += "v_fma_f64 %s, %s, %s, %s%s" % (cStr, aStr, bStr, cStr, self.endLine)

      # other precision
      else:
        printExit("Assembly doesn't support %s" % self.kernel["ProblemType"]["DataType"])

      return self.formatWithComment(kStr, "")
