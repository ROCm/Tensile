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
    print "%sModule %s:"% (indent,self.name)
    for i in self.itemList:
      if isinstance(i, Module):
        i.prettyPrint(indent+"  ")
      elif isinstance(i, str):
        print indent, '"', str(i).strip('\n'), '"'
      else: # Inst
          print indent, "%s: [ %s ]" % \
              (i.__class__.__name__, str(i).strip('\n'))

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

# Construct a waitcnt from specified lgkmcnt and vmcnt:
# lgkmcnt, vmcnt:
#   if -1 then will not be added to the wait term.
#
# If lgkmcnt=vmcnt= -1 then the waitcnt is a nop and 
# an instruction with a comment is returned.
class WaitCnt (Inst):
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
