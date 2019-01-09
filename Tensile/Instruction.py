
class Module:
  def __init__(self):
    self.instList = []
    self.itemList = []

  def __str__(self):
    return "\n".join([str(x) for x in self.instList])

  def toStr(self):
    return str(self)

  def append(self, inst):
    self.instList.append(inst)
    self.itemList.append(inst)

  def comment(self, comment):
    self.itemList.append(comment)

  def instStr(self, *args):
    params = args[0:len(args)-1]
    comment = args[len(args)-1]
    formatting = "%s"
    if len(params) > 1:
      formatting += " %s"
    for i in range(0, len(params)-2):
      formatting += ", %s"
    instStr = formatting % (params)
    self.append("%-50s // %s" % (instStr, comment))


class Inst:
  def __init__(self, *args):
    params = args[0:len(args)-1]
    comment = args[len(args)-1]
    formatting = "%s"
    if len(params) > 1:
      formatting += " %s"
    for i in range(0, len(params)-2):
      formatting += ", %s"
    instStr = formatting % (params)
    self.text = "%-50s // %s" % (instStr, comment)


  def __str__(self):
    return self.text + '\n'

  def toStr(self):
    return str(self)

