from abc  import ABC
from abc  import abstractmethod
from copy import deepcopy

class KernelWriterBase(ABC):

  def __init__(self):
    super().__init__()

    self.state = {}

    self.endLine = "\n"
    self.getGroupIdStr = "hc_get_group_id"
    self.getNumGroupsStr = "hc_get_num_groups"
    self.getLocalIdStr = "hc_get_workitem_id"
    self.getGlobalIdStr = "hc_get_workitem_absolute_id"
    self.sharedDeclStr = "__shared__ "
    self.sharedPtrStr = ""
    self.globalPtrStr = ""
    self.syncStr = "__syncthreads();"
    self.fenceStr = self.syncStr
    self.macFStr = "fmaf"
    self.macDStr = "fma"
    self.int64Str = "int64_t"
    self.uint64Str = "uint64_t"
    self.vectorComponents = ["x", "y", "z", "w"]
    self.atomicCasStr = "atomicCAS"
    self.volatileStr = ""
    self.deviceFunctionStr = "__device__ "


  def extractIndices(self, extractFrom, varPrefix, indices):
    kStr = ""
    for (i, index) in enumerate(indices):
      kStr += "  unsigned int " + varPrefix + self.indexChars[index] + " = ( " + extractFrom
      for j in reversed(list(range(i+1, len(indices)))):
        index2 = indices[j]
        kStr += " / size" + self.indexChars[index2]
      kStr += ")"

      if len(indices) > 1:
        kStr += " % size" + self.indexChars[index]
      kStr += ";" + self.endLine

    return kStr


  @abstractmethod
  def getKernelName(self):
    pass


  @abstractmethod
  def getHeaderFileString(self):
    pass


  @abstractmethod
  def getSourceFileString(self):
    pass


  ##########################
  # make class look like dict
  def keys(self):
    return list(self.state.keys())

  def __len__(self):
    return len(self.state)

  def __iter__(self):
    return iter(self.state)

  def __getitem__(self, key):
    return self.state[key]

  def __setitem__(self, key, value):
    self.state[key] = value

  def __str__(self):
    return self.getKernelName()

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return deepcopy(self.state)

  def __hash__(self):
    return hash(str(self))

  def __eq__(self, other):
    return isinstance(other, KernelWriterBase) and str(self) == str(other)

  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

