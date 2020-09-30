from Tensile.Utilities.merge import mergeLogic, removeUnusedKernels, cmpHelper
import yaml, io, pytest

logicPrefix = r"""
- DummyVersionRequirement
- DummyLanguage
- DummyScheduleName
- DummyDevice
- DummyProblemType
"""

baseLogic=logicPrefix + r"""
-
  - SolutionIndex: 0
    SolutionNameMin: InUseForSize256
  - SolutionIndex: 1
    SolutionNameMin: InUseForSize128or64
  - SolutionIndex: 2
    SolutionNameMin: UnusedSolution
- DummyIndexAssignment
-
  - - [256, 256, 1, 256]
    - [0, 111.1]
  - - [128, 128, 1, 128]
    - [1, 99.9]
  - - [64, 64, 1, 64]
    - [1, 88.8]
"""

incLogic=logicPrefix + r"""
-
  - SolutionIndex: 39
    SolutionNameMin: InUseForSize256or1024xxx
  - SolutionIndex: 1
    SolutionNameMin: InUseForSize128xxx
- DummyIndexAssignment
-
  - - [128, 128, 1, 128]
    - [1, 999.9]
  - - [256, 256, 1, 256]
    - [39, 999.9]
  - - [1024, 1024, 1, 1024]
    - [39, 999.9]
"""

notUnique=logicPrefix+r"""
-
  - SolutionIndex: 0
    SolutionNameMin: Kernel0
  - SolutionIndex: 1
    SolutionNameMin: Kernel0
"""

unique=logicPrefix+r"""
-
  - SolutionIndex: 0
    SolutionNameMin: Kernel0
  - SolutionIndex: 1
    SolutionNameMin: Kernel1
"""

def checkUniqueSolution(solutionPool):
  uniq = set()
  # note: any([False or None, True or None]) -> True
  return not any(frozenset(cmpHelper(i).items()) in uniq or uniq.add(frozenset(cmpHelper(i).items()))
                 for i in solutionPool)

@pytest.mark.parametrize("input,expectedNumKernelRemoved", [(baseLogic, 1),
                                                            (incLogic, 0)])
def test_removeUnusedKernels(input, expectedNumKernelRemoved):
  data = yaml.load(input, yaml.SafeLoader)
  dataFiltered, numKernelRemoved = removeUnusedKernels(data)

  # test if number of solution removed is correct
  assert numKernelRemoved == expectedNumKernelRemoved

  # test if solution is re-indexed
  for index, s in enumerate(dataFiltered[5]):
    assert index == s["SolutionIndex"]

  # check if size-kernel mapping is not compromised
  solutionMapBefore = {}
  solutionMapAfter = {}
  for [size, [index, _]] in data[7]:
    solutionMapBefore[tuple(size)] = [s["SolutionNameMin"] for s in data[5] if s["SolutionIndex"]==index][0]
  for [size, [index, _]] in dataFiltered[7]:
    solutionMapAfter[tuple(size)] = [s["SolutionNameMin"] for s in dataFiltered[5] if s["SolutionIndex"]==index][0]

  assert solutionMapBefore == solutionMapAfter

  # print final yaml for visual inspection
  stream = io.StringIO("")
  yaml.safe_dump(dataFiltered, stream, default_flow_style=None)
  print(stream.getvalue())

def test_mergeLogic():
  baseData = yaml.load(baseLogic, yaml.SafeLoader)
  incData = yaml.load(incLogic, yaml.SafeLoader)

  mergedData, *stats = mergeLogic(baseData, incData, False)
  assert stats == [1,2,2] # 1 sizes added, 2 kernels added, 2 kernels removed/replaced

  # print final yaml for visual inspection
  stream = io.StringIO("")
  yaml.safe_dump(mergedData, stream, default_flow_style=None)
  print(stream.getvalue())

  # check if solution matches expected. assumes SolutionNameMin is unique
  # (which is satisfied by the test data given here)
  solutionMap = {} # size -> solutionName
  for [size, [index, _]] in mergedData[7]:
    solutionMap[tuple(size)] = [s["SolutionNameMin"] for s in mergedData[5] if s["SolutionIndex"]==index][0]

  sizeList = [tuple([256, 256, 1, 256]), tuple([128, 128, 1, 128]), tuple([64, 64, 1, 64])]
  expectedSolution = [ "InUseForSize256or1024xxx", "InUseForSize128xxx", "InUseForSize128or64" ]
  for t in zip(sizeList, expectedSolution):
    size = t[0]
    expectedSolution = t[1]
    assert solutionMap[size] == expectedSolution

  # check if each solution in merged data is unique
  checkUniqueSolution(mergedData[5])

@pytest.mark.parametrize("input,expected", [(unique, True), (notUnique, False)])
def test_checkUniqueSolution(input, expected):
  data = yaml.load(input, yaml.SafeLoader)
  assert checkUniqueSolution(data[5]) == expected

if __name__ == "__main__":
    test_mergeLogic()
    test_checkUniqueSolution(unique, True)