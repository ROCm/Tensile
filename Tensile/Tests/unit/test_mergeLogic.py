from Tensile.Utilities.merge import cmpHelper, fixSizeInconsistencies, removeUnusedKernels, mergeLogic
import yaml, pytest

logicPrefix = r"""
- DummyVersionRequirement
- DummyLanguage
- DummyScheduleName
- DummyDevice
- DummyProblemType
"""

baseLogic=logicPrefix + r"""
-
  - SolutionIndex: 2
    SolutionNameMin: InUseForSize256
  - SolutionIndex: 1
    SolutionNameMin: UnusedSolution
  - SolutionIndex: 0
    SolutionNameMin: InUseForSize128or64
- DummyIndexAssignment
-
  - - [256, 256, 1, 256]
    - [2, 111.1]
  - - [128, 128, 1, 128]
    - [0, 99.9]
  - - [64, 64, 1, 64]
    - [0, 88.8]
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

notUniqueSolution=logicPrefix+r"""
-
  - SolutionIndex: 0
    SolutionNameMin: Kernel0
  - SolutionIndex: 1
    SolutionNameMin: Kernel0
"""

uniqueSolution=logicPrefix+r"""
-
  - SolutionIndex: 0
    SolutionNameMin: Kernel0
  - SolutionIndex: 1
    SolutionNameMin: Kernel1
"""

notTrimmedSize=r"""
-
  - - [128, 128, 1, 128, 128, 128, 128, 128]
    - [1, 999.9]
  - - [1024, 1024, 1, 1024]
    - [42, 0.0]
  - - [128, 128, 1, 128]
    - [39, 0.0]
  - - [1024, 1024, 1, 1024, 1024, 1024, 1024, 1024]
    - [39, 999.9]
  - - [1024, 1024, 1, 1024, 1044, 1044, 1044, 1044]
    - [39, 999.9]
"""

trimmedSize=r"""
-
  - - [128, 128, 1, 128]
    - [1, 999.9]
  - - [512, 512, 1, 512]
    - [39, 999.9]
  - - [1024, 1024, 1, 1024]
    - [42, 999.9]
"""

def checkUniqueSolution(solutionPool):
  uniq = set()
  # note: any([False or None, True or None]) -> True
  return not any(frozenset(cmpHelper(i).items()) in uniq or uniq.add(frozenset(cmpHelper(i).items()))
                 for i in solutionPool)

@pytest.mark.parametrize("sizes, expected", [
  (   trimmedSize, [[128,128,1,128],[512,512,1,512],[1024,1024,1,1024]]),
  (notTrimmedSize, [[1024,1024,1,1024],[128,128,1,128]])
  ])
def test_fixSizeInconsistencies(sizes, expected):
  data = yaml.load(sizes, yaml.SafeLoader)
  data_ = fixSizeInconsistencies(data[0], "dummy")

  for [size, [_,_]], expected_ in zip(data_[0], expected):
    assert size == expected_
  # print final yaml for visual inspection
  # stream = io.StringIO("")
  # yaml.safe_dump(data_, stream, default_flow_style=None)
  # print(stream.getvalue())

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
  # stream = io.StringIO("")
  # yaml.safe_dump(dataFiltered, stream, default_flow_style=None)
  # print(stream.getvalue())

@pytest.mark.parametrize("baseLogic, incLogic, expectedStats, expectedSizes, expectedSolutions", [
# test case #1: merge incLogic into baseLogic
  (baseLogic, incLogic, [1,2,2], # 1 sizes added, 2 kernels added, 2 kernels removed/replaced
  [(1024,1024,1,1024), (256,256,1,256), (128,128,1,128), (64,64,1,64)],
  ["InUseForSize256or1024xxx", "InUseForSize256or1024xxx", "InUseForSize128xxx", "InUseForSize128or64"]),
# test case #2: merge baseLogic into itself
  (baseLogic, baseLogic, [0,0,1], # 0 sizes added, 0 kernels added, 1 kernel removed because it's unused
  [(256,256,1,256), (128,128,1,128), (64,64,1,64)],
  ["InUseForSize256", "InUseForSize128or64", "InUseForSize128or64"]),
])
def test_mergeLogic(baseLogic, incLogic, expectedStats, expectedSizes, expectedSolutions):
  baseData = yaml.load(baseLogic, yaml.SafeLoader)
  incData = yaml.load(incLogic, yaml.SafeLoader)

  mergedData, *stats = mergeLogic(baseData, incData, False)

  # check if stats are as expected
  assert stats == expectedStats

  # print final yaml for visual inspection
  # stream = io.StringIO("")
  # yaml.safe_dump(mergedData, stream, default_flow_style=None)
  # print(stream.getvalue())

  # check if solution matches expected. assumes SolutionNameMin is uniqueSolution
  # (which is satisfied by the test data given here)
  solutionMap = {} # size -> solutionName
  for size, [index, _] in mergedData[7]:
    solutionMap[tuple(size)] = [s["SolutionNameMin"] for s in mergedData[5] if s["SolutionIndex"]==index][0]

  for size, expected in zip(expectedSizes, expectedSolutions):
    assert solutionMap[size] == expected

  # check if each solution in merged data is uniqueSolution
  assert checkUniqueSolution(mergedData[5])

@pytest.mark.parametrize("input,expected", [(uniqueSolution, True), (notUniqueSolution, False)])
def test_checkUniqueSolution(input, expected):
  data = yaml.load(input, yaml.SafeLoader)
  assert checkUniqueSolution(data[5]) == expected

if __name__ == "__main__":
    # test_mergeLogic(baseLogic, incLogic, [1,2,2], [(1024, 1024, 1, 1024), (256, 256, 1, 256), (128, 128, 1, 128), (64, 64, 1, 64)], [ "InUseForSize256or1024xxx", "InUseForSize256or1024xxx", "InUseForSize128xxx", "InUseForSize128or64"])
    # test_checkUniqueSolution(uniqueSolution, True)
    # test_fixSizeInconsistencies(trimmedSize, [[128,128,1,128],[512,512,1,512],[1024,1024,1,1024]])
    test_fixSizeInconsistencies(notTrimmedSize, [[1024,1024,1,1024],[128,128,1,128]])
