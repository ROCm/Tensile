import Structs
import FileReader
import argparse

################################################################################
# Make Index Assignments
################################################################################
def makeIndexAssignments(solution, problem):
  numIndicesC = solution.operation.numIndicesFree \
      + solution.operation.numIndicesBatch
  numIndicesA = len(solution.operation.indexAssignmentsA)
  numIndicesB = len(solution.operation.indexAssignmentsB)
  # C indices in order of descending stride
  # sort free indices, then append after batched indices
  indicesUnsortedC = []
  for i in range(0,numIndicesC):
    indexIsBatched = False
    if i in solution.operation.indexAssignmentsA:
      if i in solution.operation.indexAssignmentsB:
        indexIsBatched = True
    if indexIsBatched:
      solution.indexOrderC.append(i)
    else:
      indicesUnsortedC.append( [problem.tensorC.dimensions[i].stride, i] )
  indicesSortedC = sorted( indicesUnsortedC, \
      key = lambda x: int(x[0]), reverse=True )
  for i in range(0,len(indicesSortedC)):
    solution.indexOrderC.append( indicesSortedC[i][1] )

  # summation indices in order of descending A-stride + B-stride
  indicesSummationUnsorted = []
  for i in range(0,solution.operation.numIndicesSummation):
    sumIndex = i + numIndicesC
    assignmentA = -1
    for j in range(0,numIndicesA):
      if solution.operation.indexAssignmentsA[j] == sumIndex:
        assignmentA = j
    assignmentB = -1
    for j in range(0,numIndicesB):
      if solution.operation.indexAssignmentsB[j] == sumIndex:
        assignmentB = j
    indicesSummationUnsorted.append( \
        [problem.tensorA.dimensions[assignmentA].stride \
        + problem.tensorB.dimensions[assignmentB].stride, i] )
  indicesSummationSorted = sorted( indicesSummationUnsorted, \
      key = lambda x: int(x[0]), reverse=True )
  for i in range(0,len(indicesSummationSorted)):
    solution.indexOrderSummation.append( indicesSummationSorted[i][1] )

  # last index will belong to A or B, find out which
  solution.indexAssignmentTileDim1 = solution.indexOrderC[ numIndicesC - 1 ]
  solution.tensorAssignedDim0 = 0
  if solution.indexAssignmentTileDim1 in solution.operation.indexAssignmentsB:
    solution.tensorAssignedDim0 = 1

  # 2nd to last index must belong to other, make it so
  solution.indexAssignmentTileDim0 = solution.indexOrderC[ \
      numIndicesC - 2 ]
  solution.tensorAssignedDim1 = 1
  if solution.indexAssignmentTileDim0 in solution.operation.indexAssignmentsA:
    solution.tensorAssignedDim1 = 0

  if solution.tensorAssignedDim0 == solution.tensorAssignedDim1:
    print "SolutionCandidates::makeIndexAssignments() - ERROR TileDim0,1 same tensor"

################################################################################
# SolutionCandidates
################################################################################
class SolutionCandidates:

  # Tuneable Performance Parameters
  # skinnyness: dim1 / dim0 <= ratio[not skinny, is skinny]
  skinnyRatioWorkGroup = [ 1, 16]
  skinnyRatioMicroTile = [ 1, 2]
  skinnyRatioMacroTile = [ skinnyRatioWorkGroup[0]*skinnyRatioMicroTile[0], \
      skinnyRatioWorkGroup[1]*skinnyRatioMicroTile[1] ]
  universeUnroll = [ \
      [  1 ], [ 16, 1 ], [  8, 1 ], \
      [  2 ], [ 16, 2 ], [  8, 2 ], \
      [  4 ], [ 16, 4 ], [  8, 4 ], \
      [  8 ], [ 16, 8 ], \
      [ 16 ] ]
  universeWorkGroupDim = [ \
       [1,64],  [2,32], [4,16],  [8,8],  [16,4], [32,2],  [64,1], \
      [1,128],  [2,64], [4,32], [8,16],  [16,8], [32,4],  [64,2], [128,1], \
      [1,192],  [2,96], [3,64], [4,48],  [6,32], [8,24], [12,16], [16, 12], \
                [24,8], [32,6], [48,4],  [64,3], [96,2], [192,1], \
      [1,256], [2,128], [4,64], [8,32], [16,16], [32,8],  [64,4],  [128,2], \
               [256,1] ]



    # tile assignment - last two free indices?

  ##############################################################################
  # getSolutionCandidatesForProblem
  ##############################################################################
  def getSolutionCandidatesForProblem( self, problem ):

    numIndicesC = len(problem.tensorC.dimensions)
    numIndicesA = len(problem.tensorA.dimensions)
    numIndicesB = len(problem.tensorB.dimensions)

    # create solution object
    solution = Structs.Solution()
    solutionCandidates = []

    # Solution Correctness Parameters
    solution.operation = problem.operation
    solution.dataTypeC = problem.tensorC.dataType
    solution.dataTypeA = problem.tensorA.dataType
    solution.dataTypeB = problem.tensorB.dataType
    solution.numIndicesFree = problem.operation.numIndicesFree
    solution.numIndicesBatch = problem.operation.numIndicesBatch
    solution.numIndicesSummation = problem.operation.numIndicesSummation

    # Index Assignments
    solution.indexOrderC = []
    solution.indexOrderSummation = []
    makeIndexAssignments( solution, problem )
    solution.indexAssignmentTileDim0 = solution.indexOrderC[ \
        numIndicesC - 2 ]
    solution.indexAssignmentTileDim1 = solution.indexOrderC[ \
        numIndicesC - 1 ]

    # Problem Characteristics affecting performance
    solution.problemSizeDim0 = problem.tensorC.dimensions[ \
        solution.indexAssignmentTileDim0].size
    problemSkinnyDim0 = 0 # false
    if solution.problemSizeDim0 < 96:
      problemSkinnyDim0 = 1
    solution.problemSizeDim1 = problem.tensorC.dimensions[ \
        solution.indexAssignmentTileDim1].size
    problemSkinnyDim1 = 0
    if solution.problemSizeDim1 < 96:
      problemSkinnyDim1 = 1
    solution.indexUnroll = solution.indexOrderSummation[solution.numIndicesSummation-1]
    solution.problemSizeUnroll = -1
    for i in range(len(solution.operation.indexAssignmentsA)):
      if solution.indexUnroll == solution.operation.indexAssignmentsA[i]:
        problemSizeUnroll = problem.tensorA.dimensions[i].size
        break
    solution.tensorStrideDim0 = -1
    for i in range(len(solution.operation.indexAssignmentsA)):
      if solution.indexAssignmentTileDim0 == solution.operation.indexAssignmentsA[i]:
        solution.tensorStrideDim0 = problem.tensorA.dimensions[i].stride
        break
    solution.tensorStrideDim1 = -1
    for i in range(len(solution.operation.indexAssignmentsB)):
      if solution.indexAssignmentTileDim1 == solution.operation.indexAssignmentsB[i]:
        solution.tensorStrideDim1 = problem.tensorA.dimensions[i].stride
        break

    # Solution Universe
    for unroll in self.universeUnroll:
      # if last unroll is multiple of last/unrolled summation
      print problemSizeUnroll, unroll
      if problemSizeUnroll % unroll[len(unroll)-1] > 0:
        continue
      for workGroupDim in self.universeWorkGroupDim:
        # work-group not too skinny
        if workGroupDim[1]/workGroupDim[0] \
            > self.skinnyRatioWorkGroup[problemSkinnyDim0]:
          continue
        if workGroupDim[0]/workGroupDim[1] \
            > self.skinnyRatioWorkGroup[problemSkinnyDim1]:
          continue
        # all micro-tile dimensions
        for microTileDim0 in range(1, 16):
          for microTileDim1 in range(1, 16):
            # micro-tile not too skinny
            if microTileDim1/microTileDim0 \
                > self.skinnyRatioMicroTile[problemSkinnyDim0]:
              continue
            if microTileDim0/microTileDim1 \
                > self.skinnyRatioMicroTile[problemSkinnyDim1]:
              continue
            # macro-tile not too skinny
            macroTileDim0 = workGroupDim[0] * microTileDim0
            macroTileDim1 = workGroupDim[1] * microTileDim1
            if macroTileDim1/macroTileDim0 \
                > self.skinnyRatioMacroTile[problemSkinnyDim0]:
              continue
            if macroTileDim0/macroTileDim1 \
                > self.skinnyRatioMacroTile[problemSkinnyDim1]:
              continue
            # macro-tile not too large
            numWorkItems = workGroupDim[0] * workGroupDim[1]
            numRegisters = numWorkItems * ( microTileDim0 * microTileDim1 \
                * solution.dataTypeC.numRegistersPerElement() \
                + microTileDim0 * solution.dataTypeA.numRegistersPerElement() \
                + microTileDim1 * solution.dataTypeB.numRegistersPerElement() )
            maxRegisters = 16*16*( 4*4*4 + 4*4 + 4*4 )
            if numRegisters > maxRegisters:
              continue

            # tile exactly matches
            if solution.problemSizeDim0 % macroTileDim0 == 0 \
                and solution.problemSizeDim1 % macroTileDim1 == 0:
              solution.kernelGrid = [ 1, 1 ]
              tensorStrideDim1 = problem.tensorC.dimensions[ \
                  solution.indexAssignmentTileDim1].size
              if solution.tensorStrideDim0 % 1024 == 0:
                solution.kernelGrid[0] = solution.problemSizeDim0 / 1024;
              if solution.tensorStrideDim1 % 1024 == 0:
                solution.kernelGrid[1] = solution.problemSizeDim1 / 1024;

            # dim0,1 are both edges
            elif solution.problemSizeDim0 % macroTileDim0 != 0 \
                and solution.problemSizeDim1 % macroTileDim1 != 0:
              solution.kernelGrid = [ 2, 2 ]

            # dim0 is edge
            elif solution.problemSizeDim0 % macroTileDim0 != 0 \
                and solution.problemSizeDim1 % macroTileDim1 == 0:
              solution.kernelGrid = [ 2, 1 ]

            # dim1 is edge
            elif solution.problemSizeDim0 % macroTileDim0 == 0 \
                and solution.problemSizeDim1 % macroTileDim1 != 0:
              solution.kernelGrid = [ 1, 2 ]

            # remove prior kernels; this will be new candidate
            solution.kernels = []

            # add kernels in grid
            for dim0 in range(0,solution.kernelGrid[0]):
              for dim1 in range(1,solution.kernelGrid[1]):
                kernel = Structs.Kernel()
                kernel.unrolls = unroll
                kernel.edge[0] = solution.problemSizeDim0 % macroTileDim0 != 0 \
                    and dim0==solution.kernelGrid[0]-1
                kernel.edge[1] = solution.problemSizeDim1 % macroTileDim1 != 0 \
                    and dim1==solution.kernelGrid[1]-1
                kernel.tile.workGroupDim0 = workGroupDim[0]
                kernel.tile.workGroupDim1 = workGroupDim[1]
                kernel.tile.microTileDim0 = microTileDim0
                kernel.tile.microTileDim1 = microTileDim1
                kernel.tile.macroTileDim0 = macroTileDim0
                kernel.tile.macroTileDim1 = macroTileDim1
                solution.kernels.append( kernel )

            # include this solution as candidate
            solutionCandidates.append( solution )
    return solutionCandidates

################################################################################
# Main
################################################################################
if __name__ == "__main__":

  # arguments
  ap = argparse.ArgumentParser(description="FileReader")
  ap.add_argument("--input-file", dest="inputFiles", action="append" )
  args = ap.parse_args()

  # parse xml
  for inputFile in args.inputFiles:
    problemSet = set()
    FileReader.getProblemsFromXML( inputFile, problemSet )
  solutionCandidates = SolutionCandidates()

  for problem in problemSet:
    solutionCandidatesForProblem = \
        solutionCandidates.getSolutionCandidatesForProblem( problem )
    print problem
    print solutionCandidatesForProblem


