import Structs
import FileReader
import argparse

################################################################################
# Make Index Assignments
################################################################################
def makeIndexAssignments(solution, problem):
  numIndicesC = problem.operation.numIndicesFree \
      + problem.operation.numIndicesBatch
  numIndicesA = len(problem.operation.indexAssignmentsA)
  numIndicesB = len(problem.operation.indexAssignmentsB)
  # C indices in order of descending stride
  # sort free indices, then append after batched indices
  indicesUnsortedC = []
  for i in range(0,numIndicesC):
    indexIsBatched = False
    if i in problem.operation.indexAssignmentsA:
      if i in problem.operation.indexAssignmentsB:
        indexIsBatched = True
    if indexIsBatched:
      kernel.indexOrderC.append(i)
    else:
      indicesUnsortedC.append( [problem.tensorC.dimensions[i].stride, i] )
  indicesSortedC = sorted( indicesUnsortedC, \
      key = lambda x: int(x[0]), reverse=True )
  for i in range(0,len(indicesSortedC)):
    kernel.indexOrderC.append( indicesSortedC[i][1] )

  # summation indices in order of descending A-stride + B-stride
  indicesSummationUnsorted = []
  for i in range(0,problem.operation.numIndicesSummation):
    sumIndex = i + numIndicesC
    assignmentA = -1
    for j in range(0,numIndicesA):
      if problem.operation.indexAssignmentsA[j] == sumIndex:
        assignmentA = j
    assignmentB = -1
    for j in range(0,numIndicesB):
      if problem.operation.indexAssignmentsB[j] == sumIndex:
        assignmentB = j
    indicesSummationUnsorted.append( \
        [problem.tensorA.dimensions[assignmentA].stride \
        + problem.tensorB.dimensions[assignmentB].stride, i] )
  indicesSummationSorted = sorted( indicesSummationUnsorted, \
      key = lambda x: int(x[0]), reverse=True )
  for i in range(0,len(indicesSummationSorted)):
    kernel.indexOrderSummation.append( indicesSummationSorted[i][1] )

  # last index will belong to A or B, find out which
  kernel.indexAssignmentDim1 = kernel.indexOrderC[ numIndicesC - 1 ]
  solution.tensorAssignedDim0 = 0
  if kernel.indexAssignmentDim1 in problem.operation.indexAssignmentsB:
    solution.tensorAssignedDim0 = 1

  # 2nd to last index must belong to other, make it so
  kernel.indexAssignmentDim0 = kernel.indexOrderC[ \
      numIndicesC - 2 ]
  solution.tensorAssignedDim1 = 1
  if kernel.indexAssignmentDim0 in problem.operation.indexAssignmentsA:
    solution.tensorAssignedDim1 = 0

  if solution.tensorAssignedDim0 == solution.tensorAssignedDim1:
    print "SolutionCandidates::makeIndexAssignments() - ERROR TileDim0,1 same tensor"

################################################################################
# SolutionCandidates
################################################################################
class SolutionCandidates:

  # Tuneable Performance Parameters
  # skinnyness: dim1 / dim0 <= ratio[not skinny, is skinny]
  # increasing these parameters will test a wider variety of tiles
  skinnyRatioWorkGroup = [ 1, 16]
  skinnyRatioMicroTile = [ 1, 2]
  skinnyRatioMacroTile = [ skinnyRatioWorkGroup[0]*skinnyRatioMicroTile[0], \
      skinnyRatioWorkGroup[1]*skinnyRatioMicroTile[1] ]
  maxMicroTileSize = 16
  universeUnroll = { \
       1: [ [  1 ], [ 16, 1 ], [  8, 1 ] ], \
       2: [ [  2 ], [ 16, 2 ], [  8, 2 ] ], \
       4: [ [  4 ], [ 16, 4 ], [  8, 4 ] ], \
       8: [ [  8 ], [ 16, 8 ] ], \
      16: [ [ 16 ], [ 8 ] ] \
      }
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
    kernel = Structs.Kernel()
    solution = Structs.Solution()
    solutionCandidates = []

    # Solution Correctness Parameters
    kernel.operation = problem.operation
    kernel.dataTypeC = problem.tensorC.dataType
    kernel.dataTypeA = problem.tensorA.dataType
    kernel.dataTypeB = problem.tensorB.dataType

    # Index Assignments
    kernel.indexOrderC = []
    kernel.indexOrderSummation = []
    makeIndexAssignments( solution, problem )
    kernel.indexAssignmentDim0 = kernel.indexOrderC[ \
        numIndicesC - 2 ]
    kernel.indexAssignmentDim1 = kernel.indexOrderC[ \
        numIndicesC - 1 ]

    # Problem Characteristics affecting performance
    problemSizeDim0 = problem.tensorC.dimensions[ \
        kernel.indexAssignmentDim0].size
    problemSkinnyDim0 = 0 # false
    if problemSizeDim0 < 96:
      problemSkinnyDim0 = 1
    problemSizeDim1 = problem.tensorC.dimensions[ \
        kernel.indexAssignmentDim1].size
    problemSkinnyDim1 = 0
    if problemSizeDim1 < 96:
      problemSkinnyDim1 = 1
    kernel.indexUnroll = kernel.indexOrderSummation[ \
        problem.operation.numIndicesSummation-1]
    problemSizeUnroll = -1
    for i in range(len(problem.operation.indexAssignmentsA)):
      if kernel.indexUnroll == problem.operation.indexAssignmentsA[i]:
        problemSizeUnroll = problem.tensorA.dimensions[i].size
        break
    tensorStrideDim0 = -1
    for i in range(len(problem.operation.indexAssignmentsA)):
      if kernel.indexAssignmentDim0 == problem.operation.indexAssignmentsA[i]:
        tensorStrideDim0 = problem.tensorA.dimensions[i].stride
        break
    tensorStrideDim1 = -1
    for i in range(len(problem.operation.indexAssignmentsB)):
      if kernel.indexAssignmentDim1 == problem.operation.indexAssignmentsB[i]:
        tensorStrideDim1 = problem.tensorA.dimensions[i].stride
        break

    # only try the highest unroll level
    selectedUnroll = -1
    for unroll in [16, 8, 4, 2, 1]:
      if problemSizeUnroll % unroll == 0:
        selectedUnroll = unroll
        break

    # Solution Universe
    for unroll in self.universeUnroll[selectedUnroll]:
      # if last unroll is multiple of last/unrolled summation
      if problemSizeUnroll % unroll[len(unroll)-1] > 0:
        continue
      for workGroupDim in self.universeWorkGroupDim:
        # work-group not too skinny
        if float(workGroupDim[1])/workGroupDim[0] \
            > self.skinnyRatioWorkGroup[problemSkinnyDim0]:
          continue
        if float(workGroupDim[0])/workGroupDim[1] \
            > self.skinnyRatioWorkGroup[problemSkinnyDim1]:
          continue
        # all micro-tile dimensions
        for microTileDim0 in range(1, self.maxMicroTileSize):
          for microTileDim1 in range(1, self.maxMicroTileSize):
            # micro-tile not too skinny
            if float(microTileDim1)/microTileDim0 \
                > self.skinnyRatioMicroTile[problemSkinnyDim0]:
              continue
            if float(microTileDim0)/microTileDim1 \
                > self.skinnyRatioMicroTile[problemSkinnyDim1]:
              continue
            # macro-tile not too skinny
            macroTileDim0 = workGroupDim[0] * microTileDim0
            macroTileDim1 = workGroupDim[1] * microTileDim1
            if float(macroTileDim1)/macroTileDim0 \
                > self.skinnyRatioMacroTile[problemSkinnyDim0]:
              continue
            if float(macroTileDim0)/macroTileDim1 \
                > self.skinnyRatioMacroTile[problemSkinnyDim1]:
              continue
            # macro-tile not too large
            numWorkItems = workGroupDim[0] * workGroupDim[1]
            numRegisters = numWorkItems * ( microTileDim0 * microTileDim1 \
                * kernel.dataTypeC.numRegistersPerElement() \
                + microTileDim0 * kernel.dataTypeA.numRegistersPerElement() \
                + microTileDim1 * kernel.dataTypeB.numRegistersPerElement() )
            maxRegisters = 16*16*( 4*4*4 + 4*4 + 4*4 )
            if numRegisters > maxRegisters:
              continue

            # tile exactly matches
            if problemSizeDim0 % macroTileDim0 == 0 \
                and problemSizeDim1 % macroTileDim1 == 0:
              solution.kernelGrid = [ 1, 1 ]
              tensorStrideDim1 = problem.tensorC.dimensions[ \
                  kernel.indexAssignmentDim1].size
              if tensorStrideDim0 % 1024 == 0:
                solution.kernelGrid[0] = problemSizeDim0 / 1024;
              if tensorStrideDim1 % 1024 == 0:
                solution.kernelGrid[1] = problemSizeDim1 / 1024;

            # dim0,1 are both edges
            elif problemSizeDim0 % macroTileDim0 != 0 \
                and problemSizeDim1 % macroTileDim1 != 0:
              solution.kernelGrid = [ 2, 2 ]

            # dim0 is edge
            elif problemSizeDim0 % macroTileDim0 != 0 \
                and problemSizeDim1 % macroTileDim1 == 0:
              solution.kernelGrid = [ 2, 1 ]

            # dim1 is edge
            elif problemSizeDim0 % macroTileDim0 == 0 \
                and problemSizeDim1 % macroTileDim1 != 0:
              solution.kernelGrid = [ 1, 2 ]

            """print str(workGroupDim[0]) + "x" + str(workGroupDim[1]) + "; " \
                + str(microTileDim0) + "x" + str(microTileDim1) + "; " \
                + str(unroll) + "; " + str(numRegisters) + "/" \
                + str(maxRegisters)"""

            # remove prior kernels; this will be new candidate
            solution.kernels = []

            # add kernels in grid
            for dim0 in range(0,solution.kernelGrid[0]):
              for dim1 in range(1,solution.kernelGrid[1]):
                kernel.unrolls = unroll
                kernel.tile.workGroupDim0 = workGroupDim[0]
                kernel.tile.workGroupDim1 = workGroupDim[1]
                kernel.tile.microTileDim0 = microTileDim0
                kernel.tile.microTileDim1 = microTileDim1
                kernel.tile.macroTileDim0 = macroTileDim0
                kernel.tile.macroTileDim1 = macroTileDim1
                if problemSizeDim0 % macroTileDim0 != 0 \
                    and dim0==solution.kernelGrid[0]-1:
                  kernel.tile.macroTileDim0 = 1
                if problemSizeDim1 % macroTileDim1 != 0 \
                    and dim1==solution.kernelGrid[1]-1;
                  kernel.tile.macroTileDim1
                solution.kernels.append( kernel.copy() )

            # include this solution as candidate
            solutionCandidates.append( solution.copy() )
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

  """print "numUnrolls = " + str(len(SolutionCandidates.universeUnroll))
  print "numWorkGroups = " + str(len(SolutionCandidates.universeWorkGroupDim))
  print "numMicroTiles = " + str(SolutionCandidates.maxMicroTileSize \
      * SolutionCandidates.maxMicroTileSize)"""

  solutionCandidates = SolutionCandidates()

  for problem in problemSet:
    solutionCandidatesForProblem = \
        solutionCandidates.getSolutionCandidatesForProblem( problem )
    print "\n"
    print problem
    print "\n\n"
    print len(solutionCandidatesForProblem)
    print solutionCandidatesForProblem
    break


