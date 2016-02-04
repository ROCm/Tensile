import copy
import Structs
import FileReader
import argparse

################################################################################
# Make Index Assignments
################################################################################
def makeIndexAssignments(kernel, problem):
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
  kernel.tensorAssignedDim0 = 0
  if kernel.indexAssignmentDim1 in problem.operation.indexAssignmentsB:
    kernel.tensorAssignedDim0 = 1

  # 2nd to last index must belong to other, make it so
  kernel.indexAssignmentDim0 = kernel.indexOrderC[ \
      numIndicesC - 2 ]
  kernel.tensorAssignedDim1 = 1
  if kernel.indexAssignmentDim0 in problem.operation.indexAssignmentsA:
    kernel.tensorAssignedDim1 = 0

  if kernel.tensorAssignedDim0 == kernel.tensorAssignedDim1:
    print "SolutionCandidateGenerator::makeIndexAssignments() - ERROR TileDim0,1 same tensor"

################################################################################
# SolutionCandidateGenerator
################################################################################
class SolutionCandidateGenerator:

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

  universeBranch = [ Structs.BranchType(0), Structs.BranchType(1), \
      Structs.BranchType(2) ]



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
    makeIndexAssignments( kernel, problem )
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

    # for all unroll combinations of selected unroll level
    for unroll in self.universeUnroll[selectedUnroll]:
      kernel.unrolls = unroll
      # if last unroll is multiple of last/unrolled summation
      #if problemSizeUnroll % unroll[len(unroll)-1] > 0:
      #  continue
      for workGroup in self.universeWorkGroupDim:
        kernel.tile.workGroup = workGroup
        # only try skinny work-group if problem is skinny
        if float(workGroup[1])/workGroup[0] \
            > self.skinnyRatioWorkGroup[problemSkinnyDim0]:
          continue
        if float(workGroup[0])/workGroup[1] \
            > self.skinnyRatioWorkGroup[problemSkinnyDim1]:
          continue
        # for all micro-tile dimensions
        for microTileDim0 in range(1, self.maxMicroTileSize):
          for microTileDim1 in range(1, self.maxMicroTileSize):
            microTile = [ microTileDim0, microTileDim1 ]
            kernel.tile.microTile = microTile
            # only try skinny micro-tile if problem is skinny
            if float(microTile[1])/microTile[0] \
                > self.skinnyRatioMicroTile[problemSkinnyDim0]:
              continue
            if float(microTile[0])/microTile[1] \
                > self.skinnyRatioMicroTile[problemSkinnyDim1]:
              continue
            # only try skinny macro-tile if problem is skinny
            macroTileDim0 = workGroup[0] * microTile[0]
            macroTileDim1 = workGroup[1] * microTile[1]
            if float(macroTileDim1)/macroTileDim0 \
                > self.skinnyRatioMacroTile[problemSkinnyDim0]:
              continue
            if float(macroTileDim0)/macroTileDim1 \
                > self.skinnyRatioMacroTile[problemSkinnyDim1]:
              continue
            # macro-tile not too large
            numWorkItems = workGroup[0] * workGroup[1]
            numRegisters = numWorkItems * ( microTile[0] * microTile[1] \
                * kernel.dataTypeC.sizeOf() \
                + microTile[0] * kernel.dataTypeA.sizeOf() \
                + microTile[1] * kernel.dataTypeB.sizeOf() )
            maxRegisters = 16*16*( 4*4*4 + 4*4 + 4*4 )
            if numRegisters > maxRegisters:
              continue

            # kernel grid
            kernelGrid = [ 1, 1, 1 ]
            if tensorStrideDim0 % 1024 == 0:
              kernelGrid[0] = problemSizeDim0 / 1024;
            if tensorStrideDim1 % 1024 == 0:
              kernelGrid[1] = problemSizeDim1 / 1024;

            # for branch types
            for branchType in self.universeBranch:
              solution.kernelGrid = kernelGrid
              solution.kernels = []

              # branch - 1 exact kernel
              if branchType.isNone():
                if problemSizeDim0 % macroTileDim0 != 0 \
                    or problemSizeDim1 % macroTileDim1 != 0:
                  continue
                solution.branch = [branchType, branchType]
                kernel.tile.branch = [branchType, branchType ]
                solution.kernels.append( copy.deepcopy(kernel) )

              # branch - 2-4 kernels
              elif branchType.isMultiple():
                if problemSizeDim0 % macroTileDim0 == 0 \
                    and problemSizeDim1 % macroTileDim1 == 0:
                  continue
                solution.branch = [Structs.BranchType(0), Structs.BranchType(0)]
                # add main kernel
                kernel.tile.branch = [Structs.BranchType(0), \
                    Structs.BranchType(0)]
                solution.kernels.append( copy.deepcopy(kernel) )
                # add edge-0 kernel
                if problemSizeDim0 % macroTileDim0 != 0:
                  solution.kernelGrid[0] += 1
                  solution.branch[0] = branchType
                  kernel.tile.branch = [ branchType, Structs.BranchType(0) ]
                  solution.kernels.append( copy.deepcopy(kernel) )
                # add edge-1 kernel
                if problemSizeDim1 % macroTileDim1 != 0:
                  solution.kernelGrid[1] += 1
                  solution.branch[1] = branchType
                  kernel.tile.branch = [ Structs.BranchType(0), branchType ]
                  solution.kernels.append( copy.deepcopy(kernel) )
                # add corner-01 kernel
                if problemSizeDim0 % macroTileDim0 != 0 \
                    and problemSizeDim1 % macroTileDim1 != 0:
                  kernel.tile.branch = [ branchType, branchType ]
                  solution.kernels.append( copy.deepcopy(kernel) )

              # branch - 1 branched kernel
              elif branchType.isBranched():
                if problemSizeDim0 % macroTileDim0 == 0 \
                    and problemSizeDim1 % macroTileDim1 == 0:
                  continue
                solution.branch = [branchType, branchType]
                kernel.tile.branch = [branchType, branchType ]
                solution.kernels.append( copy.deepcopy(kernel) )

              # branch - unknown
              else:
                print "ERROR - unrecognized branchType"

              # kernels, grid, and branching specified, now add solution
              solutionCandidates.append( copy.deepcopy(solution) )
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


