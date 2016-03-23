import copy
import Structs
import FileReader
import KernelWriter
import argparse

################################################################################
# Make Index Assignments
# indicesSummation:
#    largest stride -> shortest stride
# indicesC:
#    batched largest stride (A+B) -> shortest stride
#    free largest stride (of A,B input tensor) -> shortest stride
#    last two indices must belong to different A,B and are assigned d0,d1
# TODO - should batched be mingled among free for faster performance?
################################################################################
def makeIndexAssignments(kernel, problem):
  numIndicesC = problem.operation.numIndicesFree \
      + problem.operation.numIndicesBatch
  numIndicesA = len(problem.operation.indexAssignmentsA)
  numIndicesB = len(problem.operation.indexAssignmentsB)
  # C indices in order of descending stride
  # sort free indices, then append after batched indices
  indicesBatchedUnsorted = []
  indicesFreeUnsorted = []
  for i in range(0,numIndicesC):
    indexIsBatched = False
    if i in problem.operation.indexAssignmentsA:
      if i in problem.operation.indexAssignmentsB:
        indexIsBatched = True
    if indexIsBatched:
      stride = 0
      for j in range(0,numIndicesA):
        if problem.operation.indexAssignmentsA[j] == i:
          stride += problem.tensorA.dimensions[j].stride
        if problem.operation.indexAssignmentsB[j] == i:
          stride += problem.tensorB.dimensions[j].stride
      indicesBatchedUnsorted.append([stride, i])
    else:
      stride = 0
      indexBelongsToTensor = 0
      for j in range(0,numIndicesA):
        if problem.operation.indexAssignmentsA[j] == i:
          stride = problem.tensorA.dimensions[j].stride
          indexBelongsToTensor = 0
        if problem.operation.indexAssignmentsB[j] == i:
          stride = problem.tensorB.dimensions[j].stride
          indexBelongsToTensor = 1
      indicesFreeUnsorted.append( [stride, i, indexBelongsToTensor] )

  indicesBatchedSorted = sorted( indicesBatchedUnsorted, \
      key = lambda x: int(x[0]), reverse=True )
  indicesFreeSorted = sorted( indicesFreeUnsorted, \
      key = lambda x: int(x[0]), reverse=True )
  # if last two free indices belong to same tensor
  if indicesFreeSorted[len(indicesFreeSorted)-1][2] \
      == indicesFreeSorted[len(indicesFreeSorted)-2][2]:
    # look backwards for smallest stride belonging to different tensor
    for i in range(len(indicesFreeSorted)-1, 0):
      if indicesFreeSorted[len(indicesFreeSorted)-1][2] \
          != indicesFreeSorted[i][3]:
        # remove idx i from current location
        tmp = indicesFreeSorted.pop(i)
        # and place it second to last
        indicesFreeSorted.insert(len(indicesFreeSorted)-1,tmp)
  #print indicesFreeSorted

  # the last two indices will be d0,d1; d0 is the one with the shortest C stride
  if problem.tensorC.dimensions[indicesFreeSorted[len(indicesFreeSorted)-1][1]].stride \
      > problem.tensorC.dimensions[indicesFreeSorted[len(indicesFreeSorted)-2][1]].stride: # need to swap
    #print "swapping d0,d1"
    tmp = indicesFreeSorted.pop()
    indicesFreeSorted.insert(len(indicesFreeSorted)-1,tmp)
    #print indicesFreeSorted

  kernel.indexAssignmentDim0 = indicesFreeSorted[len(indicesFreeSorted)-1][1]
  kernel.tensorAssignedDim0 = indicesFreeSorted[len(indicesFreeSorted)-1][2]
  kernel.indexAssignmentDim1 = indicesFreeSorted[len(indicesFreeSorted)-2][1]
  kernel.tensorAssignedDim1 = indicesFreeSorted[len(indicesFreeSorted)-2][2]
  strideD0 = indicesFreeSorted[len(indicesFreeSorted)-1][0]
  strideD1 = indicesFreeSorted[len(indicesFreeSorted)-2][0]
  #print "d0=%u, d1=%u" % (kernel.indexAssignmentDim0, kernel.indexAssignmentDim1)
  #print "strideD0,1 = " + str(strideD0) + ", " + str(strideD1)

  for index in indicesBatchedSorted:
    kernel.indexOrderC.append( index[1] )
  for index in indicesFreeSorted:
    kernel.indexOrderC.append( index[1] )

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


  #unrollDimStride = indicesSummationSorted[len(indicesSummationSorted)-1][0]
  unrollIndex = kernel.indexOrderSummation[len(kernel.indexOrderSummation)-1] + len(problem.tensorC.dimensions)
  kernel.indexUnroll = unrollIndex
  unrollIndexA = problem.operation.indexAssignmentsA.index(unrollIndex)
  unrollIndexB = problem.operation.indexAssignmentsB.index(unrollIndex)
  #print "unrollIndex = " + str(unrollIndex)
  #print "indexAssignmentsA = " + str(problem.operation.indexAssignmentsA)
  #print "indexAssignmentsB = " + str(problem.operation.indexAssignmentsB)
  #print "unrollIndexA,B = " + str(unrollIndexA) + ", " + str(unrollIndexB)
  unrollDimStrideA = problem.tensorA.dimensions[unrollIndexA].stride
  unrollDimStrideB = problem.tensorB.dimensions[unrollIndexB].stride
  #print "unrollStrideA,B = " + str(unrollDimStrideA) + ", " + str(unrollDimStrideB)
  #print "tensorAssignedDim0 = " + ("A" if kernel.tensorAssignedDim0==0 else "B")
  #print "strideD0 = " + str(strideD0)
  #print "strideD1 = " + str(strideD1)

  #kernel.unrollDimStrideGreaterThanTileDimStride0 = \
  #    indicesFreeSorted[len(indicesFreeSorted)-2][0] < unrollDimStride
  #kernel.unrollDimStrideGreaterThanTileDimStride1 = \
  #    indicesFreeSorted[len(indicesFreeSorted)-1][0] < unrollDimStride
  if kernel.tensorAssignedDim0 == 0: # A assigned dim0
    kernel.unrollDimStrideGreaterThanTileDimStrideA = \
      unrollDimStrideA > strideD0
    kernel.unrollDimStrideLessThanTileDimStrideB = \
      unrollDimStrideB < strideD1
  else:
    kernel.unrollDimStrideGreaterThanTileDimStrideA = \
      unrollDimStrideA > strideD1
    kernel.unrollDimStrideLessThanTileDimStrideB = \
      unrollDimStrideB < strideD0

  # print kernel name
  #kw = KernelWriter.KernelWriter(0)
  #print kw.getName(kernel)
  #print "\n"

################################################################################
# SolutionCandidateGenerator
################################################################################
class SolutionCandidateGenerator:



  # Tuneable Performance Parameters
  # skinnyness: dim1 / dim0 <= ratio[not skinny, is skinny]
  # increasing these parameters will test a wider variety of tiles
  skinnyRatioWorkGroup = [ 1, 2]
  skinnyRatioMicroTile = [ 1, 2]
  skinnyRatioMacroTile = [ skinnyRatioWorkGroup[0]*skinnyRatioMicroTile[0], \
      skinnyRatioWorkGroup[1]*skinnyRatioMicroTile[1] ]
  minMicroTileSize = 1
  maxMicroTileSize = 6
  universeUnroll = { \
       1: [ [  1 ], [ 16, 1 ], [  8, 1 ] ], \
       2: [ [  2 ], [ 16, 2 ], [  8, 2 ] ], \
       4: [ [  4 ], [ 16, 4 ], [  8, 4 ] ], \
       8: [ [  8 ], [ 16, 8 ] ], \
      16: [ [ 16 ], [ 8 ] ] \
      }
  
  """
  universeWorkGroupDim = [ \
       [1,64],  [2,32], [4,16],  [8,8],  [16,4], [32,2],  [64,1], \
      [1,128],  [2,64], [4,32], [8,16],  [16,8], [32,4],  [64,2], [128,1], \
      [1,192],  [2,96], [3,64], [4,48],  [6,32], [8,24], [12,16], [16, 12], \
                [24,8], [32,6], [48,4],  [64,3], [96,2], [192,1], \
      [1,256], [2,128], [4,64], [8,32], [16,16], [32,8],  [64,4],  [128,2], \
               [256,1] ]
  """
  universeWorkGroupDim = [ [16,16] ]
 
  # removed non-branch type
  universeBranch = [ Structs.BranchType(1), Structs.BranchType(2) ]

  ##############################################################################
  # init
  ##############################################################################
  def __init__(self, optimizeAlpha, optimizeBeta):
    self.optimizeAlpha = optimizeAlpha
    self.optimizeBeta = optimizeBeta

  ##############################################################################
  # getSolutionCandidatesForProblem
  ##############################################################################
  def getSolutionCandidatesForProblem( self, inputProblem ):
    problem = copy.deepcopy(inputProblem)

    # optimize alpha and beta?
    if not self.optimizeAlpha and problem.operation.useAlpha==0:
      problem.operation.useAlpha = True
      problem.operation.alphaType = problem.tensorC.dataType
    if not self.optimizeBeta and problem.operation.useBeta==0:
      problem.operation.useBeta = True
      problem.operation.betaType = problem.tensorC.dataType

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
    #kernel.indexAssignmentDim0 = kernel.indexOrderC[ \
    #    numIndicesC - 2 ]
    #kernel.indexAssignmentDim1 = kernel.indexOrderC[ \
    #    numIndicesC - 1 ]

    # Problem Characteristics affecting performance
    problemSizeDim0 = problem.tensorC.dimensions[ \
        kernel.indexAssignmentDim0].size
    problemSizeDim1 = problem.tensorC.dimensions[ \
        kernel.indexAssignmentDim1].size
    problemSkinnyDim0 = 0 # false
    if problemSizeDim0 < 96 and problemSizeDim1 > 1024:
      problemSkinnyDim0 = 1
    problemSkinnyDim1 = 0
    if problemSizeDim1 < 96 and problemSizeDim0 > 1024:
      problemSkinnyDim1 = 1
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
      # TODO remove this; for debugging just to one unroll
      if len(unroll) > 1:
        continue
      kernel.unrolls = unroll
      # if last unroll is multiple of last/unrolled summation
      if problemSizeUnroll % unroll[len(unroll)-1] > 0:
        continue
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
        for microTileDim0 in range(self.minMicroTileSize, \
            self.maxMicroTileSize+1):
          for microTileDim1 in range(self.minMicroTileSize, \
              self.maxMicroTileSize+1):
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
                * kernel.dataTypeC.numRegisters() \
                + microTile[0] * kernel.dataTypeA.numRegisters() \
                + microTile[1] * kernel.dataTypeB.numRegisters() )
            maxRegisters = 16*16*( 4*4*4 + 4*4 + 4*4 )
            maxRegisters /= 2; # TODO remove this; bypasses VS compiler limit string length
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
              solution.kernelGrid = copy.deepcopy(kernelGrid)
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
                solution.branch = [branchType, branchType]
                # add main kernel
                kernel.tile.branch = [Structs.BranchType(0), Structs.BranchType(0)]
                solution.kernels.append( copy.deepcopy(kernel) )
                # add edge-0 kernel
                solution.kernelGrid[0] += 1
                kernel.tile.branch = [ branchType, Structs.BranchType(0) ]
                solution.kernels.append( copy.deepcopy(kernel) )                
                # add edge-1 kernel
                solution.kernelGrid[1] += 1
                kernel.tile.branch = [ Structs.BranchType(0), branchType ]
                solution.kernels.append( copy.deepcopy(kernel) )
                # add corner-01 kernel
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
                solution.kernels.append( None )
                solution.kernels.append( None )
                solution.kernels.append( None )

              # branch - unknown
              else:
                print "ERROR - unrecognized branchType"

              # kernels, grid, and branching specified, now add solution
              # print solution
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


