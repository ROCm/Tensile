import copy
import Structs
import FileReader
import KernelWriter
import SolutionWriter
import argparse
import math
"""
TODOs
 - move loads
 - debug ddp offsets
"""


"""
3 levels
 2 - (e)xhaustive (everything that's supported; for validation)
 1 - (t)horough (for benchmarking new project; for research)
 0 - (f)ast (smallest subset "guaranteed" to find fastest solution; for production)

work-groups (threshold = 256)
 (e) all m*n <= threshold
 (t) all m*n multiple of 64 and <= threshold; exact matches for small dimensions
 (f) 8x8, 16x16, exact matches for small dimensions

micro-tiles (threshold = 8*8)
 (e) all m*n <= threshold
 (t) m=n, m,n=[2,sqrt(threshold)]; if skinny m,n within factor of 2
 (f) subset of (t): based on free index size; small tiles if problem is small, large tiles if problem is large

unrolls (threshold = 16)
 (e) all u <= threshold
 (t) 4, 8, 16; if sum < threshold, exact match also
 (f) same as (t)
 
loads
 (e) all
 (t) only 1 or N depending on transpose
 (f) same as (t)

preprocessor defines
[0] leading strides, [1] offsets, [2] everything
 (e)
     [ 0, 0, 0], \
     [ 1, 0, 0], \
     [ 0, 1, 0], \
     [ 1, 1, 0], \
     [ 1, 1, 1], \
 (t) [ 1, 0, 0], \
     [ 1, 1, 1], \
 (f) [ 1, 0, 0], \



"""


################################################################################
# SolutionCandidateGenerator
################################################################################
class SolutionCandidateGenerator:
  
  # hardware limits
  maxLocalMemoryBytes = 32768
  localMemPad = 1
  maxRegisters = 16*16*( 4*4*4 + 4*4 + 4*4 )

  # mode definitions
  modeExhaustive = 2
  modeThorough   = 1
  modeFast       = 0

  # mode selections (fully exhaustive: ~10s of Millions of solutions; don't bother)
  modeWorkGroups              = 0 # (e:1457, t:20, f:2) 
  modeMicroTiles              = 0
  modeUnrolls                 = 0
  modeLoads                   = 0
  modePreprocessorDefinitions = 0

  # thresholds
  thresholdMicroTiles = 8*8
  thresholdUnrolls    = 16
  thresholdSkinny     = 128 # if dim0 or dim1 < threshold, then problem is skinny
  ratioWorkGroupSkinny    = 2
  ratioMacroTileSkinny    = 4
  ratioMacroTileThorough  = 2
  ratioMacroTileSlow      = 2

  # Research Options
  noBranches = False # True means don't generate any solution requiring branches, i.e., only generate fastest
  noMultipleKernels = False # True means don't generate solution requiring multiple kernels, i.e., only single-kernel fastest or branched
  printDetails = False

  

  ##############################################################################
  # init
  ##############################################################################
  def __init__(self, optimizeAlpha, optimizeBeta, backend):
    self.optimizeAlpha = optimizeAlpha
    self.optimizeBeta = optimizeBeta
    self.backend = backend
    if self.backend.isHIP():
      # remove optimizations so that all kernels have identical arguments
      self.ppdUniverse = [ [True, False, False] ]
    self.kernelWriter = KernelWriter.KernelWriter(backend)
    self.solutionWriter = SolutionWriter.SolutionWriter(backend)

  ##############################################################################
  # getSolutionCandidatesForProblem
  ##############################################################################
  def getSolutionCandidatesForProblem( self, inputProblem ):
    problem = copy.deepcopy(inputProblem)
    numCandidates = 0
    fullyExhaustive = self.modeWorkGroups == self.modeExhaustive \
        and self.modeMicroTiles == self.modeExhaustive \
        and self.modeUnrolls == self.modeExhaustive \
        and self.modeLoads == self.modeExhaustive \
        and self.modePreprocessorDefinitions == self.modeExhaustive
    # optimize alpha and beta?
    if not self.optimizeAlpha and not problem.operation.useAlpha():
      if self.printDetails: print "SCG: reverting void alpha to typeC b/c not optimizing"
      problem.operation.alphaType = problem.tensorC.dataType
    if not self.optimizeBeta and not problem.operation.useBeta():
      if self.printDetails: print "SCG: reverting void beta to typeC b/c not optimizing"
      problem.operation.betaType = problem.tensorC.dataType

    numIndicesC = len(problem.tensorC.dimensions)
    numIndicesA = len(problem.tensorA.dimensions)
    numIndicesB = len(problem.tensorB.dimensions)

    # create solution object
    kernel = Structs.Kernel()
    solution = Structs.Solution()
    solutionCandidates = []

    # Solution Correctness Parameters
    kernel.dataTypeC = problem.tensorC.dataType
    kernel.dataTypeA = problem.tensorA.dataType
    kernel.dataTypeB = problem.tensorB.dataType
    kernel.problem = problem

    # Index Assignments
    kernel.indexOrderC = []
    kernel.indexOrderSummation = []
    makeIndexAssignments( kernel, problem )
    
    ###################################
    # Dimension Sizes
    ###################################
    
    # transpose work-group order for better cacheing?
    transA = not kernel.unrollDimStrideGreaterThanTileDimStrideA
    transB = not kernel.unrollDimStrideLessThanTileDimStrideB
    if transA and transB:
      kernel.transposeWorkGroupOrder = True

    # Problem Characteristics affecting performance
    problemSizeDim0 = problem.tensorC.dimensions[ kernel.indexAssignmentDim0].size
    problemSizeDim1 = problem.tensorC.dimensions[ kernel.indexAssignmentDim1].size

    problemSkinnyDim0 = problemSizeDim0 < self.thresholdSkinny
    problemSkinnyDim1 = problemSizeDim1 < self.thresholdSkinny
    
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

    inputProblem.sizeType = 0
    problem.sizeType = 0
    problemIsRectangular = True
    if (not problemSizeDim0 % 16 == 0 and not (problemSizeDim0+1) % 16 == 0) or (not problemSizeDim1 % 16 == 0 and not (problemSizeDim1+1) % 16 == 0) or (not problemSizeUnroll % 16 == 0 and not (problemSizeUnroll+1) % 16 == 0):
      inputProblem.sizeType = 1
      problem.sizeType = 1

    ###################################
    # Determine Search Universe
    ###################################
    problemIsRectangular = problem.getSizeType() == 0
    if not problemIsRectangular:
      print "WARNING: problem has unusual size; many candidates solutions will be generated."


    # work-groups
    universeWorkGroups = []
    thresholdWorkGroupSize = 256 # threads
    if self.modeWorkGroups == self.modeExhaustive:
      for m in range(1, thresholdWorkGroupSize+1):
        for n in range(1, thresholdWorkGroupSize+1):
          if m*n < thresholdWorkGroupSize:
            universeWorkGroups.append( [m,n] )
    else: # fast or thorough; both will include skinnies where applicable
      if self.modeWorkGroups == self.modeThorough:
        universeWorkGroups = [ \
            [4,16], [8,8],  [16,4], \
            [4,32], [8,16],  [16,8], [32,4], \
            [4,48], [6,32], [8,24], [12,16], [16, 12], [24,8], [32,6], [48,4], \
            [4,64], [8,32], [16,16], [32,8],  [64,4] ]
      else: # fast
        universeWorkGroups = [ [16,16], [8,8] ]
      if not problemIsRectangular and problemSizeDim0 < self.thresholdSkinny:
        if self.printDetails: print "SCG: adding skinny(dim0) work-groups"
        for workGroupDim0 in range(1, thresholdWorkGroupSize+1):
          if problemSizeDim0 % workGroupDim0 == 0:
            for workGroupSize in [256, 192, 128, 64]:
              workGroupDim1 = workGroupSize / workGroupDim0 #problemSizeDim0
              if workGroupDim1 > 0:
                universeWorkGroups.append( [workGroupDim0,workGroupDim1] )
            # break
      if not problemIsRectangular and problemSizeDim1 < self.thresholdSkinny:
        if self.printDetails: print "SCG: adding skinny(dim1) work-groups"
        for workGroupDim1 in range(1, thresholdWorkGroupSize+1):
          if problemSizeDim1 % workGroupDim1 == 0:
            for workGroupSize in [256, 192, 128, 64]:
              workGroupDim0 = workGroupSize / workGroupDim1 #problemSizeDim1
              if workGroupDim0 > 0:
                universeWorkGroups.append( [workGroupDim0,workGroupDim1] )
            # break
    if self.printDetails: print "SCG: WorkGroups(" + str(len(universeWorkGroups)) + "): " + str(universeWorkGroups)

    # micro-tiles
    if self.modeMicroTiles == self.modeExhaustive or self.modeMicroTiles == self.modeThorough:
      microTileMin = 1
      microTileMax = int(self.thresholdMicroTiles**0.5)
      # microTileRatio = microTileMax
    else:
      microTileMin = 2
      microTileMax = int(self.thresholdMicroTiles**0.5)
      # microTileRatio = 2
    # print "SCG: MicroTileRatio: " + str(microTileRatio)

    # macro-tiles
    if self.modeWorkGroups == self.modeExhaustive or self.modeMicroTiles == self.modeExhaustive:
      macroTileRatio = self.thresholdMicroTiles * self.thresholdWorkGroupSize
    elif self.modeWorkGroups == self.modeThorough or self.modeMicroTiles == self.modeThorough:
      if problemSkinnyDim0 or problemSkinnyDim1:
        macroTileRatio = self.ratioMacroTileSkinny
      elif transA and not transB:
        macroTileRatio = self.ratioMacroTileSlow
      else:
        macroTileRatio = 1
    else: # fast
      if problemSkinnyDim0 or problemSkinnyDim1:
        macroTileRatio = self.ratioMacroTileSkinny
      elif transA and not transB:
        macroTileRatio = self.ratioMacroTileSlow
      else:
        macroTileRatio = 1
    if self.printDetails: print "SCG: MacroTileRatio: " + str(macroTileRatio)
    
    # unrolls
    universeUnrolls = []
    if self.modeUnrolls == self.modeExhaustive:
      for unroll in range(1, self.thresholdUnrolls+1):
        if unroll <= problemSizeUnroll and problemSizeUnroll % unroll == 0: # exact multiple
          universeUnrolls.append( [unroll] )
        elif unroll < problemSizeUnroll: # needs trailing loop
          universeUnrolls.append( [unroll, 1] )
    else: # thorough or fast
      unrollFast = [16, 8, 4]
      for unroll in unrollFast:
        if unroll <= problemSizeUnroll and problemSizeUnroll % unroll == 0: # exact multiple
          universeUnrolls.append( [unroll] )
        elif unroll < problemSizeUnroll: # needs trailing loop
          universeUnrolls.append( [unroll, 1] )
      if problemSizeUnroll < self.thresholdUnrolls and not problemSizeUnroll in unrollFast:
        universeUnrolls.append( [problemSizeUnroll] )
    if self.printDetails: print "SCG: Unrolls(" + str(len(universeUnrolls)) + "): " + str(universeUnrolls)

    # preprocessor defines
    requireOffsets = problem.operation.useOffsets
    requireInitialStrides = problem.tensorC.dimensions[0].stride != 1 or problem.tensorA.dimensions[0].stride != 1 or problem.tensorB.dimensions[0].stride != 1
    universePreprocessorDefinitions = []
    if self.modePreprocessorDefinitions == self.modeExhaustive:
      universePreprocessorDefinitions = [ [ 0, 0, 0] ]
      if not requireInitialStrides:
        universePreprocessorDefinitions.append( [ 1, 0, 0] )
      if not requireOffsets:
        universePreprocessorDefinitions.append( [ 0, 1, 0] )
      if not requireInitialStrides and not requireOffsets:
        universePreprocessorDefinitions.append( [ 1, 1, 0] )
        universePreprocessorDefinitions.append( [ 1, 1, 1] )
    elif self.modePreprocessorDefinitions == self.modeThorough:
      universePreprocessorDefinitions = [ [ 1, 1 if requireInitialStrides else 0, 0] ]
      if not requireInitialStrides and not requireOffsets:
        universePreprocessorDefinitions.append( [ 1, 1, 1] )
    else:
      universePreprocessorDefinitions = [ [ 1, 1 if requireInitialStrides else 0, 0] ]
    if self.printDetails: print "SCG: PreprocessorDefinitions(" + str(len(universePreprocessorDefinitions)) + "): " + str(universePreprocessorDefinitions)

    
    # kernel grid
    kernelGrid = [ 1, 1, 1 ]
    """
    # wait for 4096 problem to re-appear before re-enabling kernel grid
    print kernel.unrollDimStride0, kernel.unrollDimStride1
    if not problemSkinnyDim0 and not problemSkinnyDim1 and \
        (kernel.unrollDimStride0 % 1024 == 0 or kernel.unrollDimStride1 % 1024 == 0):
      kernelGrid[0] = problemSizeDim0 / 2048;
      kernelGrid[1] = problemSizeDim1 / 2048;
      kernelGrid[2] = problemSizeUnroll / 1024
      if kernelGrid[0] == 0:
        kernelGrid[0]=1
      if kernelGrid[1] == 0:
        kernelGrid[1]=1
      if kernelGrid[2] == 0:
        kernelGrid[2]=1
      if kernelGrid[2] > 1 and not kernel.problem.operation.useBeta():
          kernel.problem.operation.betaType = problem.tensorC.dataType
          print "forcing useBeta=True due to mod1024 kernel grid"
      # print "kernelGrid = {%u, %u, %u}" % ( kernelGrid[0], kernelGrid[1], kernelGrid[2])
    """
    
    ###################################
    # begin solution universe
    ###################################

    
    ###################################
    # for unrolls
    for unroll in universeUnrolls:
      # print "unroll = " + str(unroll)
      kernel.unrolls = unroll
      
      ###################################
      # for work-groups
      for workGroup in universeWorkGroups:
        kernel.tile.workGroup = workGroup
        if fullyExhaustive:
          print "NumCandidates: " + str(numCandidates)
        
        ###################################
        # for all micro-tile dimensions
        # print microTileMin, microTileMax, workGroup[0], workGroup[1]
        for microTileDim0 in range(microTileMin, microTileMax+1):
          for microTileDim1 in range(microTileMin, microTileMax+1):
            microTile = [ microTileDim0, microTileDim1 ]
            kernel.tile.microTile = microTile

            # micro-tile not too skinny
            # if float(microTile[1])/microTile[0] > microTileRatio or float(microTile[0])/microTile[1] > microTileRatio:
              #print "micro tile too skinny %u %u %u %u" % (workGroup[0], workGroup[1], microTile[0], microTile[1])
              #continue

            # macro-tile not too skinny
            macroTileDim0 = workGroup[0] * microTile[0]
            macroTileDim1 = workGroup[1] * microTile[1]
            if float(macroTileDim1)/macroTileDim0 > macroTileRatio or float(macroTileDim0)/macroTileDim1 > macroTileRatio:
              #print "macro tile too skinny %u %u %u %u" % (workGroup[0], workGroup[1], microTile[0], microTile[1])
              continue
            
            if self.modeMicroTiles == self.modeFast: # don't accept small work-groups with large micro-tiles; pruning options
              if microTileDim0 > 0.5*workGroup[0] or microTileDim1 > 0.5*workGroup[1]:
                #print "skipping small WG and large UT: %ux%u > .5*%ux%u" % (microTile[0], microTile[1], workGroup[0], workGroup[1] )
                continue

            # macro-tile not too large
            numWorkItems = workGroup[0] * workGroup[1]
            numRegisters = numWorkItems * ( microTile[0] * microTile[1] \
                * kernel.dataTypeC.numRegisters() \
                + microTile[0] * kernel.dataTypeA.numRegisters() \
                + microTile[1] * kernel.dataTypeB.numRegisters() )
            if numRegisters > self.maxRegisters:
              continue

            # local memory not too large
            localMemoryBytes = 0
            if kernel.tensorAssignedDim0 == 0: # dim0 in tensorA
              localMemoryBytes = unroll[0] * ((macroTileDim0+self.localMemPad)*kernel.dataTypeA.numBytes() + (macroTileDim1+self.localMemPad)*kernel.dataTypeB.numBytes())
            else: # dim1 in tensorA
              localMemoryBytes = unroll[0] * ((macroTileDim0+self.localMemPad)*kernel.dataTypeB.numBytes() + (macroTileDim1+self.localMemPad)*kernel.dataTypeA.numBytes())
            if localMemoryBytes > self.maxLocalMemoryBytes:
              continue

            # load grid
            totalNumLoadsA = max(1, (workGroup[0]*microTile[0]*unroll[0])/(workGroup[0]*workGroup[1]) )
            totalNumLoadsB = max(1, (workGroup[1]*microTile[1]*unroll[0])/(workGroup[0]*workGroup[1]) )
            kernel.totalLoadSizeParaA = macroTileDim0 if kernel.unrollDimStrideGreaterThanTileDimStrideA else unroll[0]
            kernel.totalLoadSizePerpA = unroll[0] if kernel.unrollDimStrideGreaterThanTileDimStrideA else macroTileDim0
            kernel.totalLoadSizeParaB = macroTileDim1 if not kernel.unrollDimStrideLessThanTileDimStrideB else unroll[0]
            kernel.totalLoadSizePerpB = unroll[0] if not kernel.unrollDimStrideLessThanTileDimStrideB else macroTileDim1
            
            
            if fullyExhaustive:
              numCandidates += totalNumLoadsA*totalNumLoadsB*len(universePreprocessorDefinitions)*2
              continue

            # num loads parallel A
            universeNumLoadsParaA = []
            if self.modeLoads == self.modeExhaustive:
              for i in range(1, totalNumLoadsA+1):
                universeNumLoadsParaA.append(i)
            elif self.modeLoads == self.modeThorough:
                universeNumLoadsParaA.append( totalNumLoadsA )
                universeNumLoadsParaA.append( 1 )
            else:
              if not transA:
                universeNumLoadsParaA.append( totalNumLoadsA )
              else:
                universeNumLoadsParaA.append( 1 )
            # print "  optionsA = " + str(universeNumLoadsParaA)
            
            # num loads parallel B
            universeNumLoadsParaB = []
            if self.modeLoads == self.modeExhaustive:
              for i in range(1, totalNumLoadsB+1):
                universeNumLoadsParaB.append(i)
            elif self.modeLoads == self.modeThorough:
                universeNumLoadsParaB.append( totalNumLoadsB )
                universeNumLoadsParaB.append( 1 )
            else:
              if transB:
                universeNumLoadsParaB.append( totalNumLoadsB )
              else:
                universeNumLoadsParaB.append( 1 )
            # print "    optionsB = " + str(universeNumLoadsParaB)
            
            ###################################
            # for num loads parallel A
            for numLoadsParaA in universeNumLoadsParaA:
              # if False: # TODO, need this for mode scheme? true means only try perfect tiles w/o branches
              #   if totalNumLoadsA % numLoadsParaA > 0:
              #     continue
              #   if totalLoadSizeParaA%numLoadsParaA>0:
              #     continue
              #   if (workGroup[0]*workGroup[1])%(totalLoadSizeParaA/numLoadsParaA) > 0:
              #     continue
              kernel.numLoadsParaA = numLoadsParaA
              kernel.loadSizeParaA = int(math.ceil(1.0*kernel.totalLoadSizeParaA / kernel.numLoadsParaA ) ) # round up
              kernel.loadSizePerpA = int( (workGroup[0]*workGroup[1])/kernel.loadSizeParaA ) # round down
              if kernel.loadSizePerpA < 1:
                kernel.loadSizePerpA = 1
              if kernel.loadSizePerpA > kernel.totalLoadSizePerpA:
                kernel.loadSizePerpA = kernel.totalLoadSizePerpA
              kernel.numLoadsPerpA = int(math.ceil(1.0*kernel.totalLoadSizePerpA / kernel.loadSizePerpA )) # round up
              # print "  A: nl=%.1fx%.1f ls=%.1fx%.1f" % (kernel.numLoadsParaA, kernel.numLoadsPerpA, kernel.loadSizeParaA, kernel.loadSizePerpA)
              
              ###################################
              # for num loads parallel B
              for numLoadsParaB in universeNumLoadsParaB:
                # if False: # true means only try perfect tiles w/o branches
                #   if totalNumLoadsB % numLoadsParaB > 0:
                #     continue
                #   if totalLoadSizeParaB%numLoadsParaB>0:
                #     continue
                #   if (workGroup[0]*workGroup[1])%(totalLoadSizeParaB/numLoadsParaB) > 0:
                #     continue
                kernel.numLoadsParaB = numLoadsParaB
                kernel.loadSizeParaB = int(math.ceil(1.0*kernel.totalLoadSizeParaB / kernel.numLoadsParaB )) # round up
                kernel.loadSizePerpB = int((workGroup[0]*workGroup[1])/kernel.loadSizeParaB) # round down
                if kernel.loadSizePerpB < 1:
                  kernel.loadSizePerpB = 1
                if kernel.loadSizePerpB > kernel.totalLoadSizePerpB:
                  kernel.loadSizePerpB = kernel.totalLoadSizePerpB
                kernel.numLoadsPerpB = int(math.ceil(1.0*kernel.totalLoadSizePerpB / kernel.loadSizePerpB)) # round up
                # print "    B: nl=%.1fx%.1f ls=%.1fx%.1f" % (kernel.numLoadsParaB, kernel.numLoadsPerpB, kernel.loadSizeParaB, kernel.loadSizePerpB)

                  
                ###################################
                # for preprocessor definitions
                for ppdOptimization in universePreprocessorDefinitions:
                  ppdLeadingStride = ppdOptimization[0]
                  ppdOffsets       = ppdOptimization[1]
                  ppdAll           = ppdOptimization[2]
                    
                  ###################################
                  # for branch types
                  branchTypes = [ Structs.BranchType(1), Structs.BranchType(2) ]
                  for branchType in branchTypes:
                    solution.kernelGrid = copy.deepcopy(kernelGrid)
                    solution.kernels = []
                    leadingStridesOne = False
                    if problem.tensorC.dimensions[0].stride == 1 \
                        and problem.tensorA.dimensions[0].stride == 1 \
                        and problem.tensorB.dimensions[0].stride == 1:
                      leadingStridesOne = True
                    # branch - 2-4 kernels
                    if branchType.isMultiple():
                      if self.noBranches or self.noMultipleKernels:
                        if problemSizeDim0 % macroTileDim0 != 0 \
                            or problemSizeDim1 % macroTileDim1 != 0:
                          continue

                      solution.branch = [branchType, branchType]
                      if leadingStridesOne:
                        solution.ppdLeadingStride = ppdLeadingStride
                      solution.ppdOffsets = ppdOffsets # kernel 0 need offsets?
                      solution.ppdAll = 0 # kernels 1-3 will need sizes
                      # add main kernel
                      kernel.tile.branch = [Structs.BranchType(0), Structs.BranchType(0)]
                      if leadingStridesOne:
                        kernel.ppdLeadingStride = ppdLeadingStride
                      kernel.ppdOffsets = ppdOffsets
                      kernel.ppdAll = ppdAll
                      solution.kernels.append( copy.deepcopy(kernel) )
                      # add edge-0 kernel
                      solution.kernelGrid[0] += 1
                      kernel.tile.branch = [ branchType, Structs.BranchType(0) ]
                      if leadingStridesOne:
                        kernel.ppdLeadingStride = ppdLeadingStride
                      kernel.ppdOffsets = 0
                      kernel.ppdAll = 0
                      solution.kernels.append( copy.deepcopy(kernel) )
                      # add edge-1 kernel
                      solution.kernelGrid[1] += 1
                      kernel.tile.branch = [ Structs.BranchType(0), branchType ]
                      if leadingStridesOne:
                        kernel.ppdLeadingStride = ppdLeadingStride
                      kernel.ppdOffsets = 0
                      kernel.ppdAll = 0
                      solution.kernels.append( copy.deepcopy(kernel) )
                      # add corner-01 kernel
                      kernel.tile.branch = [ branchType, branchType ]
                      if leadingStridesOne:
                        kernel.ppdLeadingStride = ppdLeadingStride
                      kernel.ppdOffsets = 0
                      kernel.ppdAll = 0
                      solution.kernels.append( copy.deepcopy(kernel) )

                    # branch - 1 branched kernel
                    elif branchType.isBranched():
                      if problemSizeDim0 % macroTileDim0 == 0 \
                          and problemSizeDim1 % macroTileDim1 == 0:
                        continue
                      if kernelGrid[0] > 1 or kernelGrid[1] > 1 or kernelGrid[2] > 1: # don't use b kernels for 4096 cases b/c already not using single kernel
                        continue
                      if self.noBranches:
                        continue
                      solution.branch = [branchType, branchType]
                      if leadingStridesOne:
                        solution.ppdLeadingStride = ppdLeadingStride
                      solution.ppdOffsets = ppdOffsets
                      solution.ppdAll = ppdAll
                      kernel.tile.branch = [branchType, branchType ]
                      kernel.ppdLeadingStride = ppdLeadingStride
                      kernel.ppdOffsets = ppdOffsets
                      kernel.ppdAll = ppdAll
                      solution.kernels.append( copy.deepcopy(kernel) )
                      solution.kernels.append( None )
                      solution.kernels.append( None )
                      solution.kernels.append( None )

                    # branch - unknown
                    else:
                      print "ERROR - unrecognized branchType"

                    # kernels, grid, and branching specified, now add solution
                    # print solution
                    # print "  " + self.solutionWriter.getName(solution)
                    solutionCandidates.append( copy.deepcopy(solution) )
    if fullyExhaustive:
      print "NumCandidates: " + str(numCandidates)
    return solutionCandidates


################################################################################
# Make Index Assignments
# indicesSummation:
#    largest stride -> shortest stride
# indicesC:
#    batched largest stride (A+B) -> shortest stride
#    free largest stride (of A,B input tensor) -> shortest stride
#    last two indices must belong to different A,B and are assigned d0,d1
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

  if kernel.tensorAssignedDim0 == 0:
    kernel.indexAssignmentTileA = [0, kernel.indexAssignmentDim0]
    kernel.indexAssignmentTileB = [1, kernel.indexAssignmentDim1]
  else:
    kernel.indexAssignmentTileA = [1, kernel.indexAssignmentDim1]
    kernel.indexAssignmentTileB = [0, kernel.indexAssignmentDim0]

  strideD0 = indicesFreeSorted[len(indicesFreeSorted)-1][0]
  strideD1 = indicesFreeSorted[len(indicesFreeSorted)-2][0]
  # print "d0=%u, d1=%u" % (kernel.indexAssignmentDim0, kernel.indexAssignmentDim1)
  # print "strideD0,1 = " + str(strideD0) + ", " + str(strideD1)

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
  # print "unrollIndex = " + str(unrollIndex)
  # print "indexAssignmentsA = " + str(problem.operation.indexAssignmentsA)
  # print "indexAssignmentsB = " + str(problem.operation.indexAssignmentsB)
  # print "unrollIndexA,B = " + str(unrollIndexA) + ", " + str(unrollIndexB)
  unrollDimStrideA = problem.tensorA.dimensions[unrollIndexA].stride
  unrollDimStrideB = problem.tensorB.dimensions[unrollIndexB].stride
  kernel.unrollDimSize = problem.tensorA.dimensions[unrollIndexA].size
  # print "unrollStrideA,B = " + str(unrollDimStrideA) + ", " + str(unrollDimStrideB)
  # print "tensorAssignedDim0 = " + ("A" if kernel.tensorAssignedDim0==0 else "B")
  # print "strideD0 = " + str(strideD0)
  # print "strideD1 = " + str(strideD1)

  #kernel.unrollDimStrideGreaterThanTileDimStride0 = \
  #    indicesFreeSorted[len(indicesFreeSorted)-2][0] < unrollDimStride
  #kernel.unrollDimStrideGreaterThanTileDimStride1 = \
  #    indicesFreeSorted[len(indicesFreeSorted)-1][0] < unrollDimStride
  if kernel.tensorAssignedDim0 == 0: # A assigned dim0
    kernel.unrollDimStrideGreaterThanTileDimStrideA = \
      unrollDimStrideA > strideD0
    kernel.unrollDimStrideLessThanTileDimStrideB = \
      unrollDimStrideB < strideD1
    kernel.unrollDimStride0 = unrollDimStrideA
    kernel.unrollDimStride1 = unrollDimStrideB
  else:
    kernel.unrollDimStrideGreaterThanTileDimStrideA = \
      unrollDimStrideA > strideD1
    kernel.unrollDimStrideLessThanTileDimStrideB = \
      unrollDimStrideB < strideD0
    kernel.unrollDimStride0 = unrollDimStrideB
    kernel.unrollDimStride1 = unrollDimStrideA

  # print kernel name
  backend = Structs.Backend()
  backend.value = Structs.Backend.opencl12
  kw = KernelWriter.KernelWriter(backend)
  #print kw.getName(kernel)
  #print "\n"



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


