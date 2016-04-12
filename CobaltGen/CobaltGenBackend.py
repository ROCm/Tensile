import glob
import argparse

import FileWriter
import Structs

psMap = {} #Dict

def addToMap( exactMatch, problem, solution, time, validationStatus ):
  if exactMatch not in psMap:
    psMap[exactMatch] = {}
  if problem not in psMap[exactMatch]:
    psMap[exactMatch][problem] = {}
  if solution not in psMap[exactMatch][problem]:
    psMap[exactMatch][problem][solution] = []
  psMap[exactMatch][problem][solution].append(time)


# EXACT_MATCH:
#  deviceProfile-
#  precisions-
#  operationType-
#  indexAssignments-
#  offsetsRequired
#  initialStridesRequired

#Implementation Details
# Solution
#   kernelGrid = {2, 2, 1}
#   branch = { 1, 1}
#   ppdOffsets = false
#   ppdLeadingStride = false
#   ppdAll = false
# Kernel[0]
#   typeC, A, B
#   Operation
#   indexOrderC
#   indexOrderSummation
#   indexAssignmentDim0
#   indexAssignmentDim1
#   unrollDimStride0
#   unrollDimStride1
#   unrollDimSize
#   unrollDimStrideGreaterThanTileDimStrideA
#   unrollDimStrideLessThanTileDimStrideB
#   problem = ()
#   tile = ()
#   unrolls = {}
#   ppdOffsets = False
#   ppdLeadingStride = False
#   ppdAll = False
# Kernel[1] = null
#   

# SIZE
# PERFORMANCE_MATCH:
#  stride%1024
#  tile
#  unroll
#  branch
#  optimize away alpha, beta, offsets, initial strides

#Problem/Solution Map
# 1) Read in list
# 2) SolutionsForProblem[EXACT_MATCH].append( [Problem/Solution, time] )
# 3) for each EXACT_MATCH
# 4) for each Problem/Solution, condense all times into single time score

# if size > sizeThreshold
#     same organization as AutoGEMM
#     use fastest if multiple of teastest
#     use 2nd fastest if multiple of 2nd fastest
#     ... continue until they aren't faster than fallback
#     use fallback
# next largest size
# scan backward through sizes until you lose performance %PerfTol

# write 1 file per EXACT_MATCH
# will need to write out
# if (size0 < 32 && size1 > threshold) return skinny0()
# elif (size1 < 32 && size0 > threshold) return skinny1();
# else regular()


#P1) getSolution(Problem)
#  deviceProfile

#P2) getSolution_DeviceProfile(Problem)

#P3) getSolution_DeviceProfile_EXACT_MATCH
#  skinny
#  requires alpha, beta
#  requires offsets
#  requires leading strides

#P4 sizes
################################################################################
# Generate Backend Files
################################################################################
def GenBackendFromFiles( \
    inputFiles, \
    outputPath, \
    backend, \
    validate ):
  pass

  ##############################################################################
  # (1) accumulate set of problems
  #problemSet = set() # every problem we'll benchmark
  # for each input file, accumulate problems
  #for inputFile in inputFiles:
  #  print "status: reading problems from " + os.path.basename(inputFile)
  #  FileReader.getProblemsFromXML( inputFile, problemSet )
  #print "status: " + str(len(problemSet)) + " unique problems found"
  #for problem in problemSet:
  #  print str(problem)

  ##############################################################################
  # (2) list candidate solutions for each problem
  #solutionCandidateGenerator = \
  #    SolutionCandidateGenerator.SolutionCandidateGenerator()
  #allSolutions = set() # all solutions to be written
  #allKernels = set() # all gpu kernels to be written
  #benchmarkList = [] # problems and associated solution candidates
  #print "status: generating solution candidates for problems"
  #totalSolutions = 0
  #totalKernels = 0
  #for problem in problemSet:
  #  solutionCandidates = \
  #      solutionCandidateGenerator.getSolutionCandidatesForProblem( \
  #      problem )
  #  benchmarkList.append( [problem, solutionCandidates] )
  #  totalSolutions += len(solutionCandidates)
  #  for solution in solutionCandidates:
  #    allSolutions.add( solution )
  #  kernelsInSolutionCandidates = getKernelsFromSolutions(solutionCandidates)
  #  for kernel in kernelsInSolutionCandidates:
  #    allKernels.add( kernel )
  #    totalKernels+=1
  #print "status:   " + str(totalSolutions) + " total solutions"
  #print "status:   " + str(len(allSolutions)) + " unique solutions"
  #print "status:   " + str(totalKernels) + " total kernels"
  #print "status:   " + str(len(allKernels)) + " unique kernels"
  #kernelWriter = KernelWriter.KernelWriter(backend)
  #for kernel in allKernels:
  #  print kernelWriter.getName(kernel) + ":" + str(kernel) + ":" + str(hash(kernel))
  allKernels = set()
  allSolutions = set()
  getSolutionLogic = []

  ##############################################################################
  # (3) write benchmark files
  fileWriter = FileWriter.FileWriter(outputPath, backend)
  fileWriter.writeKernelFiles( allKernels )
  fileWriter.writeSolutionFiles( allSolutions )
  fileWriter.writeBackendFiles( getSolutionLogic )



################################################################################
# GenLibrary - Main
################################################################################
if __name__ == "__main__":

  # arguments
  ap = argparse.ArgumentParser(description="CobaltGenBackend")
  ap.add_argument("--input-path", dest="inputPath", required=True )
  ap.add_argument("--output-path", dest="outputPath", required=True )
  ap.add_argument("--backend", dest="backend", required=True, \
      choices=["OpenCL_1.2", "HIP"] )
  ap.add_argument("--enable-validation", dest="validate", action="store_true" )
  ap.add_argument("--optimize-alpha", dest="optimizeAlphaStr" )
  ap.add_argument("--optimize-beta", dest="optimizeBetaStr" )


  # parse arguments
  args = ap.parse_args()
  inputFiles = glob.glob( args.inputPath+"/*.xml" )
  backend = Structs.Backend();
  if args.backend == "OpenCL_1.2":
    backend.value = 0
  elif args.backend == "HIP":
    backend.value = 1

  # print settings
  print "CobaltGenBackend[ " + str(backend) + " ] " + str(inputFiles)

  # generate backend
  GenBackendFromFiles( \
      inputFiles, \
      args.outputPath, \
      backend, \
      args.validate )

