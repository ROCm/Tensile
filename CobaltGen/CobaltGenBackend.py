import glob
import argparse

import FileWriter
import Structs

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

