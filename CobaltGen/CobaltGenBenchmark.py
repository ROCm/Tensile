import os
import sys
import argparse
import getopt

import Structs
import FileReader
import FileWriter
import SolutionCandidateGenerator
import KernelWriter



################################################################################
# getKernelsFromSolutions
################################################################################
def getKernelsFromSolutions( solutionSet ):
  kernels = []
  for solution in solutionSet:
    for kernel in solution.kernels:
      kernels.append(kernel)
  return kernels


################################################################################
# GenBenchmark
################################################################################
def GenBenchmarkFromFiles( \
    inputFiles, \
    outputPath, \
    backend ):

  ##############################################################################
  # (1) accumulate set of problems
  problemSet = set() # every problem we'll benchmark
  # for each input file, accumulate problems
  for inputFile in inputFiles:
    print "status: reading problems from " + os.path.basename(inputFile)
    FileReader.getProblemsFromXML( inputFile, problemSet )
  print "status: " + str(len(problemSet)) + " unique problems found"
  for problem in problemSet:
    print str(problem)

  ##############################################################################
  # (2) list candidate solutions for each problem
  solutionCandidateGenerator = \
      SolutionCandidateGenerator.SolutionCandidateGenerator()
  allSolutions = set() # all solutions to be written
  allKernels = set() # all gpu kernels to be written
  benchmarkList = [] # problems and associated solution candidates
  print "status: generating solution candidates for problems"
  totalSolutions = 0
  totalKernels = 0
  for problem in problemSet:
    solutionCandidates = \
        solutionCandidateGenerator.getSolutionCandidatesForProblem( \
        problem )
    benchmarkList.append( [problem, solutionCandidates] )
    totalSolutions += len(solutionCandidates)
    for solution in solutionCandidates:
      allSolutions.add( solution )
    kernelsInSolutionCandidates = getKernelsFromSolutions(solutionCandidates)
    for kernel in kernelsInSolutionCandidates:
      allKernels.add( kernel )
      totalKernels+=1
  print "status:   " + str(totalSolutions) + " total solutions"
  print "status:   " + str(len(allSolutions)) + " unique solutions"
  print "status:   " + str(totalKernels) + " total kernels"
  print "status:   " + str(len(allKernels)) + " unique kernels"
  kernelWriter = KernelWriter.KernelWriter(backend)
  for kernel in allKernels:
    print kernelWriter.getName(kernel) + ":" + str(kernel) + ":" + str(hash(kernel))

  ##############################################################################
  # (3) write benchmark files
  fileWriter = FileWriter.FileWriter(outputPath, backend)
  fileWriter.writeKernelFiles( allKernels )
  fileWriter.writeSolutionFiles( allSolutions )
  fileWriter.writeBenchmarkFiles( benchmarkList )


################################################################################
# CobaltGenBenchmark - Main
################################################################################
if __name__ == "__main__":
  print "status: CobaltGenBenchmark.py"

  # arguments
  ap = argparse.ArgumentParser(description="CobaltGenBenchmark")
  ap.add_argument("--output-path", dest="outputPath", required=True )
  ap.add_argument("--input-file", dest="inputFiles", action="append", required=True )
  ap.add_argument("--backend", dest="backend", required=True, choices=["OpenCL1.2", "HIP"] )

  # parse arguments
  args = ap.parse_args()
  backend = Structs.Backend();
  if args.backend == "OpenCL1.2":
    backend.value = 0
  elif args.backend == "HIP":
    backend.value = 1

  # print settings
  print "status: using \"" + str(backend) + "\" backend"

  # generate benchmark
  GenBenchmarkFromFiles( \
      args.inputFiles, \
      args.outputPath, \
      backend )

