import os
import sys
import argparse
import getopt
import glob

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
    resultFile, \
    backend ):

  ##############################################################################
  # (1) accumulate set of problems
  problemSet = set() # every problem we'll benchmark
  # for each input file, accumulate problems
  for inputFile in inputFiles:
    print "CobaltGenBenchmark: reading problems from " + os.path.basename(inputFile)
    FileReader.getProblemsFromXML( inputFile, problemSet )
  print "CobaltGenBenchmark: " + str(len(problemSet)) + " unique problems found"
  #for problem in problemSet:
  #  print str(problem)

  ##############################################################################
  # (2) list candidate solutions for each problem
  solutionCandidateGenerator = \
      SolutionCandidateGenerator.SolutionCandidateGenerator()
  allSolutions = set() # all solutions to be written
  allKernels = set() # all gpu kernels to be written
  benchmarkList = [] # problems and associated solution candidates
  print "CobaltGenBenchmark: generating solution candidates for problems"
  totalSolutions = 0
  totalKernels = 0
  problemIdx = 0
  for problem in problemSet:
    solutionCandidates = \
        solutionCandidateGenerator.getSolutionCandidatesForProblem( \
        problem )
    if len(solutionCandidates) > 1000:
      print problem
    benchmarkList.append( [problem, solutionCandidates] )
    totalSolutions += len(solutionCandidates)
    for solution in solutionCandidates:
      allSolutions.add( solution )
    kernelsInSolutionCandidates = getKernelsFromSolutions(solutionCandidates)
    for kernel in kernelsInSolutionCandidates:
      if kernel != None:
        allKernels.add( kernel )
        totalKernels+=1
    print "Prob[" + str(problemIdx) + "]: " + str(len(solutionCandidates)) + "/" + str(len(allSolutions)) + " solutions"
    problemIdx += 1
  print "CobaltGenBenchmark:   " + str(totalSolutions) + " total solutions"
  print "CobaltGenBenchmark:   " + str(len(allSolutions)) + " unique solutions"
  print "CobaltGenBenchmark:   " + str(totalKernels) + " total kernels"
  print "CobaltGenBenchmark:   " + str(len(allKernels)) + " unique kernels"
  kernelWriter = KernelWriter.KernelWriter(backend)
  #for kernel in allKernels:
  #  print kernelWriter.getName(kernel) + ":" + str(kernel) + ":" + str(hash(kernel))

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

  # arguments
  ap = argparse.ArgumentParser(description="CobaltGenBenchmark")
  ap.add_argument("--input-path", dest="inputPath", required=True )
  ap.add_argument("--output-path", dest="outputPath", required=True )
  ap.add_argument("--result-file", dest="resultFile", required=True )
  ap.add_argument("--backend", dest="backend", required=True, \
      choices=["OpenCL_1.2", "HIP"] )

  # parse arguments
  args = ap.parse_args()
  inputFiles = glob.glob(args.inputPath + "/*.xml")
  backend = Structs.Backend();
  if args.backend == "OpenCL_1.2":
    backend.value = 0
  elif args.backend == "HIP":
    backend.value = 1

  # print settings
  print "CobaltGenBenchmark[ " + str(backend) + " ] " + str(inputFiles)

  # generate benchmark
  GenBenchmarkFromFiles( \
      inputFiles, \
      args.outputPath, \
      args.resultFile, \
      backend )

