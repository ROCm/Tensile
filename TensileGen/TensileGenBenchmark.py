################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

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
    solutionsPath, \
    generatedPath, \
    backend, \
    optimizeAlpha, \
    optimizeBeta):
  print "\nGenBenchmarkFromFiles:"
  print "  problemFiles=" + str(inputFiles)
  print "  solutionsPath=" + str(solutionsPath)
  print "  generatedPath=" + str(generatedPath)
  print "  backend=" + str(backend)

  ##############################################################################
  # (1) accumulate set of problems
  #problemSet = set() # every problem we'll benchmark
  problemTree = {}
  #problemTree[deviceProfile][ExactMatch] = Set() of problems
  # for each input file, accumulate problems
  for inputFile in inputFiles:
    # print "TensileGenBenchmark: reading problems from " + os.path.basename(inputFile)
    FileReader.getProblemsFromXML( inputFile, problemTree, optimizeAlpha, optimizeBeta )
  #print "TensileGenBenchmark: " + str(len(problemSet)) + " unique problem(s) found"
  #for problem in problemSet:
  #  print str(problem)

  ##############################################################################
  # (2) list candidate solutions for each problem
  solutionCandidateGenerator = \
      SolutionCandidateGenerator.SolutionCandidateGenerator(optimizeAlpha, optimizeBeta, backend )
  allSolutions = set() # all solutions to be written
  allKernels = set() # all gpu kernels to be written
  benchmarkList = {} # problems and associated solution candidates
  print "TensileGenBenchmark: generating solution candidates for problems"
  problemIdx = 0
  for deviceProfile, exactMatches in problemTree.iteritems():
    print "DeviceProfile: " + str(deviceProfile)
    for exactMatch, problemSet in exactMatches.iteritems():
      print "ExactMatch: " + str(exactMatch)

      for problem in problemSet:
        solutionCandidates = \
            solutionCandidateGenerator.getSolutionCandidatesForProblem( \
            problem )
        #if len(solutionCandidates) < 61:
        #  print problem
        #  for solution in solutionCandidates:
        #    print solution

        benchmarkList[problem] = solutionCandidates
        # print solutionCandidates
        for solution in solutionCandidates:
          # for s in allSolutions:
            # if s == solution:
            #   print "match"
            #   print s
            #   print solution
          allSolutions.add( solution )
          # print len(allSolutions)
          # print len(allSolutions)
        kernelsInSolutionCandidates = getKernelsFromSolutions(solutionCandidates)
        for kernel in kernelsInSolutionCandidates:
          if kernel != None:
            allKernels.add( kernel )
            # print kernel
        print "Prob[" + str(problemIdx) + "] \"" + str(problem) + "\": " + str(len(solutionCandidates)) + "/" + str(len(allSolutions)) + " solutions"
        problemIdx += 1
  kernelWriter = KernelWriter.KernelWriter(backend)
  #for kernel in allKernels:
  #  print kernelWriter.getName(kernel) + ":" + str(kernel) + ":" + str(hash(kernel))

  ##############################################################################
  # (3) write benchmark files
  fileWriter = FileWriter.FileWriter(generatedPath, backend, True)
  fileWriter.writeKernelFiles( allKernels )
  fileWriter.writeSolutionFiles( allSolutions )
  fileWriter.writeBenchmarkFiles( problemTree, benchmarkList )


################################################################################
# TensileGenBenchmark - Main
################################################################################
if __name__ == "__main__":

  # arguments
  ap = argparse.ArgumentParser(description="TensileGenBenchmark")
  ap.add_argument("--input-path", dest="inputPath", required=True )
  ap.add_argument("--output-path", dest="buildPath", required=True )
  ap.add_argument("--backend", dest="backend", required=True, \
      choices=["OpenCL_1.2", "HIP"] )
  ap.add_argument("--optimize-alpha", dest="optimizeAlphaStr" )
  ap.add_argument("--optimize-beta", dest="optimizeBetaStr" )
  ap.set_defaults(optimizeAlphaStr="Off")
  ap.set_defaults(optimizeBetaStr="Off")

  # parse arguments
  args = ap.parse_args()
  inputFiles = glob.glob(args.inputPath + "/*.xml")
  backend = Structs.Backend();
  if args.backend == "OpenCL_1.2":
    backend.value = 0
  elif args.backend == "HIP":
    backend.value = 1

  # print settings
  print "\nTensileGenBenchmark:"
  print "  backend=" + str(backend)
  print "  buildPath=" + args.buildPath
  print "  inputPath=" + args.inputPath
  print "  inputFiles=" + str(inputFiles)

  # generate benchmark
  GenBenchmarkFromFiles( \
      inputFiles, \
      args.buildPath, \
      backend,
      args.optimizeAlphaStr=="On" or args.optimizeAlphaStr=="ON",
      args.optimizeBetaStr=="On" or args.optimizeBetaStr=="ON" )

