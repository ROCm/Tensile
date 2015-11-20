import os
import sys
import argparse
import getopt

import FileReader
import FileWriter
import SolutionCandidateGenerator



################################################################################
# getKernelsFromSolutions
################################################################################
def getKernelsFromSolutions( solutionSet ):
  pass


################################################################################
# GenBenchmark
################################################################################
def GenBenchmarkFromInputFiles( \
    inputFiles, \
    outputPath, \
    language ):

  ##############################################################################
  # (1) accumulate set of problems
  problemSet = set() # every problem we'll benchmark
  # for each input file, accumulate problems
  for inputFile in inputFiles:
    FileReader.getProblemsFromXML( inputFile, problemSet )

  ##############################################################################
  # (2) list candidate solutions for each problem
  globalSolutionSet = set() # all solutions to be written
  globalKernelSet = set() # all gpu kernels to be written
  benchmarkList = [] # problems and associated solution candidates
  for problem in problemSet:
    solutionCandidates = \
        SolutionCandidates.getSolutionCandidatesForProblem( \
        problem )
    benchmarkList.append( [problem, solutionCandidates] )
    globalSolutionSet.add( solutionCandidates )
    globalKernelSet.add( getKernelsFromSolutions(solutionCandidates) )

  ##############################################################################
  # (3) write benchmark files
  FileWriter.writeKernelFiles( globalKernelSet )
  FileWriter.writeSolutionFiles( globalSolutionSet )
  FileWriter.writeBenchmarkFiles( benchmarkList )


################################################################################
# GenBenchmark - Main
################################################################################
if __name__ == "__main__":

  # arguments
  ap = argparse.ArgumentParser(description="CobaltGenBenchmark")
  ap.add_argument("--output-path", dest="outputPath" )
  ap.add_argument("--input-file", dest="inputFiles", action="append" )
  ap.add_argument("--language", dest="language" )

  # parse arguments
  args = ap.parse_args()

  # print settings
  print "CobaltGenBenchmark.py: using language " + args.language

  # generate benchmark
  GenBenchmarkFromFiles( \
      args.inputFiles, \
      args.outputPath, \
      args.language )

