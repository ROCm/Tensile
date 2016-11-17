################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import glob
import argparse
import os
import sys

import FileReader
import FileWriter
import Structs


################################################################################
# Generate Backend Files
################################################################################
def GenBackendFromFiles( \
    inputFiles, \
    outputPath, \
    backend, \
    optimizeAlpha, \
    optimizeBeta ):
  
  # read raw solution times
  psTimesRaw = {}
  for inputFile in inputFiles:
    print "TensileGen: Reading " + os.path.basename(inputFile)
    FileReader.getSolutionsFromXML( inputFile, psTimesRaw, optimizeAlpha, optimizeBeta )
  # print "status: created dictionary - " + str(psTimes)
  
  # structures needed to write backend
  psTimes = {}
  kernelSet = set()
  solutionSet = set()
  for deviceProfile, exactMatches in psTimesRaw.iteritems():
    psTimes[deviceProfile] = {}
    #print "DeviceProfile: " + str(deviceProfile)
    for exactMatch, problems in exactMatches.iteritems():
      rangeProblems = problems[0]
      exactProblems = problems[1]
      #print len(rangeProblems), len(exactProblems)
      psTimes[deviceProfile][exactMatch] = [[],[]]
      #print "ExactMatch: " + str(exactMatch)
      #print len(problems)
      for rangeProblem, solutionCandidates in rangeProblems.iteritems():
        for solution, solutionBenchmark in solutionCandidates.iteritems():
          avgTime = 1e100
          if len(solutionBenchmark.times) > 0 and solutionBenchmark.validationStatus != -1:
            avgTime = sum(solutionBenchmark.times) / len(solutionBenchmark.times)
            psTimes[deviceProfile][exactMatch][0].append([rangeProblem, solution, avgTime])
            
      for exactProblem, solutionCandidates in exactProblems.iteritems():
        for solution, solutionBenchmark in solutionCandidates.iteritems():
          avgTime = 1e100
          if len(solutionBenchmark.times) > 0 and solutionBenchmark.validationStatus != -1:
            avgTime = sum(solutionBenchmark.times) / len(solutionBenchmark.times)
            psTimes[deviceProfile][exactMatch][1].append([exactProblem, solution, avgTime])


      # if this exact match didn't have any psps with times, remove
      if len(psTimes[deviceProfile][exactMatch][0]) < 1 and len(psTimes[deviceProfile][exactMatch][1]) < 1:
        print "TensileGenBackend: ExactMatch %s has no benchmark times; removing." % str(exactMatch)
        psTimes[deviceProfile].pop(exactMatch, None)


    # if this device profile didn't have any exact matches with times, remove
    if len(psTimes[deviceProfile]) < 1:
      print "TensileGenBackend: Device Profile %s has no benchmark times; removing." % str(deviceProfile)
      psTimes.pop(deviceProfile, None)



  # kernelSet.remove(None)
  fileWriter = FileWriter.FileWriter(outputPath, backend, False)
  fileWriter.writeBackendFiles(psTimes)
  
  # getSolution(problem) - top level
    # which device do i match, with default
  # getSolution_DeviceProfile(problem) - device level
    # which exact match do i match, with default
  # getSolution_DeviceProfile_ExactMatch(problem) - problem level
    # which size and mod do i match, with default
        
  



################################################################################
# GenLibrary - Main
################################################################################
if __name__ == "__main__":

  # arguments
  ap = argparse.ArgumentParser(description="TensileGenBackend")
  ap.add_argument("--input-path", dest="inputPath", required=True )
  ap.add_argument("--output-path", dest="outputPath", required=True )
  ap.add_argument("--backend", dest="backend", required=True, \
      choices=["OpenCL_1.2", "HIP"] )
  ap.add_argument("--optimize-alpha", dest="optimizeAlphaStr" )
  ap.add_argument("--optimize-beta", dest="optimizeBetaStr" )
  ap.set_defaults(optimizeAlphaStr="Off")
  ap.set_defaults(optimizeBetaStr="Off")


  # parse arguments
  args = ap.parse_args()
  inputFiles = glob.glob( args.inputPath+"/*.xml" )
  if len(inputFiles) < 1:
    sys.exit("%s has no xml files" % args.inputPath);
  backend = Structs.Backend();
  if args.backend == "OpenCL_1.2":
    backend.value = 0
  elif args.backend == "HIP":
    backend.value = 1

  # print settings
  print "TensileGen: backend=%s, numInputFiles=%u" %( str(backend), len(inputFiles) )
  print "  InputPath=" + args.inputPath
  print "  OutputPath=" + args.outputPath

  #print args.optimizeAlphaStr
  #print args.optimizeBetaStr

  # generate backend
  GenBackendFromFiles( \
      inputFiles, \
      args.outputPath, \
      backend, \
      args.optimizeAlphaStr=="On" or args.optimizeAlphaStr=="ON", \
      args.optimizeBetaStr=="On" or args.optimizeBetaStr=="ON" )
  print "TensileGen: DONE."

