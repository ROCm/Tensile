import glob
import argparse
import os

import FileReader
import FileWriter
import Structs


################################################################################
# Generate Backend Files
################################################################################
def GenBackendFromFiles( \
    inputFiles, \
    outputPath, \
    backend ):
  
  # read raw solution times
  psTimesRaw = {}
  for inputFile in inputFiles:
    print "status: reading problem/solutions from " + os.path.basename(inputFile) + "\n"
    FileReader.getSolutionsFromXML( inputFile, psTimesRaw )
  # print "status: created dictionary - " + str(psTimes)
  
  # structures needed to write backend
  psTimes = {}
  kernelSet = set()
  solutionSet = set()
  for deviceProfile, exactMatches in psTimesRaw.iteritems():
    psTimes[deviceProfile] = {}
    #print "DeviceProfile: " + str(deviceProfile)
    for exactMatch, problems in exactMatches.iteritems():
      psTimes[deviceProfile][exactMatch] = []
      #print "ExactMatch: " + str(exactMatch)
      #print len(problems)
      for problem, solutionCandidates in problems.iteritems():
        #print str(solutionCandidates)
        #print len(solutionCandidates)
        #print "Problem: " + str(problem)
        # choose fastest solution
        for solution, solutionBenchmark in solutionCandidates.iteritems():
          avgTime = 1e100
          if len(solutionBenchmark.times) > 0 and solutionBenchmark.validationStatus != -1:
            avgTime = sum(solutionBenchmark.times) / len(solutionBenchmark.times)
            psTimes[deviceProfile][exactMatch].append([problem, solution, avgTime])
      # if this exact match didn't have any psps with times, remove
      if len(psTimes[deviceProfile][exactMatch]) < 1:
        print "CobaltGenBackend: ExactMatch %s has no benchmark times; removing." % str(exactMatch)
        psTimes[deviceProfile].pop(exactMatch, None)
    # if this device profile didn't have any exact matches with times, remove
    if len(psTimes[deviceProfile]) < 1:
      print "CobaltGenBackend: Device Profile %s has no benchmark times; removing." % str(deviceProfile)
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
  ap = argparse.ArgumentParser(description="CobaltGenBackend")
  ap.add_argument("--input-path", dest="inputPath", required=True )
  ap.add_argument("--output-path", dest="outputPath", required=True )
  ap.add_argument("--backend", dest="backend", required=True, \
      choices=["OpenCL_1.2", "HIP"] )
  # ap.add_argument("--enable-validation", dest="validate", action="store_true" )
  # ap.add_argument("--optimize-alpha", dest="optimizeAlphaStr" )
  # ap.add_argument("--optimize-beta", dest="optimizeBetaStr" )


  # parse arguments
  args = ap.parse_args()
  inputFiles = glob.glob( args.inputPath+"/*.xml" )
  backend = Structs.Backend();
  if args.backend == "OpenCL_1.2":
    backend.value = 0
  elif args.backend == "HIP":
    backend.value = 1

  # print settings
  print "\nCobaltGenBackend:"
  print "  backend=" + str(backend)
  print "  outputPath=" + args.outputPath
  print "  inputPath=" + args.inputPath
  print "  inputFiles=" + str(inputFiles)

  # generate backend
  GenBackendFromFiles( \
      inputFiles, \
      args.outputPath, \
      backend )

