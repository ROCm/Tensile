import glob
import argparse
import os

import FileReader
import FileWriter
import Structs





# EXACT_MATCH:
# SIZE
# PERFORMANCE_MATCH:
#  stride%1024
#  tile
#  unroll
#  branch
#  optimize away alpha, beta, offsets, initial strides

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
    backend ):
  
  # read raw solution times
  psTimes = {}
  for inputFile in inputFiles:
    print "status: reading problem/solutions from " + os.path.basename(inputFile)
    FileReader.getSolutionsFromXML( inputFile, psTimes )
  print "status: created dictionary - " + str(psTimes)
  
  # structures needed to write backend
  psMap = {}
  kernelSet = set()
  solutionSet = set()
  for deviceProfile, exactMatches in psTimes.iteritems():
    psMap[deviceProfile] = {}
    print "DeviceProfile: " + str(deviceProfile)
    for exactMatch, problems in exactMatches.iteritems():
      psMap[deviceProfile][exactMatch] = {}
      print "ExactMatch: " + str(exactMatch)
      for problem, solutionCandidates in problems.iteritems():
        print "Problem: " + str(problem)
        # choose fastest solution
        solutionCandidatesUnsorted = []
        for solution, solutionBenchmark in solutionCandidates.iteritems():
          avgTime = 1e100
          if len(solutionBenchmark.times)>0:
            avgTime = sum(solutionBenchmark.times) / len(solutionBenchmark.times)
          solutionCandidatesUnsorted.append( [solution, avgTime] )
        solutionCandidatesSorted = sorted( solutionCandidatesUnsorted, \
          key = lambda x: int(x[1]))
        fastestSolution = solutionCandidatesSorted[0][0]
        fastestSolutionTime = solutionCandidatesSorted[0][1]
        print "Winner: " + str(fastestSolutionTime) + " is " + str(fastestSolution)

        # add fastest solution to backend
        psMap[deviceProfile][exactMatch][problem] = fastestSolution
        solutionSet.add(fastestSolution)
        for kernel in fastestSolution.kernels:
          kernelSet.add(kernel)
  fileWriter = FileWriter.FileWriter(outputPath, backend, False)
  fileWriter.writeBackendFiles(psMap)
  
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
  print "\nCobaltGenBackend: backend=\"" + str(backend) + "\"; outputPath=\"" + args.outputPath + "\"; inputFiles=" + str(inputFiles)

  # generate backend
  GenBackendFromFiles( \
      inputFiles, \
      args.outputPath, \
      backend )

