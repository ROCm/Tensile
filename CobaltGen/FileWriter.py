import os
import Structs
import KernelWriter
import SolutionWriter

################################################################################
# File Writer
################################################################################
class FileWriter:

  topDirectory = "/"
  kernelSubdirectory = topDirectory + "/Kernels/"
  solutionSubdirectory = topDirectory + "/Solutions/"
  benchmarkSubdirectory = topDirectory + "/Benchmark/"
  librarySubdirectory = topDirectory + "/Library/"


  ##############################################################################
  # ensurePath
  ##############################################################################
  def ensurePath( self, path ):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
      os.makedirs(dirname)


  ##############################################################################
  # constructor
  ##############################################################################
  def __init__( self, outputPath, backend ):
    self.outputPath = outputPath
    self.backend = backend
    self.kernelWriter = KernelWriter.KernelWriter(backend)
    self.solutionWriter = SolutionWriter.SolutionWriter(self.backend)

    self.ensurePath( self.outputPath + self.topDirectory )
    self.ensurePath( self.outputPath + self.kernelSubdirectory )
    self.ensurePath( self.outputPath + self.solutionSubdirectory )
    self.ensurePath( self.outputPath + self.benchmarkSubdirectory )
    self.ensurePath( self.outputPath + self.librarySubdirectory )


  ##############################################################################
  # writeKernelFiles
  ##############################################################################
  def writeKernelFiles( self, kernelSet ):
    print "status: writing kernel files"

    # main kernel .cpp,.h files
    kernelFilePath = "${Cobalt_DIR_GENERATED}/Kernels/"
    kernelsCMakeFilePath = self.outputPath + self.kernelSubdirectory \
        + "CobaltKernels.cmake"
    allKernelsHeaderFilePath = self.outputPath + self.kernelSubdirectory \
        + "CobaltKernels.h"
    kernelsCMakeFile = open(kernelsCMakeFilePath, "w")
    allKernelsHeaderFile = open(allKernelsHeaderFilePath, "w")

    kernelsCMakeFile.write("\nset( Cobalt_KernelFiles_GENERATED_DYNAMIC\n")


    for kernel in kernelSet:
      # open .inl,.h files
      kernelName = self.kernelWriter.getName(kernel)
      kernelSourceFileName = kernelName + ".cpp"
      kernelHeaderFileName = kernelName + ".h"
      kernelSourceFilePath = self.outputPath + self.kernelSubdirectory + \
          kernelSourceFileName
      kernelHeaderFilePath = self.outputPath + self.kernelSubdirectory + \
          kernelHeaderFileName
      kernelSourceFile = open(kernelSourceFilePath, "w")
      kernelHeaderFile = open(kernelHeaderFilePath, "w")

      # get kernel file string
      kernelSourceFileString = self.getKernelSourceFileString(kernel)
      kernelHeaderFileString = self.getKernelHeaderFileString(kernel)

      # write kernel file string to file
      kernelSourceFile.write( kernelSourceFileString )
      kernelHeaderFile.write( kernelHeaderFileString )
      kernelSourceFile.close()
      kernelHeaderFile.close()

      # write the main CobaltKernels.h,cpp file which #includes these
      kernelsCMakeFile.write( "  " + kernelFilePath \
          + kernelSourceFileName + "\n")
      allKernelsHeaderFile.write( "#include \"" + kernelHeaderFileName + "\"\r")

    kernelsCMakeFile.write(")\n")
    kernelsCMakeFile.close()
    allKernelsHeaderFile.close()


  ##############################################################################
  # writeSolutionFiles
  ##############################################################################
  def writeSolutionFiles( self, solutionSet ):
    print "status: writing solution files"

    # main solution .cpp,.h files
    solutionFilePath = "${Cobalt_DIR_GENERATED}/Solutions/"
    solutionsCMakeFilePath = self.outputPath + self.solutionSubdirectory \
        + "CobaltSolutions.cmake"
    allSolutionsHeaderFilePath = self.outputPath + self.solutionSubdirectory \
        + "CobaltSolutions.h"
    solutionsCMakeFile = open(solutionsCMakeFilePath, "w")
    allSolutionsHeaderFile = open(allSolutionsHeaderFilePath, "w")

    solutionsCMakeFile.write("set( Cobalt_SolutionFiles_GENERATED_DYNAMIC\n")

    for solution in solutionSet:
      # open file
      solutionName = self.solutionWriter.getName(solution)
      solutionSourceFileName = solutionName + ".cpp"
      solutionHeaderFileName = solutionName + ".h"
      solutionSourceFilePath = self.outputPath + self.solutionSubdirectory + \
          solutionSourceFileName
      solutionHeaderFilePath = self.outputPath + self.solutionSubdirectory + \
          solutionHeaderFileName
      solutionSourceFile = open(solutionSourceFilePath, "w")
      solutionHeaderFile = open(solutionHeaderFilePath, "w")

      # get solution file string
      solutionSourceFileString = self.solutionWriter.getSourceString( solution )
      solutionHeaderFileString = self.solutionWriter.getHeaderString( solution )

      # write solution file string to file
      solutionSourceFile.write( solutionSourceFileString )
      solutionHeaderFile.write( solutionHeaderFileString )
      solutionSourceFile.close()
      solutionHeaderFile.close()

      # write the main CobaltSolutions.h,cpp file which #includes these
      solutionsCMakeFile.write( "  " + solutionFilePath \
          + solutionSourceFileName + "\n")
      allSolutionsHeaderFile.write( "#include \"" \
          + solutionHeaderFileName + "\"\n")

    solutionsCMakeFile.write(")\n")
    solutionsCMakeFile.close()
    allSolutionsHeaderFile.close()


  ##############################################################################
  # writeBenchmarkFiles
  ##############################################################################
  def writeBenchmarkFiles( self, problemSolutionCandidates ):
    print "status: writing benchmark files"
    # write a .cpp file which creates an array of problem/solution candidates
    benchmarkSourcePath = self.outputPath + self.benchmarkSubdirectory \
        + "CobaltSolutionCandidates.cpp"
    benchmarkSourceFile = open(benchmarkSourcePath, "w")
    s = ""
    s += "#include \"CobaltSolutionCandidates.h\"\n"
    s += "\n"
    s += "/* benchmark stuff */\n"
    s += "\n"

    # declarations
    numProblems = len(problemSolutionCandidates);
    numSolutions = 0
    for problemSolutionPair in problemSolutionCandidates:
      solutionList = problemSolutionPair[1]
      numSolutions += len(solutionList)
    s += "size_t numSolutionsPerProblem[numProblems];\n"
    s += "CobaltProblem problems[numProblems];\n"
    s += "CobaltSolution *solutionCandidates[numSolutions];\n"

    dataType = Structs.DataType(0)
    s += dataType.toCpp() + " alphaSingle = 2.f;\n"
    s += dataType.toCpp() + " betaSingle = 3.f;\n"
    dataType = Structs.DataType(1)
    s += dataType.toCpp() + " alphaDouble = 4.f;\n"
    s += dataType.toCpp() + " betaDouble = 5.f;\n"
    dataType = Structs.DataType(2)
    s += dataType.toCpp() + " alphaSingleComplex = { 6.f, 7.f };\n"
    s += dataType.toCpp() + " betaSingleComplex = {8.f, 9.f };\n"
    dataType = Structs.DataType(3)
    s += dataType.toCpp() + " alphaDoubleComplex = { 10.0, 11.0 };\n"
    s += dataType.toCpp() + " betaDoubleComplex = {12.0, 13.0 };\n"
    s += "\n"

    # initializeSolutionCandidates
    s += "void initializeSolutionCandidates() {\n"

    # DeviceProfile
    s += "  CobaltDeviceProfile deviceProfile;\n"
    s += "  deviceProfile.numDevices = 1;\n"
    #s += "  deviceProfile.devices[0].name = \"TODO\";\n"
    s += "  deviceProfile.devices[0].numComputeUnits = -1;\n"
    s += "  deviceProfile.devices[0].clockFrequency = -1;\n"
    s += "  CobaltProblem problem;\n"

    solutionStartIdx = 0

    # for problems
    for problemIdx in range(0,numProblems):
      problemSolutionPair = problemSolutionCandidates[problemIdx]
      problem = problemSolutionPair[0]
      solutionList = problemSolutionPair[1]
      numSolutions = len(solutionList)
      solutionEndIdx = solutionStartIdx + numSolutions

      # problem.tensorC
      s += "/* problem " + str(problemIdx) + "/" + str(numProblems) + " */\n"
      tensorDimC = len(problem.tensorC.dimensions)
      s += "  problem.tensorC.numDimensions = " + str(tensorDimC) + ";\n"
      for i in range(0,tensorDimC):
        s += "  problem.tensorC.dimensions[" + str(i) + "].stride = " \
            + str(problem.tensorC.dimensions[i].stride) + ";\n"
        s += "  problem.tensorC.dimensions[" + str(i) + "].size = " \
            + str(problem.tensorC.dimensions[i].size) + ";\n"

      # problem.tensorA
      tensorDimA = len(problem.tensorA.dimensions)
      s += "  problem.tensorA.numDimensions = " + str(tensorDimA) + ";\n"
      for i in range(0,tensorDimA):
        s += "  problem.tensorA.dimensions[" + str(i) + "].stride = " \
            + str(problem.tensorA.dimensions[i].stride) + ";\n"
        s += "  problem.tensorA.dimensions[" + str(i) + "].size = " \
            + str(problem.tensorA.dimensions[i].size) + ";\n"

      # problem.tensorB
      tensorDimB = len(problem.tensorB.dimensions)
      s += "  problem.tensorB.numDimensions = " + str(tensorDimB) + ";\n"
      for i in range(0,tensorDimB):
        s += "  problem.tensorB.dimensions[" + str(i) + "].stride = " \
            + str(problem.tensorB.dimensions[i].stride) + ";\n"
        s += "  problem.tensorB.dimensions[" + str(i) + "].size = " \
            + str(problem.tensorB.dimensions[i].size) + ";\n"

      # problem.deviceProfile
      s += "  problem.deviceProfile = deviceProfile;\n"

      # problem.operation
      s += "  problem.operation.type = " \
          + problem.operation.type.getLibString() + ";\n"

      # operation.alpha
      s += "  problem.operation.alphaType = " \
          + problem.operation.alphaType.getLibString() + ";\n"
      s += "  problem.operation.alpha = &"
      if problem.operation.alphaType.value == 0:
        s += "alphaSingle"
      elif problem.operation.alphaType.value == 1:
        s += "alphaDouble"
      if problem.operation.alphaType.value == 2:
        s += "alphaSingleComplex"
      elif problem.operation.alphaType.value == 3:
        s += "alphaDoubleComplex"
      s += ";\n"

      # operation.beta
      s += "  problem.operation.betaType = " \
          + problem.operation.betaType.getLibString() + ";\n"
      s += "  problem.operation.beta = &"
      if problem.operation.betaType.value == 0:
        s += "betaSingle"
      elif problem.operation.betaType.value == 1:
        s += "betaDouble"
      if problem.operation.betaType.value == 2:
        s += "betaSingleComplex"
      elif problem.operation.betaType.value == 3:
        s += "betaDoubleComplex"
      s += ";\n"

      # operation.indices
      s += "  problem.operation.numIndicesFree = " \
          + str(problem.operation.numIndicesFree) + ";\n"
      s += "  problem.operation.numIndicesBatch = " \
          + str(problem.operation.numIndicesBatch) + ";\n"
      s += "  problem.operation.numIndicesSummation = " \
          + str(problem.operation.numIndicesSummation) + ";\n"
      numIndicesA = len(problem.operation.indexAssignmentsA)
      for i in range(0,numIndicesA):
        s += "  problem.operation.indexAssignmentsA[" + str(i) + "] = " \
            + str(problem.operation.indexAssignmentsA[i]) + ";\n"
      numIndicesB = len(problem.operation.indexAssignmentsB)
      for i in range(0,numIndicesB):
        s += "  problem.operation.indexAssignmentsB[" + str(i) + "] = " \
            + str(problem.operation.indexAssignmentsB[i]) + ";\n"

      # store problem
      s += "  problems[" + str(problemIdx) + "] = problem;\n"
      s += "\n"

      # numSolutionsPerProblem
      s += "  numSolutionsPerProblem[" + str(problemIdx) + "] = " \
          + str(numSolutions) + ";\n"
      for i in range(0,numSolutions):
        s += "  solutionCandidates[" + str(solutionStartIdx+i) + "] = new " \
            + self.solutionWriter.getName(solutionList[i])+"( problem ); // " \
            + str(i) + "/" + str(numSolutions) + "\n"
      s += "\n"

      solutionStartIdx = solutionEndIdx
    s += "}\n"
    benchmarkSourceFile.write(s)
    benchmarkSourceFile.close()

    benchmarkHeaderPath = self.outputPath + self.benchmarkSubdirectory \
        + "CobaltSolutionCandidates.h"
    benchmarkHeaderFile = open(benchmarkHeaderPath, "w")
    s = ""
    s += "#include \"Cobalt.h\"\n"
    s += "#include \"CobaltSolutions.h\"\n"
    s += "#include \"CL/cl.h\"\n"
    s += "\n"
    s += "const size_t numProblems = " + str(numProblems) + ";\n"
    s += "const size_t numSolutions = " + str(numSolutions) + ";\n"
    s += "extern size_t numSolutionsPerProblem[numProblems];\n"
    s += "extern CobaltProblem problems[numProblems];\n"
    s += "extern CobaltSolution *solutionCandidates[numSolutions];\n"
    s += "extern float alphaSingle;\n"
    s += "extern float betaSingle;\n"
    s += "extern double alphaDouble;\n"
    s += "extern double betaDouble;\n"
    s += "extern CobaltComplexFloat alphaSingleComplex;\n"
    s += "extern CobaltComplexFloat betaSingleComplex;\n"
    s += "extern CobaltComplexDouble alphaDoubleComplex;\n"
    s += "extern CobaltComplexDouble betaDoubleComplex;\n"
    s += "\n"
    s += "void initializeSolutionCandidates();\n"
    s += "\n"
    benchmarkHeaderFile.write(s)
    benchmarkHeaderFile.close()

    # write CobaltBenchmark.cmake
    """benchmarkCMakePath = self.outputPath + self.benchmarkSubdirectory \
        + "CobaltBenchmark.cmake"
    benchmarkCMakeFile = open(benchmarkCMakePath, "w")
    s = "# CobaltBenchmark.cmake\n"
    s += "\n"
    s += "# include list of kernel files\n"
    s += "include( " + self.outputPath + self.kernelSubdirectory \
        + "CobaltKernels.cmake"
    s += "\n"
    s += "# include list of solution files\n"
    s += "include( " + self.outputPath + self.solutionSubdirectory \
        + "CobaltSolutions.cmake"
    s += "\n"
    """


  ##############################################################################
  # get source file string
  ##############################################################################
  def getKernelSourceFileString( self, kernel):
    kernelName = self.kernelWriter.getName(kernel)
    fileString = ""
    #fileString += Common.getFileHeader()
    fileString += "#ifndef KERNEL_" + kernelName.upper() + "_INL\n"
    fileString += "#define KERNEL_" + kernelName.upper() + "_INL\n"
    fileString += "\n"
    fileString += "const unsigned int %s_workGroup[3] = { %u, %u, 1 };\n" \
        % (kernelName, kernel.tile.workGroup[0], kernel.tile.workGroup[1] )
    fileString += "const unsigned int %s_microTile[2] = { %u, %u };\n" \
        % (kernelName, kernel.tile.microTile[0], kernel.tile.microTile[0] )
    fileString += "const unsigned int %s_unroll = %u;\n" \
        % (kernelName, kernel.unrolls[len(kernel.unrolls)-1])
    fileString += "\n"
    fileString += "const char * const %s_src =\"" % (kernelName)
    fileString += self.kernelWriter.getBody( kernel )
    fileString += "\";\n"
    fileString += "\n"
    fileString += "#else\n"
    fileString += "#pragma message(\"%s was overriden by user kernel.\")\n" \
        % kernelName
    fileString += "#endif\n"
    return fileString


  ##############################################################################
  # get header file string
  ##############################################################################
  def getKernelHeaderFileString( self, kernel ):
    kernelName = self.kernelWriter.getName(kernel)
    fileString = ""
    #fileString += Common.getFileHeader()
    fileString += "#ifndef KERNEL_" + kernelName.upper() + "_H\n"
    fileString += "#define KERNEL_" + kernelName.upper() + "_H\n"
    fileString += "\n"
    fileString += "extern const unsigned int %s_workGroup[0];\n" % kernelName
    fileString += "extern const unsigned int %s_workGroup[1];\n" % kernelName
    fileString += "extern const unsigned int %s_microTile[0];\n" % kernelName
    fileString += "extern const unsigned int %s_microTile[1];\n" % kernelName
    fileString += "extern const unsigned int %s_unroll;\n" % kernelName
    fileString += "extern const char * const %s_src;\n" % kernelName
    fileString += "#endif\n"
    return fileString


  ##############################################################################
  # write backend files
  ##############################################################################
  def writeBackendFiles( self, getSolutionLogic ):
    print "status: writing backend files"
    # main solution .cpp,.h files
    backendCMakeFilePath = self.outputPath + self.librarySubdirectory \
        + "CobaltLib.cmake"
    getSolutionHeaderFilePath = self.outputPath + self.librarySubdirectory \
        + "CobaltGetSolution.h"
    getSolutionSourceFilePath = self.outputPath + self.librarySubdirectory \
        + "CobaltGetSolution.cpp"

    backendCMakeFile = open(backendCMakeFilePath, "w")
    getSolutionHeaderFile = open(getSolutionHeaderFilePath, "w")
    getSolutionSourceFile = open(getSolutionSourceFilePath, "w")

    backendCMakeFile.write("# CobaltLib Backend CMakeFile\n")
    getSolutionHeaderFile.write("/* GetSolution.h */\n")
    getSolutionSourceFile.write("/* GetSolution.cpp */\n")

    backendCMakeFile.close()
    getSolutionHeaderFile.close()
    getSolutionSourceFile.close()
