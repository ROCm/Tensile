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
      kernelsCMakeFile.write( "  " + kernelFilePath \
          + kernelHeaderFileName + "\n")
      allKernelsHeaderFile.write( "#include \"" + kernelHeaderFileName + "\"\r")

    kernelsCMakeFile.write(")\n")
    kernelsCMakeFile.write("source_group(Kernels FILES ${Cobalt_KernelFiles_GENERATED_DYNAMIC} )\n")
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
          + solutionHeaderFileName + "\n")
      solutionsCMakeFile.write( "  " + solutionFilePath \
          + solutionSourceFileName + "\n")
      allSolutionsHeaderFile.write( "#include \"" \
          + solutionHeaderFileName + "\"\n")

    solutionsCMakeFile.write(")\n")
    solutionsCMakeFile.write("source_group(Solutions FILES ${Cobalt_SolutionFiles_GENERATED_DYNAMIC} )\n")
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
    s += "Cobalt::Solution *solutionCandidates[numSolutions];\n"

    dataType = Structs.DataType(Structs.DataType.single)
    s += dataType.toCpp() + " alphaSingle = 2.f;\n"
    s += dataType.toCpp() + " betaSingle = 3.f;\n"
    dataType = Structs.DataType(Structs.DataType.double)
    s += dataType.toCpp() + " alphaDouble = 4.f;\n"
    s += dataType.toCpp() + " betaDouble = 5.f;\n"
    dataType = Structs.DataType(Structs.DataType.singleComplex)
    s += dataType.toCpp() + " alphaSingleComplex = { 6.f, 7.f };\n"
    s += dataType.toCpp() + " betaSingleComplex = {8.f, 9.f };\n"
    dataType = Structs.DataType(Structs.DataType.doubleComplex)
    s += dataType.toCpp() + " alphaDoubleComplex = { 10.0, 11.0 };\n"
    s += dataType.toCpp() + " betaDoubleComplex = {12.0, 13.0 };\n"
    s += "\n"

    # initializeSolutionCandidates
    s += "void initializeSolutionCandidates() {\n"

    # DeviceProfile
    s += "  CobaltDeviceProfile deviceProfile;\n"
    s += "  deviceProfile.numDevices = 1;\n"
    s += "  sprintf_s(deviceProfile.devices[0].name, \"TODO\" );\n"
    s += "  deviceProfile.devices[0].numComputeUnits = -1;\n"
    s += "  deviceProfile.devices[0].clockFrequency = -1;\n"
    s += "\n"
    s += "  CobaltTensor tensorC;\n"
    s += "  CobaltTensor tensorA;\n"
    s += "  CobaltTensor tensorB;\n"
    s += "  std::vector<unsigned int> indexAssignmentsA(CobaltTensor::maxDimensions);\n"
    s += "  std::vector<unsigned int> indexAssignmentsB(CobaltTensor::maxDimensions);\n"
    s += "  CobaltOperationType operationType;\n"
    s += "  CobaltDataType alphaType;\n"
    s += "  CobaltDataType betaType;\n"
    s += "  CobaltStatus status;\n"
    templateInstantiationSet = set()
    solutionStartIdx = 0
    tensorSizeMaxC = 0
    tensorSizeMaxA = 0
    tensorSizeMaxB = 0

    # for problems
    for problemIdx in range(0,numProblems):
      problemSolutionPair = problemSolutionCandidates[problemIdx]
      problem = problemSolutionPair[0]
      solutionList = problemSolutionPair[1]
      numSolutions = len(solutionList)
      solutionEndIdx = solutionStartIdx + numSolutions

      # problem.tensorC
      s += "/* problem " + str(problemIdx) + "/" + str(numProblems) + " */\n"
      s += "  tensorC.dataType = " \
          + problem.tensorC.dataType.getLibString() + ";\n"
      tensorDimC = len(problem.tensorC.dimensions)
      s += "  tensorC.numDimensions = " + str(tensorDimC) + ";\n"
      for i in range(0,tensorDimC):
        s += "  tensorC.dimensions[" + str(i) + "].stride = " \
            + str(problem.tensorC.dimensions[i].stride) + ";\n"
        s += "  tensorC.dimensions[" + str(i) + "].size = " \
            + str(problem.tensorC.dimensions[i].size) + ";\n"

      # problem.tensorA
      s += "  tensorA.dataType = " \
          + problem.tensorA.dataType.getLibString() + ";\n"
      tensorDimA = len(problem.tensorA.dimensions)
      s += "  tensorA.numDimensions = " + str(tensorDimA) + ";\n"
      for i in range(0,tensorDimA):
        s += "  tensorA.dimensions[" + str(i) + "].stride = " \
            + str(problem.tensorA.dimensions[i].stride) + ";\n"
        s += "  tensorA.dimensions[" + str(i) + "].size = " \
            + str(problem.tensorA.dimensions[i].size) + ";\n"

      # problem.tensorB
      s += "  tensorB.dataType = " \
          + problem.tensorB.dataType.getLibString() + ";\n"
      tensorDimB = len(problem.tensorB.dimensions)
      s += "  tensorB.numDimensions = " + str(tensorDimB) + ";\n"
      for i in range(0,tensorDimB):
        s += "  tensorB.dimensions[" + str(i) + "].stride = " \
            + str(problem.tensorB.dimensions[i].stride) + ";\n"
        s += "  tensorB.dimensions[" + str(i) + "].size = " \
            + str(problem.tensorB.dimensions[i].size) + ";\n"

      # problem.operation
      s += "  operationType = " \
          + problem.operation.type.getLibString() + ";\n"

      # operation.alpha
      #s += "  problem.operation.useAlpha = " \
      #    + ( "true" if problem.operation.useAlpha else "false" ) + ";\n"
      s += "  alphaType = " \
          + problem.operation.alphaType.getLibString() + ";\n"

      # operation.beta
      #s += "  problem.operation.useBeta = " \
      #    + ( "true" if problem.operation.useBeta else "false" ) + ";\n"
      s += "  betaType = " \
          + problem.operation.betaType.getLibString() + ";\n"

      # operation.indices
      #s += "  problem.operation.numIndicesFree = " \
      #    + str(problem.operation.numIndicesFree) + ";\n"
      #s += "  problem.operation.numIndicesBatch = " \
      #    + str(problem.operation.numIndicesBatch) + ";\n"
      #s += "  problem.operation.numIndicesSummation = " \
      #    + str(problem.operation.numIndicesSummation) + ";\n"
      numIndicesA = len(problem.operation.indexAssignmentsA)
      for i in range(0,numIndicesA):
        s += "  indexAssignmentsA[" + str(i) + "] = " \
            + str(problem.operation.indexAssignmentsA[i]) + ";\n"
      numIndicesB = len(problem.operation.indexAssignmentsB)
      for i in range(0,numIndicesB):
        s += "  indexAssignmentsB[" + str(i) + "] = " \
            + str(problem.operation.indexAssignmentsB[i]) + ";\n"

      # store problem
      s += "  problems[" + str(problemIdx) + "] = cobaltCreateProblem(\n"
      s += "      tensorC,\n"
      s += "      tensorA,\n"
      s += "      tensorB,\n"
      s += "      &indexAssignmentsA[0],\n"
      s += "      &indexAssignmentsB[0],\n"
      s += "      operationType,\n"
      s += "      alphaType,\n"
      s += "      betaType,\n"
      s += "      deviceProfile,\n"
      s += "      &status);\n"
      s += "\n"

      # numSolutionsPerProblem
      s += "  numSolutionsPerProblem[" + str(problemIdx) + "] = " \
          + str(numSolutions) + ";\n"
      for i in range(0,numSolutions):
        s += "  solutionCandidates[" + str(solutionStartIdx+i) + "] = new Cobalt::" \
            + self.solutionWriter.getName(solutionList[i])+self.solutionWriter.getTemplateArgList(solutionList[i])+"( *(problems[" + str(problemIdx) + "]->pimpl) ); // " \
            + str(i) + "/" + str(numSolutions) + "\n"
        templateInstantiationSet.add(self.solutionWriter.getTemplateArgList(solutionList[i]))
      s += "\n"
      solutionStartIdx = solutionEndIdx

      # max tensor size
      for dimension in problem.tensorC.dimensions:
        tensorSizeDimC = dimension.stride * dimension.size \
            * problem.tensorC.dataType.numBytes()
        if tensorSizeDimC > tensorSizeMaxC:
          tensorSizeMaxC = tensorSizeDimC
          print "tensorSizeMaxC = " + str(tensorSizeMaxC)
      for dimension in problem.tensorA.dimensions:
        tensorSizeDimA = dimension.stride * dimension.size \
            * problem.tensorA.dataType.numBytes()
        if tensorSizeDimA > tensorSizeMaxA:
          tensorSizeMaxA = tensorSizeDimA
          print "tensorSizeMaxA = " + str(tensorSizeMaxA)
      for dimension in problem.tensorB.dimensions:
        tensorSizeDimB = dimension.stride * dimension.size \
            * problem.tensorB.dataType.numBytes()
        if tensorSizeDimB > tensorSizeMaxB:
          tensorSizeMaxB = tensorSizeDimB
          print "tensorSizeMaxB = " + str(tensorSizeMaxB)

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
    s += "const size_t tensorSizeMaxC = " + str(tensorSizeMaxC) + ";\n"
    s += "const size_t tensorSizeMaxA = " + str(tensorSizeMaxA) + ";\n"
    s += "const size_t tensorSizeMaxB = " + str(tensorSizeMaxB) + ";\n"
    s += "extern size_t numSolutionsPerProblem[numProblems];\n"
    s += "extern CobaltProblem problems[numProblems];\n"
    s += "extern Cobalt::Solution *solutionCandidates[numSolutions];\n"
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

    # explicit template instantiation
    templateInstantiationsPath = self.outputPath + self.solutionSubdirectory \
        + "SolutionTemplateInstantiations.inl"
    templateInstantiationsFile = open(templateInstantiationsPath, "w")
    templateInstantiationsFile.write("/* explicit template instantiations for base classes of generated solutions */\n\n")
    for templateInstantiationStr in templateInstantiationSet:
      templateInstantiationsFile.write("template class Cobalt::SolutionOpenCL" \
          +templateInstantiationStr + ";\n")


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
    fileString += "#ifndef KERNEL_" + kernelName.upper() + "_CPP\n"
    fileString += "#define KERNEL_" + kernelName.upper() + "_CPP\n"
    fileString += "\n"
    fileString += "#include \"" + kernelName + ".h\"\n"
    fileString += "\n"
    fileString += "cl_kernel " + kernelName + "_kernel = nullptr;\n"
    #fileString += "// const size_t %s_workGroup[3] = { %u, %u, 1 };\n" \
    #    % (kernelName, kernel.tile.workGroup[0], kernel.tile.workGroup[1] )
    #fileString += "// const size_t %s_microTile[2] = { %u, %u };\n" \
    #    % (kernelName, kernel.tile.microTile[0], kernel.tile.microTile[0] )
    #fileString += "// const size_t %s_unroll = %u;\n" \
    #    % (kernelName, kernel.unrolls[len(kernel.unrolls)-1])
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
    fileString += "#include \"CL/cl.h\"\n"
    fileString += "\n"
    fileString += "extern const size_t %s_workGroup[3];\n" % kernelName
    fileString += "extern const size_t %s_microTile[2];\n" % kernelName
    fileString += "extern const size_t %s_unroll;\n" \
        % (kernelName)
    fileString += "extern const char * const %s_src;\n" % kernelName
    fileString += "extern cl_kernel %s_kernel;\n" % kernelName
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
