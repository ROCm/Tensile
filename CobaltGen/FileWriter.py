import os
import Structs
import KernelWriter
import SolutionWriter
import SolutionSelectionWriter

################################################################################
# File Writer
################################################################################
class FileWriter:

  kernelSubdirectory = "/Kernels/"
  solutionSubdirectory = "/Solutions/"
  otherSubdirectory = "/Other/"


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
  def __init__( self, outputPath, backend, forBenchmark ):
    self.outputPath = outputPath
    self.backend = backend
    self.kernelWriter = KernelWriter.KernelWriter(backend)
    self.solutionWriter = SolutionWriter.SolutionWriter(self.backend)

    self.ensurePath( self.outputPath )
    self.ensurePath( self.outputPath + self.kernelSubdirectory )
    self.ensurePath( self.outputPath + self.solutionSubdirectory )
    self.ensurePath( self.outputPath + self.otherSubdirectory )

    self.forBenchmark = forBenchmark
    self.cobaltDirGenerated = "${CobaltLib_DIR_GENERATED}"
    if self.forBenchmark:
      self.cobaltDirGenerated = "${CobaltBenchmark_DIR_GENERATED}"



  ##############################################################################
  # writeKernelFiles
  ##############################################################################
  def writeKernelFiles( self, kernelSet ):
    print "status: writing kernel files"

    # main kernel .cpp,.h files
    kernelFilePath = self.cobaltDirGenerated + self.kernelSubdirectory
    kernelsCMakeFilePath = self.outputPath + self.kernelSubdirectory \
        + "CobaltKernels.cmake"
    allKernelsHeaderFilePath = self.outputPath + self.kernelSubdirectory \
        + "CobaltKernels.h"
    kernelsCMakeFile = open(kernelsCMakeFilePath, "w")
    allKernelsHeaderFile = open(allKernelsHeaderFilePath, "w")

    if self.forBenchmark:
      kernelsCMakeFile.write("\nset( CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC\n")
    else:
      kernelsCMakeFile.write("\nset( CobaltLib_KernelFiles_GENERATED_DYNAMIC\n")


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
      allKernelsHeaderFile.write( "#include \"" + kernelHeaderFileName + "\"\n")

    kernelsCMakeFile.write(")\n")
    if self.forBenchmark:
      kernelsCMakeFile.write("source_group(CobaltGen\\\\Kernels FILES ${CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC} )\n")
    else:
      kernelsCMakeFile.write("source_group(CobaltGen\\\\Kernels FILES ${CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC} )\n")
    kernelsCMakeFile.close()
    allKernelsHeaderFile.close()


  ##############################################################################
  # writeSolutionFiles
  ##############################################################################
  def writeSolutionFiles( self, solutionSet ):
    print "status: writing solution files"

    # main solution .cpp,.h files
    solutionFilePath = self.cobaltDirGenerated + self.solutionSubdirectory
    solutionsCMakeFilePath = self.outputPath + self.solutionSubdirectory \
        + "CobaltSolutions.cmake"
    allSolutionsHeaderFilePath = self.outputPath + self.solutionSubdirectory \
        + "CobaltSolutions.h"
    solutionsCMakeFile = open(solutionsCMakeFilePath, "w")
    allSolutionsHeaderFile = open(allSolutionsHeaderFilePath, "w")

    if self.forBenchmark:
      solutionsCMakeFile.write("set( CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC\n")
    else:
      solutionsCMakeFile.write("set( CobaltLib_SolutionFiles_GENERATED_DYNAMIC\n")

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
    if self.forBenchmark:
      solutionsCMakeFile.write("source_group(CobaltGen\\\\Solutions FILES ${CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC} )\n")
    else:
      solutionsCMakeFile.write("source_group(CobaltGen\\\\Solutions FILES ${CobaltLib_SolutionFiles_GENERATED_DYNAMIC} )\n")
    solutionsCMakeFile.close()
    allSolutionsHeaderFile.close()


  ##############################################################################
  # writeBenchmarkFiles
  ##############################################################################
  def writeBenchmarkFiles( self, problemSolutionCandidates ):
    print "status: writing benchmark files"

    numProblems = len(problemSolutionCandidates);
    numSolutions = 0


    templateInstantiationSet = set()
    solutionStartIdx = 0
    tensorSizeMaxC = 0
    tensorSizeMaxA = 0
    tensorSizeMaxB = 0
    solutionEndIdx = -1

    # for problems
    for problemIdx in range(0,numProblems):
      problemSolutionPair = problemSolutionCandidates[problemIdx]
      problem = problemSolutionPair[0]
      solutionList = problemSolutionPair[1]
      numSolutions = len(solutionList)
      solutionEndIdx = numSolutions

      problemName = str(problem)
      # open problem file
      problemFileNameBase = "init_" + problemName + "_candidates"
      problemSourcePath = self.outputPath + self.otherSubdirectory \
          + problemFileNameBase + ".cpp"
      problemSourceFile = open(problemSourcePath, "w")
      problemHeaderPath = self.outputPath + self.otherSubdirectory \
          + problemFileNameBase + ".h"
      problemHeaderFile = open(problemHeaderPath, "w")

      s = "" # source file string
      s += "#include \"" + problemFileNameBase + ".h\"\n"
      s += "\n"
      s += "/* problem " + str(problemIdx) + "/" + str(numProblems) + ": " + problemName + " */\n"
      s += "\n"

      # initializeSolutionCandidates
      s += "void init_" + problemName + "_candidates(CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates) {\n"

      # DeviceProfile
      s += "\n"
      s += "  CobaltDeviceProfile deviceProfile = cobaltCreateEmptyDeviceProfile();\n"
      s += "  deviceProfile.numDevices = %u;\n" % len(problem.deviceProfile.devices)
      for i in range(0,len(problem.deviceProfile.devices)):
        s += "  sprintf(deviceProfile.devices[%u].name, \"%s\" );\n" % (i, problem.deviceProfile.devices[i].name)
      s += "\n"


      # problem.tensorC
      s += "\n  /* tensorC */\n"
      s += "  CobaltTensor tensorC = cobaltCreateEmptyTensor();\n"
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
      s += "\n  /* tensorA */\n"
      s += "  CobaltTensor tensorA = cobaltCreateEmptyTensor();\n"
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
      s += "\n  /* tensorB */\n"
      s += "  CobaltTensor tensorB = cobaltCreateEmptyTensor();\n"
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
      s += "\n  /* operation */\n"
      s += "  CobaltOperationType operationType = " \
          + problem.operation.type.getLibString() + ";\n"
      s += "  CobaltDataType alphaType = " \
          + problem.operation.alphaType.getLibString() + ";\n"
      s += "  CobaltDataType betaType = " \
          + problem.operation.betaType.getLibString() + ";\n"
      s += "  bool useOffsets = " \
          + ("true" if problem.operation.useOffsets else "false") + ";\n"
      numIndicesA = len(problem.operation.indexAssignmentsA)
      s += "  std::vector<unsigned int> indexAssignmentsA("+str(numIndicesA)+");\n"
      for i in range(0,numIndicesA):
        s += "  indexAssignmentsA[" + str(i) + "] = " \
            + str(problem.operation.indexAssignmentsA[i]) + ";\n"
      numIndicesB = len(problem.operation.indexAssignmentsB)
      s += "  std::vector<unsigned int> indexAssignmentsB("+str(numIndicesB)+");\n"
      for i in range(0,numIndicesB):
        s += "  indexAssignmentsB[" + str(i) + "] = " \
            + str(problem.operation.indexAssignmentsB[i]) + ";\n"
      s += "\n"
      s += "  CobaltStatus status;\n"
      # store problem
      s += "  *problem = cobaltCreateProblem(\n"
      s += "      tensorC,\n"
      s += "      tensorA,\n"
      s += "      tensorB,\n"
      s += "      &indexAssignmentsA[0],\n"
      s += "      &indexAssignmentsB[0],\n"
      s += "      operationType,\n"
      s += "      alphaType,\n"
      s += "      betaType,\n"
      s += "      useOffsets,\n"
      s += "      deviceProfile,\n"
      s += "      &status);\n"
      s += "\n"

      for i in range(0,numSolutions):
        s += "  solutionCandidates->push_back( new Cobalt::" \
            + self.solutionWriter.getName(solutionList[i])+self.solutionWriter.getTemplateArgList(solutionList[i])+"( *((*problem)->pimpl) ) ); // " \
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
      for dimension in problem.tensorA.dimensions:
        tensorSizeDimA = dimension.stride * dimension.size \
            * problem.tensorA.dataType.numBytes()
        if tensorSizeDimA > tensorSizeMaxA:
          tensorSizeMaxA = tensorSizeDimA
          #print "tensorSizeMaxA = " + str(tensorSizeMaxA)
      for dimension in problem.tensorB.dimensions:
        tensorSizeDimB = dimension.stride * dimension.size \
            * problem.tensorB.dataType.numBytes()
        if tensorSizeDimB > tensorSizeMaxB:
          tensorSizeMaxB = tensorSizeDimB
          #print "tensorSizeMaxB = " + str(tensorSizeMaxB)

      s += "}\n"
      problemSourceFile.write(s)
      problemSourceFile.close()

      # problem header
      h = "" # header file string
      h += "#ifndef " + problemFileNameBase.upper() + "_H\n"
      h += "#define " + problemFileNameBase.upper() + "_H\n"
      h += "\n"
      h += "#include \"Cobalt.h\"\n"
      h += "#include \"Solution.h\"\n"
      h += "#include <vector>\n"
      h += "\n"
      for i in range(0,numSolutions):
        h += "#include \""+ self.solutionWriter.getName(solutionList[i]) + ".h\"\n"
      h += "\n"
      h += "void init_" + problemName + "_candidates(CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates);\n"
      h += "\n"
      h += "#endif\n"
      problemHeaderFile.write(h)
      problemHeaderFile.close()

    ########################################
    # top level benchmark file
    benchmarkSourcePath = self.outputPath + self.otherSubdirectory \
      + "CobaltSolutionCandidates.cpp"
    benchmarkSourceFile = open(benchmarkSourcePath, "w")
    s = ""
    s += "#include \"CobaltSolutionCandidates.h\"\n"
    s += "#include <cstdio>\n"
    s += "\n"
    # include candidates
    for problemIdx in range(0,numProblems):
      problemSolutionPair = problemSolutionCandidates[problemIdx]
      problem = problemSolutionPair[0]
      problemName = str(problem)
      problemFileNameBase = "init_" + problemName + "_candidates"
      s += "#include \"" + problemFileNameBase + ".h\"\n"
    # init function
    s += "\n"
    s += "void initializeSolutionCandidates(CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates, size_t problemIndex) {\n"
    s += "  switch( problemIndex ) {\n"
    for problemIdx in range(0,numProblems):
      problemSolutionPair = problemSolutionCandidates[problemIdx]
      problem = problemSolutionPair[0]
      problemName = str(problem)
      s += "  case " + str(problemIdx) + ":\n"
      s += "    init_" + problemName + "_candidates(problem, solutionCandidates);\n"
      s += "    break;\n"

    s += "  default:\n"
    s += "    printf(\"Oops: index too large.\\n\");\n"
    s += "  }\n"
    s += "}\n"
    benchmarkSourceFile.write(s)
    benchmarkSourceFile.close()

    ###########################################
    # top level benchmark header file
    benchmarkHeaderPath = self.outputPath + self.otherSubdirectory \
        + "CobaltSolutionCandidates.h"
    benchmarkHeaderFile = open(benchmarkHeaderPath, "w")
    h = "#ifndef COBALT_SOLUTION_CANDIDATES_H\n"
    h += "#define COBALT_SOLUTION_CANDIDATES_H\n"
    h += "#include \"Cobalt.h\"\n"
    h += "#include \"Solution.h\"\n"
    h += "#include \"CobaltSolutions.h\"\n"
    h += "#include <vector>\n"
    #if self.backend.isOpenCL():
    #  h += "#include \"CL/cl.h\"\n"
    
    h += "\n"
    h += "const size_t numProblems = " + str(numProblems) + ";\n"
    h += "const size_t tensorSizeMaxC = " + str(tensorSizeMaxC) + ";\n"
    h += "const size_t tensorSizeMaxA = " + str(tensorSizeMaxA) + ";\n"
    h += "const size_t tensorSizeMaxB = " + str(tensorSizeMaxB) + ";\n"
    h += "\n"
    h += "void initializeSolutionCandidates(CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates, size_t problemIndex);\n"
    h += "\n"
    h += "#endif\n"
    h += "\n"
    benchmarkHeaderFile.write(h)
    benchmarkHeaderFile.close()

    # explicit template instantiation
    templateInstantiationsPath = self.outputPath + self.solutionSubdirectory \
        + "SolutionTemplateInstantiations.inl"
    templateInstantiationsFile = open(templateInstantiationsPath, "w")
    templateInstantiationsFile.write("/* explicit template instantiations for base classes of generated solutions */\n\n")
    for templateInstantiationStr in templateInstantiationSet:
      templateInstantiationsFile.write("template class Cobalt::SolutionGPU" \
          +templateInstantiationStr + ";\n")
      if self.backend.isOpenCL():
        templateInstantiationsFile.write(
            "template class Cobalt::SolutionOpenCL" \
            +templateInstantiationStr + ";\n")
      else:
        templateInstantiationsFile.write(
            "template class Cobalt::SolutionHIP" \
            +templateInstantiationStr + ";\n")


    # write CobaltBenchmark.cmake
    benchmarkCMakePath = self.outputPath + self.otherSubdirectory \
        + "CobaltBenchmark.cmake"
    benchmarkCMakeFile = open(benchmarkCMakePath, "w")
    s = "# CobaltBenchmark.cmake\n"
    s += "\n"
    s += "include( ${CobaltBenchmark_KernelFiles_CMAKE_DYNAMIC} )\n"
    s += "include( ${CobaltBenchmark_SolutionFiles_CMAKE_DYNAMIC} )\n"
    s += "\n"
    s += "set( CobaltBenchmark_SRC_GENERATED_DYNAMIC\n"
    for problemIdx in range(0,numProblems):
      problemSolutionPair = problemSolutionCandidates[problemIdx]
      problem = problemSolutionPair[0]
      problemName = str(problem)
      problemFileNameBase = "init_" + problemName + "_candidates"
      s += "  ${CobaltBenchmark_DIR_GENERATED}" + self.otherSubdirectory + problemFileNameBase + ".cpp\n"
      s += "  ${CobaltBenchmark_DIR_GENERATED}" + self.otherSubdirectory + problemFileNameBase + ".h\n"
    s += ")\n"
    s += "\n"
    s += "source_group(CobaltGen\\\\Benchmark FILES\n"
    s += "  ${CobaltBenchmark_SRC_GENERATED_STATIC}\n"
    s += "  ${CobaltBenchmark_SRC_GENERATED_DYNAMIC} )\n"
    s += "\n"
    benchmarkCMakeFile.write(s)
    benchmarkCMakeFile.close()



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
    #fileString += "cl_kernel " + kernelName + "_kernel = nullptr;\n"

    # backend pre
    fileString += "\n"
    if self.backend.isOpenCL():
      fileString += "const char * const %s_src =\"" % (kernelName)

    # write kernel body
    fileString += self.kernelWriter.getBody( kernel )

    # backend post
    if self.backend.isOpenCL():
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
    if self.backend.isHIP():
      fileString += "#include <hip_runtime.h>\n"
      fileString += "\n"
    if self.backend.isOpenCL():
      fileString += "extern const char * const %s_src;\n" % kernelName
    else:
      fileString += self.kernelWriter.getSignature(kernel)
      fileString += ";\n"

    fileString += "#endif\n"
    return fileString


  ##############################################################################
  # write backend files
  ##############################################################################
  def writeBackendFiles( self, psTimes ):
    print "status: writing backend files"
     # (1) Write Top-Level Solution Selection files
    sslw = SolutionSelectionWriter.SolutionSelectionWriter(psTimes, self.backend)
    baseName = "CobaltGetSolution"
    sslSourcePath = self.outputPath + self.otherSubdirectory + baseName + ".cpp"
    sslSourceFile = open(sslSourcePath, "w")
    sslHeaderPath = self.outputPath + self.otherSubdirectory + baseName + ".h"
    sslHeaderFile = open(sslHeaderPath, "w")
    sslSourceString, sslHeaderString = sslw.writeGetSolutionTop() # match device
    sslSourceFile.write(sslSourceString)
    sslSourceFile.close()
    sslHeaderFile.write(sslHeaderString)
    sslHeaderFile.close()

    for deviceProfile, exactMatches in psTimes.iteritems():
      # print str(deviceProfile), str(exactMatches)
      # (2) Write Device-Level Solution Selection files
      baseName = "CobaltGetSolution_" + deviceProfile.libString()
      sslSourcePath = self.outputPath + self.otherSubdirectory + baseName + ".cpp"
      sslSourceFile = open(sslSourcePath, "w")
      sslHeaderPath = self.outputPath + self.otherSubdirectory + baseName + ".h"
      sslHeaderFile = open(sslHeaderPath, "w")
      sslSourceString, sslHeaderString = sslw.writeGetSolutionForDevice(deviceProfile, exactMatches) # match exact
      sslSourceFile.write(sslSourceString)
      sslSourceFile.close()
      sslHeaderFile.write(sslHeaderString)
      sslHeaderFile.close()

      for exactMatch, problemSolutionPairs in exactMatches.iteritems():
        baseName = "CobaltGetSolution_" + exactMatch.libString()

        # (7) Write CSV for verification
        #print "Writing CSV for %s" % baseName
        #csvPath = self.outputPath + self.otherSubdirectory + baseName + "_SolutionSpeeds.csv"
        #csvFile = open(csvPath, "w")
        #s = sslw.writePSPsToCSV(exactMatch, problemSolutionPairs)
        #csvFile.write(s)
        #csvFile.close()



        # (3) Write Exact-Match-Level Solution Selection files
        sslSourcePath = self.outputPath + self.otherSubdirectory + baseName + ".cpp"
        sslSourceFile = open(sslSourcePath, "w")
        sslHeaderPath = self.outputPath + self.otherSubdirectory + baseName + ".h"
        sslHeaderFile = open(sslHeaderPath, "w")
        sslSourceString, sslHeaderString = sslw.writeGetSolutionForExactMatch(exactMatch, problemSolutionPairs) # match size and mod
        sslSourceFile.write(sslSourceString)
        sslSourceFile.close()
        sslHeaderFile.write(sslHeaderString)
        sslHeaderFile.close()



    # (4) Write Kernel Files
    self.writeKernelFiles(sslw.getKernelSet())

    # (5) Write Solution Files
    self.writeSolutionFiles(sslw.getSolutionSet())

    # (6) Write CMake File
    backendCMakePath = self.outputPath + self.otherSubdirectory + "CobaltLib.cmake"
    backendCMakeFile = open(backendCMakePath, "w")
    s = sslw.writeCobaltLibCMake(self.otherSubdirectory)
    backendCMakeFile.write(s)
    backendCMakeFile.close()
