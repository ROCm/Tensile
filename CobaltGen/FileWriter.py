################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

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
  minimumXMLSubdirectory = "/MinimumXMLs/"

  cmakeHeader =  "################################################################################\n" + "# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.\n" + "################################################################################\n\n"
  cHeader = "/*******************************************************************************\n" + " * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.\n" + " ******************************************************************************/\n\n"

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
    self.ensurePath( self.outputPath + self.minimumXMLSubdirectory )

    self.forBenchmark = forBenchmark
    self.cobaltDirGenerated = "${CobaltLib_DIR_GENERATED}"
    if self.forBenchmark:
      self.cobaltDirGenerated = "${CobaltBenchmark_DIR_GENERATED}"



  ##############################################################################
  # writeKernelFiles
  ##############################################################################
  def writeKernelFiles( self, kernelSet ):
    print "CobaltGen: Writing kernel files."

    # main kernel .cpp,.h files
    kernelFilePath = self.cobaltDirGenerated + self.kernelSubdirectory
    kernelsCMakeFilePath = self.outputPath + self.kernelSubdirectory \
        + "CobaltKernels.cmake"
    allKernelsHeaderFilePath = self.outputPath + self.kernelSubdirectory \
        + "CobaltKernels.h"
    kernelsCMakeFile = open(kernelsCMakeFilePath, "w")
    kernelsCMakeFile.write(self.cmakeHeader)
    allKernelsHeaderFile = open(allKernelsHeaderFilePath, "w")
    allKernelsHeaderFile.write(self.cHeader)

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
      kernelSourceFile.write(self.cHeader)
      kernelHeaderFile = open(kernelHeaderFilePath, "w")
      kernelHeaderFile.write(self.cHeader)

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
      kernelsCMakeFile.write("source_group(CobaltGen\\\\Kernels FILES ${CobaltLib_KernelFiles_GENERATED_DYNAMIC} )\n")
    kernelsCMakeFile.close()
    allKernelsHeaderFile.close()


  ##############################################################################
  # writeSolutionFiles
  ##############################################################################
  def writeSolutionFiles( self, solutionSet ):
    print "CobaltGen: Writing solution files."

    # main solution .cpp,.h files
    solutionFilePath = self.cobaltDirGenerated + self.solutionSubdirectory
    solutionsCMakeFilePath = self.outputPath + self.solutionSubdirectory \
        + "CobaltSolutions.cmake"
    allSolutionsHeaderFilePath = self.outputPath + self.solutionSubdirectory \
        + "CobaltSolutions.h"
    solutionsCMakeFile = open(solutionsCMakeFilePath, "w")
    solutionsCMakeFile.write(self.cmakeHeader)
    allSolutionsHeaderFile = open(allSolutionsHeaderFilePath, "w")
    allSolutionsHeaderFile.write(self.cHeader)

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
      solutionSourceFile.write(self.cHeader)
      solutionHeaderFile = open(solutionHeaderFilePath, "w")
      solutionHeaderFile.write(self.cHeader)

      # get solution file string
      solutionSourceFileString = self.solutionWriter.getSourceString( solution )
      solutionHeaderFileString = self.solutionWriter.getHeaderString( solution )

      # write solution file string to file
      solutionSourceFile.write( solutionSourceFileString )
      solutionHeaderFile.write( solutionHeaderFileString )
      solutionSourceFile.close()
      solutionHeaderFile.close()
      # print "Wrote: " + solutionName

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
  def writeBenchmarkFiles( self, problemTree, problemSolutionCandidates ):
    print "CobaltGen: Writing benchmark files."

    numSolutions = 0
    benchmarkNumExactMatches = 0
    benchmarkExactMatchNames = []
    benchmarkExactMatchNumProblems = []

    templateInstantiationSet = set()
    solutionStartIdx = 0
    tensorSizeMaxC = 0
    tensorSizeMaxA = 0
    tensorSizeMaxB = 0
    solutionEndIdx = -1

    for deviceProfile, exactMatches in problemTree.iteritems():
      for exactMatch, problemSet in exactMatches.iteritems():
        print "ExactMatch: " + str(exactMatch)

        benchmarkExactMatchNames.append(str(exactMatch))
        benchmarkExactMatchNumProblems.append(len(problemSet))
        benchmarkNumExactMatches += 1

        problemList = list(problemSet)
        # initializeSolutionCandidates(&problem, &solutionCandidates, exactMatchIdx, problemIdx);

        exactMatchName = str(exactMatch)
        exactMatchFileNameBase = "init_" + exactMatchName + "_candidates"
        exactMatchSourcePath = self.outputPath + self.otherSubdirectory \
            + exactMatchFileNameBase + ".cpp"
        exactMatchSourceFile = open(exactMatchSourcePath, "w")
        exactMatchSourceFile.write(self.cHeader)
        exactMatchHeaderPath = self.outputPath + self.otherSubdirectory \
            + exactMatchFileNameBase + ".h"
        exactMatchHeaderFile = open(exactMatchHeaderPath, "w")
        exactMatchHeaderFile.write(self.cHeader)

        s = "" # source file string
        s += "#include \"" + exactMatchFileNameBase + ".h\"\n"
        s += "\n"
        s += "/* ExactMatch: " + exactMatchName + " */\n"
        s += "\n"

        # initializeSolutionCandidates
        s += "void init_" + exactMatchName + "_candidates(CobaltDeviceProfile & deviceProfile, CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates, size_t problemIndex) {\n"
        s += "  switch( problemIndex ) {\n"
        for problemIdx in range(0,len(problemList)):
          problem = problemList[problemIdx]
          problemName = str(problem)
          s += "  case " + str(problemIdx) + ": init_" + problemName + "_candidates(deviceProfile, problem, solutionCandidates); break;\n"
        s += "  default: printf(\"Oops: index too large.\\n\");\n"
        s += "  }\n"
        s += "}\n"
        exactMatchSourceFile.write(s)
        exactMatchSourceFile.close()


        h = "" # header file string
        h += "#ifndef " + exactMatchFileNameBase.upper() + "_H\n"
        h += "#define " + exactMatchFileNameBase.upper() + "_H\n"
        h += "\n"
        h += "#include \"Cobalt.h\"\n"
        h += "#include \"Solution.h\"\n"
        h += "#include <vector>\n"
        h += "\n"
        for i in range(0,len(problemList)):
          problemFileNameBase = "init_" + str(problemList[i]) + "_candidates"
          h += "#include \""+ problemFileNameBase + ".h\"\n"
        h += "\n"
        h += "void init_" + exactMatchName + "_candidates(CobaltDeviceProfile & deviceProfile, CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates, size_t problemIndex);\n"
        h += "\n"
        h += "#endif\n"
        exactMatchHeaderFile.write(h)
        exactMatchHeaderFile.close()

        # for problems belonging to exactMatch
        for problemIdx in range(0,len(problemList)):
          problem = problemList[problemIdx]

          solutionSet = problemSolutionCandidates[problem]

          problemName = str(problem)
          # open problem file
          problemFileNameBase = "init_" + problemName + "_candidates"
          problemSourcePath = self.outputPath + self.otherSubdirectory \
              + problemFileNameBase + ".cpp"
          problemSourceFile = open(problemSourcePath, "w")
          problemSourceFile.write(self.cHeader)
          problemHeaderPath = self.outputPath + self.otherSubdirectory \
              + problemFileNameBase + ".h"
          problemHeaderFile = open(problemHeaderPath, "w")
          problemHeaderFile.write(self.cHeader)

          s = "" # source file string
          s += "#include \"" + problemFileNameBase + ".h\"\n"
          s += "\n"
          s += "/* problem " + str(problemIdx) + "/" + str(len(problemList)) + ": " + problemName + " */\n"
          s += "\n"

          # initializeSolutionCandidates
          s += "void init_" + problemName + "_candidates(CobaltDeviceProfile & deviceProfile, CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates) {\n"

          # CobaltDeviceProfile &
          # s += "\n"
          # s += "  CobaltDeviceProfile deviceProfile = cobaltCreateEmptyDeviceProfile();\n"
          # s += "  deviceProfile.numDevices = %u;\n" % len(problem.deviceProfile.devices)
          # for i in range(0,len(problem.deviceProfile.devices)):
          #   s += "  sprintf(deviceProfile.devices[%u].name, \"%s\" );\n" % (i, problem.deviceProfile.devices[i].name)
          # s += "\n"


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
          # store problem
          s += "  CobaltStatus status = cobaltCreateProblem(\n"
          s += "      problem,\n"
          s += "      tensorC,\n"
          s += "      tensorA,\n"
          s += "      tensorB,\n"
          s += "      &indexAssignmentsA[0],\n"
          s += "      &indexAssignmentsB[0],\n"
          s += "      operationType,\n"
          s += "      alphaType,\n"
          s += "      betaType,\n"
          s += "      useOffsets,\n"
          s += "      deviceProfile);\n"
          s += "\n"

          idx = 0
          numSolutions = len(solutionSet)
          for solution in solutionSet:
            s += "  solutionCandidates->push_back( new Cobalt::" \
                + self.solutionWriter.getName(solution)+self.solutionWriter.getTemplateArgList(solution)+"( *((*problem)->pimpl) ) ); // " \
                + str(idx) + "/" + str(numSolutions) + "\n"
            templateInstantiationSet.add(self.solutionWriter.getTemplateArgList(solution))
            idx += 1
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
          for solution in solutionSet:
            h += "#include \""+ self.solutionWriter.getName(solution) + ".h\"\n"
          h += "\n"
          h += "void init_" + problemName + "_candidates(CobaltDeviceProfile & deviceProfile, CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates);\n"
          h += "\n"
          h += "#endif\n"
          problemHeaderFile.write(h)
          problemHeaderFile.close()

    ########################################
    # top level benchmark file
    benchmarkSourcePath = self.outputPath + self.otherSubdirectory \
      + "CobaltSolutionCandidates.cpp"
    benchmarkSourceFile = open(benchmarkSourcePath, "w")
    benchmarkSourceFile.write(self.cHeader)
    s = ""
    s += "#include \"CobaltSolutionCandidates.h\"\n"
    s += "#include <cstdio>\n"
    s += "\n"
    # include candidates

    for deviceProfile, exactMatches in problemTree.iteritems():
      for exactMatch, problemSet in exactMatches.iteritems():
        exactMatchName = str(exactMatch)
        exactMatchFileNameBase = "init_" + exactMatchName + "_candidates"
        s += "#include \"" + exactMatchFileNameBase + ".h\"\n"
    # init function
    s += "\n"
    s += "void initializeSolutionCandidates(CobaltDeviceProfile & deviceProfile, CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates, size_t exactMatchIndex, size_t problemIndex) {\n"
    s += "  switch( exactMatchIndex ) {\n"
    exactMatchIdx = 0
    for deviceProfile, exactMatches in problemTree.iteritems():
      for exactMatch, problemSet in exactMatches.iteritems():
        exactMatchName = str(exactMatch)
        s += "  case " + str(exactMatchIdx) + ": init_" + exactMatchName + "_candidates(deviceProfile, problem, solutionCandidates, problemIndex); break;\n"
        exactMatchIdx += 1
    s += "  default: printf(\"Oops: index too large.\\n\");\n"
    s += "  }\n"
    s += "}\n"
    benchmarkSourceFile.write(s)
    benchmarkSourceFile.close()

    ###########################################
    # top level benchmark header file
    benchmarkHeaderPath = self.outputPath + self.otherSubdirectory \
        + "CobaltSolutionCandidates.h"
    benchmarkHeaderFile = open(benchmarkHeaderPath, "w")
    benchmarkHeaderFile.write(self.cHeader)
    h = "#ifndef COBALT_SOLUTION_CANDIDATES_H\n"
    h += "#define COBALT_SOLUTION_CANDIDATES_H\n"
    h += "#include \"Cobalt.h\"\n"
    h += "#include \"Solution.h\"\n"
    h += "#include \"CobaltSolutions.h\"\n"
    h += "#include <vector>\n"
    #if self.backend.isOpenCL():
    #  h += "#include \"CL/cl.h\"\n"

    h += "\n"
    h += "static const size_t tensorSizeMaxC = " + str(tensorSizeMaxC) + ";\n"
    h += "static const size_t tensorSizeMaxA = " + str(tensorSizeMaxA) + ";\n"
    h += "static const size_t tensorSizeMaxB = " + str(tensorSizeMaxB) + ";\n"
    h += "static const size_t benchmarkNumExactMatches = " + str(benchmarkNumExactMatches) + ";\n"
    h += "static const char *benchmarkExactMatchNames[] = {\n"
    h += "    \"" + benchmarkExactMatchNames[0]
    for i in range(1, len(benchmarkExactMatchNames)):
      h += "\",\n    \"" + benchmarkExactMatchNames[i]
    h += "\"\n    };\n"

    h += "static const size_t benchmarkExactMatchNumProblems[] = {\n"
    h += "    " + str(benchmarkExactMatchNumProblems[0])
    for i in range(1, len(benchmarkExactMatchNumProblems)):
      h += ",\n    " + str(benchmarkExactMatchNumProblems[i])
    h += "\n    };\n"


    # write device profile
    for deviceProfile, exactMatches in problemTree.iteritems():
      for exactMatch, problemSet in exactMatches.iteritems():
        for p in problemSet:
          problem = p
          break
        break
      break
    problem = problemList[0]
    dp = problem.deviceProfile
    h += "\n"
    h += "static const char *benchmarkDeviceName = \"" + dp.devices[0].name + "\";\n"
    h += "static const unsigned int benchmarkDeviceNumComputeUnits = " + str(dp.devices[0].numComputeUnits) + ";\n"
    h += "static const unsigned int benchmarkDeviceClockFrequency = " + str(dp.devices[0].clockFrequency) + ";\n"
    h += "static const unsigned int benchmarkDeviceFlopsPerClock = " + str(dp.devices[0].flopsPerClock) + ";\n"

    h += "\n"
    h += "void initializeSolutionCandidates(CobaltDeviceProfile & deviceProfile, CobaltProblem * problem, std::vector<Cobalt::Solution *> *solutionCandidates, size_t exactMatchIndex, size_t problemIndex);\n"
    h += "\n"
    h += "#endif\n"
    h += "\n"
    benchmarkHeaderFile.write(h)
    benchmarkHeaderFile.close()

    self.writeTemplateInstantiations(templateInstantiationSet)

    # write CobaltBenchmark.cmake
    benchmarkCMakePath = self.outputPath + self.otherSubdirectory \
        + "CobaltBenchmark.cmake"
    benchmarkCMakeFile = open(benchmarkCMakePath, "w")
    benchmarkCMakeFile.write(self.cmakeHeader)
    s = "# CobaltBenchmark.cmake\n"
    s += "\n"
    s += "include( ${CobaltBenchmark_KernelFiles_CMAKE_DYNAMIC} )\n"
    s += "include( ${CobaltBenchmark_SolutionFiles_CMAKE_DYNAMIC} )\n"
    s += "\n"
    s += "set( CobaltBenchmark_SRC_GENERATED_DYNAMIC\n"
    for deviceProfile, exactMatches in problemTree.iteritems():
      for exactMatch, problemSet in exactMatches.iteritems():
        exactMatchName = str(exactMatch)
        exactMatchFileNameBase = "init_" + exactMatchName + "_candidates"
        s += "  ${CobaltBenchmark_DIR_GENERATED}" + self.otherSubdirectory + exactMatchFileNameBase + ".cpp\n"
        s += "  ${CobaltBenchmark_DIR_GENERATED}" + self.otherSubdirectory + exactMatchFileNameBase + ".h\n"
        for problem in problemSet:
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
      fileString += "#include <hip/hip_runtime.h>\n"
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
    #print "status: writing backend files"
     # (1) Write Top-Level Solution Selection files
    sslw = SolutionSelectionWriter.SolutionSelectionWriter(psTimes, self.backend)
    baseName = "CobaltGetSolution"
    sslSourcePath = self.outputPath + self.otherSubdirectory + baseName + ".cpp"
    sslSourceFile = open(sslSourcePath, "w")
    sslSourceFile.write(self.cHeader)
    sslHeaderPath = self.outputPath + self.otherSubdirectory + baseName + ".h"
    sslHeaderFile = open(sslHeaderPath, "w")
    sslHeaderFile.write(self.cHeader)
    sslSourceString, sslHeaderString = sslw.writeGetSolutionTop() # match device
    sslSourceFile.write(sslSourceString)
    sslSourceFile.close()
    sslHeaderFile.write(sslHeaderString)
    sslHeaderFile.close()

    templateInstantiationSet = set()

    for deviceProfile, exactMatches in psTimes.iteritems():
      #print str(deviceProfile)
      # (2) Write Device-Level Solution Selection files
      baseName = "CobaltGetSolution_" + deviceProfile.libString()
      sslSourcePath = self.outputPath + self.otherSubdirectory + baseName + ".cpp"
      sslSourceFile = open(sslSourcePath, "w")
      sslSourceFile.write(self.cHeader)
      sslHeaderPath = self.outputPath + self.otherSubdirectory + baseName + ".h"
      sslHeaderFile = open(sslHeaderPath, "w")
      sslHeaderFile.write(self.cHeader)
      sslSourceString, sslHeaderString = sslw.writeGetSolutionForDevice(deviceProfile, exactMatches) # match exact
      sslSourceFile.write(sslSourceString)
      sslSourceFile.close()
      sslHeaderFile.write(sslHeaderString)
      sslHeaderFile.close()

      for exactMatch, pspTypes in exactMatches.iteritems():
        rangePSPs = pspTypes[0]
        exactPSPs = pspTypes[1]

        rangePSPs = sslw.bucketSortRangePSPs( rangePSPs ) # sort into size groups

        numExactTiles = 0
        numFallbacks = 0
        for sizeGroup in rangePSPs:
          numExactTiles += len(sizeGroup[0])
          numFallbacks += len(sizeGroup[1])
        print "CobaltGen: Writing backend for %s(%u,%u,%u)." % (str(exactMatch), numExactTiles, numFallbacks, len(exactPSPs))
        # only support this exact match if some benchmark times existed
        # otherwise none of the other files for it will have been written

        baseName = "CobaltGetSolution_" + exactMatch.libString()

        # (7) Write CSV for verification
        if sslw.printStatus: print "%s::writeCSV(%s)" % (str(exactMatch), baseName)
        csvPath = self.outputPath + self.otherSubdirectory + baseName + "_perf.csv"
        csvFile = open(csvPath, "w")
        s = sslw.writePSPsToCSV(exactMatch, rangePSPs, exactPSPs)
        csvFile.write(s)
        csvFile.close()



        # (3) Write Exact-Match-Level Solution Selection files
        sslSourcePath = self.outputPath + self.otherSubdirectory + baseName + ".cpp"
        sslSourceFile = open(sslSourcePath, "w")
        sslSourceFile.write(self.cHeader)
        sslHeaderPath = self.outputPath + self.otherSubdirectory + baseName + ".h"
        sslHeaderFile = open(sslHeaderPath, "w")
        sslHeaderFile.write(self.cHeader)
        #print "calling writeGetSolutionForExactMatch"
        sslw.pruneSolutions(exactMatch, rangePSPs, exactPSPs)
        sslSourceString, sslHeaderString, fastestPSPs = sslw.writeGetSolutionForExactMatch(exactMatch, rangePSPs, exactPSPs) # match size and mod
        if sslw.printStatus: print "%s::writeMinimumXML()" % str(exactMatch)
        self.writeMinimumXML( exactMatch, fastestPSPs )
        sslSourceFile.write(sslSourceString)
        sslSourceFile.close()
        sslHeaderFile.write(sslHeaderString)
        sslHeaderFile.close()



    # (4) Write Kernel Files
    self.writeKernelFiles(sslw.getKernelSet())

    # add solutions to template
    for solution in sslw.getSolutionSet():
      templateInstantiationSet.add(self.solutionWriter.getTemplateArgList(solution))

    # (5) Write Solution Files
    self.writeSolutionFiles(sslw.getSolutionSet())

    # (6) Write CMake File
    backendCMakePath = self.outputPath + self.otherSubdirectory + "CobaltLib.cmake"
    backendCMakeFile = open(backendCMakePath, "w")
    backendCMakeFile.write(self.cmakeHeader)
    s = sslw.writeCobaltLibCMake(self.otherSubdirectory)
    backendCMakeFile.write(s)
    backendCMakeFile.close()
    self.writeTemplateInstantiations(templateInstantiationSet)

  def writeTemplateInstantiations( self, templateInstantiationSet ):
    # explicit template instantiation
    templateInstantiationsPath = self.outputPath + self.solutionSubdirectory \
        + "SolutionTemplateInstantiations.inl"
    templateInstantiationsFile = open(templateInstantiationsPath, "w")
    templateInstantiationsFile.write(self.cHeader)
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
    print "CobaltGen: Writing explicit template instantiations."
    templateInstantiationsFile.close()

################################################################################
# write the minimum set of xml entries requried to reproduce library backend
################################################################################
  def writeMinimumXML( self, exactMatch, fastestPSPs ):
    minXMLPath = self.outputPath + self.minimumXMLSubdirectory \
        + str(exactMatch) + ".xml"
    minXMLFile= open(minXMLPath, "w")

    s = ""
    s += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    s += "<CobaltLog>\n"

    for psp in fastestPSPs:
      problem = psp[0]
      solution = psp[1]
      time = psp[2]
      s += " <TE>\n"
      s += "  <S>\n"
      s += "   <P>\n"
      # tensorC
      s += "    <TC t=\"%u\" n=\"%u\" " % (problem.tensorC.dataType.value, len(problem.tensorC.dimensions) )
      for i in range(0, len(problem.tensorC.dimensions)):
        dim = problem.tensorC.dimensions[i]
        s += "st%u=\"%u\" sz%u=\"%u\" " % (i, dim.stride, i, dim.size)
      s += "/>\n"
      # tensorA
      s += "    <TA t=\"%u\" n=\"%u\" " % (problem.tensorA.dataType.value, len(problem.tensorA.dimensions) )
      for i in range(0, len(problem.tensorA.dimensions)):
        dim = problem.tensorA.dimensions[i]
        s += "st%u=\"%u\" sz%u=\"%u\" " % (i, dim.stride, i, dim.size)
      s += "/>\n"
      # tensorB
      s += "    <TB t=\"%u\" n=\"%u\" " % (problem.tensorB.dataType.value, len(problem.tensorB.dimensions) )
      for i in range(0, len(problem.tensorB.dimensions)):
        dim = problem.tensorB.dimensions[i]
        s += "st%u=\"%u\" sz%u=\"%u\" " % (i, dim.stride, i, dim.size)
      s += "/>\n"
      # operation
      s += "    <O t=\"%u\" a=\"%u\" b=\"%u\" o=\"%u\" nF=\"%u\" nB=\"%u\" nS=\"%u\" >\n" % ( \
          problem.operation.type.value, \
          problem.operation.alphaType.value, \
          problem.operation.betaType.value, \
          problem.operation.useOffsets, \
          problem.operation.numIndicesFree, \
          problem.operation.numIndicesBatch, \
          problem.operation.numIndicesSummation )
      # indexAssignmentsA
      s += "     <IA n=\"%u\" " % ( len(problem.operation.indexAssignmentsA) )
      for i in range(0, len(problem.operation.indexAssignmentsA)):
        ia = problem.operation.indexAssignmentsA[i]
        s += "i%u=\"%u\" " % (i, ia)
      s += "/>\n"
      # indexAssignmentsB
      s += "     <IB n=\"%u\" " % ( len(problem.operation.indexAssignmentsB) )
      for i in range(0, len(problem.operation.indexAssignmentsB)):
        ia = problem.operation.indexAssignmentsB[i]
        s += "i%u=\"%u\" " % (i, ia)
      s += "/>\n"
      s += "    </O>\n"
      # device profile
      s += "    <DP n=\"%u\" " % ( len(problem.deviceProfile.devices) )
      for i in range(0, len(problem.deviceProfile.devices)):
        device = problem.deviceProfile.devices[i]
        s += "d%u=\"%s\" CU%u=\"%u\" MHz%u=\"%u\" FPC%u=\"%u\" " % (\
            i, device.name, \
            i, device.numComputeUnits, \
            i, device.clockFrequency, \
            i, device.flopsPerClock )
      s += "/>\n"
      s += "   </P>\n"
      # implementation details
      s += "   <ID kG0=\"%u\" kG1=\"%u\" kG2=\"%u\" b0=\"%u\" b1=\"%u\" ppdO=\"%u\" ppdLS=\"%u\" ppdAll=\"%u\" >\n" % ( \
          solution.kernelGrid[0], \
          solution.kernelGrid[1], \
          solution.kernelGrid[2], \
          solution.branch[0].value, \
          solution.branch[1].value, \
          solution.ppdOffsets, \
          solution.ppdLeadingStrides, \
          solution.ppdAll )
      for i in range(0, len(solution.kernels)):
        kernel = solution.kernels[i]
        if kernel == None:
          continue
        s += "    <K i=\"%u\" wG0=\"%u\" wG1=\"%u\" mT0=\"%u\" mT1=\"%u\" b0=\"%u\" b1=\"%u\" nlpaA=\"%u\" lspaA=\"%u\" tspaA=\"%u\" nlpeA=\"%u\" lspeA=\"%u\" tspeA=\"%u\" nlpaB=\"%u\" lspaB=\"%u\" tspaB=\"%u\" nlpeB=\"%u\" lspeB=\"%u\" tspeB=\"%u\" u0=\"%u\" u1=\"%u\" />\n" % ( \
            i,
            kernel.tile.workGroup[0], \
            kernel.tile.workGroup[1], \
            kernel.tile.microTile[0], \
            kernel.tile.microTile[1], \
            kernel.tile.branch[0].value, \
            kernel.tile.branch[1].value, \
            kernel.numLoadsParaA, \
            kernel.loadSizeParaA, \
            kernel.totalLoadSizeParaA, \
            kernel.numLoadsPerpA, \
            kernel.loadSizePerpA, \
            kernel.totalLoadSizePerpA, \
            kernel.numLoadsParaB, \
            kernel.loadSizeParaB, \
            kernel.totalLoadSizeParaB, \
            kernel.numLoadsPerpB, \
            kernel.loadSizePerpB, \
            kernel.totalLoadSizePerpB, \
            kernel.unrolls[0], \
            0 if len(kernel.unrolls)<2 else kernel.unrolls[1] )
      s += "   </ID>\n"
      s += "  </S>\n"
      s += "  <B t=\"%f\" u=\"ms\" />\n" % (time)
      s += " </TE>\n"

    s += "</CobaltLog>\n"
    minXMLFile.write( s )
    minXMLFile.close()
