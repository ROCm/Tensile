from Common import globalParameters, HR, pushWorkingPath, popWorkingPath, print1, CHeader
from SolutionStructs import Solution
from SolutionWriter import SolutionWriter
from KernelWriter import KernelWriter
import YAMLIO

import os
from subprocess import Popen
from shutil import copy as shutil_copy
from shutil import rmtree


################################################################################
# Main
################################################################################
def main( config ):
  libraryLogicPath = os.path.join(globalParameters["WorkingPath"], \
      globalParameters["LibraryLogicPath"])
  pushWorkingPath(globalParameters["LibraryClientPath"])


  ##############################################################################
  # Copy Source Files
  ##############################################################################
  pushWorkingPath("source")
  filesToCopy = [
      "Client.cpp",
      "Client.h",
      "ReferenceCPU.h",
      "MathTemplates.cpp",
      "MathTemplates.h",
      "Tools.h",
      "CMakeLists.txt",
      "TensileConfig.cmake"
      ]

  for f in filesToCopy:
    filename = os.path.join(globalParameters["SourcePath"], f)
    shutil_copy(
        os.path.join(globalParameters["SourcePath"], f),
        globalParameters["WorkingPath"] )
  if globalParameters["Backend"] == "OCL":
    shutil_copy(
        os.path.join(globalParameters["SourcePath"], "FindOpenCL.cmake"),
        globalParameters["WorkingPath"] )
  else:
    shutil_copy(
        os.path.join(globalParameters["SourcePath"], "FindHIP.cmake"),
        globalParameters["WorkingPath"] )
    shutil_copy(
        os.path.join(globalParameters["SourcePath"], "FindHCC.cmake"),
        globalParameters["WorkingPath"] )

  ##############################################################################
  # Read Logic Files
  ##############################################################################
  logicFiles = [os.path.join(libraryLogicPath, f) for f \
      in os.listdir(libraryLogicPath) \
      if os.path.isfile(os.path.join(libraryLogicPath, f))]
  print1("LogicFiles: %s" % logicFiles)
  functions = []
  functionNames = []
  for logicFileName in logicFiles:
    (scheduleName, problemType, solutionsForType, skinnyLogic0, skinnyLogic1, \
        diagonalLogic) = YAMLIO.readLibraryLogicForProblemType(logicFileName)
    functions.append((scheduleName, problemType))
    functionNames.append("tensile_%s_%s" % (scheduleName, problemType))

  ##############################################################################
  # Write Generated Header
  ##############################################################################
  forBenchmark = False
  solutions = None
  problemSizes = None
  stepName = None
  writeClientParameters(forBenchmark, solutions, problemSizes, stepName, \
      functions)
  popWorkingPath() # source

  ##############################################################################
  # Run Build Script
  ##############################################################################
  # if redo=true, clobber the build directory
  if globalParameters["ForceRedoLibraryClient"]:
    rmtree(os.path.join(globalParameters["WorkingPath"], "build"), \
        ignore_errors=True)
  pushWorkingPath("build")

  # write runScript
  path = globalParameters["WorkingPath"]
  forBenchmark = False
  runScriptName = writeRunScript(path, libraryLogicPath, forBenchmark)

  # run runScript
  process = Popen(runScriptName, cwd=globalParameters["WorkingPath"])
  status = process.communicate()
  popWorkingPath() # build

  popWorkingPath() # LibraryClient


################################################################################
# Write Run Script
################################################################################
def writeRunScript(path, libraryLogicPath, forBenchmark):
  # create run.bat or run.sh which builds and runs
  runScriptName = os.path.join(path, \
    "run.%s" % ("bat" if os.name == "nt" else "sh") )
  runScriptFile = open(runScriptName, "w")
  echoLine = "@echo." if os.name == "nt" else "echo"
  if os.name != "nt":
    runScriptFile.write("#!/bin/sh\n")
  runScriptFile.write("%s & echo %s & echo # Configuring CMake for Client & echo %s\n" \
      % (echoLine, HR, HR))
  runScriptFile.write("cmake")
  # backend
  runScriptFile.write(" -DTensile_BACKEND=%s" \
      % globalParameters["Backend"])
  if forBenchmark:
    # for benchmark client
    runScriptFile.write(" -DTensile_CLIENT_BENCHMARK=ON")
  else:
    # for library client
    runScriptFile.write(" -DTensile_ROOT=%s" % os.path.join(globalParameters["ScriptPath"], "..") )
    runScriptFile.write(" -DTensile_CLIENT_BENCHMARK=OFF")
    runScriptFile.write(" -DTensile_LOGIC_PATH=%s" % libraryLogicPath)
    runScriptFile.write(" -DTensile_LIBRARY_PRINT_DEBUG=%s" \
        % ("ON" if globalParameters["LibraryPrintDebug"] else "OFF"))
    runScriptFile.write(" -DTensile_SHORT_FILE_NAMES=%s" \
        % ("ON" if globalParameters["ShortNames"] else "OFF"))
  # for both
  if os.name == "nt":
    runScriptFile.write(" -DCMAKE_GENERATOR_PLATFORM=x64")
  runScriptFile.write(" -DTensile_MERGE_FILES=%s" \
      % ("ON" if globalParameters["MergeFiles"] else "OFF"))
  runScriptFile.write(" ../source\n")
  runScriptFile.write("%s & echo %s & echo # Building Client & echo %s\n" \
      % (echoLine, HR, HR))
  runScriptFile.write("cmake --build . --config %s%s\n" \
      % (globalParameters["CMakeBuildType"], " -- -j 8" \
      if os.name != "nt" else "") )
  #if os.name != "nt":
  #  runScriptFile.write("find .\n")
  runScriptFile.write("%s & echo %s & echo # Running Client & echo %s\n" \
      % (echoLine, HR, HR))
  if os.name == "nt":
    runScriptFile.write(os.path.join(globalParameters["CMakeBuildType"], \
        "client.exe") )
  else:
    runScriptFile.write("./client")
  runScriptFile.close()
  if os.name != "nt":
    os.chmod(runScriptName, 0777)
  return runScriptName


################################################################################
# Write Generated Benchmark Parameters
################################################################################
def writeClientParameters(forBenchmark, solutions, problemSizes, stepName, \
    functionList):
  h = ""

  ##############################################################################
  # Min Naming
  ##############################################################################
  if forBenchmark:
    kernelNames = []
    kernels = []
    for solution in solutions:
      solutionKernels = solution.getKernels()
      for kernel in solutionKernels:
        if kernel not in kernels:
          kernels.append(kernel)

    solutionSerialNaming = Solution.getSerialNaming(solutions)
    kernelSerialNaming = Solution.getSerialNaming(kernels)
    solutionMinNaming = Solution.getMinNaming(solutions)
    kernelMinNaming = Solution.getMinNaming(kernels)
    solutionWriter = SolutionWriter( \
        solutionMinNaming, solutionSerialNaming, \
        kernelMinNaming, kernelSerialNaming)
    kernelWriter = KernelWriter( \
        kernelMinNaming, kernelSerialNaming)

  if forBenchmark:
    if globalParameters["MergeFiles"]:
      h += "#include \"Solutions.h\"\n"
    else:
      for solution in solutions:
        solutionName = solutionWriter.getSolutionName(solution)
        #if globalParameters["ShortNames"]:
        #  solutionFileName = \
        #      Solution.getNameSerial(solution, solutionSerialNaming)
        #else:
        #  solutionFileName = Solution.getNameMin(solution, solutionMinNaming)
        h += "#include \"" + solutionName + ".h\"\n"
    h += "\n"
  else:
    h += "#include \"Tensile.h\"\n"


  h += "typedef enum {\n"
  h += "    enum_float,\n"
  h += "    enum_double,\n"
  h += "    enum_TensileComplexFloat,\n"
  h += "    enum_TensileComplexDouble\n"
  h += "} DataTypeEnum;\n"
  h += "\n"

  h += "const char indexChars[%u] = \"%s" \
      % (len(globalParameters["IndexChars"])+1, \
      globalParameters["IndexChars"][0])
  for i in range(1, len(globalParameters["IndexChars"])):
    h += globalParameters["IndexChars"][i]
  h += "\";\n"

  h += "unsigned int functionIdx;\n"
  h += "unsigned int dataTypeIdx;\n"
  h += "unsigned int problemTypeIdx;\n"
  h += "\n"

  ##############################################################################
  # Problem Types
  ##############################################################################
  #dataTypes = []
  #problemTypes = []
  #functionSerialToDataTypeAndIdx = []
  dataTypes = []
  problemTypes = []
  problemTypesForDataType = {} # for data type
  schedulesForProblemType = {} # for problem type
  functionInfo = [] # dataTypeIdx, problemTypeIdx, idxWithinDataType, idxWithinProblemType

  if forBenchmark:
    problemType = solutions[0]["ProblemType"]
    dataType = problemType["DataType"]
    dataTypes.append(dataType)
    problemTypes.append(problemType)
    problemTypesForDataType[dataType] = [problemType]
    schedulesForProblemType[problemType] = solutions
    numProblemTypes = 1
    for solution in solutions:
      functionInfo.append([ 0, 0, 0, 0, 0, 0 ])
  else:
    for functionIdx in range(0, len(functionList)):
      function = functionList[functionIdx]
      scheduleName = function[0]
      problemType = function[1]
      dataType = problemType["DataType"]
      if dataType not in dataTypes:
        dataTypes.append(dataType)
        problemTypesForDataType[dataType] = []
      if problemType not in problemTypesForDataType[dataType]:
        problemTypesForDataType[dataType].append(problemType)
        schedulesForProblemType[problemType] = []
      schedulesForProblemType[problemType].append(scheduleName)

    # sort
    dataTypes = sorted(dataTypes)
    for dataType in dataTypes:
      problemTypesForDataType[dataType] = \
          sorted(problemTypesForDataType[dataType])
      for problemType in problemTypesForDataType[dataType]:
        schedulesForProblemType[problemType] = \
            sorted(schedulesForProblemType[problemType])

    # assign info
    functionIdxSerial = 0
    problemTypeIdxSerial = 0
    for dataTypeIdxSerial in range(0, len(dataTypes)):
      dataType = dataTypes[dataTypeIdxSerial]
      functionIdxForDataType = 0
      for problemTypeIdxForDataType in range(0, \
          len(problemTypesForDataType[dataType])):
        problemType = \
            problemTypesForDataType[dataType][problemTypeIdxForDataType]
        problemTypes.append(problemType)
        functionIdxForProblemType = 0
        for functionIdxForProblemType in range(0, \
            len(schedulesForProblemType[problemType])):
          schedule = \
              schedulesForProblemType[problemType][functionIdxForProblemType]
          functionInfo.append([ \
              dataTypeIdxSerial, \
              problemTypeIdxForDataType, \
              problemTypeIdxSerial, \
              functionIdxSerial,\
              functionIdxForDataType,\
              functionIdxForProblemType, \
              ])
          functionIdxForProblemType += 1
          functionIdxForDataType += 1
          functionIdxSerial += 1
      problemTypeIdxSerial += 1
    numProblemTypes = problemTypeIdxSerial
    numFunctions = functionIdxSerial
    h += "const unsigned int numFunctions = %u;\n" % numFunctions

  ##############################################################################
  # Data Types
  ##############################################################################
  h += "/* data types */\n"
  numDataTypes = len(dataTypes)
  h += "const unsigned int numDataTypes = %u;\n" % numDataTypes
  h += "const DataTypeEnum dataTypeEnums[numDataTypes] = { enum_%s" \
      % dataTypes[0].toCpp()
  for dataTypeIdx in range(1, numDataTypes):
    h += ", enum_%s" % dataTypes[dataTypeIdx].toCpp();
  h += " };\n"
  # bytes per elements
  h += "const unsigned int bytesPerElement[numDataTypes] = { %u" \
      % (dataTypes[0].numBytes())
  for dataTypeIdx in range(1, numDataTypes):
    dataType = dataTypes[dataTypeIdx]
    h += ", %u" % dataType.numBytes()
  h += " };\n"
  # flops per mac
  h += "const unsigned int numFlopsPerMac[numDataTypes] = { %u" \
      % (2 if dataTypes[0].isReal() else 8)
  for dataTypeIdx in range(1, numDataTypes):
    dataType = dataTypes[dataTypeIdx]
    h += ", %u" % (2 if dataType.isReal() else 8)
  h += " };\n"
  for dataTypeIdx in range(0, numDataTypes):
    h += "#define Tensile_DATA_TYPE_%s\n" \
        % dataTypes[dataTypeIdx].toCpp().upper()

  ##############################################################################
  # Problem Types
  ##############################################################################
  h += "/* problem types */\n"
  h += "const unsigned int numProblemTypes = %u;\n" % numProblemTypes
  # Num C Indices
  h += "const unsigned int numIndicesC[numProblemTypes] = { %u" \
      % problemTypes[0]["NumIndicesC"]
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    h += ", %u" % problemType["NumIndicesC"]
  h += " };\n"

  # Num AB Indices
  maxNumIndicesAB = len(problemTypes[0]["IndexAssignmentsA"])
  h += "const unsigned int numIndicesAB[numProblemTypes] = { %u" \
      % len(problemTypes[0]["IndexAssignmentsA"])
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    numIndicesAB = len(problemType["IndexAssignmentsA"])
    h += ", %u" % numIndicesAB
    maxNumIndicesAB = max(numIndicesAB, maxNumIndicesAB)
  h += " };\n"
  h += "const unsigned int maxNumIndicesAB = %u;\n" % maxNumIndicesAB
  # Index Assignments A
  h += "const unsigned int indexAssignmentsA[numProblemTypes][maxNumIndicesAB] = {\n"
  for problemTypeIdx in range(0, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    indices = problemType["IndexAssignmentsA"]
    h += "  { %u" % indices[0]
    for i in range(1, maxNumIndicesAB):
      if i < len(indices):
        h += ", %u" % indices[i]
      else:
        h += ", -1"
    if problemTypeIdx < numProblemTypes-1:
      h += " },\n"
    else:
      h += " }\n"
  h += "};\n"
  # Index Assignments B
  h += "const unsigned int indexAssignmentsB[numProblemTypes][maxNumIndicesAB] = {\n"
  for problemTypeIdx in range(0, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    indices = problemType["IndexAssignmentsB"]
    h += "  { %u" % indices[0]
    for i in range(1, maxNumIndicesAB):
      if i < len(indices):
        h += ", %u" % indices[i]
      else:
        h += ", -1"
    if problemTypeIdx < numProblemTypes-1:
      h += " },\n"
    else:
      h += " }\n"
  h += "};\n"
  # beta
  h += "bool useBeta[numProblemTypes] = { %s" \
      % ("true" if problemTypes[0]["UseBeta"] else "false")
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    h += ", %s" % ("true" if problemType["UseBeta"] else "false")
  h += " };\n"
  # Complex Conjugates
  h += "const bool complexConjugateA[numProblemTypes] = { %s" \
      % ("true" if problemTypes[0]["ComplexConjugateA"] else "false" )
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    h += ", %s" % ("true" if problemTypes[0]["ComplexConjugateA"] else "false" )
  h += " };\n"
  h += "const bool complexConjugateB[numProblemTypes] = { %s" \
      % ("true" if problemTypes[0]["ComplexConjugateB"] else "false" )
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    h += ", %s" % ("true" if problemTypes[0]["ComplexConjugateB"] else "false" )
  h += " };\n"
  h += "\n"

  if not forBenchmark:
    h += "// dataTypeIdxSerial, problemTypeIdxForDataType, problemTypeIdxSerial, functionIdxSerial, functionIdxForDataType, functionIdxForProblemType\n"
    first = True
    h += "const unsigned int functionInfo[numFunctions][6] = {\n"
    for info in functionInfo:
      h += "%s{ %u, %u, %u, %u, %u, %u }" % ("  " if first else ",\n  ", \
          info[0], info[1], info[2], info[3], info[4], info[5] )
      first = False
    h += " };\n"


  ##############################################################################
  # Problem Sizes
  ##############################################################################
  maxNumIndices = problemTypes[0]["TotalIndices"]
  if not forBenchmark:
    for problemType in problemTypes:
      maxNumIndices = max(problemType["TotalIndices"], maxNumIndices)
  h += "const unsigned int maxNumIndices = %u;\n" % maxNumIndices
  h += "const unsigned int totalIndices[numProblemTypes] = { %u" \
      % problemTypes[0]["TotalIndices"]
  for problemTypeIdx in range(1, numProblemTypes):
      h += ", %u" % problemTypes[problemTypeIdx]["TotalIndices"]
  h += " };\n"
  h += "unsigned int userSizes[maxNumIndices];\n"
  if forBenchmark:
    h += "/* problem sizes */\n"
    h += "const bool indexIsSized[maxNumIndices] = {"
    for i in range(0, problemSizes.totalIndices):
      h += " %s" % ("true" if problemSizes.indexIsSized[i] else "false")
      if i < problemSizes.totalIndices-1:
        h += ","
    h += " };\n"

    h += "const unsigned int numIndicesSized = %u;\n" \
        % len(problemSizes.indicesSized)
    h += "const unsigned int indicesSized[numIndicesSized][4] = {\n"
    h += "// { min, stride, stride_incr, max }\n"
    for i in range(0, len(problemSizes.indicesSized)):
      r = problemSizes.indicesSized[i]
      h += "  { %u, %u, %u, %u }" % (r[0], r[1], r[2], r[3])
      if i < len(problemSizes.indicesSized)-1:
        h += ","
      h += "\n"
    h += "  };\n"

    numIndicesMapped = len(problemSizes.indicesMapped)
    h += "const unsigned int numIndicesMapped = %u;\n" % numIndicesMapped
    if numIndicesMapped > 0:
      h += "#define Tensile_INDICES_MAPPED 1\n"
      h += "const unsigned int indicesMapped[numIndicesMapped] = {"
      for i in range(0, numIndicesMapped):
        h += " %u" % problemSizes.indicesMapped[i]
        if i < numIndicesMapped-1:
          h += ","
      h += " };\n"
    else:
      h += "#define Tensile_INDICES_MAPPED 0\n"
  #else:
  #  h += "const unsigned int indicesMapped[1] = { 0 }; // dummy\n"

  ##############################################################################
  # Max Problem Sizes
  ##############################################################################
  if forBenchmark:
    sizeC = 1
    sizeA = 1
    sizeB = 1
    problemType = problemTypes[0]
    for idx in range(0, problemType["NumIndicesC"]):
      sizeC *= problemSizes.indexMax[idx]
    for idx in problemType["IndexAssignmentsA"]:
      sizeA *= problemSizes.indexMax[idx]
    for idx in problemType["IndexAssignmentsB"]:
      sizeB *= problemSizes.indexMax[idx]
    h += "size_t maxSizeC = %u;\n" % (sizeC)
    h += "size_t maxSizeA = %u;\n" % (sizeA)
    h += "size_t maxSizeB = %u;\n" % (sizeB)
    h += "\n"
  else:
    h += "size_t maxSizeC;\n"
    h += "size_t maxSizeA;\n"
    h += "size_t maxSizeB;\n"
    h += "\n"

  ##############################################################################
  # Current Problem Size
  ##############################################################################
  h += "/* current problem size */\n"
  h += "unsigned int fullSizes[maxNumIndices];\n"
  if forBenchmark:
    h += "const unsigned int numProblems = %u;\n" \
        % problemSizes.totalProblemSizes
    h += "unsigned int currentSizedIndexSizes[numIndicesSized];\n"
    h += "unsigned int currentSizedIndexIncrements[numIndicesSized];\n"
  h += "\n"

  ##############################################################################
  # Solutions
  ##############################################################################
  if forBenchmark:
    h += "/* solutions */\n"
    # Problem Type Indices
    h += "const unsigned int numSolutions = %u;\n" % len(solutions)
    h += "float solutionPerf[numProblems][numSolutions]; // milliseconds\n"
    h += "\n"
    # Solution Ptrs
    h += "typedef TensileStatus (*SolutionFunctionPointer)(\n"
    argList = solutionWriter.getArgList(solutions[0])
    for i in range(0, len(argList)):
      h += "  %s%s" % (argList[i], ",\n" if i < len(argList)-1 else ");\n\n")
    h += "const SolutionFunctionPointer solutions[numSolutions] = {\n"
    for i in range(0, len(solutions)):
      solution = solutions[i]
      solutionName = solutionWriter.getSolutionName(solution)
      h += "  %s" % solutionName
      if i < len(solutions)-1:
        h += ","
      h += "\n"
    h += " };\n"
    h += "\n"
    # Solution Names
    h += "const char *solutionNames[numSolutions] = {\n"
    for i in range(0, len(solutions)):
      solution = solutions[i]
      solutionName = solutionWriter.getSolutionName(solution)
      h += "  \"%s\"" % solutionName
      if i < len(solutions)-1:
        h += ","
      h += "\n"
    h += " };\n"
    h += "\n"
  else:
    # Function Names
    functionNames = []
    for dataType in dataTypes:
      for problemType in problemTypesForDataType[dataType]:
        for scheduleName in schedulesForProblemType[problemType]:
          functionNames.append("tensile_%s_%s" % (scheduleName, problemType))
    h += "const char *functionNames[numFunctions] = {\n"
    for functionIdx in range(0, len(functionNames)):
      functionName = functionNames[functionIdx]
      h += "    \"%s\"%s\n" % (functionName, \
          "," if functionIdx < len(functionNames)-1 else "" )
    h += " };\n"

  ##############################################################################
  # Runtime Structures
  ##############################################################################
  h += "/* runtime structures */\n"
  h += "TensileStatus status;\n"
  if globalParameters["Backend"] == "OCL":
    h += "unsigned int platformIdx = %u;\n" \
        % (globalParameters["Platform"])
    h += "unsigned int deviceIdx = %u;\n" \
        % (globalParameters["Device"])
    h += "cl_platform_id platform;\n"
    h += "cl_device_id device;\n"
    h += "cl_context context;\n"
    h += "cl_command_queue stream;\n"
  else:
    h += "hipStream_t stream;\n"
    h += "int deviceIdx = %u;\n" \
        % (globalParameters["Device"])
  h += "\n"
  h += "void *deviceC;\n"
  h += "void *deviceA;\n"
  h += "void *deviceB;\n"

  ##############################################################################
  # Benchmarking and Validation Parameters
  ##############################################################################
  h += "/* benchmarking parameters */\n"
  h += "unsigned int numEnqueuesPerSync = %u;\n" \
      % (globalParameters["EnqueuesPerSync"])
  h += "unsigned int numSyncsPerBenchmark = %u;\n" \
      % (globalParameters["SyncsPerBenchmark"])
  h += "unsigned int numElementsToValidate = %s;\n" \
      % (str(globalParameters["NumElementsToValidate"]) \
      if globalParameters["NumElementsToValidate"] >= 0 \
      else "0xFFFFFFFF" )
  h += "unsigned int validationMaxToPrint = %u;\n" \
      % globalParameters["ValidationMaxToPrint"]
  h += "bool validationPrintValids = %s;\n" \
      % ("true" if globalParameters["ValidationPrintValids"] else "false")
  h += "size_t validationStride;\n"
  h += "unsigned int dataInitType = %s;\n" % globalParameters["DataInitType"]
  h += "\n"

  ##############################################################################
  # Generated Call to Reference
  ##############################################################################
  h += "/* generated call to reference */\n"
  h += "template<typename DataType>\n"
  h += "TensileStatus generatedCallToReferenceCPU(\n"
  h += "    unsigned int *sizes,\n"
  h += "    DataType *referenceC,\n"
  h += "    DataType *initialA,\n"
  h += "    DataType *initialB,\n"
  h += "    DataType alpha,\n"
  h += "    DataType beta) {\n"
  h += "  return tensileReferenceCPU(\n"
  h += "      referenceC,\n"
  h += "      initialA,\n"
  h += "      initialB,\n"
  h += "      alpha,\n"
  h += "      beta,\n"
  h += "      totalIndices[problemTypeIdx],\n"
  h += "      sizes,\n"
  h += "      numIndicesC[problemTypeIdx],\n"
  h += "      numIndicesAB[problemTypeIdx],\n"
  h += "      indexAssignmentsA[problemTypeIdx],\n"
  h += "      indexAssignmentsB[problemTypeIdx],\n"
  h += "      complexConjugateA[problemTypeIdx],\n"
  h += "      complexConjugateB[problemTypeIdx],\n"
  h += "      validationStride );\n"
  h += "};\n"
  h += "\n"

  ##############################################################################
  # Generated Call to Solution
  ##############################################################################
  if forBenchmark:
    problemType = solutions[0]["ProblemType"]
    h += "/* generated call to solution */\n"
    h += "template<typename DataType>\n"
    h += "void generatedCallToSolution(\n"
    h += "    unsigned int solutionIdx,\n"
    h += "    unsigned int *sizes,\n"
    h += "    DataType alpha,\n"
    h += "    DataType beta ) {\n"
    h += "  // calculate parameters assuming packed data\n"
    # strides
    indexChars = globalParameters["IndexChars"]
    firstStride = 1
    if problemType["UseInitialStrides"]:
      firstStride = 0
    lastStrideC = problemType["NumIndicesC"]
    lastStrideA = len(problemType["IndexAssignmentsA"])
    lastStrideB = len(problemType["IndexAssignmentsB"])

    # calculate strides
    for i in range(0,lastStrideC):
      h += "  unsigned int strideC%u%s = 1" % (i, indexChars[i])
      for j in range(0, i):
        h += "*sizes[%i]" % j
      h += ";\n"
    for i in range(0,lastStrideA):
      h += "  unsigned int strideA%u%s = 1" % (i, \
          indexChars[problemType["IndexAssignmentsA"][i]])
      for j in range(0, i):
        h += "*sizes[%i]" % \
          problemType["IndexAssignmentsA"][j]
      h += ";\n"
    for i in range(0,lastStrideB):
      h += "  unsigned int strideB%u%s = 1" % (i, \
          indexChars[problemType["IndexAssignmentsB"][i]])
      for j in range(0, i):
        h += "*sizes[%i]" % \
          problemType["IndexAssignmentsB"][j]
      h += ";\n"
    for i in range(0, problemType["TotalIndices"]):
      h += "  unsigned int size%s = sizes[%u];\n" % (indexChars[i], i)
    h += "\n"

    # function call
    h += "  // call solution function\n"
    if globalParameters["Backend"] == "OCL":
      h += "  solutions[solutionIdx]( static_cast<cl_mem>(deviceC), static_cast<cl_mem>(deviceA), static_cast<cl_mem>(deviceB),\n"
    else:
      typeName = dataTypes[0].toCpp()
      h += "  solutions[solutionIdx]( static_cast<%s *>(deviceC), static_cast<%s *>(deviceA), static_cast<%s *>(deviceB),\n" \
          % (typeName, typeName, typeName)
    h += "      alpha,\n"
    if problemType["UseBeta"]:
      h += "      beta,\n"
    h += "      0, 0, 0, // offsets\n"
    for i in range(firstStride,lastStrideC):
      h += "      strideC%u%s,\n" % (i, indexChars[i])
    for i in range(firstStride,lastStrideA):
      h += "      strideA%u%s,\n" % (i, \
          indexChars[problemType["IndexAssignmentsA"][i]])
    for i in range(firstStride,lastStrideB):
      h += "      strideB%u%s,\n" % (i, \
          indexChars[problemType["IndexAssignmentsB"][i]])
    for i in range(0, problemType["TotalIndices"]):
      h += "      size%s,\n" % indexChars[i]
    h += "      stream,\n"
    h += "      0, NULL, NULL); // events\n"
    h += "};\n"
    h += "\n"
  else:
    ############################################################################
    # Generated Call to Function
    ############################################################################
    h += "/* generated call to function */\n"
    h += "template<typename DataType>\n"
    h += "TensileStatus generatedCallToFunction(\n"
    h += "    unsigned int *sizes,\n"
    h += "    DataType alpha,\n"
    h += "    DataType beta );\n\n"

    for dataType in dataTypes:
      typeName = dataType.toCpp()
      functionsForDataType = []
      for problemType in problemTypesForDataType[dataType]:
        for scheduleName in schedulesForProblemType[problemType]:
          functionsForDataType.append([scheduleName, problemType])
      h += "template<>\n"
      h += "inline TensileStatus generatedCallToFunction<%s>(\n" % typeName
      h += "    unsigned int *sizes,\n"
      h += "    %s alpha,\n" % typeName
      h += "    %s beta ) {\n\n" % typeName

      h += "  unsigned int functionIdxForDataType = functionInfo[functionIdx][4];\n"

      for functionIdx in range(0, len(functionsForDataType)):
        function = functionsForDataType[functionIdx]
        scheduleName = function[0]
        problemType = function[1]
        if len(functionsForDataType)> 1:
          if functionIdx == 0:
            h += "  if (functionIdxForDataType == %u) {\n" % functionIdx
          elif functionIdx == len(functionsForDataType)-1:
            h += "  } else {\n"
          else:
            h += "  } else if (functionIdxForDataType == %u) {\n" % functionIdx

        # strides
        indexChars = globalParameters["IndexChars"]
        firstStride = 1
        if problemType["UseInitialStrides"]:
          firstStride = 0
        lastStrideC = problemType["NumIndicesC"]
        lastStrideA = len(problemType["IndexAssignmentsA"])
        lastStrideB = len(problemType["IndexAssignmentsB"])

        # calculate strides
        for i in range(0,lastStrideC):
          h += "    unsigned int strideC%u%s = 1" % (i, indexChars[i])
          for j in range(0, i):
            h += "*sizes[%i]" % j
          h += ";\n"
        for i in range(0,lastStrideA):
          h += "    unsigned int strideA%u%s = 1" % (i, \
              indexChars[problemType["IndexAssignmentsA"][i]])
          for j in range(0, i):
            h += "*sizes[%i]" % \
              problemType["IndexAssignmentsA"][j]
          h += ";\n"
        for i in range(0,lastStrideB):
          h += "    unsigned int strideB%u%s = 1" % (i, \
              indexChars[problemType["IndexAssignmentsB"][i]])
          for j in range(0, i):
            h += "*sizes[%i]" % \
              problemType["IndexAssignmentsB"][j]
          h += ";\n"
        for i in range(0, problemType["TotalIndices"]):
          h += "    unsigned int size%s = sizes[%u];\n" % (indexChars[i], i)

        # function call
        h += "    // call solution function\n"
        h += "    return tensile_%s_%s(\n" % (scheduleName, problemType)
        if globalParameters["Backend"] == "OCL":
          h += "        static_cast<cl_mem>(deviceC),\n"
          h += "        static_cast<cl_mem>(deviceA),\n"
          h += "        static_cast<cl_mem>(deviceB),\n"
        else:
          h += "        static_cast<%s *>(deviceC),\n" % typeName
          h += "        static_cast<%s *>(deviceA),\n" % typeName
          h += "        static_cast<%s *>(deviceB),\n" % typeName
        h += "        alpha,\n"
        if problemType["UseBeta"]:
          h += "        beta,\n"
        h += "        0, 0, 0, // offsets\n"
        for i in range(firstStride,lastStrideC):
          h += "        strideC%u%s,\n" % (i, indexChars[i])
        for i in range(firstStride,lastStrideA):
          h += "        strideA%u%s,\n" % (i, \
              indexChars[problemType["IndexAssignmentsA"][i]])
        for i in range(firstStride,lastStrideB):
          h += "        strideB%u%s,\n" % (i, \
              indexChars[problemType["IndexAssignmentsB"][i]])
        for i in range(0, problemType["TotalIndices"]):
          h += "        size%s,\n" % indexChars[i]
        h += "        stream,\n"
        h += "        0, NULL, NULL); // events\n"
      if len(functionsForDataType) > 1:
        h += "  }\n" # close last if
      h += "};\n" # close callToFunction

  ##############################################################################
  # Results File Name
  ##############################################################################
  if forBenchmark:
    h += "/* results file name */\n"
    resultsFileName = os.path.join(globalParameters["WorkingPath"], \
        "../../Data","%s.csv" % stepName)
    resultsFileName = resultsFileName.replace("\\", "\\\\")
    h += "const char *resultsFileName = \"%s\";\n" % resultsFileName

  ##############################################################################
  # Write File
  ##############################################################################
  clientParametersFile = open(os.path.join(globalParameters["WorkingPath"], \
      "ClientParameters.h"), "w")
  clientParametersFile.write(CHeader)
  clientParametersFile.write(h)
  clientParametersFile.close()

