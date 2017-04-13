# This script only gets called by CMake
from Common import globalParameters, HR, print1, print2, printExit, ensurePath, CHeader, CMakeHeader, assignGlobalParameters
from SolutionStructs import Solution
import YAMLIO
from SolutionWriter import SolutionWriter
from KernelWriter import KernelWriter

import os
import os.path
import argparse
import sys
from shutil import copy as shutil_copy


################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
def writeSolutionsAndKernels(outputPath, solutions, \
    solutionWriter, kernelWriter):
  print1("# Writing Solutions and Kernels")
  if not globalParameters["MergeFiles"]:
    ensurePath(os.path.join(outputPath, "Solutions"))
    ensurePath(os.path.join(outputPath, "Kernels"))

  #solutionFileNames = []
  kernelNames = []
  kernels = []

  ##############################################################################
  # Min Naming
  ##############################################################################
  for solution in solutions:
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
      if kernel not in kernels:
        kernels.append(kernel)

  if globalParameters["ShortNames"] and not globalParameters["MergeFiles"] :
    solutionSerialNaming = Solution.getSerialNaming(solutions)
    kernelSerialNaming = Solution.getSerialNaming(kernels)
  else:
    solutionSerialNaming = None
    kernelSerialNaming = None
  solutionMinNaming = Solution.getMinNaming(solutions)
  kernelMinNaming = Solution.getMinNaming(kernels)

  ##############################################################################
  # Write Solutions
  ##############################################################################
  if globalParameters["MergeFiles"]:
    solutionSourceFile = open(os.path.join(outputPath, \
        "Solutions.cpp"), "w")
    solutionHeaderFile = open(os.path.join(outputPath, \
        "Solutions.h"), "w")
    solutionSourceFile.write("#include \"Solutions.h\"\n")
    solutionHeaderFile.write("#include \"TensileTypes.h\"\n")
    solutionHeaderFile.write("#include \"Kernels.h\"\n")
    solutionHeaderFile.write("#include \"SolutionHelper.h\"\n")
    solutionHeaderFile.write("#include \"Tools.h\"\n")
  for solution in solutions:
    # get solution name
    if not globalParameters["MergeFiles"]:
      if globalParameters["ShortNames"]:
        solutionFileName = \
            Solution.getNameSerial(solution, solutionSerialNaming)
      else:
        solutionFileName = Solution.getNameMin(solution, solutionMinNaming)

    # write solution.cpp
    if not globalParameters["MergeFiles"]:
      solutionSourceFile = open(os.path.join(outputPath, \
          "Solutions", solutionFileName+".cpp"), "w")
    solutionSourceFile.write(CHeader)
    solutionSourceFile.write( \
        solutionWriter.getSourceFileString(solution))
    if not globalParameters["MergeFiles"]:
      solutionSourceFile.close()

    # write solution.h
    if not globalParameters["MergeFiles"]:
      solutionHeaderFile = open(os.path.join(outputPath, \
          "Solutions", solutionFileName+".h"), "w")
    solutionHeaderFile.write(CHeader)
    solutionHeaderFile.write( \
        solutionWriter.getHeaderFileString(solution))
    if not globalParameters["MergeFiles"]:
      solutionHeaderFile.close()
  # close merged
  if not globalParameters["MergeFiles"]:
    solutionHeaderFile.close()

  ##############################################################################
  # Write Kernels
  ##############################################################################
  if globalParameters["MergeFiles"]:
    kernelSourceFile = open(os.path.join(outputPath, \
        "Kernels.cpp"), "w")
    kernelHeaderFile = open(os.path.join(outputPath, \
        "Kernels.h"), "w")
    kernelSourceFile.write("#include \"Kernels.h\"\n")
    kernelHeaderFile.write("#pragma once\n")
    if globalParameters["RuntimeLanguage"] == "HIP":
      kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
    else:
      kernelHeaderFile.write("#include <string>\n")
  for kernel in kernels:
    # get kernel name
    if not globalParameters["MergeFiles"]:
      if globalParameters["ShortNames"]:
        kernelName = Solution.getNameSerial(kernel, kernelSerialNaming)
      else:
        kernelName = Solution.getNameMin(kernel, kernelMinNaming)
      kernelNames.append(kernelName)

    # write kernel.cpp
    if not globalParameters["MergeFiles"]:
      kernelSourceFile = open(os.path.join(outputPath, \
          "Kernels", kernelName+".cpp"), "w")
    kernelSourceFile.write(CHeader)
    kernelSourceFile.write( kernelWriter.getSourceFileString(kernel))
    if not globalParameters["MergeFiles"]:
      kernelSourceFile.close()

    # write kernel.h
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile = open(os.path.join(outputPath, \
          "Kernels", kernelName+".h"), "w")
    kernelHeaderFile.write(CHeader)
    kernelHeaderFile.write( kernelWriter.getHeaderFileString(kernel))
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile.close()
  # close merged
  if globalParameters["MergeFiles"]:
    kernelHeaderFile.close()


################################################################################
# Write Logic
################################################################################
def writeLogic(outputPath, logicList, solutionWriter ):
  print1("# Writing Library Logic")

  if not globalParameters["MergeFiles"]:
    ensurePath(os.path.join(outputPath, "Logic"))
  indexChars = globalParameters["IndexChars"]

  # the header will always be merged into "Tensile.h"
  # includes for merged files
  s = ""
  h = ""
  h += "#pragma once\n"
  h += "#include \"TensileTypes.h\"\n"
  #h += "\nTensileStatus tensileSetup();\n"
  #h += "\nTensileStatus tensileTeardown();\n"
  s += "#include \"Tensile.h\"\n"
  s += "#include \"Solutions.h\"\n"

  # solution naming
  solutions = []
  for logicProblemType in logicList:
    problemTypeSolutions = logicProblemType[2]
    solutions.extend(problemTypeSolutions)
  if globalParameters["ShortNames"]:
    solutionSerialNaming = Solution.getSerialNaming(solutions)
  else:
    solutionMinNaming = Solution.getMinNaming(solutions)


  # for each ProblemType
  for logicProblemType in logicList:

    # get logic parameters for problem type
    scheduleName = logicProblemType[0]
    problemType = logicProblemType[1]
    solutions = logicProblemType[2]
    indexOrder = logicProblemType[3]
    logic = logicProblemType[4]

    # solution names
    solutionNames = []
    for solution in solutions:
      if globalParameters["ShortNames"]:
        solutionNames.append(Solution.getNameSerial(solution, \
            solutionSerialNaming) )
      else:
        solutionNames.append(Solution.getNameMin(solution, \
            solutionMinNaming) )

    functionName = "tensile_%s_%s" % (scheduleName, problemType)

    # reset individual file string
    if not globalParameters["MergeFiles"]:
      filePrefix   = "Tensile_%s_%s" % (scheduleName, problemType)
      #s = "#include \"%s.h\"" % filePrefix
      s = "#include \"Tensile.h\""
      for solutionName in solutionNames:
        h += "#include \"%s.h\"\n" % solutionName

    # function argument list
    argList = solutionWriter.getArgList(solutions[0])

    # declare function in header
    h += "\nTensileStatus %s(\n" % functionName
    for i in range(0, len(argList)):
      h += "    %s%s" % (argList[i], ",\n" if i < len(argList)-1 else ");\n\n")

    # implement function in source
    s += "\nTensileStatus %s(\n" % functionName
    for i in range(0, len(argList)):
      s += "    %s%s" % (argList[i], ",\n" if i < len(argList)-1 else ") {\n\n")

    """
    indent = "  "
    s += "%ssize_t sizeC = size%s" % ( indent, indexChars[0])
    for i in range(1, problemType["NumIndicesC"]):
      s += "*size%s" % indexChars[i]
    s += ";\n"
    s += "%ssize_t sizeSum = size%s" % ( indent, \
        indexChars[problemType["IndicesSummation"][0]])
    for i in range(1, len(problemType["IndicesSummation"])):
      s += "*size%s" % indexChars[problemType["IndicesSummation"][i]]
    s += ";\n\n"
    """
    print2(solutionNames)

    logicStr = writeLogicRec(0, indexOrder, logic, solutionNames, problemType)
    s += logicStr
    s += "\n}\n"

    # open and close individual files
    if not globalParameters["MergeFiles"]:
      logicSourceFile = open(os.path.join(outputPath, "Logic", \
          "%s.cpp" % filePrefix), "w")
      logicSourceFile.write(s)
      logicSourceFile.close()

  # close merged files
  if globalParameters["MergeFiles"]:
    logicSourceFile = open(os.path.join(outputPath, \
        "TensileFunctions.cpp"), "w")
    logicSourceFile.write(s)
    logicSourceFile.close()

  logicHeaderFile = open(os.path.join(outputPath, \
      "Tensile.h"), "w")
  logicHeaderFile.write(h)
  logicHeaderFile.close()

################################################################################
# Write Logic Recursive
################################################################################
def writeLogicRec(depth, indexOrder, logic, solutionNames, problemType):
  indexChars = globalParameters["IndexChars"]
  indent = "  "
  indent += "  "*depth
  s = ""
  lowestLevel = depth == len(indexOrder)-1
  numRules = len(logic)
  for ruleIdx in range(0, numRules):
    rule = logic[ruleIdx]
    threshold = rule[0]
    if lowestLevel:
      solutionIdx = rule[1]
      solutionCall = writeSolutionCall(solutionNames[solutionIdx],problemType)
      if threshold > 0:
        s += "%sif (size%s < %u) return %s;\n" \
            % (indent, indexChars[indexOrder[depth]], threshold, solutionCall)
      else:
        s += "%sreturn %s;\n" % (indent, solutionCall)
    else:
      if threshold > 0:
        s += "%sif (size%s < %u) {\n" \
            % (indent, indexChars[indexOrder[depth]], threshold)
      else:
        s += "%s{\n" % (indent)
      s += writeLogicRec(depth+1, indexOrder, rule[1], solutionNames, \
          problemType)
      s += "%s}\n" % (indent)
  return s


################################################################################
# Write Solution Call
################################################################################
def writeSolutionCall(solutionName, problemType):
  indexChars = globalParameters["IndexChars"]
  s = ""
  s += "%s(" % solutionName
  # solution parameters
  s += " dataC, dataA, dataB, alpha"
  if problemType["UseBeta"]:
    s += ", beta"
  s += ", offsetC, offsetA, offsetB"
  firstStride = 1
  if problemType["UseInitialStrides"]:
    firstStride = 0
  lastStrideC = problemType["NumIndicesC"]
  lastStrideA = len(problemType["IndexAssignmentsA"])
  lastStrideB = len(problemType["IndexAssignmentsB"])
  for i in range(firstStride,lastStrideC):
    s += ", strideC%u%s" % (i, indexChars[i])
  for i in range(firstStride,lastStrideA):
    s += ", strideA%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsA"][i]])
  for i in range(firstStride,lastStrideB):
    s += ", strideB%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsB"][i]])
  for i in range(0, problemType["TotalIndices"]):
    s += ", size%s" % indexChars[i]
  s += ", stream, numInputEvents, inputEvents, outputEvent )"
  return s




################################################################################
# Write CMake
################################################################################
def writeCMake(outputPath, solutions, libraryStaticFiles, clientName ):
  print1("# Writing Custom CMake")
  ##############################################################################
  # Min Naming
  ##############################################################################
  kernels = []
  for solution in solutions:
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
      if kernel not in kernels:
        kernels.append(kernel)

  if globalParameters["ShortNames"] and not globalParameters["MergeFiles"] :
    solutionSerialNaming = Solution.getSerialNaming(solutions)
    kernelSerialNaming = Solution.getSerialNaming(kernels)
  else:
    solutionSerialNaming = None
    kernelSerialNaming = None
  solutionMinNaming = Solution.getMinNaming(solutions)
  kernelMinNaming = Solution.getMinNaming(kernels)
  solutionWriter = SolutionWriter( \
      solutionMinNaming, solutionSerialNaming, \
      kernelMinNaming, kernelSerialNaming)
  kernelWriter = KernelWriter( \
      kernelMinNaming, kernelSerialNaming)

  generatedFile = open(os.path.join(outputPath, "Generated.cmake"), "w")
  generatedFile.write(CMakeHeader)
  #generatedFile.write("set( ClientName %s)\n\n" % clientName )
  generatedFile.write("set( TensileClient_SOLUTIONS\n")

  # write solution names
  if globalParameters["MergeFiles"]:
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions.h\n")
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions.cpp\n")
  else:
    for solution in solutions:
      solutionName = solutionWriter.getSolutionName(solution)
      generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions/%s.h\n" \
          % (solutionName) )
      generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions/%s.cpp\n" \
          % (solutionName) )
  generatedFile.write("  )\n")

  # write kernel names
  generatedFile.write("set( TensileClient_KERNELS\n")
  if globalParameters["MergeFiles"]:
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Kernels.h\n")
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Kernels.cpp\n")
  else:
    for kernel in kernels:
      kernelName = kernelWriter.getKernelName(kernel)
      generatedFile.write("  ${CMAKE_SOURCE_DIR}/Kernels/%s.h\n" % (kernelName))
      generatedFile.write("  ${CMAKE_SOURCE_DIR}/Kernels/%s.cpp\n" % kernelName)
  generatedFile.write("  )\n")


  generatedFile.write("set( TensileClient_SOURCE\n")
  for fileName in libraryStaticFiles:
    # copy file
    shutil_copy( os.path.join(globalParameters["SourcePath"], fileName), \
        outputPath )
    # add file to cmake
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/%s\n" % fileName)
  generatedFile.write("  )\n\n")

  # close generated cmake
  generatedFile.close()



################################################################################
# Tensile Create Library
################################################################################
def TensileCreateLibrary():
  print1("")
  print1(HR)
  print1("# Tensile Create Library")
  print2(HR)
  print2("")

  ##############################################################################
  # Parse Command Line Arguments
  ##############################################################################
  print2("Arguments: %s" % sys.argv)
  argParser = argparse.ArgumentParser()
  argParser.add_argument("LogicPath", help="Path to LibraryLogic.yaml files.")
  argParser.add_argument("OutputPath", help="Where to write library files?")
  argParser.add_argument("RuntimeLanguage", help="Which runtime language?", \
      choices=["OCL", "HIP", "HSA"])
  argParser.add_argument("KernelLanguage", help="Which kernel language?", \
      choices=["OCL", "HIP", "ASM"])
  argParser.add_argument("--merge-files", dest="MergeFiles", \
      action="store_true")
  argParser.add_argument("--no-merge-files", dest="MergeFiles", \
      action="store_false")
  argParser.add_argument("--short-file-names", dest="ShortNames", \
      action="store_true")
  argParser.add_argument("--no-short-file-names", dest="ShortNames", \
      action="store_false")
  argParser.add_argument("--library-print-debug", dest="LibraryPrintDebug", \
      action="store_true")
  argParser.add_argument("--no-library-print-debug", dest="LibraryPrintDebug", \
      action="store_false")
  args = argParser.parse_args()

  logicPath = args.LogicPath
  outputPath = args.OutputPath
  print2("OutputPath: %s" % outputPath)
  ensurePath(outputPath)
  arguments = {}
  arguments["RuntimeLanguage"] = args.RuntimeLanguage
  arguments["KernelLanguage"] = args.KernelLanguage
  arguments["MergeFiles"] = args.MergeFiles
  arguments["ShortNames"] = args.ShortNames
  arguments["LibraryPrintDebug"] = args.LibraryPrintDebug
  assignGlobalParameters(arguments)

  if not os.path.exists(logicPath):
    printExit("LogicPath %s doesn't exist" % logicPath)

  logicFiles = [os.path.join(logicPath, f) for f in os.listdir(logicPath) \
      if (os.path.isfile(os.path.join(logicPath, f)) \
      and os.path.splitext(f)[1]==".yaml")]

  print1("# LibraryLogicFiles:" % logicFiles)
  for logicFile in logicFiles:
    print1("#   %s" % logicFile)

  ##############################################################################
  # Parse config files
  ##############################################################################
  solutions = []
  logicList = []
  for logicFileName in logicFiles:
    (scheduleName, problemType, solutionsForType, indexOrder, logic) \
        = YAMLIO.readLibraryLogicForProblemType(logicFileName)
    logicList.append((scheduleName, problemType, solutionsForType, \
        indexOrder, logic ))
    for solution in solutionsForType:
      if solution not in solutions:
        solutions.append(solution)

  # create solution writer and kernel writer
  kernels = []
  for solution in solutions:
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
      if kernel not in kernels:
        kernels.append(kernel)
  if globalParameters["ShortNames"] and not globalParameters["MergeFiles"]:
    solutionSerialNaming = Solution.getSerialNaming(solutions)
    kernelSerialNaming = Solution.getSerialNaming(kernels)
  else:
    solutionSerialNaming = None
    kernelSerialNaming = None
  solutionMinNaming = Solution.getMinNaming(solutions)
  kernelMinNaming = Solution.getMinNaming(kernels)
  solutionWriter = SolutionWriter( \
      solutionMinNaming, solutionSerialNaming, \
      kernelMinNaming, kernelSerialNaming)
  kernelWriter = KernelWriter( \
      kernelMinNaming, kernelSerialNaming)

  # write solutions and kernels
  writeSolutionsAndKernels(outputPath, solutions, solutionWriter, kernelWriter)

  libraryStaticFiles = [
      "SetupTeardown.cpp",
      "TensileTypes.h",
      "SolutionHelper.cpp",
      "SolutionHelper.h",
      "Tools.cpp",
      "Tools.h" ]

  # write cmake
  clientName = "LibraryClient"
  writeCMake(outputPath, solutions, libraryStaticFiles, clientName )

  # write logic
  writeLogic(outputPath, logicList, solutionWriter)
  print1("# Tensile Library Writer DONE")
  print1(HR)
  print1("")

################################################################################
# Main
################################################################################
if __name__ == "__main__":
    TensileCreateLibrary()
