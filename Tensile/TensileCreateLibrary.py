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
# This script only gets called by CMake
from Common import globalParameters, HR, print1, print2, printExit, ensurePath, CHeader, CMakeHeader, assignGlobalParameters
from SolutionStructs import Solution
import YAMLIO
from SolutionWriter import SolutionWriter
from KernelWriterSource import KernelWriterSource

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
      kernelHeaderFile.write("#include <hip/hip_fp16.h>\n")
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
def writeLogic(outputPath, logicData, solutionWriter ):
  print1("# Writing Library Logic")

  if not globalParameters["MergeFiles"]:
    ensurePath(os.path.join(outputPath, "Logic"))
  indexChars = globalParameters["IndexChars"]

  # Tensile.h
  h = ""
  h += "#pragma once\n"
  h += "#include \"TensileTypes.h\"\n"

  # TensileInternal.h
  ih = ""
  ih += "#include \"Tensile.h\"\n"
  ih += "#include \"SolutionHelper.h\"\n"
  ih += "#include <map>\n"
  ih += "#include <tuple>\n"

  # Tensile.cpp
  s = ""
  s += "#include \"Tensile.h\"\n"
  s += "#include \"TensileInternal.h\"\n"
  s += "#include \"Solutions.h\"\n"

  ########################################
  # problemType
  for problemType in logicData:

    # function argument list
    argListSizes = solutionWriter.getArgList(problemType, False, False)
    argListStream = solutionWriter.getArgList(problemType, False, True)
    argListData = solutionWriter.getArgList(problemType, True, True)

    # declare tensile_ProblemType
    h += "\n// enqueue solution\n"
    h += "TensileStatus tensile_%s(\n" % problemType
    for i in range(0, len(argListData)):
      h += "    %s %s%s" \
          % (argListData[i][0], argListData[i][1], \
          ",\n" if i < len(argListData)-1 else ");\n\n")

    # declare TensileSolutionPointer_ProblemType
    h += "\n// solution pointer\n"
    h += "typedef TensileStatus (*TensileSolutionPointer_%s)(\n" \
        % problemType
    for i in range(0, len(argListData)):
      h += "    %s %s%s" % (argListData[i][0], argListData[i][1], ",\n" \
          if i < len(argListData)-1 else ");\n\n")

    # declare tensileGetSolutionPointer_ProblemType
    h += "\n// get solution pointer\n"
    h += "TensileSolutionPointer_%s tensileGetSolutionPointer_%s(\n" \
        % (problemType, problemType)
    for i in range(0, len(argListStream)):
      h += "    %s %s%s" \
          % (argListStream[i][0], argListStream[i][1], \
          ",\n" if i < len(argListStream)-1 else ");\n\n")

    # declare tensileName_
    h += "// get solution name\n"
    h += "char* tensileGetSolutionName_%s(\n" \
        % (problemType)
    for i in range(0, len(argListStream)):
      h += "    %s %s%s" \
          % (argListStream[i][0], argListStream[i][1], \
          ",\n" if i < len(argListStream)-1 else ");\n\n")


    # get solution naming for problem type
    solutionsForProblemType = []
    for scheduleTuple in logicData[problemType]:
      solutionsForSchedule = scheduleTuple[2]
      for solution in solutionsForSchedule:
        if solution not in solutionsForProblemType:
          solutionsForProblemType.append(solution)
    if globalParameters["ShortNames"]:
      solutionSerialNaming = Solution.getSerialNaming(solutionsForProblemType)
    else:
      solutionMinNaming = Solution.getMinNaming(solutionsForProblemType)

    # solution names for problem type
    solutionNamesForProblemType = []
    for solution in solutionsForProblemType:
      if globalParameters["ShortNames"]:
        solutionName = Solution.getNameSerial(solution, solutionSerialNaming)
      else:
        solutionName = Solution.getNameMin(solution, solutionMinNaming)
      solutionNamesForProblemType.append(solutionName)

    # reset problemType source
    if not globalParameters["MergeFiles"]:
      filePrefix = "Tensile_%s" % (problemType)
      s = "#include \"Tensile.h\"\n"
      s += "#include \"TensileInternal.h\"\n"
      for solutionName in solutionNamesForProblemType:
        s += "#include \"%s.h\"\n" % solutionName

    ########################################
    # implement per-Schedule functions in source
    s += "// per-schedule functions\n"
    for scheduleTuple in logicData[problemType]:

      # get logic parameters for problem type
      scheduleName  = scheduleTuple[0]
      deviceNames   = scheduleTuple[1]
      solutionsForSchedule = scheduleTuple[2]
      indexOrder    = scheduleTuple[3]
      logic         = scheduleTuple[4]

      # solution names for schedule
      solutionNamesForSchedule = []
      for solution in solutionsForSchedule:
        if globalParameters["ShortNames"]:
          solutionName = Solution.getNameSerial(solution, solutionSerialNaming)
        else:
          solutionName = Solution.getNameMin(solution, solutionMinNaming)
        solutionNamesForSchedule.append(solutionName)

      # function tensileGetSolutionPointerUncached_Schedule_ProblemType
      s += "\n// problem size -> solution logic\n"
      s += "TensileSolutionPointer_%s tensileGetSolutionPointerUncached_%s_%s(\n" \
          % (problemType, scheduleName, problemType)
      for i in range(0, len(argListSizes)):
        s += "    %s %s%s" \
            % (argListSizes[i][0], argListSizes[i][1], \
            ",\n" if i < len(argListSizes)-1 else ") {\n\n")
      logicStr = writeLogicRec(0, indexOrder, logic, solutionNamesForSchedule, \
          problemType, True)
      s += logicStr
      s += "\n}\n"

      # function tensileGetSolutionName_Schedule_ProblemType
      s += "\n// get solution name for problem size\n"
      s += "char * tensileGetSolutionName_%s_%s(\n" \
          % (scheduleName, problemType)
      for i in range(0, len(argListSizes)):
        s += "    %s %s%s" \
            % (argListSizes[i][0], argListSizes[i][1], \
            ",\n" if i < len(argListSizes)-1 else ") {\n\n")
      logicStr = writeLogicRec(0, indexOrder, logic, solutionNamesForSchedule, \
          problemType, False)
      s += logicStr
      s += "\n}\n"

    ########################################
    # implement problem-type functions in source
    s += "// per-problemType functions\n"

    # problem type templates
    problemTypeTemplate = "%s" % ("cl_command_queue" \
        if globalParameters["RuntimeLanguage"] == "OCL" else "hipStream_t")
    for i in range(0, problemType["TotalIndices"]):
      problemTypeTemplate += ", unsigned int"
    ih += "typedef std::tuple<%s> Key_%s;\n" \
        % (problemTypeTemplate, problemType)
    ih += "typedef std::map<Key_%s, TensileSolutionPointer_%s> Map_%s;\n" \
        % (problemType, problemType, problemType)
    ih += "extern Map_%s solutionMap_%s;\n" % (problemType, problemType)

    # implement tensileGetSolutionPointerUncached_ProblemType
    for ptr in [True, False]:
      returnType = "PointerUncached" if ptr else "Name"
      s += "\n// return solution %s\n" % returnType
      s += ("TensileSolutionPointer_%s "%problemType) if ptr else "char *"
      s += "tensileGetSolution%s_%s(\n" \
          % (returnType, problemType)
      for i in range(0, len(argListStream)):
        s += "    %s %s%s" \
            % (argListStream[i][0], argListStream[i][1], \
            ",\n" if i < len(argListStream)-1 else ") {\n")

      # choose from schedules based on device name
      schedules = logicData[problemType]
      numSchedules = len(schedules)
      if numSchedules > 1:

        # get device name
        if globalParameters["RuntimeLanguage"] == "OCL":
          s += "get device name opencl;\n"
        else:
          s += "get device name hip;\n"

        for scheduleIdx in range(0, numSchedules):
          schedule = schedules[scheduleIdx]
          deviceNames = schedule[1]
          if scheduleIdx > 0:
            s += "else "
          if scheduleIdx < numSchedules-1:
            s += "if ("
            for deviceNameIdx in range(0, len(deviceNames)):
              deviceName = deviceNames[deviceNameIdx]
              if deviceNameIdx > 0:
                s += " && "
                s += "name == \"%s\"" % deviceName
            s += ")"
          s += "{"
          s += "  return tensileGetSolution%s_%s_%s(" \
              % ( returnType, scheduleName, problemType)
          for i in range(0, len(argListSizes)):
            s += "%s%s" \
                % (argListSizes[i][1],
                    ", " if i < len(argListSizes)-1 else ");\n")
            s += "}"
      else: # == 1
        schedule = schedules[0]
        scheduleName = schedule[0]
        s += "  return tensileGetSolution%s_%s_%s(" \
            % ( returnType, scheduleName, problemType)
        for i in range(0, len(argListSizes)):
          s += "%s%s" \
              % (argListSizes[i][1],
                  ", " if i < len(argListSizes)-1 else ");\n")
      s += "\n}\n"


    # implement tensileGetSolutionPointer_ProblemType
    s += "\n// return solution pointer; user calls it\n"
    s += "Map_%s solutionMap_%s;\n" % (problemType, problemType)
    s += "TensileSolutionPointer_%s tensileGetSolutionPointer_%s(\n" \
        % (problemType, problemType)
    for i in range(0, len(argListStream)):
      s += "    %s %s%s" \
          % (argListStream[i][0], argListStream[i][1], \
          ",\n" if i < len(argListStream)-1 else ") {\n")
    # create key
    s += "Key_%s key = std::make_tuple(stream" % (problemType)
    for i in range(0, problemType["TotalIndices"]):
      s += ", size%s" % globalParameters["IndexChars"][i]
    s += ");\n"
    # check for key in map
    s += "  Map_%s::iterator iter = solutionMap_%s.find(key);\n" \
        % (problemType, problemType)
    s += "  if (iter != solutionMap_%s.end()) {\n" % problemType
    s += "    return iter->second;\n"
    s += "  } else {\n"
    s += "    TensileSolutionPointer_%s ptr = tensileGetSolutionPointerUncached_%s(\n" \
        % (problemType, problemType)
    for i in range(0, len(argListStream)):
      s += "%s%s" \
          % (argListStream[i][1], ", " if i < len(argListStream)-1 else ");\n")
    s += "    solutionMap_%s[key] = ptr;\n" % problemType
    s += "    return ptr;\n"
    s += "  }\n"
    s += "}\n"

    # declare tensile_ProblemType
    s += "\n// main call to solution; enqueues a kernel\n"
    s += "TensileStatus tensile_%s(\n" % problemType
    for i in range(0, len(argListData)):
      s += "    %s %s%s" \
          % (argListData[i][0], argListData[i][1], \
          ",\n" if i < len(argListData)-1 else ") {\n")
    s += "    TensileSolutionPointer_%s ptr = tensileGetSolutionPointer_%s(\n" \
        % (problemType, problemType)
    for i in range(0, len(argListStream)):
      s += "%s%s" \
          % (argListStream[i][1], ", " if i < len(argListStream)-1 else ");\n")
    s += "    return ptr("
    for i in range(0, len(argListData)):
      s += "%s%s" \
          % (argListData[i][1], ", " if i < len(argListData)-1 else ");\n")
    s += "}\n"

    # open and close problemType files
    if not globalParameters["MergeFiles"]:
      logicSourceFile = open(os.path.join(outputPath, "Logic", \
          "%s.cpp" % filePrefix), "w")
      logicSourceFile.write(s)
      logicSourceFile.close()

  # close merged files
  if globalParameters["MergeFiles"]:
    logicSourceFile = open(os.path.join(outputPath, \
        "Tensile.cpp"), "w")
    logicSourceFile.write(s)
    logicSourceFile.close()

  logicHeaderFile = open(os.path.join(outputPath, \
      "Tensile.h"), "w")
  logicHeaderFile.write(h)
  logicHeaderFile.close()

  internalHeaderFile = open(os.path.join(outputPath, \
      "TensileInternal.h"), "w")
  internalHeaderFile.write(ih)
  internalHeaderFile.close()

################################################################################
# Write Logic Recursive
################################################################################
def writeLogicRec(depth, indexOrder, logic, solutionNames, problemType, ptr):
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
      solutionName = solutionNames[solutionIdx]
      if ptr:
        returnValue = solutionName
      else:
        returnValue = "\"%s\"" % solutionName
      if threshold > 0:
        s += "%sif (size%s <= %u) return %s;\n" \
            % (indent, indexChars[indexOrder[depth]], threshold, returnValue)
      else:
        s += "%sreturn %s;\n" % (indent, returnValue)
    else:
      if threshold > 0:
        s += "%sif (size%s <= %u) {\n" \
            % (indent, indexChars[indexOrder[depth]], threshold)
      else:
        s += "%s{\n" % (indent)
      s += writeLogicRec(depth+1, indexOrder, rule[1], solutionNames, \
          problemType, ptr)
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
  kernelWriter = KernelWriterSource( \
      kernelMinNaming, kernelSerialNaming)

  generatedFile = open(os.path.join(outputPath, "Generated.cmake"), "w")
  generatedFile.write(CMakeHeader)
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
  logicData = {} # keys are problemTypes, values are schedules
  for logicFileName in logicFiles:
    (scheduleName, deviceNames, problemType, solutionsForSchedule, \
        indexOrder, logic) \
        = YAMLIO.readLibraryLogicForSchedule(logicFileName)
    if problemType not in logicData:
      logicData[problemType] = []
    logicData[problemType].append((scheduleName, deviceNames, \
        solutionsForSchedule, indexOrder, logic ))
    for solution in solutionsForSchedule:
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
  kernelWriter = KernelWriterSource( \
      kernelMinNaming, kernelSerialNaming)

  # write solutions and kernels
  writeSolutionsAndKernels(outputPath, solutions, solutionWriter, kernelWriter)

  libraryStaticFiles = [
      "TensileTypes.h",
      "SolutionHelper.cpp",
      "SolutionHelper.h",
      "Tools.cpp",
      "Tools.h" ]

  # write cmake
  clientName = "LibraryClient"
  writeCMake(outputPath, solutions, libraryStaticFiles, clientName )

  # write logic
  writeLogic(outputPath, logicData, solutionWriter)
  print1("# Tensile Library Writer DONE")
  print1(HR)
  print1("")

################################################################################
# Main
################################################################################
if __name__ == "__main__":
    TensileCreateLibrary()
