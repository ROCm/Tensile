################################################################################
# Copyright (C) 2016-2019 Advanced Micro Devices, Inc. All rights reserved.
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
from __future__ import print_function
from .Common import globalParameters, HR, print1, print2, printExit, ensurePath, CHeader, CMakeHeader, assignGlobalParameters, ProgressBar, listToInitializer
from .SolutionStructs import Solution
from . import YAMLIO
from .SolutionWriter import SolutionWriter
from .KernelWriterSource import KernelWriterSource
from .KernelWriterAssembly import KernelWriterAssembly
from . import Utils

from shutil import copy as shutil_copy

import argparse
import itertools
import multiprocessing
import os
import subprocess
import sys
import time


################################################################################
def processKernelSource(kernel, kernelWriterSource, kernelWriterAssembly):
    """
    Process a single kernel, return results.
    """
    kernelWriter = kernelWriterSource if kernel["KernelLanguage"] == "Source" else kernelWriterAssembly
    # get kernel name
    kernelName = kernelWriter.getKernelName(kernel)
    #sys.stderr.write("kernel:%s\n"% kernelName)
    (err, src) = kernelWriter.getSourceFileString(kernel)

    header = kernelWriter.getHeaderFileString(kernel)

    return (err, src, header, kernelName)

def processKernelSourceWithArgs(args):
    """
    Multiprocessing aid.  Wraps up processKernelSource and takes only a single argument.
    """
    return processKernelSource(*args)

def linkCombinedCodeObjectFile(kernels, kernelsBetaOnly, kernelWriterSource, kernelWriterAssembly, outputPath):
    kernelsToLink = [kernelWriterAssembly.getKernelName(k) for k in kernels if k['KernelLanguage'] == 'Assembly']
    asmDir = kernelWriterAssembly.getAssemblyDirectory()

    objectFiles = [os.path.join(asmDir, k + '.o') for k in kernelsToLink]

    coFile = os.path.join(outputPath, 'TensileLibrary.co')

    args = kernelWriterAssembly.getLinkCodeObjectArgs(objectFiles, coFile)

    subprocess.check_call(args)

################################################################################
def prepAsm():
  """
  Create and prepare the assembly directory  - called ONCE per output dir:
  """
  asmPath = ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly") )
  assemblerFileName = os.path.join(asmPath, \
      "asm.%s"%("bat" if os.name=="nt" else "sh"))
  assemblerFile = open(assemblerFileName, "w")
  if os.name == "nt":
    assemblerFile.write("echo Windows: Copying instead of Assembling\n")
    assemblerFile.write("copy %1.s %1.o\n")
    assemblerFile.write("copy %1.o %1.co\n")
  else:
    assemblerFile.write("#!/bin/sh %s\n" % ("-x" if globalParameters["PrintLevel"] >=2  else ""))
    assemblerFile.write("# usage: asm.sh kernelName ASM_ARGS\n")
    assemblerFile.write("# example: asm.sh kernelName -mcpu=gfx900\n")
    assemblerFile.write("f=$1\n")
    assemblerFile.write("shift\n")
    assemblerFile.write("ASM=%s\n"%globalParameters["AssemblerPath"])
    # cannot use globalParameters["CurrentISA"] because it might be (0,0,0)
    defaultIsa = (9,0,0)
    assemblerFile.write( \
      "${ASM} -x assembler -target amdgcn--amdhsa %s $@ -c -o $f.o $f.s\n" % \
      ("-mno-code-object-v3" if \
      globalParameters["AsmCaps"][defaultIsa]["HasCodeObjectV3"] else ""))
    assemblerFile.write("${ASM} -target amdgcn--amdhsa $f.o -o $f.co\n")
  assemblerFile.close()
  os.chmod(assemblerFileName, 0o777)

################################################################################
def buildKernelSourceAndHeaderFiles(results, outputPath, kernelsWithBuildErrs, \
      kernelSourceFile, kernelHeaderFile):
  """
  Logs errors and writes appropriate info to kernelSourceFile and kernelHeaderFile.

  Arguments:
    results:              list of (err, src, header, kernelName)
    outputPath:           path to source directory
    kernelsWithBuildErrs: Dictionary to be updated with kernels that have errors
    kernelSourceFile:     File to write source data to
    kernelHeaderFile:     File to write header data to
  """

  for (err,src,header,kernelName) in results:
    if err:
      kernelsWithBuildErrs[kernelName] = err
      #print "*** warning: invalid kernel#%s"%kernelName

    # write kernel.cpp
    if not globalParameters["MergeFiles"]:
      kernelSourceFile = open(os.path.join(outputPath, \
          "Kernels", kernelName+".cpp"), "w")
      kernelSourceFile.write(CHeader)

    kernelSourceFile.write(src)

    if not globalParameters["MergeFiles"]:
      kernelSourceFile.close()
      # write kernel.h
      kernelHeaderFile = open(os.path.join(outputPath, \
          "Kernels", kernelName+".h"), "w")
      kernelHeaderFile.write(CHeader)

    kernelHeaderFile.write(header)

    if not globalParameters["MergeFiles"]:
      kernelHeaderFile.close()

################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
def writeSolutionsAndKernels(outputPath, problemTypes, solutions, kernels, kernelsBetaOnly, \
    solutionWriter, kernelWriterSource, kernelWriterAssembly):
  start = time.time()
  print1("# Writing Kernels...")
  if not globalParameters["MergeFiles"]:
    ensurePath(os.path.join(outputPath, "Solutions"))
    ensurePath(os.path.join(outputPath, "Kernels"))

  ##############################################################################
  # Write Kernels
  ##############################################################################
  if globalParameters["MergeFiles"]:
    kernelSourceFile = open(os.path.join(outputPath, \
        "Kernels.cpp"), "w")
    kernelHeaderFile = open(os.path.join(outputPath, \
        "Kernels.h"), "w")
    kernelSourceFile.write(CHeader)
    kernelHeaderFile.write(CHeader)
    kernelSourceFile.write("#include \"Kernels.h\"\n")
    kernelHeaderFile.write("#pragma once\n")
    if globalParameters["RuntimeLanguage"] == "HIP":
      kernelHeaderFile.write("// Also set env var HCC_ENABLE_PRINTF=1 for printf\n")
      kernelHeaderFile.write("#define HCC_ENABLE_ACCELERATOR_PRINTF\n\n")
      kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
      kernelHeaderFile.write("#include \"TensileTypes.h\"\n")
      kernelHeaderFile.write("#include \"KernelHeader.h\"\n")
      kernelHeaderFile.write("\n\n")
      kernelHeaderFile.write("__device__ inline int GenDot4(int a, int b, int c) { \n")
      kernelHeaderFile.write("  typedef struct { int c0:8,c1:8,c2:8,c3:8; } C4I8;\n")
      kernelHeaderFile.write("  typedef union { int32_t i; C4I8 z; } PkInt8x4;\n")
      kernelHeaderFile.write("  PkInt8x4 va, vb; va.i = a; vb.i = b;\n")

      kernelHeaderFile.write("//if (__oclc_ISA_version == 906)\n")
      kernelHeaderFile.write("//{\n")
      kernelHeaderFile.write("//    return (float)__llvm_amdgcn_sdot4(a, b, c, true);\n")
      kernelHeaderFile.write("//}\n")
      kernelHeaderFile.write("//else\n")
      kernelHeaderFile.write("//{\n")
      kernelHeaderFile.write("      return c + (vb.z.c3*va.z.c3 + vb.z.c2*va.z.c2 + vb.z.c1*va.z.c1 + vb.z.c0*va.z.c0); }\n")
      kernelHeaderFile.write("//}\n")
      kernelHeaderFile.write("\n\n")
    else:
      kernelHeaderFile.write("#include <string>\n")

  kernelsWithBuildErrs = {}

  prepAsm()

  if globalParameters["CpuThreads"] == 0:
    cpus = 0
  else:
    cpu_count = multiprocessing.cpu_count()
    cpuThreads = globalParameters["CpuThreads"]
    cpus = cpu_count*abs(cpuThreads) if cpuThreads < 0 \
           else min(cpu_count, cpuThreads)

  kIter = list(zip(kernels, itertools.repeat(kernelWriterSource), itertools.repeat(kernelWriterAssembly)))
  if cpus > 1:
    print("# Launching kernel compilation processes (cpus=%u kernels=%u)" % (cpus, len(kernels)))

    pool = multiprocessing.Pool(cpus)

    results = pool.map(processKernelSourceWithArgs, kIter)

    pool.close()
  else:
    print("# Compiling kernels (no multiprocessing, kernels=%u)" % (len(kernels)))
    if globalParameters['ShowProgressBar']:
      kIter = Utils.tqdm(kIter)

    results = list(map(processKernelSourceWithArgs, kIter))

  buildKernelSourceAndHeaderFiles(results, outputPath, kernelsWithBuildErrs, kernelSourceFile, kernelHeaderFile)


  if len(kernelsWithBuildErrs) > 0:
    print("\nKernel compilation failed in one or more subprocesses. May want to set CpuThreads=0 and re-run to make debug easier")
    printExit("** kernel compilation failure **")


  # beta-only kernels
  for kernel in kernelsBetaOnly:
    kernelWriter = kernelWriterSource
    kernelName = kernelWriter.getKernelNameBetaOnly(kernel)

    # write kernel.cpp
    if not globalParameters["MergeFiles"]:
      kernelSourceFile = open(os.path.join(outputPath, \
          "Kernels", kernelName+".cpp"), "w")
      kernelSourceFile.write(CHeader)

    (err, src) = kernelWriter.getSourceFileStringBetaOnly(kernel)
    kernelSourceFile.write(src)
    if err:
      print("*** warning: invalid kernel#%u"%kernelName)
    if not globalParameters["MergeFiles"]:
      kernelSourceFile.close()
    # write kernel.h
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile = open(os.path.join(outputPath, \
          "Kernels", kernelName + ".h"), "w")
      kernelHeaderFile.write(CHeader)
    kernelHeaderFile.write( kernelWriter.getHeaderFileStringBetaOnly(kernel))
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile.close()

  # close merged
  if globalParameters["MergeFiles"]:
    kernelHeaderFile.close()

  linkCombinedCodeObjectFile(kernels, kernelsBetaOnly, kernelWriterSource, kernelWriterAssembly, outputPath)

  stop = time.time()
  print("# Kernel Building elapsed time = %.1f secs" % (stop-start))

  print1("# Writing Solutions")
  if globalParameters["ShowProgressBar"]:
    progressBar = ProgressBar(len(solutions))
  ##############################################################################
  # Write Solutions
  ##############################################################################
  if globalParameters["MergeFiles"]:
    solutionSourceFile = open(os.path.join(outputPath, \
        "Solutions.cpp"), "w")
    solutionHeaderFile = open(os.path.join(outputPath, \
        "Solutions.h"), "w")
    if globalParameters["MergeFiles"]:
      solutionSourceFile.write(CHeader)
      solutionHeaderFile.write(CHeader)
    solutionSourceFile.write("#include \"Solutions.h\"\n")
    solutionSourceFile.write("#include <algorithm>\n")
    solutionHeaderFile.write("#include \"TensileTypes.h\"\n")
    solutionHeaderFile.write("#include \"Kernels.h\"\n")
    solutionHeaderFile.write("#include \"SolutionHelper.h\"\n")
    solutionHeaderFile.write("#include \"Tools.h\"\n")
    if globalParameters["CodeFromFiles"]:
      solutionHeaderFile.write("#include <unistd.h>\n")


  # Write a solution pointer typedef for each problemType:
  h = ""
  for problemType in problemTypes:
    #print "p=", problemType
    argListAll = solutionWriter.getArgList(problemType, True, True, True, True)
    # declare TensileSolutionPointer_ProblemType
    h += "\n// solution pointer\n"
    h += "typedef TensileStatus (*TensileSolutionPointer_%s)(\n" \
        % problemType
    for i in range(0, len(argListAll)):
      h += "    %s %s%s" % (argListAll[i][0], argListAll[i][1], ",\n" \
          if i < len(argListAll)-1 else ");\n\n")
    h += "\n"

  solutionHeaderFile.write(h)
#
  for solution in solutions:
    # get solution name
    if not globalParameters["MergeFiles"]:
      solutionFileName = solutionWriter.getSolutionName(solution)

    # write solution.cpp
    if not globalParameters["MergeFiles"]:
      solutionSourceFile = open(os.path.join(outputPath, \
          "Solutions", solutionFileName+".cpp"), "w")
      solutionSourceFile.write(CHeader)
    solutionSourceFile.write( \
        solutionWriter.getProblemSourceString(solution["ProblemType"], solution, kernelsWithBuildErrs))
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
    if globalParameters["ShowProgressBar"]:
      progressBar.increment()
  # close merged
  if not globalParameters["MergeFiles"]:
    solutionHeaderFile.close()

  if globalParameters["ExitAfterKernelGen"]:
    printExit("** Exiting after kernel generation due to ExitAfterKernelGen=1")


################################################################################
# Write Logic
################################################################################
def writeLogic(outputPath, logicData, solutionWriter ):
  print1("# Writing Library Logic")

  if not globalParameters["MergeFiles"]:
    ensurePath(os.path.join(outputPath, "Logic"))

  # Tensile.h
  h = ""
  h += "#pragma once\n"
  h += "#include \"TensileTypes.h\"\n"
  h += "#include \"SolutionHelper.h\"\n"
  h += "#include \"SolutionMapper.h\"\n"

  # TensileInternal.h
  ih = ""
  ih += "#include \"Tensile.h\"\n"

  # Tensile.cpp
  s = ""
  s += "#include \"Solutions.h\"\n"
  s += "#include \"Tensile.h\"\n"
  s += "#include \"TensileInternal.h\"\n"
  s += "#include \"SolutionMapper.h\"\n"

  ########################################
  # problemType
  for problemType in logicData:

    # function argument list
    argListSizes = solutionWriter.getArgList(problemType, False, False, False, False)
    argListData  = solutionWriter.getArgList(problemType, False, True, True, True)
    argListAll  = solutionWriter.getArgList(problemType, True, True, True, True)

    # declare tensile_ProblemType
    h += "\n// enqueue solution\n"
    h += "TensileStatus tensile_%s(\n" % problemType
    for i in range(0, len(argListData)):
      h += "    %s %s%s" \
          % (argListData[i][0], argListData[i][1], \
          ",\n" if i < len(argListData)-1 else ");\n\n")


    numSizes = problemType["TotalIndices"];
    firstStride = 0 if problemType["UseInitialStrides"] else 1
    lastStrideA = len(problemType["IndexAssignmentsA"])
    lastStrideB = len(problemType["IndexAssignmentsB"])
    lastStrideC = problemType["NumIndicesC"]
    h += "typedef ProblemKey<%u> ProblemKey_%s;\n" % (numSizes,problemType)
    h += "typedef ProblemDims<%u,%u,%u,%u,%u> ProblemDims_%s;\n" \
        % (firstStride, lastStrideC, lastStrideA, lastStrideB, numSizes, problemType)
    h += "typedef SolutionMapper<ProblemDims_%s, ProblemKey_%s> SolutionMapper_%s;\n" \
            % (problemType, problemType, problemType)

    # declare tensileGetSolutionPointer_ProblemType
    h += "\n// get solution pointer\n"
    h += "SolutionMapper_%s::SolutionRuntime *\n" % (problemType)
    h += "tensileGetSolutionPointer_%s(\n" % (problemType)
    for i in range(0, len(argListSizes)):
      h += "    %s %s%s" \
          % (argListSizes[i][0], argListSizes[i][1], \
          ",\n" if i < len(argListSizes)-1 else ");\n\n")

    # declare tensileName_
    h += "// get solution name\n"
    h += "const char * tensileGetSolutionName_%s(\n" \
        % (problemType)
    for i in range(0, len(argListSizes)):
      h += "    %s %s%s" \
          % (argListSizes[i][0], argListSizes[i][1], \
          ",\n" if i < len(argListSizes)-1 else ");\n\n")


    # get solution naming for problem type
    solutionsForProblemType = []
    for scheduleTuple in logicData[problemType]:
      solutionsForSchedule = scheduleTuple[2]
      for solution in solutionsForSchedule:
        if solution not in solutionsForProblemType:
          solutionsForProblemType.append(solution)

    # solution names for problem type
    solutionNamesForProblemType = []
    for solution in solutionsForProblemType:
      solutionName = solutionWriter.getSolutionName(solution)
      solutionNamesForProblemType.append(solutionName)

    # reset problemType source
    if not globalParameters["MergeFiles"]:
      filePrefix = "Tensile_%s" % (problemType)
      s = "#include \"TensileTypes.h\"\n"
      s = "#include \"Tensile.h\"\n"
      s += "#include \"TensileInternal.h\"\n"
      for solutionName in solutionNamesForProblemType:
        s += "#include \"%s.h\"\n" % solutionName

    ########################################
    # Per-problem constants here:
    # These are common for all schedules and thus do not include schedule name (vega,hip,etc)
    s += "\n"
    s += "/*******************************************************************************\n"
    s += "* Per-Problem Functions for %s\n" % problemType
    s += "*******************************************************************************/\n"

    s += "// Problem type include the index assignments for free, summation, batch:\n"
    s += "static const ProblemType problemType_%s( " % problemType
    s += listToInitializer(problemType["IndicesFree"]) + ", "
    s += listToInitializer(problemType["IndicesSummation"]) + ", "
    s += listToInitializer(problemType["IndicesBatch"])
    s += ");\n"

    s += "\n"
    s += "// Master solution mapper is the entry point for problem->solution mapping\n"
    s += "// There is one master solution mapper per problem type\n"
    s += "// The master solution mapper contains pointers to the solution mappers for each device\n"
    s += "static MasterSolutionMapper<ProblemDims_%s> masterSolutionMapper_%s;\n " % (problemType,problemType)


    ########################################
    # implement per-Schedule functions in source
    s += "\n"
    s += "/*******************************************************************************\n * Per-Schedule Functions\n *******************************************************************************/"
    for scheduleTuple in logicData[problemType]:

      # get logic parameters for problem type
      scheduleName  = scheduleTuple[0]
      deviceNames   = scheduleTuple[1]
      solutionsForSchedule = scheduleTuple[2]
      indexOrder    = scheduleTuple[3]
      exactLogic    = scheduleTuple[4]
      rangeLogic    = scheduleTuple[5]

      # solution names for schedule
      solutionNamesForSchedule = []
      for solution in solutionsForSchedule:
        solutionName = solutionWriter.getSolutionName(solution)
        solutionNamesForSchedule.append(solutionName)

      s += "\n\n"
      schedProbName = "%s_%s" % (scheduleName, problemType)
      s += writeSolutionAndExactTable(scheduleName, deviceNames, schedProbName, problemType, \
              solutionsForSchedule, solutionNamesForSchedule, exactLogic)


    # Per-problem function here:
    # function tensileGetSolutionPointer_ProblemType
    del schedProbName
    del scheduleName
    s += "\n// problem dims -> solution logic\n"
    s += "SolutionMapper_%s::SolutionRuntime *\n" % (problemType)
    s += "tensileGetSolutionPointer_%s(\n" % (problemType)
    for i in range(0, len(argListSizes)):
      s += "    %s %s%s" \
          % (argListSizes[i][0], argListSizes[i][1], \
          ",\n" if i < len(argListSizes)-1 else ") {\n\n")

    exactLogicStr = writeExactLogic(problemType, indexOrder, \
                                    solutionsForSchedule, exactLogic, \
                                    solutionNamesForSchedule, True)
    if rangeLogic != None:
      print("** warning: ignored ranges in logic file, these should have been expanded with ExpandRanges=1 during Tensile phase 3")
    s += "  /* exact mappings */\n"
    s += exactLogicStr
    s += "\n  return nullptr;\n"
    s += "\n}\n"

    # function tensileGetSolutionName_Schedule_ProblemType
    s += "\n// get solution name for problem dims\n"
    s += "const char * tensileGetSolutionName_%s(\n" \
        % (problemType)
    for i in range(0, len(argListSizes)):
      s += "    %s %s%s" \
          % (argListSizes[i][0], argListSizes[i][1], \
          ",\n" if i < len(argListSizes)-1 else ") {\n\n")

    exactLogicStr = writeExactLogic(problemType, indexOrder, \
                                    solutionsForSchedule, exactLogic, \
                                    solutionNamesForSchedule, False)
    s += "  /* exact mappings */\n"
    s += exactLogicStr
    #s += "  return NULL; // none\n"
    s += "\n}\n"

    ########################################
    # implement problem-type functions in source
    s += "/*******************************************************************************\n * Per-ProblemType Functions\n *******************************************************************************/"


    # declare tensile_ProblemType
    s += "\n// main call to solution; enqueues a kernel\n"
    s += "TensileStatus tensile_%s(\n" % problemType
    for i in range(0, len(argListData)):
      s += "    %s %s%s" \
          % (argListData[i][0], argListData[i][1], \
          ",\n" if i < len(argListData)-1 else ") {\n")
    s += "    auto solution = tensileGetSolutionPointer_%s(\n" % (problemType)
    for i in range(0, len(argListSizes)):
      s += "        %s%s" \
          % (argListSizes[i][1], ", " if i < len(argListSizes)-1 else ");")
      s += "\n"
    s += "    if (solution) {\n"
    s += "      TensileSolutionPointer_%s f = reinterpret_cast<TensileSolutionPointer_%s> (solution->_info->_functionPtr);\n" \
      % (problemType, problemType)
    s += "      auto solutionLock = &solution->_lock;\n"
    s += "      return f("
    for i in range(0, len(argListAll)):
      s += "%s%s" \
          % (argListAll[i][1], ", " if i < len(argListAll)-1 else ");\n")
    s += "    } else {\n"
    #s += "      printf(\"solution not valid, returning fail\\n\");"
    s += "      return tensileStatusFailure; // no solution found\n"
    s += "    }\n"
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


def writeSolutionAndExactTable(scheduleName, deviceNames, schedProbName, problemType, \
                               solutionsForSchedule, solutionNames, exactLogic):
  s = ""
  s += "namespace { // Start schedule '%s'\n" % scheduleName

  s += "// solution table - function, name, assertion requirements\n"
  s += "static const SolutionInfo solutionTable_%s[] = {\n" % (schedProbName)
  for i in range(0, len(solutionsForSchedule)):
    solution = solutionsForSchedule[i]
    solutionName = solutionNames[i]
    s += "  {(void*)%s, \"%s\", {%d, %d, %d, %d} }%s // %d" % \
      (solutionName, solutionName, \
        solution["AssertSummationElementMultiple"], \
        solution["AssertFree0ElementMultiple"], \
        solution["AssertFree1ElementMultiple"], \
        solution["AssertMinApproxSize"], \
        "," if i < len(solutionsForSchedule)-1 else "", \
        i)
    s += "\n"

  s += "};\n\n"

  # Write the exact problems here
  s += "// table of exact problem dims and selected solutionIdx\n"
  s += "static const std::pair<const ProblemKey_%s, int> embeddedExactTable_%s[] = {\n" % (problemType,schedProbName)
  for ruleIdx in range(0, len(exactLogic)):
    rule = exactLogic[ruleIdx]
    problemSize = rule[0]
    solutionIdx = rule[1][0]
    solutionGFlops = rule[1][1]
    s += " { {"
    for i in range(0, len(problemSize)):
      if i == 0:
        s += "%u" % problemSize[i];
      else:
        s += ", %u" % problemSize[i];
    s += "}, %u}" % (solutionIdx)
    s += "," if ruleIdx != len(exactLogic)-1 else " "
    s += " // %.0f GFlop/s" % (solutionGFlops)
    s += "\n";
  s += "};\n\n"

  # Create a solution mapper and init with the table above:
  s += "// The solution master constructor here adds device to the master solution mapper\n"
  s += "// The entrypoint to find a solution for this problem is through the master solution master\n"
  s += "static SolutionMapper_%s solutionMapper_%s(\n" % (problemType, schedProbName)
  s += "  \"%s\", // schedule+problem name\n" % (schedProbName) 
  s += "  {%s}, // Device names\n" % (', '.join('"{0}"'.format(w) for w in deviceNames))
  s += "  &masterSolutionMapper_%s, // add to this master solution mapper\n" % (problemType)
  s += "  solutionTable_%s, %u,\n" % (schedProbName, len(solutionsForSchedule))
  s += "  embeddedExactTable_%s, %u,\n" % (schedProbName, len(exactLogic))
  s += "  &problemType_%s);\n" % (problemType)

  s += "} // end anonymous namespace\n" 
  return s


################################################################################
# Write Range Logic Recursive
# ptr :
#   True : write logic to return the function pointer
#   False : write logic to return the function name
################################################################################
def writeExactLogic(problemType, indexOrder,
                    solutionsForSchedule, exactLogic, \
                    solutionNames, ptr):
  s = ""
  s += "  ProblemDims_%s pdims(" % problemType
  indexChars = globalParameters["IndexChars"]
  firstStride = 0 if problemType["UseInitialStrides"] else 1
  lastStrideC = problemType["NumIndicesC"]
  lastStrideA = len(problemType["IndexAssignmentsA"])
  lastStrideB = len(problemType["IndexAssignmentsB"])
  for i in range(firstStride,lastStrideC):
    if i != firstStride: s += ", "
    s += "strideC%u%s" % (i, indexChars[i])
  for i in range(firstStride,lastStrideA):
    s += ", strideA%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsA"][i]])
  for i in range(firstStride,lastStrideB):
    s += ", strideB%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsB"][i]])
  for i in range(0,len(indexOrder)):
    s += ", size%s" % indexChars[i]
  s += ");\n"

  s += "  auto solutionMapper = reinterpret_cast<SolutionMapper_%s *> (masterSolutionMapper_%s.mapper());\n"  \
      % (problemType, problemType)
  if ptr:
    s += "  return solutionMapper->getSolutionWithFallback(pdims,&masterSolutionMapper_%s);\n" % problemType
  else:
    s += "  return solutionMapper->getSolutionWithFallback(pdims,&masterSolutionMapper_%s)->_info->_name;\n" % problemType

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
def writeCMake(outputPath, solutions, kernels, libraryStaticFiles, clientName ):
  print1("# Writing Custom CMake")
  ##############################################################################
  # Min Naming
  ##############################################################################
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
  kernelWriterSource = KernelWriterSource( \
      kernelMinNaming, kernelSerialNaming)
  kernelWriterAssembly = KernelWriterAssembly( \
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
      kernelName = kernelWriterSource.getKernelName(kernel) if kernel["KernelLanguage"] == "Source" else kernelWriterAssembly.getKernelName(kernel) 
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
  arguments["MergeFiles"] = args.MergeFiles
  arguments["ShortNames"] = args.ShortNames
  arguments["LibraryPrintDebug"] = args.LibraryPrintDebug
  arguments["CodeFromFiles"] = False
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
  newMasterLibrary = None
  for logicFileName in Utils.tqdm(logicFiles, "Reading logic files"):
    (scheduleName, deviceNames, problemType, solutionsForSchedule, \
        indexOrder, exactLogic, rangeLogic, newLibrary) \
        = YAMLIO.readLibraryLogicForSchedule(logicFileName)
    if problemType not in logicData:
      logicData[problemType] = []
    logicData[problemType].append((scheduleName, deviceNames, \
        solutionsForSchedule, indexOrder, exactLogic, rangeLogic ))
    for solution in solutionsForSchedule:
      if solution not in solutions:
        solutions.append(solution)

    if newMasterLibrary is None:
        newMasterLibrary = newLibrary
    else:
        newMasterLibrary.merge(newLibrary)

  # create solution writer and kernel writer
  kernels = []
  kernelsBetaOnly = []
  for solution in solutions:
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
      if kernel not in kernels:
        kernels.append(kernel)
    solutionKernelsBetaOnly = solution.getKernelsBetaOnly()
    for kernel in solutionKernelsBetaOnly:
      if kernel not in kernelsBetaOnly:
        kernelsBetaOnly.append(kernel)

  # if any kernels are assembly, append every ISA supported

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
  kernelWriterSource = KernelWriterSource( \
      kernelMinNaming, kernelSerialNaming)
  kernelWriterAssembly = KernelWriterAssembly( \
      kernelMinNaming, kernelSerialNaming)

  # write solutions and kernels
  problemTypes = list(logicData.keys())
  writeSolutionsAndKernels(outputPath, problemTypes, solutions, kernels, kernelsBetaOnly, \
      solutionWriter, kernelWriterSource, kernelWriterAssembly)

  libraryStaticFiles = [
      "SolutionMapper.h",
      "TensileTypes.h",
      "KernelHeader.h",
      "SolutionHelper.cpp",
      "SolutionHelper.h",
      "Tools.cpp",
      "Tools.h" ]

  # write cmake
  clientName = "LibraryClient"
  writeCMake(outputPath, solutions, kernels, libraryStaticFiles, clientName )

  # write logic
  writeLogic(outputPath, logicData, solutionWriter)

  masterFile = os.path.join(outputPath, "TensileLibrary.yaml")
  newMasterLibrary.applyNaming(kernelMinNaming)
  YAMLIO.write(masterFile, Utils.state(newMasterLibrary))

  print1("# Tensile Library Writer DONE")
  print1(HR)
  print1("")

################################################################################
# Main
################################################################################
if __name__ == "__main__":
    TensileCreateLibrary()
