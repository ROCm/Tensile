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

if __name__ == "__main__":
    print("This file can no longer be run as a script.  Run 'Tensile/bin/TensileCreateLibrary' instead.")
    exit(1)

from . import Common
from . import EmbeddedData
from . import Utils
from . import YAMLIO
from .Common import globalParameters, HR, print1, print2, printExit, ensurePath, \
                   CHeader, CMakeHeader, assignGlobalParameters, ProgressBar, \
                   listToInitializer
from .KernelWriterAssembly import KernelWriterAssembly
from .KernelWriterSource import KernelWriterSource
from .SolutionStructs import Solution
from .SolutionWriter import SolutionWriter

import argparse
import collections
import itertools
import os
import shutil
import subprocess
import sys
import time

################################################################################
def processKernelSource(kernel, kernelWriterSource, kernelWriterAssembly):
    """
    Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    try:
        kernelWriter = kernelWriterSource if kernel["KernelLanguage"] == "Source" else kernelWriterAssembly
        # get kernel name
        kernelName = kernelWriter.getKernelName(kernel)
        #sys.stderr.write("kernel:%s\n"% kernelName)
        (err, src) = kernelWriter.getSourceFileString(kernel)
        header = kernelWriter.getHeaderFileString(kernel)
    except RuntimeError:
        return (1, "", "", kernelName)

    return (err, src, header, kernelName)

def getAssemblyCodeObjectFiles(kernels, kernelWriterAssembly, outputPath):
    destDir = ensurePath(os.path.join(outputPath, 'library'))
    asmDir = kernelWriterAssembly.getAssemblyDirectory()
    assemblyKernels = list([k for k in kernels if k['KernelLanguage'] == 'Assembly'])
    if len(assemblyKernels) == 0:
        return []

    if globalParameters["MergeFiles"]:
        archs = collections.defaultdict(list)
        for k in assemblyKernels:
            archs[tuple(k['ISA'])].append(k)

        coFiles = []
        for arch, archKernels in archs.items():
            objectFiles = list([kernelWriterAssembly.getKernelName(k) + '.o' \
                                for k in archKernels \
                                if k['KernelLanguage'] == 'Assembly'])
            if len(objectFiles) == 0:
                continue

            archName = 'gfx'+''.join(map(str,arch))
            coFile = os.path.join(destDir, 'TensileLibrary_{}.co'.format(archName))
            args = kernelWriterAssembly.getLinkCodeObjectArgs(objectFiles, coFile)
            subprocess.check_call(args, cwd=asmDir)
            coFiles.append(coFile)

        return coFiles

    else:
        assemblyKernelNames = [kernelWriterAssembly.getKernelName(k) for k in assemblyKernels]
        origCOFiles = [os.path.join(asmDir,  k + '.co') for k in assemblyKernelNames]
        newCOFiles  = [os.path.join(destDir, k + '.co') for k in assemblyKernelNames]
        for src, dst in Utils.tqdm(zip(origCOFiles, newCOFiles), "Copying code objects"):
            shutil.copyfile(src, dst)

        return newCOFiles

def which(p):
    exes = [p+x for x in ['', '.exe', '.bat']]
    system_path = os.environ['PATH'].split(os.pathsep)
    for dirname in system_path+['/opt/rocm/bin']:
        for exe in exes:
            candidate = os.path.join(os.path.expanduser(dirname), exe)
            if os.path.isfile(candidate):
                return candidate
    return None


def buildSourceCodeObjectFile(CxxCompiler, outputPath, kernelFile):
    buildPath = ensurePath(os.path.join(globalParameters['WorkingPath'], 'code_object_tmp'))
    destDir = ensurePath(os.path.join(outputPath, 'library'))
    (_, filename) = os.path.split(kernelFile)
    (base, _) = os.path.splitext(filename)

    objectFilename = base + '.o'
    objectFilepath = os.path.join(buildPath, objectFilename)

    soFilename = base + '.so'
    soFilepath = os.path.join(buildPath, soFilename)

    archs = ['gfx'+''.join(map(str,arch)) for arch in globalParameters['SupportedISA'] \
             if globalParameters["AsmCaps"][arch]["SupportedISA"]]

    archFlags = ['--amdgpu-target=' + arch for arch in archs]

    if (CxxCompiler == 'hcc'):

      hipFlags = subprocess.check_output([which('hcc-config'), '--cxxflags']).decode().split(' ')
      # when HCC_HOME is defined -I/opt/rocm/include is *not* part of
      # hcc-config --cxxflags; so we need hipconfig -C to be safe
      hipFlags += subprocess.check_output([which('hipconfig'), '-C']).decode().split(' ')
      hipLinkFlags = subprocess.check_output([which('hcc-config'), '--ldflags', '--shared']).decode().split(' ')

      hipFlags += ['-I', outputPath, '-fPIC']

      compileArgs = [which('hcc')] + hipFlags + [kernelFile, '-c', '-o', objectFilepath]

      linkArgs = [globalParameters['AssemblerPath']] + hipLinkFlags + archFlags + [objectFilepath, '-shared', '-o', soFilepath]
      extractArgs = [globalParameters['ExtractKernelPath'], '-i', soFilename]

      #print(' '.join(compileArgs))
      subprocess.check_call(compileArgs)

      #print(' '.join(linkArgs))
      subprocess.check_call(linkArgs)

      #print(' '.join(extractArgs))
      subprocess.check_call(extractArgs, cwd=buildPath)

      coFilenames = ["{0}-000-{1}.hsaco".format(soFilename, arch) for arch in archs]
    elif (CxxCompiler == "hipcc"):

      hipFlags = ["--genco", "-D__HIP_HCC_COMPAT_MODE__=1"]

      hipFlags += ['-I', outputPath]

      compileArgs = [which('hipcc')] + hipFlags + archFlags + [kernelFile, '-c', '-o', soFilepath]

      #print(' '.join(compileArgs))
      subprocess.check_call(compileArgs)

      coFilenames = [soFilename]
    else:
      raise RuntimeError("Unknown compiler {}".format(CxxCompiler))

    extractedCOs = [os.path.join(buildPath, name) for name in coFilenames]
    destCOs = [os.path.join(destDir, name) for name in coFilenames]

    for (src, dst) in zip(extractedCOs, destCOs):
        shutil.copyfile(src, dst)

    return destCOs

def buildSourceCodeObjectFiles(CxxCompiler, kernelFiles, outputPath):
    args = zip(itertools.repeat(CxxCompiler), itertools.repeat(outputPath), kernelFiles)

    coFiles = Common.ParallelMap(buildSourceCodeObjectFile, args, "Compiling source kernels",
                                 method=lambda x: x.starmap)

    return itertools.chain.from_iterable(coFiles)

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
      "${ASM} -x assembler -target amdgcn-amd-amdhsa %s $@ -c -o $f.o $f.s\n" % \
      ("-mno-code-object-v3" if \
      globalParameters["AsmCaps"][defaultIsa]["HasCodeObjectV3"] and \
      globalParameters["CodeObjectVersion"] == "V2" else "-mcode-object-v3"))
    assemblerFile.write("${ASM} -target amdgcn-amd-amdhsa $f.o -o $f.co\n")
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

  sourceFilenames = []

  for (err,src,header,kernelName) in results:
    if err:
      kernelsWithBuildErrs[kernelName] = err
      #print "*** warning: invalid kernel#%s"%kernelName

    # write kernel.cpp
    if not globalParameters["MergeFiles"]:
      filename = os.path.join(outputPath, "Kernels", kernelName+".cpp")
      sourceFilenames.append(filename)
      kernelSourceFile = open(filename, "w")
      kernelSourceFile.write(CHeader)

    kernelSourceFile.write(src)

    if not globalParameters["MergeFiles"]:
      kernelSourceFile.close()
      # write kernel.h
      kernelHeaderFile = open(os.path.join(outputPath, "Kernels", kernelName+".h"), "w")
      kernelHeaderFile.write(CHeader)

    kernelHeaderFile.write(header)

    if not globalParameters["MergeFiles"]:
      kernelHeaderFile.close()

  return sourceFilenames

################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
def writeSolutionsAndKernels(outputPath, CxxCompiler, problemTypes, solutions, kernels, kernelsBetaOnly, \
    solutionWriter, kernelWriterSource, kernelWriterAssembly, errorTolerant=False):
  start = time.time()

  codeObjectFiles = []

  print1("# Writing Kernels...")
  if not globalParameters["MergeFiles"]:
    ensurePath(os.path.join(outputPath, "Solutions"))
    ensurePath(os.path.join(outputPath, "Kernels"))

  ##############################################################################
  # Write Kernels
  ##############################################################################
  kernelFiles = []
  if globalParameters["MergeFiles"]:
    kernelSourceFilename = os.path.join(outputPath, "Kernels.cpp")
    kernelHeaderFilename = os.path.join(outputPath, "Kernels.h")

    kernelFiles.append(kernelSourceFilename)
    kernelSourceFile = open(kernelSourceFilename, "w")
    kernelHeaderFile = open(kernelHeaderFilename, "w")
    kernelSourceFile.write(CHeader)
    kernelHeaderFile.write(CHeader)
    #kernelSourceFile.write("#define HCC_ENABLE_ACCELERATOR_PRINTF\n")
    kernelSourceFile.write("#include \"Kernels.h\"\n")
    kernelHeaderFile.write("#pragma once\n")
    if globalParameters["RuntimeLanguage"] == "HIP":
      kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
      kernelHeaderFile.write("#include <hip/hip_hcc.h>\n\n")
    kernelHeaderFile.write("#include \"KernelHeader.h\"\n\n")
  else:
    kernelSourceFile = None
    kernelHeaderFile = None

  kernelsWithBuildErrs = {}

  prepAsm()

  kIter = zip(kernels, itertools.repeat(kernelWriterSource), itertools.repeat(kernelWriterAssembly))
  results = Common.ParallelMap(processKernelSource, kIter, "Generating kernels", method=lambda x: x.starmap)

  removeKernels = []
  removeSolutions = []
  removeResults = []
  for kernIdx in range(0, len(results)):
    (err,src,header,kernelName) = results[kernIdx]
    if(err == -2):
      removeKernels.append(kernels[kernIdx])
      removeSolutions.append(solutions[kernIdx])
      removeResults.append(results[kernIdx])
  for kern in removeKernels:
      kernels.remove(kern)
  for solut in removeSolutions:
      solutions.remove(solut)
  for rel in removeResults:
      results.remove(rel)

  kernelFiles += buildKernelSourceAndHeaderFiles(results, outputPath, kernelsWithBuildErrs, kernelSourceFile, kernelHeaderFile)

  kernelsToBuild = list(kernels)
  if errorTolerant:
      def success(kernel):
          writer = kernelWriterAssembly if kernel['KernelLanguage'] == 'Assembly' else kernelWriterSource
          kernelName = writer.getKernelName(kernel)
          return kernelName not in kernelsWithBuildErrs
      kernelsToBuild = list(filter(success, kernelsToBuild))

  if False:#len(kernelsWithBuildErrs) > 0:
    print("\nKernel compilation failed in one or more subprocesses. May want to set CpuThreads=0 and re-run to make debug easier")
    printExit("** kernel compilation failure **")


  # beta-only kernels
  for kernel in kernelsBetaOnly:
    kernelWriter = kernelWriterSource
    kernelName = kernelWriter.getKernelNameBetaOnly(kernel)

    # write kernel.cpp
    if not globalParameters["MergeFiles"]:
      kernelSourceFilename = os.path.join(outputPath, "Kernels", kernelName+".cpp")
      kernelSourceFile = open(kernelSourceFilename, "w")
      kernelSourceFile.write(CHeader)
      kernelFiles.append(kernelSourceFilename)

    (err, src) = kernelWriter.getSourceFileStringBetaOnly(kernel)
    kernelSourceFile.write(src)
    if err:
      print("*** warning: invalid kernel#%u"%kernelName)
    if not globalParameters["MergeFiles"]:
      kernelSourceFile.close()
    # write kernel.h
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile = open(os.path.join(outputPath, "Kernels", kernelName + ".h"), "w")
      kernelHeaderFile.write(CHeader)
    kernelHeaderFile.write( kernelWriter.getHeaderFileStringBetaOnly(kernel))
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile.close()

  # close merged
  if globalParameters["MergeFiles"]:
    kernelSourceFile.close()
    kernelHeaderFile.close()

  kernelsToBuild += kernelsBetaOnly

  codeObjectFiles += buildSourceCodeObjectFiles(CxxCompiler, kernelFiles, outputPath)
  codeObjectFiles += getAssemblyCodeObjectFiles(kernelsToBuild, kernelWriterAssembly, outputPath)

  stop = time.time()
  print("# Kernel Building elapsed time = %.1f secs" % (stop-start))

  print1("# Writing Solutions")
  if globalParameters["ShowProgressBar"]:
    progressBar = ProgressBar(len(solutions))
  ##############################################################################
  # Write Solutions
  ##############################################################################

  solutionSourceFilename = os.path.join(outputPath, "Solutions.cpp")
  solutionHeaderFilename = os.path.join(outputPath, "Solutions.h")

  solutionSourceFile = open(solutionSourceFilename, "w")
  solutionHeaderFile = open(solutionHeaderFilename, "w")
  solutionSourceFile.write(CHeader)
  solutionHeaderFile.write(CHeader)

  solutionSourceFile.write("#include \"Solutions.h\"\n")
  solutionSourceFile.write("#include <algorithm>\n")

  solutionHeaderFile.write("#include \"TensileTypes.h\"\n")
  solutionHeaderFile.write("#include \"SolutionHelper.h\"\n")
  solutionHeaderFile.write("#include \"Tools.h\"\n")
  if globalParameters["CodeFromFiles"]:
    solutionHeaderFile.write("#include <unistd.h>\n")
  if globalParameters["MergeFiles"]:
    solutionHeaderFile.write("#include \"Kernels.h\"\n")


  # Write a solution pointer typedef for each problemType:
  h = ""
  for problemType in problemTypes:
    #print "p=", problemType
    argListAll = solutionWriter.getArgList(problemType, True, True, True, True)
    # declare TensileSolutionPointer_ProblemType
    h += "\n// solution pointer\n"
    h += "typedef TensileStatus (*TensileSolutionPointer_%s)(\n" % problemType
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

  return codeObjectFiles

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

  sourceIncludes = ""
  sourceIncludes += "#include \"Solutions.h\"\n"
  sourceIncludes += "#include \"Tensile.h\"\n"
  sourceIncludes += "#include \"TensileInternal.h\"\n"
  sourceIncludes += "#include \"SolutionMapper.h\"\n"

  s = sourceIncludes

  ########################################
  # problemType
  for problemType in logicData:

    # function argument list
    argListSizes = solutionWriter.getArgList(problemType, False, False, False, False)
    argListData  = solutionWriter.getArgList(problemType, False, True, True, True)
    argListAll  = solutionWriter.getArgList(problemType, True, True, True, True)
    
    # tensile initializer
    h += "\nvoid tensileInitialize();\n\n"

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
    lastStrideD = problemType["NumIndicesC"]
    h += "typedef ProblemKey<%u> ProblemKey_%s;\n" % (numSizes,problemType)
    h += "typedef ProblemDims<%u,%u,%u,%u,%u,%u> ProblemDims_%s;\n" \
        % (firstStride, lastStrideD, lastStrideC, lastStrideA, lastStrideB, numSizes, problemType)
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
      s = sourceIncludes

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
    s += listToInitializer(problemType["IndicesBatch"]) + ", "
    s += listToInitializer(problemType["IndexAssignmentsA"]) + ", "
    s += listToInitializer(problemType["IndexAssignmentsB"])
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

  s += "\n"
  s += writeTensileInitialize(logicData)

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


def writeTensileInitialize(logicData):

  s = "/*******************************************************************************\n"
  s += "* Tensilze initializer\n"
  s += "*******************************************************************************/\n"
  s += "void tensileInitialize() {\n"

  for problemType in logicData:
    s += "  masterSolutionMapper_%s.initialize();\n" % problemType
    
    for scheduleTuple in logicData[problemType]:
      scheduleName  = scheduleTuple[0]
      deviceNames   = scheduleTuple[1]


      schedProbName = "%s_%s" % (scheduleName, problemType)
      s += "  solutionMapper_%s.initializeMappers(" % (schedProbName)
      s += "{%s}," % (', '.join('"{0}"'.format(w) for w in deviceNames))
      s += "&masterSolutionMapper_%s);\n" % (problemType)
      
  s += "}"

  return s

def writeSolutionAndExactTable(scheduleName, deviceNames, schedProbName, problemType, \
                               solutionsForSchedule, solutionNames, exactLogic):
  s = ""
  s += "namespace { // Start schedule '%s'\n" % scheduleName

  s += "// solution table - function, name, assertion requirements\n"
  s += "static const SolutionInfo solutionTable_%s[] = {\n" % (schedProbName)
  for i in range(0, len(solutionsForSchedule)):
    solution = solutionsForSchedule[i]
    solutionName = solutionNames[i]
    s += "  {(void*)%s, \"%s\", {%d, %d, %d, %d, %d, %d, %d} }%s // %d" % \
      (solutionName, solutionName, \
        solution["AssertSummationElementMultiple"], \
        solution["AssertFree0ElementMultiple"], \
        solution["AssertFree1ElementMultiple"], \
        0, \
        solution["LdcEqualsLdd"], \
        solution["PackBatchDims"]==2, \
        solution["PackBatchDims"]==1, \
        "," if i < len(solutionsForSchedule)-1 else "", \
        i)
    s += "\n"

  s += "};\n\n"

  # Write the exact problems here
  s += "// table of exact problem dims and selected solutionIdx\n"
  s += "static const std::pair<const ProblemKey_%s, int> embeddedExactTable_%s[] = {\n" % (problemType,schedProbName)
  numSizes = problemType["TotalIndices"]
  for ruleIdx in range(0, len(exactLogic)):
    rule = exactLogic[ruleIdx]
    problemSize = rule[0][:numSizes]
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
  lastStrideD = problemType["NumIndicesC"]
  lastStrideC = problemType["NumIndicesC"]
  lastStrideA = len(problemType["IndexAssignmentsA"])
  lastStrideB = len(problemType["IndexAssignmentsB"])
  for i in range(firstStride,lastStrideD):
    if i != firstStride: s += ", "
    s += "strideD%u%s" % (i, indexChars[i])
  for i in range(firstStride,lastStrideC):
    s += ", strideC%u%s" % (i, indexChars[i])
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
  s += " dataD, dataC, dataA, dataB, alpha"
  if problemType["UseBeta"]:
    s += ", beta"
  s += ", offsetC, offsetA, offsetB"
  firstStride = 1
  if problemType["UseInitialStrides"]:
    firstStride = 0
  lastStrideD = problemType["NumIndicesC"]
  lastStrideC = problemType["NumIndicesC"]
  lastStrideA = len(problemType["IndexAssignmentsA"])
  lastStrideB = len(problemType["IndexAssignmentsB"])
  for i in range(firstStride,lastStrideD):
    s += ", strideD%u%s" % (i, indexChars[i])
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
    shutil.copy( os.path.join(globalParameters["SourcePath"], fileName), \
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
  argParser.add_argument("LogicPath",       help="Path to LibraryLogic.yaml files.")
  argParser.add_argument("OutputPath",      help="Where to write library files?")
  argParser.add_argument("RuntimeLanguage", help="Which runtime language?", choices=["OCL", "HIP", "HSA"])
  argParser.add_argument("--cxx-compiler",           dest="CxxCompiler",       choices=["hcc", "hipcc"],       action="store", default="hcc")
  argParser.add_argument("--code-object-version",    dest="CodeObjectVersion", choices=["V2", "V3"], action="store", default="V2")
  argParser.add_argument("--merge-files",            dest="MergeFiles",        action="store_true")
  argParser.add_argument("--no-merge-files",         dest="MergeFiles",        action="store_false")
  argParser.add_argument("--short-file-names",       dest="ShortNames",        action="store_true")
  argParser.add_argument("--no-short-file-names",    dest="ShortNames",        action="store_false")
  argParser.add_argument("--library-print-debug",    dest="LibraryPrintDebug", action="store_true")
  argParser.add_argument("--no-library-print-debug", dest="LibraryPrintDebug", action="store_false")
  argParser.add_argument("--embed-library",          dest="EmbedLibrary",
                         help="Embed (new) library files into static variables.  Specify the name of the library.")

  argParser.add_argument("--embed-library-key",      dest="EmbedLibraryKey", default=None,
                         help="Access key for embedding library files.")
  args = argParser.parse_args()

  logicPath = args.LogicPath
  outputPath = args.OutputPath
  CxxCompiler = args.CxxCompiler
  print2("OutputPath: %s" % outputPath)
  ensurePath(outputPath)
  arguments = {}
  arguments["RuntimeLanguage"] = args.RuntimeLanguage
  arguments["CodeObjectVersion"] = args.CodeObjectVersion
  arguments["CxxCompiler"] = args.CxxCompiler
  arguments["MergeFiles"] = args.MergeFiles
  arguments["ShortNames"] = args.ShortNames
  arguments["LibraryPrintDebug"] = args.LibraryPrintDebug
  arguments["CodeFromFiles"] = False
  arguments["EmbedLibrary"] = args.EmbedLibrary
  assignGlobalParameters(arguments)

  print1("# CodeObjectVersion from TensileCreateLibrary: %s" % arguments["CodeObjectVersion"])
  print1("# CxxCompiler       from TensileCreateLibrary: %s" % CxxCompiler)

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

  libraries = Common.ParallelMap(YAMLIO.readLibraryLogicForSchedule, logicFiles, "Reading logic files")

  for logic in Utils.tqdm(libraries, "Processing logic data"):
    (scheduleName, deviceNames, problemType, solutionsForSchedule, \
       indexOrder, exactLogic, rangeLogic, newLibrary) = logic

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

  libraryStaticFiles = [
      "SolutionMapper.h",
      "TensileTypes.h",
      "tensile_bfloat16.h",
      "KernelHeader.h",
      "SolutionHelper.cpp",
      "SolutionHelper.h",
      "Tools.cpp",
      "Tools.h" ]

  # write cmake
  clientName = "LibraryClient"
  writeCMake(outputPath, solutions, kernels, libraryStaticFiles, clientName )

  # write solutions and kernels
  problemTypes = list(logicData.keys())
  codeObjectFiles = writeSolutionsAndKernels(outputPath, CxxCompiler, problemTypes, solutions,
                                             kernels, kernelsBetaOnly,
                                             solutionWriter,
                                             kernelWriterSource, kernelWriterAssembly)

  # write logic
  writeLogic(outputPath, logicData, solutionWriter)

  newLibraryDir = ensurePath(os.path.join(outputPath, 'library'))
  
  masterFile = os.path.join(newLibraryDir, "TensileLibrary.yaml")
  newMasterLibrary.applyNaming(kernelMinNaming)
  YAMLIO.write(masterFile, Utils.state(newMasterLibrary))

  if args.EmbedLibrary is not None:
      embedFileName = os.path.join(outputPath, "library/{}.cpp".format(args.EmbedLibrary))
      with EmbeddedData.EmbeddedDataFile(embedFileName) as embedFile:
          embedFile.embed_file(newMasterLibrary.cpp_base_class, masterFile, nullTerminated=True,
                               key=args.EmbedLibraryKey)

          for co in Utils.tqdm(codeObjectFiles):
              embedFile.embed_file("SolutionAdapter", co, nullTerminated=False,
                                   key=args.EmbedLibraryKey)

  print1("# Tensile Library Writer DONE")
  print1(HR)
  print1("")

