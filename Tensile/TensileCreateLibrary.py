################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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
from . import LibraryIO
from . import Utils
from .Common import globalParameters, HR, print1, print2, printExit, ensurePath, \
                   CHeader, CMakeHeader, assignGlobalParameters, \
                   listToInitializer
from .KernelWriterAssembly import KernelWriterAssembly
from .KernelWriterSource import KernelWriterSource
from .KernelWriter import KernelWriter
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
from copy import deepcopy

################################################################################
def processKernelSource(kernel, kernelWriterSource, kernelWriterAssembly):
    """
    Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    try:
        kernelWriter = kernelWriterSource if kernel["KernelLanguage"] == "Source" else kernelWriterAssembly
        # get kernel name
        kernelName = kernelWriter.getKernelFileBase(kernel)
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

    archs = collections.defaultdict(list)
    for k in assemblyKernels:
      archs[tuple(k['ISA'])].append(k)
    coFiles = []
    for arch, archKernels in archs.items():
      archName = 'gfx'+''.join(map(str,arch))
      objectFiles = list([kernelWriterAssembly.getKernelFileBase(k) + '.o' \
                          for k in archKernels \
                          if k['KernelLanguage'] == 'Assembly'])
      if len(objectFiles) == 0:
        continue
      if globalParameters["MergeFiles"]:
      #  archName = 'gfx'+''.join(map(str,arch))
        coFile = os.path.join(destDir, 'TensileLibrary_{}.co'.format(archName))
        if "PackageLibrary" in globalParameters and globalParameters["PackageLibrary"]:
          destArchDir = ensurePath(os.path.join(destDir, archName))
          coFile = os.path.join(destArchDir, 'TensileLibrary_{}.co'.format(archName))
        args = kernelWriterAssembly.getLinkCodeObjectArgs(objectFiles, coFile)
        subprocess.check_call(args, cwd=asmDir)
        coFiles.append(coFile)
      else:
        assemblyKernelNames = [kernelWriterAssembly.getKernelFileBase(k) for k in archKernels]
        origCOFiles = [os.path.join(asmDir,  k + '.co') for k in assemblyKernelNames]
        newCOFiles  = []
        if globalParameters["PackageLibrary"]:
          newCOFiles = [os.path.join(destDir, archName, k + '.co') for k in assemblyKernelNames]
        else:
          newCOFiles = [os.path.join(destDir, k + '_' + archName + '.co') for k in assemblyKernelNames]

        for src, dst in Utils.tqdm(zip(origCOFiles, newCOFiles), "Copying code objects"):
          shutil.copyfile(src, dst)
        coFiles += newCOFiles

    return coFiles

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

    def isSupported(arch):
        return globalParameters["AsmCaps"][arch]["SupportedISA"] and \
               globalParameters["AsmCaps"][arch]["SupportedSource"]

    archs = ['gfx'+''.join(map(str,arch)) for arch in globalParameters['SupportedISA'] \
             if isSupported(arch)]

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
      extractArgs = [globalParameters['ExtractKernelPath'], '-i', os.path.join(buildPath,soFilename)]

      if globalParameters["PrintCodeCommands"]:
        print(' '.join(compileArgs))
      subprocess.check_call(compileArgs)

      if globalParameters["PrintCodeCommands"]:
        print(' '.join(linkArgs))
      subprocess.check_call(linkArgs)

      if globalParameters["PrintCodeCommands"]:
        print(' '.join(extractArgs))
      subprocess.check_call(extractArgs, cwd=buildPath)

      coFilenames = ["{0}-000-{1}.hsaco".format(soFilename, arch) for arch in archs]
    elif (CxxCompiler == "hipcc"):

      hipFlags = ["--genco", "-D__HIP_HCC_COMPAT_MODE__=1"] #needs to be fixed when Maneesh's change is made available

      hipFlags += ['-I', outputPath]

      compileArgs = [which('hipcc')] + hipFlags + archFlags + [kernelFile, '-c', '-o', os.path.join(buildPath, objectFilename)]

      if globalParameters["PrintCodeCommands"]:
        print('hipcc:', ' '.join(compileArgs))
      subprocess.check_call(compileArgs)

      for arch in archs:
        infile = os.path.join(buildPath, objectFilename)
        outfile = os.path.join(buildPath, "{0}-000-{1}.hsaco".format(soFilename, arch))
        bundlerArgs = [globalParameters["ClangOffloadBundlerPath"], "-type=o", "-targets=hip-amdgcn-amd-amdhsa-%s" % arch, "-inputs=%s" % infile, "-outputs=%s" % outfile, "-unbundle"]
        if globalParameters["PrintCodeCommands"]:
          print(' '.join(bundlerArgs))
        subprocess.check_call(bundlerArgs)

      coFilenames = ["{0}-000-{1}.hsaco".format(soFilename, arch) for arch in archs]
    else:
      raise RuntimeError("Unknown compiler {}".format(CxxCompiler))

    destCosList = []
    if "PackageLibrary" in globalParameters and globalParameters["PackageLibrary"]:
      for arch in archs:
        ensurePath(os.path.join(destDir, arch))
        archCoFilenames = [name for name in coFilenames if arch in name]
        extractedCOs = [os.path.join(buildPath, name) for name in archCoFilenames]
        destCOs = [os.path.join(destDir, arch, name) for name in archCoFilenames]
        destCosList += destCOs
        if globalParameters["PrintCodeCommands"]:
          print ("# copy source code objects    : ", extractedCOs)
          print ("# to dest source code objects : ", destCOs)
        for (src, dst) in zip(extractedCOs, destCOs):
          shutil.copyfile(src, dst)
    else:
      coFilenames = [name for name in coFilenames]
      extractedCOs = [os.path.join(buildPath, name) for name in coFilenames]
      destCOs = [os.path.join(destDir, name) for name in coFilenames]
      destCosList += destCOs
      for (src, dst) in zip(extractedCOs, destCOs):
        shutil.copyfile(src, dst)

    return destCosList

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

    # Don't create a file for empty kernels.
    #####
    if len(src.strip()) == 0 and globalParameters["NewClient"] > 1:
      continue

    #if kernelSourceFile:
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

  # For NewClient builds: Push working path into build_tmp folder because there may be more than
  # one process running this script. This is to avoid build directory clashing.
  # NOTE: file paths must not contain the lower case word 'kernel' or the
  # /opt/rocm/bin/extractkernel will fail.
  # See buildSourceCodeObjectFile:167 for the call to this binary.
  if globalParameters["NewClient"] == 2:
    Common.pushWorkingPath('build_tmp')
    Common.pushWorkingPath(os.path.basename(outputPath).upper())

  print1("# Writing Kernels...")
  kernelFiles = []
  kernelSourceFile = None
  kernelHeaderFile = None

  if not globalParameters["MergeFiles"]:
    if globalParameters["LegacyComponents"]:
      ensurePath(os.path.join(outputPath, "Solutions"))
    ensurePath(os.path.join(outputPath, "Kernels"))

  ##############################################################################
  # Write Kernels
  ##############################################################################
  if globalParameters["MergeFiles"]:
    kernelSourceFilename = os.path.join(outputPath, "Kernels.cpp")
    kernelHeaderFilename = os.path.join(outputPath, "Kernels.h")

    kernelFiles.append(kernelSourceFilename)
    kernelSourceFile = open(kernelSourceFilename, "w")
    kernelHeaderFile = open(kernelHeaderFilename, "w")
    kernelSourceFile.write(CHeader)
    kernelHeaderFile.write(CHeader)
    kernelSourceFile.write("#include \"Kernels.h\"\n")
    kernelHeaderFile.write("#pragma once\n")
    if globalParameters["RuntimeLanguage"] == "HIP":
      kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
      kernelHeaderFile.write("#include <hip/hip_ext.h>\n\n")
    kernelHeaderFile.write("#include \"KernelHeader.h\"\n\n")

  kernelsWithBuildErrs = {}

  prepAsm()

  kIter = zip(kernels, itertools.repeat(kernelWriterSource), itertools.repeat(kernelWriterAssembly))
  results = Common.ParallelMap(processKernelSource, kIter, "Generating kernels", method=lambda x: x.starmap)
  #do we need this?
  #print(len(results))

  removeKernels = []
  removeSolutions = []
  removeResults = []
  for kernIdx, res in Utils.tqdm(enumerate(results)):
    (err,src,header,kernelName) = res
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
    if kernelSourceFile:
      kernelSourceFile.close()
    if kernelHeaderFile:
      kernelHeaderFile.close()

  kernelsToBuild += kernelsBetaOnly

  codeObjectFiles += buildSourceCodeObjectFiles(CxxCompiler, kernelFiles, outputPath)
  codeObjectFiles += getAssemblyCodeObjectFiles(kernelsToBuild, kernelWriterAssembly, outputPath)

  stop = time.time()
  print("# Kernel Building elapsed time = %.1f secs" % (stop-start))

  if globalParameters["LegacyComponents"]:
    print1("# Writing Solutions")
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

    for solution in Utils.tqdm(solutions):
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
    # close merged
    if not globalParameters["MergeFiles"]:
      solutionHeaderFile.close()

    if globalParameters["ExitAfterKernelGen"]:
      printExit("** Exiting after kernel generation due to ExitAfterKernelGen=1")

  if globalParameters["NewClient"] == 2:
    Common.popWorkingPath() # build_tmp
    Common.popWorkingPath() # workingDir

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
  for problemType in Utils.tqdm(logicData):

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


    numSizes = problemType["TotalIndices"]
    assert(not problemType["UseInitialStridesCD"])
    firstStride = 0 if problemType["UseInitialStridesAB"] else 1
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
        solution["AssertMinApproxSize"], \
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
        s += "%u" % problemSize[i]
      else:
        s += ", %u" % problemSize[i]
    s += "}, %u}" % (solutionIdx)
    s += "," if ruleIdx != len(exactLogic)-1 else " "
    s += " // %.0f GFlop/s" % (solutionGFlops)
    s += "\n"
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
  firstStrideAB = 0 if problemType["UseInitialStridesAB"] else 1
  firstStrideCD = 0 if problemType["UseInitialStridesCD"] else 1
  lastStrideD = problemType["NumIndicesC"]
  lastStrideC = problemType["NumIndicesC"]
  lastStrideA = len(problemType["IndexAssignmentsA"])
  lastStrideB = len(problemType["IndexAssignmentsB"])
  for i in range(firstStrideCD,lastStrideD):
    if i != firstStrideCD: s += ", "
    s += "strideD%u%s" % (i, indexChars[i])
  for i in range(firstStrideCD,lastStrideC):
    s += ", strideC%u%s" % (i, indexChars[i])
  for i in range(firstStrideAB,lastStrideA):
    s += ", strideA%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsA"][i]])
  for i in range(firstStrideAB,lastStrideB):
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
  firstStrideAB = firstStrideCD = 1
  if problemType["UseInitialStridesAB"]:
    firstStrideAB = 0
  if problemType["UseInitialStridesCD"]:
    firstStrideCD = 0
  lastStrideD = problemType["NumIndicesC"]
  lastStrideC = problemType["NumIndicesC"]
  lastStrideA = len(problemType["IndexAssignmentsA"])
  lastStrideB = len(problemType["IndexAssignmentsB"])
  for i in range(firstStrideCD,lastStrideD):
    s += ", strideD%u%s" % (i, indexChars[i])
  for i in range(firstStrideCD,lastStrideC):
    s += ", strideC%u%s" % (i, indexChars[i])
  for i in range(firstStrideAB,lastStrideA):
    s += ", strideA%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsA"][i]])
  for i in range(firstStrideAB,lastStrideB):
    s += ", strideB%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsB"][i]])
  for i in range(0, problemType["TotalIndices"]):
    s += ", size%s" % indexChars[i]
  s += ", stream, numInputEvents, inputEvents, outputEvent )"
  return s

def buildObjectFileNames(solutionWriter, kernelWriterSource, kernelWriterAssembly, solutions, kernels, betaOnlyKernels):

  # Build lists of output object names
  sourceKernelNames = []
  asmKernelNames = []
  betaKernelNames = []
  solutionFiles = []
  sourceKernelFiles = []
  asmKernelFiles = []
  sourceLibFiles = []
  asmLibFiles = []

  sourceKernels = list([k for k in kernels if k['KernelLanguage'] == 'Source'])
  asmKernels = list([k for k in kernels if k['KernelLanguage'] == 'Assembly'])

  # Helper for architecture
  def isSupported(arch):
        return globalParameters["AsmCaps"][arch]["SupportedISA"] and \
               globalParameters["AsmCaps"][arch]["SupportedSource"]

  # Build a list of kernel object names.
  for kernel in sourceKernels:
    sourceKernelNames += [kernelWriterSource.getKernelFileBase(kernel)]

  for kernel in asmKernels:
    asmKernelNames += [kernelWriterAssembly.getKernelFileBase(kernel)]

  for kernel in betaOnlyKernels:
    betaKernelNames += [kernelWriterSource.getKernelNameBetaOnly(kernel)]

  # Source based kernels are built for all supported architectures
  sourceArchs = ['gfx'+''.join(map(str,arch)) for arch in globalParameters['SupportedISA'] \
             if isSupported(arch)]

  # Asm based kernels target the configured ISA
  asmArchs = collections.defaultdict(list)
  for (kernelName, kernel) in zip(asmKernelNames, asmKernels):
    asmArchs[kernelName].append('gfx'+''.join(map(str, tuple(kernel['ISA']))))

  # Build list of solution files
  if globalParameters["LegacyComponents"]:
    if not globalParameters["MergeFiles"]:
      for solution in solutions:
        solutionFiles += [
          "%s.h"   % (solutionWriter.getSolutionName(solution)),
          "%s.cpp" % (solutionWriter.getSolutionName(solution))]
    else:
      solutionFiles += ["Solutions.h", "Solutions.cpp"]

  # Build a list of source files
  if not globalParameters["MergeFiles"]:
    for kernelName in (sourceKernelNames + asmKernelNames):
      sourceKernelFiles += [
        "%s.h"   % (kernelName),
        "%s.cpp" % (kernelName)]
  else:
    sourceKernelFiles += ["Kernels.h", "Kernels.cpp"]

  # Build a list of assembly files
  for asmKernelName in asmKernelNames:
      asmKernelFiles += [
        "%s.s"  % (asmKernelName),
        "%s.o"  % (asmKernelName),
        "%s.co" % (asmKernelName)]

  # Build a list of lib names from source
  cxxCompiler = globalParameters["CxxCompiler"]
  if not globalParameters["MergeFiles"]:

    allSources = sourceKernelNames + betaKernelNames
    if globalParameters["NewClient"] <= 1:
      allSources += asmKernelNames

    for kernelName in (allSources):
      # Hcc compiler expects to have a different extension than hipclang.
      # Hcc compiler also makes one file per architecture
      if (cxxCompiler == 'hcc' or cxxCompiler == 'hipcc'):
        sourceLibFiles += ["%s.so-000-%s.hsaco" % (kernelName, arch) for arch in sourceArchs]
      else:
        raise RuntimeError("Unknown compiler {}".format(cxxCompiler))
  else: # Merge
    if (cxxCompiler == 'hcc' or cxxCompiler == 'hipcc'):
      sourceLibFiles += ["Kernels.so-000-%s.hsaco" % (arch) for arch in sourceArchs]
    else:
      raise RuntimeError("Unknown compiler {}".format(cxxCompiler))

  # Build a list of asm lib names
  if globalParameters["MergeFiles"]:
    # Find all unique arch values for current asm kernels
    uniqueArchs = set(itertools.chain(*asmArchs.values()))
    asmLibFiles += ["TensileLibrary_%s.co" % (arch) for arch in uniqueArchs]
  else:
    for asmKernelName, archs in asmArchs.items():
      asmLibFiles += ["%s_%s.co" % (asmKernelName, str(arch)) for arch in archs]

  return (solutionFiles, sourceKernelFiles, asmKernelFiles, sourceLibFiles, asmLibFiles)

def buildObjectFilePaths(prefixDir, solutionFiles, sourceKernelFiles, asmKernelFiles, sourceLibFiles, asmLibFiles):

  solutionPaths = []
  sourceKernelPaths = []
  asmKernelPaths = []
  sourceLibPaths = []
  asmLibPaths = []

  # Build full paths for solution files
  if globalParameters["LegacyComponents"]:
    solutionDir = ""
    if not globalParameters["MergeFiles"]:
      solutionDir = os.path.join(prefixDir, "Solutions")
    else:
      solutionDir = prefixDir

    for solutionFile in solutionFiles:
      solutionPaths += [ os.path.join(solutionDir, solutionFile) ]

  # Build full paths for source kernel files
  sourceKernelDir = ""
  if not globalParameters["MergeFiles"]:
    sourceKernelDir = os.path.join(prefixDir, "Kernels")
  else:
    sourceKernelDir = prefixDir

  for sourceKernelFile in sourceKernelFiles:
    sourceKernelPaths += [ os.path.join(sourceKernelDir, sourceKernelFile) ]

  # Build full paths for asm kernel files
  asmKernelDir = os.path.join(prefixDir, "assembly")

  for asmKernelFile in asmKernelFiles:
    asmKernelPaths += [ os.path.join(asmKernelDir, asmKernelFile) ]

  # Build full paths for source and asm library files
  libDir = os.path.join(prefixDir, "library")

  for sourceLibFile in sourceLibFiles:
    sourceLibPaths += [ os.path.join(libDir, sourceLibFile) ]

  for asmLibFile in asmLibFiles:
    if globalParameters["PackageLibrary"]:
      # Asm lib files are enumerated in the form of
      # KernelName_gfxXXXXX.co
      # Strip the gfxXXXX portion and use that as a subdirectory
      asmLibFileNoExt = str(os.path.splitext(asmLibFile)[0])
      asmArch = asmLibFileNoExt[asmLibFileNoExt.find("_gfx"):]

      # asmArch contains _gfxXXXX. Don't use the underscore in new path
      asmLibPaths += [ os.path.join(
        libDir, asmArch[1:], asmLibFile.replace(asmArch, ''))]

    else:
      asmLibPaths += [ os.path.join(libDir, asmLibFile) ]

  return (solutionPaths, sourceKernelPaths, asmKernelPaths, sourceLibPaths, asmLibPaths)


################################################################################
# Write CMake
################################################################################
def writeCMake(outputPath, solutionFiles, kernelFiles, libraryStaticFiles):
  print1("# Writing Custom CMake")

  # Build output file paths, using relative CMake symbol
  cmakeSrcDir = "${CMAKE_SOURCE_DIR}"
  (solutionPaths, sourceKernelPaths, asmKernelPaths, sourceLibPaths, asmLibPaths) = \
    buildObjectFilePaths(cmakeSrcDir, solutionFiles, kernelFiles, [], [], [])

  # Build full paths the static library files
  staticFilePaths = []
  for staticFile in libraryStaticFiles:
    staticFilePaths += [ os.path.join(cmakeSrcDir, staticFile) ]

  # Proceed to generate cmake file
  generatedFile = open(os.path.join(outputPath, "Generated.cmake"), "w")
  generatedFile.write(CMakeHeader)

  # write TensileClient_SOLUTIONS symbol
  if globalParameters["LegacyComponents"]:
    generatedFile.write("set( TensileClient_SOLUTIONS\n")
    for solutionFile in solutionPaths:
      generatedFile.write("  %s\n" % (solutionFile))
    generatedFile.write("  )\n")

  # write TensileClient_KERNELS symbol
  generatedFile.write("set( TensileClient_KERNELS\n")
  for kernelFile in sourceKernelPaths:
    generatedFile.write("  %s\n" % (kernelFile))
  generatedFile.write("  )\n")

  # write TensileClient_SOURCE symbol
  generatedFile.write("set( TensileClient_SOURCE\n")
  for staticFile in staticFilePaths:
    generatedFile.write("  %s\n" % (staticFile))
  generatedFile.write("  )\n\n")

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
  argParser.add_argument("--cxx-compiler",           dest="CxxCompiler",       choices=["hcc", "hipcc"],       action="store", default="hipcc")
  argParser.add_argument("--code-object-version",    dest="CodeObjectVersion", choices=["V2", "V3"], action="store", default="V3")
  argParser.add_argument("--architecture",           dest="Architecture",      choices=["all", "gfx000", "gfx803", "gfx900", "gfx906", "gfx908"], action="store", default="all")
  argParser.add_argument("--merge-files",            dest="MergeFiles",        action="store_true")
  argParser.add_argument("--no-merge-files",         dest="MergeFiles",        action="store_false")
  argParser.add_argument("--short-file-names",       dest="ShortNames",        action="store_true")
  argParser.add_argument("--no-short-file-names",    dest="ShortNames",        action="store_false")
  argParser.add_argument("--library-print-debug",    dest="LibraryPrintDebug", action="store_true")
  argParser.add_argument("--no-library-print-debug", dest="LibraryPrintDebug", action="store_false")
  argParser.add_argument("--no-enumerate",           action="store_true", help="Do not run rocm_agent_enumerator.")
  argParser.add_argument("--package-library",        dest="PackageLibrary",    action="store_true", default=False)
  argParser.add_argument("--no-legacy-components",   dest="LegacyComponents",  action="store_false", default=True)
  argParser.add_argument("--embed-library",          dest="EmbedLibrary",
                         help="Embed (new) library files into static variables.  Specify the name of the library.")
  argParser.add_argument("--new-client-only",        action="store_true")

  argParser.add_argument("--embed-library-key",      dest="EmbedLibraryKey", default=None,
                         help="Access key for embedding library files.")
  argParser.add_argument("--version", help="Version string to embed into library file.")
  argParser.add_argument("--generate-manifest-and-exit",   dest="GenerateManifestAndExit", action="store_true",
                          default=False, help="Output manifest file with list of expected library objects and exit.")
  argParser.add_argument("--library-format", dest="LibraryFormat", choices=["yaml", "msgpack"], \
      action="store", default="msgpack", help="select which library format to use")
  args = argParser.parse_args()

  logicPath = args.LogicPath
  outputPath = args.OutputPath
  CxxCompiler = args.CxxCompiler
  libraryFormat = args.LibraryFormat
  print2("OutputPath: %s" % outputPath)
  ensurePath(outputPath)
  arguments = {}
  arguments["RuntimeLanguage"] = args.RuntimeLanguage
  arguments["CodeObjectVersion"] = args.CodeObjectVersion
  arguments["Architecture"] = args.Architecture
  arguments["CxxCompiler"] = args.CxxCompiler
  arguments["MergeFiles"] = args.MergeFiles
  arguments["ShortNames"] = args.ShortNames
  arguments["LibraryPrintDebug"] = args.LibraryPrintDebug
  arguments["CodeFromFiles"] = False
  arguments["EmbedLibrary"] = args.EmbedLibrary
  arguments["LibraryFormat"] = args.LibraryFormat
  if args.no_enumerate:
    arguments["ROCmAgentEnumeratorPath"] = False
  arguments["PackageLibrary"] = args.PackageLibrary
  arguments["LegacyComponents"] = args.LegacyComponents

  if args.new_client_only:
    arguments["NewClient"] = 2

  # Output manifest only applies to the new client
  arguments["GenerateManifestAndExit"] = args.new_client_only and args.GenerateManifestAndExit

  assignGlobalParameters(arguments)

  print1("# CodeObjectVersion from TensileCreateLibrary: %s" % arguments["CodeObjectVersion"])
  print1("# CxxCompiler       from TensileCreateLibrary: %s" % CxxCompiler)
  print1("# Architecture      from TensileCreateLibrary: %s" % arguments["Architecture"])
  print1("# LibraryFormat     from TensileCreateLibrary: %s" % libraryFormat)

  if not os.path.exists(logicPath):
    printExit("LogicPath %s doesn't exist" % logicPath)

  # Translate GPU targets to filter filenames in Tensile_LOGIC directory
  mapArchitecture = {'all':'_','gfx000':'none', 'gfx803':'r9nano',
        'gfx900':'vega10', 'gfx906':'vega20', 'gfx908':'arcturus'}

  for key in mapArchitecture:
    if arguments["Architecture"] == key:
      arguments["Architecture"] = mapArchitecture[key]

  logicFiles = [os.path.join(logicPath, f) for f in os.listdir(logicPath) \
      if (os.path.isfile(os.path.join(logicPath, f)) \
      and os.path.splitext(f)[1]==".yaml") \
      and arguments["Architecture"] in os.path.splitext(f)[0] \
      or "hip" in os.path.splitext(f)[0]]

  print1("# LibraryLogicFiles:" % logicFiles)
  for logicFile in logicFiles:
    print1("#   %s" % logicFile)

  ##############################################################################
  # Parse config files
  ##############################################################################
  solutions = []
  logicData = {} # keys are problemTypes, values are schedules

  libraries = Common.ParallelMap(LibraryIO.readLibraryLogicForSchedule, logicFiles, "Reading logic files")

  masterLibraries = {}
  fullMasterLibrary = None
  for logic in Utils.tqdm(libraries, "Processing logic data"):
    (scheduleName, deviceNames, problemType, solutionsForSchedule, \
       indexOrder, exactLogic, rangeLogic, newLibrary, architectureName) = logic

    if not globalParameters["PackageLibrary"]:
      if fullMasterLibrary is None:
        fullMasterLibrary = deepcopy(newLibrary)
        fullMasterLibrary.version = args.version
      else:
        fullMasterLibrary.merge(deepcopy(newLibrary))

    if globalParameters["PackageLibrary"]:
      if architectureName in masterLibraries:
        masterLibraries[architectureName].merge(deepcopy(newLibrary))
      else:
        masterLibraries[architectureName] = deepcopy(newLibrary)

    if problemType not in logicData:
      logicData[problemType] = []
    logicData[problemType].append((scheduleName, deviceNames, \
        solutionsForSchedule, indexOrder, exactLogic, rangeLogic ))
    for solution in solutionsForSchedule:
      if solution not in solutions:
        solutions.append(solution)

    if globalParameters["PackageLibrary"]:
      if architectureName in masterLibraries:
        masterLibraries[architectureName].merge(deepcopy(newLibrary))
      else:
        masterLibraries[architectureName] = deepcopy(newLibrary)
        masterLibraries[architectureName].version = args.version

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
      if KernelWriter.getKernelNameBetaOnly(kernel) not in \
          [KernelWriter.getKernelNameBetaOnly(k) for k in kernelsBetaOnly]:
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

  if globalParameters["LegacyComponents"]:
    staticFiles = [
      "SolutionMapper.h",
      "TensileTypes.h",
      "tensile_bfloat16.h",
      "KernelHeader.h",
      "SolutionHelper.cpp",
      "SolutionHelper.h",
      "Tools.cpp",
      "Tools.h" ]
  else:
    staticFiles = [
      "TensileTypes.h",
      "tensile_bfloat16.h",
      "KernelHeader.h" ]

  # Build a list of files to be expected
  (solutionFiles,
   sourceKernelFiles,
   asmKernelFiles,
   sourceLibFiles,
   asmLibFiles) = buildObjectFileNames(solutionWriter, kernelWriterSource, \
    kernelWriterAssembly, solutions, kernels, kernelsBetaOnly)

  (solutionPaths,
   sourceKernelPaths,
   asmKernelPaths,
   sourceLibPaths,
   asmLibPaths) = buildObjectFilePaths(outputPath, solutionFiles, sourceKernelFiles, \
    asmKernelFiles, sourceLibFiles, asmLibFiles)

  if globalParameters["GenerateManifestAndExit"] == True:

    # Generate manifest file
    libraryPath = os.path.join(outputPath, "library")
    ensurePath(libraryPath)
    generatedFile = open(os.path.join(libraryPath, "TensileManifest.txt"), "w")

    libraryFilename = "TensileLibrary.yaml" if globalParameters["LibraryFormat"] == "yaml" else "TensileLibrary.dat"

    # Manifest file contains YAML file, output library paths and cpp source for embedding.
    for filePath in [os.path.join(libraryPath, libraryFilename)] + sourceLibPaths + asmLibPaths:
      generatedFile.write("%s\n" %(filePath) )
    generatedFile.close()

    return

  # generate cmake for the source kernels
  writeCMake(outputPath, solutionFiles, sourceKernelFiles, staticFiles)

  # Make sure to copy the library static files.
  for fileName in staticFiles:
    shutil.copy( os.path.join(globalParameters["SourcePath"], fileName), \
      outputPath )

    # write solutions and kernels
  problemTypes = list(logicData.keys())

  codeObjectFiles = writeSolutionsAndKernels(outputPath, CxxCompiler, problemTypes, solutions,
                                             kernels, kernelsBetaOnly,
                                             solutionWriter,
                                             kernelWriterSource, kernelWriterAssembly)

  sanityCheck0 = set(codeObjectFiles) - set(sourceLibPaths + asmLibPaths)
  sanityCheck1 = set(sourceLibPaths + asmLibPaths) - set(codeObjectFiles)

  assert len(sanityCheck0) == 0, "Unexpected code object files: {}".format(sanityCheck0)
  assert len(sanityCheck1) == 0, "Missing expected code object files: {}".format(sanityCheck1)

  if globalParameters["LegacyComponents"]:
    # write logic
    writeLogic(outputPath, logicData, solutionWriter)

  archs = ['gfx'+''.join(map(str,arch)) for arch in globalParameters['SupportedISA'] \
             if globalParameters["AsmCaps"][arch]["SupportedISA"]]
  newLibraryDir = ensurePath(os.path.join(outputPath, 'library'))

  libraryWriter = LibraryIO.configWriter(args.LibraryFormat)
  tensileLibraryFilename = "TensileLibrary.yaml" if args.LibraryFormat == "yaml" \
                           else "TensileLibrary.dat"
  if globalParameters["PackageLibrary"]:
    for archName, newMasterLibrary in masterLibraries.items():
      if (archName in archs):
        archPath = ensurePath(os.path.join(newLibraryDir, archName))
        masterFile = os.path.join(archPath, tensileLibraryFilename)
        newMasterLibrary.applyNaming(kernelMinNaming)
        libraryWriter.write(masterFile, Utils.state(newMasterLibrary))
  else:
    masterFile = os.path.join(newLibraryDir, tensileLibraryFilename)
    fullMasterLibrary.applyNaming(kernelMinNaming)
    libraryWriter.write(masterFile, Utils.state(fullMasterLibrary))

  theMasterLibrary = fullMasterLibrary
  if globalParameters["PackageLibrary"]:
    theMasterLibrary = list(masterLibraries.values())[0]
  if args.EmbedLibrary is not None:
      embedFileName = os.path.join(outputPath, "library/{}.cpp".format(args.EmbedLibrary))
      with EmbeddedData.EmbeddedDataFile(embedFileName) as embedFile:
          embedFile.embed_file(theMasterLibrary.cpp_base_class, masterFile, nullTerminated=True,
                               key=args.EmbedLibraryKey)

          for co in Utils.tqdm(codeObjectFiles):
              embedFile.embed_file("SolutionAdapter", co, nullTerminated=False,
                                   key=args.EmbedLibraryKey)

  print1("# Tensile Library Writer DONE")
  print1(HR)
  print1("")

