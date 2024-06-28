################################################################################
#
# Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

# This script only gets called by CMake

if __name__ == "__main__":
    print(
        "This file can no longer be run as a script.  Run 'Tensile/bin/TensileCreateLibrary' instead."
    )
    exit(1)

from . import Common
from . import ClientExecutable
from . import EmbeddedData
from . import LibraryIO
from . import Utils
from .Common import (
    getArchitectureName,
    globalParameters,
    HR,
    tPrint,
    printExit,
    ensurePath,
    CHeader,
    CMakeHeader,
    assignGlobalParameters,
    gfxName,
    architectureMap,
    printWarning,
    supportedCompiler,
    which,
)
from .KernelWriterAssembly import KernelWriterAssembly
from .KernelWriterSource import KernelWriterSource
from .SolutionLibrary import MasterSolutionLibrary
from .SolutionStructs import Solution
from .Utilities.String import splitDelimitedString
from .Utilities.Profile import profile
from .Utilities.toFile import toFile

import argparse
import collections
import itertools
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings

from copy import deepcopy
from typing import Set, List, Tuple, Dict
from pathlib import Path

TENSILE_MANIFEST_FILENAME = "TensileManifest.txt"
TENSILE_LIBRARY_DIR = "library"


################################################################################
def processKernelSource(kernel, kernelWriterSource, kernelWriterAssembly):
    """
    Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    try:
        kernelWriter = (
            kernelWriterSource if kernel["KernelLanguage"] == "Source" else kernelWriterAssembly
        )
        # get kernel name
        kernelName = kernelWriter.getKernelFileBase(kernel)
        (err, src) = kernelWriter.getSourceFileString(kernel)
        header = kernelWriter.getHeaderFileString(kernel)
        # will be put in Kernels.h/cpp if None
        filename = kernel._state.get("codeObjectFile", None)

    except RuntimeError:
        return (1, "", "", kernelName, None)

    return (err, src, header, kernelName, filename)


def getAssemblyCodeObjectFiles(kernels, kernelWriterAssembly, outputPath):
    destDir = ensurePath(os.path.join(outputPath, "library"))
    asmDir = kernelWriterAssembly.getAssemblyDirectory()
    assemblyKernels = list([k for k in kernels if k["KernelLanguage"] == "Assembly"])
    if len(assemblyKernels) == 0:
        return []

    archs = collections.defaultdict(list)
    for k in assemblyKernels:
        archs[tuple(k["ISA"])].append(k)
    coFiles = []
    for arch, archKernels in archs.items():
        archName = gfxName(arch)
        objectFiles = list(
            [
                kernelWriterAssembly.getKernelFileBase(k) + ".o"
                for k in archKernels
                if "codeObjectFile" not in k
            ]
        )

        numObjectFiles = len([1 for k in archKernels if k["KernelLanguage"] == "Assembly"])

        if numObjectFiles == 0:
            continue
        if (
            globalParameters["MergeFiles"]
            or globalParameters["NumMergedFiles"] > 1
            or globalParameters["LazyLibraryLoading"]
        ):

            # Group kernels from placeholder libraries
            coFileMap = collections.defaultdict(list)

            if len(objectFiles):
                coFileMap[os.path.join(destDir, "TensileLibrary_" + archName + ".co")] = objectFiles

            for kernel in archKernels:
                coName = kernel.get("codeObjectFile", None)
                if coName:
                    coFileMap[os.path.join(destDir, coName + ".co")] += [
                        kernelWriterAssembly.getKernelFileBase(kernel) + ".o"
                    ]

            for coFile, objectFiles in coFileMap.items():
                args = []
                if os.name == "nt":
                    # On Windows, the objectFiles list command line (including spaces)
                    # exceeds the limit of 8191 characters, so using response file

                    responseArgs = objectFiles
                    responseFile = os.path.join(asmDir, "clangArgs.txt")
                    with open(responseFile, "wt") as file:
                        file.write(" ".join(responseArgs))
                        file.flush()

                    args = kernelWriterAssembly.getLinkCodeObjectArgs(["@clangArgs.txt"], coFile)
                else:
                    args = kernelWriterAssembly.getLinkCodeObjectArgs(objectFiles, coFile)

                # change to use  check_output to force windows cmd block util command finish
                try:
                    out = subprocess.check_output(args, stderr=subprocess.STDOUT, cwd=asmDir)
                    tPrint(3, out)
                except subprocess.CalledProcessError as err:
                    print(err.output)
                    raise

                coFiles.append(coFile)
        else:
            # no mergefiles

            assemblyKernelNames = [kernelWriterAssembly.getKernelFileBase(k) for k in archKernels]
            origCOFiles = [os.path.join(asmDir, k + ".co") for k in assemblyKernelNames]
            newCOFiles = [
                os.path.join(destDir, k + "_" + archName + ".co") for k in assemblyKernelNames
            ]

            for src, dst in (
                zip(origCOFiles, newCOFiles)
                if globalParameters["PrintLevel"] == 0
                else Utils.tqdm(zip(origCOFiles, newCOFiles), "Copying code objects")
            ):
                shutil.copyfile(src, dst)
            coFiles += newCOFiles

    return coFiles


def splitArchs():
    # Helper for architecture
    def isSupported(arch):
        return (
            globalParameters["AsmCaps"][arch]["SupportedISA"]
            and globalParameters["AsmCaps"][arch]["SupportedSource"]
        )

    if ";" in globalParameters["Architecture"]:
        wantedArchs = globalParameters["Architecture"].split(";")
    else:
        wantedArchs = globalParameters["Architecture"].split("_")
    archs = []
    cmdlineArchs = []
    if "all" in wantedArchs:
        for arch in globalParameters["SupportedISA"]:
            if isSupported(arch):
                if arch == (9, 0, 6) or arch == (9, 0, 8) or arch == (9, 0, 10):
                    if arch == (9, 0, 10):
                        archs += [gfxName(arch) + "-xnack+"]
                        cmdlineArchs += [gfxName(arch) + ":xnack+"]
                    archs += [gfxName(arch) + "-xnack-"]
                    cmdlineArchs += [gfxName(arch) + ":xnack-"]
                else:
                    archs += [gfxName(arch)]
                    cmdlineArchs += [gfxName(arch)]
    else:
        for arch in wantedArchs:
            archs += [re.sub(":", "-", arch)]
            cmdlineArchs += [arch]
    return archs, cmdlineArchs


def buildSourceCodeObjectFile(CxxCompiler, outputPath, kernelFile):
    buildPath = ensurePath(os.path.join(globalParameters["WorkingPath"], "code_object_tmp"))
    destDir = ensurePath(os.path.join(outputPath, "library"))
    (_, filename) = os.path.split(kernelFile)
    (base, _) = os.path.splitext(filename)

    if "CmakeCxxCompiler" in globalParameters and globalParameters["CmakeCxxCompiler"] is not None:
        os.environ["CMAKE_CXX_COMPILER"] = globalParameters["CmakeCxxCompiler"]

    objectFilename = base + ".o"
    soFilename = base + ".so"

    coFilenames = []

    if supportedCompiler(CxxCompiler):
        archs, cmdlineArchs = splitArchs()

        archFlags = ["--offload-arch=" + arch for arch in cmdlineArchs]

        hipFlags = [
            "-D__HIP_HCC_COMPAT_MODE__=1"
        ]  # needs to be fixed when Maneesh's change is made available
        hipFlags += (
            ["--genco"] if CxxCompiler == "hipcc" else ["--cuda-device-only", "-x", "hip", "-O3"]
        )
        # if CxxCompiler == "amdclang++":
        # hipFlags += ["-mllvm", "-amdgpu-early-inline-all=true", "-mllvm", "-amdgpu-function-calls=false"]
        hipFlags += ["-I", outputPath]

        # Add build-id for builds with rocm 5.3+
        compilerVer = globalParameters["HipClangVersion"].split(".")[:2]
        compilerVer = [int(c) for c in compilerVer]
        if len(compilerVer) >= 2 and (
            compilerVer[0] > 5 or (compilerVer[0] == 5 and compilerVer[1] > 2)
        ):
            hipFlags += ["-Xoffload-linker", "--build-id"]

        launcher = shlex.split(os.environ.get("Tensile_CXX_COMPILER_LAUNCHER", ""))

        if os.name == "nt":
            hipFlags += [
                "-std=c++14",
                "-fms-extensions",
                "-fms-compatibility",
                "-fPIC",
                "-Wno-deprecated-declarations",
            ]
            compileArgs = (
                launcher
                + [which(CxxCompiler)]
                + hipFlags
                + archFlags
                + [kernelFile, "-c", "-o", os.path.join(buildPath, objectFilename)]
            )
        else:
            compileArgs = (
                launcher
                + [which(CxxCompiler)]
                + hipFlags
                + archFlags
                + [kernelFile, "-c", "-o", os.path.join(buildPath, objectFilename)]
            )

        tPrint(2, "hipcc:" + " ".join(compileArgs))
        # change to use  check_output to force windows cmd block util command finish
        try:
            out = subprocess.check_output(compileArgs, stderr=subprocess.STDOUT)
            tPrint(3, out)
        except subprocess.CalledProcessError as err:
            print(err.output)
            raise

        # get hipcc version due to compatiblity reasons
        # If we aren't using hipcc what happens?
        hipccver = globalParameters["HipClangVersion"].split(".")
        hipccMaj = int(hipccver[0])
        hipccMin = int(hipccver[1])

        # for hipclang 5.2 and above, clang offload bundler changes the way input/output files are specified
        inflag = "-inputs"
        outflag = "-outputs"
        if (hipccMaj == 5 and hipccMin >= 2) or hipccMaj >= 6:
            inflag = "-input"
            outflag = "-output"

        infile = os.path.join(buildPath, objectFilename)
        bundler = globalParameters["ClangOffloadBundlerPath"]
        if bundler is None:
            raise ValueError(
                "No bundler available; set TENSILE_ROCM_OFFLOAD_BUNDLER_PATH to point to clang-offload-bundler."
            )
        try:
            bundlerArgs = [bundler, "-type=o", "%s=%s" % (inflag, infile), "-list"]
            listing = (
                subprocess.check_output(bundlerArgs, stderr=subprocess.STDOUT).decode().split("\n")
            )
            for target in listing:
                matched = re.search("gfx.*$", target)
                if matched:
                    arch = re.sub(":", "-", matched.group())
                    if "TensileLibrary" in base and "fallback" in base:
                        outfile = os.path.join(buildPath, "{0}_{1}.hsaco".format(base, arch))
                    elif "TensileLibrary" in base:
                        variant = [t for t in ["", "xnack-", "xnack+"] if t in target][-1]
                        baseVariant = base + "-" + variant if variant else base
                        if arch in baseVariant:
                            outfile = os.path.join(buildPath, baseVariant + ".hsaco")
                        else:
                            outfile = None
                    else:
                        outfile = os.path.join(
                            buildPath, "{0}-000-{1}.hsaco".format(soFilename, arch)
                        )

                    # Compilation
                    if outfile:
                        coFilenames.append(os.path.split(outfile)[1])
                        # bundlerArgs = [bundler, "-type=o", "-targets=%s" % target, "-inputs=%s" % infile, "-outputs=%s" % outfile, "-unbundle"]
                        bundlerArgs = [
                            bundler,
                            "-type=o",
                            "-targets=%s" % target,
                            "%s=%s" % (inflag, infile),
                            "%s=%s" % (outflag, outfile),
                            "-unbundle",
                        ]
                        tPrint(2, " ".join(bundlerArgs))
                        # change to use  check_output to force windows cmd block util command finish
                        out = subprocess.check_output(bundlerArgs, stderr=subprocess.STDOUT)
                        tPrint(3, out)

        except subprocess.CalledProcessError as err:
            tPrint(1, err.output)
            for i in range(len(archs)):
                outfile = os.path.join(buildPath, "{0}-000-{1}.hsaco".format(soFilename, archs[i]))
                coFilenames.append(os.path.split(outfile)[1])
                # bundlerArgs = [bundler, "-type=o", "-targets=hip-amdgcn-amd-amdhsa--%s" % cmdlineArchs[i], "-inputs=%s" % infile, "-outputs=%s" % outfile, "-unbundle"]
                bundlerArgs = [
                    bundler,
                    "-type=o",
                    "-targets=hip-amdgcn-amd-amdhsa--%s" % cmdlineArchs[i],
                    "%s=%s" % (inflag, infile),
                    "%s=%s" % (outflag, outfile),
                    "-unbundle",
                ]
                tPrint(2, " ".join(bundlerArgs))
                # change to use  check_output to force windows cmd block util command finish
                try:
                    out = subprocess.check_output(bundlerArgs, stderr=subprocess.STDOUT)
                    tPrint(3, out)
                except subprocess.CalledProcessError as err:
                    tPrint(1, err.output)
                    raise
    else:
        raise RuntimeError("Unknown compiler {}".format(CxxCompiler))

    coFilenames = [name for name in coFilenames]
    extractedCOs = [os.path.join(buildPath, name) for name in coFilenames]
    destCOsList = [os.path.join(destDir, name) for name in coFilenames]
    for src, dst in zip(extractedCOs, destCOsList):
        shutil.copyfile(src, dst)

    return destCOsList


def buildSourceCodeObjectFiles(CxxCompiler, kernelFiles, outputPath):
    args = zip(itertools.repeat(CxxCompiler), itertools.repeat(outputPath), kernelFiles)
    coFiles = Common.ParallelMap(buildSourceCodeObjectFile, args, "Compiling source kernels")

    return itertools.chain.from_iterable(coFiles)


################################################################################
def prepAsm(kernelWriterAssembly):
    """
    Create and prepare the assembly directory  - called ONCE per output dir:
    """
    asmPath = ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly"))
    isa = globalParameters["CurrentISA"]
    assemblerFileName = os.path.join(asmPath, "asm-new.%s" % ("bat" if os.name == "nt" else "sh"))
    assemblerFile = open(assemblerFileName, "w")
    if os.name == "nt":
        assemblerFile.write("@echo off\n")
        assemblerFile.write("set f=%1\n\n")
        assemblerFile.write("set arg2=--wave64\n")
        assemblerFile.write("if [%2] NEQ [] set arg2=%2\n\n")
        assemblerFile.write("set /A wave=64\n")
        assemblerFile.write("if %arg2% EQU --wave32 set /A wave=32\n\n")

        assemblerFile.write("set h={gfxName}\n".format(gfxName=Common.gfxName(isa)))

        cArgs32 = " ".join(
            kernelWriterAssembly.getCompileArgs("%f%.s", "%f%.o", isa=isa, wavefrontSize=32)
        )
        cArgs64 = " ".join(
            kernelWriterAssembly.getCompileArgs("%f%.s", "%f%.o", isa=isa, wavefrontSize=64)
        )
        lArgs = " ".join(kernelWriterAssembly.getLinkCodeObjectArgs(["%f%.o"], "%f%.co"))

        assemblerFile.write(f"if %wave% == 32 ({cArgs32}) else ({cArgs64})\n")
        assemblerFile.write(f"{lArgs}\n")
        assemblerFile.write("copy %f%.co ..\..\..\library\%f%_%h%.co\n")
    else:
        assemblerFile.write(
            "#!/bin/sh {log}\n".format(log="-x" if globalParameters["PrintLevel"] >= 3 else "")
        )
        assemblerFile.write("# usage: asm-new.sh kernelName(no extension) [--wave32]\n")

        assemblerFile.write("f=$1\n")
        assemblerFile.write("shift\n")
        assemblerFile.write('if [ ! -z "$1" ] && [ "$1" = "--wave32" ]; then\n')
        assemblerFile.write("    wave=32\n")
        assemblerFile.write("    shift\n")
        assemblerFile.write("else\n")
        assemblerFile.write("    wave=64\n")
        assemblerFile.write("fi\n")

        assemblerFile.write("h={gfxName}\n".format(gfxName=Common.gfxName(isa)))

        cArgs32 = kernelWriterAssembly.getCompileArgs("$f.s", "$f.o", isa=isa, wavefrontSize=32)
        cArgs64 = kernelWriterAssembly.getCompileArgs("$f.s", "$f.o", isa=isa, wavefrontSize=64)
        lArgs = kernelWriterAssembly.getLinkCodeObjectArgs(["$f.o"], "$f.co")

        assemblerFile.write("if [ $wave -eq 32 ]; then\n")
        assemblerFile.write(" ".join(cArgs32) + "\n")
        assemblerFile.write("else\n")
        assemblerFile.write(" ".join(cArgs64) + "\n")
        assemblerFile.write("fi\n")

        assemblerFile.write(" ".join(lArgs) + "\n")

        assemblerFile.write("cp $f.co ../../../library/${f}_$h.co\n")
        assemblerFile.write("mkdir -p ../../../asm_backup && ")
        assemblerFile.write("cp $f.s ../../../asm_backup/$f.s\n")

    assemblerFile.close()
    os.chmod(assemblerFileName, 0o777)


################################################################################
def buildKernelSourceAndHeaderFiles(results, outputPath, kernelsWithBuildErrs):
    """
    Logs errors and writes appropriate info to kernelSourceFile and kernelHeaderFile.

    Arguments:
      results:              list of (err, src, header, kernelName, filename)
      outputPath:           path to source directory
      kernelsWithBuildErrs: Dictionary to be updated with kernels that have errors
      kernelSourceFile:     File to write source data to
      kernelHeaderFile:     File to write header data to

    Returns:
      sourceFilenames:      Array containing source kernel filenames
    """

    # Find kernels to write
    kernelsToWrite = []
    filesToWrite = collections.defaultdict(list)
    validKernelCount = 0
    for err, src, header, kernelName, filename in results:

        # Keep track of kernels with errors
        if err:
            kernelsWithBuildErrs[kernelName] = err

        # Don't create a file for empty kernels
        if len(src.strip()) == 0:
            continue

        kernelsToWrite.append((err, src, header, kernelName))

        # Create list of files
        if filename:
            filesToWrite[os.path.join(os.path.normcase(outputPath), filename)].append(
                (err, src, header, kernelName)
            )
        elif globalParameters["MergeFiles"]:
            kernelSuffix = ""
            if globalParameters["NumMergedFiles"] > 1:
                kernelSuffix = validKernelCount % globalParameters["NumMergedFiles"]

            filesToWrite[
                os.path.join(os.path.normcase(outputPath), "Kernels" + kernelSuffix)
            ].append((err, src, header, kernelName))
        else:
            filesToWrite[os.path.join(os.path.normcase(outputPath), kernelName)].append(
                (err, src, header, kernelName)
            )

        validKernelCount += 1

    # Ensure there's at least one kernel file for helper kernels
    if globalParameters["LazyLibraryLoading"] or (
        globalParameters["MergeFiles"] and not kernelsToWrite
    ):
        kernelSuffix = ""
        if globalParameters["NumMergedFiles"] > 1:
            kernelSuffix = "0"

        filesToWrite[os.path.join(os.path.normcase(outputPath), "Kernels" + kernelSuffix)] = []

    # Write kernel data to files
    # Parse list of files and write kernels
    for filename, kernelList in filesToWrite.items():
        with open(filename + ".h", "w", encoding="utf-8") as kernelHeaderFile, open(
            filename + ".cpp", "w", encoding="utf-8"
        ) as kernelSourceFile:

            kernelSourceFile.write(CHeader)
            kernelHeaderFile.write(CHeader)
            kernelSourceFile.write('#include "{}.h"\n'.format(filename))
            kernelHeaderFile.write("#pragma once\n")
            if globalParameters["RuntimeLanguage"] == "HIP":
                kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
                kernelHeaderFile.write("#include <hip/hip_ext.h>\n\n")
            kernelHeaderFile.write('#include "KernelHeader.h"\n\n')

            for err, src, header, kernelName in kernelList:
                kernelSourceFile.write(src)
                kernelHeaderFile.write(header)

    sourceFilenames = [filePrefix + ".cpp" for filePrefix in filesToWrite]

    return sourceFilenames


################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
def writeKernels(
    outputPath,
    CxxCompiler,
    problemTypes,
    solutions,
    kernels,
    kernelHelperObjs,
    kernelWriterSource,
    kernelWriterAssembly,
    errorTolerant=False,
):
    start = time.time()

    codeObjectFiles = []

    # Push working path into build_tmp folder because there may be more than
    # one process running this script. This is to avoid build directory clashing.
    # NOTE: file paths must not contain the lower case word 'kernel' or the
    # /opt/rocm/bin/extractkernel will fail.
    # See buildSourceCodeObjectFile:167 for the call to this binary.
    Common.pushWorkingPath("build_tmp")
    Common.pushWorkingPath(os.path.basename(outputPath).upper())

    tPrint(1, "# Writing Kernels...")
    kernelFiles = []
    kernelSourceFile = None
    kernelHeaderFile = None

    if (
        not globalParameters["MergeFiles"]
        or globalParameters["NumMergedFiles"] > 1
        or globalParameters["LazyLibraryLoading"]
    ):
        ensurePath(os.path.join(outputPath, "Kernels"))

    ##############################################################################
    # Write Kernels
    ##############################################################################
    kernelsWithBuildErrs = {}

    prepAsm(kernelWriterAssembly)

    # Kernels may be intended for different co files, but generate the same .o file
    # Mark duplicate kernels to avoid race condition
    # @TODO improve organization so this problem doesn't appear
    objFilenames = set()
    for kernel in kernels:
        if kernel["KernelLanguage"] == "Assembly":
            base = kernelWriterAssembly.getKernelFileBase(kernel)
            if base in objFilenames:
                kernel.duplicate = True
            else:
                objFilenames.add(base)
                kernel.duplicate = False

    kIter = zip(
        kernels,
        itertools.repeat(kernelWriterSource),
        itertools.repeat(kernelWriterAssembly),
    )
    results = Common.ParallelMap(processKernelSource, kIter, "Generating kernels")

    removeKernels = []
    removeSolutions = []
    removeResults = []
    for kernIdx, res in (
        enumerate(results)
        if globalParameters["PrintLevel"] == 0
        else Utils.tqdm(enumerate(results))
    ):
        (err, src, header, kernelName, filename) = res
        if err == -2:
            if not errorTolerant:
                print(
                    "\nKernel generation failed for kernel: {}".format(
                        kernels[kernIdx]["SolutionIndex"]
                    )
                )
                print(kernels[kernIdx]["SolutionNameMin"])
            removeKernels.append(kernels[kernIdx])
            removeSolutions.append(solutions[kernIdx])
            removeResults.append(results[kernIdx])
    if len(removeKernels) > 0 and not errorTolerant:
        printExit("** kernel generation failure **")
    for kern in removeKernels:
        kernels.remove(kern)
    for solut in removeSolutions:
        solutions.remove(solut)
    for rel in removeResults:
        results.remove(rel)

    kernelFiles += buildKernelSourceAndHeaderFiles(results, outputPath, kernelsWithBuildErrs)

    kernelsToBuild = list(kernels)
    if errorTolerant:

        def success(kernel):
            writer = (
                kernelWriterAssembly
                if kernel["KernelLanguage"] == "Assembly"
                else kernelWriterSource
            )
            kernelName = writer.getKernelName(kernel)
            return kernelName not in kernelsWithBuildErrs

        kernelsToBuild = list(filter(success, kernelsToBuild))
    elif len(kernelsWithBuildErrs) > 0:
        print(
            "\nKernel compilation failed in one or more subprocesses. May want to set CpuThreads=0 and re-run to make debug easier"
        )
        printExit("** kernel compilation failure **")

    # Put all kernel helper objects into the first merged kernel file
    if globalParameters["NumMergedFiles"] > 1 and len(kernelFiles) > 0:
        kernelFilename = kernelFiles[0].replace(".cpp", "")
        kernelSourceFile = open(kernelFilename + ".cpp", "a", encoding="utf-8")
        kernelHeaderFile = open(kernelFilename + ".h", "a", encoding="utf-8")
    elif globalParameters["MergeFiles"] or globalParameters["LazyLibraryLoading"]:
        kernelSourceFilename = os.path.join(os.path.normcase(outputPath), "Kernels.cpp")
        kernelHeaderFilename = os.path.join(os.path.normcase(outputPath), "Kernels.h")
        kernelSourceFile = open(kernelSourceFilename, "a", encoding="utf-8")
        kernelHeaderFile = open(kernelHeaderFilename, "a", encoding="utf-8")

    # handle helper kernel function
    for ko in kernelHelperObjs:
        kernelName = ko.getKernelName()

        # write kernel.cpp
        if not globalParameters["MergeFiles"]:
            kernelSourceFilename = os.path.join(outputPath, "Kernels", kernelName + ".cpp")
            kernelSourceFile = open(kernelSourceFilename, "w")
            kernelSourceFile.write(CHeader)
            kernelFiles.append(kernelSourceFilename)

        (err, src) = ko.getSourceFileString()
        kernelSourceFile.write(src)
        if err:
            print("*** warning: invalid kernel#%u" % kernelName)

        if not globalParameters["MergeFiles"]:
            kernelSourceFile.close()

        # write kernel.h
        if not globalParameters["MergeFiles"]:
            kernelHeaderFile = open(
                os.path.join(os.path.normcase(outputPath), "Kernels", kernelName + ".h"),
                "w",
            )
            kernelHeaderFile.write(CHeader)

        kernelHeaderFile.write(ko.getHeaderFileString())

        if not globalParameters["MergeFiles"]:
            kernelHeaderFile.close()

    # close merged
    if globalParameters["MergeFiles"]:
        if kernelSourceFile:
            kernelSourceFile.close()
        if kernelHeaderFile:
            kernelHeaderFile.close()

    if not globalParameters["GenerateSourcesAndExit"]:
        codeObjectFiles += buildSourceCodeObjectFiles(CxxCompiler, kernelFiles, outputPath)
        codeObjectFiles += getAssemblyCodeObjectFiles(
            kernelsToBuild, kernelWriterAssembly, outputPath
        )

    stop = time.time()
    tPrint(1, "# Kernel Building elapsed time = %.1f secs" % (stop - start))

    Common.popWorkingPath()  # outputPath.upper()

    if globalParameters["CleanupBuildFiles"]:
        shutil.rmtree(globalParameters["WorkingPath"])

    Common.popWorkingPath()  # build_tmp

    return codeObjectFiles


##############################################################################
# Min Naming / Solution and Kernel Writers
##############################################################################
def getKernelWriters(solutions, kernels):

    # if any kernels are assembly, append every ISA supported
    kernelSerialNaming = Solution.getSerialNaming(kernels)

    solutionMinNaming = Solution.getMinNaming(solutions)
    kernelMinNaming = Solution.getMinNaming(kernels)
    kernelWriterSource = KernelWriterSource(kernelMinNaming, kernelSerialNaming)
    kernelWriterAssembly = KernelWriterAssembly(kernelMinNaming, kernelSerialNaming)

    return (
        kernelWriterSource,
        kernelWriterAssembly,
        kernelMinNaming,
        solutionMinNaming,
    )


################################################################################
# copy static cpp files and headers
################################################################################
def copyStaticFiles(outputPath=None):
    if outputPath is None:
        outputPath = globalParameters["WorkingPath"]
    libraryStaticFiles = [
        "TensileTypes.h",
        "tensile_bfloat16.h",
        "tensile_float8_bfloat8.h",
        "hip_f8_impl.h",
        "KernelHeader.h",
    ]

    for fileName in libraryStaticFiles:
        # copy file
        shutil.copy(os.path.join(globalParameters["SourcePath"], fileName), outputPath)

    return libraryStaticFiles


def buildObjectFileNames(
    kernelWriterSource, kernelWriterAssembly, solutions, kernels, kernelHelperObjs
):

    # Build lists of output object names
    sourceKernelNames = []
    asmKernelNames = []
    kernelHelperObjNames = []

    solutionFiles = []
    sourceKernelFiles = []
    asmKernelFiles = []
    sourceLibFiles = []
    asmLibFiles = []

    sourceKernels = list([k for k in kernels if k["KernelLanguage"] == "Source"])
    asmKernels = list([k for k in kernels if k["KernelLanguage"] == "Assembly"])

    # Build a list of kernel object names.
    for kernel in sourceKernels:
        sourceKernelNames += [kernelWriterSource.getKernelFileBase(kernel)]

    for kernel in asmKernels:
        asmKernelNames += [kernelWriterAssembly.getKernelFileBase(kernel)]

    kernelHelperObjNames = [ko.getKernelName() for ko in kernelHelperObjs]

    cxxCompiler = globalParameters["CxxCompiler"]

    # Source based kernels are built for all supported architectures
    if supportedCompiler(cxxCompiler):
        sourceArchs, _ = splitArchs()
    else:
        raise RuntimeError("Unknown compiler %s" % cxxCompiler)

    # Asm based kernels target the configured ISA
    asmArchs = collections.defaultdict(list)
    for kernelName, kernel in zip(asmKernelNames, asmKernels):
        asmArchs[kernelName].append(gfxName(kernel["ISA"]))

    # Build a list of source files
    if not globalParameters["MergeFiles"]:
        for kernelName in sourceKernelNames + asmKernelNames + kernelHelperObjNames:
            sourceKernelFiles += ["%s.h" % (kernelName), "%s.cpp" % (kernelName)]
    elif globalParameters["NumMergedFiles"] > 1:
        for kernelIndex in range(0, globalParameters["NumMergedFiles"]):
            sourceKernelFiles += [
                "Kernels%s.h" % str(kernelIndex),
                "Kernels%s.cpp" % str(kernelIndex),
            ]
        for kernelName in kernelHelperObjNames:
            sourceKernelFiles += ["%s.h" % (kernelName), "%s.cpp" % (kernelName)]
    else:
        sourceKernelFiles += ["Kernels.h", "Kernels.cpp"]

    # Build a list of assembly files
    for asmKernelName in asmKernelNames:
        asmKernelFiles += [
            "%s.s" % (asmKernelName),
            "%s.o" % (asmKernelName),
            "%s.co" % (asmKernelName),
        ]

    # Build a list of lib names from source
    if not globalParameters["MergeFiles"]:

        allSources = sourceKernelNames + kernelHelperObjNames

        for kernelName in allSources:
            if supportedCompiler(cxxCompiler):
                sourceLibFiles += [
                    "%s.so-000-%s.hsaco" % (kernelName, arch) for arch in sourceArchs
                ]
            else:
                raise RuntimeError("Unknown compiler {}".format(cxxCompiler))
    elif globalParameters["NumMergedFiles"] > 1:
        if supportedCompiler(cxxCompiler):
            for kernelIndex in range(0, globalParameters["NumMergedFiles"]):
                sourceLibFiles += [
                    "Kernels%d.so-000-%s.hsaco" % (kernelIndex, arch) for arch in sourceArchs
                ]
        else:
            raise RuntimeError("Unknown compiler {}".format(cxxCompiler))
    elif globalParameters["LazyLibraryLoading"]:
        fallbackLibs = list(
            set(
                [
                    kernel._state["codeObjectFile"]
                    for kernel in kernels
                    if "fallback" in kernel._state.get("codeObjectFile", "")
                ]
            )
        )
        sourceLibFiles += [
            "{0}_{1}.hsaco".format(name, arch)
            for name, arch in itertools.product(fallbackLibs, sourceArchs)
        ]
        if supportedCompiler(cxxCompiler):
            sourceLibFiles += ["Kernels.so-000-%s.hsaco" % (arch) for arch in sourceArchs]
    else:  # Merge
        if supportedCompiler(cxxCompiler):
            sourceLibFiles += ["Kernels.so-000-%s.hsaco" % (arch) for arch in sourceArchs]
        else:
            raise RuntimeError("Unknown compiler {}".format(cxxCompiler))

    # Returns names for all xnack versions
    def addxnack(name, ext):
        arch = re.search(r"gfx.*$", name).group()
        if arch in sourceArchs:
            return [name + ext]
        else:
            return [name + xnack[len(arch) :] + ext for xnack in sourceArchs if arch in xnack]

    # Build a list of asm lib names
    if globalParameters["LazyLibraryLoading"]:

        # If assembly kernel with codeObjectFile specified
        cond = (
            lambda k: "codeObjectFile" in k._state
            and "fallback" not in k._state["codeObjectFile"]
            and k._state["KernelLanguage"] == "Assembly"
        )

        asmLibFiles += list(
            set([kernel._state["codeObjectFile"] + ".co" for kernel in kernels if cond(kernel)])
        )

        # If architecture specific source kernel with codeObjectFile specified
        cond = (
            lambda k: "codeObjectFile" in k._state
            and "fallback" not in k._state["codeObjectFile"]
            and k._state["KernelLanguage"] == "Source"
        )

        sourceLibFiles += list(
            set(
                itertools.chain.from_iterable(
                    [
                        addxnack(kernel._state["codeObjectFile"], ".hsaco")
                        for kernel in kernels
                        if cond(kernel)
                    ]
                )
            )
        )

    elif globalParameters["MergeFiles"]:
        # Find all unique arch values for current asm kernels
        uniqueArchs = set(itertools.chain(*asmArchs.values()))
        asmLibFiles += ["TensileLibrary_%s.co" % (arch) for arch in uniqueArchs]

    else:
        for asmKernelName, archs in asmArchs.items():
            asmLibFiles += ["%s_%s.co" % (asmKernelName, str(arch)) for arch in archs]

    return (
        solutionFiles,
        sourceKernelFiles,
        asmKernelFiles,
        sourceLibFiles,
        asmLibFiles,
    )


def buildObjectFilePaths(
    prefixDir,
    solutionFiles,
    sourceKernelFiles,
    asmKernelFiles,
    sourceLibFiles,
    asmLibFiles,
    masterLibraries,
):
    solutionPaths = []
    sourceKernelPaths = []
    asmKernelPaths = []
    sourceLibPaths = []
    asmLibPaths = []
    libMetadataPaths = []

    # Build full paths for source kernel files
    sourceKernelDir = ""
    if not globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1:
        sourceKernelDir = os.path.join(prefixDir, "Kernels")
    else:
        sourceKernelDir = prefixDir

    for sourceKernelFile in sourceKernelFiles:
        sourceKernelPaths += [os.path.join(sourceKernelDir, sourceKernelFile)]

    # Build full paths for asm kernel files
    asmKernelDir = os.path.join(prefixDir, "assembly")

    for asmKernelFile in asmKernelFiles:
        asmKernelPaths += [os.path.join(asmKernelDir, asmKernelFile)]

    # Build full paths for source and asm library files
    libDir = os.path.join(prefixDir, "library")

    libraryExt = ".yaml" if globalParameters["LibraryFormat"] == "yaml" else ".dat"
    if not globalParameters["SeparateArchitectures"] and not globalParameters["LazyLibraryLoading"]:
        libMetadataPaths = [os.path.join(libDir, "TensileLibrary" + libraryExt)]

    for sourceLibFile in sourceLibFiles:
        sourceLibPaths += [os.path.join(libDir, sourceLibFile)]

    # Use set because of duplicate fallback libraries
    newMetadataPaths = set()
    if "full" not in masterLibraries.keys():
        for arch, lib in masterLibraries.items():
            if globalParameters["LazyLibraryLoading"]:
                newMetadataPaths.add(
                    os.path.join(libDir, "TensileLibrary_lazy_" + arch + libraryExt)
                )
            else:
                newMetadataPaths.add(os.path.join(libDir, "TensileLibrary_" + arch + libraryExt))
            for name, placeholder in lib.lazyLibraries.items():
                newMetadataPaths.add(os.path.join(libDir, name + libraryExt))

    libMetadataPaths += list(newMetadataPaths)

    for asmLibFile in asmLibFiles:
        # Asm lib files are enumerated in the form of
        # KernelName_gfxXXXXX.co
        asmLibPaths += [os.path.join(libDir, asmLibFile)]

    return (
        solutionPaths,
        sourceKernelPaths,
        asmKernelPaths,
        sourceLibPaths,
        asmLibPaths,
        libMetadataPaths,
    )


################################################################################
# Write CMake
################################################################################
def writeCMake(outputPath, solutionFiles, kernelFiles, libraryStaticFiles, masterLibraries):
    tPrint(1, "# Writing Custom CMake")

    # Build output file paths, using relative CMake symbol
    cmakeSrcDir = "${CMAKE_SOURCE_DIR}"
    (
        solutionPaths,
        sourceKernelPaths,
        asmKernelPaths,
        sourceLibPaths,
        asmLibPaths,
        _,
    ) = buildObjectFilePaths(cmakeSrcDir, solutionFiles, kernelFiles, [], [], [], masterLibraries)

    # Build full paths the static library files
    staticFilePaths = []
    for staticFile in libraryStaticFiles:
        staticFilePaths += [os.path.join(cmakeSrcDir, staticFile)]

    # Proceed to generate cmake file
    generatedFile = open(os.path.join(os.path.normcase(outputPath), "Generated.cmake"), "w")
    generatedFile.write(CMakeHeader)

    # write TensileClient_KERNELS symbol
    generatedFile.write("set( TensileClient_KERNELS\n")
    for kernelFile in sourceKernelPaths:
        generatedFile.write("  %s\n" % (kernelFile))
    generatedFile.write("  )\n")

    # write TensileClient_SOURCE symbol
    generatedFile.write("set( TensileClient_SOURCE\n")
    for fileName in libraryStaticFiles:
        generatedFile.write("  ${CMAKE_SOURCE_DIR}/%s\n" % fileName)
    generatedFile.write("  )\n\n")

    generatedFile.close()


################################################################################
# Generate Kernel Objects From Solutions
################################################################################
def generateKernelObjectsFromSolutions(solutions):
    # create solution writer and kernel writer
    kernels = []
    kernelHelperObjs = []
    kernelHelperNames = set()

    for solution in solutions:
        kernels += solution.getKernels()
        solutionHelperKernels = solution.getHelperKernelObjects()
        kernelHelperObjs += solutionHelperKernels
        for ko in solutionHelperKernels:
            kernelHelperNames.add(ko.getKernelName())

    # remove duplicates while preserving order
    kernels = list(dict.fromkeys(kernels))
    kernelHelperObjs = list(dict.fromkeys(kernelHelperObjs))
    return (kernels, kernelHelperObjs, kernelHelperNames)


def addNewLibrary(masterLibraries, newLibrary, architectureName, version):
    """Adds master solution library to a master solution libraries dict.

    For a given architecture, add the new library to a dictionary containing
    libraries for all architectures, compute the starting index for the new
    library, then remap the indexes for all of the solutions associated with
    the library.

    Args:
        masterLibraries: A dictionary containing all master solution libraries for all architectures.
        newLibrary: A master solution library to add to the dictionary.
        architectureName: The name of the architecture (or key) associated with the library.
        version: User provided version for the library.

    Returns:
        A tuple of
    """
    masterLibraries[architectureName] = deepcopy(newLibrary)
    archIndex = MasterSolutionLibrary.ArchitectureIndexMap(architectureName)
    masterLibraries[architectureName].remapSolutionIndicesStartingFrom(archIndex)
    masterLibraries[architectureName].version = version
    return archIndex


def makeMasterLibraries(logicList, separate, version) -> Dict[str, MasterSolutionLibrary]:
    """Creates a dictionary of master solution libraries.

    Iterates through a list of LibraryLogic objects creating
    master solution libraries and modifying the indexing as
    rereuiqred.

    Args:
        logicFiles: List of LibraryLogic objects.
        separate: Separate libraries by architecture.
        version: User provided version for the libraries.

    Returns:
        An architecture separated master solution libraries
        or a single master solution library for all architectures.
    """
    masterLibraries = {}
    nextSolIndex = {}
    fullMasterLibrary = None

    for logic in logicList:
        (_, architectureName, _, solutionsForSchedule, _, newLibrary) = logic
        if separate:
            if architectureName in masterLibraries:
                nextSolIndex[architectureName] = masterLibraries[architectureName].merge(
                    deepcopy(newLibrary), nextSolIndex[architectureName]
                )
            else:
                nextSolIndex[architectureName] = addNewLibrary(
                    masterLibraries, newLibrary, architectureName, version
                )
        else:
            if fullMasterLibrary is None:
                fullMasterLibrary = deepcopy(newLibrary)
                fullMasterLibrary.version = version
            else:
                fullMasterLibrary.merge(deepcopy(newLibrary))
    return {"full": fullMasterLibrary} if fullMasterLibrary is not None else masterLibraries


def addFallbacks(masterLibraries, version: str) -> None:
    """Adds fallback librarys.

    Given a master solution library, add a fallback and if the corresponding
    architecture is unsupported, replace the library altogether with a fallback.

    Args:
        masterLibraries: A dictionary containing the master solution libraries.
        version: User provided version for the library.
    """
    archs, _ = splitArchs()
    for key, value in masterLibraries.items():
        if key != "fallback":  # why?
            value.insert(deepcopy(masterLibraries["fallback"]))  # why?
    for archName in archs:
        archName = archName.split("-", 1)[0]
        if archName not in masterLibraries:
            tPrint(1, "Using fallback for arch: " + archName)
            masterLibraries[archName] = deepcopy(masterLibraries["fallback"])
            # I don't think this is the right place for this
            masterLibraries[archName].version = version
    masterLibraries.pop("fallback")


def applyNaming(masterLibraries) -> None:
    """Assigns the solution code object file name for lazy libraries.

    Given a master solution library with lazy libraries, assigns the
    key associated with the lazy library (or name) as the value
    assiciated with the corresponding solution's code object file.

    Args:
        masterLibraries: A dictionary containing the master solution libraries.
    """
    for masterLibrary in masterLibraries.values():  # belongs elsewhere
        for name, lib in masterLibrary.lazyLibraries.items():  # belongs elsewhere
            for sol in lib.solutions.values():  # belongs elsewhere
                sol.originalSolution._state["codeObjectFile"] = name  # belongs elsewhere


def makeSolutions(
    masterLibraries: dict, separate: bool
):  # -> Generator[Solution]:# is breaking tensile
    """Extracts the solutions from the master solution library.

    Given a master solution library, forms a flattened generator that
    yields solutions by iterating over all of the solutions contained
    in the master solution libraries. If using separate architectures,
    but not using lazy loading lazyLibraries should be an empty dict.

    Args:
        masterLibraries: A dictionary containing the master solution libraries.

    Returns:
        List of library logic tuples.
    """
    gen1 = (
        sol.originalSolution
        for masterLibrary in masterLibraries.values()
        for sol in masterLibrary.solutions.values()
    )
    gen2 = (
        sol.originalSolution
        for masterLibrary in masterLibraries.values()
        for lib in masterLibrary.lazyLibraries.values()
        for sol in lib.solutions.values()
    )
    return itertools.chain(gen1, gen2)


def parseLibraryLogicFiles(logicFiles: List[str]) -> List[LibraryIO.LibraryLogic]:
    """Load and parse logic files.

    Given a list of paths to yaml files containing library logic, load the files
    into memory and parse the data into a named tuple or LibraryLogic. This operation
    is parallelized over N processes.

    Args:
        logicFiles: List of paths to logic files.

    Returns:
        List of library logic tuples.
    """
    return Common.ParallelMap(
        LibraryIO.parseLibraryLogicFile, logicFiles, "Reading logic files", multiArg=False
    )


def generateLogicData(
    logicFiles: List[str], version: str, printLevel: int, separate: bool
) -> MasterSolutionLibrary:
    """Generates a dictionary of master solution libraries.

    Args:
        logicFiles: List of paths to logic files.
        version: User provided version for the library.
        printLevel: Level of debug printing requested.
        separate: Separate libraries by architecture.

    Returns:
        A dictionary of architecture separated master solution libraries or
        a single master solution library for all architectures.
    """
    libraries = parseLibraryLogicFiles(logicFiles)
    # conditionally add progress bar
    logicList = libraries if not printLevel else Utils.tqdm(libraries, "Processing logic data")
    masterLibraries = makeMasterLibraries(logicList, separate, version)
    if separate:
        addFallbacks(masterLibraries, version)

    return masterLibraries


def generateSolutions(
    masterLibraries: Dict[str, MasterSolutionLibrary], separate: bool
) -> List[Solution]:
    """Generates a list of solutions.

    Args:
        masterLibraries: A dictionary of master solutions libraries.
        separate: Separate libraries by architecture.

    Returns:
        A solution list.
    """
    # remove duplicates while preserving order
    return list(dict.fromkeys(makeSolutions(masterLibraries, separate)))


################################################################################
# Write Benchmark Client Files
################################################################################
def writeBenchmarkClientFiles(libraryWorkingPath, tensileSourcePath, solutions, cxxCompiler):

    if not globalParameters["GenerateSourcesAndExit"]:
        copyStaticFiles(libraryWorkingPath)

    kernels, kernelsBetaOnly, _ = generateKernelObjectsFromSolutions(solutions)
    kernelWriterSource, kernelWriterAssembly, kernelMinNaming, _ = getKernelWriters(
        solutions, kernels
    )

    # write solution, kernels and CMake
    problemType = solutions[0]["ProblemType"]
    codeObjectFiles = writeKernels(
        libraryWorkingPath,
        cxxCompiler,
        [problemType],
        solutions,
        kernels,
        kernelsBetaOnly,
        kernelWriterSource,
        kernelWriterAssembly,
        errorTolerant=True,
    )

    newLibraryDir = ensurePath(os.path.join(libraryWorkingPath, "library"))
    newLibraryFile = os.path.join(newLibraryDir, "TensileLibrary.yaml")
    newLibrary = MasterSolutionLibrary.BenchmarkingLibrary(solutions)
    newLibrary.applyNaming(kernelMinNaming)

    LibraryIO.writeYAML(newLibraryFile, Utils.state(newLibrary))

    return (codeObjectFiles, newLibrary)


def WriteClientLibraryFromSolutions(solutionList, libraryWorkingPath, tensileSourcePath=None):

    if tensileSourcePath == None:
        tensileSourcePath = os.path.dirname(os.path.realpath(__file__))
    firstSolution = deepcopy(solutionList[0])
    problemType = firstSolution["ProblemType"].state
    problemType["DataType"] = problemType["DataType"].value
    problemType["DataTypeA"] = problemType["DataTypeA"].value
    problemType["DataTypeB"] = problemType["DataTypeB"].value
    problemType["DestDataType"] = problemType["DestDataType"].value
    problemType["ComputeDataType"] = problemType["ComputeDataType"].value
    problemType["MathDataTypeA"] = problemType["MathDataTypeA"].value
    problemType["MathDataTypeB"] = problemType["MathDataTypeB"].value
    problemType["F32XdlMathOp"] = problemType["F32XdlMathOp"].value
    cxxCompiler = globalParameters["CxxCompiler"]

    effectiveWorkingPath = os.path.join(libraryWorkingPath, "library")
    ensurePath(effectiveWorkingPath)
    mataDataFilePath = os.path.join(effectiveWorkingPath, "metadata.yaml")

    metaData = {"ProblemType": problemType}
    LibraryIO.writeYAML(mataDataFilePath, metaData)

    codeObjectFiles, newLibrary = writeBenchmarkClientFiles(
        libraryWorkingPath, tensileSourcePath, solutionList, cxxCompiler
    )

    return (codeObjectFiles, newLibrary)


################################################################################
# Write Master Solution Index CSV
################################################################################
def writeMasterSolutionIndexCSV(outputPath, masterLibraries):
    libraryPath = os.path.join(outputPath, "library")
    ensurePath(libraryPath)
    try:
        with open(os.path.join(libraryPath, "TensileMasterSolutionIndex.csv"), "w") as indexFile:
            indexFile.write(
                "architectureName,libraryName,libraryIndex,solutionIndex,solutionName\n"
            )
            for arch, lib in masterLibraries.items():
                for lazylibname, lazylibvals in lib.lazyLibraries.items():
                    for solidx, solution in lazylibvals.solutions.items():
                        line = ",".join(
                            str(x)
                            for x in [
                                arch,
                                lazylibname,
                                solidx,
                                solution.index,
                                solution.name,
                            ]
                        )
                        indexFile.write("%s\n" % (line))
    except IOError as err:
        tPrint(1, "Error writing MasterSolutionIndex %s" % err)


def verifyManifest(manifest: Path) -> bool:
    """Verifies whether the files listed in the manifest exist on disk.

    Args:
        manifest: Path to the manifest file.

    Returns:
        True if all files exist on disk, otherwise False.
    """
    with open(manifest, mode="r") as generatedFiles:
        for f in generatedFiles.readlines():
            if not Path(f.rstrip()).exists():
                return False
    return True


def findLogicFiles(
    path: Path,
    logicArchs: Set[str],
    lazyLoading: bool,
    experimentalDir: str,
    extraMatchers: Set[str] = {"hip"},
) -> List[str]:
    """Recursively searches the provided path for logic files.

    Args:
        path: The path to the directory to search.
        logicArchs: Target logic archiectures. These are interepreted as filename substrings
            for which logic files are to be included.
        extraMatchers: Additional directories to include for logic files.

    Returns:
        A list of Path objects representing the found YAML files.
    """
    isMatch = lambda file: any((arch in file.stem for arch in logicArchs.union(extraMatchers)))
    isExperimental = lambda path: not experimentalDir in str(path)

    extensions = ["*.yaml", "*.yml"]
    logicFiles = filter(isMatch, (file for ext in extensions for file in path.rglob(ext)))
    if not lazyLoading:
        if not experimentalDir:
            printWarning(
                "Configuration parameter `ExperimentalLogicDir` is an empty string, "
                "logic files may be filtered incorrectly."
            )
        logicFiles = filter(isExperimental, logicFiles)

    return list(str(l) for l in logicFiles)


def sanityCheck(
    srcLibPaths: List[str],
    asmLibPaths: List[str],
    codeObjectPaths: List[str],
    genSourcesAndExit: bool,
):
    """Verifies that generated code object paths match associated library paths.

    Args:
        srcLibPaths: Source library paths (.hsaco).
        asmLibPaths: Assembly library paths (.co).
        coPaths: Code object paths containing generated kernels; should contain all assembly
            and source library paths.
        genSourcesAndExit: Flag identifying whether only source file should be generated.

    Raises:
        ValueError: If code object paths do not match library paths.
    """
    libPaths = set([Path(p).resolve() for p in srcLibPaths + asmLibPaths])
    coPaths = set([Path(p).resolve() for p in codeObjectPaths])

    extraCodeObjects = coPaths - libPaths
    if extraCodeObjects:
        raise ValueError(
            f"Sanity check failed; unexpected code object files: "
            f"{[p.name for p in extraCodeObjects]}"
        )

    if not genSourcesAndExit:
        extraLibs = libPaths - coPaths
        if extraLibs:
            raise ValueError(
                f"Sanity check failed; missing expected code object files: "
                f"{[p.name for p  in extraLibs]}"
            )


def generateClientConfig(
    outputPath: Path,
    masterFile: Path,
    codeObjectFiles: List[str],
    configFile: str = "best-solution.ini",
) -> None:
    """Generates a client config file.

    Generates a client config file corresponding to a master library file and code-object parameters
    created by a TensileCreateLibrary invocation. Also sets best-solution-mode to True.

    Args:
        outputPath: The path to the tensile output directory where output files are written.
        masterFile: Path to the master library file (.dat or .yaml).
        codeObjectFiles: List of code object files created by TensileCreateLibrary.
        configFile: Name of config file written to the output directory.
    """
    iniFile = outputPath / configFile

    def param(key, value):
        f.write(f"{key}={value}\n")

    with open(iniFile, "w") as f:
        if not masterFile.is_file():
            warnings.warn(
                UserWarning(f"{masterFile} does not exist. best-solution.ini may be invalid.")
            )

        param("library-file", masterFile)
        for coFile in codeObjectFiles:
            codeObject: Path = outputPath / coFile
            if not codeObject.is_file():
                warnings.warn(
                    UserWarning(f"{codeObject} does not exist. best-solution.ini may be invalid.")
                )

            param("code-object", outputPath / coFile)

        param("best-solution", True)


def generateLazyMasterFileList(
    masterFileList: List[Tuple[str, MasterSolutionLibrary]]
) -> List[Tuple[str, MasterSolutionLibrary]]:
    """Generates a list of tuples that represent the name and the state associated with the lazy master libraries.

    This function takes a list of MasterSolutionLibraries and traverses each lazy libraries.
    It collects the items (i.e. the name and corresponding master file) and adds them to list
    of master files.

    Args:
        masterLibraries: A list of name / master solution library pairs.

    Returns:
        List of pairs of master solutions libraries and the corresponding name.
    """
    return [t for _, lib in masterFileList for t in lib.lazyLibraries.items()]


def generateMasterFileList(
    masterLibraries: dict, archs: List[str], lazy: bool
) -> List[Tuple[str, MasterSolutionLibrary]]:
    """Generates a list of tuples that represent the name and the state associated with the master libraries.

    This function takes a dictionary with keys corresponding to a target architecture and values
    corresponding to the master solution library for that architecture. The function generates a
    tuple consisting of a MasterSolutionLibrary and the associated name. When not separating architectures,
    the key full will appear in masterLibraries indicating that all libraries are combinded into a
    single master library.

    Args:
        masterLibraries: A dictionary of architecture name / master solution library pairs.
        archs: A list of supported architectures.
        lazy: If True, add lazy library master files.

    Returns:
        List of pairs of master solutions libraries and the corresponding name.
    """
    if "full" in masterLibraries.keys():
        return [("TensileLibrary", masterLibraries["full"])]

    baseName = "TensileLibrary_lazy_" if lazy else "TensileLibrary_"
    result = [
        (baseName + arch, masterLibrary)
        for arch, masterLibrary in masterLibraries.items()
        if arch in archs
    ]

    return result + generateLazyMasterFileList(result) if lazy else result


def writeMasterFile(
    libraryPath: Path, format: str, naming: dict, name: str, lib: MasterSolutionLibrary
) -> None:
    """Writes a master file to disk as a .yaml or .dat file.

    Args:
        libraryPath: Path to library subdirectory located in the tensile output directory.
        format: Output format of written file (.dat or .yaml).
        naming: Kernel minimum naming.
        name: Name of the masterfile.
        lib: Master solution library data.
    """
    lib.applyNaming(naming)
    LibraryIO.write(str(libraryPath / name), Utils.state(lib), format)


################################################################################
# Tensile Create Library
################################################################################
@profile
def TensileCreateLibrary():

    ##############################################################################
    # Parse Command Line Arguments
    ##############################################################################
    def splitExtraParameters(par):
        """
        Allows the --global-parameters option to specify any parameters from the command line.
        """

        (key, value) = par.split("=")
        value = eval(value)
        return (key, value)

    tPrint(3, "Arguments: %s" % sys.argv)
    argParser = argparse.ArgumentParser()
    argParser.add_argument("LogicPath", help="Path to LibraryLogic.yaml files.")
    argParser.add_argument("OutputPath", help="Where to write library files?")
    argParser.add_argument(
        "RuntimeLanguage", help="Which runtime language?", choices=["OCL", "HIP", "HSA"]
    )

    compilerChoices = ["amdclang++", "hipcc"] if os.name != "nt" else ["clang++", "hipcc"]
    argParser.add_argument(
        "--cxx-compiler",
        dest="CxxCompiler",
        choices=compilerChoices,
        action="store",
        default=compilerChoices[0],
        help="C++ compiler used when generating binaries."
        " On linux, amdclang++ (default) or hipcc. On Windows clang++ (default) or hipcc.",
    )

    argParser.add_argument("--cmake-cxx-compiler", dest="CmakeCxxCompiler", action="store")
    argParser.add_argument(
        "--code-object-version",
        dest="CodeObjectVersion",
        choices=["default", "V4", "V5"],
        action="store",
    )
    argParser.add_argument(
        "--architecture",
        dest="Architecture",
        type=str,
        action="store",
        default="all",
        help="Architectures to generate a library for. When specifying multiple options, "
        "use quoted, semicolon delimited architectures, e.g., --architecture='gfx908;gfx1012'. "
        "Supported archiectures include: " + " ".join(architectureMap.keys()),
    )
    argParser.add_argument("--merge-files", dest="MergeFiles", action="store_true")
    argParser.add_argument("--no-merge-files", dest="MergeFiles", action="store_false")
    argParser.add_argument(
        "--num-merged-files",
        dest="NumMergedFiles",
        type=int,
        default=1,
        help="Number of files the kernels should be written into.",
    )
    argParser.add_argument("--short-file-names", dest="ShortNames", action="store_true")
    argParser.add_argument("--no-short-file-names", dest="ShortNames", action="store_false")
    argParser.add_argument(
        "--library-print-debug",
        dest="LibraryPrintDebug",
        action="store_true",
        help="Deprecated",
    )
    argParser.add_argument(
        "--no-library-print-debug", dest="LibraryPrintDebug", action="store_false"
    )
    argParser.add_argument(
        "--no-enumerate", action="store_true", help="Do not run rocm_agent_enumerator."
    )
    argParser.add_argument(
        "--embed-library",
        dest="EmbedLibrary",
        help="Embed (new) library files into static variables.  Specify the name of the library.",
    )

    argParser.add_argument(
        "--embed-library-key",
        dest="EmbedLibraryKey",
        default=None,
        help="Access key for embedding library files.",
    )
    argParser.add_argument("--version", help="Version string to embed into library file.")
    argParser.add_argument(
        "--generate-manifest-and-exit",
        dest="GenerateManifestAndExit",
        action="store_true",
        default=False,
        help="Output manifest file with list of expected library objects and exit.",
    )
    argParser.add_argument(
        "--verify-manifest",
        dest="VerifyManifest",
        action="store_true",
        default=False,
        help="Verify manifest file against generated library files and exit.",
    )
    argParser.add_argument(
        "--library-format",
        dest="LibraryFormat",
        choices=["yaml", "msgpack"],
        action="store",
        default="msgpack",
        help="select which library format to use",
    )
    argParser.add_argument(
        "--generate-sources-and-exit",
        dest="GenerateSourcesAndExit",
        action="store_true",
        default=False,
        help="Output source files only and exit.",
    )
    argParser.add_argument(
        "--jobs",
        "-j",
        dest="CpuThreads",
        type=int,
        default=-1,
        help="Number of parallel jobs to launch.",
    )
    argParser.add_argument(
        "--verbose",
        "-v",
        dest="PrintLevel",
        type=int,
        default=1,
        help="Set printout verbosity level.",
    )
    argParser.add_argument(
        "--separate-architectures",
        dest="SeparateArchitectures",
        action="store_true",
        default=False,
        help="Separates TensileLibrary file by architecture",
    )
    argParser.add_argument(
        "--lazy-library-loading",
        dest="LazyLibraryLoading",
        action="store_true",
        default=False,
        help="Loads Tensile libraries when needed instead of upfront.",
    )
    argParser.add_argument(
        "--build-client",
        dest="BuildClient",
        action="store_true",
        help="Build Tensile client",
    )
    argParser.add_argument(
        "--client-config",
        dest="ClientConfig",
        action="store_true",
        help="Create client config for setting the library and code object files",
    )
    argParser.add_argument("--global-parameters", nargs="+", type=splitExtraParameters, default=[])
    argParser.add_argument(
        "--ignore-asm-cap-cache",
        dest="IgnoreAsmCapCache",
        action="store_true",
        default=False,
        help="Ignore asm cap cache and derive the asm caps at runtime",
    )
    argParser.add_argument(
        "--write-master-solution-index",
        dest="WriteMasterSolutionIndex",
        action="store_true",
        default=False,
        help="Output master solution index in csv format.",
    )
    args = argParser.parse_args()

    logicPath = args.LogicPath
    outputPath = args.OutputPath
    CxxCompiler = args.CxxCompiler
    libraryFormat = args.LibraryFormat
    tPrint(3, "OutputPath: %s" % outputPath)
    ensurePath(outputPath)
    outputPath = os.path.abspath(outputPath)
    arguments = {}
    arguments["RuntimeLanguage"] = args.RuntimeLanguage
    arguments["CodeObjectVersion"] = args.CodeObjectVersion
    arguments["Architecture"] = args.Architecture
    arguments["SeparateArchitectures"] = args.SeparateArchitectures
    arguments["LazyLibraryLoading"] = args.LazyLibraryLoading
    arguments["CxxCompiler"] = args.CxxCompiler
    if args.CmakeCxxCompiler:
        os.environ["CMAKE_CXX_COMPILER"] = args.CmakeCxxCompiler
    arguments["MergeFiles"] = args.MergeFiles
    arguments["NumMergedFiles"] = args.NumMergedFiles
    arguments["ShortNames"] = args.ShortNames
    arguments["LibraryPrintDebug"] = args.LibraryPrintDebug
    arguments["CodeFromFiles"] = False
    arguments["EmbedLibrary"] = args.EmbedLibrary
    arguments["LibraryFormat"] = args.LibraryFormat
    if args.no_enumerate:
        arguments["ROCmAgentEnumeratorPath"] = False

    arguments["GenerateManifestAndExit"] = args.GenerateManifestAndExit
    arguments["VerifyManifest"] = args.VerifyManifest

    arguments["GenerateSourcesAndExit"] = args.GenerateSourcesAndExit
    if arguments["GenerateSourcesAndExit"]:
        # Generated sources are preserved and go into output dir
        arguments["WorkingPath"] = outputPath

    arguments["CpuThreads"] = args.CpuThreads
    arguments["PrintLevel"] = args.PrintLevel
    arguments["IgnoreAsmCapCache"] = args.IgnoreAsmCapCache
    arguments["WriteMasterSolutionIndex"] = args.WriteMasterSolutionIndex

    for key, value in args.global_parameters:
        arguments[key] = value

    globalParameters["PrintLevel"] = arguments["PrintLevel"]

    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "# Tensile Create Library")
    tPrint(3, HR)
    tPrint(3, "")

    assignGlobalParameters(arguments)

    manifestFile = Path(outputPath) / TENSILE_LIBRARY_DIR / TENSILE_MANIFEST_FILENAME
    manifestFile.parent.mkdir(exist_ok=True)

    if globalParameters["VerifyManifest"]:
        if verifyManifest(manifestFile):
            tPrint(1, "Successfully verified all files in manifest were generated")
            return
        else:
            printExit("Failed to verify all files in manifest")

    tPrint(
        1,
        "# CodeObjectVersion from TensileCreateLibrary: %s" % arguments["CodeObjectVersion"],
    )
    tPrint(1, "# CxxCompiler       from TensileCreateLibrary: %s" % CxxCompiler)
    tPrint(
        1,
        "# Architecture      from TensileCreateLibrary: %s" % arguments["Architecture"],
    )
    tPrint(1, "# LibraryFormat     from TensileCreateLibrary: %s" % libraryFormat)

    if not os.path.exists(logicPath):
        printExit("LogicPath %s doesn't exist" % logicPath)

    # CLI uses `;` delimiters, CMake uses `_` delimiters
    logicArchs = splitDelimitedString(arguments["Architecture"], {";", "_"})
    logicArchs = {name for name in (getArchitectureName(gfxName) for gfxName in logicArchs) if name}

    if globalParameters["LazyLibraryLoading"] and not (
        globalParameters["MergeFiles"] and globalParameters["SeparateArchitectures"]
    ):
        printExit(
            "--lazy-library-loading requires --merge-files and --separate-architectures enabled"
        )

    logicFiles = findLogicFiles(
        Path(logicPath),
        logicArchs,
        lazyLoading=globalParameters["LazyLibraryLoading"],
        experimentalDir=globalParameters["ExperimentalLogicDir"],
    )

    tPrint(1, "# LibraryLogicFiles:" % logicFiles)
    for logicFile in logicFiles:
        tPrint(1, "#   %s" % logicFile)

    masterLibraries = generateLogicData(
        logicFiles, args.version, args.PrintLevel, args.SeparateArchitectures
    )
    solutions = generateSolutions(masterLibraries, args.SeparateArchitectures)

    if globalParameters["LazyLibraryLoading"] and arguments["WriteMasterSolutionIndex"]:
        writeMasterSolutionIndexCSV(outputPath, masterLibraries)

    kernels, kernelHelperObjs, _ = generateKernelObjectsFromSolutions(solutions)

    # if any kernels are assembly, append every ISA supported
    kernelWriterSource, kernelWriterAssembly, kernelMinNaming, _ = getKernelWriters(
        solutions, kernels
    )

    staticFiles = copyStaticFiles(outputPath)

    (solutionFiles, sourceKernelFiles, asmKernelFiles, sourceLibFiles, asmLibFiles) = (
        buildObjectFileNames(
            kernelWriterSource,
            kernelWriterAssembly,
            solutions,
            kernels,
            kernelHelperObjs,
        )
    )

    (_, _, _, sourceLibPaths, asmLibPaths, libMetadataPaths) = buildObjectFilePaths(
        outputPath,
        solutionFiles,
        sourceKernelFiles,
        asmKernelFiles,
        sourceLibFiles,
        asmLibFiles,
        masterLibraries,
    )

    toFile(Path(manifestFile), libMetadataPaths + sourceLibPaths + asmLibPaths)
    if globalParameters["GenerateManifestAndExit"]:
        return

    if not arguments["GenerateSourcesAndExit"]:
        writeCMake(outputPath, solutionFiles, sourceKernelFiles, staticFiles, masterLibraries)

    # Make sure to copy the library static files.
    for fileName in staticFiles:
        shutil.copy(os.path.join(globalParameters["SourcePath"], fileName), outputPath)

    codeObjectFiles = writeKernels(
        outputPath,
        CxxCompiler,
        None,
        solutions,
        kernels,
        kernelHelperObjs,
        kernelWriterSource,
        kernelWriterAssembly,
    )

    sanityCheck(
        sourceLibPaths,
        asmLibPaths,
        codeObjectFiles,
        globalParameters["GenerateSourcesAndExit"],
    )

    tPrint(2, f"codeObjectFiles: {codeObjectFiles}")
    tPrint(2, f"sourceLibPaths + asmLibPaths: {sourceLibPaths + asmLibPaths}")

    archs = [
        gfxName(arch)
        for arch in globalParameters["SupportedISA"]
        if globalParameters["AsmCaps"][arch]["SupportedISA"]
    ]

    newLibraryDir = Path(outputPath) / "library"
    newLibraryDir.mkdir(exist_ok=True)

    masterFileList = generateMasterFileList(masterLibraries, archs, args.LazyLibraryLoading)

    for name, lib in masterFileList:
        writeMasterFile(newLibraryDir, args.LibraryFormat, kernelMinNaming, name, lib)

    if args.EmbedLibrary or args.ClientConfig:
        masterFile, fullMasterLibrary = masterFileList[0]
        ext = ".yaml" if globalParameters["LibraryFormat"] == "yaml" else ".dat"

    if args.EmbedLibrary:
        embedFileName = Path(outputPath) / "library" / args.EmbedLibrary
        EmbeddedData.generateLibrary(
            embedFileName,
            args.EmbedLibraryKey,
            (newLibraryDir / masterFile).with_suffix(ext),
            fullMasterLibrary.cpp_base_class,
            codeObjectFiles,
        )

    if args.BuildClient:
        tPrint(1, "# Building Tensile Client")
        ClientExecutable.getClientExecutable(outputPath)

    if args.ClientConfig:
        generateClientConfig(Path(outputPath), Path(masterFile).with_suffix(ext), codeObjectFiles)

    tPrint(1, "# Tensile Library Writer DONE")
    tPrint(1, HR)
    tPrint(1, "")
