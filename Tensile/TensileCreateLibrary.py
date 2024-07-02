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

from io import TextIOWrapper
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
    architectureMap,
    ensurePath,
    CHeader,
    CMakeHeader,
    assignGlobalParameters,
    gfxName,
    printWarning,
    supportedCompiler,
    which,
)
from .KernelWriterAssembly import KernelWriterAssembly
from .KernelWriterSource import KernelWriterSource
from .KernelWriterBase import KernelWriterBase
from .SolutionLibrary import MasterSolutionLibrary
from .SolutionStructs import Solution
from .Utilities.String import splitDelimitedString
from .Utilities.Profile import profile
from .Utilities.toFile import toFile
from .TensileCreateLib.ParseArguments import parseArguments

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
from typing import Dict, Any, Set, List, Tuple, Callable
from pathlib import Path

TENSILE_MANIFEST_FILENAME = "TensileManifest.txt"
TENSILE_LIBRARY_DIR = "library"


################################################################################
def processKernelSource(kernel, kernelWriterSource, kernelWriterAssembly):
    """Generate source for a single kernel.
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
def prepAsm(kernelWriterAssembly: KernelWriterAssembly, buildPath: Path):
    """Create and prepare the assembly directory. - called ONCE per output directory.

    This function is called once per output directory. It creates a directory
    "assembly" under the provided **buildPath**, and generates a bash script for
    compiling object files into code object files.

    Args:
        kernelWriterAssembly: Assembly writer object.
        buildPath: Path to directory where assembly files will be written.
    """
    asmPath = buildPath / "assembly"
    asmPath.mkdir(exist_ok=True)

    isa = globalParameters["CurrentISA"]
    assemblerFileName = asmPath / f"asm-new.{"bat" if os.name == "nt" else "sh"}"

    with open(assemblerFileName, "w") as assemblerFile:
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

    os.chmod(assemblerFileName, 0o777)


################################################################################
def buildKernelSourceAndHeaderFiles(results, outputPath):
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
    kernelsWithBuildErrs = {}
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


def filterProcessingErrors(
    kernels: List[Solution],
    solutions: List[Solution],
    results: List[Any],
    printLevel: int,
    errorTolerant: bool,
) -> Tuple[List[Solution], List[Solution], List[Any]]:
    """Filters out processing errors from lists of kernels, solutions, and results.

    This function iterates through the results of **processKernelSource** and identifies
    any errors encountered during processing. If an error is found (-2 error code),
    the corresponding kernel, solution, and result are appended to separate lists
    for removal. After processing, items identified for removal are deleted from the
    original lists of kernels, solutions, and results.

    Args:
        kernels: List of Solution objects representing kernels.
        solutions: List of Solution objects associated with kernels.
        results: List of tuples representing processing results.
        printLevel: Print level indicator.

    Returns:
        Tuple[List[Solution], List[Solution], List[Any]]: Tuple containing filtered lists
        of kernels, solutions, and results after removing items with processing errors.

    Raises:
        KeyError: If 'PrintLevel' key is not found in the params dictionary.
    """
    ## TODO(bstefanuk): Create ticket to refactor and test this function... there's lots of room for optimization
    ## For each kernel we check if there was an error while processing it
    ## If there is an error, then we append to three lists
    ## afterwards, we loop over those lists and remove items from the list of kernels
    removeKernels = []
    removeSolutions = []
    removeResults = []
    for kernIdx, res in enumerate(results) if printLevel == 0 else Utils.tqdm(enumerate(results)):
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
    return kernels, solutions, results


def filterBuildErrors(
    kernels: List[Solution],
    kernelsWithBuildErrors: List[Solution],
    writerSelectionFn: Callable[[str], KernelWriterSource | KernelWriterAssembly],
    errorTolerant: bool,
):
    """
    Filters a list of kernels based on build errors and error tolerance.

    Args:
        kernels: A list of `Solution` objects representing kernels to filter.
        kernelsWithBuildErrors: A list of `Solution` objects that have build errors.
        errorTolerant: A boolean indicating whether to tolerate build errors.

    Returns:
        A filtered list of kernels (**Solution** objects) that are eligible for building.

    Raises:
        SystemExit: If **error_tolerant** is False and any kernels have build errors.
    """
    if not errorTolerant and len(kernelsWithBuildErrors) > 0:
        printExit(
            "** kernel compilation failure **"
            "Kernel compilation failed in one or more subprocesses. "
            "Consider setting CpuThreads=0 and re-run to make debugging easier."
        )

    def noBuildError(kernel):
        kernelName = writerSelectionFn(kernel["KernelLanguage"]).getKernelName(kernel)
        return kernelName not in kernelsWithBuildErrors

    return list(filter(noBuildError, kernels))


def markDuplicateKernels(
    kernels: List[Solution], kernelWriterAssembly: KernelWriterAssembly
) -> List[Solution]:
    """Marks duplicate kernels based on their generated file base names.

    Kernels written in Assembly language may generate duplicate output file names,
    leading to potential race conditions. This function identifies such duplicates within
    the provided list of Solution objects and marks them to prevent issues.

    Args:
        kernels: A list of Solution objects representing kernels to be processed.

    Returns:
        A modified list of Solution objects where kernels identified as duplicates
        are marked with a `duplicate` attribute indicating their duplication status.
    """
    # Kernels may be intended for different .co files, but generate the same .o file
    # Mark duplicate kernels to avoid race condition
    # @TODO improve organization so this problem doesn't appear
    ## TODO(bstefanuk): Create ticket to "improve organization so this problem doesn't appear"
    visited = set()
    for kernel in kernels:
        if kernel["KernelLanguage"] == "Assembly":
            curr = kernelWriterAssembly.getKernelFileBase(kernel)
            kernel.duplicate = curr in visited
            visited.add(curr)
    return kernels


def generateKernelSourcesAndHeaders(
    kernelObj: KernelWriterBase,
    srcFile: TextIOWrapper | None,
    hdrFile: TextIOWrapper | None,
    outputPath: Path,
    mergeFiles: bool,
):
    """Writes a source and header file to disk for a kernel object and writes them to specified files or merges them.

    Args:
        kernelObj: The kernel object implementing `KernelWriterBase` interface, providing source and header strings.
        srcFile: The text file object to write the kernel's C++ source code. If `mergeFiles` is False, should be initially None.
        hdrFile: The text file object to write the kernel's C++ header code. If `mergeFiles` is False, should be initially None.
        outputPath: The directory path where generated files should be saved.
        mergeFiles: Flag indicating whether to merge source and header into existing files (`srcFile` and `hdrFile` must be pre-initialized).

    Raises:
        ValueError: If `srcFile` or `hdrFile` is None and `mergeFiles` is True, indicating file operations cannot proceed.

    Notes:
        - If `mergeFiles` is False, new `.cpp` and `.h` files are created in the `outputPath/Kernels` directory named after the kernel.
        - Closes `srcFile` and `hdrFile` after writing if `mergeFiles` is False. Otherwise it is the user's responsibility.
    """
    if (srcFile is None or hdrFile is None) and mergeFiles:
        raise ValueError(
            "Cannot conduct file operations on NoneType, `srcFile` and `hdrFile` must be pre-initialized if `mergeFiles == true`"
        )

    kernelName = kernelObj.getKernelName()

    if not mergeFiles:
        srcFilename = outputPath / "Kernels" / f"{kernelName}.cpp"
        srcFile = open(srcFilename, "w")
        srcFile.write(CHeader)

        hdrFilename = outputPath / "Kernels" / f"{kernelName}.h"
        hdrFile = open(hdrFilename, "w")
        hdrFile.write(CHeader)

    err, src = kernelObj.getSourceFileString()
    srcFile.write(src)
    if err:
        printWarning(f"*** Invalid kernel {kernelName}")

    hdr = kernelObj.getHeaderFileString()
    hdrFile.write(hdr)

    if not mergeFiles:
        srcFile.close()
        hdrFile.close()


################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
def writeKernels(
    outputPath: str,
    cxxCompiler: str,
    params: Dict[str, Any],
    solutions: List[Solution],
    kernels: List[Solution],
    kernelHelperObjs: List[KernelWriterBase],  # TODO(bstefanuk): Verify this type is correct
    kernelWriterSource: KernelWriterSource,
    kernelWriterAssembly: KernelWriterAssembly,
    errorTolerant: bool = False,
):
    start = time.time()

    # Push working path into build_tmp folder because there may be more than
    # one process running this script. This is to avoid build directory clashing.
    # NOTE: file paths must not contain the lower case word 'kernel' or the
    # /opt/rocm/bin/extractkernel will fail.
    # See buildSourceCodeObjectFile:167 for the call to this binary.
    ## TODO(bstefanuk): Is there a way to get this to work without change global state?
    Common.pushWorkingPath("build_tmp")
    Common.pushWorkingPath(os.path.basename(outputPath).upper())

    tPrint(1, "# Writing Kernels...")
    kernelSourceFile = None
    kernelHeaderFile = None

    ## TODO(bstefanuk): Is this even used? My shows nothing popluated in this file?
    if not params["MergeFiles"] or params["NumMergedFiles"] > 1 or params["LazyLibraryLoading"]:
        ensurePath(os.path.join(outputPath, "Kernels"))

    ## TODO(bstef): This uses global state from "WorkingPath", perhaps replace with the params
    ## variable... but may there are side effects.
    prepAsm(kernelWriterAssembly, Path(globalParameters["WorkingPath"]))

    kernels = markDuplicateKernels(kernels, kernelWriterAssembly)

    kIter = zip(
        kernels,
        itertools.repeat(kernelWriterSource),
        itertools.repeat(kernelWriterAssembly),
    )
    results = Common.ParallelMap(processKernelSource, kIter, "Generating kernels")
    kernels, solutions, results = filterProcessingErrors(
        kernels, solutions, results, params["PrintLevel"], errorTolerant
    )

    kernelFiles, kernelsWithBuildErrors = buildKernelSourceAndHeaderFiles(results, outputPath)

    ## TODO(bstefanuk): I'm not convinced this is returning distinct output above just calling the base class function
    ## KernelWriter.getKernelName(...)
    writerSelector = lambda lang: kernelWriterAssembly if lang == "Assembly" else kernelWriterSource
    kernelsToBuild = filterBuildErrors(
        kernels, kernelsWithBuildErrors, writerSelector, errorTolerant
    )

    # Put all kernel helper objects into the first merged kernel file
    if globalParameters["NumMergedFiles"] > 1 and len(kernelFiles) > 0:
        kernelFilename = kernelFiles[0].replace(".cpp", "")
        ## TODO(bstefanuk): There's a potential bug here if we use NumMergeFiles=2 we open a file
        ## then on line 767 if we also specify NoMergeFiles we open another file, bind it to the same
        ## handle, but don't actually close the original file we opened. Same thing for the header
        ## file on line 785
        kernelSourceFile = open(kernelFilename + ".cpp", "a", encoding="utf-8")
        kernelHeaderFile = open(kernelFilename + ".h", "a", encoding="utf-8")
    elif globalParameters["MergeFiles"] or globalParameters["LazyLibraryLoading"]:
        kernelSourceFilename = os.path.join(os.path.normcase(outputPath), "Kernels.cpp")
        kernelHeaderFilename = os.path.join(os.path.normcase(outputPath), "Kernels.h")
        kernelSourceFile = open(kernelSourceFilename, "a", encoding="utf-8")
        kernelHeaderFile = open(kernelHeaderFilename, "a", encoding="utf-8")

    for ko in kernelHelperObjs:
        newKernelFile = generateKernelSourcesAndHeaders(
            ko, kernelSourceFile, kernelHeaderFile, outputPath, params["MergeFiles"]
        )
        kernelFiles.append(newKernelFile)

    # close merged
    if globalParameters["MergeFiles"]:
        if kernelSourceFile:
            kernelSourceFile.close()
        if kernelHeaderFile:
            kernelHeaderFile.close()

    codeObjectFiles = []
    if not globalParameters["GenerateSourcesAndExit"]:
        codeObjectFiles += buildSourceCodeObjectFiles(cxxCompiler, kernelFiles, outputPath)
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
    for arch, lib in masterLibraries.items():
        if globalParameters["LazyLibraryLoading"]:
            newMetadataPaths.add(os.path.join(libDir, "TensileLibrary_lazy_" + arch + libraryExt))
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


################################################################################
# Generate Logic Data and Solutions
################################################################################
def generateLogicDataAndSolutions(logicFiles, version: str):
    libraries = Common.ParallelMap(
        LibraryIO.parseLibraryLogicFile,
        logicFiles,
        "Reading logic files",
        multiArg=False,
    )
    solutions = []
    masterLibraries = {}
    fullMasterLibrary = None

    nextSolIndex = {}
    for logic in (
        libraries
        if globalParameters["PrintLevel"] == 0
        else Utils.tqdm(libraries, "Processing logic data")
    ):
        (_, architectureName, _, solutionsForSchedule, _, newLibrary) = logic

        if globalParameters["SeparateArchitectures"] or globalParameters["LazyLibraryLoading"]:

            if architectureName in masterLibraries:
                nextSolIndex[architectureName] = masterLibraries[architectureName].merge(
                    deepcopy(newLibrary), nextSolIndex[architectureName]
                )
            else:
                masterLibraries[architectureName] = deepcopy(newLibrary)
                archIndexMap = MasterSolutionLibrary.ArchitectureIndexMap(architectureName)
                masterLibraries[architectureName].remapSolutionIndicesStartingFrom(archIndexMap)
                nextSolIndex[architectureName] = archIndexMap
                masterLibraries[architectureName].version = version
        else:
            if fullMasterLibrary is None:
                fullMasterLibrary = deepcopy(newLibrary)
                fullMasterLibrary.version = version
            else:
                fullMasterLibrary.merge(deepcopy(newLibrary))

    (archs, _) = splitArchs()
    if globalParameters["SeparateArchitectures"] or globalParameters["LazyLibraryLoading"]:
        if "fallback" in masterLibraries.keys():
            for key, value in masterLibraries.items():
                if key != "fallback":
                    value.insert(deepcopy(masterLibraries["fallback"]))
            for archName in archs:
                archName = archName.split("-", 1)[0]
                if archName not in masterLibraries:
                    tPrint(1, "Using fallback for arch: " + archName)
                    masterLibraries[archName] = deepcopy(masterLibraries["fallback"])
                    masterLibraries[archName].version = version

            masterLibraries.pop("fallback")

        for _, masterLibrary in masterLibraries.items():
            for _, sol in masterLibrary.solutions.items():
                solutions.append(sol.originalSolution)
            for name, lib in masterLibrary.lazyLibraries.items():
                for _, sol in lib.solutions.items():
                    sol.originalSolution._state["codeObjectFile"] = name
                    solutions.append(sol.originalSolution)
    else:
        for _, sol in fullMasterLibrary.solutions.items():
            solutions.append(sol.originalSolution)

    # remove duplicates while preserving order
    solutions = list(dict.fromkeys(solutions))
    return solutions, masterLibraries, fullMasterLibrary


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
    codeObjectFiles = writeKernels(
        libraryWorkingPath,
        cxxCompiler,
        globalParameters,  #  TODO(bstefanuk): This may be a bad idea since it will be a ref and can be mutated
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
    """Generates a list of tuples that represent the name and the state associated with the architecture
        specific master libraries.

    This function takes a dictionary with keys corresponding to a target architecture and values
    corresponding to the master solution library for that architecture. The function generates a
    tuple consisting of a MasterSolutionLibrary and the associated name.

    Args:
        masterLibraries: A dictionary of architecture name / master solution library pairs.
        archs: A list of supported architectures.
        lazy: If True, add lazy library master files.

    Returns:
        List of pairs of master solutions libraries and the corresponding name.
    """
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

    tPrint(3, "Arguments: %s" % sys.argv)
    args = parseArguments()

    lazyLoading = args["LazyLibraryLoading"]
    separateArchs = args["SeparateArchitectures"]
    mergeFiles = args["MergeFiles"]
    embedLibrary = args["EmbedLibrary"]
    cxxCompiler = args["CxxCompiler"]
    libraryFormat = args["LibraryFormat"]
    logicPath = args["LogicPath"]
    outputPath = args["OutputPath"]

    globalParameters["PrintLevel"] = args["PrintLevel"]

    tPrint(3, "OutputPath: %s" % outputPath)
    ensurePath(outputPath)
    outputPath = os.path.abspath(outputPath)

    assignGlobalParameters(args)

    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "# Tensile Create Library")
    tPrint(3, HR)
    tPrint(3, "")

    manifestFile = Path(outputPath) / TENSILE_LIBRARY_DIR / TENSILE_MANIFEST_FILENAME
    manifestFile.parent.mkdir(exist_ok=True)

    if args["VerifyManifest"]:
        if verifyManifest(manifestFile):
            tPrint(1, "Successfully verified all files in manifest were generated")
            return
        else:
            printExit("Failed to verify all files in manifest")

    tPrint(
        1,
        "# CodeObjectVersion from TensileCreateLibrary: %s" % args["CodeObjectVersion"],
    )
    tPrint(1, "# CxxCompiler       from TensileCreateLibrary: %s" % cxxCompiler)
    tPrint(
        1,
        "# Architecture      from TensileCreateLibrary: %s" % args["Architecture"],
    )
    tPrint(1, "# CxxCompiler       from TensileCreateLibrary: %s" % cxxCompiler)
    tPrint(1, "# Architecture      from TensileCreateLibrary: %s" % args["Architecture"])
    tPrint(1, "# LibraryFormat     from TensileCreateLibrary: %s" % libraryFormat)

    if not os.path.exists(logicPath):
        printExit("LogicPath %s doesn't exist" % logicPath)

    # CLI uses `;` delimiters, CMake uses `_` delimiters
    logicArchs = splitDelimitedString(args["Architecture"], {";", "_"})
    logicArchs = {name for name in (getArchitectureName(gfxName) for gfxName in logicArchs) if name}

    if lazyLoading and not (mergeFiles and separateArchs):
        printExit(
            "--lazy-library-loading requires --merge-files and --separate-architectures enabled"
        )

    logicFiles = findLogicFiles(
        Path(logicPath),
        logicArchs,
        lazyLoading=lazyLoading,
        experimentalDir=globalParameters["ExperimentalLogicDir"],
    )

    tPrint(1, f"# LibraryLogicFiles: found {len(logicFiles)} files")
    tPrint(1, "#      set --verbose=2 to view all files")
    tPrint(2, "#   " + "\n#   ".join(logicFiles))

    solutions, masterLibraries, fullMasterLibrary = generateLogicDataAndSolutions(
        logicFiles, args["Version"]
    )

    if lazyLoading and args["WriteMasterSolutionIndex"]:
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
    if args["GenerateManifestAndExit"]:
        return

    if not args["GenerateSourcesAndExit"]:
        writeCMake(outputPath, solutionFiles, sourceKernelFiles, staticFiles, masterLibraries)

    # Make sure to copy the library static files.
    for fileName in staticFiles:
        shutil.copy(os.path.join(globalParameters["SourcePath"], fileName), outputPath)

    codeObjectFiles = writeKernels(
        outputPath,
        cxxCompiler,
        args,
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

    # do we need this or have we already done this?
    archs = [
        gfxName(arch)
        for arch in globalParameters["SupportedISA"]
        if globalParameters["AsmCaps"][arch]["SupportedISA"]
    ]

    newLibraryDir = Path(outputPath) / "library"
    newLibraryDir.mkdir(exist_ok=True)

    masterFileList = (
        generateMasterFileList(masterLibraries, archs, lazyLoading)
        if separateArchs
        else [("TensileLibrary", fullMasterLibrary)]
    )
    for name, lib in masterFileList:
        writeMasterFile(newLibraryDir, libraryFormat, kernelMinNaming, name, lib)
    masterFile, fullMasterLibrary = masterFileList[0]

    ext = ".yaml" if libraryFormat == "yaml" else ".dat"
    if embedLibrary:
        embedFileName = Path(outputPath) / "library" / embedLibrary
        EmbeddedData.generateLibrary(
            embedFileName,
            args["EmbedLibraryKey"],
            (newLibraryDir / masterFile).with_suffix(ext),
            fullMasterLibrary.cpp_base_class,
            codeObjectFiles,
        )

    if args["BuildClient"]:
        tPrint(1, "# Building Tensile Client")
        ClientExecutable.getClientExecutable(outputPath)

    if args["ClientConfig"]:
        generateClientConfig(Path(outputPath), Path(masterFile).with_suffix(ext), codeObjectFiles)

    tPrint(1, "# Tensile Library Writer DONE")
    tPrint(1, HR)
    tPrint(1, "")
