################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
import gc
import collections
import functools
import itertools
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

from joblib import Parallel

#from viztracer import log_sparse

from . import Common, LibraryIO, Utils
from .Common import (
    HR,
    CHeader,
    CMakeHeader,
    assignGlobalParameters,
    ensurePath,
    getArchitectureName,
    gfxName,
    globalParameters,
    printExit,
    printWarning,
    supportedCompiler,
    tPrint,
    which,
)
from .KernelWriterAssembly import KernelWriterAssembly
from .KernelWriterBase import KernelWriterBase
from .KernelWriterSource import KernelWriterSource
from .SolutionLibrary import MasterSolutionLibrary
from .SolutionStructs import Solution
from .TensileCreateLib.KernelFileContext import KernelFileContextManager
from .TensileCreateLib.ParseArguments import parseArguments
from .Utilities.Profile import profile
from .Utilities.String import splitDelimitedString
from .Utilities.toFile import toFile

TENSILE_MANIFEST_FILENAME = "TensileManifest.txt"
TENSILE_LIBRARY_DIR = "library"


ProcessedKernelResult = Tuple[int, str, str, str, Optional[str]]


def processKernelSource(kernel, kernelWriter) -> ProcessedKernelResult:
    """Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    try:
        # get kernel name
        kernelName = kernelWriter.getKernelFileBase(kernel)
        (err, src) = kernelWriter.getSourceFileString(kernel)
        header = kernelWriter.getHeaderFileString(kernel)
        # will be put in Kernels.h/cpp if None
        filename = kernel.get("codeObjectFile", None)

    except RuntimeError:
        printWarning(
            "Gracefully handling unknown runtime error when generating kernel: %s"
            % kernel["KernelName"]
        )
        return (1, "", "", kernelName, None)

    return (err, src, header, kernelName, filename)


def getAssemblyCodeObjectFiles(kernels, kernelWriterAssembly, outputPath, removeTemporaries):
    destDir = ensurePath(os.path.join(outputPath, "library"))
    asmDir = kernelWriterAssembly.getAssemblyDirectory()
    
    if len(kernels) == 0:
        return []

    archs = collections.defaultdict(list)
    for k in kernels:
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

        numObjectFiles = len(kernels)

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

                tPrint(2, "Linking objects into co files: " + " ".join(args))

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


def buildSourceCodeObjectFile(CxxCompiler, outputPath, kernelFile, removeTemporaries):
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

        # needs to be fixed when Maneesh's change is made available
        hipFlags = ["-D__HIP_HCC_COMPAT_MODE__=1"]
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

        tPrint(2, f"Build object file command: {compileArgs}")
        # change to use  check_output to force windows cmd block util command finish
        try:
            out = subprocess.check_output(compileArgs, stderr=subprocess.STDOUT)
            tPrint(3, out)
        except subprocess.CalledProcessError as err:
            with open(kernelFile, "r") as f, open("test.cpp", "w") as o:
                o.write(f.read())
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
                        tPrint(2, "Build source code object file: " + " ".join(bundlerArgs))
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
                tPrint(2, "Build source code object file: " + " ".join(bundlerArgs))
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
        if removeTemporaries:
            shutil.move(src, dst)
        else:
            shutil.copyfile(src, dst)

    return destCOsList


def buildSourceCodeObjectFiles(CxxCompiler, kernelFiles, outputPath, removeTemporaries):
    args = zip(
        itertools.repeat(CxxCompiler),
        itertools.repeat(outputPath),
        kernelFiles,
        itertools.repeat(removeTemporaries),
    )

    coFiles = []
    for k in kernelFiles:
        coFile = buildSourceCodeObjectFile(CxxCompiler, outputPath, k, removeTemporaries)
        coFiles.append(coFile)

    return coFiles

    # coFiles = buildSourceCodeObjectFile, args, "Compiling source kernels")

    # return itertools.chain.from_iterable(coFiles)


################################################################################
def prepAsm(
    kernelWriterAssembly: KernelWriterAssembly,
    isLinux: bool,
    buildPath: Path,
    isa: Tuple[int, int, int],
    printLevel: int,
):
    """Create and prepare the assembly directory; called ONCE per output directory.

    This function is called once per output directory. It creates a directory
    "assembly" under the provided **buildPath**, and generates a bash script for
    compiling object files into code object files.

    Args:
        kernelWriterAssembly: Assembly writer object.
        buildPath: Path to directory where assembly files will be written.
    """
    asmPath = buildPath / "assembly"
    asmPath.mkdir(exist_ok=True)

    assemblerFileName = asmPath / f"asm-new.{'sh' if isLinux else 'bat'}"

    with open(assemblerFileName, "w") as assemblerFile:
        if isLinux:
            assemblerFile.write("#!/bin/sh {log}\n".format(log="-x" if printLevel >= 3 else ""))
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
        else:
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
    os.chmod(assemblerFileName, 0o777)


ProcessedKernelLookup = Dict[str, Any]


def collectFilesToWrite(
    results: List[ProcessedKernelResult],
    outputPath: Path,
    lazyLoading: bool,
    mergeFiles: bool,
    numMergedFiles: int,
) -> ProcessedKernelLookup:
    """Collects and organizes kernel files to be written based on the provided results.

    Args:
        results: A list of processed kernel results, each containing
            error status, source code, header, kernel name, and filename.
        outputPath: The path where the output files should be written.
        lazyLoading: Flag indicating whether lazy loading is enabled.
        mergeFiles: Flag indicating whether kernel files should be merged.
        numMergedFiles: The number of files to merge into if merging is enabled.

    Returns:
        A tuple containing:
            - A dictionary mapping file paths to kernel data:
                (error status, source code, header code, kernel name).
            - A dictionary mapping kernel names to an error status.
    """
    pathJoin = lambda x: os.path.join(os.path.normcase(outputPath), x)

    filesToWrite = collections.defaultdict(list)
    validKernelCount = 0

    for err, src, header, kernelName, filename in results:
        if not src.strip():
            continue

        kernPath = pathJoin(kernelName)
        if filename:
            kernPath = pathJoin(filename)
        elif mergeFiles:
            suffix = str(validKernelCount % numMergedFiles) if numMergedFiles > 1 else ""
            kernPath = pathJoin(f"Kernels{suffix}")

        filesToWrite[kernPath].append((err, src, header, kernelName))
        validKernelCount += 1

    # Ensure there's at least one kernel file for helper kernels
    if lazyLoading or (mergeFiles and validKernelCount == 0):
        kernelSuffix = "0" if numMergedFiles > 1 else ""
        filesToWrite[pathJoin(f"Kernels{kernelSuffix}")] = []

    return filesToWrite


def generateKernelSourceAndHeaderFiles(
    filesToWrite: ProcessedKernelLookup,
) -> List[str]:
    """Generates kernel source and header files.

    Arguments:
        fileToWrite: A dictionary mapping file paths to kernel data:
            (error status, source code, header code, kernel name).

    Returns:
        A list containing source kernel filenames, and a dictionary with kernels that
            encountered build errors.
    """

    def writeHeaderPreface(hdrFile):
        hdrFile.write(CHeader)
        hdrFile.write("#pragma once\n")
        if globalParameters["RuntimeLanguage"] == "HIP":
            hdrFile.write("#include <hip/hip_runtime.h>\n")
            hdrFile.write("#include <hip/hip_ext.h>\n\n")
        hdrFile.write('#include "KernelHeader.h"\n\n')

    def writeSourcePreface(srcFile, filename):
        srcFile.write(CHeader)
        srcFile.write(f'#include "{filename}.h"\n')

    # Write kernel data to files
    for filename, kernelList in filesToWrite.items():
        # fmt: off
        with open(f"{filename}.h", "w", encoding="utf-8") as hdrFile, \
             open(f"{filename}.cpp", "w", encoding="utf-8") as srcFile:
        # fmt: on
            writeHeaderPreface(hdrFile)
            writeSourcePreface(srcFile, filename)
            for _, src, header, _ in kernelList:
                srcFile.write(src)
                hdrFile.write(header)

    return [filePrefix + ".cpp" for filePrefix in filesToWrite]


def markDuplicateKernels(
    kernels: List[Solution], kernelWriter
) -> List[Solution]:
    """Marks duplicate assembly kernels based on their generated base file names.

    Kernels written in Assembly language may generate duplicate output file names,
    leading to potential race conditions. This function identifies such duplicates within
    the provided list of Solution objects and marks them to prevent issues.

    Args:
        kernels: A list of Solution objects representing kernels to be processed.

    Returns:
        A modified list of Solution objects where kernels identified as duplicates
        are marked with a `duplicate` attribute indicating their duplication status.

    Notes:
        This function sets the "duplicate" attribute on Solution objects, and thereby prepares
        kernels for **processKernelSource**, which requires "duplicate" to be set.
    """
    # Kernels may be intended for different .co files, but generate the same .o file
    # Mark duplicate kernels to avoid race condition
    # @TODO improve organization so this problem doesn't appear
    visited = set()
    count = 0
    for kernel in kernels:
        #if kernel["KernelLanguage"] == "Assembly":
        curr = kernelWriter.getKernelFileBase(kernel)
        kernel.duplicate = curr in visited
        count += kernel.duplicate
        visited.add(curr)
    if count:
        printWarning(f"Found {count} duplicate kernels, these will be ignored")
    return kernels


def filterProcessingErrors(
    kernels: List[Solution],
    results: List[ProcessedKernelResult],
    errorTolerant: bool,
):
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
    removeKernels = []
    removeResults = []
    for kernIdx, res in (
        enumerate(results)
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
            removeResults.append(results[kernIdx])
    if len(removeKernels) > 0 and not errorTolerant:
        printExit("** kernel generation failure **")
    for kern in removeKernels:
        kernels.remove(kern)
    for rel in removeResults:
        results.remove(rel)


def filterBuildErrors(
    kernels: List[Solution],
    kernelsWithBuildErrors: Dict[str, int],
    kernelWriter: Any,
    ignoreErr: bool,
) -> List[Solution]:
    """Filters a list of kernels based on build errors and error tolerance.

    Args:
        kernels: A list of `Solution` objects representing kernels to filter.
        kernelsWithBuildErrors: A list of `Solution` objects that have build errors.
        errorTolerant: A boolean indicating whether to tolerate build errors.

    Returns:
        A filtered list of kernels (**Solution** objects) that are eligible for building.

    Raises:
        SystemExit: If **ignoreErr** is False and any kernels have build errors.
    """
    if not ignoreErr and len(kernelsWithBuildErrors) > 0:
        raise RuntimeError(
            "Kernel compilation failed in one or more subprocesses. "
            "Consider setting CpuThreads=0 and re-run to debug."
        )

    def noBuildError(kernel):
        kernelName = kernelWriter.getKernelName(kernel)
        return kernelName not in kernelsWithBuildErrors

    return list(filter(noBuildError, kernels))


def getKernelSourceAndHeaderCode(ko: KernelWriterBase) -> Tuple[int, List[str], List[str], str]:
    """Get the source and header content for a kernel object.

    Arguments:
        ko: Kernel object to extract content from.

    Returns:
        Tuple of data: (error code, source code, header code, kernel name)
    """
    name = ko.getKernelName()
    err, src = ko.getSourceFileString() # why are we doing this twice?
    hdr = ko.getHeaderFileString()
    return err, [CHeader, src], [CHeader, hdr], name


def writeKernelHelpers(
    kernelHelperObj: KernelWriterBase,
    kernelSourceFile: Optional[TextIOWrapper],
    kernelHeaderFile: Optional[TextIOWrapper],
    outputPath: Path,
    kernelFiles: List[str],
):
    """Writes the source and header code generated by a kernel helper object to specified files or a new file.

    Args:
        kernelHelperObj: The kernel helper object providing source and header code.
        kernelSourceFile: The file object for the kernel's source. If None, a new file is created.
        kernelHeaderFile: The file object for the kernel's header. If None, a new file is created.
        outputPath: The directory path where new files should be saved if `kernelSourceFile` and
            `kernelHeaderFile` are None.
        kernelFiles: A list of kernel file names to be updated with the new kernel name if new
            files are created.

    Notes:
        - If `kernelSourceFile` and `kernelHeaderFile` are provided, the source and header code
          are appended to these files.
        - If these file objects are None, new `.cpp` and `.h` files are created in the
          `outputPath/Kernels` directory named after the kernel.
        - The function appends the new kernel name to `kernelFiles` if new files are created.
    """
    err, srcCode, hdrCode, kernelName = getKernelSourceAndHeaderCode(kernelHelperObj)
    if err:
        printWarning(f"Invalid kernel: {kernelName} may be corrupt")
    if kernelSourceFile and kernelHeaderFile:  # Append to existing files => mergeFiles == True
        toFile(kernelSourceFile, srcCode)
        toFile(kernelHeaderFile, hdrCode)
    else:  # Write to new a file for each helper => mergeFiles == False. Default behaviour when called through rocBLAS
        srcFilename = Path(outputPath) / "Kernels" / f"{kernelName}.cpp"
        hdrFilename = Path(outputPath) / "Kernels" / f"{kernelName}.h"
        toFile(srcFilename, srcCode)
        toFile(hdrFilename, hdrCode)
        kernelFiles.append(str(srcFilename))


################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
def writeSourceKernels(
    outputPath: str,
    cxxCompiler: str,
    params: Dict[str, Any],
    kernels: List[Solution],
    kernelHelperObjs: List[KernelWriterBase],
    kernelWriterSource: KernelWriterSource,
    errorTolerant: bool = False,
    removeTemporaries: bool = True,
):
    start = time.time()

    # Push working path into build_tmp folder because there may be more than
    # one process running this script. This is to avoid build directory clashing.
    # NOTE: file paths must not contain the lower case word 'kernel' or the
    # /opt/rocm/bin/extractkernel will fail.
    # See buildSourceCodeObjectFile:167 for the call to this binary.

    ## TODO: Is there a way to get this to work without changing global state?
    Common.pushWorkingPath("build_tmp")
    Common.pushWorkingPath(os.path.basename(outputPath).upper())

    tPrint(1, "# Writing Kernels...")


    results = [processKernelSource(k, kernelWriterSource) for k in kernels]

    filesToWrite = collectFilesToWrite(
        results,
        Path(outputPath),
        params["LazyLibraryLoading"],
        params["MergeFiles"],
        params["NumMergedFiles"],
    )

    kernelFiles = generateKernelSourceAndHeaderFiles(filesToWrite)

    outPath = Path(outputPath)
    with KernelFileContextManager(
        params["LazyLibraryLoading"],
        params["MergeFiles"],
        params["NumMergedFiles"],
        outPath,
        kernelFiles,
    ) as (srcFile, hdrFile):
        for ko in kernelHelperObjs:
            writeKernelHelpers(ko, srcFile, hdrFile, outPath, kernelFiles)

    codeObjectFiles = []
    #if not globalParameters["GenerateSourcesAndExit"]:
    #    codeObjectFiles += buildSourceCodeObjectFiles(
    #        cxxCompiler, kernelFiles, outputPath, removeTemporaries
    #    )
     
    stop = time.time()
    tPrint(1, "# Kernel Building elapsed time = %.1f secs" % (stop - start))

    Common.popWorkingPath()  # outputPath.upper()
    Common.popWorkingPath()  # build_tmp


def writeAssemblyKernels(
    outputPath: str,
    cxxCompiler: str,
    params: Dict[str, Any],
    kernels: List[Solution],
    kernelHelperObjs: List[KernelWriterBase],
    kernelWriterAssembly: KernelWriterAssembly,
    errorTolerant: bool = False,
    removeTemporaries: bool = True,
):
    start = time.time()
    Common.pushWorkingPath("build_tmp")
    Common.pushWorkingPath(os.path.basename(outputPath).upper())

    ## TODO: This may be unused
    if not params["MergeFiles"] or params["NumMergedFiles"] > 1 or params["LazyLibraryLoading"]:
        ensurePath(os.path.join(outputPath, "Kernels"))

    ## This uses global state from "WorkingPath"
    prepAsm(
        kernelWriterAssembly,
        os.name != "nt",
        # Use globalParameters here, not params
        Path(globalParameters["WorkingPath"]),
        globalParameters["CurrentISA"],
        params["PrintLevel"],
    )

    kernels = markDuplicateKernels(kernels, kernelWriterAssembly)  
    results = [processKernelSource(k, kernelWriterAssembly) for k in kernels]

    filterProcessingErrors(kernels, results, errorTolerant)
    kernelsWithBuildErrors = {kernelName: err for err, _, _, kernelName, _ in results if err}
    kernelsToBuild = filterBuildErrors(
        kernels, kernelsWithBuildErrors, kernelWriterAssembly, errorTolerant
    )

    if not globalParameters["GenerateSourcesAndExit"]:
        codeObjectFiles = getAssemblyCodeObjectFiles(
            kernelsToBuild,
            kernelWriterAssembly,
            outputPath,
            removeTemporaries,
        )

    stop = time.time()
    tPrint(1, "# Kernel Building elapsed time = %.1f secs" % (stop - start))

    Common.popWorkingPath()  # outputPath.upper()
    Common.popWorkingPath()  # build_tmp





##############################################################################
# Min Naming / Solution and Kernel Writers
##############################################################################
def getKernelWriters(kernels: List[Solution], removeTemporaries):

    # if any kernels are assembly, append every ISA supported
    kernelSerialNaming = Solution.getSerialNaming(kernels)

    kernelMinNaming = Solution.getMinNaming(kernels)
    kernelWriterSource = KernelWriterSource(kernelMinNaming, kernelSerialNaming, removeTemporaries)
    kernelWriterAssembly = KernelWriterAssembly(
        kernelMinNaming, kernelSerialNaming, removeTemporaries
    )

    return (kernelWriterSource, kernelWriterAssembly)


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


################################################################################
# Generate Kernel Objects From Solutions
################################################################################
def generateKernelObjectsFromSolutions(kernels: List[Solution]):
    helpers = itertools.chain(k.getHelperKernelObjects() for k in kernels)
    for h in helpers:
        for ko in h:
            ko.getKernelName()
    list(dict.fromkeys(helpers))
    return helpers


def addNewLibrary(
    masterLibraries: Dict[str, MasterSolutionLibrary],
    newLibrary: MasterSolutionLibrary,
    architectureName: str,
) -> int:
    """Adds new master solution library to a master solution libraries dict.

    For a given architecture, add the new library to a dictionary containing
    libraries for all architectures, compute the starting index for the new
    library, then remap the indexes for all of the solutions associated with
    the library.

    Args:
        masterLibraries: A dictionary containing all master solution libraries for all architectures.
        newLibrary: A master solution library to add to the dictionary.
        architectureName: The name of the architecture (or key) associated with the library.

    Returns:
        Index to the last solution of the library associated with current architecture.
    """
    masterLibraries[architectureName] = newLibrary
    archIndex = MasterSolutionLibrary.ArchitectureIndexMap(architectureName)
    masterLibraries[architectureName].remapSolutionIndicesStartingFrom(archIndex)
    return archIndex


def makeMasterLibraries(
    logicList: List[LibraryIO.LibraryLogic], separate: bool
) -> Dict[str, MasterSolutionLibrary]:
    """Creates a dictionary of master solution libraries.

    Iterates through a list of LibraryLogic objects creating
    master solution libraries and modifying the solution
    indexing as required.

    Args:
        logicFiles: List of LibraryLogic objects.
        separate: Separate libraries by architecture.

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
                    newLibrary, nextSolIndex[architectureName]
                )
            else:
                nextSolIndex[architectureName] = addNewLibrary(
                    masterLibraries, newLibrary, architectureName
                )
        else:
            if fullMasterLibrary:
                fullMasterLibrary.merge(newLibrary)
            else:
                fullMasterLibrary = newLibrary

    return {"full": fullMasterLibrary} if fullMasterLibrary is not None else masterLibraries



def applyNaming(masterLibraries: Dict[str, MasterSolutionLibrary]) -> None:
    """Assigns the solution code object file name for lazy libraries.

    Given a master solution library with lazy libraries, assigns the
    key associated with the lazy library (or name) as the value
    assiciated with the corresponding solution's code object file.

    Args:
        masterLibraries: A dictionary containing the master solution libraries.
    """
    for masterLibrary in masterLibraries.values():
        for name, lib in masterLibrary.lazyLibraries.items():
            for sol in lib.solutions.values():
                sol.originalSolution["codeObjectFile"] = name


def makeSolutions(
    masterLibraries: dict, separate: bool
):  # -> Generator[Solution]:# is breaking tensile
    """Extracts the solutions from the master solution library.

    Given a master solution library, forms a flattened generator that
    yields solutions by iterating over all of the solutions contained
    in the master solution libraries. If using separate architectures
    but not using lazy loading, lazyLibraries should be an empty dict.

    Args:
        masterLibraries: A dictionary containing the master solution libraries.

    Returns:
        Generator representing a sequence of library logic tuples.
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
    """Load and parse logic (yaml) files.

    Given a list of paths to yaml files containing library logic, load the files
    into memory and parse the data into a named tuple (i.e. LibraryLogic). This
    operation is parallelized over N processes.

    Args:
        logicFiles: List of paths to logic files.

    Returns:
        List of library logic tuples.
    """
    libraryLogics = []
    for f in logicFiles:
        logic = LibraryIO.parseLibraryLogicFile(f)
        libraryLogics.append(logic)

    return libraryLogics





def generateSolutions(libraryLogics: List[LibraryIO.LibraryLogic]):
    """Generates a list of solutions.

    Args:
        masterLibraries: A dictionary of master solutions libraries.
        separate: Separate libraries by architecture.

    Returns:
        A solution list.
    """
    tmp = (l for ll in libraryLogics for l in ll.solutions)
    return list(dict.fromkeys(tmp))
 
    # return (l for ll in libraryLogics for l in ll.solutions)


def findLogicFiles(
    path: Path,
    logicArchs: Set[str],
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

    return list(str(l) for l in logicFiles)





################################################################################
# Tensile Create Library
################################################################################
def run(removeTemporaries, outputPath, cxxCompiler, args, logicFiles):

    print("processing on: ", os.getpid())

    libraryLogics = parseLibraryLogicFiles(logicFiles)
    solns = list(generateSolutions(libraryLogics))
    kernels = list((s.getKernels() for s in solns))
    kernelHelperObjs = generateKernelObjectsFromSolutions(kernels)
    kernelWriterSource, kernelWriterAssembly = getKernelWriters(kernels, removeTemporaries)
    asmKernels = [k for k in kernels if k["KernelLanguage"] == "Assembly"]
    writeAssemblyKernels(
        outputPath,
        cxxCompiler,
        args,
        asmKernels,
        kernelHelperObjs,
        kernelWriterAssembly,
        removeTemporaries=removeTemporaries,
    )

    # srcKernels = [k for k in kernels if k["KernelLanguage"] != "Assembly"]
    # print([f for f in logicFiles if "hip" in f])
    # writeSourceKernels(
    #     outputPath,
    #     cxxCompiler,
    #     args,
    #     srcKernels,
    #     kernelHelperObjs,
    #     kernelWriterSource,
    #     removeTemporaries=removeTemporaries,
    # )    

    return kernels

# weights are assumed reverse sorted
def multifit(weights, num_bins):
    bins = [0] * num_bins
    result = [[] for _ in range(0, num_bins)]
    
    def find_min_bin(bins):
        return bins.index(min(bins))
    
    for weight in weights:
        min_bin_index = find_min_bin(bins)
        bins[min_bin_index] += weight[0]
        result[min_bin_index].append(weight[1])
    
    for bin in bins:
        print("size: ", bin)
    
    return result

@profile
def TensileCreateLibrary():

    args = parseArguments()

    lazyLoading = args["LazyLibraryLoading"]
    separateArchs = args["SeparateArchitectures"]
    mergeFiles = args["MergeFiles"]
    cxxCompiler = args["CxxCompiler"]
    libraryFormat = args["LibraryFormat"]
    logicPath = args["LogicPath"]
    outputPath = args["OutputPath"]
    removeTemporaries = not args["KeepBuildTmp"]
    numPasses = args["NumPasses"]
    cpuThreads = args["CpuThreads"]

    globalParameters["PrintLevel"] = args["PrintLevel"]

    ensurePath(outputPath)
    outputPath = os.path.abspath(outputPath)
    copyStaticFiles(outputPath)

    cacheFile = Path(outputPath).parent / "asm-cache.yaml"
    capabilitiesCache = LibraryIO.initAsmCapsCache(cacheFile)

    assignGlobalParameters(args, capabilitiesCache)

    if globalParameters["CacheAsmCaps"]:
        LibraryIO.writeAsmCapsCache(cacheFile, globalParameters["AsmCaps"])

    if not os.path.exists(logicPath):
        printExit("LogicPath %s doesn't exist" % logicPath)

    logicArchs = splitDelimitedString(args["Architecture"], {";", "_"})
    logicArchs = {name for name in (getArchitectureName(gfxName) for gfxName in logicArchs) if name}
    logicFiles = sorted([(os.path.getsize(f), f) for f in findLogicFiles(Path(logicPath), logicArchs)], reverse=True)
    batchedLogicFiles = multifit(logicFiles, numPasses*cpuThreads)

    print(len(logicFiles))
    print(len(batchedLogicFiles))

    parallelFunc = functools.partial(run, removeTemporaries, outputPath, cxxCompiler, args)

    for i in range(0, numPasses):
        print("pass ", i)
        start = cpuThreads * i
        stop = cpuThreads * (i+1)
        results = Common.ParallelMap(parallelFunc, batchedLogicFiles[start:stop], cpuThreads / numPasses, "Running TCL...", multiArg=False)

        for result in results:
            print(type(result))
        del results
        
    newLibraryDir = Path(outputPath) / "library"
    newLibraryDir.mkdir(exist_ok=True)

    if removeTemporaries:
        buildTmp = Path(outputPath).parent / "build_tmp"
        if buildTmp.exists() and buildTmp.is_dir():
            shutil.rmtree(buildTmp)
