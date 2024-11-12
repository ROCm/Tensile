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

import collections
import functools
import itertools
import os
import re
import shlex
import shutil
import subprocess
import time
import glob
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union
from itertools import chain

from . import Common, LibraryIO, Utils
from .Kernel import Name
from .Common import (
    HR,
    ArchInfo,
    CHeader,
    CMakeHeader,
    Capabilities,
    IsaVersion,
    RocmPaths,
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
from .Utilities.toFile import toFile
from .Utilities.String import splitDelimitedString
from .Utilities.RequiredParameters import getRequiredParametersMin

TENSILE_MANIFEST_FILENAME = "TensileManifest.txt"
TENSILE_LIBRARY_DIR = "library"


class ProcessedKernelResult(NamedTuple):
    error: int
    source: str
    header: str
    kernelName: str
    filename: Optional[str] = None


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

def addBuildIDForROCm53Plus():
    compilerVer = globalParameters['HipClangVersion'].split(".")[:2]
    compilerVer = [int(c) for c in compilerVer]
    if len(compilerVer) >= 2 and (compilerVer[0] > 5 or (compilerVer[0] == 5 and compilerVer[1] > 2)):
      return ["-Xlinker", "--build-id"]
    else:
        return []


def getLinkCodeObjectArgs(assembler, objectFileNames, coFileName, *moreArgs):
    rv = [assembler,
          '-target', 'amdgcn-amd-amdhsa']
    rv += addBuildIDForROCm53Plus()
    rv += moreArgs
    rv += ['-o', coFileName]

    rv += objectFileNames
    return rv


def gatherCOFilesForLinking(kernels, kernelMinNaming):
    coFileMap = collections.defaultdict(list)
    for kernel in kernels:
        coName = kernel.get("codeObjectFile", None)
        if coName:
            coFileMap[coName + ".co"].append(
                Name.getKernelFileBase(kernel, kernelMinNaming) + ".o" )# would be nice if we didn't need to compute name
    return coFileMap


def linkCodeObjectFiles(coFileMap, destDir, asmDir):
    coFiles = []
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
            args = getLinkCodeObjectArgs("amdclang++", ["@clangArgs.txt"], coFile)
        else:
            args = getLinkCodeObjectArgs("amdclang++", objectFiles, os.path.join(destDir, coFile))

        tPrint(2, "Linking objects into co files: " + " ".join(args))

        try:
            # change to use check_output to force windows cmd block util command finish            
            out = subprocess.check_output(args, stderr=subprocess.STDOUT, cwd=asmDir)
            tPrint(3, out)
        except subprocess.CalledProcessError as err:
            print("Failed to link in linkCodeObjectFiles:")
            print("Subprocess command is: out = subprocess.check_output(args, stderr=subprocess.STDOUT, cwd=asmDir)")
            print("Subprocess input is: \n\nargs:", ' '.join(args))
            print("asmDir: ", asmDir)
            print("See the following error output: \n\n", err.output)
            raise
        coFiles.append(coFile)
    return coFiles


def getAssemblyCodeObjectFiles(coFileMap, outputPath):
    asmDir = Path("build_tmp") / Path(outputPath).stem.upper() / "assembly"
    asmDir.mkdir(parents=True, exist_ok=True)

    destDir = Path(outputPath) / "library"
    destDir.mkdir(parents=True, exist_ok=True)

    return linkCodeObjectFiles(coFileMap, destDir, asmDir)


def splitArchs(caps: Capabilities, archInfo: ArchInfo):
    # Helper for architecture
    def isSupported(arch):
        return caps.Asm[arch]["SupportedISA"] and caps.Asm[arch]["SupportedSource"]

    archs = []
    cmdlineArchs = []

    if "all" in archInfo.Archs:
        for arch in archInfo.SupportedIsas:
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
        for arch in archInfo.Archs:
            archs += [re.sub(":", "-", arch)]
            cmdlineArchs += [arch]
    return archs, cmdlineArchs


def buildSourceKernelObjectFile(
    CxxCompiler, outputPath, caps: Capabilities, rocmPaths: RocmPaths, archInfo: ArchInfo, kernelFile
):
    buildPath = Path("build_tmp") / Path(outputPath).stem.upper() / "code_object_tmp"
    buildPath.mkdir(parents=True, exist_ok=True)

    (_, filename) = os.path.split(kernelFile)
    (base, _) = os.path.splitext(filename)

    if rocmPaths.CmakeCxxCompiler is not None:
        os.environ["CMAKE_CXX_COMPILER"] = rocmPaths.CmakeCxxCompiler

    objectFilename = base + ".o"

    if supportedCompiler(CxxCompiler):
        _, cmdlineArchs = splitArchs(caps, archInfo)

        archFlags = ["--offload-arch=" + arch for arch in cmdlineArchs]

        # needs to be fixed when Maneesh's change is made available
        hipFlags = ["-D__HIP_HCC_COMPAT_MODE__=1"]
        hipFlags += (
            ["--genco"] if CxxCompiler == "hipcc" else ["--cuda-device-only", "-x", "hip", "-O3"]
        )
        hipFlags += ["-I", outputPath]

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
                + [which(CxxCompiler, rocmPaths.Bin)]
                + hipFlags
                + archFlags
                + [kernelFile, "-c", "-o", os.path.join(buildPath, objectFilename)]
            )
        else:
            compileArgs = (
                launcher
                + [which(CxxCompiler, rocmPaths.Bin)]
                + hipFlags
                + archFlags
                + [kernelFile, "-c", "-o", os.path.join(buildPath, objectFilename)]
            )

        tPrint(0, f"Build object file command: {compileArgs}")
        # change to use  check_output to force windows cmd block util command finish
        try:
            out = subprocess.check_output(compileArgs, stderr=subprocess.STDOUT)
            tPrint(3, out)
        except subprocess.CalledProcessError as err:
            print("Failed to build source kernel object file in buildSourceKernelObjectFile:")
            print("Subprocess command is: out = subprocess.check_output(compileArgs, stderr=subprocess.STDOUT)")
            print("Subprocess input is: \n\nargs:", " ".join(compileArgs))
            print("See the following error output: \n\n", err.output)

            with open(kernelFile, "r") as f, open("test.cpp", "w") as o:
                o.write(f.read())
            print(err.output)
            raise
    else:
        raise RuntimeError("Unknown compiler {}".format(CxxCompiler))
    
    return base


def makeHsaCOFilePath(target, base, buildPath):
    matched = re.search("gfx.*$", target)
    outfile = None
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
            outfile = os.path.join(
                buildPath, "{0}-000-{1}.hsaco".format(base + ".so", arch)
            )
    return outfile


def setInOutFlags():
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
    return inflag, outflag
    

def buildSourceKernelCodeObjectFile(
    CxxCompiler, outputPath, base, caps: Capabilities, rocmPaths: RocmPaths, archInfo: ArchInfo, removeTemporaries
):
    buildPath = Path(outputPath).parent / Path("build_tmp") / Path(outputPath).stem.upper() / "code_object_tmp"
    destDir = Path(outputPath) / "library"

    buildPath.mkdir(parents=True, exist_ok=True)
    destDir.mkdir(parents=True, exist_ok=True)
    archs, cmdlineArchs = splitArchs(caps, archInfo)
    inflag, outflag = setInOutFlags()
    infile = os.path.join(buildPath, base + ".o")
    bundler = rocmPaths.Bundler
    coFilenames = []
    try:
        bundlerArgs = [bundler, "-type=o", f"{inflag}={infile}", "-list"]
        listing = (
            subprocess.check_output(bundlerArgs, stderr=subprocess.STDOUT).decode().split("\n")
        )
        for target in listing:
            outfile = makeHsaCOFilePath(target, base, buildPath)
            if outfile:
                coFilenames.append(os.path.split(outfile)[1])
                bundlerArgs = [bundler, "-type=o", f"-targets={target}", f"{inflag}={infile}", f"{outflag}={outfile}", "-unbundle"]
                tPrint(2, "Build source code object file: " + " ".join(bundlerArgs))
                out = subprocess.check_output(bundlerArgs, stderr=subprocess.STDOUT)
                tPrint(3, out)
    except subprocess.CalledProcessError as err:
        print("Failed to unbundle in buildSourceKernelCodeObjectFile:")
        print("Subprocess comand is: out = subprocess.check_output(bundlerArgs, stderr=subprocess.STDOUT)")
        print("Subprocess input is: \nargs:", bundlerArgs)
        print("See the following error output: \n\n", err.output)
        print("Attempting to recover...")
        note = " ".join([f"-targets=hip-amdgcn-amd-amdhsa--{arch}" for arch in cmdlineArchs])
        print(f"Using {note}")

        for i in range(len(archs)):
            outfile = os.path.join(buildPath, "{0}-000-{1}.hsaco".format(base + ".so", archs[i]))
            coFilenames.append(os.path.split(outfile)[1])
            bundlerArgs = [bundler, "-type=o", f"-targets=hip-amdgcn-amd-amdhsa--{cmdlineArchs[i]}"]
            bundlerArgs += [f"{inflag}={infile}", f"{outflag}={outfile}", "-unbundle"]
            tPrint(2, "Build source code object file: " + " ".join(bundlerArgs))
            try:
                out = subprocess.check_output(bundlerArgs, stderr=subprocess.STDOUT)
                tPrint(3, out)
            except subprocess.CalledProcessError as err:
                tPrint(1, err.output)
                raise


    return coFilenames


def buildSourceKernelCodeObjectFiles(CxxCompiler, baseNames, outputPath, caps, rocmPaths, archInfo, removeTemporaries, cores):
    args = zip(
        itertools.repeat(CxxCompiler),
        itertools.repeat(outputPath),
        baseNames,
        itertools.repeat(caps),
        itertools.repeat(rocmPaths),
        itertools.repeat(archInfo),                        
        itertools.repeat(removeTemporaries),
    )
    return Common.ParallelMap(buildSourceKernelCodeObjectFile, args, cores, "Compiling source kernels")


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


ProcessedKernelLookup = Dict[Union[Path, str], List[ProcessedKernelResult]]


def collectFilesToWrite(
    results: List[ProcessedKernelResult],
    outputPath: Path,
    lazyLoading: bool,
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
        - A dictionary mapping file paths to kernel data:
            (error status, source code, header code, kernel name).
    """
    pathJoin = lambda x: os.path.join(os.path.normcase(outputPath), x)

    srcKernelMap = collections.defaultdict(list)
    validKernelCount = 0

    for err, src, header, kernelName, filename in results:
        if not src.strip():
            continue

        kernPath = pathJoin(kernelName)
        if filename:
            kernPath = pathJoin(filename)
        else:
            suffix = str(validKernelCount % numMergedFiles) if numMergedFiles > 1 else ""
            kernPath = pathJoin(f"Kernels{suffix}")

        srcKernelMap[kernPath].append(ProcessedKernelResult(err, src, header, kernelName))
        validKernelCount += 1

    # Ensure there's at least one kernel file for helper kernels
    if lazyLoading or (validKernelCount == 0):
        kernelSuffix = "0" if numMergedFiles > 1 else ""
        srcKernelMap[pathJoin(f"Kernels{kernelSuffix}")] = []

    return srcKernelMap 


def generateKernelSourceAndHeaderFiles(
    srcCodeMap: ProcessedKernelLookup,
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
    filenames = set()
    for filename, kernelList in srcCodeMap.items():
        filenames.add(filename)
        # fmt: off
        with open(f"{filename}.h", "w", encoding="utf-8") as hdrFile, \
                open(f"{filename}.cpp", "w", encoding="utf-8") as srcFile:
        # fmt: on
            writeHeaderPreface(hdrFile)
            writeSourcePreface(srcFile, filename)
            for _, src, header, _ in kernelList:
                srcFile.write(src)
                hdrFile.write(header)

    return [filePrefix + ".cpp" for filePrefix in filenames]


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
    for kernIdx, res in enumerate(results):
        (err, src, header, kernelName, filename) = res
        if err !=0 :
            print(kernelName, err)
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
    err, src = ko.getSourceFileString()
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
    if kernelSourceFile and kernelHeaderFile:
        toFile(kernelSourceFile, srcCode)
        toFile(kernelHeaderFile, hdrCode)


################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
def writeSourceKernels(
    outputPath: str,
    params: Dict[str, Any],
    kernels: List[Solution],
    kernelWriterSource: KernelWriterSource,
):
    start = time.time()
    outPath = Path(outputPath)
    results = [processKernelSource(k, kernelWriterSource) for k in kernels]
    srcKernelMap= collectFilesToWrite(results, outPath, params["LazyLibraryLoading"], numMergedFiles=1)
    stop = time.time()
    total = stop - start
    numKernels = len(srcKernelMap)    
    tPrint(1, f"# Source kernel Writing elapsed time = {total} secs, kps = {float(numKernels)/total}, kernels = {numKernels}")

    return srcKernelMap 


# should be writeAssemblyKernelsAndBuildCodeObjects
# or we should separate building the code objects from
# writing asm kernels.
def writeAssemblyKernels(
    outputPath: str,
    kernels: List[Solution],
    kernelWriterAssembly: KernelWriterAssembly,
    errorTolerant: bool = False,
):
    start = time.time()

    kernels = markDuplicateKernels(kernels, kernelWriterAssembly)  
    results = [processKernelSource(k, kernelWriterAssembly) for k in kernels] # the generates both .o and .co files for asm kernels (unlike source kernesl)

    filterProcessingErrors(kernels, results, errorTolerant)
    kernelsWithBuildErrors = {kernelName: err for err, _, _, kernelName, _ in results if err}
    kernelsToBuild = filterBuildErrors(
        kernels, kernelsWithBuildErrors, kernelWriterAssembly, errorTolerant
    )

    stop = time.time()
    total = stop-start
    numKernels = len(kernelsToBuild)
    tPrint(1, f"# Assembly kernel Building elapsed time = {total} secs, kps = {float(numKernels)/total}, kernels = {numKernels}")    

    return kernelsToBuild # we realy just need the .co files on disk


##############################################################################
# Min Naming / Solution and Kernel Writers
##############################################################################
def getKernelWriters(
    kernels: List[Solution], removeTemporaries, rocmPaths, capabilities, archInfo, assemblyDirectory, kernelMinNaming
):

    # if any kernels are assembly, append every ISA supported
    kernelSerialNaming = Solution.getSerialNaming(kernels)
    kernelWriterSource = KernelWriterSource(
        getRequiredParametersMin(), kernelSerialNaming, capabilities, archInfo, rocmPaths, removeTemporaries
    )
    kernelWriterAssembly = KernelWriterAssembly(
        getRequiredParametersMin(),
        kernelSerialNaming,
        rocmPaths.Assembler,
        capabilities,
        archInfo,
        assemblyDirectory,
        removeTemporaries,
    )

    return kernelWriterSource, kernelWriterAssembly


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


def updateMasterLibrary(
    gfxName: str,
    masterLib: MasterSolutionLibrary, 
    masterLibraries: Dict[str, MasterSolutionLibrary], 
    nextIdx: Dict[str, int], 
) -> None:
        tPrint(0, f"gfxNames in MSL: {masterLibraries.keys()}")
        if gfxName in masterLibraries:
            nextIdx[gfxName] = masterLibraries[gfxName].merge(masterLib, nextIdx[gfxName])
        else:
            nextIdx[gfxName] = addNewLibrary(masterLibraries, masterLib, gfxName)


def addFallbacksToMasterLibraries(masterLibraries: Dict[str, MasterSolutionLibrary], caps, archInfo) -> None:
    """Adds fallback library.

    Given a master solution library, add a fallback and if the corresponding
    architecture is unsupported, replace the library altogether with a fallback.

    Args:
        masterLibraries: A dictionary containing the master solution libraries.
    """
    archs, _ = splitArchs(caps, archInfo)

    for arch, msl in masterLibraries.items():
        if arch != "fallback":
            msl.insert(masterLibraries["fallback"])

    for archName in archs:
        archName = archName.split("-", 1)[0]
        if archName not in masterLibraries:
            tPrint(1, "Using fallback for arch: " + archName)
            masterLibraries[archName] = masterLibraries["fallback"]

    masterLibraries.pop("fallback")


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


def parseLibraryLogicFiles(
    logicFiles: List[str], caps: Capabilities
) -> List[LibraryIO.LibraryLogic]:
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
    #print(logicFiles)
    #for d in logicFiles:
    #    print(d)
    files = glob.glob(logicFiles + "/*.yaml")
    print(files)
    for f in files:
        print(f)
        yamlDict = LibraryIO.readYAML(f)
        logic = LibraryIO.parseLibraryLogicData(yamlDict, caps)
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


def findLogicFiles(
    path: Path,
    logicArchs: Set[str],
    extraMatchers: Set[str] = {"hip"},
    experimentalDir: str = "experimental",
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
    isNotExperimental = lambda path: not experimentalDir in str(path)
    extensions = ["*.yaml", "*.yml"]
    logicFiles = filter(isMatch, (file for ext in extensions for file in path.rglob(ext)))
    logicFiles = filter(isNotExperimental, logicFiles)
    
    return set(str(l.parent) for l in logicFiles)


# @ray.remote
def run(
    removeTemporaries,
    outputPath,
    args,
    capabilities: Capabilities,
    rocmPaths: RocmPaths,
    archInfo: ArchInfo,
    kernelMinNaming,
    logicFiles,
):
    procnum = os.getpid()
    tPrint(0, f"processing on: {procnum}")
    libraryLogics = parseLibraryLogicFiles(logicFiles, capabilities)

    solns = list(generateSolutions(libraryLogics))
    kernels = list((s.getKernels() for s in solns))
    # tPrint(0, f"Library logic file: {logicFiles}, kernels: {kernels}")

    kernelHelperObjs = generateKernelObjectsFromSolutions(kernels)
    asmDir = Path(os.path.join(Path(outputPath).parent, "build_tmp", Path(outputPath).stem.upper(), "assembly"))
    asmDir.mkdir(parents=True, exist_ok=True)
    kernelWriterSource, kernelWriterAssembly = getKernelWriters(
        kernels, removeTemporaries, rocmPaths, capabilities, archInfo, str(asmDir), kernelMinNaming
    )

    #srcKernels = [k for k in kernels if k["KernelLanguage"] == "Source"]
    #filesToWrite = writeSourceKernels(
    #  outputPath,
    #  args,
    #  srcKernels,
    #  kernelWriterSource,
    #)

    asmKernels = [k for k in kernels if k["KernelLanguage"] == "Assembly"]
    asmKernels = writeAssemblyKernels(
        outputPath,
        asmKernels,
        kernelWriterAssembly,
    )

    coFileMap = gatherCOFilesForLinking(asmKernels, kernelMinNaming)

    return getAssemblyCodeObjectFiles(
        coFileMap,
        outputPath,
    )


# # Relevant to source kernel compilation
# def relocateSourceKernelCodeObjectFiles(coFilenames, outputPath, removeTemporaries):
#     buildPath = Path(outputPath).parent / Path("build_tmp") / Path(outputPath).stem.upper() / "code_object_tmp"
#     destDir = Path(outputPath) / "library"
#     extractedCOs = [os.path.join(buildPath, name) for name in set(coFilenames)]
#     destCOsList = [os.path.join(destDir, name) for name in set(coFilenames)]
#     for src, dst in zip(extractedCOs, destCOsList):
#         if removeTemporaries:
#             shutil.move(src, dst)
#         else:
#             shutil.copyfile(src, dst)


@profile
def TensileCreateLibrary():

    args = parseArguments()

    cxxCompiler = args["CxxCompiler"]
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

    # Below code can be removed after deepcopy removal is complete
    archInfo, capabilities, rocmPaths = assignGlobalParameters(args, capabilitiesCache)

    if capabilities.AsmIsCached:
        LibraryIO.writeAsmCapsCache(cacheFile, capabilities.Asm)

    if not os.path.exists(logicPath):
        printExit("LogicPath %s doesn't exist" % logicPath)

    # converts logicArchs from gfx to common name, e.g., aldebaran, aquavanjaram
    logicArchs: Set[str] = {name for name in (getArchitectureName(gfxName) for gfxName in archInfo.Archs) if name}

    kernelMinNaming = getRequiredParametersMin()

    parallelFunc = functools.partial(run, removeTemporaries, outputPath, args, capabilities, rocmPaths, archInfo, kernelMinNaming)

    logicFiles = list(findLogicFiles(Path(logicPath), logicArchs))

    total = len(logicFiles)
    chunk_size = int(total / numPasses)
    remainder = total % numPasses    

    coFiles = []
    for p in range(0, numPasses):
        start = p * chunk_size
        stop = (p+1) * chunk_size
        rvs = Common.ParallelMap(parallelFunc, logicFiles[start:stop], cpuThreads, "Running TCL...", multiArg=False)
        for r in rvs:
            coFiles.extend(r)
    if remainder:
        rvs = Common.ParallelMap(parallelFunc, logicFiles[-remainder:], cpuThreads, "Running TCL...", multiArg=False)
        for r in rvs:
            coFiles.extend(r)

    # # Relevant to source kernel compilation
    #for coFileMap_, files, kho in rvs:
    #    filesToWrite.append(files)
    #    kernelHelperObjs.extend(kho)
    #kernelFiles = generateKernelSourceAndHeaderFiles(filesToWrite)
    #with KernelFileContextManager(True, True, 1, outputPath, kernelFiles) as (srcFile, hdrFile):
    #    for ko in kernelHelperObjs:
    #        writeKernelHelpers(ko, srcFile, hdrFile, outputPath, kernelFiles)
    #parallelFunc = functools.partial(buildSourceKernelObjectFile, cxxCompiler, outputPath, capabilities, rocmPaths, archInfo)
    #basenames = Common.ParallelMap(parallelFunc, kernelFiles, cpuThreads, "Building Source Kernel Code objects", multiArg=False)
    #results = buildSourceKernelCodeObjectFiles(cxxCompiler, basenames, outputPath, capabilities, rocmPaths, archInfo, removeTemporaries, cpuThreads)
    #coFilenames = []
    #for result in results:
    #     coFilenames.extend(result)
    #relocateSourceKernelCodeObjectFiles(coFilenames, outputPath, removeTemporaries)

    # # MSL creation
    # newLibraryDir = Path(outputPath) / "library"
    # newLibraryDir.mkdir(exist_ok=True)
    # baseName = "TensileLibrary_"
    # if "fallback" in masterLibraries:
    #     addFallbacksToMasterLibraries(masterLibraries, capabilities, archInfo)
    # for arch, masterLib in masterLibraries.items():
    #     for name, lib in [(baseName + "lazy_" + arch, masterLib)] + list(masterLib.lazyLibraries.items()):
    #         tPrint(1, f"Writing MSLibrary: {name}")
    #         lib.applyNaming(getRequiredParametersMin())  # <-- This should be able to be replaced directly with `name`?
    #         LibraryIO.write(str(newLibraryDir / name), Utils.state(lib), args["LibraryFormat"])

    if removeTemporaries:
        buildTmp = Path(outputPath).parent / "build_tmp"
        if buildTmp.exists() and buildTmp.is_dir():
            shutil.rmtree(buildTmp)



