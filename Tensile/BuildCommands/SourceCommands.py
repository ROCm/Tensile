import itertools
import os
import re
import shlex
import subprocess

from typing import Union, List
from pathlib import Path

from ..Common import (
    globalParameters,
    tPrint,
    ensurePath,
    supportedCompiler,
    ParallelMap,
    splitArchs,
    which,
    ParallelMap,
    IsaVersion,
)
from .SharedCommands import compressCodeObject


def _compileSourceObjectFile(
    cmdlineArchs: List[str],
    cxxCompiler: str,
    cxxSrcPath: str,
    objDestPath: str,
    outputPath: str,
    compilerVer: IsaVersion,
):
    """
    Compiles a HIP source file (.cpp) into an object file (.o).

    Args:
        cmdlineArchs: List of architectures for offloading.
        cxxCompiler: The C++ compiler to use.
        kernelFile: The path to the kernel source file.
        buildPath: The build directory path.
        objFilename: The name of the output object file.
        outputPath: The output directory path.
        globalParameters (dict): A dictionary of global parameters.

    Raises:
        subprocess.CalledProcessError: If the compilation command fails.
    """
    archFlags = ["--offload-arch=" + arch for arch in cmdlineArchs]

    # needs to be fixed when Maneesh's change is made available
    hipFlags = ["-D__HIP_HCC_COMPAT_MODE__=1"]
    hipFlags += (
        ["--genco"] if cxxCompiler == "hipcc" else ["--cuda-device-only", "-x", "hip", "-O3"]
    )
    hipFlags += ["-I", outputPath]

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

    args = (
        launcher
        + [which(cxxCompiler)]
        + hipFlags
        + archFlags
        + [str(cxxSrcPath), "-c", "-o", objDestPath]
    )

    tPrint(2, f"Compile source object file command: {args}")
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        tPrint(2, f"Output: {out}" if out else "")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error compiling source object file: {err.output}\nFailed command: {' '.join(args)}"
        )


def _computeSourceCodeObjectPath(target: str, base: str, buildPath: Union[Path, str], arch) -> Path:
    """
    Generates the output file path based on the target, base, and build path.

    Args:
        target (str): The target architecture string.
        base (str): The base name for the output file.
        buildPath (str): The build directory path.

    Returns:
        str: The generated output file path.
    """
    coPath = None
    buildPath = Path(buildPath)
    if "TensileLibrary" in base and "fallback" in base:
        coPath = buildPath / "{0}_{1}.hsaco.raw".format(base, arch)
    elif "TensileLibrary" in base:
        variant = [t for t in ["", "xnack-", "xnack+"] if t in target][-1]
        baseVariant = base + "-" + variant if variant else base
        if arch in baseVariant:
            coPath = buildPath / (baseVariant + ".hsaco.raw")
        else:
            raise RuntimeError(
                "Failed to compute code object name:"
                f"Could not find variant {variant} in {baseVariant}"
            )
    else:
        coPath = buildPath / "{0}.so-000-{1}.hsaco.raw".format(base, arch)
    return coPath


def _listTargetTriples(bundler: str, infile: str):
    args = [bundler, "--type=o", f"--input={infile}", "-list"]
    try:
        listing = subprocess.check_output(args, stderr=subprocess.STDOUT).decode().split("\n")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error listing target triples in object files: {err.output}\nFailed command: {' '.join(args)}"
        )
    return listing


def _unbundleSourceCodeObjects(bundler: str, target: str, infile: str, outfileRaw: str):
    """
    Unbundles source code object files using the Clang Offload Bundler.

    Args:
        bundler (str): The path to the Clang Offload Bundler executable.
        target (str): The target architecture string.
        inflag (str): The input flag for the bundler.
        infile (str): The input file path.
        outflag (str): The output flag for the bundler.
        outfileRaw (str): The output raw file path.

    Raises:
        RuntimeError: If unbundling the source code object file fails.
    """
    args = [
        bundler,
        "--type=o",
        f"--targets={target}",
        f"--input={infile}",
        f"--output={outfileRaw}",
        "--unbundle",
    ]

    tPrint(2, "Unbundling source code object file: " + " ".join(args))
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        tPrint(2, f"Output: {out}" if out else "")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error unbundling source code object file: {err.output}\nFailed command: {' '.join(args)}"
        )


def _buildSourceCodeObjectFile(
    cxxCompiler: str,
    outputPath: Union[Path, str],
    kernelFile: Union[Path, str],
    removeTemporaries: bool,
):

    buildPath = Path(ensurePath(os.path.join(globalParameters["WorkingPath"], "code_object_tmp")))
    destPath = Path(ensurePath(os.path.join(outputPath, "library")))
    kernelPath = Path(kernelFile)

    objectFilename = kernelPath.stem + ".o"
    coPathsRaw = []
    coPaths = []

    if "CmakeCxxCompiler" in globalParameters and globalParameters["CmakeCxxCompiler"] is not None:
        os.environ["CMAKE_CXX_COMPILER"] = globalParameters["CmakeCxxCompiler"]

    if not supportedCompiler(cxxCompiler):
        raise RuntimeError(
            f"Cannot compiler source code object files: unknown compiler: {cxxCompiler}"
        )

    _, cmdlineArchs = splitArchs()

    # Add build-id for builds with rocm 5.3+
    compilerVer = globalParameters["HipClangVersion"].split(".")[:2]
    compilerVer = [int(c) for c in compilerVer]

    objPath = str(buildPath / objectFilename)
    _compileSourceObjectFile(
        cmdlineArchs, cxxCompiler, kernelFile, objPath, outputPath, compilerVer
    )

    bundler = globalParameters["ClangOffloadBundlerPath"]
    if not bundler:
        raise RuntimeError(
            "No bundler found; set TENSILE_ROCM_OFFLOAD_BUNDLER_PATH to point to clang-offload-bundler"
        )

    for target in _listTargetTriples(bundler, objPath):
        if match := re.search("gfx.*$", target):
            arch = re.sub(":", "-", match.group())
            coPathRaw = _computeSourceCodeObjectFilename(target, kernelPath.stem, buildPath, arch)
            _unbundleSourceCodeObjects(bundler, target, objPath, coPathRaw)

            outfile = os.path.join(destPath, coPathRaw.stem)
            compressCodeObject(coPathRaw, outfile, arch, bundler)
            coPathsRaw.append(coPathRaw)
            coPaths.append(outfile)

    if removeTemporaries:
        for file in coPathsRaw:
            os.remove(file)

    return coPathsRaw


def buildSourceCodeObjectFiles(CxxCompiler, kernelFiles, outputPath, removeTemporaries):
    args = zip(
        itertools.repeat(CxxCompiler),
        itertools.repeat(outputPath),
        kernelFiles,
        itertools.repeat(removeTemporaries),
    )
    coFiles = ParallelMap(_buildSourceCodeObjectFile, args, "Compiling source kernels")

    return itertools.chain.from_iterable(coFiles)
