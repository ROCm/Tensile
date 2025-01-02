import collections
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Union

from .. import Utils
from ..Common import ensurePath, gfxName, globalParameters, printWarning, tPrint
from ..KernelWriterAssembly import KernelWriterAssembly
from ..SolutionStructs import Solution
from .SharedCommands import compressCodeObject


def _linkIntoCodeObject(
    objFiles: List[str], coPathDest: Union[Path, str], kernelWriterAssembly: KernelWriterAssembly
):
    """Links object files into a code object file.

    Args:
        objectFiles: A list of object files to be linked.
        coPathDest: The destination path for the code object file.
        kernelWriterAssembly: An instance of KernelWriterAssembly to get link arguments.

    Raises:
        RuntimeError: If linker invocation fails.
    """
    args = []
    if os.name == "nt":
        # On Windows, the objectFiles list command line (including spaces)
        # exceeds the limit of 8191 characters, so using response file
        with tempfile.NamedTemporaryFile(mode="wt", delete=False) as f:
            f.write(" ".join(o.replace("\\", "/") for o in objFiles))
            f.flush()
            args = kernelWriterAssembly.getLinkCodeObjectArgs([f"@{f.name}"], str(coPathDest))
    else:
        args = kernelWriterAssembly.getLinkCodeObjectArgs(objFiles, str(coPathDest))

    tPrint(2, "Linking objects into co files: " + " ".join(args))
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        tPrint(2, f"Output: {out}")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error linking object files in to code object: {err.output}\nFailed command: {' '.join(args)}"
        )


def buildAssemblyCodeObjectFiles(
    bundler: str,
    kernels: List[Solution],
    writer: KernelWriterAssembly,
    outputPath: Union[Path, str],
    removeTemporaries: bool,
):

    countAsmKernels = lambda kernels: sum(k["KernelLanguage"] == "Assembly" for k in kernels)

    extObj = ".o"
    extCo = ".co"
    extCoRaw = ".co.raw"

    destDir = Path(ensurePath(os.path.join(outputPath, "library")))
    asmDir = Path(writer.getAssemblyDirectory())

    assemblyKernels = [k for k in kernels if k["KernelLanguage"] == "Assembly"]
    if len(assemblyKernels) == 0:
        return []

    archKernelMap = collections.defaultdict(list)
    for k in assemblyKernels:
        archKernelMap[tuple(k["ISA"])].append(k)

    coFiles = []
    for arch, archKernels in archKernelMap.items():
        gfx = gfxName(arch)
        objectFiles = [
            str(asmDir / (writer.getKernelFileBase(k) + extObj))
            for k in archKernels
            if "codeObjectFile" not in k
        ]

        if countAsmKernels(archKernels) == 0:
            continue

        if (
            globalParameters["MergeFiles"]
            or globalParameters["NumMergedFiles"] > 1
            or globalParameters["LazyLibraryLoading"]
        ):

            # Group kernels from placeholder libraries
            coFileMap = collections.defaultdict(list)

            if len(objectFiles):
                coFileMap[asmDir / ("TensileLibrary_" + gfx + extCoRaw)] = objectFiles

            for kernel in archKernels:
                coName = kernel.get("codeObjectFile", None)
                if coName:
                    coFileMap[asmDir / (coName + extCoRaw)].append(
                        str(asmDir / (writer.getKernelFileBase(kernel) + extObj))
                    )

            for coFileRaw, objFiles in coFileMap.items():

                _linkIntoCodeObject(objFiles, coFileRaw, writer)
                coFile = destDir / coFileRaw.name.replace(extCoRaw, extCo)
                compressCodeObject(coFileRaw, coFile, gfx, bundler)
                coFiles.append(coFile)

        else:
            # no mergefiles
            assemblyKernelNames = [writer.getKernelFileBase(k) for k in archKernels]
            origCOFiles = [os.path.join(asmDir, k + ".co") for k in assemblyKernelNames]
            newCOFiles = [os.path.join(destDir, k + "_" + gfx + ".co") for k in assemblyKernelNames]

            for src, dst in (
                zip(origCOFiles, newCOFiles)
                if globalParameters["PrintLevel"] == 0
                else Utils.tqdm(zip(origCOFiles, newCOFiles), desc="Relocating code objects")
            ):
                shutil.copyfile(src, dst)
            coFiles += newCOFiles
            printWarning("Code object files are not compressed in `--no-merge-files` build mode.")

    return coFiles
