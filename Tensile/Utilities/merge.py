################################################################################
# Copyright 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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

import yaml
import os
import sys
import argparse
from copy import deepcopy
from enum import IntEnum

verbosity = 1

def ensurePath(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

def allFiles(startDir):
    current = os.listdir(startDir)
    files = []
    for filename in [_current for _current in current if os.path.splitext(_current)[-1].lower() == '.yaml']:
        fullPath = os.path.join(startDir,filename)
        if os.path.isdir(fullPath):
            files = files + allFiles(fullPath)
        else:
            files.append(fullPath)
    return files

def reindexSolutions(data):
    for i, _ in enumerate(data[5]):
        data[5][i]["SolutionIndex"] = i
    return data

def fixSizeInconsistencies(sizes, fileType):
    duplicates = list()
    for i in range(0,len(sizes)):
        currSize = sizes[i][0]
        # >= so size will be trimmed when a SolutionTag is included
        if len(currSize) >= 8:
            currSize = currSize[:-4]
            if currSize in (item for index in sizes for item in index):
                duplicates.append(i-len(duplicates))
            else:
                sizes[i][0] = currSize
    sizes_ = deepcopy(sizes)
    if len(duplicates) > 0:
        for i in duplicates:
            sizes_.pop(i)
        verbose(len(duplicates), "duplicate size(s) removed from", fileType, "logic file")
    return sizes_, len(sizes_)

# remove dict key "SolutionIndex" from dict
def cmpHelper(sol):
    return {k:v for k, v in sol.items() if k!="SolutionIndex"}

def addKernel(solutionPool, solution):
    for item in solutionPool:
        if cmpHelper(item) == cmpHelper(solution):
            index = item["SolutionIndex"]
            debug("...Reuse previously existed kernel", end="")
            break
    else:
        index = len(solutionPool)
        _solution = deepcopy(solution) # if we don't we will see some subtle errors
        _solution["SolutionIndex"] = index
        solutionPool.append(_solution)
        debug("...A new kernel has been added", end="")
    debug("({}) {}".format(index, solutionPool[index]["SolutionNameMin"] if "SolutionNameMin" in solutionPool[index] else "(SolutionName N/A)"))
    return solutionPool, index

def removeUnusedKernels(origData, prefix=""):
    origNumSolutions = len(origData[5])

    kernelsInUse = [ index for _, [index, _] in origData[7] ]
    for i, solution in enumerate(origData[5]):
        solutionIndex = solution["SolutionIndex"]
        origData[5][i]["__InUse__"] = True if solutionIndex in kernelsInUse else False

    # debug prints
    for o in [o for o in origData[5] if o["__InUse__"]==False]:
        debug("{}Solution ({}) {} is unused".format(
            prefix,
            o["SolutionIndex"],
            o["SolutionNameMin"] if "SolutionNameMin" in o else "(SolutionName N/A)"))

    # filter out dangling kernels
    origData[5] = [ {k: v for k, v in o.items() if k != "__InUse__"}
                    for o in origData[5] if o["__InUse__"]==True ]

    # reindex solutions
    idMap = {} # new = idMap[old]
    for i, solution in enumerate(origData[5]):
        idMap[solution["SolutionIndex"]] = i
        origData[5][i]["SolutionIndex"] = i
    for i, [size, [oldSolIndex, eff]] in enumerate(origData[7]):
        origData[7][i] = [size, [idMap[oldSolIndex], eff]]

    numInvalidRemoved = origNumSolutions - len(origData[5])
    return origData, numInvalidRemoved

def loadData(filename):
    try:
        stream = open(filename, "r")
    except IOError:
        print("Cannot open file: ", filename)
        sys.stdout.flush()
        sys.exit(-1)
    data = yaml.load(stream, yaml.SafeLoader)
    return data

# this is for complying the behavior of legacy merge script, where incremental logic
# file always replaces the base logic file even it's slower in performance -
# in the future we may let default force merge policy = False
def defaultForceMergePolicy(incFile):
    if "arcturus" in incFile:
        forceMerge = False
    else:
        forceMerge = True

    return forceMerge

def msg(*args, **kwargs):
    for i in args: print(i, end=" ")
    print(**kwargs)

def verbose(*args, **kwargs):
    if verbosity < 1: return
    msg(*args, **kwargs)

def debug(*args, **kwargs):
    if verbosity < 2: return
    msg(*args, **kwargs)

# Tags distinguishing solution types
# Can be added to size key to allow solutions of each type to be present
# in logic file for a given size
class SolutionTag(IntEnum):
    VALU = 0
    MFMA = 1

    def __str__(self):
        return ["VALU", "MFMA"][self]
    def __repr__(self):
        return str(self)

def getSolutionTag(solution):
    if solution.get("EnableMatrixInstruction", False) or solution.get("MatrixInstruction", False):
        return SolutionTag.MFMA
    else:
        return SolutionTag.VALU

def findSolutionWithIndex(solutionData, solIndex):
    # Check solution at the index corresponding to solIndex first
    if solIndex < len(solutionData) and solutionData[solIndex]["SolutionIndex"] == solIndex:
        return solutionData[solIndex]
    else:
        debug("Searching for index...")
        solution = [s for s in solutionData if s["SolutionIndex"]==solIndex]
        assert(len(solution) == 1)
        return solution[0]

def addSolutionTagToKeys(solutionMap, solutionPool):
    return [[[getSolutionTag(findSolutionWithIndex(solutionPool, idx))] + keys, [idx, eff]]
            for [keys, [idx, eff]] in solutionMap]

def removeSolutionTagFromKeys(solutionMap):
    return [[keys[1:], [idx, incEff]] for keys, [idx, incEff] in solutionMap]

# returns merged logic data as list
def mergeLogic(origData, incData, forceMerge, trimSize=True, addMfmaTag=False):
    origNumSizes = len(origData[7])
    origNumSolutions = len(origData[5])

    incNumSizes = len(incData[7])
    incNumSolutions = len(incData[5])

    verbose(origNumSizes, "sizes and", origNumSolutions, "kernels in base logic file")
    verbose(incNumSizes, "sizes and", incNumSolutions, "kernels in incremental logic file")

    # Add SolutionTag to distinguish mfma and non-mfma solutions
    origTaggedSizes = addSolutionTagToKeys(origData[7], origData[5])
    incTaggedSizes  = addSolutionTagToKeys(incData[7],  incData[5])
    if addMfmaTag:
        origData[7] = origTaggedSizes
        incData[7]  = incTaggedSizes
    # Print warning if addMfmaTag=False results in removed sizes
    else:
        origSet       = {tuple(size) for size, [_, _] in origData[7]}
        origTaggedSet = {tuple(size) for size, [_, _] in origTaggedSizes}
        incSet        = {tuple(size) for size, [_, _] in incData[7]}
        incTaggedSet  = {tuple(size) for size, [_, _] in incTaggedSizes}

        if len(origSet) != len(origTaggedSet):
            verbose("Warning:", len(origTaggedSet) - len(origSet), "duplicate sizes are present in base logic",
                    "that may not be handled correctly unless --add_mfma_tag is used")
        if len(incSet) != len(incTaggedSet):
            verbose("Warning:", len(incTaggedSet) - len(incSet), "duplicate sizes are present in incremental logic",
                    "that may not be handled correctly unless --add_mfma_tag is used")



    if trimSize:
        # trim 8-tuple gemm size format to 4-tuple [m, n, b, k]
        # TODO future gemm size could include dictionary format so need robust preprocessing
        [origData[7], origNumSizes] = fixSizeInconsistencies(origData[7], "base")
        [incData[7], incNumSizes] = fixSizeInconsistencies(incData[7], "incremental")

    origData, numOrigRemoved = removeUnusedKernels(origData, "Base logic file: ")
    incData, numIncRemoved = removeUnusedKernels(incData, "Inc logic file: ")

    solutionPool = deepcopy(origData[5])
    solutionMap = deepcopy(origData[7])

    origDict = {tuple(origSize): [i, origEff] for i, [origSize, [origIndex, origEff]] in enumerate(origData[7])}
    for incSize, [incIndex, incEff] in incData[7]:
        incSolution = findSolutionWithIndex(incData[5], incIndex)

        try:
            j, origEff = origDict[tuple(incSize)]
            if incEff > origEff or forceMerge:
                if incEff > origEff:
                    verbose("[O]", incSize, "already exists and has improved in performance.", end="")
                elif forceMerge:
                    verbose("[!]", incSize, "already exists but does not improve in performance.", end="")
                verbose("Efficiency:", origEff, "->", incEff, "(force_merge=True)" if forceMerge else "")
                solutionPool, index = addKernel(solutionPool, incSolution)
                solutionMap[j][1] = [index, incEff]
            else:
                verbose("[X]", incSize, "already exists but does not improve in performance.", end="")
                verbose("Efficiency:", origEff, "->", incEff)
        except KeyError:
            verbose("[-]", incSize, "has been added to solution table, Efficiency: N/A ->", incEff)
            solutionPool, index = addKernel(solutionPool, incSolution)
            solutionMap.append([incSize,[index, incEff]])

    verbose(numOrigRemoved, "unused kernels removed from base logic file")
    verbose(numIncRemoved, "unused kernels removed from incremental logic file")

    # Remove SolutionTag for yaml output
    if addMfmaTag:
        solutionMap = removeSolutionTagFromKeys(solutionMap)

    mergedData = deepcopy(origData)
    mergedData[5] = solutionPool
    mergedData[7] = solutionMap
    mergedData, numReplaced = removeUnusedKernels(mergedData, "Merged data: ")

    numSizesAdded = len(solutionMap)-len(origData[7])
    numSolutionsAdded = len(solutionPool)-len(origData[5])
    numSolutionsRemoved = numReplaced+numOrigRemoved # incremental file not counted

    return [mergedData, numSizesAdded, numSolutionsAdded, numSolutionsRemoved]

def avoidRegressions(originalDir, incrementalDir, outputPath, forceMerge, trimSize=True, addMfmaTag=False):
    originalFiles = allFiles(originalDir)
    incrementalFiles = allFiles(incrementalDir)
    ensurePath(outputPath)

    # filter the incremental logic files that have the corresponding base file
    incrementalFiles = [ i for i in incrementalFiles
                         if os.path.split(i)[-1] in [os.path.split(o)[-1] for o in originalFiles] ]

    for incFile in incrementalFiles:
        basename = os.path.split(incFile)[-1]
        origFile = os.path.join(originalDir, basename)
        forceMerge = defaultForceMergePolicy(incFile) if forceMerge is None else forceMerge

        msg("Base logic file:", origFile, "| Incremental:", incFile, "| Merge policy: %s"%("Forced" if forceMerge else "Winner"), "| Trim size:", trimSize,
        "| Add MFMA tag:", addMfmaTag)
        origData = loadData(origFile)
        incData = loadData(incFile)

        # So far "SolutionIndex" in logic yamls has zero impact on actual 1-1 size mapping (but the order of the Solution does)
        # since mergeLogic() takes that value very seriously so we reindex them here so it doesn't choke on duplicated SolutionIndex
        origData = reindexSolutions(origData)
        incData = reindexSolutions(incData)

        mergedData, *stats = mergeLogic(origData, incData, forceMerge, trimSize, addMfmaTag)
        msg(stats[0], "size(s) and", stats[1], "kernel(s) added,", stats[2], "kernel(s) removed")

        with open(os.path.join(outputPath, basename), "w") as outFile:
            yaml.safe_dump(mergedData,outFile,default_flow_style=None)
        msg("File written to", os.path.join(outputPath, basename))
        msg("------------------------------")

# partialLogicFilePaths: list of full paths to partial logic files
# outputDir: Directory to write the final result to
# forceMerge:
# trimSize:
# Expects: that all the partial logic files
# have the same base name, but are located
# in different folders.
# Provides: one final logic file that is the
# merged result of all partial files.
# This is useful for when a tuning task is
# shared between multiple machines who each
# will provide a partial result.
def mergePartialLogics(partialLogicFilePaths, outputDir, forceMerge, trimSize=True, addMfmaTag=False):
    logicFiles = deepcopy(partialLogicFilePaths)
    ensurePath(outputDir)

    baseLogicFile = logicFiles.pop(0)
    baseLogicData = loadData(baseLogicFile)
    msg("Base logic file:", baseLogicFile)
    for f in logicFiles:
        forceMerge = defaultForceMergePolicy(f) if forceMerge is None else forceMerge

        msg("Incremental file:", f, "| Merge policy: %s"%("Forced" if forceMerge else "Winner"), "| Trim size:", trimSize)
        incLogicData = loadData(f)

        # So far "SolutionIndex" in logic yamls has zero impact on actual 1-1 size mapping (but the order of the Solution does)
        # since mergeLogic() takes that value very seriously so we reindex them here so it doesn't choke on duplicated SolutionIndex
        baseLogicData = reindexSolutions(baseLogicData)
        incLogicData = reindexSolutions(incLogicData)

        mergedData, *stats = mergeLogic(baseLogicData, incLogicData, forceMerge, trimSize, addMfmaTag)
        msg(stats[0], "size(s) and", stats[1], "kernel(s) added,", stats[2], "kernel(s) removed")

        # Use the merged data as the base data for the next partial logic file
        baseLogicData = deepcopy(mergedData)


    baseFileName = os.path.basename(baseLogicFile)
    outputFilePath = os.path.join(outputDir, baseFileName)
    with open(outputFilePath, "w") as outFile:
        yaml.safe_dump(baseLogicData, outFile, default_flow_style=None)
    msg("File written to", outputFilePath)
    msg("------------------------------")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("original_dir", help="The library logic directory without tuned sizes")
    argParser.add_argument("incremental_dir", help="The incremental logic directory")
    argParser.add_argument("output_dir", help="The output logic directory")
    argParser.add_argument("-v", "--verbosity", help="0: summary, 1: verbose, 2: debug", default=1, type=int)
    argParser.add_argument("--force_merge", help="Merge previously known sizes unconditionally. Default behavior if not arcturus", default="none")
    argParser.add_argument("--notrim", help="Do not trim long size format down to short format (m,n,b,k). Default is --trim", action="store_false")
    argParser.add_argument("--add_mfma_tag", help="Add a tag to the size key for usage of MFMA instructions, allowing for both a MFMA and non-MFMA kernel to exist for the same size. Default doesn't add this tag.", action="store_true")

    args = argParser.parse_args(sys.argv[1:])
    originalDir = args.original_dir
    incrementalDir = args.incremental_dir
    outputPath = args.output_dir
    verbosity = args.verbosity
    forceMerge = args.force_merge.lower()
    trimSize = args.notrim
    addMfmaTag = args.add_mfma_tag

    if forceMerge in ["none"]: forceMerge=None
    elif forceMerge in ["true", "1"]: forceMerge=True
    elif forceMerge in ["false", "0"]: forceMerge=False

    avoidRegressions(originalDir, incrementalDir, outputPath, forceMerge, trimSize, addMfmaTag)
