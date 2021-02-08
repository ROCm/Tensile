import yaml
import os
import sys
import argparse
from copy import deepcopy
from collections import defaultdict
from enum import Enum
import pdb

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
        if len(currSize) == 8:
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

def findSolutionWithIndex(solutionData, solIndex):
    #Return index if 
    solution = solutionData[solIndex]
    if solution["SolutionIndex"] == solIndex:
        return solution
    else:
        print("Searching for index...")
        solution = [s for s in solutionData if s["SolutionIndex"]==solIndex]
        assert(len(solution) == 1)
        return solution[0]

class SolutionUpdate(Enum):
    Replace=0
    Append =1
    Reject =2

# returns (efficiency[, replacedIndex]) or None if nothing is replaced
def findOriginalSolutionToUpdate(incSolution, incSize, incEff, origSolutionData, origSolutionDict, forceMerge, includeKernelVariants = False):
    try:
        origSolutions = origSolutionDict[tuple(incSize)]
        origJ, origIndex, origEff = origSolutions[0]
        originalSolution = None
        if includeKernelVariants:
            for j, index, eff in origSolutions:
                solution = findSolutionWithIndex(origSolutionData, index)
                if solution["EnableMatrixInstruction"] == incSolution["EnableMatrixInstruction"]:
                    origJ, origIndex, origEff = j, index, eff
                    originalSolution = solution
                    break
            if not originalSolution:
                verbose("[O]", incSize, "already exists but has a different value for EnableMatrixInstruction."
                                        " New entry has been added to the solution table")
                return SolutionUpdate.Append, 
        else:
            originalSolution = findSolutionWithIndex(origSolutionData, origIndex)
        
        if incEff > origEff or forceMerge:
            if incEff > origEff:
                verbose("[O]", incSize, "already exists and has improved in performance.", end="")
            elif forceMerge:
                verbose("[!]", incSize, "already exists but does not improve in performance.", end="")
            verbose("Efficiency:", origEff, "->", incEff, "(force_merge=True)" if forceMerge else "")
            return SolutionUpdate.Replace, origJ
        else:
            verbose("[X]", incSize, " already exists but does not improve in performance.", end="")
            verbose("Efficiency:", origEff, "->", incEff)
    except KeyError:
        verbose("[-]", incSize, "has been added to solution table, Efficiency: N/A ->", incEff)
        return SolutionUpdate.Append,
    
    return SolutionUpdate.Reject,

# returns merged logic data as list
def mergeLogic(origData, incData, forceMerge, trimSize=True, includeKernelVariants=False):
    origNumSizes = len(origData[7])
    origNumSolutions = len(origData[5])

    incNumSizes = len(incData[7])
    incNumSolutions = len(incData[5])

    verbose(origNumSizes, "sizes and", origNumSolutions, "kernels in base logic file")
    verbose(incNumSizes, "sizes and", incNumSolutions, "kernels in incremental logic file")

    unmergedKernelParameters = {"EnableMatrixInstruction": False} if includeKernelVariants else {}

    if trimSize:
        # trim 8-tuple gemm size format to 4-tuple [m, n, b, k]
        # TODO future gemm size could include dictionary format so need robust preprocessing
        [origData[7], origNumSizes] = fixSizeInconsistencies(origData[7], "base")
        [incData[7], incNumSizes] = fixSizeInconsistencies(incData[7], "incremental")

    origData, numOrigRemoved = removeUnusedKernels(origData, "Base logic file: ")
    incData, numIncRemoved = removeUnusedKernels(incData, "Inc logic file: ")

    solutionPool = deepcopy(origData[5])
    solutionMap = deepcopy(origData[7])

    origDict = {} 
    for i, [origSize, [origIndex, origEff]] in enumerate(origData[7]):
        origDict[tuple(origSize)] = origDict.get(tuple(origSize), []) + [[i, origIndex, origEff]] 

    for incSize, [incIndex, incEff] in incData[7]:
        incSolution = findSolutionWithIndex(incData[5], incIndex)

        result, *index = findOriginalSolutionToUpdate(incSolution,  
                                                      incSize, 
                                                      incEff, 
                                                      origData[5], 
                                                      origDict, 
                                                      forceMerge,
                                                      includeKernelVariants)

        if result == SolutionUpdate.Append:
            solutionPool, kernelIndex = addKernel(solutionPool, incSolution)
            solutionMap.append([incSize,[kernelIndex, incEff]])
            #pdb.set_trace()
        elif result == SolutionUpdate.Replace:
            assert(index)
            replacedIndex = index[0]
            solutionPool, kernelIndex = addKernel(solutionPool, incSolution)
            pdb.set_trace()
            solutionMap[replacedIndex][1] = [kernelIndex, incEff]

    verbose(numOrigRemoved, "unused kernels removed from base logic file")
    verbose(numIncRemoved, "unused kernels removed from incremental logic file")

    mergedData = deepcopy(origData)
    mergedData[5] = solutionPool
    mergedData[7] = solutionMap
    mergedData, numReplaced = removeUnusedKernels(mergedData, "Merged data: ")

    numSizesAdded = len(solutionMap)-len(origData[7])
    numSolutionsAdded = len(solutionPool)-len(origData[5])
    numSolutionsRemoved = numReplaced+numOrigRemoved # incremental file not counted

    return [mergedData, numSizesAdded, numSolutionsAdded, numSolutionsRemoved]

def avoidRegressions(originalDir, incrementalDir, outputPath, forceMerge, trimSize=True):
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

        msg("Base logic file:", origFile, "| Incremental:", incFile, "| Merge policy: %s"%("Forced" if forceMerge else "Winner"), "| Trim size:", trimSize)
        origData = loadData(origFile)
        incData = loadData(incFile)

        # So far "SolutionIndex" in logic yamls has zero impact on actual 1-1 size mapping (but the order of the Solution does)
        # since mergeLogic() takes that value very seriously so we reindex them here so it doesn't choke on duplicated SolutionIndex
        origData = reindexSolutions(origData)
        incData = reindexSolutions(incData)

        mergedData, *stats = mergeLogic(origData, incData, forceMerge, trimSize, True)
        msg(stats[0], "size(s) and", stats[1], "kernel(s) added,", stats[2], "kernel(s) removed")

        with open(os.path.join(outputPath, basename), "w") as outFile:
            yaml.safe_dump(mergedData,outFile,default_flow_style=None)
        msg("File written to", os.path.join(outputPath, basename))
        msg("------------------------------")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("original_dir", help="The library logic directory without tuned sizes")
    argParser.add_argument("incremental_dir", help="The incremental logic directory")
    argParser.add_argument("output_dir", help="The output logic directory")
    argParser.add_argument("-v", "--verbosity", help="0: summary, 1: verbose, 2: debug", default=1, type=int)
    argParser.add_argument("--force_merge", help="Merge previously known sizes unconditionally. Default behavior if not arcturus", default="none")
    argParser.add_argument("--notrim", help="Do not trim long size format down to short format (m,n,b,k). Default is --trim", action="store_false")

    args = argParser.parse_args(sys.argv[1:])
    originalDir = args.original_dir
    incrementalDir = args.incremental_dir
    outputPath = args.output_dir
    verbosity = args.verbosity
    forceMerge = args.force_merge.lower()
    trimSize = args.notrim

    if forceMerge in ["none"]: forceMerge=None
    elif forceMerge in ["true", "1"]: forceMerge=True
    elif forceMerge in ["false", "0"]: forceMerge=False

    avoidRegressions(originalDir, incrementalDir, outputPath, forceMerge, trimSize)
