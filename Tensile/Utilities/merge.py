import yaml
import os
import sys
import argparse
from copy import deepcopy

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

def fixSolutionIndexBug(kernels):
    for i in range(0,len(kernels)):
        kernels[i]["SolutionIndex"] = i
    return kernels

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
    for i, item in enumerate(solutionPool):
        if cmpHelper(item) == cmpHelper(solution):
            index = i
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
    for i in range(0,len(origData[5])):
        solutionIndex = origData[5][i]["SolutionIndex"]
        for item in range(0,len(origData[7])):
            index = origData[7][item][1][0]
            if index == solutionIndex:
                origData[5][i]["__valid__"] = True
                break
        else:
            origData[5][i]["__valid__"] = False

    # debug prints
    for o in [o for o in origData[5] if o["__valid__"]==False]:
        debug("{}Solution ({}) {} is unused".format(
            prefix,
            o["SolutionIndex"],
            o["SolutionNameMin"] if "SolutionNameMin" in o else "(SolutionName N/A)"))

    # filter out dangling kernels
    origData[5] = [ {k: v for k, v in o.items() if k != "__valid__"}
                    for o in origData[5] if o["__valid__"]==True ]

    # reindex solutions
    for i in range(0,len(origData[5])):
        oldSolIndex = origData[5][i]["SolutionIndex"]
        origData[5][i]["SolutionIndex"] = i
        for item in range(0,len(origData[7])):
            index = origData[7][item][1][0]
            if index == oldSolIndex:
                origData[7][item][1][0] = i

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

# returns merged logic data as list
def mergeLogic(origData, incData, forceMerge, trimSize=True):
    origNumSizes = len(origData[7])
    origNumSolutions = len(origData[5])

    incNumSizes = len(incData[7])
    incNumSolutions = len(incData[5])

    verbose(origNumSizes, "sizes and", origNumSolutions, "kernels in base logic file")
    verbose(incNumSizes, "sizes and", incNumSolutions, "kernels in incremental logic file")

    if trimSize:
        # trim 8-tuple gemm size format to 4-tuple [m, n, b, k]
        # TODO future gemm size could include dictionary format so need robust preprocessing
        [origData[7], origNumSizes] = fixSizeInconsistencies(origData[7], "base")
        [incData[7], incNumSizes] = fixSizeInconsistencies(incData[7], "incremental")

    origData, numOrigRemoved = removeUnusedKernels(origData, "Base logic file: ")
    incData, numIncRemoved = removeUnusedKernels(incData, "Inc logic file: ")

    solutionPool = deepcopy(origData[5])
    solutionMap = deepcopy(origData[7])
    for i, [incSize, [incIndex, incEff]] in enumerate(incData[7]):
        incSolution = [s for s in incData[5] if s["SolutionIndex"]==incIndex] # TODO this is slow
        assert len(incSolution)==1
        incSolution = incSolution[0]
        for j, [origSize, [origIndex, origEff]] in enumerate(origData[7]):
            if incSize != origSize:
                continue
            if incEff > origEff or forceMerge:
                if incEff > origEff:
                    verbose("[O]", origSize, "already exists and has improved in performance.", end="")
                elif forceMerge:
                    verbose("[!]", origSize, "already exists but does not improve in performance.", end="")
                verbose("Efficiency:", origEff, "->", incEff, "(force_merge=True)" if forceMerge else "")
                solutionPool, index = addKernel(solutionPool, incSolution)
                solutionMap[j][1] = [index, incEff]
            else:
                verbose("[X]", origSize, " already exists but does not improve in performance.", end="")
                verbose("Efficiency:", origEff, "->", incEff)
            break
        else:
            verbose("[-]", incSize, "has been added to solution table, Efficiency: N/A ->", incEff)
            solutionPool, index = addKernel(solutionPool, incSolution)
            solutionMap.append([incSize,[index, incEff]])

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
        mergedData, *stats = mergeLogic(loadData(origFile), loadData(incFile), forceMerge, trimSize)
        msg(stats[0], "sizes and", stats[1], "kernels added,", stats[2], "kernels removed")

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
