import yaml
import os
import sys
import argparse

def ensurePath(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

def allFiles(startDir):
    current = os.listdir(startDir)
    files = []
    for filename in current:
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
    if len(duplicates) > 0:
        for i in duplicates:
            sizes.pop(i)
        print(len(duplicates), "duplicate size(s) removed from ", fileType, " logic file")
    return [sizes,len(sizes)]

def addKernel(incData, origData, improvedKernels, incIndex, currIndex):
    tempData = incData[5][incIndex]
    tempData["SolutionIndex"] = currIndex
    currIndex = currIndex + 1
    improvedKernels[incIndex] = tempData
    origData[5].append(improvedKernels[incIndex])
    return [incData, origData, improvedKernels, incIndex, currIndex]

def removeUnusedKernels(origData):
    unusedKernels = list()
    for i in range(0,len(origData[5])):
        kernelIndex = origData[5][i]["SolutionIndex"]
        isUsed = False
        for item in range(0,len(origData[7])):
            index = origData[7][item][1][0]
            if index == kernelIndex:
                isUsed = True
                break
        if isUsed == False:
            uIndex = i-len(unusedKernels)
            if uIndex not in unusedKernels:
                unusedKernels.append(uIndex)

    for i in unusedKernels:
        origData[5].pop(i)
    for i in range(0,len(origData[5])):
        oldSolIndex = origData[5][i]["SolutionIndex"]
        origData[5][i]["SolutionIndex"] = i
        for item in range(0,len(origData[7])):
            index = origData[7][item][1][0]
            if index == oldSolIndex:
                origData[7][item][1][0] = i

    print("Removed ",len(unusedKernels), " unused kernels from output logic file")
    return origData

def loadData(filename):
    try:
        stream = open(filename, "r")
    except IOError:
        print("Cannot open file: ", filename)
        sys.stdout.flush()
        sys.exit(-1)
    data = yaml.load(stream, yaml.SafeLoader)
    return data

def avoidRegressions():

    userArgs = sys.argv[1:]
    argParser = argparse.ArgumentParser()
    argParser.add_argument("original_dir", help="The library logic directory without tuned sizes")
    argParser.add_argument("incremental_dir", help="The incremental logic directory")
    argParser.add_argument("output_dir", help="The output logic directory")
    argParser.add_argument("force_merge", help="Merge previously known sizes unconditionally. Default behavior if not arcturus", nargs='?', default="true")

    args = argParser.parse_args(userArgs)
    originalFiles = allFiles(args.original_dir)
    incrementalFiles = allFiles(args.incremental_dir)
    outputPath = args.output_dir
    forceMerge = args.force_merge
    ensurePath(outputPath)

    for incFile in incrementalFiles:
        with open(incFile):
            if "arcturus" in incFile:
                forceMerge = "false"
            incData = loadData(incFile)
            improvedKernels = dict()
            for origFile in originalFiles:
                fileSplit = origFile.split('/')
                logicFile = fileSplit[len(fileSplit)-1]
                if logicFile in incFile:
                    print("Logic file: ", logicFile)
                    with open(origFile):
                        origData = loadData(origFile)
                        numSizes = len(origData[7])
                        incNumSizes = len(incData[7])
                        print(numSizes, " sizes in original logic file")
                        print(incNumSizes, " sizes in tuned logic file")
                        print(len(origData[5]), " kernels in original logic file")
                        print(len(incData[5]), " kernels in tuned logic file")
                        [origData[7], numSizes] = fixSizeInconsistencies(origData[7], "original")
                        origData[5] = fixSolutionIndexBug(origData[5])
                        [incData[7], incNumSizes] = fixSizeInconsistencies(incData[7], "incremental")
                        incData[5] = fixSolutionIndexBug(incData[5])
                        currIndex = len(origData[5])
                        for i in range(0,len(incData[7])):
                            incSize = incData[7][i][0]
                            incIndex = incData[7][i][1][0]
                            incEff = incData[7][i][1][1]
                            isOld = False
                            for j in range(0,len(origData[7])):
                                origSize = origData[7][j][0]
                                origEff = origData[7][j][1][1]
                                if incSize == origSize:
                                    isOld = True
                                    if incEff < origEff and forceMerge == "false":
                                        print(origSize, " already exists but has regressed in performance. Kernel is unchanged")
                                        print("Old Efficiency: ", origEff, "New efficiency: ", incEff)
                                    else:
                                        if incIndex in improvedKernels.keys():
                                            print(origSize, " already exists and has improved in performance, and uses a previously known kernel.")
                                            print("Old Efficiency: ", origEff, "New efficiency: ", incEff)
                                        if incIndex not in improvedKernels.keys():
                                            print(origSize, " already exists and has improved in performance. A new kernel has been added.")
                                            print("Old Efficiency: ", origEff, ", New Efficiency: ", incEff)
                                            [incData, origData, improvedKernels, incIndex, currIndex] = addKernel(incData, origData, improvedKernels, incIndex, currIndex)
                                        origData[7][j][1][0] = improvedKernels[incIndex]["SolutionIndex"]
                                        origData[7][j][1][1] = incEff
                            if isOld == False:
                                if incIndex in improvedKernels.keys():
                                    print(incSize, " has been added to solution table, and uses a previously known kernel. Efficiency: ", incEff)
                                else:
                                    print(incSize, " has been added to solution table. A new kernel has been added. Efficiency: ", incEff)
                                    [incData, origData, improvedKernels, incIndex, currIndex] = addKernel(incData, origData, improvedKernels, incIndex, currIndex)
                                origData[7].append([incSize,[improvedKernels[incIndex]["SolutionIndex"], incEff]])
                        print(len(origData[7])-numSizes, " sizes and ", len(improvedKernels.keys())," kernels have been added to ", logicFile)
                        origData = removeUnusedKernels(origData)
                    with open(outputPath+'/'+logicFile, "w") as outFile:
                        yaml.safe_dump(origData,outFile,default_flow_style=None)

if __name__ == "__main__":
    avoidRegressions()
