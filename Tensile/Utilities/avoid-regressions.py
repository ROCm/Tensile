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


def avoidRegressions():

    userArgs = sys.argv[1:]
    argParser = argparse.ArgumentParser()
    argParser.add_argument("original_dir", help="the library logic directory without tuned sizes")
    argParser.add_argument("incremental_dir", help="the incremental logic directory")
    argParser.add_argument("output_dir", help="the output logic directory")

    args = argParser.parse_args(userArgs)
    originalFiles = allFiles(args.original_dir)
    incrementalFiles = allFiles(args.incremental_dir)
    outputPath = args.output_dir
    ensurePath(outputPath)

    for f in incrementalFiles:
        with open(f) as incFile:
            incData = yaml.safe_load(incFile)
            improvedKernels = dict()
            for g in originalFiles:
                fileSplit = g.split('/')
                logicFile = fileSplit[len(fileSplit)-1]
                if logicFile in f:
                    print("Logic file: ", logicFile)
                    with open(g) as origFile:
                        origData = yaml.safe_load(origFile)
                        currIndex = len(origData[5])
                        numSizes = len(origData[7])
                        incNumSizes = len(incData[7])
                        print(numSizes, " sizes in original logic file")
                        print(incNumSizes, " sizes in tuned logic file")
                        for i in range(0,len(incData[7])):
                            incSize = incData[7][i][0]
                            incIndex = incData[7][i][1][0]
                            incEff = incData[7][i][1][1]
                            isOld = False
                            if len(incSize) == 8:
                                incSize = incSize[:-4]
                            for j in range(0,numSizes):
                                origSize = origData[7][j][0]
                                origIndex = origData[7][j][1][0]
                                origEff = origData[7][j][1][1]
                                if incSize == origSize:
                                    print(origData[7][j])
                                    isOld = True
                                    if incEff < origEff:
                                        print(origSize, " already exists but has regressed in performance. Kernel is unchanged")
                                    else:
                                        if incIndex in improvedKernels.keys():
                                            print(origSize, " already exists and has improved in performance, and uses a previously known kernel.")
                                        if incIndex not in improvedKernels.keys():
                                            print(origSize, " already exists and has improved in performance. A new kernel has been added.")
                                            print("Old Efficiency: ", origEff, ", New Efficiency: ", incEff)
                                            tempData = incData[5][incIndex]
                                            tempData["SolutionIndex"] = currIndex
                                            currIndex = currIndex + 1
                                            improvedKernels[incIndex] = tempData
                                            origData[5].append(improvedKernels[incIndex])
                                        origData[7][j][1][0] = improvedKernels[incIndex]["SolutionIndex"]
                                        origData[7][j][1][1] = incEff
                            if isOld == False:
                                if incIndex in improvedKernels.keys():
                                    print(incSize, " has been added to solution table, and uses a previously known kernel. Efficiency: ", incEff)
                                else:
                                    print(incSize, " has been added to solution table. A new kernel has been added. Efficiency: ", incEff)
                                    tempData = incData[5][incIndex]
                                    tempData["SolutionIndex"] = currIndex
                                    currIndex = currIndex + 1
                                    improvedKernels[incIndex] = tempData
                                    origData[5].append(improvedKernels[incIndex])
                                origData[7].append([incSize,[improvedKernels[incIndex]["SolutionIndex"], incEff]])
                        print(len(origData[7])-numSizes, " sizes and ", len(improvedKernels.keys())," kernels have been added to ", logicFile)
                    with open(outputPath+'/'+logicFile, "w") as outFile:
                        yaml.safe_dump(origData,outFile,default_flow_style=None)

if __name__ == "__main__":
    avoidRegressions()
