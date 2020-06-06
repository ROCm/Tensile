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
            currIndex = len(incData[5])
            regressedKernels = dict()
            for i in range(0,len(incData[7])):
                currSize = incData[7][i][0]
                if len(currSize) == 8:
                    currSize = currSize[:-4]
                for g in originalFiles:
                    fileSplit = g.split('/')
                    if fileSplit[len(fileSplit)-1] in f:
                        with open(g) as largeFile:
                            origData = yaml.safe_load(largeFile)
                            for j in range(0,len(origData[7])):
                                if currSize == origData[7][j][0]:
                                    print("Size: ",origData[7][j][0])
                                    index = origData[7][j][1][0]
                                    eff = origData[7][j][1][1]
                                    if index in regressedKernels.keys():
                                        incData[7][i][1][0] = regressedKernels[index]["SolutionIndex"]
                                    else:
                                        tempData = origData[5][index]
                                        tempData["SolutionIndex"] = currIndex
                                        currIndex = currIndex + 1
                                        regressedKernels[index] = tempData
                                        incData[5].append(regressedKernels[index])
                                        incData[7][i][1][0] = regressedKernels[index]["SolutionIndex"]
                                    incData[7][i][1][1] = eff
                        with open(outputPath+'/'+fileSplit[len(fileSplit)-1], "w") as outFile:
                            yaml.safe_dump(incData,outFile,default_flow_style=None)

if __name__ == "__main__":
    avoidRegressions()
