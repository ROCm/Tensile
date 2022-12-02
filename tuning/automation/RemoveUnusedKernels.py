################################################################################
#
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

import argparse
import os
import yaml

def parseArgs():
    argParser = argparse.ArgumentParser()

    h = {"inDir"    : "Directory containing input logic files", \
         "outDir"   : "Output directory for modified logic files"}
    
    argParser.add_argument("inDir", metavar="input-dir", type=str, help=h["inDir"])
    argParser.add_argument("outDir", metavar="output-dir", type=str, help=h["outDir"])

    return argParser.parse_args()

def allFiles(startDir):
    current = os.listdir(startDir)
    files = []
    for filename in current:
        fullPath = os.path.join(startDir, filename)
        if os.path.isdir(fullPath):
            files = files + allFiles(fullPath)
        else:
            files.append(fullPath)
    return files

def main():
    print("Remove Unused Kernels")
    print("=====================")
    args = parseArgs()

    files = allFiles(args.inDir)
    for inFile in files:
        print("Reading: {}".format(inFile))
        with open(inFile) as yamlFile:
            data = yaml.safe_load(yamlFile)
            
        solutions = data[5]
        logic = data[7]
        indicesUsed = {x[1][0] for x in logic}
        minSolutions = [s for s in solutions if s['SolutionIndex'] in indicesUsed]
        indexMap = {k:minSolutions.index(solutions[k]) for k in indicesUsed}
        for (idx, s) in enumerate(minSolutions):
            minSolutions[idx]['SolutionIndex'] = idx
        for l in logic:
            l[1][0] = indexMap[l[1][0]]
        data[5] = minSolutions
        data[7] = logic

        relFile = os.path.relpath(inFile, args.inDir)
        outFile = os.path.join(args.outDir, relFile)
        print("Writing: {}".format(outFile))
        os.makedirs(os.path.dirname(outFile), exist_ok=True)
        with open(outFile, "w") as yamlFile:
            yaml.safe_dump(data, yamlFile, default_flow_style=None)

if __name__ == "__main__":
    main()
